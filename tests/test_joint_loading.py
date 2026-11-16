import json
import threading

import pytest
import torch

from test_adaptive_dataset import source_dataset
from test_joint_training import joint_dataset, config_for
from wavefront_denoiser.joint_data import JointPatchDataset
from wavefront_denoiser.joint_loading import joint_batches
from wavefront_denoiser.joint_training import train_joint, training_config
from wavefront_denoiser.joint_checkpoint import load_joint_checkpoint


@pytest.mark.parametrize('depth',[1,2,4,8])
def test_prefetch_preserves_exact_samples_order_and_rng(joint_dataset, monkeypatch,depth):
    dataset = JointPatchDataset(joint_dataset,partition='train',patch_size=5,patches_per_window=2)
    main = threading.get_ident()
    original_plan, original_load = dataset.plan,dataset.load_plan
    def plan(index):
        assert threading.get_ident()==main
        return original_plan(index)
    monkeypatch.setattr(dataset,'plan',plan)
    outcomes = []
    for prefetch in (False,True):
        def load(plan):
            assert (threading.get_ident()!=main)==prefetch
            return original_load(plan)
        monkeypatch.setattr(dataset,'load_plan',load)
        torch.manual_seed(13)
        generator = torch.Generator().manual_seed(17)
        # Eight windows at batch 3 exercises the partial batch.
        with joint_batches(dataset,batch_size=3,generator=generator,shuffle=True,prefetch=prefetch,prefetch_batches=depth) as batches:
            rows = list(batches)
        outcomes.append((rows,torch.get_rng_state(),generator.get_state()))
    for a,b in zip(outcomes[0][0],outcomes[1][0]):
        assert a['sequence_id']==b['sequence_id']
        torch.testing.assert_close(a['layers'],b['layers'],rtol=0,atol=0)
        for name in a['buffers']:
            torch.testing.assert_close(a['buffers'][name],b['buffers'][name],rtol=0,atol=0)
    for a,b in zip(outcomes[0][1:],outcomes[1][1:]):
        assert torch.equal(a,b)


def test_prefetch_failure_and_early_exit_join_worker(joint_dataset,monkeypatch):
    dataset = JointPatchDataset(joint_dataset,partition='train',patch_size=5)
    original = dataset.load_plan
    def fail(plan):
        raise ValueError('loading failed')
    monkeypatch.setattr(dataset,'load_plan',fail)
    with pytest.raises(ValueError,match='loading failed'):
        with joint_batches(dataset,batch_size=1,generator=torch.Generator(),prefetch=True) as batches:
            next(batches)
    monkeypatch.setattr(dataset,'load_plan',original)
    with joint_batches(dataset,batch_size=1,generator=torch.Generator(),prefetch=True) as batches:
        next(batches)
    assert not [t for t in threading.enumerate() if t.name.startswith('joint-loading')]


@pytest.mark.parametrize('depth',[1,2,4,8])
def test_prefetch_overlaps_consumer_and_bounds_lookahead(depth):
    planned=[]
    started,release=threading.Event(),threading.Event()
    class Dataset:
        def __len__(self):
            return 100
        def plan(self,index):
            planned.append(index)
            return index
        def load_plan(self,index):
            if index==1:
                started.set()
                if not release.wait(timeout=5):
                    raise RuntimeError('Consumer did not release test worker')
            return {'index':index}
    with joint_batches(Dataset(),batch_size=1,generator=torch.Generator(),prefetch=True,prefetch_batches=depth) as batches:
        try:
            assert next(batches)['index'].item()==0
            assert started.wait(timeout=5)  # Next preparation runs while the consumer holds batch 0.
            assert planned==list(range(depth+1))  # Current batch plus bounded lookahead.
        finally:
            release.set()
    assert not [t for t in threading.enumerate() if t.name.startswith('joint-loading')]


def test_prefetch_and_synchronous_training_match_exactly(joint_dataset,tmp_path):
    states=[]
    for policy in (None,'cached_v1','prefetch_v1'):
        options={} if policy is None else {'loader_policy':policy}
        path,_=config_for(joint_dataset,**options)
        output=tmp_path/str(policy)
        train_joint(path,output,'cpu')
        model,payload=load_joint_checkpoint(output/'last.pt')
        states.append((model.state_dict(),payload['training']['rng']))
    for weights,rng in states[1:]:
        for name,value in states[0][0].items():
            torch.testing.assert_close(value,weights[name],rtol=0,atol=0)
        for name in ('torch','generator'):
            assert torch.equal(rng[name],states[0][1][name])


@pytest.mark.parametrize('policy',[None,False,1,'workers_v1',{}])
def test_invalid_loader_policy_is_rejected(joint_dataset,policy):
    path,config=config_for(joint_dataset)
    config['loader_policy']=policy
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='loader_policy'):
        training_config(path)


def test_resume_rejects_changed_loader_policy(joint_dataset,tmp_path):
    path,config=config_for(joint_dataset,epochs=1,loader_policy='cached_v1')
    train_joint(path,tmp_path/'first','cpu')
    config.update(epochs=2,loader_policy='prefetch_v1')
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='configuration'):
        train_joint(path,tmp_path/'resumed','cpu',resume=tmp_path/'first/last.pt')


@pytest.mark.parametrize('key,value,policy',[
    ('decoded_cache_frames',0,'prefetch_v1'),('decoded_cache_frames',True,'prefetch_v1'),
    ('decoded_cache_frames',8,None),('prefetch_batches',0,'prefetch_v1'),
    ('prefetch_batches',1.5,'prefetch_v1'),('prefetch_batches',2,'cached_v1')])
def test_invalid_cache_and_queue_options(joint_dataset,key,value,policy):
    options={} if policy is None else {'loader_policy':policy}
    options[key]=value
    path,_=config_for(joint_dataset,**options)
    with pytest.raises(ValueError,match=key):
        training_config(path)


@pytest.mark.parametrize('key,new_value',[('decoded_cache_frames',32),('prefetch_batches',4)])
def test_resume_rejects_changed_cache_or_queue(joint_dataset,tmp_path,key,new_value):
    path,config=config_for(joint_dataset,epochs=1,loader_policy='prefetch_v1',decoded_cache_frames=8,prefetch_batches=1)
    train_joint(path,tmp_path/'first','cpu')
    config.update(epochs=2)
    config[key]=new_value
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='configuration'):
        train_joint(path,tmp_path/'changed','cpu',resume=tmp_path/'first/last.pt')
