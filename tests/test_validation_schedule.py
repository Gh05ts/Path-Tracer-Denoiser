import importlib.util
import json
from pathlib import Path

import pytest
import torch

from test_adaptive_dataset import source_dataset
from test_joint_training import joint_dataset,config_for
from wavefront_denoiser import joint_training as training
from wavefront_denoiser.joint_checkpoint import load_joint_checkpoint


@pytest.mark.parametrize('epochs,expected',[(3,[1,2,3]),(12,[1,2,3,4,5,10,12]),(15,[1,2,3,4,5,10,15])])
def test_global_schedule_and_final_epoch(epochs,expected):
    config=dict(epochs=epochs,validation_every_epochs=5,validation_warmup_epochs=5)
    assert [e for e in range(1,epochs+1) if training.validation_due(e,config)]==expected


@pytest.mark.parametrize('key,value',[('validation_every_epochs',0),('validation_every_epochs',True),
    ('validation_warmup_epochs',-1),('finite_check_policy','skip')])
def test_invalid_scheduling_settings(joint_dataset,key,value):
    path,_=config_for(joint_dataset,**{key:value})
    with pytest.raises(ValueError,match=key):
        training.training_config(path)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_grouped_flags_include_every_tensor_and_empty_groups(device):
    if device=='cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    tensors=[torch.ones(3,device=device),torch.ones(4,device=device),torch.ones(2,device=device)]
    for bad in (float('nan'),float('inf'),-float('inf')):
        for index in range(3):
            values=[t.clone() for t in tensors];values[index][-1]=bad
            assert training.finite_group_flags([[],tensors,values],device)==[True,True,False]


@pytest.mark.parametrize('network',['sampler','denoiser'])
def test_nonfinite_gradient_blocks_optimizer(joint_dataset,tmp_path,monkeypatch,network):
    original=training.JointNTASAD
    def create(**kw):
        model=original(**kw)
        list(getattr(model,network).parameters())[-1].register_hook(lambda grad: grad*float('nan'))
        return model
    monkeypatch.setattr(training,'JointNTASAD',create)
    monkeypatch.setattr(torch.optim.Adam,'step',lambda *a,**k: pytest.fail('Unsafe optimizer update'))
    path,_=config_for(joint_dataset,epochs=1,finite_check_policy='grouped_v1')
    with pytest.raises(ValueError,match=f'Non-finite {network} training gradient'):
        training.train_joint(path,tmp_path/'bad','cpu')


def test_nonfinite_weight_after_update_is_rejected(joint_dataset,tmp_path,monkeypatch):
    original=torch.optim.Adam.step
    def step(opt,*a,**kw):
        result=original(opt,*a,**kw)
        with torch.no_grad():
            opt.param_groups[-1]['params'][-1].fill_(float('inf'))
        return result
    monkeypatch.setattr(torch.optim.Adam,'step',step)
    path,_=config_for(joint_dataset,epochs=1,finite_check_policy='grouped_v1')
    with pytest.raises(ValueError,match='Non-finite joint weights'):
        training.train_joint(path,tmp_path/'bad','cpu')


def test_sparse_validation_metrics_parity_resume_and_report(joint_dataset,tmp_path,monkeypatch):
    saved=[]
    original=training.save_joint_checkpoint
    def save(path,*args,**kwargs):
        saved.append((Path(path).parent.name,Path(path).name,kwargs['epoch']))
        return original(path,*args,**kwargs)
    monkeypatch.setattr(training,'save_joint_checkpoint',save)
    spec=importlib.util.spec_from_file_location('report',Path(__file__).parents[1]/'tools/report_joint_pilot.py')
    report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)
    path,config=config_for(joint_dataset,epochs=4)
    training.train_joint(path,tmp_path/'legacy','cpu')
    config.update(validation_every_epochs=3,finite_check_policy='grouped_v1')
    path.write_text(json.dumps(config))
    summary=training.train_joint(path,tmp_path/'full','cpu')
    assert summary['best_epoch'] in (0,3,4)
    assert [e for parent,name,e in saved if parent=='full' and name=='last.pt']==[1,2,3,4]
    rows=[json.loads(x) for x in (tmp_path/'full/history.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in rows if r['validation'] is not None]==[3,4]
    assert all(r['timing']['validation_windows']==0 and r['timing']['validation_seconds']==0 for r in rows[:2])
    assert all((tmp_path/'full'/name).exists() for name in ('model.pt','last.pt'))
    config['epochs']=1;path.write_text(json.dumps(config))
    training.train_joint(path,tmp_path/'first','cpu')
    config['epochs']=4;path.write_text(json.dumps(config))
    training.train_joint(path,tmp_path/'resumed','cpu',resume=tmp_path/'first/last.pt')
    expected,payload=load_joint_checkpoint(tmp_path/'legacy/last.pt')
    for name in ('full','resumed'):
        model,p=load_joint_checkpoint(tmp_path/name/'last.pt')
        for key,value in expected.state_dict().items():
            torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
        assert torch.equal(payload['training']['rng']['torch'],p['training']['rng']['torch'])
        assert torch.equal(payload['training']['rng']['generator'],p['training']['rng']['generator'])
    text=report.make_report(tmp_path/'full')
    assert 'not evaluated' in text
    resumed=[json.loads(x) for x in (tmp_path/'resumed/history.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in resumed if r['validation'] is not None]==[3,4]
