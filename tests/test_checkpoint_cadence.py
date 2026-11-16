import json
import importlib.util
from pathlib import Path

import pytest
import torch

from test_adaptive_dataset import source_dataset
from test_joint_training import joint_dataset,config_for
from wavefront_denoiser import joint_training as training
from wavefront_denoiser.joint_checkpoint import load_joint_checkpoint


@pytest.mark.parametrize('value',[0,-1,True,1.5])
def test_invalid_checkpoint_interval(tmp_path,value):
    from wavefront_denoiser.training_defaults import joint_training_config
    config=joint_training_config();config['checkpoint_every_epochs']=value
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='checkpoint_every_epochs'):
        training.training_config(path)


def test_checkpoint_interval_final_best_and_resume(joint_dataset,tmp_path,monkeypatch):
    calls=[];original=training.save_joint_checkpoint
    def save(path,*args,**kwargs):
        calls.append((path.parent.name,path.name,kwargs['epoch']))
        return original(path,*args,**kwargs)
    monkeypatch.setattr(training,'save_joint_checkpoint',save)
    path,config=config_for(joint_dataset,epochs=5,validation_every_epochs=3,finite_check_policy='grouped_v1')
    training.train_joint(path,tmp_path/'legacy','cpu')
    config['checkpoint_every_epochs']=3;path.write_text(json.dumps(config))
    training.train_joint(path,tmp_path/'sparse','cpu')
    assert [e for parent,name,e in calls if parent=='legacy' and name=='last.pt']==[1,2,3,4,5]
    assert [e for parent,name,e in calls if parent=='sparse' and name=='last.pt']==[3,5]
    assert [e for parent,name,e in calls if parent=='legacy' and name=='model.pt']==[e for parent,name,e in calls if parent=='sparse' and name=='model.pt']
    config['epochs']=3;path.write_text(json.dumps(config))
    training.train_joint(path,tmp_path/'first','cpu')
    config['epochs']=5;path.write_text(json.dumps(config))
    training.train_joint(path,tmp_path/'resumed','cpu',resume=tmp_path/'first/last.pt')
    assert [e for parent,name,e in calls if parent=='resumed' and name=='last.pt']==[5]
    expected,payload=load_joint_checkpoint(tmp_path/'legacy/last.pt')
    for name in ('sparse','resumed'):
        actual,p=load_joint_checkpoint(tmp_path/name/'last.pt')
        for key,value in expected.state_dict().items():
            torch.testing.assert_close(value,actual.state_dict()[key],rtol=0,atol=0)
        for key in ('torch','generator'):
            assert torch.equal(payload['training']['rng'][key],p['training']['rng'][key])
    rows=[json.loads(x) for x in (tmp_path/'sparse/history.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in rows if r['timing']['resume_checkpoint_saved']]==[3,5]
    # Supply only the metadata needed by the report's full-inventory projection.
    (joint_dataset/'capture_lock.json').write_text(json.dumps({'plan':{'scenes':[
        {'id':p,'partition':p} for p in ('train','validation')]},'configs':{
        p:{'scene':p,'frames':list(range(8))} for p in ('train','validation')}}))
    spec=importlib.util.spec_from_file_location('report',Path(__file__).parents[1]/'tools/report_joint_pilot.py')
    report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)
    text=report.make_report(tmp_path/'sparse')
    assert 'Resume checkpoint: every 3 epochs and final' in text
    assert '1000 epochs:' in text
