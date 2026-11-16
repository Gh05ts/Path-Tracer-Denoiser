import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import os

import pytest

from test_joint_training import joint_dataset, config_for
from test_adaptive_dataset import source_dataset

ORIGINAL_RUN = subprocess.run


spec = importlib.util.spec_from_file_location("benchmark_joint", Path(__file__).resolve().parents[1]/"tools/benchmark_joint.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_benchmark_aggregates_windows_and_excludes_warmup():
    def row(seconds,windows,steps):
        return dict(training_seconds=seconds,validation_seconds=seconds*2,
            training_windows=windows,validation_windows=3,training_steps=steps,
            training_data_seconds=seconds/2,checkpoint_seconds=.1)
    summary = {"config":{"window_length":5,"patch_size":128,"validation_patch_size":256},
        "initial_validation":{"loss":1.},"performance":{"epochs":[row(100,4,1),row(2,4,2),row(4,3,2)],
        "cuda_peak_allocated_bytes":None,"cuda_peak_reserved_bytes":None,"process_lifetime_peak_rss_bytes":42}}
    result = benchmark.summarize(summary,1)
    assert result["training_windows_per_second"] == pytest.approx(7/6)
    assert result["training_pixels_per_second"] == pytest.approx(7*5*128**2/6)
    assert result["validation_windows_per_second"] == pytest.approx(6/12)
    assert result["training_windows"] == 7
    assert result["optimizer_updates"] == 4
    assert result["training_data_wait_seconds"] == 3
    with pytest.raises(ValueError,match="measured epochs"):
        benchmark.summarize(summary,3)


def test_batch_sweep_runs_isolated_cases_and_records_provenance(joint_dataset,tmp_path,monkeypatch):
    # The capture fixture substitutes subprocess.run only to create its EXRs.
    # Restore it now so both training cases execute as real fresh processes.
    monkeypatch.setattr(subprocess,"run",ORIGINAL_RUN)
    monkeypatch.setattr(benchmark,"git",lambda *args: (
        "test-commit" if args == ("rev-parse","HEAD") else
        "test-branch" if args == ("branch","--show-current") else ""))
    config,_ = config_for(joint_dataset,epochs=1,radiance_gradient_floor=1e-6)
    output = tmp_path/"benchmark"
    assert benchmark.main([str(config),str(output),"--device","cpu","--batches","1","2",
        "--warmup-epochs","1","--measured-epochs","1"]) == 0
    manifest = json.loads((output/"manifest.json").read_text())
    results = json.loads((output/"comparison.json").read_text())
    assert manifest["commit"] == "test-commit"
    assert manifest["dirty"] is False
    assert [r["optimizer_updates"] for r in results] == [4,2]
    assert results[0]["inventory_sha256"] == results[1]["inventory_sha256"]
    assert all(r["training_windows"] == 4 for r in results)
    # Identical seeds and fresh processes must initialize identical models.
    assert results[0]["initial_validation"] == results[1]["initial_validation"]
    for name in ("train-batch-1","train-batch-2"):
        software = json.loads((output/name/"software.json").read_text())
        assert Path(software["import_path"]).is_relative_to(benchmark.ROOT/"src")
        assert software["autocast"] is False
    with pytest.raises(SystemExit):
        benchmark.main([str(config),str(output),"--device","cpu"])


@pytest.mark.parametrize('sweep',['loader','crops','safety'])
def test_loader_sweep_keeps_batches_and_initial_state_fixed(joint_dataset,tmp_path,monkeypatch,sweep):
    monkeypatch.setattr(subprocess,"run",ORIGINAL_RUN)
    monkeypatch.setattr(benchmark,"git",lambda *args: (
        "test-commit" if args == ("rev-parse","HEAD") else
        "test-branch" if args == ("branch","--show-current") else ""))
    config,_=config_for(joint_dataset,epochs=1,radiance_gradient_floor=1e-6)
    output=tmp_path/'loaders'
    assert benchmark.main([str(config),str(output),'--device','cpu','--sweep',sweep,
        '--warmup-epochs','1','--measured-epochs','1'])==0
    rows=json.loads((output/'comparison.json').read_text())
    assert [r['loader_policy'] for r in rows]==([None,'cached_v1','prefetch_v1'] if sweep=='loader'
        else [None,None,None] if sweep=='safety' else ['prefetch_v1','prefetch_fused_v1'])
    assert len({r['batch'] for r in rows})==1
    assert len({r['inventory_sha256'] for r in rows})==1
    assert all(r['initial_validation']==rows[0]['initial_validation'] for r in rows)
    assert all(r['last_validation']==rows[0]['last_validation'] for r in rows)


def test_cache_sweep_records_parity_hashes_and_discards_own_checkpoints(joint_dataset,tmp_path,monkeypatch):
    monkeypatch.setattr(subprocess,'run',ORIGINAL_RUN)
    monkeypatch.setattr(benchmark,'git',lambda *args: 'test-commit' if args==('rev-parse','HEAD') else '')
    config,_=config_for(joint_dataset,epochs=1,radiance_gradient_floor=1e-6)
    output=tmp_path/'caches'
    assert benchmark.main([str(config),str(output),'--device','cpu','--sweep','cache',
        '--cache-frames','8','32','--prefetch-depths','4','--discard-checkpoints',
        '--warmup-epochs','1','--measured-epochs','1'])==0
    rows=json.loads((output/'comparison.json').read_text())
    assert [row['decoded_cache_frames'] for row in rows]==[8,32]
    assert all(row['prefetch_batches']==4 for row in rows)
    assert len({row['model_state_sha256'] for row in rows})==1
    assert len({row['training_state_sha256'] for row in rows})==1
    assert not list(output.rglob('*.pt'))
    assert len(list(output.glob('*/summary.json')))==2


def test_resource_monitor_terminates_case_at_ram_limit(tmp_path,monkeypatch):
    monkeypatch.setattr(benchmark,'host_memory',lambda:(100,10))
    audit=tmp_path/'resources.json'
    with (tmp_path/'log').open('w') as log:
        code=benchmark.run_case([sys.executable,'-c','import time; time.sleep(10)'],
            log,dict(os.environ),.8,'cpu',audit)
    assert code==124
    assert json.loads(audit.read_text())['violation']=='host RAM limit'
