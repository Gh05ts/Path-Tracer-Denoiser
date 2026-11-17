"""Numerical controls and bounded-run failure paths for the diagnostic launcher."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch

from test_adaptive_dataset import source_dataset
from test_joint_training import joint_dataset, config_for
from wavefront_denoiser.joint import JointNTASAD
from wavefront_denoiser.joint_checkpoint import load_joint_checkpoint
from wavefront_denoiser.joint_data import JointPatchDataset
from wavefront_denoiser.joint_training import _batch, train_joint, training_config
from wavefront_denoiser.loss import sequence_loss
from wavefront_denoiser.model import INPUT_CHANNELS
from wavefront_denoiser.ntasad_radiance import radiance_gradient_policy
from wavefront_denoiser.sample_layers import replay_sample_layers


@pytest.fixture
def diagnostic():
    path = Path(__file__).parents[1]/"tools/diagnose_quality_collapse.py"
    spec = importlib.util.spec_from_file_location("quality_collapse_diagnostic", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def sample_batch(root):
    sample = JointPatchDataset(root, partition="train", patch_size=5, augment=False)[0]
    return {"buffers": {k: v[None] for k, v in sample["buffers"].items()}, "layers": sample["layers"][None]}


def build(diagnostic, mode):
    torch.manual_seed(7)
    ordinary = JointNTASAD(4, 4)
    model = diagnostic.DiagnosticJoint(ordinary.specification(), mode)
    model.load_state_dict(ordinary.state_dict())
    return ordinary, model


def test_tracing_preserves_exact_predictions_gradients_adam_and_rng(diagnostic, joint_dataset):
    torch.set_num_threads(1)
    ordinary, model = build(diagnostic, "joint")
    batch = sample_batch(joint_dataset)
    buffers, layers, reference = _batch(batch, "cpu")
    optimizers = [torch.optim.Adam(m.parameters(), lr=.0003) for m in (ordinary, model)]
    state = torch.get_rng_state().clone()
    model.trace.enabled = True
    with radiance_gradient_policy(1e-6):
        predictions = []
        for m, optimizer in zip((ordinary, model), optimizers):
            prediction = m.forward_sequence(buffers, layers, training_reference=reference, transformed=True)
            predictions.append(prediction)
            sequence_loss(prediction.radiance, reference, transformed=True)["loss"].backward()
            optimizer.step()
    torch.testing.assert_close(predictions[0].radiance, predictions[1].radiance, rtol=0, atol=0)
    torch.testing.assert_close(predictions[0].counts, predictions[1].counts, rtol=0, atol=0)
    for a, b in zip(ordinary.parameters(), model.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    frames, gradients = model.trace.backward_stats(reference)
    assert len(frames) == len(gradients) == 4
    assert model.trace.features and len(model.trace.logits) == 4
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize("mode,frozen", [("uniform", "sampler"), ("frozen_denoiser", "denoiser")])
def test_component_freezing_and_uniform_recurrent_parity(diagnostic, joint_dataset, mode, frozen):
    torch.set_num_threads(1)
    ordinary, model = build(diagnostic, mode)
    buffers, layers, reference = _batch(sample_batch(joint_dataset), "cpu")
    before = copy.deepcopy(getattr(model, frozen).state_dict())
    with radiance_gradient_policy(1e-6):
        prediction = model.forward_sequence(buffers, layers, training_reference=reference, transformed=True)
        if mode == "uniform":
            counts = torch.full_like(prediction.counts, model.average_spp)
            current = torch.stack([replay_sample_layers(layers[:, t], counts[:, t]) for t in range(5)], 1)
            control = ordinary.denoiser.forward_sequence(dict(buffers, radiance=current, sample_count=counts), transformed=True)
            torch.testing.assert_close(prediction.radiance, control.radiance, rtol=0, atol=0)
            assert torch.all(prediction.counts == 4)
        sequence_loss(prediction.radiance, reference, transformed=True)["loss"].backward()
        active = "denoiser" if mode == "uniform" else "sampler"
        assert all(p.grad is None for p in getattr(model, frozen).parameters())
        assert any(p.grad is not None and torch.any(p.grad != 0) for p in getattr(model, active).parameters())
        optimizer = diagnostic.optimizer_for(model, {"learning_rate": .0003}, {})
        optimizer.step()
        for key, value in getattr(model, frozen).state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_uniform_evaluation_ignores_reference_in_predictions_and_labels_partial(diagnostic, joint_dataset):
    _, model = build(diagnostic, "joint")
    path, _ = config_for(joint_dataset)
    config = training_config(path)
    data = diagnostic.dataset(config, "validation")
    try:
        normal = diagnostic.evaluate(model, data, config, "cpu", time.monotonic()+30)
        uniform = diagnostic.evaluate(model, data, config, "cpu", time.monotonic()+30, uniform=True)
        partial = diagnostic.evaluate(model, data, config, "cpu", time.monotonic()-1)
        assert normal["complete"] and uniform["complete"]
        assert uniform["learned_frame_paths"] == uniform["learned_frame_pixels"]*4
        assert not partial["complete"] and partial["metrics"] is None and partial["windows"] == 0
    finally:
        data.close()


def test_diagnostic_baseline_matches_production_epoch_and_anomaly_replays(diagnostic, joint_dataset, tmp_path):
    path, _ = config_for(joint_dataset, epochs=1, learning_rate=.0003, radiance_gradient_floor=1e-6)
    config = training_config(path)
    source = tmp_path/"source"
    train_joint(path, source/"train", "cpu")
    base = tmp_path/"diagnostic"; base.mkdir()
    plan = {"source_run": str(source), "max_epochs": 1, "arm_seconds": 60, "report_reserve_seconds": 1,
            "detail_every_steps": 1, "anomaly_thresholds": {"sampler_gradient_norm": 1e-12}}
    with diagnostic.execution_policy(config):
        summary = diagnostic.train_arm(base, config, plan, "baseline", "inventory-fixture", time.monotonic()+90, "cpu")
        probe = diagnostic.probe_snapshot(base/"baseline", "cpu")
    _, original = load_joint_checkpoint(source/"train/last.pt")
    saved = torch.load(base/"baseline/last.pt", weights_only=True)
    assert summary["completed_epochs"] == 1
    assert summary["historical_epoch_deltas"][0]["deltas"] == {
        "loss": 0., "zero_fraction": 0., "sampler_gradient_norm": 0., "denoiser_gradient_norm": 0.}
    for key, value in original["state_dict"].items():
        torch.testing.assert_close(value, saved["state_dict"][key], rtol=0, atol=0)
    for transfer in probe["transfers"]:
        assert transfer["paths_before"] == transfer["paths_after"]
    assert (base/"baseline/anomaly-after.pt").is_file()
    before = torch.load(base/"baseline/anomaly-before.pt", weights_only=True)
    after = torch.load(base/"baseline/anomaly-after.pt", weights_only=True)
    batch = torch.load(base/"baseline/anomaly-batch.pt", weights_only=True)["batch"]
    replay = diagnostic.DiagnosticJoint(before["model_spec"], before["arm"]["mode"])
    replay.load_state_dict(before["state_dict"])
    optimizer = diagnostic.optimizer_for(replay, config, before["arm"])
    optimizer.load_state_dict(before["optimizer"])
    buffers, layers, reference = _batch(batch, "cpu")
    with diagnostic.execution_policy(config):
        prediction = replay.forward_sequence(buffers, layers, training_reference=reference, transformed=True)
        sequence_loss(prediction.radiance, reference, transformed=True)["loss"].backward()
        optimizer.step()
    for key, value in replay.state_dict().items():
        torch.testing.assert_close(value, after["state_dict"][key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="Incompatible"):
        load_joint_checkpoint(base/"baseline/last.pt")


def test_budget_does_not_start_an_epoch_and_report_survives_partial_arms(diagnostic, joint_dataset, tmp_path):
    path, _ = config_for(joint_dataset)
    config = training_config(path)
    source = tmp_path/"source"; (source/"train").mkdir(parents=True)
    (source/"train/history.jsonl").write_text("")
    plan = {"source_run": str(source), "arms": ["baseline", "uniform"], "max_epochs": 12,
            "arm_seconds": 1, "report_reserve_seconds": 1, "detail_every_steps": 1, "anomaly_thresholds": {}}
    base = tmp_path/"diagnostic"; base.mkdir()
    with diagnostic.execution_policy(config):
        summary = diagnostic.train_arm(base, config, plan, "baseline", "fixture", time.monotonic(), "cpu")
    assert summary["completed_epochs"] == summary["global_step"] == 0
    assert summary["stop_reason"] == "arm_time_budget"
    assert not summary["initial"]["complete"]
    diagnostic.write(base/"plan.json", plan)
    diagnostic.report(base, "overall_deadline_stopped")
    assert "incomplete" in (base/"report.md").read_text()
    assert "not started" in (base/"report.md").read_text()


def test_no_cuda_preflight_creates_no_outputs(diagnostic, monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is required"):
        diagnostic.launch(tmp_path/"not-created", tmp_path/"unused-plan.json", {}, {})
    assert not (tmp_path/"not-created").exists()


def test_optional_repairs_are_separate_optimizer_policies(diagnostic):
    _, model = build(diagnostic, "joint")
    optimizer = diagnostic.optimizer_for(model, {"learning_rate": .0003}, diagnostic.ARMS["sampler-lr-low"])
    assert [g["name"] for g in optimizer.param_groups] == ["sampler", "denoiser"]
    assert [g["lr"] for g in optimizer.param_groups] == pytest.approx([.00003, .0003])


@pytest.mark.parametrize("reason", ["memory_guard_stopped", "overall_deadline_stopped"])
def test_launcher_stops_worker_and_preserves_partial_report(diagnostic, monkeypatch, tmp_path, reason):
    plan = diagnostic.read(Path(__file__).parents[1]/"configs/quality-collapse-diagnostic.json")
    source = tmp_path/"original/train"; source.mkdir(parents=True)
    for name in ("initial.pt", "best-trained.pt"):
        (source/name).write_bytes(b"fixture")
    plan["source_run"] = str(source.parent)
    plan["total_seconds"] = 0 if reason == "overall_deadline_stopped" else 30
    monkeypatch.setattr(diagnostic, "WORKSPACE", tmp_path)
    monkeypatch.setattr(diagnostic, "preflight", lambda *args: None)
    used = 80 if reason == "memory_guard_stopped" else 1
    monkeypatch.setattr(diagnostic, "memory_sample", lambda: {
        "host_total_bytes": 100, "host_used_bytes": used, "gpu_total_bytes": 100, "gpu_used_bytes": 1})
    original_popen = subprocess.Popen
    processes = []
    def sleeper(*args, **kwargs):
        process = original_popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                start_new_session=True)
        processes.append(process)
        return process
    monkeypatch.setattr(diagnostic.subprocess, "Popen", sleeper)
    # check_output uses Popen too, so keep Git provenance read independent.
    monkeypatch.setattr(diagnostic.subprocess, "check_output", lambda args, **kwargs:
                        "fixture-head\n" if kwargs.get("text") else b"")
    base = tmp_path/"runs/diagnostic"
    assert diagnostic.launch(base, tmp_path/"plan.json", plan, {}) == 1
    assert processes[0].poll() is not None
    assert diagnostic.read(base/"status.json")["status"] == reason
    assert reason in (base/"report.md").read_text()
    assert (source/"initial.pt").read_bytes() == b"fixture"
