import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")


def test_dataset_launcher_trains_one_joint_model_and_creates_missing_sidecar(tmp_path, monkeypatch):
    import wavefront_denoiser.training as training
    path = Path(__file__).parents[1] / "tools/train_dataset.py"
    spec = importlib.util.spec_from_file_location("train_dataset", path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    dataset = tmp_path / "data"
    dataset.mkdir()
    (dataset / "capture_lock.json").write_text(json.dumps({"configs": {
        "train": {"frames": [{}, {}, {}, {}, {}]}, "validation": {"frames": [{}, {}, {}, {}, {}]}}}))
    partitions = {"train": ["train"], "validation": ["validation"], "test": ["sealed_test"]}
    monkeypatch.setattr(launcher, "read_split_manifest", lambda _: {"partitions": partitions})
    monkeypatch.setattr(launcher, "audit_dataset_split", lambda *a, **kw: {})
    checked = []
    monkeypatch.setattr(launcher, "verify_frame", lambda directory, config, index: checked.append((directory, index)))
    calls = []
    monkeypatch.setattr(training, "train_denoiser", lambda *a, **kw: calls.append((a, kw)))
    # This is the state of a capture started by the old two-model launcher.
    (dataset / "train-spatial.json").write_text('{"retired": true}')
    (dataset / "train-recurrent.json").write_text('{"retired": true}')
    output = tmp_path / "joint-run"
    resume = tmp_path / "last.pt"
    assert launcher.main(["--dataset", str(dataset), "--output", str(output),
                          "--device", "cpu", "--resume", str(resume)]) == 0
    assert len(calls) == 1 and len(checked) == 10
    args, kwargs = calls[0]
    assert args == (dataset / "train.json", output, "cpu")
    assert kwargs == {"resume": resume}
    config = json.loads((dataset / "train.json").read_text())
    assert config["epochs"] == 1000 and config["window_length"] == 5
    assert config["base_channels"] == 48 and config["learning_rate"] == .001
    assert all("sealed_test" not in str(directory) for directory, _ in checked)
    # Editing the current config is respected by later launcher invocations.
    config["epochs"] = 2000
    (dataset / "train.json").write_text(json.dumps(config))
    launcher.main(["--dataset", str(dataset), "--output", str(output), "--device", "cpu"])
    assert json.loads((dataset / "train.json").read_text())["epochs"] == 2000
