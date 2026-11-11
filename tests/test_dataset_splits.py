from copy import deepcopy
import json

import pytest

from wavefront_denoiser.dataset_splits import audit_dataset_split, read_split_manifest


@pytest.fixture
def split_fixture(make_frame, tmp_path):
    spec = {"format": "wavefront-dataset-split", "version": 1, "data_root": "data",
            "partitions": {"train": ["train"], "validation": ["val"], "test": ["test"]},
            "groups": {name: {"scene": f"scene_{name}", "family": f"family_{name}", "capture_config": f"{name}.json"}
                       for name in ("train", "val", "test")}}
    filename = tmp_path / "split.json"
    filename.write_text(json.dumps(spec))
    paths = {}
    for number, name in enumerate(("train", "val", "test"), start=1):
        paths[name] = make_frame(f"data/{name}/frame", sequence=name, width=8, height=8,
            mutate=lambda c, n=number: c["reference.R"].fill(5 + n),
            metadata_mutate=lambda m, n=number, key=name: m.update(scene=f"scene_{key}", scene_fingerprint=f"{n:x}"))
    return filename, spec, paths


def rewrite(path, transform):
    metadata = json.loads(path.read_text())
    transform(metadata)
    path.write_text(json.dumps(metadata))


def test_independent_partitions_are_metadata_only(split_fixture):
    filename, _, paths = split_fixture
    # Deliberately corrupt the withheld image. Auditing must not open test pixels.
    (paths["test"] / "frame.exr").write_bytes(b"test image is sealed from training")
    audit = audit_dataset_split(filename)
    assert audit["status"] == "ready"
    assert audit["test_pixels_loaded"] is False
    assert audit["groups"]["test"]["frames"] == 1


def test_family_cannot_cross_partitions(split_fixture):
    filename, _, _ = split_fixture
    rewrite(filename, lambda spec: spec["groups"]["val"].update(family="family_train"))
    with pytest.raises(ValueError, match="family crosses"):
        read_split_manifest(filename)


def test_same_scene_with_different_family_labels_is_rejected(split_fixture):
    filename, _, _ = split_fixture
    rewrite(filename, lambda spec: spec["groups"]["val"].update(scene="scene_train"))
    with pytest.raises(ValueError, match="Scene crosses"):
        read_split_manifest(filename)


def test_renamed_scene_identity_cannot_cross_partitions(split_fixture):
    filename, _, paths = split_fixture
    rewrite(paths["val"] / "manifest.json", lambda metadata: metadata.update(scene_fingerprint="1"))
    with pytest.raises(ValueError, match="Scene identity crosses"):
        audit_dataset_split(filename)


def test_duplicate_image_content_cannot_cross_partitions(split_fixture):
    filename, _, paths = split_fixture
    train = json.loads((paths["train"] / "manifest.json").read_text())
    rewrite(paths["test"] / "manifest.json", lambda metadata: metadata.update(image_sha256=train["image_sha256"]))
    with pytest.raises(ValueError, match="Image content crosses"):
        audit_dataset_split(filename)


def test_test_can_be_reserved_before_capture(split_fixture):
    filename, _, paths = split_fixture
    (paths["test"] / "manifest.json").unlink()
    audit = audit_dataset_split(filename, allow_missing_test=True)
    assert audit["status"] == "test_reserved"
    with pytest.raises(ValueError, match="Missing dataset group"):
        audit_dataset_split(filename)


def test_duplicate_membership_rejected(split_fixture):
    filename, _, _ = split_fixture
    rewrite(filename, lambda spec: spec["partitions"]["test"].append("train"))
    with pytest.raises(ValueError, match="exactly one"):
        read_split_manifest(filename)


def test_legacy_captures_need_recapture_for_strict_split(split_fixture):
    filename, _, paths = split_fixture
    rewrite(paths["train"] / "manifest.json", lambda metadata: metadata.pop("scene_fingerprint"))
    with pytest.raises(ValueError, match="recapture"):
        audit_dataset_split(filename)


def test_group_names_cannot_escape_dataset_root(split_fixture):
    filename, spec, _ = split_fixture
    spec["groups"]["../escape"] = spec["groups"].pop("test")
    spec["partitions"]["test"] = ["../escape"]
    filename.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="exactly one"):
        read_split_manifest(filename)


def test_training_never_loads_reserved_test_pixels(split_fixture, tmp_path, monkeypatch, make_frame):
    torch = pytest.importorskip("torch")
    import wavefront_denoiser.sequence_data as data
    import wavefront_denoiser.sequences as sequences
    from wavefront_denoiser.training import train_denoiser
    from wavefront_denoiser.runtime import evaluate_partition, evaluate_sequences

    filename, _, paths = split_fixture
    # Sequence training needs five consecutive frames in each observed group.
    for name, number in (("train", 1), ("val", 2)):
        for index in range(1, 5):
            make_frame(f"data/{name}/frame_{index}", index=index, previous=True, sequence=name,
                width=8, height=8, mutate=lambda c, n=number: c["reference.R"].fill(5+n),
                metadata_mutate=lambda m, n=number, key=name: m.update(scene=f"scene_{key}", scene_fingerprint=f"{n:x}"))
    (paths["test"] / "frame.exr").write_bytes(b"withheld pixels must never be opened")
    original = data.load_frame
    opened = []

    def guarded(path, *args, **kwargs):
        assert paths["test"] not in path.parents
        opened.append(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(data, "load_frame", guarded)
    monkeypatch.setattr(sequences, "load_frame", guarded)
    config = {"split_manifest": filename.name, "epochs": 1, "patch_size": 4,
              "patches_per_window": 1, "base_channels": 4, "batch_size": 1, "cpu_threads": 1}
    config_path = tmp_path / "train.json"
    config_path.write_text(json.dumps(config))
    output = tmp_path / "run"
    train_denoiser(config_path, output, "cpu")
    assert opened
    payload = torch.load(output / "model.pt", weights_only=True)
    assert payload["dataset_split"]["split_audit"]["partitions"]["test"] == ["test"]
    with pytest.raises(ValueError, match="Reserved test"):
        evaluate_sequences(output / "model.pt", tmp_path / "data", ["test"], "cpu")
    report = evaluate_partition(output / "model.pt", filename, "validation", "cpu")
    assert report["partition"] == "validation"
    assert report["test_used_for_checkpoint_selection"] is False
    rewrite(paths["val"] / "manifest.json", lambda metadata: metadata.update(frame_index=9))
    with pytest.raises(ValueError, match="changed since"):
        evaluate_partition(output / "model.pt", filename, "test", "cpu")


def test_strict_training_requires_reference_qualification(split_fixture, tmp_path):
    pytest.importorskip("torch")
    from wavefront_denoiser.training import train_denoiser

    filename, _, _ = split_fixture
    rewrite(filename, lambda spec: spec.update(require_reference_policy=True))
    config_path = tmp_path / "train.json"
    config_path.write_text(json.dumps({"split_manifest": filename.name}))
    with pytest.raises(ValueError, match="qualified reference"):
        train_denoiser(config_path, tmp_path / "must-not-exist", "cpu")
    assert not (tmp_path / "must-not-exist").exists()


def test_declared_fixed_reference_budget_is_enforced_without_claiming_qualification(split_fixture):
    filename, _, _ = split_fixture
    rewrite(filename, lambda spec: spec.update(declared_reference_samples=16))
    audit = audit_dataset_split(filename)
    assert audit["declared_reference_samples"] == 16
    assert audit["reference_samples"] is None
    rewrite(filename, lambda spec: spec.update(declared_reference_samples=1024))
    with pytest.raises(ValueError, match="declared dataset budget"):
        audit_dataset_split(filename)


def test_declared_frame_count_rejects_incomplete_clips(split_fixture):
    filename, _, _ = split_fixture
    rewrite(filename, lambda spec: spec["groups"]["train"].update(frame_count=8))
    with pytest.raises(ValueError, match="Incomplete frame indices"):
        audit_dataset_split(filename)
