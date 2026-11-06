import json

import pytest

from wavefront_denoiser import FrameFormatError, SequenceDataset, discover_sequences
from wavefront_denoiser.cli import main


def test_windows_stay_in_sequence_and_reset_initial_history(make_frame, tmp_path):
    for sequence in ("a", "b"):
        for index in range(3):
            make_frame(f"{sequence}/frame_{index:06}", index=index, sequence=sequence, previous=index > 0)
    dataset = SequenceDataset(tmp_path, length=2, tensors=False)
    assert len(dataset) == 4
    assert dataset[1]["frame_indices"] == [1, 2]
    assert dataset[1]["buffers"]["radiance"].shape == (2, 2, 3, 3)
    assert not dataset[1]["buffers"]["history_valid"][0].any()
    assert dataset[1]["buffers"]["history_valid"][1].all()
    assert len(SequenceDataset(tmp_path, length=2, sequence_ids=["b"], tensors=False)) == 2


def test_torch_sequence_batch(make_frame, tmp_path):
    torch = pytest.importorskip("torch")
    for index in range(2):
        make_frame(f"frame_{index:06}", index=index, previous=index > 0)
    dataset = SequenceDataset(tmp_path, length=2)
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1)))
    assert tuple(batch["buffers"]["radiance"].shape) == (1, 2, 3, 2, 3)


def test_partial_export_is_not_a_frame(make_frame, tmp_path):
    make_frame()
    make_frame("frame_000001.partial", index=1, previous=True)
    assert len(discover_sequences(tmp_path)["fixture"]) == 1


@pytest.mark.parametrize("indices", [(0, 2), (0, 0)])
def test_broken_history_and_duplicate_indices_rejected(make_frame, tmp_path, indices):
    make_frame("first", index=indices[0])
    make_frame("second", index=indices[1], previous=indices[1] > 0)
    with pytest.raises(FrameFormatError, match="history"):
        discover_sequences(tmp_path)


def test_camera_history_must_match(make_frame, tmp_path):
    make_frame()
    make_frame("frame_000001", index=1, previous=True,
               metadata_mutate=lambda m: m["previous_camera"].update(position=[1, 0, 0]))
    with pytest.raises(FrameFormatError, match="history"):
        discover_sequences(tmp_path)


def test_cli_validates_pixels(make_frame, tmp_path, capsys):
    make_frame()
    assert main(["validate", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["validated_frames"] == 1


def test_empty_or_short_sequences_fail_clearly(make_frame, tmp_path):
    with pytest.raises(FrameFormatError, match="No completed"):
        discover_sequences(tmp_path)
    make_frame()
    with pytest.raises(ValueError, match="long enough"):
        SequenceDataset(tmp_path, length=5)
