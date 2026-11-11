"""Inspect completed preview clips; never discovers reserved test captures.

Usage: python tools/inspect_temporal_pilot.py runs/ntasad-pilot-v1/preview
Writes measurements and a display-only contact sheet beside the previews.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from wavefront_denoiser.frames import load_frame
from wavefront_denoiser.sequences import SequenceDataset
from wavefront_denoiser.sequence_data import SequencePatchDataset


def motion_error(frame):
    """Reconstruct primary hits from forward depth, then project into history."""
    b, metadata = frame.buffers, frame.metadata
    valid = b["history_valid"][..., 0] > 0
    if not valid.any():
        return None
    camera, previous = metadata["camera"], metadata["previous_camera"]
    h, w = b["depth"].shape[:2]
    y, x = np.mgrid[:h, :w]
    tangent = np.tan(camera["vertical_fov_radians"] / 2)
    directions = (np.array(camera["forward"]) +
                  ((2 * (x + .5) / w - 1) * tangent * w / h)[..., None] * camera["right"] +
                  ((1 - 2 * (y + .5) / h) * tangent)[..., None] * camera["up"])
    point = np.array(camera["position"]) + b["depth"] * directions
    relative = point - previous["position"]
    depth = relative @ previous["forward"]
    half_height = depth * np.tan(previous["vertical_fov_radians"] / 2)
    # Background depth can be zero; it is excluded by the history-valid mask.
    with np.errstate(divide="ignore", invalid="ignore"):
        px = .5 * ((relative @ previous["right"]) / (half_height * w / h) + 1) * w - .5
        py = .5 * (1 - (relative @ previous["up"]) / half_height) * h - .5
    expected = np.stack((px - x, py - y), axis=-1)
    error = float(np.max(np.abs(expected[valid] - b["motion"][valid])))
    if error > .002:
        raise ValueError(f"Camera/depth reprojection disagrees with motion: {error} pixels")
    return error


def inspect(root):
    records = []
    sheet = Image.new("RGB", (192 * 8, 152 * 3))
    draw = ImageDraw.Draw(sheet)
    for row, scene in enumerate(("hurricane", "demo", "deer")):
        directory = root / scene
        paths = sorted(directory.glob("frame_*/manifest.json"))
        if len(paths) != 8:
            raise ValueError(f"Expected eight completed frames in {directory}")
        frames = [load_frame(path) for path in paths]
        group = f"ntasad_pilot_{scene}_00"
        sequence = SequenceDataset(directory, length=5, sequence_ids=[group])
        patches = SequencePatchDataset(directory, sequence_ids=[group], patch_size=128)
        if len(sequence) != 4 or len(patches) != 4:
            raise ValueError("Expected four five-frame windows")
        for i in range(4):
            sample = patches[i]
            assert sample["buffers"]["radiance"].shape == (5, 3, 128, 128)
            assert not sample["buffers"]["history_valid"][0].any()
        per_frame = []
        for i, (path, frame) in enumerate(zip(paths, frames)):
            b = frame.buffers
            assert all(np.isfinite(v).all() for v in b.values())
            if i == 0:
                assert not b["history_valid"].any()
            normal = np.linalg.norm(b["normal"], axis=-1)
            hit = b["guide_valid"][..., 0] > 0
            motion = np.linalg.norm(b["motion"], axis=-1)
            valid = b["history_valid"][..., 0] > 0
            per_frame.append(dict(frame=i, guide_valid_fraction=float(hit.mean()),
                                  history_valid_fraction=float(valid.mean()),
                                  motion_reprojection_max_error=motion_error(frame),
                                  motion_p50_valid=float(np.median(motion[valid])) if valid.any() else None,
                                  motion_p95_valid=float(np.quantile(motion[valid], .95)) if valid.any() else None,
                                  normal_length_median=float(np.median(normal[hit])) if hit.any() else None,
                                  exr_bytes=(path.parent / "frame.exr").stat().st_size))
            rgb = np.maximum(b["reference"], 0)
            rgb = np.power(rgb / (1 + rgb), 1 / 2.2)
            preview = Image.fromarray(np.uint8(np.clip(rgb, 0, 1) * 255))
            sheet.paste(preview, (i * 192, row * 152))
            draw.text((i * 192, row * 152 + 130), f"{scene} {i}", fill="white")
        records.append(dict(scene=scene, windows=4, frames=per_frame))
    report = dict(status="numeric_checks_passed_visual_review_required", clips=records,
                  references_qualified=False, test_pixels_loaded=False,
                  display_transform="(rgb/(1+rgb))**(1/2.2)")
    (root.parent / "inspection.json").write_text(json.dumps(report, indent=2) + "\n")
    sheet.save(root.parent / "contact-sheet.png")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preview_root", type=Path)
    inspect(parser.parse_args().preview_root)
