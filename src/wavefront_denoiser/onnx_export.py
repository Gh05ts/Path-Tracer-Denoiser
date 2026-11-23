"""Fixed-resolution, dense spatial inference ABI for the CUDA renderer.

This exports the existing spatial fallback, including HDR transforms and kernel
reconstruction. It does not change training or silently enable temporal history.
The static tracer intentionally specializes Python validation/shape branches;
the caller must enforce the manifest's input contract before inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import warnings

import numpy as np
import torch
from torch import nn

from .checkpoint import CHECKPOINT_FORMAT, load_checkpoint
from .joint_checkpoint import CHECKPOINT_FORMAT as JOINT_FORMAT, load_joint_checkpoint
from .ntasad_radiance import GAMMA, MAX_ENCODED, MAX_RADIANCE


INPUT_NAME = "frame_linear_v1"
OUTPUT_NAME = "denoised_linear_v1"
CHANNELS = ["radiance.R", "radiance.G", "radiance.B", "albedo.R", "albedo.G",
            "albedo.B", "normal.X", "normal.Y", "normal.Z", "depth"]


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


class SpatialExport(nn.Module):
    """N10HW linear/native guides -> N3HW linear HDR, dense uniform samples.

    Uniform count factors cancel in the spatial pyramid for both supported
    reconstruction versions. One is therefore a sufficient representative count.
    Normals remain signed world-space values; depth is camera-forward distance.
    """

    def __init__(self, denoiser):
        super().__init__()
        self.denoiser = denoiser

    @staticmethod
    def buffers(frame):
        one = torch.ones_like(frame[:, :1])
        return {"radiance": frame[:, :3], "albedo": frame[:, 3:6],
                   "normal": frame[:, 6:9], "depth": frame[:, 9:10],
                   "guide_valid": one, "sample_count": one.to(torch.int64),
                   "motion": torch.zeros_like(frame[:, :2]),
                   "history_valid": torch.zeros_like(one)}

    def reference(self, frame):
        result = self.denoiser.forward_spatial(self.buffers(frame))
        return result if isinstance(result, torch.Tensor) else result.radiance

    def forward(self, frame):
        result = self.denoiser._step(self.buffers(frame), None, initialize=False)
        encoded = result if isinstance(result, torch.Tensor) else result.radiance
        # ONNX has Exp but no Expm1. Keep the trained filtering path unchanged;
        # check the decoded result against the original expm1 implementation.
        return (torch.exp(encoded.clamp(0, MAX_ENCODED).pow(GAMMA)) - 1).clamp(0, MAX_RADIANCE)


def load_spatial(checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    kind = payload.get("format") if isinstance(payload, dict) else None
    # Each loader validates all architecture/version/policy metadata and weights.
    if kind == JOINT_FORMAT:
        model, payload = load_joint_checkpoint(checkpoint)
        model = model.denoiser
    elif kind == CHECKPOINT_FORMAT:
        model, payload = load_checkpoint(checkpoint, torch.device("cpu"))
    else:
        raise ValueError("Expected a supported uniform or joint NTASAD checkpoint")
    if payload["parameter_dtype"] != "float32":
        raise ValueError("Renderer export requires float32 checkpoint weights")
    return SpatialExport(model).eval(), payload


def example_frame(height, width, seed=73):
    generator = torch.Generator().manual_seed(seed)
    frame = torch.rand((1, 10, height, width), generator=generator)
    frame[:, :3] *= 32
    frame[:, 6:9] = 2 * frame[:, 6:9] - 1
    frame[:, 9:] *= 100
    return frame


@torch.inference_mode()
def export_spatial(checkpoint, output, *, width, height):
    """Publish ONNX + ABI manifest only after structural and numerical checks."""
    if any(type(v) is not int or not 1 <= v <= 0x7FFFFFFF for v in (width, height)):
        raise ValueError("width and height must be positive int32 dimensions")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Export destination already exists: {output}")
    # Optional dependencies stay out of ordinary training/runtime imports.
    import onnx
    import onnxruntime as ort

    model, payload = load_spatial(checkpoint)
    frame = example_frame(height, width)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".partial-", dir=output.parent))
    try:
        path = staging / "denoiser.onnx"
        # Fixed-size TorchScript export supports the existing Python validation
        # and unfold reference implementation without modifying trained code.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", torch.jit.TracerWarning)
            torch.onnx.export(model, (frame,), path, input_names=[INPUT_NAME],
                              output_names=[OUTPUT_NAME], opset_version=17,
                              dynamo=False, external_data=False)
        graph = onnx.load(path)
        graph.producer_name = "wavefront-denoiser-export"
        onnx.helper.set_model_props(graph, {"wavefront.abi": "dense-spatial-v1",
                                           "wavefront.checkpoint_sha256": sha256(checkpoint)})
        onnx.checker.check_model(graph, full_check=True)
        onnx.save(graph, path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, min(4, torch.get_num_threads()))
        session = ort.InferenceSession(str(path), sess_options=options,
                                       providers=["CPUExecutionProvider"])
        checks = []
        # Different values detect accidentally constant-folded data branches.
        for name, value in (("hdr", frame), ("black", torch.zeros_like(frame)),
                            ("alternate", example_frame(height, width, seed=104))):
            expected = model.reference(value).numpy()
            actual = session.run([OUTPUT_NAME], {INPUT_NAME: value.numpy()})[0]
            np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=3e-4)
            checks.append({"case": name, "max_abs_error": float(np.abs(actual - expected).max())})
        manifest = {"format": "wavefront-denoiser-onnx", "version": 1,
                    "abi": "dense-spatial-v1", "mode": "spatial_fallback",
                    "checkpoint_sha256": sha256(checkpoint),
                    "checkpoint_architecture": payload["architecture"],
                    "epoch": payload["epoch"], "opset": 17,
                    "onnx": "denoiser.onnx", "onnx_sha256": sha256(path),
                    "input": {"name": INPUT_NAME, "dtype": "float32", "layout": "NCHW",
                              "shape": [1, 10, height, width], "channels": CHANNELS},
                    "output": {"name": OUTPUT_NAME, "dtype": "float32", "layout": "NCHW",
                               "shape": [1, 3, height, width], "color_space": "linear_scene_referred_rgb"},
                    "requirements": {"sampling": "dense_uniform", "history": False,
                                     "albedo": "linear_rgb", "normal": "signed_world_space",
                                     "depth": "camera_forward_distance"},
                    "verification": {"backend": "onnxruntime_cpu", "rtol": 3e-4,
                                     "atol": 3e-4, "cases": checks}}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if output.exists():
            raise FileExistsError(f"Export destination already exists: {output}")
        staging.rename(output)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path, help="New export directory")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    try:
        manifest = export_spatial(args.checkpoint, args.output, width=args.width, height=args.height)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Export failed: {error}\n")
    print(json.dumps({"output": str(args.output), "verification": manifest["verification"]}, indent=2))


if __name__ == "__main__":
    main()
