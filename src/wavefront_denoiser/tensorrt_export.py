"""Build and numerically verify an FP32 TensorRT engine from a spatial export."""

import argparse
import json
from pathlib import Path
import tempfile
import os

import numpy as np

from .onnx_export import INPUT_NAME, OUTPUT_NAME, example_frame, sha256


def cuda_checked(result):
    if int(result[0]) != 0:
        raise RuntimeError(f"CUDA operation failed: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


def infer(engine, frame):
    """One explicit-stream inference with owned buffers; suitable for validation."""
    from cuda.bindings import runtime as cuda

    frame = np.ascontiguousarray(frame, dtype=np.float32)
    if tuple(frame.shape) != tuple(engine.get_tensor_shape(INPUT_NAME)):
        raise ValueError("Input shape does not match the fixed-resolution TensorRT engine")
    output = np.empty(tuple(engine.get_tensor_shape(OUTPUT_NAME)), dtype=np.float32)
    stream = cuda_checked(cuda.cudaStreamCreate())
    buffers = []
    try:
        device_input = cuda_checked(cuda.cudaMalloc(frame.nbytes))
        buffers.append(device_input)
        device_output = cuda_checked(cuda.cudaMalloc(output.nbytes))
        buffers.append(device_output)
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("Cannot create TensorRT execution context")
        if not (context.set_tensor_address(INPUT_NAME, device_input)
                and context.set_tensor_address(OUTPUT_NAME, device_output)):
            raise RuntimeError("Cannot bind TensorRT buffers")
        cuda_checked(cuda.cudaMemcpyAsync(device_input, frame.ctypes.data, frame.nbytes,
                                         cuda.cudaMemcpyKind.cudaMemcpyHostToDevice, stream))
        if not context.execute_async_v3(stream):
            raise RuntimeError("TensorRT inference failed")
        cuda_checked(cuda.cudaMemcpyAsync(output.ctypes.data, device_output, output.nbytes,
                                         cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream))
        cuda_checked(cuda.cudaStreamSynchronize(stream))
        return output
    finally:
        cuda.cudaStreamSynchronize(stream)
        for pointer in reversed(buffers):
            cuda.cudaFree(pointer)
        cuda.cudaStreamDestroy(stream)


def build_engine(export, output, *, workspace_mib=1024):
    import onnxruntime as ort
    import tensorrt as trt
    from cuda.bindings import runtime as cuda

    export, output = Path(export), Path(output)
    report_path = output.with_suffix(output.suffix + ".json")
    if output.exists() or report_path.exists():
        raise FileExistsError(f"Engine or report destination already exists: {output}")
    if type(workspace_mib) is not int or workspace_mib < 1:
        raise ValueError("workspace_mib must be positive")
    manifest = json.loads((export / "manifest.json").read_text())
    if (manifest.get("format") != "wavefront-denoiser-onnx" or manifest.get("version") != 1
            or manifest.get("abi") != "dense-spatial-v1" or manifest.get("opset") != 17
            or manifest.get("input", {}).get("name") != INPUT_NAME
            or manifest.get("output", {}).get("name") != OUTPUT_NAME):
        raise ValueError("Expected a dense-spatial-v1 export manifest")
    onnx_path = export / "denoiser.onnx"
    if sha256(onnx_path) != manifest["onnx_sha256"]:
        raise ValueError("ONNX checksum does not match its export manifest")
    logger = trt.Logger(trt.Logger.WARNING)
    with trt.Builder(logger) as builder, builder.create_network(0) as network, \
            trt.OnnxParser(network, logger) as parser:
        if not parser.parse_from_file(str(onnx_path)):
            raise ValueError("TensorRT ONNX parse failed:\n" + "\n".join(
                str(parser.get_error(i)) for i in range(parser.num_errors)))
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mib * 1024 * 1024)
        # Establish FP32 parity before introducing reduced-precision tactics.
        config.clear_flag(trt.BuilderFlag.TF32)
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT engine build failed")
        plan = bytes(serialized)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan)
    if engine is None:
        raise RuntimeError("Cannot reload TensorRT engine")
    if (engine.num_io_tensors != 2 or list(engine.get_tensor_shape(INPUT_NAME)) != manifest["input"]["shape"]
            or list(engine.get_tensor_shape(OUTPUT_NAME)) != manifest["output"]["shape"]):
        raise ValueError("Engine tensors do not match the export ABI")
    for name in (INPUT_NAME, OUTPUT_NAME):
        if (engine.get_tensor_dtype(name) != trt.float32
                or engine.get_tensor_location(name) != trt.TensorLocation.DEVICE
                or engine.get_tensor_format(name) != trt.TensorFormat.LINEAR):
            raise ValueError("Engine requires FP32 linear device input/output")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"])
    height, width = manifest["input"]["shape"][-2:]
    frame = example_frame(height, width).numpy()
    checks = []
    for name, value in (("hdr", frame), ("black", np.zeros_like(frame)),
                        ("alternate", example_frame(height, width, seed=104).numpy())):
        expected = session.run([OUTPUT_NAME], {INPUT_NAME: value})[0]
        actual = infer(engine, value)
        np.testing.assert_allclose(actual, expected, rtol=5e-4, atol=5e-4)
        checks.append({"case": name, "max_abs_error": float(np.abs(actual - expected).max())})
    properties = cuda_checked(cuda.cudaGetDeviceProperties(cuda_checked(cuda.cudaGetDevice())))
    report = {"format": "wavefront-denoiser-tensorrt", "version": 1,
              "abi": manifest["abi"], "checkpoint_sha256": manifest["checkpoint_sha256"],
              "onnx_sha256": manifest["onnx_sha256"], "tensorrt_version": trt.__version__,
              "gpu": properties.name.decode(), "precision": "fp32_tf32_disabled",
              "input": manifest["input"], "output": manifest["output"],
              "verification": {"backend": "tensorrt_vs_onnxruntime", "rtol": 5e-4,
                               "atol": 5e-4, "cases": checks}}
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=output.name + ".partial-", dir=output.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(plan)
        if output.exists() or report_path.exists():
            raise FileExistsError(f"Engine or report destination already exists: {output}")
        # Publication is after verification; a failed build cannot replace a model.
        os.rename(temporary, output)
        report["engine_sha256"] = sha256(output)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workspace-mib", type=int, default=1024)
    args = parser.parse_args()
    report = build_engine(args.export, args.output, workspace_mib=args.workspace_mib)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
