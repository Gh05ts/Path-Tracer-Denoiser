# Wavefront denoiser

The recurrent kernel-prediction denoiser, paper audit, sequence training, and
resumable checkpoints are described in [NTASAD progress](docs/ntasad-progress.md).

The new BMW/Bistro/interior/Dragon dataset workflow, camera trajectories,
1080p capture, frame resume, SSH background launcher and fresh training commands
are in [dataset generation](dataset_generation/README.md). Scripts live in `tools/`;
image/guide outputs default to the workspace’s `../dataset/` directory.

A separate research project for denoising captures from the CUDA wavefront
renderer in `../Wavefront-Path-Tracer`. It includes a spatial HDR model, training and evaluation
commands, single-image inference, and a validated capture pipeline. The spatial
model processes each still image independently. The recurrent model consumes
motion and temporal history, with a separate `train-recurrent` command. Its
five-frame crops, loss and default architecture now follow the paper's denoiser
specification, with unresolved implementation details documented in the audit.

Whole-scene train/validation/test splits and measured reference budgets are
available in [reference convergence and independent splits](docs/reference-convergence-and-splits.md).

## Set up

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
# Add the spatial model, PyTorch, and preview dependencies:
python -m pip install -e '.[train]'
```

Build the optional capture executable from the renderer directory:

```bash
cmake -S ../Wavefront-Path-Tracer -B ../Wavefront-Path-Tracer/build -DPATHTRACER_BUILD_DATASET_CAPTURE=ON
cmake --build ../Wavefront-Path-Tracer/build --target dataset_capture frame_export_tests --parallel 2
```

The renderer target requires the OpenEXR C++ development package and pkg-config.
The normal renderer build has no new dependencies when capture is disabled.
Set `CMAKE_CUDA_ARCHITECTURES` for your GPU when configuring a new build.

## Independent dataset

The new `data/spatial-independent-v2` dataset uses 4-SPP inputs and 8,192-SPP
references: eight demo/prism training images, two deer validation images, and
two held-out Sponza test images. The budget was selected from independent
reference replicas on train/validation scenes; the test scene was excluded.

```bash
python -m wavefront_denoiser split-audit configs/dataset_split_v2.json
python -m wavefront_denoiser train \
  configs/spatial_independent_train_v2.json runs/spatial-independent-new --device auto
```

The audit reads metadata only. Training uses the declared train/validation
partitions and records the split provenance in `model.pt`. The independent
trained run is in `runs/spatial-independent-v2`; its model is
selected using deer validation before the frozen Sponza evaluation. See
[the experiment results and limits](docs/independent-baseline-results.md).
Calibration results, replicas, and the plot are in `runs/reference-convergence-v2`.
Version 1 is retained for provenance; its deer cameras missed the mesh, so its
validation metrics are unsuitable as a meaningful unseen-object check.

## Controlled lighting study

A matched-budget sky-only versus diverse-lighting experiment is available
through `lighting-study`. It holds scene surfaces and cameras fixed and reports
results by lighting condition across three paired training seeds. See
[the protocol, results and interpretation](docs/lighting-experiment.md).

## Spatial denoising

A small trained checkpoint and dataset are included in this workspace under
`runs/spatial-baseline-hdr` and `data/spatial-baseline`. These generated artifacts
are excluded from Git. Run inference on one still image:

```bash
python -m wavefront_denoiser denoise \
  runs/spatial-baseline-hdr/model.pt \
  data/spatial-baseline/prism_validation/frame_000000 \
  runs/denoised-prism --device auto
```

The new output directory receives linear HDR `denoised.exr`, a side-by-side
`comparison.png`, and `report.json`. Inference also accepts an input-only EXR
with the named spatial guide channels; no high-sample reference is required.
The existing capture tool produces paired images for supervised training.

Train another run and evaluate selected validation groups:

```bash
python -m wavefront_denoiser train configs/spatial_train.json runs/spatial-new --device auto
python -m wavefront_denoiser evaluate runs/spatial-new/model.pt data/spatial-baseline \
  --sequence-id demo_spatial_validation --sequence-id prism_spatial_validation --device auto
```

`auto` selects CUDA when available, otherwise CPU. Outputs must use new
directories. The small example trains on eight 128x96 views and validates on
four separate views of the same two scenes, using 4-SPP inputs and 512-SPP
references. It is an initial baseline, not a general-purpose trained denoiser.

To regenerate that dataset, use new output directories or first relocate the
existing example. From the renderer root:

```bash
for name in demo_train demo_validation prism_train prism_validation; do
  build/dataset_capture "../denoiser/configs/spatial_${name}_capture.json" \
    "../denoiser/data/spatial-baseline/${name}"
done
```

Capture frame ordering is ignored by spatial training. The `sequence_id`
metadata names select disjoint capture groups, rather than temporal inputs.
See [the model, loss, metrics, and limits](docs/spatial-baseline.md).

## Capture and validate a short sequence

An RTX 3090 capture is included at `data/example-demo` for immediate reader
validation after installation:

```bash
python -m wavefront_denoiser validate data/example-demo
```

Run the capture command from the renderer root so scene presets resolve:

```bash
cd ../new
build/dataset_capture ../denoiser/configs/demo_capture.json ../denoiser/data/demo-smoke
cd ../denoiser
python -m wavefront_denoiser validate data/demo-smoke
```

Each frame directory contains `frame.exr` and `manifest.json`. Exports reject
existing frame directories. Interrupted `.partial` directories are ignored by
discovery; inspect or relocate them before retrying that output path.
The example's 16-SPP references are for pipeline smoke testing. Check reference
convergence before generating a training dataset.

```python
from wavefront_denoiser import SequenceDataset

dataset = SequenceDataset("data", length=3, sequence_ids=["demo_camera_smoke"])
sample = dataset[0]
print(sample["buffers"]["radiance"].shape)  # T, C, H, W
```

Use explicit scene/trajectory IDs for training and validation splits. Windows
stay within a sequence and start with invalid recurrent history. Without
PyTorch, `SequenceDataset(..., tensors=False)` returns THWC NumPy arrays.

## Validation

```bash
python -m pytest -q
# Cross-language EXR fixture, from the renderer directory:
# build/tests/frame_export_tests /tmp/frame-fixture
# WAVEFRONT_CPP_FIXTURE=/tmp/frame-fixture python -m pytest -q
# Actual CUDA tests (run where the GPU is accessible):
WAVEFRONT_CAPTURE_EXECUTABLE=../new/build/dataset_capture python -m pytest -q tests/test_capture_gpu.py
```

GPU tests compare tiled, full-frame, and persistent execution on partial edge
tiles, check camera history, and exercise HDR photon-map captures. The ordinary
reader tests require no GPU.

## Scope and next steps

The capture tool supports camera motion through static scenes, fixed uniform
SPP, and both existing wavefront backends. Input and reference paths use disjoint
PCG32 sample ranges, also disjoint across frames in a capture invocation. The
scene remains resident across the sequence. A driver and optional photon map
are currently rebuilt per frame.

Guides follow first-hit surface motion. Mirrors, glass, and moving lighting need
careful temporal evaluation; their apparent radiance motion is not described
fully by first-hit geometry. Background history is currently invalidated.
The reference uses the same integrator and fixed photon-map settings as the
input. It is a matched-estimator reference, rather than a claim of an unbiased
physical solution.

The implementation lives in `src/wavefront_denoiser/`; the original top-level
scaffold directories remain available for later experiments. The frame contract is in
[docs/frame-format.md](docs/frame-format.md).

Next: train using the independent split, expand the spatial dataset, and perform
one final unseen-scene evaluation after model selection, followed by recurrent
kernel-prediction training and temporal evaluation.
Distributed capture, adaptive sampling, and render checkpoint integration are
later steps. The capture executable uses the local task API and writes no
renderer checkpoints.
