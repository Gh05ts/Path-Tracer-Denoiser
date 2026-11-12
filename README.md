# Wavefront denoiser

One **NTASAD spatial/temporal kernel-prediction U-Net** denoises captures from
`../Wavefront-Path-Tracer`. The same network predicts spatial filters at three
scales, a temporal filter for warped history, and their blend weights. Spatial
and temporal filtering train together through one sequence loss and produce one
checkpoint. The earlier independent residual spatial model has been removed.

The paper's second network is the **adaptive sampler**. It is not implemented
here yet; current inputs are uniformly sampled. This is a reconstruction of the
published denoiser, not a claim of paper-quality results or exact author code.
See the [paper audit and model API](docs/ntasad-progress.md) for architecture,
source links, implementation choices and remaining differences.

## Set up

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train,test]'
```

The local environment is `/tmp/ntasad-venv/bin/python`. Use that interpreter in
place of `python3` if the environment is not activated.

## View captured EXR images

Export both the noisy `radiance` and cleaner `reference` layers as PNGs:

```bash
python3 tools/generate_images.py /path/to/frame.exr
python3 tools/generate_images.py /path/to/frame.exr --output /path/to/previews/
python3 tools/generate_images.py /path/to/frame.exr --output /path/to/bmw.png
```

By default, this writes `frame-radiance.png` and `frame-reference.png` beside
the EXR. A directory changes their location; a PNG filename changes their base
name (`bmw-radiance.png` and `bmw-reference.png`). Existing PNGs are replaced.
The tool reads the EXR directly without a manifest and requires both named RGB
layers. PNGs use Reinhard tone mapping and gamma 2.2 for display; HDR data in the
EXR remains unchanged. Dependencies are included in the setup above.

Both layers and dataset contact sheets use exposure `1.0`. Bistro's sun/lamp
balance is authored in its derived glTF lighting settings, with no scene-specific
display transform or renderer branch. Override PNG exposure explicitly with
`--exposure MULTIPLIER`; this never changes HDR data or lighting balance.

## Dataset and training

The current [dataset workflow](dataset_generation/README.md) uses v4: 1080p,
4-SPP noisy inputs, 1024-SPP references, eight-frame camera clips, and independent
train/validation/test scenes. Capture commands and frame-level resume are
unchanged by model unification. A capture already running can finish normally.

After train/validation captures are complete, run from this directory:

```bash
python3 tools/train_dataset.py --device cuda

# Survive an SSH disconnect:
DATASET_PYTHON=/tmp/ntasad-venv/bin/python \
  ./tools/run_dataset_background.sh train --device cuda
```

This trains one model into `runs/denoiser-v4-denoiser/`, with `model.pt` selected by
validation loss and `last.pt` for epoch-boundary resume. It creates
`../dataset/denoiser-v4/train.json` if missing, including for a capture started
before unification. Existing obsolete `train-spatial.json` and
`train-recurrent.json` sidecars are ignored; capture configs, locks and images
are unchanged. Edit `train.json` to change training settings. No existing
checkpoint is loaded unless `--resume` is supplied.

Five-frame training windows contain one noisy initialization frame and four
recurrent network steps. Back-propagation passes through the entire window.
The loss gives equal weight to spatial L1 on the final frame and temporal L1 on
the final two-frame difference, in the paper's gamma/log domain. Defaults are
Adam at 0.001, Xavier initialization, 1000 epochs, and shared 128×128 sequence
crops/flips/90-degree rotations. Five fixed 256×256 validation crops per window
bound memory for the 1080p dataset; these are crop-based validation losses.
References retain their declared fixed-budget, unqualified convergence status.

```bash
# Increase epochs in train.json first; resume into a new run directory:
python3 tools/train_dataset.py --device cuda \
  --resume runs/denoiser-v4-denoiser/last.pt \
  --output runs/denoiser-v4-resumed

# Generic sequence config, independent of the capture launcher:
python3 -m wavefront_denoiser train configs/train.json runs/denoiser-new --device auto
```

The generic config is a template; supply actual dataset paths and sequence IDs,
or a whole-scene `split_manifest`. Test pixels are excluded from training and
checkpoint selection. Training checks the declared reference policy and exact
train/validation inventory; strict evaluation checks the frozen split again.

## Inference and evaluation

One checkpoint supports both modes. Sequence inference is the paper's recurrent
path: each independent clip starts from its own noisy frame zero, then subsequent
frames use the learned spatial and temporal filters. No reference image enters
the model, and only the current frame and history stay resident.

```bash
python3 -m wavefront_denoiser denoise-sequence \
  runs/denoiser-v4-denoiser/model.pt ../dataset/denoiser-v4 runs/video-output \
  --sequence-id bmw_view_00_linear_slow --device cuda

# A still uses the same U-Net's learned spatial branch with history disabled:
python3 -m wavefront_denoiser denoise \
  runs/denoiser-v4-denoiser/model.pt \
  ../dataset/denoiser-v4/bmw_view_00_linear_slow/frame_000000 \
  runs/still-output --device cuda

python3 -m wavefront_denoiser evaluate runs/denoiser-v4-denoiser/model.pt \
  --split-manifest ../dataset/denoiser-v4/split.json --partition validation --device cuda

# Use the reserved test partition only after checkpoint selection is frozen:
python3 -m wavefront_denoiser evaluate runs/denoiser-v4-denoiser/model.pt \
  --split-manifest ../dataset/denoiser-v4/split.json --partition test --device cuda
```

Outputs contain linear HDR `denoised.exr`, display-only comparison PNGs, and
JSON provenance/metrics. Sequence initialization frames are labelled as noisy
initialization and excluded from denoised-image averages. Temporal metrics compare
finite differences in the paper's transformed domain, without motion warping.
Still inference accepts a named-channel EXR without reference or temporal guides;
sequence inference requires motion/history channels and consecutive manifests.
The still fallback is not a separately trained model or the paper's recurrent
warm-up procedure. Full-resolution runtime uses a reference PyTorch kernel
implementation; tiled inference and real-time performance are not established.

## Capture tools and validation

The renderer can be built with optional dataset/OpenEXR support:

```bash
cmake -S ../Wavefront-Path-Tracer -B ../Wavefront-Path-Tracer/build \
  -DPATHTRACER_BUILD_DATASET_CAPTURE=ON
cmake --build ../Wavefront-Path-Tracer/build --target dataset_capture frame_export_tests --parallel 2
python3 -m wavefront_denoiser validate ../dataset/denoiser-v4
python3 -m pytest -q
```

The frame contract is in [frame-format.md](docs/frame-format.md). Reference
replicas and whole-scene audit tools are documented in
[reference convergence and splits](docs/reference-convergence-and-splits.md).
Historical spatial-baseline and lighting-study reports describe retired
experiments; their old model checkpoints and training commands are unsupported.
`lighting-study` now retains capture preparation only.

The uniform captures support moving cameras through static geometry. Object
animation, adaptive sample layers, zero-sample inputs, an adaptive sampling
network and joint sampler/denoiser optimization remain future work. First-hit
motion does not fully describe reflected or refracted appearance, and background
history is invalidated by the current frame contract.

## Distributed capture

The [distributed clip generator](distributed/README.md) lets separate GPU
machines resume the same locked v4 dataset through a shared R2 task queue.
Each task covers an entire eight-frame clip, including its required layers and
decoded cache. Existing local progress and archived scenes are reused.
