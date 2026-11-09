# NTASAD denoiser audit and implementation

Target: Hasselgren et al., *Neural Temporal Adaptive Sampling and Denoising*,
Eurographics 2020. Sources used for this audit:

* [Main paper](https://research.nvidia.com/sites/default/files/pubs/2020-05_Neural-Temporal-Adaptive/NTASAD_preprint.pdf), Figure 3 and sections 2.1–2.2.
* [Supplement](https://research.nvidia.com/sites/default/files/publications/supplemental.pdf), section 1 and Figure 2.

This is a reconstruction from the publication, not an independently verified
copy of the authors' implementation. The denoiser now follows substantially
more of the published specification than the original compact prototype.
Adaptive sampling, joint sampler/denoiser training, the Falcor renderer, and
reproduction of the authors' training scenes remain outside this stage.
The original residual spatial U-Net is a separate baseline and is unchanged.
No trained NTASAD checkpoint or captured training dataset is included here.

## Audit

| Component | Published description | Current implementation |
| --- | --- | --- |
| Encoder | Figure 3: 48 channels, scales through I/32 | Default 48 channels; two full-resolution convolutions and five downsampling stages with one convolution each |
| Decoder | Figure 3: 96-channel coarse levels, 64 at full resolution | Five stages with two convolutions each; 96 channels at I/16, I/8, I/4 and I/2, then 64 at I |
| Convolutions | 3x3 with ReLU; final convolutions predict filter parameters | 3x3 throughout; ReLU in feature blocks, unactivated prediction heads |
| Skip connections | Concatenated U-Net skips; supplement contrasts these with the small sampler's residual skips | Concatenate encoder outputs at matching resolutions |
| Kernel outputs | 26 at I/4 and I/2; 51 at I | Two 5x5 spatial kernels with scale blends; finest spatial and temporal 5x5 kernels with temporal blend |
| Reconstruction | Equation (3), recursively combined scales | Average-pool image pyramid, nearest upsampling, recursive low-frequency replacement |
| Reprojection | Equation (1), bilinear grid_sample | Backward pixel-space motion, bilinear warp, preserved gradients |
| Guides | Noisy RGB, history RGB, normals, depth, motion and albedo | 15 channels; signed world-space normals, forward depth and pixel-unit motion from capture |
| Radiance | Clamp to [0,65535], then log(1+x)^(1/2.2); log-space training | Same input transform; filtering and recurrence in transformed space; linear HDR at the public output API |
| Initialization | Xavier; initialize first recurrent frame from noisy uniform samples | Xavier-uniform weights, zero biases; frame zero is noisy RGB initialization, followed by network steps |
| Sequence loss | Equal spatial and temporal L1; final frame and last two-frame difference | Equation (4) in transformed space; temporal differences are not motion-warped |
| Training windows | Typically five frames: initialization plus four unrolled iterations | Five by default; no history detach within a window |
| Augmentation | Random crops, x/y flips and 90-degree rotations | Shared crop and augmentation across inputs, references and guides; motion components transformed consistently |
| Optimization | Adam, initial learning rate 0.001, 1000 epochs | Those defaults; batch size 4 and 128-pixel patches are configurable local choices |

### Details the sources do not fully specify

The figure is not executable code. Exact skip attachment, network pooling,
feature normalization, border padding, kernel normalization and blend activation
are not fully specified in the text. We use average pooling, nearest decoder
upsampling, concatenation after upsampling, softmax kernel weights and sigmoid
blends. Coarse-scale blend logits are nearest-upsampled to control the next finer
scale. These choices must not be presented as independently verified author code.
The reconstructed default model has **1,114,311 parameters**, including the three
kernel heads. The paper mentions approximately 850k weights for its modified
DASR comparison; we have not reconciled that approximate count with the figure,
so exact parameter-count equivalence is not established.

Guide normalization/coordinate spaces are not fully documented. Our depth and
motion retain native capture units; normals remain world-space and signed.
Motion is now a network input, rather than only a warp input. Sample counts and
validity masks are checked/used for validity but are not extra learned features.
Albedo remains RGB for the denoiser; the paper's grayscale conversion is described
for the deferred adaptive sampler. No albedo demodulation is applied.

We interpret the stated radiance preprocessing and log-space training as
filtering and recurring in that transformed space. History is encoded **before**
bilinear warping. A port that filters linear radiance and only transforms its loss
would be a different implementation; the publication does not provide a complete
operator-by-operator code listing resolving every domain boundary.

Additional boundary conventions are explicit:

* Replicate-pad features and spatial radiance to multiples of 32, then crop
  output. Motion is evaluated in the original history image, before padding.
* Exclude invalid history taps and renormalize kernels; an invalid center forces
  temporal weight to zero even when neighboring history is valid. Background
  history remains invalid under the existing renderer contract.
* Equation (3) can overshoot. Clamp transformed reconstruction to the range
  corresponding to linear [0,65535], ensuring finite HDR output and consistent
  streaming/sequence state. The paper specifies the input clamp but does not
  specify output overshoot handling.
* The fractional radiance power has an undefined/infinite derivative at black.
  Use a zero subgradient at exactly zero without biasing positive values.
* Rotate/flip motion-vector components with image coordinates. World-space normal
  components retain their values while their pixel locations move.

Kernel filtering uses a memory-heavy `unfold` reference implementation, not an
optimized Falcor/CUDA path. No real-time performance or paper-quality result is
claimed. Mixed precision and zero-sample adaptive inputs are not implemented.

## Model API

```python
import torch
from wavefront_denoiser import SequenceDataset
from wavefront_denoiser.recurrent import INPUT_CHANNELS, TemporalKernelDenoiser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = SequenceDataset("data/temporal", length=5, sequence_ids=["camera_train"])
batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1)))
buffers = {name: batch["buffers"][name].to(device) for name in INPUT_CHANNELS}
model = TemporalKernelDenoiser().to(device)  # default widths from Figure 3
linear_rgb = model.forward_sequence(buffers)  # N,T,3,H,W
training_rgb = model.forward_sequence(buffers, transformed=True)
```

`model(frame_buffers, history=previous_linear_output)` supports streaming with
NCHW buffers. `history=None` initializes from current noisy RGB without a network
pass. `forward_sequence` always begins with fresh history and accepts optional
binary `reset[N,T]` to initialize individual batch items again. There is no hidden
state or input mutation, and gradients remain attached between frames.
Width overrides retain the stage topology for tests/ablations; they are not the
paper's default widths. The example above constructs an untrained model.

## Train and resume

After creating completed camera trajectories and editing their sequence IDs:

```bash
python -m wavefront_denoiser train-recurrent \
  configs/recurrent_train.json runs/temporal-first --device auto
python -m wavefront_denoiser train-recurrent \
  configs/recurrent_train.json runs/temporal-resumed --device auto \
  --resume runs/temporal-first/last.pt
```

`configs/recurrent_train.json` is a template: its dataset path and sequence IDs
must match captures that have not yet been generated. Inputs and references are
paired captures, currently uniform positive SPP. Crop size must fit the images.
Each training access selects one shared crop/augmentation for the entire window;
references and guides stay aligned. Crop-local history bounds are invalidated.

Validation defaults to full-image windows without random augmentation and computes the
same final-frame/final-difference loss. `validation_patch_size` optionally selects
deterministic center/corner crops, with `validation_crops_per_window` from 1–5.
The new 1080p dataset configuration uses five 256-pixel validation crops to bound
GPU memory; those losses are crop metrics, not full-resolution evaluation scores.
It does not yet perform long-video metric
aggregation. Training and validation IDs are explicit and disjoint. For strict
whole-scene separation, supply `split_manifest` instead of manual IDs; the existing
metadata audit and declared reference policy are enforced without loading test
pixels. Manual IDs alone do not establish scene independence.

`model.pt` is the best validation checkpoint **within this invocation**, including
its initial model as a candidate; `last.pt` is the latest completed epoch for
resume. A resumed invocation selects its best among its starting checkpoint and
subsequent epochs, not among weights from the previous run that were not loaded.
History and summary JSON report that scope. Resume writes to a new directory;
`epochs` is the total target epoch count, not an additional number of epochs.
Other configuration and the train/validation manifest/image inventory must match.
No gradient clipping, scheduler, hue permutation or grayscale augmentation is
added to this kernel-prediction training path. Crop/batch size, Adam defaults and
validation selection are implementation choices where the paper gives no detail.

## Checkpoints

`recurrent_checkpoint` uses the separate `wavefront-recurrent-checkpoint` format.
The architecture is now `ntasad_uniform_unet_v2`, feature version 2. Earlier
compact v1 recurrent checkpoints are rejected rather than silently changing
preprocessing. Spatial checkpoints retain their existing behavior.

```python
from wavefront_denoiser.recurrent_checkpoint import (
    save_recurrent_checkpoint, load_recurrent_checkpoint, restore_recurrent_training,
)

save_recurrent_checkpoint(
    "runs/temporal/last.pt", model, optimizer=optimizer,
    epoch=completed_epochs, global_step=completed_steps,
    config=config, dataset_split=split_audit, generator=loader_generator,
)
model, metadata = load_recurrent_checkpoint("runs/temporal/last.pt", device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loader_generator = torch.Generator()
progress = restore_recurrent_training(model, optimizer, metadata, generator=loader_generator)
```

Weights are stored on CPU and loaded with restricted `weights_only=True`
deserialization, compatibility checks and finite-weight checks. A unique
partial file is atomically replaced into the target; this is not an fsync
power-loss guarantee. A weights-only save omits optimizer and RNG state and
cannot resume training. Loading returns an eval model; restoring training changes
it to training mode.

Resumable saves include optimizer type, named parameter groups, moments and
hyperparameters, progress, JSON configuration/provenance, and Python, NumPy,
PyTorch CPU, initialized CUDA and optional DataLoader-generator RNG states.
Save after a completed epoch/optimizer step without pending gradient accumulation.
Recurrent image history, worker state, schedulers, AMP scalers, pending gradients
and mid-epoch sampler positions are not saved. Dataset metadata is automatically
compared on CLI resume; standalone checkpoint APIs leave that comparison to callers.
CUDA RNG restoration requires matching device count. The standalone API accepts
`restore_rng=False` for deliberate device migration without exact random replay;
CLI resume currently restores RNG and expects a matching environment.

Tests cover transforms, black gradients, model stages and dimensions, kernels,
reprojection, visibility, cross-frame/reset gradients, coherent motion augmentation,
loss placement, checkpoint compatibility, interrupted writes and exact next-update
resume. A real synthetic two-epoch training run is compared with one epoch plus
resume. Test devices select CUDA when available and otherwise CPU.

## Deferred work and dataset requirements

### Dataset follow-up (2026-11-09)

The [dataset plan](ntasad-dataset-plan.md) inventories all four local glTF assets
and proposes independent scene families, eight-frame clips and five-frame
training windows. A [small RTX 3090 capture pilot](ntasad-pilot-results.md) now
contains three eight-frame train/validation previews and 24 independent
reference-replica jobs in ignored local run directories. Sequence/crop and
geometric-motion checks passed. Hurricane and demo qualified at 1,024 SPP, but
deer failed the fixed highlight rule even at 4,096 SPP; the sweep selected no
dataset-wide reference budget. Test pixels remain unopened. Improve framing and
investigate background-heavy highlight measurements before a second pilot or
bulk capture. No new model was trained, and adaptive sampling remains deferred.

### Bistro renderer follow-up

Bistro has since been downloaded and its DDS textures converted to PNG. The
renderer now imports its specular/glossiness materials and punctual lights;
see [renderer support and checks](../../Wavefront-Path-Tracer/docs/gltf-materials-and-lights.md).
A single 320×180 appearance preview is in `runs/bistro-material-light-preview-v2`.
It has no dataset partition assignment and its 64-SPP reference is unqualified.
This renderer change requires new reference qualification before training captures.

### HDR environment follow-up

HDR environments are now supported through the scene manifest, including
importance sampling, MIS, intensity and yaw rotation. See
[renderer configuration and validation](../../Wavefront-Path-Tracer/docs/hdr-environments.md).
The Breakfast Room preview uses its supplied HDR map at 320×180 and 128 reference
SPP; it is an appearance check, not a qualified training sequence or scene split.
The final preview is in `runs/breakfast-room-hdr-preview-v2`. Standard/persistent
captures agree within FP32 tolerance, with identical guides. Eight focused
renderer test executables passed, including HDR energy, shadows and distributed
map transfer. Adaptive per-pixel sampling remains deferred.

### Production dataset workflow follow-up

[`dataset_generation/`](../dataset_generation/README.md) now defines the requested
1920×1080 dataset: BMW, Bistro, Living Room, Breakfast Room and Dragon train;
Bathroom and Classroom2 validate; Sponza is reserved for final evaluation.
Six motion types and three/two camera anchors produce 90 train, 24 validation
and 12 test eight-frame clips (1008 planned frames). Training references are a
declared 1024-SPP budget with independent 4-SPP inputs. A separate report script
requests 4096 SPP while preserving those exact noisy sample IDs and cameras.
New-scene convergence qualification remains outstanding; a representative
train/validation-only replica plan is generated without fabricating a policy.

Capture resumes per completed, checksum-verified frame, with camera/settings,
renderer binary and all referenced asset bytes locked. An unfinished frame is
rendered again; renderer sample checkpoints and distributed capture are not
integrated. The `nohup` background launcher was checked with an explicit SIGHUP.
Fresh spatial/recurrent training entry points use the generated independent
split. Recurrent training keeps five-frame shared 128-pixel crops/augmentations,
decodes/crops one frame at a time, and uses five fixed 256-pixel validation crops
for this dataset. Older configurations still default to full-frame validation,
and their checkpoint configuration can resume.

The new pilot contains **two**, not eight, completed BMW frames at 1080p and
4+1024 SPP in `../../dataset/denoiser-v1`. Capture plus full verification took 370.5 seconds;
a repeat captured zero frames and skipped both. Frame one has 98.85% valid
history and 28.26-pixel 95th-percentile motion. A separate low-resolution framing
check rendered all 1008 planned camera poses; after correcting Bistro/Breakfast
Room wall crossings it flagged no near-surface camera poses. Sheets and details
are in `../../dataset/denoiser-v1-cameras/`; pilot metrics are in
`../../dataset/denoiser-v1/pilot-results.json`. These checks do not establish convergence
or denoising quality. The GPU-enabled suite passed **155 tests**, with one optional
cross-language fixture skipped. Out-of-order frame-range captures reproduce
uninterrupted images exactly, and report-budget changes preserve noisy inputs.
Bulk generation, complete production clips and production retraining have not
been run. Adaptive sampling remains deferred.

### Persistent output and motion profiles follow-up

Executable entry points now live in `tools/`: `generate_dataset.py`,
`generate_report_references.py`, `check_dataset_cameras.py`, `train_dataset.py`
and `run_dataset_background.sh`. Image/guide/preview/log defaults are under the
workspace's `../dataset/`; `/tmp` remains only the development Python environment
and renderer build. The v1 pilot and diagnostic captures were moved and retained;
both completed BMW EXRs verified and skipped without rendering again.

The default source is now `dataset_generation/plan-v2.json`. It retains the same
126 clips / 1008 planned frames and independent scene partitions, cycling slow
0.6×, medium 1.4× and fast 2.8× amplitudes across motion types and viewpoints.
Every training motion appears at all three speeds across its three anchors.
V2 has separate output, group IDs and sample ranges; it does not mix with v1.
After moving one Bathroom start pose farther from a wall, all 1008 low-resolution
GPU camera checks completed with no near-surface flags. Median frame-p95 motion
estimates for BMW linear slow/medium/fast clips are about 17/39/75 pixels at 1080p.
These are scaled diagnostic estimates, not native-1080p v2 captures. Across all
clips the corresponding profile medians are about 10/22/44 pixels.

Camera-check sheets explicitly label their 4-SPP diagnostic references. Their
paired radiance/albedo GIFs expose motion without stochastic guide noise.
Appearance previews use 4+16 SPP; real pilot/capture uses 4+1024 SPP. The v1 pilot
now also has a labelled `comparison.png` and full-resolution noisy/reference
viewing PNGs. No trained denoiser output is claimed. The paper's animated-object
cases remain uncovered by this static-geometry capture workflow, and its exact
camera speeds are not specified. The relocated launcher again completed a plan
job after explicit SIGHUP. The Python suite passed 151 tests with 12 optional
GPU/fixture tests skipped; the separate actual GPU camera checks validated all
v2 frames. Bulk v2 capture and production retraining have not started.

### Original implementation-stage requirements

The paper reports nine training scenes, 16–25 eight-frame clips per scene,
1,368 frames total, approximately 1,024-SPP training references, and 4,096-SPP
evaluation references. A proposed 4,096-SPP training reference budget exceeds its
training budget, but should still be convergence-qualified on representative
new scenes before bulk rendering. Capture diverse camera trajectories and use
explicit scene partitions. No new scene assets or training images were generated
by this implementation change.

The existing renderer supplies motion for static geometry and moving cameras.
It does not provide animated-object motion, and first-hit geometry motion does
not fully describe reflected/refracted appearance or moving illumination.
That remains a dataset/renderer difference from the paper's animated scenes.

Adaptive sample prediction and joint training are intentionally deferred. They
require independent sample layers (1,2,4,8,16,32 SPP), the sampling-gradient
approximation, budget normalization, and per-pixel/zero-sample renderer support.
The current frame schema requires uniform positive sample counts. None of those
renderer or export-contract changes are part of this denoiser-only work.
