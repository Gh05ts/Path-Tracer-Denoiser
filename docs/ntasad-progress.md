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
There is now one denoiser model. The original residual spatial U-Net, its trainer
and checkpoint format have been removed; the adaptive sampler remains deferred.
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
from wavefront_denoiser.model import INPUT_CHANNELS, NTASADDenoiser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = SequenceDataset("data/temporal", length=5, sequence_ids=["camera_train"])
batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1)))
buffers = {name: batch["buffers"][name].to(device) for name in INPUT_CHANNELS}
model = NTASADDenoiser().to(device)  # default widths from Figure 3
linear_rgb = model.forward_sequence(buffers)  # N,T,3,H,W
training_rgb = model.forward_sequence(buffers, transformed=True)
```

`model(frame_buffers, history=previous_linear_output)` supports streaming with
NCHW buffers. `history=None` initializes from current noisy RGB without a network
pass. `forward_spatial(frame_buffers)` runs the same learned U-Net with zero
history for a still image; it does not train or load a separate model.
`forward_sequence` always begins with fresh history and accepts optional
binary `reset[N,T]` to initialize individual batch items again. There is no hidden
state or input mutation, and gradients remain attached between frames.
Width overrides retain the stage topology for tests/ablations; they are not the
paper's default widths. The example above constructs an untrained model.

## Train and resume

After creating completed camera trajectories and editing their sequence IDs:

```bash
python -m wavefront_denoiser train \
  configs/train.json runs/temporal-first --device auto
python -m wavefront_denoiser train \
  configs/train.json runs/temporal-resumed --device auto \
  --resume runs/temporal-first/last.pt
```

`configs/train.json` is a template: its dataset path and sequence IDs
must match captures that have not yet been generated. Inputs and references are
paired captures, currently uniform positive SPP. Crop size must fit the images.
Each training access selects one shared crop/augmentation for the entire window;
references and guides stay aligned. Crop-local history bounds are invalidated.

Validation defaults to full-image windows without random augmentation and computes the
same final-frame/final-difference loss. `validation_patch_size` optionally selects
deterministic center/corner crops, with `validation_crops_per_window` from 1–5.
The new 1080p dataset configuration uses five 256-pixel validation crops to bound
GPU memory; those losses are crop metrics, not full-resolution evaluation scores.
Full-sequence evaluation streams the selected clips, excludes noisy initialization
frames, and reports HDR image metrics plus transformed-domain temporal L1.
Training and validation IDs are explicit and disjoint. For strict
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

`checkpoint` uses one `wavefront-denoiser-checkpoint` format for training,
still inference and streaming sequence inference.
The architecture is now `ntasad_spatiotemporal_unet_v1`, feature version 2.
Retired spatial and recurrent checkpoint formats are rejected. Start this
unified model from scratch.

```python
from wavefront_denoiser.checkpoint import (
    save_checkpoint, load_checkpoint, restore_training,
)

save_checkpoint(
    "runs/temporal/last.pt", model, optimizer=optimizer,
    epoch=completed_epochs, global_step=completed_steps,
    config=config, dataset_split=split_audit, generator=loader_generator,
)
model, metadata = load_checkpoint("runs/temporal/last.pt", device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loader_generator = torch.Generator()
progress = restore_training(model, optimizer, metadata, generator=loader_generator)
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

### BMW HDR firefly correction and v3 pilot

The user stopped the v2 bulk capture after spotting white floor specks. CPU
replay traced two reproduced outliers to HDR-sun paths through glass and chrome.
An older triangle-normal bug treated dielectric exits as entries; rough metals
also used an ad hoc fuzzy reflection model without direct-light sampling.
The renderer now preserves dielectric boundary orientation, uses GGX with a
matching PDF/light sampling for nonzero-roughness metals in both execution
modes, corrects photon interface classification, and declares integrator version
4. See the [full diagnosis and remaining limitation](../../Wavefront-Path-Tracer/docs/diagnostics/bmw-hdr-fireflies/README.md).

At the user's request the original `/tmp/ntasad-capture-build` was updated for
both renderer and capture, rather than using a separate build. All 18 CTest
tests passed; the relevant Python suite passed 72 with one optional fixture
skipped, including actual GPU capture/mode/resume checks. Two corrected native
1080p BMW frames at 4+1024 SPP took 388.7 seconds. Floor-region outlier counts
fell 124→2 and 112→0; all buffers are finite and frame-one history validity is
99.29%. A repeat skipped both frames without rendering.

The new default `plan-v3.json` retains v2 cameras, motion profiles, sample ranges
and scene splits, and outputs to `../../dataset/denoiser-v3`. V2 is preserved as
diagnostic data. Training/report defaults follow v3; no production model was
trained and bulk generation has not restarted.

Reference quality remains unresolved: independent 320×180 BMW 1024/4096-SPP
replicas failed the existing 0.1 noise-fraction threshold (whole-image ratios
0.506/0.174; highlights 0.508/0.175). Almost all disagreement is concentrated in
a few glass/HDR-sun highlights. CPU replay confirms a remaining interior-leather
→ glass → sun path with roulette amplification, beyond the repaired orientation
bug. Results and comparisons are under
`../../dataset/diagnostics/bmw-hdr-fireflies/`. The v3 references remain explicitly
fixed-budget/unqualified. Address glass/HDR variance or qualify a higher budget
before another final-data bulk run. Adaptive pixel sampling remains deferred.

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


### Remaining BMW variance: roulette isolation

A current-header CPU-only study at the remaining glass/leather/sun pixel
`(166,38)` used two disjoint 65,536-sample ranges. Disabling roulette within the
same 12-bounce cap reduces the known sun sample from 66,862 to 10,671 (6.27×),
but it still supplies about 96% of one range's red total. Independent range means
remain far apart: 0.169485 versus 0.006548 without roulette, compared with
1.026532 versus 0.006718 with the current schedule. This is selected-pixel
variance evidence, not full-image convergence or a reliable mean estimate.
The renderer executable and sampling policy remain unchanged. See the renderer
diagnostic README's roulette study and `dataset/diagnostics/bmw-hdr-fireflies/roulette-study/`.
Bulk generation/training remain stopped; v3 remains fixed-budget/unqualified,
v2 stays separate, and adaptive sampling remains deferred.


### Glass transport follow-up — integrator 5

Further BMW replay found actual transport errors beyond roulette: a combined
1 mm origin shift and 1 mm intersection cutoff skipped a reproduced 1.865 mm
windshield exit; smooth normals could choose the wrong medium side or send
specular directions through the wrong geometric hemisphere; Schlick strongly
underestimated near-critical glass exit reflection. The renderer now uses
precision-scaled ray spawning, a 1e-6 traversal cutoff, geometric boundary
classification with a consistent normal fallback, exact dielectric Fresnel and
radiance eta-squared transmission. Roulette's depth schedule remains unchanged,
with inverse eta scaling keeping its survival heuristic eta-neutral. Visible
terminal HDR glass branches are integrated with their Fresnel weights rather
than receiving full sun radiance on a rare coin flip. Geometry still occludes
those connections; no clamp, transparent-shadow approximation or scene edit is
used. Both shading modes and photon interfaces use the shared boundary logic.

Integrator version is now **5**. Only the existing `/tmp/ntasad-capture-build`
is rebuilt, with one build job. V2 and the two native v3 pilot frames are
preserved. Their integrator-4 provenance cannot be resumed/mixed with this build.
The default plan remains `plan-v3.json`; its 1024-SPP budget has not been changed.

Four new 320×180 BMW replicas retain the v4 cameras and sample ranges. The
1024-SPP whole-image noise fraction improves 0.506→0.223 but still fails 0.1;
4096 SPP improves 0.174→0.073 (highlights 0.072) and passes for this one calibration
view. This is not qualification of the production budget, native resolution,
other cameras or other scenes. The largest remaining 1024-SPP outlier replays
as glass → interior leather → glass → sun with correctly paired boundaries.
Open LightGlass meshes and a rear-light winding candidate were also recorded
without altering assets. See the renderer diagnostic README and
`../../dataset/diagnostics/bmw-hdr-fireflies/glass-investigation/` for comparisons,
per-path evidence, topology audit and reproducible helpers.

Bulk generation/training remain stopped. Adaptive pixel sampling remains deferred.


Final validation: **18 renderer tests and 80 relevant Python tests passed**, with
one optional fixture skipped. Both native v3 pilot EXRs retain their original
checksums/integrator-4 provenance. BMW mode diagnostics have identical guides
but one 128-SPP reference pixel differs by 1.96e-5. A same-binary 4096-SPP repeat
has reference RMS difference 3.61e-6 (maximum 3.63e-4), so bit-identical BMW
rerendering is not established; these small differences do not explain the
remaining sun outliers. The official v5 calibration report was regenerated with
the final executable and still yields 0.223 / 0.073 noise fractions.

### Visible-speck regression and retained HDR sampling correction — integrator 8

The user's image review exposed a v5 regression hidden by the global squared-error
summary. At 1024 SPP, windshield pixels with linear luminance >1 and >10 times
their eight-neighbor median increase from 1/2 per replica in v4 to 8/8 in v5.
The score's 4-SPP denominator also changes between integrators. Neither that
ratio nor a few selected path improvements establish fewer visible fireflies.
The 1024/4096 levels use disjoint sample ranges, not nested image refinements.

Chrome-region GPU outliers replay as chrome → diffuse surface → paired glass →
HDR sun, or as diffuse surfaces seen between grille parts → glass → sun.
Radiance-only sun removal eliminates those selected spikes with the same RNG
and geometry. Diffuse continuation now mixes cosine and environment directions,
traces the real intervening glass, and weights by the full mixture density;
direct-light MIS uses the matching density. The HDR alias table now covers the
bilinear texture footprint rather than relying on a tiny uniform probability
in bright neighboring cells. Incorrect diffuse geometric fallback/resampling
is replaced by zero-contribution rejection with a matching PDF. Controlled
slab/sun and tilted-normal tests validate energy independently. No production
radiance clamp, transparent-shadow shortcut, material/asset edit or adaptive
pixel allocation is used.

The diffuse-only candidate (integrator 6) sharply reduces windshield/window and
grille variance. An extension to finite GGX (integrator 7) improved selected
chrome/headlight paths but created larger outliers around another headlight;
whole-image noise worsened. That experiment is **rejected and preserved**, not
promoted based on its selected successes or its 4096-SPP scalar passing flag.
Integrator **8** restores the v6 traversal/continuation estimator exactly,
with finite GGX retaining its original matched BSDF sampling. Its new version
prevents resuming the intervening experiment. Further glass/chrome connections
and subpixel direct-sun reflections remain reference-quality limitations.

All builds use only `/tmp/ntasad-capture-build` with `-j1`. Two native v3 pilots
still match their original EXR checksums and retain integrator-4 provenance;
they cannot be mixed/resumed with the current build. V2 stays separate, the
default remains `plan-v3.json` with its existing 1024-SPP references, and bulk
generation/training remain stopped. Adaptive pixel sampling remains deferred.
See the renderer diagnosis and
`../../dataset/diagnostics/bmw-hdr-fireflies/hdr-spot-investigation/` for counts,
fixed-exposure comparisons, source/binary hashes and accepted/rejected evidence.

### BMW appearance/material audit — integrator 9 candidate

The user's brown windshield/headlight-detail review leaves v6/v8 appearance
unqualified. Longer CPU replays at two changed windshield pixels show that
cosine-only sampling also resolves the warm glass → leather → glass → HDR-sun
contribution, with much larger uncertainty; that finding is not validation of
the entire glass model. The supplied Cycles reference is denoised and uses a
different filter/display transform; its clamp/glossy-filter settings are not
fully recorded. The original PBRT source enables regularization.

The loader also had a confirmed feature loss: BMW requires clearcoat on 12
materials, but it and specular controls were silently dropped. All 114 glTF
meshes / 792,746 triangles, including the headlight geometry, are accounted for.
Integrator 9 adds opaque metallic/roughness specular and clearcoat layers with
matched finite/delta sampling and direct MIS. Transmission remains a partial
smooth-solid model: BMW omits the volume extension, which means thin glass in
glTF. This semantic difference and the remaining HDR-sun tails are unresolved
reference concerns, not a reason to restart generation.

The matched-camera 700×500 comparison, material/geometry audit and replay logs
are under `../../dataset/diagnostics/bmw-hdr-fireflies/appearance-investigation/`.
All compilation remains serial in the original `/tmp/ntasad-capture-build`.
Native v3 pilots retain integrator-4 provenance; v2 stays separate and plan-v3
camera/sample/split/default budgets are unchanged. Bulk generation and training
remain stopped; adaptive pixel sampling remains deferred.

### Headlight reflective frame — integrator 10 candidate

The coating-only v9 comparison restored paint/floor reflections but retained a
dark headlight center and failed the 1024-SPP noise rule. All 18 renderer and
80 relevant Python tests passed, with one optional fixture skipped; its source
and images remain archived. Scalar 4096-SPP success did not qualify production.

CPU replay then isolated an inner-chrome smooth normal whose ideal reflection
points below the actual triangle: smooth view cosine 0.06875 versus geometric
0.62039. Rejecting its narrow lobe makes the headlight cylinder appear black.
A bounded CPU ablation using a valid reflective frame restores the cylinder;
the dark-center red mean rises 0.03279 → 0.08357 while the neighboring cylinder
changes only 0.13447 → 0.13746. Integrator 10 corrects base/coat reflective
frames consistently in BRDF evaluation, continuation and MIS, preserving the
diffuse normal. It is a documented smooth-normal policy, not radiance clamping,
roughening, mesh editing or the rejected generic GGX guiding experiment.

The warm windshield caustics and incomplete thin/solid glTF distinction remain
separate reference concerns. Generation/training remain stopped, all builds
remain serial in `/tmp/ntasad-capture-build`, and production plans/pilots remain
separate and unchanged. See the renderer diagnosis and appearance comparisons.

Final integrator-10 validation passes **18 renderer tests and 80 relevant Python
tests**, with one optional fixture skipped. GPU headlight-center red rises
0.03569 → 0.06877 at 1024 SPP; all guides/counts and both original native pilot
EXR hashes are unchanged. The supplied reference still differs. The final
1024/4096 noise fractions are 0.16413 / 0.05753: 1024 fails, and the absolute
4096 noise estimate worsens versus v9 (0.04925 versus 0.03739). This is an
appearance/model correction, **not** a general firefly-free result. A remaining
grille outlier replays as three chrome bounces → paint → paired glass → HDR sun,
with an individual red contribution of 41,672. The refracted connection proposal
and native reference quality remain unresolved. Bulk generation/training remain
stopped; no default SPP/plan or adaptive sampling change follows from these checks.

### Fresh dataset with accepted indirect clamp — 2026-11-10

The user accepted the renderer's indirect radiance clamp of 10 and authorized
resetting the dataset. The validated integrator-11 executable in the original
`/tmp/ntasad-capture-build` matches the recorded clamp-validation SHA-256
`e4c694544e3491fb0a3bdb2923b461065817759f0ed3c1f891460a164bba6a52`;
the existing build is current, so no build or new build directory was needed.

`plan-v3.json` now explicitly records `indirect_radiance_clamp: 10.0`. The
generator carries it into all capture and reference-check configs, and resume
verification rejects frames with missing or different clamp metadata. V2 plans
and outputs remain separate and unchanged. Cameras, sample ranges, splits,
1920×1080 resolution, eight-frame clips and 4+1024-SPP budgets are preserved.

The previous active v3 directory was moved, without copying or deleting EXRs,
to `dataset/archive/denoiser-v3-before-clamp-2026-11-10`. Both original pilot
EXR hashes were verified after the move. A fresh `dataset/denoiser-v3` contains
the complete 126-clip / 1008-frame plan, split and training configs, bound to
the validated renderer and referenced asset hashes, with zero captured frames.
All diagnostics and older datasets remain intact. Bulk generation has not
been launched by the assistant; the user receives the background launch command.
Training and adaptive sampling remain deferred. Reference status remains
`fixed_budget_not_convergence_qualified`.

Dataset planning/resume, reference-convergence, frame, sequence and split checks:
**77 passed, 1 optional fixture skipped**. Tests cover explicit clamping in
production/report configs, invalid clamp settings and refusing mismatched
completed-frame metadata. Renderer code was unchanged in this reset; the
existing integrator-11 validation records all 18 renderer tests passing.


## 2026-11-10: Faster camera profiles and expanded viewpoints (v4)

The paused v3 generation produced two complete BMW frames. Both were verified
against their original v3 camera/settings and EXR checksums and remain untouched.
The renderer capture tool already accepts arbitrary per-frame camera poses and
preserves the preceding camera for motion guides, so no renderer changes were
needed for this adjustment.

The default is now `dataset_generation/plan-v4.json`, outputting to
`../../dataset/denoiser-v4`. Slow remains 0.6×; medium is 1.8× and fast 3.6×,
approximately 29% above v3's medium/fast amplitudes over the same eight frames.
Anchors can now supply an explicit world-space `camera` (position, target, up,
FOV), while legacy offsets/yaw continue to use the original scene camera.
Training scenes retain their three nearby views and add three separated views;
held-out scenes add one view each, for three anchors. All six motions remain
continuous within each clip. Training grows from 90 clips / 720 frames to
180 clips / 1440 frames; validation has 36 clips / 288 frames and Sponza test has
18 clips / 144 frames. Total: 234 clips / 1872 frames.

V4 retains 1080p, 4+1024 SPP, integrator 11, clamp 10, depth 12, caustics off,
and the independent scene split. Sample allocation starts at 700,000,000,
after v3 and below the independent convergence-replica range at 1,000,000,000.
Report references retain identical cameras/input sample IDs. Production configs,
split and training configs are prepared and locked to the checked binary.
No v4 production captures or training were started.

All 1872 low-resolution diagnostic poses passed the final near-surface check
with zero flags. Earlier Bistro/Bathroom collisions were corrected before this
final sweep; the initial v4 diagnostic/config drafts remain under
`../../dataset/archive/denoiser-v4*-draft-2026-11-10`. Framing checks do not prove
that every point between discrete cameras is collision-free. Diagnostic sheets,
albedo/radiance motion GIFs and motion summaries are under
`../../dataset/denoiser-v4-cameras`; `viewpoint-overview.png` compares all anchor
positions. BMW linear clip median frame-p95 motion at 1080p measured approximately
17 / 50 / 96 pixels per frame (v3: 17 / 39 / 75).

Validation: Python suite 161 passed, 12 optional tests skipped; targeted dataset
generation tests 28 passed after the final camera adjustments. Historical v1–v3
plans remain unchanged and explicit `--plan dataset_generation/plan-v3.json`
continues to target the paused v3 output. Reference quality remains the declared
fixed-budget status; this change does not qualify convergence.


## 2026-11-10: One unified spatial/temporal denoiser

Rechecked the main paper, Figure 3 and sections 2.1–2.2. The two networks in
NTASAD are the adaptive sampler and the denoiser; spatial and temporal kernel
prediction belong to one denoiser. The existing recurrent prototype already
contained that shared network, while the dataset launcher still trained the
older independent residual spatial baseline as a second model.

The active implementation is now `model.NTASADDenoiser`, with one `training`
module, `checkpoint` module, and `wavefront-denoiser-checkpoint` format. The
residual spatial model/trainer/checkpoint code and obsolete model configs were
removed. Old spatial/recurrent checkpoint formats are deliberately unsupported;
the next production model starts fresh. Historical lighting-study model training
was retired with its baseline; its capture preparation and paired-result analysis
remain available. Historical reports/artifacts were not used as initialization.

The U-Net shares its encoder/decoder across three spatial 5×5 filters, one temporal
5×5 filter and learned blends. It retains Figure 3's reconstructed widths/depth,
15 guide/history/noisy features, bilinear reprojection, transformed-domain
recurrence, one noisy warm-up frame plus four learned steps, and final-frame
spatial L1 plus final-difference temporal L1, weighted equally. A still-image
fallback disables history and uses these same learned spatial kernels. It is
an inference mode, not another model or separate spatial pretraining stage.
The paper's under-specified operator choices and parameter-count discrepancy
remain documented above; unification does not establish exact author-code or
image-quality equivalence.

The public `train` command and `tools/train_dataset.py` now train one model.
Dataset preparation writes only `train.json`, defaulting to Xavier initialization,
Adam 0.001, 1000 epochs, 128-pixel shared training crops and fixed 256-pixel
validation crops. The launcher defaults to `runs/denoiser-v4-denoiser/`, supports
explicit `--resume`, and verifies train/validation before touching training data.
A capture launched before this change can finish with its existing sidecars;
the first training launch creates a missing `train.json` and ignores obsolete
sidecars. No active capture output, camera plan, renderer binary, frame, or lock
was modified during model unification. No production training was started.

`denoise`, `denoise-sequence`, and `evaluate` all load the same checkpoint.
Streaming inference resets history at every clip, labels frame zero as noisy
initialization, and accepts input-only captures without reference channels.
Sequence evaluation excludes noisy initialization from image metrics and reports
transformed-domain temporal finite-difference error on subsequent learned pairs.
Reserved-test evaluation still requires the frozen split, and changed observed
inventories are rejected before test pixels are opened. Output names are checked
and failed inference removes its incomplete staging directory.

Validation covers joint gradients into spatial/temporal kernels and blends,
shared still-image inference, one-checkpoint train/infer/evaluate integration,
input-only sequence inference, boundary resets, output cleanup, training-launcher
migration, test-pixel isolation, and exact epoch-boundary optimizer/RNG resume.
The adaptive sampler, zero-SPP support and joint sampling optimization remain
future work; this change makes the implemented denoiser one model.

Final unification checks: **151 tests passed, 12 optional GPU/cross-language
tests skipped** on CPU. CLI help exposes one trainer and both inference modes;
the source contains one model class and one trainer. The default network retains
1,114,311 parameters.
