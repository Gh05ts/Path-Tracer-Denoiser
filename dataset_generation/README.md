# Denoiser dataset generation

This directory holds the camera plans and scene manifests for the new static-scene
dataset. Executable scripts live in `../tools/`. Run the commands below from `Path-Tracer-Denoiser` with
the package installed (`pip install -e '.[train,test]'`). The environment used for
local checks is `/tmp/ntasad-venv/bin/python`; replace `python3` with your activated
environment's Python when needed.

`plan-v4.json` is the editable source of truth. It uses 1920×1080 captures, eight
frames per clip, independent 4-SPP noisy inputs and 1024-SPP references, depth 12,
Russian roulette from depth 5, 256×256 render tiles, caustics disabled, and an
explicit `indirect_radiance_clamp` of **10** for both noisy inputs and references.
Adaptive sampling remains deferred.

V4 retains v3's corrected integrator **11**, glass/GGX and glTF material fixes,
and clamp 10, while expanding viewpoints and increasing medium/fast motion.
The paused v3 output remains untouched in `../dataset/denoiser-v3` with two
completed BMW frames. Its original plan remains `plan-v3.json`; resume it only
with that explicit plan. V4 uses a separate dataset ID, output directory and
sample range, so its changed trajectories cannot mix with the paused captures.
See the renderer
[diagnosis](../../Wavefront-Path-Tracer/docs/diagnostics/bmw-hdr-fireflies/README.md).

**Current status (2026-11-10):** all 234 v4 configs and the split are prepared
and locked to the checked integrator-11 renderer. Bulk generation and training
remain stopped. References retain their declared
fixed-budget status; clamping does not confer convergence qualification.

| Partition | Scenes | Clips | Frames |
| --- | --- | ---: | ---: |
| Train | BMW, Bistro, Living Room, Breakfast Room, Dragon | 180 | 1440 |
| Validation | Bathroom, Classroom2 | 36 | 288 |
| Final evaluation | Sponza | 18 | 144 |

This assigns Bathroom and Classroom to validation and reserves Sponza for final
evaluation. A scene and all its camera clips belong to one partition. Neither
model trains on validation or test scenes. Training and checkpoint selection do
not load test pixels. Camera previews may display the test scene for framing;
do not use its denoising results to tune hyperparameters.

Each training scene has **six anchors × six motions**; held-out scenes have
**three anchors × six motions**. V3's original nearby anchors are retained;
new anchors have explicit world-space `camera` poses in other parts of the scene.
BMW adds views around the car, Bistro moves along the street, interiors add
translated/reoriented views, Dragon adds side/elevated views, and Sponza adds a
view from the opposite end. An anchor's optional `camera` supplies its own
position, target, up and FOV; `offset` and `yaw_degrees` remain supported relative
to that camera. Without an explicit camera, legacy plans use the scene camera.

Motions are lateral linear translation, forward/backward dolly, wave translation,
a small circle in the camera's right/up plane, an orbit arc about its focus,
and pan/tilt. Each eight-frame clip is continuous, without camera cuts. Local
motion scales stay smaller in confined interiors. Research interiors/BMW use
their supplied HDR lighting, Bistro uses its imported lights, and Sponza retains
its top light.

V4 cycles **slow (0.6×), medium (1.8×), and fast (3.6×)** across anchors and
motion types. Medium and fast are approximately **29% faster** than v3's 1.4×
and 2.8×; slow is unchanged. Each training motion appears at every speed from
two viewpoints. Multipliers scale translation/arc/pan amplitude over the same
eight frames, not playback speed. These are engineering choices, not camera
speeds specified by the paper. V1–v3 plans and outputs remain available separately.

## Inspect and pilot

The renderer must be built with dataset capture/OpenEXR support. In the current
workspace the executable is `/tmp/ntasad-capture-build/dataset_capture`. Keep that
binary unchanged during a dataset run; production use should put the build in a
persistent directory rather than depending on `/tmp` surviving a reboot.

```bash
# Write the complete configs and split without rendering. The subsequent capture
# can bind this planned output to its renderer binary.
python3 tools/generate_dataset.py --stage plan

# Check all planned cameras of all 234 clips, pooled by scene to avoid
# repeatedly loading Bistro. Outputs are diagnostic, never training captures.
python3 tools/check_dataset_cameras.py \
  --renderer /tmp/ntasad-capture-build/dataset_capture

# Eight-frame 320×180 appearance/guide previews, one clip per scene.
python3 tools/generate_dataset.py --stage preview \
  --renderer /tmp/ntasad-capture-build/dataset_capture

# Two real 1080p frames, 4+1024 SPP. Reused by the later complete capture.
python3 tools/generate_dataset.py --stage pilot --scenes bmw \
  --frame-range 0 2 --output ../dataset/denoiser-v4 \
  --renderer /tmp/ntasad-capture-build/dataset_capture

# Extend that exact pilot to a full eight-frame clip; five-frame windows can now
# be checked. Omitting --scenes selects BMW and Breakfast Room for the pilot.
python3 tools/generate_dataset.py --stage pilot --scenes bmw \
  --output ../dataset/denoiser-v4 \
  --renderer /tmp/ntasad-capture-build/dataset_capture
```

The current `../dataset/denoiser-v4-cameras/viewpoint-overview.png` compares
original and added viewpoints across all scenes (4-SPP diagnostics).
`tools/check_dataset_cameras.py` saves one PNG sheet per scene and `camera-checks.json` with
near-surface warnings. These are framing checks, not a proof that an entire
continuous path is collision-free. Diagnostic sheets can adjust exposure for
viewing; EXR radiance stays linear and unmodified. The ordinary preview/pilot
stages save `contact-sheet.png` with first/middle/last reference frames,
`comparison.png` with explicitly labelled noisy/reference pairs, and full-size
viewing PNGs under `previews/`. These PNGs are display-only; EXRs remain linear.
The camera checker renders 1-SPP input and **4-SPP diagnostic references**, not
1024-SPP training references. Its noisy sheets are labelled accordingly. Appearance
previews use 4+16 SPP; actual pilot/capture uses 4+1024 SPP. No image here is a
trained denoiser output. The camera checker additionally saves paired radiance /
noise-free albedo-guide GIFs in `motion-previews/`, plus per-clip motion estimates
scaled to the dataset resolution and history-valid fractions in `camera-checks.json`.
Inspect previews
before committing GPU time to bulk capture. A new `--output` is required if the
plan/assets/renderer change after capture begins.

## Run through an SSH disconnect

A foreground process can die when SSH disconnects. This launcher uses `nohup`,
redirects stdin/stdout/stderr, and prints its PID and log path:

```bash
DATASET_PYTHON=/tmp/ntasad-venv/bin/python \
  ./tools/run_dataset_background.sh \
  --plan dataset_generation/plan-v4.json \
  --renderer /tmp/ntasad-capture-build/dataset_capture
```

That command captures the complete plan in `../dataset/denoiser-v4`. It works in one
go and skips verified completed frames when rerun. It survives SSH hangups while
the host/process remains running; after a reboot or crash, start it again.
The background worker was explicitly tested by sending SIGHUP during a plan job.

```bash
# Use the exact log path printed by the launcher:
tail -f ../dataset/jobs/JOB.log
# To stop a foreground monitor, Ctrl-C stops tail, not the capture.
ps -p PID -o pid,etime,args
```

All image, guide, preview and capture-log defaults are under `../dataset`
(the workspace's `new/dataset`), never `/tmp`. The renderer binary and temporary Python
environment shown above are separate from the output directory. Model checkpoints
continue to use the denoiser's `runs/` directory.

To run a subset, append `--scenes bmw bistro`,
`--partitions train validation`, or explicit IDs such as
`--groups bmw_view_00_circle_slow`. `--frame-range START END` captures the indicated
half-open range in each selected clip without changing its original frame
indices, preceding camera, or sample allocation. Omit it later to finish the
remaining frames.

A per-output advisory lock rejects concurrent capture/training writers.
`capture_lock.json` records the plan, generated cameras/settings, renderer SHA-256
and every referenced glTF/GLB, buffer, image and HDR digest. Missing assets and
unresolved Git LFS pointers fail before capture. Resume validates frame metadata,
cameras and EXR SHA-256 before skipping a frame. Newly captured frames also pass
the complete channel/finite-value/guide validation. Corrupt completed captures
cause an error, not a silent skip or overwrite. Incomplete `.partial` exports are
preserved under `recovery.partial/` and the unfinished frame is rendered again.
Sampling within a partly rendered frame is not checkpointed. Completed frames
remain usable even if the process dies before the clip finishes.

The raw 22-channel float32/uint32 payload is about 174 MiB/frame, or 318 GiB for
all 1872 frames before lossless EXR compression. Actual disk use and render time
must be measured from the pilot. The report dataset requires separate space.
Local capture uses the renderer task API; this launcher does not distribute
capture or resume the renderer's per-sample local/distributed checkpoint files.

## Reference quality

1024 SPP is the requested fixed training reference budget. It is **not yet
convergence-qualified on these new scenes**. Files explicitly declare
`fixed_budget_not_convergence_qualified`. The scene split/audit still enforces
the declared minimum reference count and complete frame indices, while the
training configs explicitly allow this fixed-budget policy. This does not
fabricate or bypass a convergence certificate for the earlier qualified-only
capture workflow.

Each normal output also receives `reference-check-plan.json`: independent
256/1024/4096-SPP replicas of the first and last cameras from one clip per
train/validation scene, at that output's resolution. Test scenes are excluded.
This is representative-view evidence, not proof of convergence for every frame.
Run it separately before relying on the references for reported accuracy:

```bash
python3 -m wavefront_denoiser reference-convergence \
  ../dataset/denoiser-v4/reference-check-plan.json ../dataset/denoiser-v4-reference-check \
  --capture-executable /tmp/ntasad-capture-build/dataset_capture \
  --renderer-root ../Wavefront-Path-Tracer
```

The existing convergence command expects a new output directory and does not
resume interrupted sweeps. It reports whole-image and brightest-5% reference
noise relative to the noisy input. A failed 1024-SPP check remains a failed check;
the dataset generator does not quietly escalate SPP or mark a policy qualified.

## Guides and training from scratch

Each `GROUP/frame_NNNNNN/` contains `frame.exr` and `manifest.json`. Exported EXR
channels include linear noisy/reference RGB, albedo RGB, signed world shading and
geometric normals, forward depth, backward pixel motion, surface/history masks,
and input/reference sample counts. The manifest includes current/previous
cameras, frame indices, scene/render fingerprints, environment and integrator
settings, sample allocation and image SHA-256. No PNG or tone-mapped image is
used as a training target.

Generated `split.json` and **`train.json`** feed one spatial/temporal NTASAD
U-Net. Spatial kernels at three scales, temporal history filtering, and blend
weights train jointly; there is no separate spatial model or pretraining stage.
The adaptive sampler remains a future second network.

Training uses five-frame windows, one noisy initialization plus four recurrent
steps, and shared 128×128 crops/flips/90-degree rotations across all frames,
references and guides. Motion-vector components rotate/flip with image axes;
world normal components retain their coordinates. Crop borders invalidate
unavailable history. Full frames are decoded one at a time and cropped before
stacking tensors. Each eight-frame clip supplies four five-frame windows, with
four random patches per window per epoch.

Checkpoint selection uses the same final-frame spatial plus final-difference
temporal L1 loss on five fixed 256×256 validation crops per window. These are
crop-based validation losses, not full-1080p evaluation metrics. Defaults are
Xavier initialization, Adam at 0.001, 48 base channels and 1000 epochs. Edit the
single `train.json` to change settings; capture resume preserves edits.

```bash
# One model, fresh random weights; verifies train/validation without test pixels.
python3 tools/train_dataset.py --device cuda

DATASET_PYTHON=/tmp/ntasad-venv/bin/python \
  ./tools/run_dataset_background.sh train --device cuda

# Explicit resume into a new output; increase epochs in train.json first.
python3 tools/train_dataset.py --device cuda \
  --resume runs/denoiser-v4-denoiser/last.pt \
  --output runs/denoiser-v4-resumed
```

The default run is `runs/denoiser-v4-denoiser/`: `model.pt` is selected on
validation and `last.pt` resumes optimizer/RNG state at a completed epoch.
Training can begin once train/validation groups are complete; test may remain
reserved. This change does not start training automatically.

A capture started before unification can finish unchanged. On the first training
launch, a missing `train.json` is created after inventory verification; obsolete
`train-spatial.json`/`train-recurrent.json` sidecars are ignored. Capture locks,
configs, frame indices, sample allocation and EXR bytes remain intact. The
training launcher retains its exclusive writer lock, so it cannot start against
a dataset still being captured.

Still and streaming sequence inference both use this same checkpoint; see the
[README](../README.md). Sequence initialization follows the paper; the still
fallback runs only the shared learned spatial branch. The reference runtime
supports full-frame sequence inference but does not add tiled inference.

## Separate report references

```bash
# Validation + evaluation scenes by default; separate ../dataset/denoiser-v4-reports4096.
DATASET_PYTHON=/tmp/ntasad-venv/bin/python \
  ./tools/run_dataset_background.sh reports \
  --renderer /tmp/ntasad-capture-build/dataset_capture

# Limit expensive report capture to selected groups if desired:
python3 tools/generate_report_references.py --groups sponza_view_00_linear_slow \
  --renderer /tmp/ntasad-capture-build/dataset_capture
```

Report captures have 4096-SPP references and identical 4-SPP inputs/cameras to
their corresponding training/evaluation captures. An explicit 8192-sample stride
keeps each frame's sample IDs unchanged when the reference count increases;
1024-SPP references are a prefix of the report reference sample range. They are
not independent replicas for convergence measurement. Report references never
overwrite training targets. The report script generates captures, not formatted
figures or denoiser metrics.

## Relation to the paper's motion

The [paper](https://research.nvidia.com/sites/default/files/pubs/2020-05_Neural-Temporal-Adaptive/NTASAD_preprint.pdf),
section 2.2, uses nine animated scenes and 16–25 eight-frame clips per scene,
chosen to expose substantially different views. Figure 5 explicitly shows moving
characters with a static camera; the results discuss large, complex camera and
object motion. It does not prescribe a camera velocity or pixels-per-frame range
that we can reproduce. Our moving-camera/static-geometry dataset therefore does
not cover its object-animation cases, even after adding speeds.

At 1920×1080, the original BMW pilot's 95th-percentile guide motion was about
28 pixels/frame. A 160-pixel-wide sheet reduces that to about 2.3 pixels/frame,
so thumbnails can look nearly static. Use the animated albedo guides and motion
measurements to judge temporal displacement; inspect separate viewpoints to
judge scene coverage. Increasing motion alone does not replace viewpoint diversity.

Historical v1/v2 checks: the relocated v1 pilot resumes by skipping both completed frames;
all 1008 v2 camera poses passed the final near-surface check after one Bathroom
camera adjustment. BMW linear slow/medium/fast clips estimate approximately
17/39/75 pixels per frame (median frame-p95, scaled to 1080p). The Python suite
passed 151 tests, with 12 optional tests skipped; camera captures were checked
separately on the RTX 3090. The moved background launcher passed SIGHUP testing.


V4 checks (2026-11-10): all 1872 diagnostic camera poses passed the near-surface
check with zero flags after Bistro/Bathroom adjustments. The viewpoint overview
and per-scene sheets are under `../dataset/denoiser-v4-cameras/`. BMW linear
slow/medium/fast measured about 17/50/96 pixels per frame (median frame-p95 at
1080p), confirming that slow stayed unchanged and medium/fast increased. Python
suite: 161 passed, 12 optional tests skipped; final camera-plan tests: 28 passed.
The paused v3's two completed frames passed their original camera/checksum checks.
