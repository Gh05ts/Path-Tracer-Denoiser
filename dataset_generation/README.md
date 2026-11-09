# Denoiser dataset generation

This directory holds the camera plans and scene manifests for the new static-scene
dataset. Executable scripts live in `../tools/`. Run the commands below from `Path-Tracer-Denoiser` with
the package installed (`pip install -e '.[train,test]'`). The environment used for
local checks is `/tmp/ntasad-venv/bin/python`; replace `python3` with your activated
environment's Python when needed.

`plan-v2.json` is the editable source of truth. It uses 1920×1080 captures, eight
frames per clip, independent 4-SPP noisy inputs and 1024-SPP references, depth 12,
Russian roulette from depth 5, 256×256 render tiles, and caustics disabled.
Adaptive sampling remains deferred.

| Partition | Scenes | Clips | Frames |
| --- | --- | ---: | ---: |
| Train | BMW, Bistro, Living Room, Breakfast Room, Dragon | 90 | 720 |
| Validation | Bathroom, Classroom2 | 24 | 192 |
| Final evaluation | Sponza | 12 | 96 |

This assigns Bathroom and Classroom to validation and reserves Sponza for final
evaluation. A scene and all its camera clips belong to one partition. Neither
model trains on validation or test scenes. Training and checkpoint selection do
not load test pixels. Camera previews may display the test scene for framing;
do not use its denoising results to tune hyperparameters.

Each training scene has three anchors × six motions; held-out scenes have two
anchors × six motions. Motions are lateral linear translation, forward/backward
dolly, wave translation, a small circle in the camera's right/up plane, an orbit
arc about its focus, and pan/tilt. The orbit is a small arc, not a full revolution
inside a room. Each clip is a continuous motion, without camera cuts. Local
motion scales are deliberately smaller in confined interiors. Source glTF camera
transforms seed the anchors; Dragon uses the existing Cornell staging and Sponza
uses an explicit atrium camera. Research interiors/BMW use their supplied HDR
lighting, Bistro uses its imported lights, and Sponza retains its top light.

V2 cycles **slow (0.6×), medium (1.4×), and fast (2.8×)** motion across anchors
and motion types. Every training motion appears at all three speeds across the
three viewpoints; this keeps the original 18 clips per training scene. Multipliers
scale the scene's translation/arc/pan amplitude over the same eight frames, not
the playback speed of an identical path. Interiors retain smaller base distances.
These are our engineering choices, not camera speeds specified by the paper.
The original plan (`plan.json`) and its completed two-frame BMW pilot are preserved
in `../dataset/denoiser-v1`. V2 uses separate IDs/sample ranges/output, and never
mixes its new trajectories into that pilot.

## Inspect and pilot

The renderer must be built with dataset capture/OpenEXR support. In the current
workspace the executable is `/tmp/ntasad-capture-build/dataset_capture`. Keep that
binary unchanged during a dataset run; production use should put the build in a
persistent directory rather than depending on `/tmp` surviving a reboot.

```bash
# Write the complete configs and split without rendering. The subsequent capture
# can bind this planned output to its renderer binary.
python3 tools/generate_dataset.py --stage plan

# Check all planned cameras of all 126 clips, pooled by scene to avoid
# repeatedly loading Bistro. Outputs are diagnostic, never training captures.
python3 tools/check_dataset_cameras.py \
  --renderer /tmp/ntasad-capture-build/dataset_capture

# Eight-frame 320×180 appearance/guide previews, one clip per scene.
python3 tools/generate_dataset.py --stage preview \
  --renderer /tmp/ntasad-capture-build/dataset_capture

# Two real 1080p frames, 4+1024 SPP. Reused by the later complete capture.
python3 tools/generate_dataset.py --stage pilot --scenes bmw \
  --frame-range 0 2 --output ../dataset/denoiser-v2 \
  --renderer /tmp/ntasad-capture-build/dataset_capture

# Extend that exact pilot to a full eight-frame clip; five-frame windows can now
# be checked. Omitting --scenes selects BMW and Breakfast Room for the pilot.
python3 tools/generate_dataset.py --stage pilot --scenes bmw \
  --output ../dataset/denoiser-v2 \
  --renderer /tmp/ntasad-capture-build/dataset_capture
```

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
  --renderer /tmp/ntasad-capture-build/dataset_capture
```

That command captures the complete plan in `../dataset/denoiser-v2`. It works in one
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

The raw 22-channel float32/uint32 payload is about 174 MiB/frame, or 171 GiB for
all 1008 frames before lossless EXR compression. Actual disk use and render time
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
  ../dataset/denoiser-v2/reference-check-plan.json ../dataset/denoiser-v2-reference-check \
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

Generated `split.json`, `train-spatial.json` and `train-recurrent.json` can be
consumed directly by the denoiser. The recurrent configuration uses five-frame
windows, shared 128×128 random crops/flips/90-degree rotations across all frames,
references and guides. Motion-vector components rotate/flip with image axes;
world normal components retain their coordinates. Crop borders invalidate
unavailable history. Full frames are decoded one at a time and cropped before
stacking tensors, bounding host memory. There are four overlapping five-frame
windows in each eight-frame clip and four random patches per window per epoch.

Recurrent checkpoint selection uses five deterministic 256×256 crops (center
and corners) per validation window, without augmentation. This is a crop-based
validation loss, not a full-1080p evaluation metric. Full-frame validation remains
the default for older configurations. The spatial model uses 128×128 training
patches and full-image validation. Initial epochs are 40 spatial / 1000 recurrent;
edit the generated training JSONs to change them. Capture resume preserves those
training JSON edits. Full-resolution recurrent report inference is a separate
runtime/memory task; this generator does not add tiled recurrent inference.

```bash
# Fails if train/validation clips are incomplete. Fresh random weights by default.
# Writes separate spatial/ and recurrent/ runs; never loads an old model silently.
python3 tools/train_dataset.py --device cuda

# The same training command detached from SSH:
DATASET_PYTHON=/tmp/ntasad-venv/bin/python \
  ./tools/run_dataset_background.sh train --device cuda

# Either model alone:
python3 tools/train_dataset.py --models recurrent --device cuda \
  --output runs/denoiser-v2-recurrent

# Only an explicitly requested recurrent resume loads weights/Adam/RNG state.
# Increase epochs in the same training config first; use a fresh output directory.
python3 tools/train_dataset.py --models recurrent --device cuda \
  --recurrent-resume runs/denoiser-v2-models/recurrent/last.pt \
  --output runs/denoiser-v2-resumed
```

Training can begin once all train/validation groups are complete; test groups may
remain reserved. Default capture generates all three partitions. This setup
creates capture/training scripts; it does not start bulk generation or production
training automatically.

## Separate report references

```bash
# Validation + evaluation scenes by default; separate ../dataset/denoiser-v2-reports4096.
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

Local checks: the relocated v1 pilot resumes by skipping both completed frames;
all 1008 v2 camera poses passed the final near-surface check after one Bathroom
camera adjustment. BMW linear slow/medium/fast clips estimate approximately
17/39/75 pixels per frame (median frame-p95, scaled to 1080p). The Python suite
passed 151 tests, with 12 optional tests skipped; camera captures were checked
separately on the RTX 3090. The moved background launcher passed SIGHUP testing.
