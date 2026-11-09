# NTASAD camera dataset: pilot and scale-up

This extends the recurrent implementation in [ntasad-progress.md](ntasad-progress.md).
Uniform 4-SPP inputs, eight-frame camera clips, five-frame training windows,
shared 128-pixel crops/augmentations, and static geometry are the initial contract.
Adaptive sampling remains deferred. A small pilot does not establish production
readiness: inspect its images and timing before selecting the production set.

The first pilot has now run; see [its results](ntasad-pilot-results.md). Its
reference sweep failed the dataset-wide qualification rule, so bulk capture
remains deferred.

## Local asset inspection

Inspection of glTF JSON/GLB JSON chunks on 2026-11-09 found:

| Asset | Meshes / materials | Declared extensions | Existing staging |
| --- | --- | --- | --- |
| DragonAttenuation.gltf | 2 / 3 | transmission, volume, variants | cornell.json |
| GlassHurricaneCandleHolder.glb | 2 / 2 | transmission, volume | hurricane.json |
| GlassBrokenWindow.glb | 3 / 3 | transmission | none |
| sponza/Sponza.gltf | 1 / 25 | none | sponza.json |

All external buffer/image URIs resolve locally; none of these assets declares
animations. These are inventory counts, not rendered scene complexity. Accessor
bounds are mesh-local and do not establish world-space camera framing: node
transforms and scene placement must be respected. BrokenWindow still needs a
scene manifest and a framing/material preview. Preserve bundled license files;
review licensing before distributing any derived dataset.

The loader reads transmission, IOR, attenuation and thickness, but maps any
positive transmission factor to its dielectric material category. It does not
implement glTF material-variant selection. Appearance equivalence to a glTF
reference viewer is therefore not established. Preview glass silhouettes,
textures, alpha masking, normals and exposure before accepting a scene.

The local set also includes demo geometry and OBJ deer, bunny and crystals, plus
a procedural prism. `rtweek` points to an external Lucy file and is not a ready
local scene. Multiple objects in the same Cornell room do not count as independent
scene families. The four glTF files alone cannot provide nine diverse training
scenes plus independent validation and test scenes.

## Small pilot

`configs/ntasad_pilot_split_v1.json` preregisters two training clips (hurricane,
demo), one validation clip (deer), and one reserved test clip (Sponza).
All have eight explicitly listed cameras, 192×128 pixels, max depth 12, caustics
disabled and disjoint sample-index blocks. The initial reference field is a
1,024-SPP target; only convergence evidence can qualify it. Do not use the direct
clip captures as qualified training data before that check.

Run direct train/validation captures into `runs/ntasad-pilot-v1/preview`, outside
the final dataset root. Keep Sponza pixels unopened during pilot development.
First inspect hurricane; then demo and deer. Check all eight views for actual
object coverage (the earlier deer dataset missed its object), camera continuity,
finite HDR channels, valid depths/normals, and motion sign/magnitude. Frame zero
must reset history; later frames must identify the preceding camera and contain
useful valid reprojection. Check occlusion edges and backgrounds separately;
first-hit motion does not track reflected/refracted appearance.

Load each completed clip with `SequenceDataset(length=5)` and verify four windows
per clip, and aligned shared crops through `RecurrentPatchDataset`. Record frame
time, peak GPU memory, EXR bytes/frame and pilot wall time. Do not train or tune
against reserved test pixels.

`ntasad_pilot_convergence_v1.json` registers independent replicas at 256, 1,024
and 4,096 SPP for hurricane endpoints, demo middle and deer middle. The existing
rule requires reference noise ≤10% of 4-SPP input error both over the entire image
and the brightest 5%; a selected level and all higher tested levels must pass
every case. Keep the rule fixed. If 1,024 fails, increase the measured budget or
revise the scene set with a new plan, rather than relabeling 1,024 as converged.
Earlier spatial captures needed 8,192 SPP, so failure is plausible.

From the denoiser directory, with an absolute capture executable and renderer root:

```bash
python -m wavefront_denoiser reference-convergence \
  configs/ntasad_pilot_convergence_v1.json runs/ntasad-pilot-convergence-v1 \
  --capture-executable /tmp/ntasad-capture-build/dataset_capture \
  --renderer-root ../Wavefront-Path-Tracer
```

The existing `capture-split` command captures all partitions, including test.
Do not invoke it during preview development. After qualification and freezing the
pilot protocol, it can create the complete audited pilot dataset:

```bash
python -m wavefront_denoiser capture-split configs/ntasad_pilot_split_v1.json \
  runs/ntasad-pilot-convergence-v1/report.json \
  --capture-executable /tmp/ntasad-capture-build/dataset_capture \
  --renderer-root ../Wavefront-Path-Tracer
python -m wavefront_denoiser split-audit configs/ntasad_pilot_split_v1.json
```

`ntasad_pilot_train_v1.json` is a two-epoch integration configuration, not a
training-quality experiment. It retains five-frame windows and shared crops.

## Production proposal after pilot acceptance

Target nine genuinely different training scene families × 19 clips × eight
frames = 1,368 training frames, matching the reported scale rather than claiming
the original scenes. Reserve at least two additional families each for validation
and test, initially eight clips per family (128 frames per partition). Acquire
additional licensed assets before finalizing names. Prioritize furnished interiors,
textured architecture, outdoor vegetation, small clutter, glossy metals, diffuse
indirect lighting and glass. Avoid filling the nine slots with Cornell variants.

All Cornell-backed dragon/hurricane/window/prism/crystal variants belong to one
training family; demo stays train, deer stays validation, Sponza stays test.
Those assignments preserve existing baseline provenance. Shared primary assets,
room layouts and lighting variants remain in their parent partition. Freeze a
manifest before rendering; record asset/buffer/texture hashes and licenses in
addition to the renderer fingerprint, which is not a complete dependency hash.

Start production at 512×384, subject to pilot framing, time and memory. Use smooth
short dollies, arcs, lateral reveals and modest yaw/pitch, with both low and higher
motion, fine texture and disocclusion. Eight-frame clips are independent units;
never concatenate unrelated cameras. Change light/material settings only between
clips. Static geometry/camera-only motion is an explicit departure from animated
NTASAD scenes. Each clip yields four overlapping five-frame windows; split by
scene before creating windows, crops or augmentations.

Requalify references at production resolution on representative train/validation
views, including difficult highlights and motion endpoints. Target 1,024 SPP for
training; preregister a separate 4,096-SPP evaluation target and qualify it using
train/validation calibration only. Current capture-split uses one global budget;
implement and test separate reference policies before claiming mixed budgets.
Never choose budgets or checkpoints using test-scene convergence results.

For the 1,368-frame training proposal alone, 512×384 at 4+1,024 SPP costs
276.5 billion camera samples before path depth, replicas or evaluation. Measure
time and storage rather than extrapolating samples as equal cost across scenes.
Use the pilot's slowest representative scene plus a safety margin for scheduling.
Capture one production clip per scene before releasing the full batch. Keep exact
configs, logs, renderer revision, dependency hashes, sample intervals, scene splits
and qualification reports with the dataset. Resume only completed clip units;
current capture commands do not provide a general interrupted-batch resume.
