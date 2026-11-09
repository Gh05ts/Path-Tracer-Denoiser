# Camera pilot results — 2026-11-09

**Decision: do not generate the bulk dataset yet.** The three preview clips are
usable for pipeline inspection, but the registered reference sweep returned
`insufficient_budget`, with no selected reference SPP. No qualified dataset or
new training run was created. Sponza test pixels were not captured or loaded.

See [the dataset plan](ntasad-dataset-plan.md) and
`configs/ntasad_pilot_convergence_v1.json` for the fixed pilot protocol.

## Captures and sequence checks

RTX 3090 captures completed for hurricane, demo and deer: eight frames each at
192×128, 4 input + 1,024 reference SPP, max depth 12, caustics disabled. The
1,024-SPP preview references remain unqualified as a dataset-wide policy.

All 24 frames loaded through the frame contract and passed finite-buffer checks.
Each clip produced four five-frame windows; shared augmented 128×128 crops loaded
successfully, including history reset at the beginning of each window. Camera
history was consecutive. Motion was independently checked by reconstructing
world points from forward depth and projecting them into the preceding camera.
The maximum component discrepancy was below 0.002 pixels. First-frame history
masks were empty and valid surface normals had median length approximately one.

| Clip | EXR storage | History-valid image fraction, frames 1–7 | Valid motion median | Clip wall time |
| --- | --- | --- | --- | --- |
| Hurricane | 5.23 MiB | 98.63–98.69% | ~0.20 px | not instrumented |
| Demo | 5.28 MiB | 52.76–53.02% | ~0.45 px | 3.40 s |
| Deer | 4.15 MiB | 13.61–13.83% | ~0.15 px | 3.56 s |

Wall times include scene setup and export, not just kernels. GPU peak memory
was not sampled; process RSS from `time` is not GPU memory. Do not extrapolate
these simple scenes to textured production interiors without further measurement.

Visual review of all eight reference thumbnails per clip confirmed that the
objects remain in frame and camera motion is smooth. Hurricane occupies a small
part of its Cornell room; deer occupies only about 14% of the image. These are
low-motion pipeline clips, not sufficient production diversity or a strong test
of disocclusion. Add tighter framing, more parallax and foreground reveals in
the next pilot. The Sponza camera is a reserved placeholder, not visually verified.

## Independent reference qualification

Two independent replicas per level and view; entries below are estimated
single-reference noise as a percentage of the 4-SPP input error, whole / brightest
5%. Both must be ≤10%, at the selected level and every higher measured level.

| Calibration camera | 256 SPP | 1,024 SPP | 4,096 SPP |
| --- | --- | --- | --- |
| Hurricane frame 0 | 13.14 / 14.04 | 6.74 / 7.32 | 3.26 / 3.47 |
| Hurricane frame 7 | 13.09 / 13.24 | 7.20 / 7.70 | 2.92 / 2.69 |
| Demo frame 4 | 12.46 / 11.77 | 6.26 / 4.76 | 3.13 / 2.94 |
| Deer frame 4 | 12.55 / 19.43 | 6.19 / 15.98 | 3.14 / 11.64 |

Hurricane and demo pass at 1,024 SPP. Deer fails highlights even at 4,096 SPP.
Inspection of the highest-level anchor's highlight mask found no primary
surface hits in either demo or deer highlights: the brightest pixels are sky.
For deer the highlight input-error denominator is only about 0.0001723 linear
RMSE, and reference highlight noise at 4,096 is about 0.00002005. This makes the
relative test sensitive to small background differences. The pilot does not
establish whether these arise from sampling, boundary coverage or floating-point
accumulation; investigate before interpreting them as useful object convergence.
Do not retroactively exclude sky or relax the threshold to obtain a pass.

Next: preregister closer views and a background/foreground diagnostic alongside
the existing whole/highlight rule, inspect background accumulation behavior, then
repeat calibration. If the rule still fails, explicitly measure higher reference
budgets. Requalify at production resolution and on new scenes before scaling up.

## Artifacts and build

Ignored local artifacts remain in:

* `runs/ntasad-pilot-v1/preview/`: 24 paired HDR frames and manifests.
* `runs/ntasad-pilot-v1/contact-sheet.png`, `inspection.json`, and `provenance.json`.
* `runs/ntasad-pilot-convergence-v1/`: exact plan, 24 replica jobs, logs,
  reference/input EXRs, `report.json` and `summary.csv`.

The inspection can be repeated using:

```bash
python tools/inspect_temporal_pilot.py runs/ntasad-pilot-v1/preview
```

The capture executable was built in `/tmp/ntasad-capture-build` with Release,
sm86, GCC 8.5.0 and CUDA 13.3.73. CMake 4.4.4/Ninja were installed in the existing
temporary Python environment; OpenEXR C++ 2.5.10 was built into
`/tmp/ntasad-capture-deps`. The renderer's pinned NexusBVH submodule was initialized.
CMake now conditionally creates its optional benchmark target when the source
exists; this checkout lacks `benchmarks/render_benchmark.cpp`.

The reference-convergence and dataset-split tests passed: **21 tests**. This is
in addition to the real capture, sequence, crop and geometric-motion checks.
Existing uncommitted recurrent implementation changes were preserved.
