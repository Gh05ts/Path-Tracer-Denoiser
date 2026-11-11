# Controlled lighting experiment

Historical report: the independent spatial model and its training/checkpoint
commands were retired on 2026-11-10. The current single NTASAD denoiser workflow
is in [the README](../README.md).

This experiment tests whether covering explicit area-light noise during training
helps a fixed spatial denoiser. All interventions use the same demo surface
geometry/materials and cameras. It compares matched training budgets rather
than increasing architecture size or adding more optimizer steps.

## Intervention and controls

| Condition | Sky | Area emitter | Side length | Linear RGB radiance |
| --- | --- | --- | ---: | ---: |
| sky | yes | none | — | — |
| large_area | no | center [-1,6,0] | 4 | [24,24,24] |
| small_area | no | center [-1,6,0] | 0.5 | [1536,1536,1536] |
| side_area_sky | yes | center [-4,5,-3] | 0.5 | [1536,1536,1536] |

The large and small area emitters have equal area-times-radiance, changing
light size and emitted radiance together while preserving emitted power.
Image exposure, visibility, and estimator variance are not thereby equal.
The side condition also changes placement and adds the sky; this experiment
compares lighting coverage jointly, rather than isolating each lighting factor.
Scene surfaces remain fixed, while the actual emitter geometry changes.
All emitters face downward and remain outside the cameras. The runner requires
primary albedo, shading/geometric normals, depth and validity to match exactly
across lighting variants at every train/validation/probe camera.

Each training arm has 16 images at 128x96 and 4 SPP:

- Control: four fixed cameras under sky-only lighting, with four independent
  input/reference sample ranges per camera.
- Diverse: the same four cameras under each of the four lighting conditions.

One sky group is shared byte-for-byte between arms; other captures use separate
PCG32 sample intervals, disjoint from calibration and all references. Both arms
use 64 aligned random 64x64 crops per epoch, batch size 4, 16 optimizer steps
per epoch and 20 epochs. Architecture (119,603 parameters), Adam learning rate
0.001, linear loss weight 0.1 and the seed are matched for each pair. Three
predeclared seeds (7, 19, 41) provide descriptive seed-to-seed comparisons.
They are not a statistical significance claim; the capture dataset is fixed
across these model-training seeds.

Validation uses one separate camera under all four lights. The selected epoch
minimizes mean validation log-L1, requiring HDR RMSE no worse than noisy input.
Identity epoch zero is eligible. Checkpoint selection is identical for both
arms. All six checkpoints and their hashes are frozen together before loading
the two additional probe cameras under the same four lights.

This is a **within-scene lighting/camera ablation**. Train, validation and probe
views share geometry. Lighting conditions are covered by diversified training;
the probe tests new cameras, not unseen lighting types or independent scenes.
The existing v2 scene-disjoint model is also measured as context, but its
training scenes/budget differ, so it is not the causal control arm.

Sponza is additionally reported as an already observed development benchmark.
It does not select any epoch or setting in this experiment and is not a fresh
independent test. No Sponza training pixels enter either arm.

## Reference qualification

The initial pilot captured two independent reference replicas at 2,048, 8,192
and 32,768 SPP for each condition at the validation camera. It applied the
existing 10% reference-noise-to-input-error qualification over both the full
image and brightest 5% of pixels, requiring every higher measured budget to
pass too. The side-light-plus-sky condition failed even at 32,768 SPP (10.3%
whole-image noise fraction); the runner stopped before dataset capture/training.

That pilot is retained in `runs/lighting-reference-pilot-v1`. The protocol was
extended with 65,536 SPP for the side condition, reusing all 24 original
calibration captures with their actual sample ranges, cameras and source
settings checked. The 4-SPP input used as the error baseline stays fixed at the
original 32,768-SPP capture's input while extending the reference anchor. This
prevents qualification improving merely by choosing a noisier input realization.
At 65,536 SPP the side condition passes at 8.39% whole / 6.56% highlight noise.

Each condition uses a qualifying reference count with a minimum 8,192-SPP
floor: sky, large area and small area use 8,192 SPP; side light plus sky uses
65,536 SPP. The rule is the same across conditions, while reference budgets
and remaining errors differ. This controls target noise relative to input
error, rather than spending equal reference-render compute everywhere.
Calibration uses separate sample intervals starting at 100,000,000; dataset
intervals start at 200,000,000. The framing previews use another range.

This is empirical camera-sampling qualification, not proof of unbiased or fully
converged ground truth. It is measured at a calibration camera, not every view.
Caustics are disabled. Radiance is exported as one combined RGB image: the
experiment does not measure direct/indirect path fractions separately. The
condition names describe source configuration, not a direct-only render.

## Reproduction

Build the updated optional `../new/build/dataset_capture`, then run from the
denoiser project with training dependencies installed:

```bash
python -m wavefront_denoiser lighting-study configs/lighting_experiment_v1.json \
  runs/lighting-experiment-new --capture-executable ../new/build/dataset_capture \
  --device cuda
```

The checked-in protocol reuses the saved pilot reference captures. Set
`reuse_reference_captures` to `null` to render the full reference experiment
again. Reuse is rejected if calibration scene settings or cameras change.

Outputs must be new. `--phase capture` only prepares reference calibration and
data. After that completes, `--phase train` trains, freezes and evaluates the
prepared experiment. Changing the protocol after capture is rejected. Once
models exist, the runner refuses to retune in place. For another experiment,
use a new protocol/output directory.

The output retains the exact protocol, capture configs/logs, reference replicas
and noise report, dataset inventory, six training configs/histories/checkpoints,
freeze record, per-light/per-image probe and observed-benchmark metrics, and
float32 HDR prediction EXRs with PNG comparisons. Surface-only metrics exclude
background pixels. PNGs use the same fixed Reinhard/sRGB display transform.

## Measured results

The original model remains unchanged. These are two-camera probe means per
lighting condition, averaged over the three training seeds for each new arm.
The last column is the mean paired reduction in HDR RMSE for diverse versus
control; positive values favor diverse training.

| Condition | Noisy HDR RMSE | Control | Diverse | Existing v2 | Diverse vs control |
| --- | ---: | ---: | ---: | ---: | ---: |
| sky | 0.03753 | 0.03756 | 0.02673 | 0.02399 | +28.8% |
| large_area | 0.38407 | 0.38342 | 0.19132 | 0.26552 | +50.1% |
| small_area | 2.57947 | 2.57585 | 0.92573 | 0.98876 | +64.1% |
| side_area_sky | 1.37955 | 1.37640 | 0.21029 | 0.18595 | +84.7% |

Diverse training improves mean paired HDR RMSE in 3 of the 3 area-light
conditions. See the per-seed values and surface-only metrics before drawing a
generalization conclusion. The comparison isolates training-lighting coverage
on this one scene/camera protocol; it does not identify lighting as the sole
cause of the original Sponza result.

![Lighting comparison](../runs/lighting-experiment-v1/lighting_comparison.png)

![Fixed seed-7 comparisons](../runs/lighting-experiment-v1/comparison_grid.png)

The previews use predeclared seed 7, not the best seed on probe results. Columns
are input, existing v2, control, diverse and reference.

### Selected checkpoints and observed Sponza benchmark

Sponza numbers use its existing two 4-SPP/8,192-reference-SPP captures. Reference
convergence on Sponza has not been qualified. These results are descriptive
development-benchmark measurements, not a new independent test.

| Arm | Seed | Selected epoch | Sponza HDR RMSE | Sponza display PSNR |
| --- | ---: | ---: | ---: | ---: |
| control | 7 | 5 | 0.36821 | 19.37 dB |
| diverse | 7 | 20 | 0.31331 | 22.22 dB |
| control | 19 | 6 | 0.36815 | 19.36 dB |
| diverse | 19 | 17 | 0.31445 | 21.90 dB |
| control | 41 | 3 | 0.36821 | 19.37 dB |
| diverse | 41 | 20 | 0.32167 | 22.20 dB |
| existing_v2 | — | 20 | 0.33500 | 22.12 dB |

Across seeds, Sponza HDR RMSE averages 0.31648 for diverse training versus
0.36819 for the sky-only control and 0.33500 for the existing v2 model. This is
about 14.0% below control and 5.5% below v2. Average display PSNR (22.11 dB)
is essentially unchanged from v2 (22.12 dB). Relative to v2, the new model also
loses on the sky and side-light probe conditions while improving on the two
pure area-light conditions. It is not an across-the-board replacement.

The sky-only controls selected early checkpoints (epochs 3–6), while diverse
models selected epochs 17–20 under the same mixed-light validation criterion.
The comparison therefore measures the training distribution together with the
predeclared checkpoint-selection workflow. It does not compare two models
forced to retain epoch 20, nor establish that more lighting coverage alone
solves texture, geometry, rare-path, or resolution differences in Sponza.

Models retain the original execution paths in their training provenance;
checkpoint paths in `freeze.json` and dataset inventory entries are relative
to the experiment output. Portable reproduction uses the protocol/CLI above.
All six new checkpoints were frozen before probe/benchmark evaluation.
`verification.json` records data, frozen-hash, saved-HDR and metric checks.
