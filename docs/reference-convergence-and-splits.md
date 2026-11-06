# Reference convergence and independent dataset splits

The original v1 experiment below is retained as history. Visual inspection
during training found that its deer cameras missed the mesh. Version 2 corrects
those cameras, reruns reference qualification, and reuses unchanged train/test
EXRs. Use `configs/dataset_split_v2.json` and
`configs/spatial_independent_train_v2.json` for the current baseline. See
[the training and frozen evaluation results](independent-baseline-results.md).

The original spatial example selected checkpoints using separate views of the
same scenes used for training. It remains a pipeline demonstration. The v1
dataset introduced here reserves whole scenes:

| Partition | Scenes | Images | Purpose |
| --- | --- | --- | --- |
| Train | Demo, prism | 8 | Gradient updates |
| Validation | Deer | 2 | Checkpoint selection and development metrics |
| Test | Sponza | 2 | Final evaluation after freezing a checkpoint |

`configs/dataset_split_v1.json` is the explicit assignment. Each group declares
a scene family as well as its scene name. Related Cornell variants belong to
one family. Families, scene names, captured scene fingerprints, and EXR content
digests cannot cross partitions. These checks prevent accidental leakage through
renamed capture groups. They are not a guarantee of completely disjoint assets
or visual concepts: scenes can still share primitives, materials, or loaders.

## Independent reference estimates

`configs/reference_convergence_v1.json` registers calibration views from the
training and validation partitions only. Sponza is excluded. For each camera,
the experiment renders two independent estimates at 512, 2,048, and 8,192 SPP.
All input and reference sample intervals are disjoint across those jobs.
Scene, camera, dimensions, and integrator settings stay fixed.

The sample variance across replicas estimates the noise of one reference at
that SPP. With two replicas A and B:

```text
estimated_reference_noise_rmse = sqrt(mean((A - B)**2) / 2)
```

The highest-SPP replicas are averaged to create an approximate comparison
anchor. The report also estimates the anchor's remaining noise. Comparisons
at the highest level use the other replica, rather than an average containing
the image being evaluated. The anchor is not declared noise-free ground truth.

Before capture, the qualification rule was set to reference noise no more than
10% of the error in one 4-SPP input against the anchor. It must pass both over
the entire image and over the brightest 5% of pixels, selected by the anchor's
luminance. A budget qualifies only if it and every higher measured budget pass
for every calibration view. If no budget qualifies, the report says
`insufficient_budget`; dataset capture refuses to silently use it.

This is an empirical screen using a small number of replicas and cameras,
not a confidence interval or proof of full convergence. It measures camera
sampling at fixed integrator settings. Truncation, estimator bias, photon count,
gather radius, and fixed-photon-map bias are not tested. The current v1 example
has photon-mapped caustics disabled.

The GPU experiment selected 8,192 SPP. Estimated noise fractions were:

| Calibration scene | 512 SPP, whole / highlights | 2,048 SPP, whole / highlights | 8,192 SPP, whole / highlights |
| --- | --- | --- | --- |
| Demo | 9.04% / 18.00% | 4.54% / 11.03% | 2.21% / 6.91% |
| Prism | 10.54% / 11.31% | 5.34% / 5.73% | 2.71% / 2.97% |
| Deer | 9.05% / 18.57% | 4.48% / 15.35% | 2.25% / 4.39% |

Only 8,192 SPP passed the 10% rule for both regions in all three cases. The
lower budgets still illustrate the expected reduction in whole-image sampling
noise; bright regions require a more conservative budget in this experiment.
The result and plot are in `runs/reference-convergence-v1`.

## Commands

Run from the denoiser directory after building the updated renderer capture
tool. Use fresh output directories; completed outputs are not overwritten.

```bash
python -m wavefront_denoiser reference-convergence \
  configs/reference_convergence_v1.json runs/reference-convergence-new \
  --capture-executable ../new/build/dataset_capture --plot
```

The core analysis needs the normal dataset dependencies. `--plot` additionally
needs `pip install -e '.[analysis]'`. The output contains the exact job configs,
capture logs, HDR replicas, `report.json`, and `summary.csv`; the optional
`convergence.png` visualizes the noise fractions.

```bash
python -m wavefront_denoiser capture-split \
  configs/dataset_split_v1.json runs/reference-convergence-new/report.json \
  --capture-executable ../new/build/dataset_capture
python -m wavefront_denoiser split-audit configs/dataset_split_v1.json
```

The capture command uses the measured reference budget for all registered
groups and writes a separate `data/spatial-independent-v1` dataset. It preserves
the convergence report and its checksum in a reference policy. Test references
receive the same preregistered budget; test-scene convergence is deliberately
not used to tune that budget. Qualification on calibration cameras does not
guarantee that every new view or test reference has converged.

The audit reads JSON metadata and checks file existence, but never reads EXR
pixels. Its `test_pixels_loaded` field therefore describes the audit itself,
not a security restriction on who can open files. Metadata checksums and image
digests record the inventory; test pixel integrity is checked when explicitly
loading test images for final evaluation.

## Training and final evaluation

```bash
python -m wavefront_denoiser train \
  configs/spatial_independent_train.json runs/spatial-independent-new --device auto
python -m wavefront_denoiser evaluate runs/spatial-independent-new/model.pt \
  --split-manifest configs/dataset_split_v1.json --partition validation --device auto
```

Training expands its input groups from the split manifest, checks scene
separation and reference qualification, and reads only train/validation pixels.
It writes `split_audit.json` and records the same provenance in the model
checkpoint. Existing legacy configurations still work, but do not provide the
whole-scene independent-test guarantee. `require_reference_policy: false` is an
explicit override for synthetic tests or deliberately unqualified experiments.

Only after selecting and freezing the model, run the final test explicitly:

```bash
python -m wavefront_denoiser evaluate runs/spatial-independent-new/model.pt \
  --split-manifest configs/dataset_split_v1.json --partition test --device auto
```

This verifies the split and train/validation inventory against the checkpoint,
then reports the checkpoint hash with the final test metrics. Generic group
evaluation refuses the checkpoint's reserved test groups. The workflow guard
cannot prevent manually opening images or repeatedly tuning against test
results; preserving test independence also requires that experimental discipline.

The original dataset-only step did not train a model or evaluate test quality.
The subsequent version 2 training and evaluation are documented separately.
Renderer checkpoint formats and
render resume behavior are unchanged; only capture metadata gains a scene
fingerprint. That fingerprint excludes the display name, camera, and sample
budget and reuses the renderer's scene/config hashing, including primary asset
contents. It is not a complete cryptographic dependency manifest for every
external buffer or texture.
