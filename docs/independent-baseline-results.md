# Independent spatial baseline: corrected cameras and frozen test

The current checkpoint is `runs/spatial-independent-v2/model.pt`. It is a
single-frame residual U-Net, not an NTASAD reproduction. The renderer itself
was not modified for this training experiment.

## Camera quality check

The first training attempt used `dataset_split_v1.json`. Its deer validation
previews mostly show ground and sky: the cameras were near the origin, while
the loaded mesh spans about 186 x 219 x 65 world units after its configured
scale. Those views miss the animal. The numeric improvement on that run is
not a meaningful unseen-object validation result.

Version 2 preserves the same scene assignments, training hyperparameters,
training pixels and Sponza test pixels. It corrects only the two deer cameras,
using positions `[200,110,330]` and `[-220,120,330]` aimed at `[-15,104,16]`.
Low-SPP previews confirmed the full mesh is visible before qualification and
training. No Sponza EXR was decoded while making this correction. The original
run is retained in `runs/spatial-independent-v1` with a superseded status.

The camera correction required a new split manifest and reference experiment,
rather than silently changing the frozen v1 data. The v2 convergence experiment
again selected 8,192 SPP using the same 10% whole-image/highlight rule over
independent 512/2,048/8,192-SPP replicas. See `runs/reference-convergence-v2`.
Training and test EXRs were copied byte-for-byte; only their metadata group IDs
were versioned. `data/spatial-independent-v2/reuse_provenance.json` records this.
Caustics remain disabled; this does not test photon-map convergence.

## Training and selection

The run used 8 demo/prism training views, 2 deer validation views, 4-SPP inputs,
8,192-SPP references, 128x96 images, and the unchanged 119,603-parameter model.
Adam ran for 20 epochs with seed 7, batch size 4, 64x64 patches, 8 patches per
frame, learning rate 0.001 and linear loss weight 0.1. The epoch with the lowest
mean deer log-L1 was eligible only if mean HDR RMSE did not exceed noisy input.
Epoch 20 won. The sky occupies much of the deer views, so full-image averages
should be read with the high-gradient diagnostics and previews.

The checkpoint was frozen before decoding test pixels. `freeze.json` records
its hash, selection rule, split/config/reference-policy hashes and source
hashes. Stored staging paths were relocated before freezing; weights and
architecture were unchanged. Only this frozen model was evaluated on Sponza;
test results did not choose its epoch, loss, architecture or settings.

## Results

Values are unweighted means over two images per partition. The reference is
a finite-sample, matched-integrator estimate. The reference budget was qualified
on train/validation cameras; convergence on Sponza itself was not measured.

| Partition / metric | Noisy input | Denoised |
| --- | ---: | ---: |
| Deer validation / HDR RMSE | 0.038091 | 0.032944 |
| Deer validation / Relative MSE | 0.012970 | 0.010163 |
| Deer validation / Log-L1 | 0.006615 | 0.005448 |
| Deer validation / Display PSNR (dB) | 31.218706 | 32.323177 |
| Sponza test / HDR RMSE | 0.368286 | 0.335000 |
| Sponza test / Relative MSE | 0.280473 | 0.120479 |
| Sponza test / Log-L1 | 0.049648 | 0.040170 |
| Sponza test / Display PSNR (dB) | 19.388931 | 22.119557 |

Both test images improve, but not equally. The bright Sponza view's HDR RMSE
changes from 0.69613 to 0.65075; the dark view changes from 0.04044 to 0.01925.
The aggregate test HDR reduction is about 9.0%, and substantial noise remains.
The fixed Reinhard/sRGB previews improve more than the linear HDR metric;
these two measures should not be conflated.

Deer validation HDR RMSE decreases about 13.5%, with a 1.10 dB display PSNR
increase. The top 10% reference-gradient regions improve modestly; fine details
and silhouettes still need work. The brightest 5% region mostly selects clean
sky here, rather than object highlights. Sky RMSE increases from roughly
0.000179 to 0.000314: a small absolute regression hidden by aggregate gains.
`validation_regions.json` records both diagnostics. They did not alter the
predeclared checkpoint rule.

![Noisy, denoised and reference comparisons](../runs/spatial-independent-v2/comparison_grid.png)

![Validation history](../runs/spatial-independent-v2/training.png)

## Performance

The RTX 3090 took 50.4 seconds for training and validation
inside the training call. Warm batch-1 float32 inference at 128x96, over 100
iterations after warmup, measured:

| Scope | Median | 95th percentile |
| --- | ---: | ---: |
| Device forward and HDR decode | 0.679 ms | 0.724 ms |
| Feature construction, upload, forward, decode, download | 2.410 ms | 2.526 ms |

Model loading and EXR/PNG I/O are excluded. Peak PyTorch allocated memory was
about 9.31 MiB including resident models; this is
not total process/device memory. The benchmark uses one validation frame,
PyTorch 2.7.0 / CUDA 12.8 and four CPU threads. This
does not establish performance or quality at production resolution.

## Artifacts and reproduction

`runs/spatial-independent-v2` contains `model.pt`, the exact expanded training
configuration, split audit, per-epoch history, summary, validation and test
reports, `freeze.json`, benchmark, and linear HDR predictions. Each prediction
also includes a PNG comparison and JSON report. `execution/` preserves the
original staging scripts as execution provenance, not portable entry points.

Frozen model SHA-256:

```text
ff60b7c84de08975a88ababca30917b4cb23e2786e80a5c0e925ac6df6b068ff
```

For a new training run from the denoiser directory:

```bash
python -m wavefront_denoiser train configs/spatial_independent_train_v2.json \
  runs/spatial-independent-new --device cuda
python -m wavefront_denoiser evaluate runs/spatial-independent-new/model.pt \
  --split-manifest configs/dataset_split_v2.json --partition validation --device cuda
```

To denoise with the existing frozen model:

```bash
python -m wavefront_denoiser denoise runs/spatial-independent-v2/model.pt \
  data/spatial-independent-v2/v2_sponza_test/frame_000000 \
  runs/sponza-prediction-new --device cuda
```

This is a tiny baseline with two training scenes and one test scene. It does
not establish broad generalization, caustic quality, temporal behavior or
full-resolution performance. Future changes informed by these Sponza results
must treat Sponza as an already observed benchmark and reserve additional
unseen scenes for a fresh final test. The immediate priority is more diverse
training scenes/SPP levels, larger views, and denoiser comparisons before
adding temporal NTASAD components.
