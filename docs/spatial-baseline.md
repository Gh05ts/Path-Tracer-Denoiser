# Spatial HDR baseline

This model denoises one image at a time. It has no temporal state, frame order,
motion-vector input, or animated-scene requirement. Capture group IDs retain
the existing `sequence_id` name solely to select explicit data splits. Every
captured view is an independent training example; the spatial loader permits
gaps in frame numbering.

## Model and features

`SpatialDenoiser` is a two-downsampling residual U-Net with 3x3 convolutions,
ReLU activations, average pooling, bilinear upsampling, and skip connections.
The default 16-channel base has 119,603 parameters. A zero-initialized output
head starts as an identity transform in log-radiance space.

Feature format v1 has 12 channels:

| Input | Channels | Transform |
| --- | --- | --- |
| Noisy linear RGB | 3 | `log1p(L)` |
| Linear albedo | 3 | Unchanged |
| World-space shading normal | 3 | Signed, unchanged |
| Camera-forward depth | 1 | `depth / (1 + depth)` |
| Surface validity | 1 | Binary |
| Actual sample count | 1 | `1 / sqrt(count)` |

All features come from the noisy input or deterministic guides. References,
geometric normals, motion, and history do not enter the model. At least one
path per pixel is required. The model pads to a multiple of four internally
and crops back to the original dimensions, including odd and tiny images.

The network predicts a residual added to the input's `log1p(L)`. Training
predictions remain unconstrained so initially black noisy pixels can receive
useful gradients. Inference applies `max(expm1(prediction), 0)` and rejects
non-finite results. HDR values above one are retained in the output EXR.

## Training and model selection

Training uses aligned random input/reference crops and Adam. The loss is:

```text
mean(abs(predicted_log - log1p(reference)))
  + linear_loss_weight
    * mean(((expm1(predicted_log) - reference) / (1 + reference)) ** 2)
```

The default linear weight is 0.1. Gradients are checked for finite values and
clipped to norm 1. The HDR term discourages errors around high-radiance features
that a compressed-space loss alone can underweight.

Train and validation capture IDs must be supplied explicitly and be disjoint.
There is no automatic split of neighboring frames. Each epoch evaluates full
validation images. Checkpoints are selected by mean validation log-L1, with a
guard requiring mean linear HDR RMSE no worse than the noisy-input baseline.
The initial identity model is also a candidate: `best_epoch: 0` means training
did not produce a qualifying improvement. This selection rule protects the
reported aggregate, not every individual image or every metric.

Each run writes `config.json`, `split.json`, `history.jsonl`, `summary.json`, and
`model.pt`. Weights include architecture/feature versions and load using
PyTorch's `weights_only=True`. Run directories must be new. The current command
does not resume optimizer state; a checkpoint is for inference/evaluation.
The random seed is recorded; bitwise agreement across GPU/PyTorch versions is
not guaranteed.

## Inference

The command accepts a capture directory, capture manifest, or a standalone EXR.
Standalone inputs need these named channels:

```text
radiance.R/G/B, albedo.R/G/B, normal.X/Y/Z,
depth.Z, guide_valid.Y, sample_count.N
```

Spatial inputs use the [frame conventions](frame-format.md), with zero-origin
EXR windows and float32 channels except uint32 counts. No reference image is
required. If complete reference RGB channels are present, inference also
reports error metrics and includes the reference in the preview.

A new output directory receives `denoised.exr` (linear float32 `R/G/B`),
`comparison.png`, and `report.json`. Only the PNG uses the fixed display
transform: Reinhard `L/(1+L)`, followed by the sRGB transfer function. Each panel
uses the same transform. Input captures are never modified.

Inference is currently full-frame PyTorch. There is no tiled inference, C++
runtime, or automatic hook in the renderer's final image output yet.

## Metrics and limits

Evaluation reports noisy and denoised metrics for every image and their
unweighted image means:

- Linear HDR RMSE.
- Relative MSE, using denominator `reference**2 + 0.01` per RGB component.
- Log-L1 in `log1p` radiance space.
- PSNR after the fixed display transform, capped at 120 dB for identical images.

The small example uses 4-SPP inputs and 512-SPP matched-renderer references at
128x96. It trains on eight views and validates on four separate views of the
same two procedural scenes. This checks learning and the complete workflow;
it does not establish generalization to new scenes, high-resolution quality,
other SPP levels, or converged references. Photon-mapped caustics are disabled
in this training example.

The example checkpoint selected epoch 20 on an RTX 3090. Its four validation
images had these mean results against the 512-SPP references:

| Metric | Noisy input | Denoised |
| --- | --- | --- |
| Linear HDR RMSE | 0.36173 | 0.23079 |
| Relative MSE | 2.68040 | 0.51787 |
| Log-L1 | 0.03380 | 0.02372 |
| Display PSNR | 26.35 dB | 30.60 dB |

These are validation results used in checkpoint selection, not an independent
test set. Per-image values are saved in the run's `evaluation.json`.

A subsequent run uses whole-scene splits and qualified 8,192-SPP references.
See [the independent baseline results](independent-baseline-results.md). More
scenes, materials, SPP levels, resolutions, and comparisons with established
denoisers are still needed before attempting the NTASAD spatial architecture.
