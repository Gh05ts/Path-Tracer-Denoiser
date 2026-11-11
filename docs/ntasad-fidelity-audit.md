# NTASAD paper and supplement fidelity audit

This audit precedes long training and performance work. It compares the current
LARGE Python sampler/denoiser to Hasselgren et al., *Neural Temporal Adaptive
Sampling and Denoising* (Eurographics 2020), including the supplement, and the
positive-count renderer derivative cited from DASR. It does not establish
equivalence to unpublished author code or reproduce the paper's quality results.

Sources:

* [NVIDIA publication and original attachments](https://research.nvidia.com/publication/2020-05_neural-temporal-adaptive-sampling-and-denoising).
* [Main paper](https://research.nvidia.com/sites/default/files/pubs/2020-05_Neural-Temporal-Adaptive/NTASAD_preprint.pdf), sections 2.1–2.2, Figure 3, equations (1)–(4).
* [Supplement](https://research.nvidia.com/sites/default/files/publications/supplemental.pdf), section 1, Figures 1–2.
* [DASR](https://cseweb.ucsd.edu/~viscomp/projects/dasr/paper/EGSR18_Sampling_LoRes.pdf), section 4.2, equations (3)–(8).

The locally audited PDF SHA-256 values are, respectively:

```text
NTASAD:    b7e21ddc22efcba0da7141594b15b216782df4cfceb088008117210b2a4b804c
Supplement:eaacc1bafe343a5485900598cba0a59d61ab4b7659442a157ad4fc472b6badaa
DASR:      fcd3451bb2dfd58ddf89a3628c5a21d8723272215dec6ba0ca62067ccea2c94c
```

## Correction: downsample the pixel-mean image

Previously the adaptive wrapper downsampled encoded colors using their integer
sample counts as weights. For two pixels with values 2 and 10 and counts 1 and
3, that operation produces 8 rather than the image mean 6. It therefore changes
the spatial reconstruction even on dense inputs, and changes it again if counts
change while the rendered pixel means remain fixed. The prior uniform-parity
tests used equal counts and did not expose this difference.

Section 2.1 describes filtering downscaled versions of the noisy image; the
selected implementation of its downsampler is a box filter. The corrected
`pixel_mean_v2` pyramid weights each available pixel equally. Dense images now
agree with ordinary box downsampling for **all positive count maps**, with the
same backbone weights. For missing pixels, available-pixel normalization is
retained, and fractional coverage propagates between levels. This ensures that
original pixels remain equally weighted across a 4×4 footprint with unequal
2×2 coverage. It does not average unequal-sized block means equally.

The paper does not explicitly prescribe box pooling, sample-count weighting,
or its zero-sample handling. This correction removes an additional sample-mass
heuristic and makes the adaptive implementation consistent with the selected
image-pyramid interpretation. It is not evidence of an unpublished author
implementation's exact sparse downsampler. Averaging remains in transformed
space; this is not a new unbiased linear-radiance estimator.

## Requirement-by-requirement comparison

| Source | Published behavior | Implementation and finding |
| --- | --- | --- |
| Figure 2 | Separate sampler and denoiser, both conditioned on warped previous denoised output | `JointNTASAD` implements both recurrent paths; neither history path is detached. Reference RGB is absent from prediction features. |
| Figure 3, LARGE encoder | Two full-resolution 48-channel convolutions; one 48-channel convolution at each of I/2, I/4, I/8, I/16, I/32 | `NTASADDenoiser` and `AdaptiveSampler` use these feature widths and five downsampling stages. |
| Figure 3, LARGE decoder | Two feature convolutions at each decoder level: 96 channels at I/16, I/8, I/4, I/2; 64 at I | Both networks use these widths. Concatenated skips agree with the supplement's description of the larger network. Exact skip attachment within the block remains an interpretation. |
| Section 2.1 | 3×3 convolutional kernels and ReLU feature blocks | All feature and prediction convolutions are 3×3. Prediction heads are unactivated; kernel softmax and scalar sigmoid follow separately. |
| Figure 3 and section 2.1 | 5×5 spatial kernels at I, I/2, I/4; an additional 5×5 history kernel at I | Heads have 51, 26, 26 channels. Filtering is shared across RGB channels and comes from one denoiser backbone. |
| Equation (1) | Backward bilinear motion warp at x+mx, y+my | Pixel-space backward motion, bilinear interpolation, no rounding to integer motion. Existing gather joint backend is retained; separate tests cover values, interior derivatives and boundary conventions. |
| Equation (2) | Per-image spatial softmax scaled by pixel count and target SPP; independent rounding | `paper_softmax_v1` follows this equation, rejecting bank overflow. Default `capped_softmax_v1` remains an explicitly named capacity variant. No minimum count or integer-budget repair is introduced. |
| Equation (3) | Recursive fine − alpha·U(D(fine)) + alpha·U(coarse), nearest U | Both reconstructions use this low-frequency replacement, not a direct convex blend of scales. Default adaptive D now preserves the dense pixel-mean pyramid at unequal SPP. |
| Finest temporal blend | Filter warped history and linearly blend with multiscale current output | Two separate finest kernels and a temporal scalar; history is not merely concatenated into a direct RGB predictor. |
| Section 2.1 features | Current noisy RGB, RGB history, RGB albedo, normal, depth, motion for the denoiser | 15 features. Count and validity are used as contracts/masks, not extra learned features. Guide conventions remain those of capture. |
| Section 2.2 sampler features | Albedo/history colors converted to CCIR 601 gray; current geometry guides | Eight channels, coefficients 0.2989/0.587/0.114. No current noisy RGB or reference input. |
| Training initialization | First frame uses uniform noisy RGB at target SPP; four learned recurrent steps in a typical five-frame window | `JointNTASAD` replays uniform frame zero, initializes state directly, then invokes both networks four times. There is no learned denoising pass on initialization. |
| Equation (4) | Equal spatial and temporal L1; spatial loss only on last frame, temporal difference only on final pair | `sequence_loss` follows this structure, with no motion warp of temporal differences and no direct losses on earlier outputs. Earlier networks receive gradients through recurrence. |
| Section 2.2 radiance | Input clamp 0..65535 and log(1+x)^(1/2.2); log-space training | Exact transform at positive values and exact black forward value. Features, spatial kernels, history and loss use transformed radiance; public APIs decode to linear HDR. Domain boundaries are documented below. |
| Section 2.2 optimizer | Xavier initialization; Adam, initial LR 0.001; 1000 epochs in reported runs | Xavier-uniform conv weights, zero biases, Adam defaults and LR match. Batch/crop choices and model selection are local settings; no staged DASR loss/schedule is imported. |
| Section 2.2 augmentation | Aligned random crops, x/y flips, 90-degree rotations | Joint crops transform layers, reference and guides together. Motion components change with image coordinates; captured world-space normal components retain their basis. No direct-prediction-only hue augmentation is added. |
| Section 2.2 replay | Independent 1/2/4/8/16/32-SPP means combined by binary selection | Replay weights selected layers by SPP before dividing in linear radiance. Six layers represent 0..63, despite the paper's broader description of up to 64 samples. Capture provenance checks independence. |
| DASR equation (8) | dI/ds approximately (reference − I)/s for positive counts | Linear-domain RGB dot product against upstream gradient, actual rounded positive count denominator, then ordinary encoder chain rule. No target gradient or target-dependent forward values. Zero-count behavior remains an explicit extension. |

## Supplement: SMALL is a distinct architecture

Supplement Figure 2 retains all denoiser resolutions but uses 24 channels in
the full-resolution encoder and levels I/2 through I/16, **48 at the I/32
bottleneck**, 48 in the four coarse decoder levels, and 32 at full resolution.
The output heads remain 26/26/51. Thus simply setting `base_channels=24` is
not the exact depicted SMALL denoiser: the existing uniform-width encoder
would give a 24-channel bottleneck. The current supported reproduction target
is LARGE; arbitrary narrower smoke-test widths are ablations.

Supplement Figure 1's SMALL sampler is also distinct: two resolutions (I,
I/2), eight features, and residual rather than concatenated skips. A narrow
six-resolution `AdaptiveSampler(8)` is not that network. Neither SMALL topology
is claimed as implemented. The supplement's later figures report comparative
quality/generalization and SVGF variants; they add no denoiser equations to
implement. Their empirical claims require trained evaluations, not unit tests.

## Details not fixed by the publication

* Network downsampling uses average pooling, upsampling uses nearest neighbors,
  and skips concatenate before the decoder feature convolutions. No exact
  pooling operator, padding scheme or normalization constants are supplied.
* Kernel weights use a softmax, and blend weights use a sigmoid. The paper does
  not give their activation formulas. The 26-channel heads' scale weights are
  nearest-upsampled to control composition at the next finer level, consistent
  with the finest head's 51 channels being occupied by two kernels and the
  temporal scalar. Exact scalar placement is not independently verified.
* Radiance is transformed before pyramid construction and history warping;
  grayscale follows transformed-history warping. The publication specifies
  input transformation and log-space training, but not every operator's domain
  boundary or the ordering of grayscale and nonlinear preprocessing.
* Missing-tap masking, normalized available-history warping, center geometric
  validity gates, sparse scale fallbacks, and output availability are engineering
  extensions. All failed/zero predictions remain in the training loss.
* Output bounding to the encoded input range prevents scale-composition
  overshoots from creating invalid HDR. The paper explicitly bounds inputs,
  not outputs. Black uses a finite zero encoder subgradient. These choices are
  retained and must remain part of experiment provenance.
* `unit_denominator` and `zero` extend the undefined zero-count DASR derivative.
  Hard masks and binary layer selections are not differentiated. In particular,
  a masked zero-count tap can suppress its local allocation gradient; a finite
  zero-policy denominator does not fix that limitation. No sampler superiority
  follows from the gradient smoke checks.
* The LARGE denoiser has 1,114,311 parameters (1,112,976 convolution weights);
  the sampler has 1,037,457 (1,036,224 weights). Excluding biases does not explain
  the paper's approximate 850k statement for its modified DASR comparison.
  That statement is not an exact parameter inventory for the adaptive KP model.
  Author-code parameter-count equivalence remains unverified; the audit does
  not silently change Figure 3's channel counts to force a numerical match.

## Checkpoint and training compatibility

The uniform model, uniform checkpoint format and uniform training semantics
are unchanged. The adaptive default is
`ntasad_adaptive_spatiotemporal_unet_v2` with the same feature version 3 and
unchanged learned parameter shapes. New joint checkpoints use envelope v2,
architecture `ntasad_joint_v2`, and an explicit `reconstruction_version`.

Original v1 joint checkpoints still load using `sample_mass_v1`; neither their
outputs nor optimizer/RNG state is reinterpreted as the corrected policy.
Continuing their training requires `"reconstruction_version": "sample_mass_v1"`
in the config. Loading for inference selects that policy automatically. Saving
a loaded legacy model uses a v2 envelope with the legacy reconstruction
identity and policy intact. Switching policies during ordinary resume is
rejected before an output directory is created. A new v2 experiment starts
fresh; copying weights intentionally is a separately labelled warm start.

## Verification and next steps

Independent numerical checks cover dense unequal-SPP parity with the uniform
box-pyramid implementation, sparse count invariance at fixed pixel means,
multi-level coverage, and an explicit legacy result that differs from the
corrected policy. Checkpoint tests exercise original v1 loading, Adam's next
update, re-saving legacy semantics, v2 identity checks, policy mismatch
rejection and explicit legacy trainer continuation. Existing checks cover
kernel/warp derivatives, final-pair loss, input isolation and exact CPU resume.

The GPU-enabled model/data/checkpoint suite passed **266 tests**, including the
C++ EXR interoperability fixture, with no failures or skips in that selection.
The attempted full suite stalled in the existing renderer capture test
`test_gpu_v5_full_clip_pipeline_reuses_source_and_resumes`, on frame zero of
its tiny 17×13 source capture. Its child process was stopped after several
minutes without producing a frame. Both renderer capture modules were then
excluded from the completed run. This is a current capture-validation gap;
the earlier 278-test handoff result must not be reported as a fresh full pass.
Renderer diagnosis was deferred in keeping with the requested scope.

The actual pre-audit Demo pilot checkpoint was evaluated before editing. Its
outputs, validity, expected allocations and counts are bitwise identical after
loading with the compatibility path on CPU. No historical checkpoints were
rewritten. GPU checks use full default 48-channel networks over five frames of
captured 17×13 input and a synthetic 128×128 crop: all four learned allocations
and both networks receive finite nonzero gradients, one Adam step changes both
networks, and checkpoint round-trip outputs match bitwise.

Renderer-workspace artifacts live under `build/ntasad-v5/fidelity-audit/`:
`gpu-smoke.json`, `pytest.xml`, and the reviewed change manifest/patch. Tiny
captured inputs, synthetic crops and regression success establish computation
and compatibility, not trained denoising quality or sampler gains. Production
v4 data, native-resolution overfit/convergence and fixed-denoiser matched-budget
uniform/random/learned sampler comparisons remain outstanding.

Grid-sample/backend changes, throughput modes, loader optimization, and long
training have not been introduced. Revisit measured performance only after
reviewing this fidelity baseline and before long training. Renderer adaptive
scheduling and CUDA inference integration remain the last phase.
