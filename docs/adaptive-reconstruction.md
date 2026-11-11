# Adaptive reconstruction and the renderer-gradient estimator

The v5 layers can now feed a count-aware reconstruction model and a
differentiable training-time replay operator. The [joint trainer](joint-training.md)
now combines these building blocks with crop-aware data, explicit allocation
capacity and resumable two-network checkpoints.

## Paper derivation and radiance domain

NTASAD section 2.1 cites Kuznetsov et al. for the renderer-gradient approximation.
The [DASR paper](https://cseweb.ucsd.edu/~viscomp/projects/dasr/paper/EGSR18_Sampling_LoRes.pdf),
section 4.2, derives the following expression in equation (8):

```text
d I_s / d s ≈ (I_reference - I_s) / s,       s > 0
```

It obtains this by averaging independent appended samples and taking their
mean to a reference estimate, then letting the additional sample count tend to
zero. The original method starts with positive sample counts; this expression
does not define a derivative at zero. It is an estimator, not the derivative of
the discontinuous integer layer-selection function.

The audited PDF has SHA-256
`fcd3451bb2dfd58ddf89a3628c5a21d8723272215dec6ba0ca62067ccea2c94c`.
[NTASAD](https://research.nvidia.com/sites/default/files/pubs/2020-05_Neural-Temporal-Adaptive/NTASAD_preprint.pdf)
still supplies the model, five-frame unrolling and equal spatial/temporal L1
loss. DASR's different losses and staged training schedule are not imported.

Our estimator runs in **linear radiance**, where independent layer means can
be combined by sample count. The existing gamma/log encoder follows as an
ordinary differentiable operator; its chain-rule derivative is preserved.
Using `(encode(reference)-encode(noisy))/s` instead would be a different
surrogate. Existing HDR clamping and the zero subgradient at exactly black
remain unchanged.

## Training replay

`renderer_gradient.differentiable_replay` takes N,6,3,H,W layer means,
differentiable N1HW expected allocations, and an independent N3HW linear target.
It rounds the allocation and uses the existing exact binary replay in forward.
Changing the target cannot change a forward pixel. It returns linear `radiance`,
integer `counts` and the selected `zero_policy`.

Backward sends the RGB dot product of the upstream gradient and equation (8)
to each expected allocation, using the **actual rounded positive count** in
the denominator. It retains ordinary gradients into layer values, but returns
no gradient into the target. Inference uses `replay_sample_layers` directly and
does not need a reference. Targets must come from the disjoint reference ranges;
they must never be supplied as sampler features or forward image values.

The caller must explicitly select one zero-count extension:

| `zero_policy` | Positive counts | Zero counts |
| --- | --- | --- |
| `unit_denominator` | Equation (8) | Denominator 1, giving a finite reference-minus-zero surrogate |
| `zero` | Equation (8) | Local allocation derivative set to zero |

Neither policy traces a mandatory pilot sample or changes the forward count.
Neither extension is specified by the papers. The policy is part of experiment
provenance and is saved in the joint checkpoint/configuration.

The operator rejects negative/nonfinite allocations and rounded counts outside
0..63. It does not silently clip, impose a minimum, or repair rounded budgets.
A separate finite-capacity policy is provided by `allocation.allocate_samples`
and documented with the joint trainer. Higher-order differentiation of this custom estimator is
unsupported; ordinary first-order optimization is supported.

Reconstruction uses hard integer-derived tap/availability masks and pyramid
weights. This surrogate does **not** differentiate those discrete decisions.
At a missing current tap, masking can suppress its local radiance gradient;
the unit-denominator extension does not manufacture a gradient through an
excluded tap. Spatial softmax and shared network weights still couple pixel
allocations. This limitation needs to be measured during sampler experiments.

## Sparse reconstruction

`adaptive_model.AdaptiveNTASADDenoiser` owns the existing 15-feature U-Net as
`network`. Its reconstruction uses these explicit engineering choices:

* Ignore current RGB where integer `sample_count` is zero, including its feature
  input. Require finite nonnegative RGB and integer nonnegative counts.
* Build the pyramid from equally weighted available **transformed pixel means**
  (`reconstruction_version=pixel_mean_v2`). Propagate fractional pixel coverage
  between levels. With positive counts everywhere this is ordinary box
  downsampling, including when sample counts differ. Zero coverage never dilutes
  a sampled neighbor. `sample_mass_v1` retains the earlier count-weighted pyramid
  only for explicitly selected legacy experiments/checkpoints.
* Apply spatial kernels with missing taps excluded and weights renormalized.
  Track whether each kernel neighborhood has support.
* Where both scale estimates exist, apply equation (3), using a normalized
  downsample of available fine estimates for its low-frequency term. Use a
  coarse-only estimate fully when fine support is missing; retain the fine
  estimate when coarse support is missing.
* Warp history color and its availability separately. Divide the bilinear color
  numerator by available coverage, then filter only available temporal taps.
  The renderer's geometric center mask still gates temporal reuse.
* Blend spatial and temporal estimates only when both exist. A history-only
  estimate is used fully; a spatial-only estimate is used fully. With neither,
  return zero and mark the output unavailable.

Filtering and recurrence remain in the existing gamma/log domain, with the
existing output clamp and replicate padding. Constant signals survive sparse
holes, and the fully sampled path agrees with the uniform model when backbone
weights match, within the tested floating-point tolerance. Weighted transformed
pixel means are a documented domain choice, not an unbiased linear estimator.
See the [fidelity audit](ntasad-fidelity-audit.md) for the source comparison and
why the previous sample-mass weighting was removed from the default path.

These missing-data, scale and fallback rules are not fully specified by the
publication. Availability represents usable support, not confidence or a claim
that a prediction is accurate. Do not use it to exclude failed reconstructions
from the training loss or inflate evaluation scores.

## State and compatibility

Streaming returns `AdaptiveReconstruction(radiance, valid)`: linear N3HW RGB and
boolean N1HW availability. Preserve both in history, so a missing output cannot
be reused as measured black. A real sampled black pixel can have valid=True.
Pass `history_available=state.valid` to `AdaptiveSampler` so it uses the same
normalized warp; omitting that new argument retains the previous sampler API.

`history=None` initializes from dense uniform noisy samples. Sequence resets
require the same dense uniform initialization, independently for each batch
item. Sparse stills use `forward_spatial`; they also return availability.
`forward_sequence` returns NT3HW/NT1HW state, and `transformed=True` keeps its
radiance in gamma/log space for `sequence_loss`.

The default adaptive architecture is `ntasad_adaptive_spatiotemporal_unet_v2`, feature
version 3. Existing uniform model/trainer/checkpoint semantics are unchanged.
The uniform checkpoint saver rejects this wrapper instead of labelling adaptive
weights as a uniform model. The joint trainer uses a distinct two-network
checkpoint format, including its allocation, gradient and warp versions.
Original joint v1 checkpoints load with `sample_mass_v1`; no weights or optimizer
moments are migrated to different reconstruction semantics. New joint v2
checkpoints record the selected reconstruction version explicitly. Continuing a
v1 training run requires `reconstruction_version: "sample_mass_v1"` in its config.
For an intentional warm start, copy only the compatible backbone weights:

```python
adaptive.network.load_state_dict(uniform.state_dict())
```

The main paper-reproduction experiment can still initialize both networks from
scratch; warm starts are a separately labelled experiment.

## One recurrent training step

```python
from wavefront_denoiser.adaptive import AdaptiveSampler
from wavefront_denoiser.adaptive_model import AdaptiveNTASADDenoiser
from wavefront_denoiser.renderer_gradient import differentiable_replay

sampler = AdaptiveSampler()
denoiser = AdaptiveNTASADDenoiser()
# frame0 contains dense uniform noisy radiance and the usual capture buffers.
state = denoiser(frame0)

allocation = sampler(current_guides, state.radiance, average_spp=4,
                     history_available=state.valid)
render = differentiable_replay(layers, allocation.expected, reference,
                               zero_policy="unit_denominator")
current = dict(current_guides, radiance=render.radiance,
               sample_count=render.counts)
state = denoiser(current, state)
```

`current_guides` must include the denoiser's guide/motion/mask buffers as well as
the sampler's guides. The target is only supplied to the replay estimator.
Repeat for four learned iterations and apply the existing final-frame/final-pair
loss. The example constructs untrained CPU models; move models and inputs to
the same device/dtype in a real run.

## Validation and remaining work

The full suite passes **246 tests with no skips** on an RTX 3090, including
**32 new tests** for reconstruction and the estimator. Checks cover shared-weight
uniform parity, singleton/odd shapes, sparse black/HDR constants, empty frames,
normalized history warping and its RGB/motion derivatives, reset boundaries,
no target-dependent forward values, exact layer-value gradients, RGB reduction,
softmax/transform chain rules, and a five-frame update of both networks.

The estimator is checked against equation (8) and the independent appended-mean
limit. A finite-difference `gradcheck` of discontinuous integer replay would be
the wrong oracle; the smooth normalized warp is separately gradchecked.

A separate RTX 3090 smoke used the captured 17x13 v5 fixture, five frames and
16-SPP references. It explicitly zero-initialized the sampler head for uniform
4-SPP numerical checking. The final loss produced finite nonzero gradients at
all four allocations, sampler head and denoiser head; one Adam update changed
both heads. This is a gradient-path check, not production training, learned
sampling effectiveness or a denoising-quality result.

The crop-aware joint trainer, explicit capacity/budget policy, resumable
two-network checkpoints and small training/validation runs are now implemented;
see [joint training](joint-training.md) for the subsequent 278-test validation.
Production v4 inputs remain expected at `../dataset/denoiser-v4`; their arrival
and validation are still deferred. Live per-pixel CUDA scheduling is separate.
