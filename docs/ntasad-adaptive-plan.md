# NTASAD adaptive sampler: audit and implementation plan

## Current milestone

The renderer in `../new` is a CUDA wavefront tracer, with graph and cooperative
persistent execution. This project already contains a reconstructed uniform-SPP
NTASAD denoiser. The first adaptive milestone adds a separate sampler, equation
(2) normalization, and forward replay of independent sample layers. It does not
yet connect the sampler to live per-pixel CUDA rendering or provide joint
training. Independent layer-only capture, the v5 planner and a validated reader
are now implemented; see [the v5 workflow](../dataset_generation/V5.md).
[Sparse reconstruction and the DASR gradient](adaptive-reconstruction.md) are
also implemented, with explicit availability state and zero-count policies.

Primary sources, read including the architecture figures:

* [Hasselgren et al., NTASAD, sections 2.1–2.2 and Figure 3](https://research.nvidia.com/sites/default/files/pubs/2020-05_Neural-Temporal-Adaptive/NTASAD_preprint.pdf).
* [Supplement, section 1 and Figures 1–2](https://research.nvidia.com/sites/default/files/publications/supplemental.pdf).

The downloaded paper SHA-256 is
`b7e21ddc22efcba0da7141594b15b216782df4cfceb088008117210b2a4b804c`;
the supplement is
`eaacc1bafe343a5485900598cba0a59d61ab4b7659442a157ad4fc472b6badaa`.
These identify the audited PDFs; PDFs are not committed.

## Independent denoiser check

| Paper requirement | Source implementation and assessment |
| --- | --- |
| Bilinear backward history warp, equation (1) | `temporal.reproject_history`: previous-minus-current pixel motion, differentiable `grid_sample`; sign, fractional motion and singleton axes tested |
| Three spatial 5x5 kernels and one temporal 5x5 kernel | `model.NTASADDenoiser`: 26/26/51 output parameters, normalized kernels and learned blends |
| Recursive low-frequency replacement, equation (3) | `temporal.combine_scales`: fine + alpha*(upsampled coarse - upsampled downsampled fine); analytic oracle tested |
| Recurrent history from previous denoised RGB | Explicit history, no detach within windows; reset prevents gradients crossing a sequence cut |
| One initialization frame plus four network iterations | Default five-frame windows, initial uniform noisy frame, reset tested |
| Final spatial L1 plus last temporal-difference L1, equation (4) | `loss.sequence_loss`: equal weights, no warp of temporal differences, in transformed space; placement tested |
| Clamp HDR to 65535; log(1+x)^(1/2.2) | `ntasad_radiance`: matching transform, invertible in range, finite black subgradient |
| Xavier / Adam / learning rate 0.001 / 1000 epochs | Existing defaults match; synthetic training/resume test verifies the next updates |
| Shared crop, flips and 90-degree rotations | `sequence_data`: aligned windows, adjusted pixel-space motion, unchanged world-space normal components |
| Grayscale albedo/history for sampler | New `adaptive.AdaptiveSampler`: CCIR 601 weights 0.2989/0.587/0.114; denoiser retains RGB |

The initial pre-v4 audit passed **159 tests**, with 12 optional GPU/fixture
skips, on CPU before edits. V4 subsequently unified the denoiser, trainer,
runtime and checkpoint modules; current verification is recorded below. Tests
verify executable behavior, not the authors' weights or their quality/performance
results. The current default denoiser contains
1,114,311 parameters; exact parameter-count equivalence to the publication is
unresolved. Pooling, skip attachment, normalization, boundary padding, masked
history taps and radiance-domain boundaries remain reconstruction choices
described in [the earlier audit](ntasad-progress.md). They should not be silently
changed in existing checkpoints based on ambiguous figure details.

No trained NTASAD checkpoint was found in the initial inspection of `runs/`;
the inspected checkpoints were historical spatial baselines, now unsupported by
the unified trainer. Production v4 frames are being generated on another machine.
No claim of denoising quality or learned
adaptive behavior follows from these unit tests. The initial sandbox check
reported CUDA unavailable. Execution outside the sandbox exposes an RTX 3090
and has now passed the GPU validation described below.

## Stage 1: sampler and sample representation (implemented)

`adaptive.AdaptiveSampler` follows the existing Figure 3 U-Net reconstruction
with five downsampling stages, 48-channel encoders, 96-channel coarse decoders,
64-channel fine decoder, and a scalar prediction head. It contains 1,037,457
parameters. Smaller width overrides are testing/ablation choices, **not** the
supplement's SMALL sampler. Implementing that two-level, eight-channel residual
sampler is a later performance experiment.

The eight input features are grayscale albedo (1), signed world normals (3),
depth (1), backward motion (2), and grayscale warped denoised history (1).
Current noisy radiance and reference radiance are never consumed. History is
linear RGB at the API, encoded before warping to match the current denoiser.
The exact domain order is a documented reconstruction choice. An explicit
`history=None` returns uniform counts for initialization.

`normalize_sample_map` applies softmax across the original H*W pixels **per
image**, multiplies by H*W*average_spp, and rounds independently. Padding is
cropped before normalization. It returns differentiable `expected` allocations
and non-differentiable int64 `counts`. FP16/BF16 normalization uses FP32.

Equation (2) preserves the expected budget. Its rounded total can drift; e.g.,
two expectations of 1.5 round to two counts of 2. No minimum, cap, exact-total
repair or straight-through gradient is silently added. Logit concentration can
produce zero counts or counts above the available layer capacity. These are
integration concerns that must be handled explicitly in stage 2.

`sample_layers.replay_sample_layers` combines six independent **linear RGB
means**, weighted by the layer SPP, then divides by actual count. Count 13 uses
layers 1+4+8. Count zero returns zero radiance with its count retained by the
caller; values above 63 fail. The operation differentiates layer values but
does not approximate differentiation through integer sample selection.

```python
from wavefront_denoiser.adaptive import AdaptiveSampler
from wavefront_denoiser.sample_layers import replay_sample_layers

sampler = AdaptiveSampler()
# guides: NCHW albedo, normal, depth, motion and history_valid
# previous_denoised: linear HDR N3HW (None on initialization)
allocation = sampler(guides, previous_denoised, average_spp=4)
# layers: independent N,6,3,H,W RGB means, ordered 1/2/4/8/16/32 SPP
noisy = replay_sample_layers(layers, allocation.counts)
```

This snippet demonstrates the forward interfaces only. It is not a joint
training example, and the existing uniform denoiser rejects zero counts.

## Stage 2: capture and sparse reconstruction implemented

The companion manifest, v5 planner, verified copy/resume path, independent
layer exporter/reader and zero-sample reconstruction in items 1, 2 and 4 below
are implemented. The explicit capacity/budget policy (item 3) is now available
with the joint trainer; live adaptive rendering (item 5) remains future work.

1. Define a separate, versioned companion-layer manifest. Record the six
   disjoint absolute sample intervals, per-layer SPP, means-vs-sums convention,
   EXR hashes, renderer/scene fingerprints, exact cameras and frame indices.
   Reuse the existing first-hit guide contract. Validate disjointness from
   input, training reference and report reference intervals for each frame.
2. Add capture of RGB layers without regenerating references six times. Prefer
   a dedicated layer export beside the current immutable frame directories.
   Lock cameras, scene/integrator version and pixel/sample RNG keys. Keep the
   uniform v1 frame reader and existing training runs compatible.
3. Define out-of-range allocation policy before training. The paper gives
   independent rounding but does not specify how to enforce the finite layer
   capacity. Six layers 1..32 contain **63** samples, although the paper also
   describes up to 64 individual samples. Do not silently clip to 63 or force
   minimum 1 and call it exact equation (2). Record any chosen bounded/budget
   variant as an ablation and log expected/actual path totals.
4. Add a separate adaptive denoiser version with zero-count semantics. Exclude
   missing current-frame taps, normalize sampled radiance correctly at each
   image-pyramid scale, and provide defined all-missing-neighborhood behavior.
   Maintain valid temporal reconstruction in holes. Test count 0/1/63, empty
   neighborhoods, disocclusion, reset, black and bright HDR. Do not treat an
   absent sample as a measured black sample or silently change the unified
   uniform model's checkpoint semantics.
5. Extend local CUDA task execution to accept a per-pixel count map; trace only
   active pixels in each sample round, compact primary queues, use absolute
   per-pixel sample IDs, and divide radiance sums by each pixel's count. Support
   full-frame and partial edge tiles; compare graph and persistent backends
   against uniform rendering and independent single-pixel/range renders.
   Distributed tasks and renderer checkpoints can follow after local parity.

Renderer integration seams are documented in
[`../new/docs/NTASAD_ADAPTIVE.md`](../../new/docs/NTASAD_ADAPTIVE.md).

## Dataset work on the other machine

Preserve the active v4 dataset and derive a separate v5 companion dataset from
its exact camera configurations. V4's `plan-v4.json` adds explicit anchor cameras,
uses slow/medium/fast multipliers of 0.6/1.8/3.6, and plans 234 eight-frame clips
(1872 frames). Its camera poses, motion-specific trajectories, split, sample IDs,
and generated configs must be preserved when adding layers; do not regenerate
v5 from the older v3 plan.

Reuse the completed v4 4-SPP means as the **4-SPP layer**, plus the independent
1024-SPP references and matching guides. Copy only checksum-verified completed
frames, preserve source provenance, and check cameras, assets and integrator
settings against each new layer capture. Render the missing independent
**1/2/8/16/32-SPP** layers using disjoint ranges outside both training and report
references. A single 4-SPP mean cannot be split into its constituent samples;
it can be reused as one complete binary layer. Neither a reference image nor
synthetic noise generated from it replaces the missing layers.

The paper's practical training representation is six independent rendered means
at 1/2/4/8/16/32 SPP, with fixed guides/cameras per frame. At 1920x1080, six FP32
RGB layers require about **142.4 MiB/frame**, or **260.3 GiB for 1872 v4 frames**
before compression, guides and references. Reusing the existing 4-SPP layer
leaves about **216.9 GiB** for the five new RGB layers. Avoid storing 64 RGB samples per
pixel unless another experiment requires them. Start with complete eight-frame
clips from one train and one validation scene; validate replay and independence
before expanding storage or using test scenes.

Companion layers can be generated later if the other machine preserves the
exact scene assets, camera trajectory, renderer revision, integrator settings,
and RNG conventions. Allocate new sample intervals explicitly within a reserved
stride or a new disjoint range; do not assume that unused sample IDs are free
without checking both 1024-SPP and 4096-SPP reference ranges. The separate v5
launcher now generates those companion configurations; the existing v4 launcher
and camera configurations remain unchanged.

## Stage 3: joint training implemented

The renderer-gradient estimator is implemented and checked against DASR
equation (8), with explicit zero-count extensions. Synthetic and real-fixture
five-frame gradient/update checks pass. The crop-aware joint trainer and
resumable two-network checkpoint now implement the requirements below. Its
default capped-softmax allocation is explicitly labelled as a bounded variant;
the unbounded paper policy remains selectable with overflow rejection. Joint
warping uses deterministic bilinear gathers for exact CUDA resume. See
[joint training](joint-training.md) for configuration, checkpoint and data rules.

1. Use the implemented `renderer_gradient.differentiable_replay`: linear-domain
   DASR equation (8) for positive rounded counts, with independent references
   used only in backward. Record its explicit `zero_policy` and gradient version.
   Its analytic tests are separate from the integer forward replay. Finite-difference checks of a
   discontinuous integer renderer are not proof of the estimator's correctness.
2. Unroll initialization plus four sampler -> replay -> denoiser iterations,
   warp previous denoised history with current guides, and preserve gradients
   through both history consumers. Apply only the existing final-frame spatial
   and final-pair temporal loss. Do not add a target sample-map loss.
3. Define a separate v5 joint-training configuration using the inherited v4 scene
   partitions. Save both networks, optimizer, RNG, layer inventory, budget/cap
   policy and sampler feature version in a distinct resumable checkpoint. Keep
   the unified uniform model's checkpoint and trainer compatible; do not
   reintroduce the retired independent spatial model or its trainer.
4. Demonstrate a real sampler update from final reconstruction loss, nonzero
   history gradients, exact interrupted/resumed next update, and a small
   train/validation overfit check before full dataset training. Keep test data
   excluded from model selection and convergence-budget selection.

## Stage 4: quality and performance validation

Measure sampler effectiveness separately using a fixed denoiser and identical
evaluation cameras: compare uniform allocation, a seeded random allocation with
the same integer budget, and learned allocation. Keep reconstruction weights
fixed across these comparisons, then report a separate comparison with the
jointly trained sampler/denoiser. Also compare the unified model's spatial-only
fallback and its recurrent output; these share one denoiser rather than using
the retired spatial baseline. Compare at **measured actual path totals**, and report
rounding/cap deviations rather than assuming the requested average is achieved.
Use the same explicit display transform for paper-style PSNR and tPSNR
(temporal finite differences); also retain linear/transformed-domain metrics.
Evaluate complete trajectories with camera cuts, disocclusion, bright specular
motion, mirrors/glass, slow/fast camera motion, and long-run drift. The present
static-geometry dataset does not cover animated-object motion from the paper.

Measure sampler, warp, render and denoiser time independently, peak GPU memory,
and full-frame throughput. The current `unfold` kernel application is a
correctness reference and needs CUDA optimization before any real-time claim.

## Verification commands

From the denoiser root, using an environment with the `[train,test]` extras:

```bash
python -m pytest -q tests/test_adaptive.py
python -m pytest -q
```

The sampler tests use independent numerical oracles for the softmax Jacobian,
per-image budget, backward warp, grayscale coefficients, all 64 binary layer
combinations, and layer-value gradients. Integer-count gradients and CUDA
execution are deliberately not claimed by these checks.

Original pre-v4 CPU verification: **190 passed, 11 skipped**, including all 30 sampler
tests and the optional cross-language test using a freshly rebuilt
`../new/build/tests/frame_export_tests` EXR fixture. The skipped tests require
GPU capture. The temporary system-backed test environment used an older pytest
and emitted its existing `pythonpath` configuration warning; `PYTHONPATH=src`
was supplied explicitly. Existing NumPy/networkx deprecation warnings also
remain. No CUDA capture or trained recurrent quality evaluation was run.

After the v4 sync, restoration was rebased onto commit `ab6f0fb`. Sampler code
and layer replay had no overlapping changes. README/audit text was merged into
the unified-model documentation, and references now use `model.NTASADDenoiser`,
`loss` and `sequence_data`. The v4 camera plan, capture generator, unified
model/trainer/runtime and checkpoint format were not modified by restoration.
The v5 follow-up above now inherits all 234 clips / 1872 frames and retains
the completed v4 4-SPP layer and references.

Restored-v4 CPU verification: **183 passed, 11 GPU capture tests skipped**,
including all **31 sampler tests** and the C++ EXR fixture. The added compatibility
test passes the unified model's spatial output into the sampler and verifies
gradients into both networks. The count differs from the pre-v4 run because
v4 retired tests for the independent spatial model and added unified runtime
coverage. The same existing environment warnings remain; CUDA capture and
joint training are still unverified.

Step 1–2 verification now passes **214 tests, with no skips**, using an RTX 3090
outside the sandbox. This includes 13 v5 metadata/copy/resume tests, seven new
GPU layer/pipeline tests, the existing GPU capture suite and C++ EXR fixture.
The rebuilt C++ `frame_export_tests` also passes CTest. Actual source copies,
independent-range rendering, tiled/full/persistent parity, five-frame tensor
loading and CUDA layer replay were checked on two complete eight-frame fixture
clips. The production source will be `../dataset/denoiser-v4`; its frames are
not present here yet, so production v5 reuse/pilot validation is deferred until
they arrive. No bulk capture or adaptive model training has been started.

The subsequent reconstruction/gradient milestone passes **246 tests with no
skips** on the RTX 3090, including 32 new tests. It adds a separate adaptive
model wrapper with available-history reprojection, sparse pyramid/filter
fallbacks, and a first-order DASR count surrogate. A real 17x13, five-frame v5
fixture produced finite allocation gradients and an Adam update in both heads.
This controlled smoke used a zero-initialized sampler head for uniform 4-SPP
counts; it is not production training or an effectiveness measurement. See
[the operator specification and limits](adaptive-reconstruction.md).

The joint-training milestone now passes **278 tests with no skips**. It adds
bounded/unbounded allocation policies, aligned crops, the public `train-adaptive`
CLI, verified inventories and atomic two-network optimizer/RNG checkpoints.
Strict bilinear-gather recurrence makes CPU and RTX 3090 epoch resume exact.
A real two-scene pilot completed eight updates with nonzero gradients in both
networks, decreasing train/validation losses and bitwise-identical full/resumed
weights. This qualifies the pipeline; production data and separate sampler
effectiveness experiments remain pending. See [joint training](joint-training.md).
