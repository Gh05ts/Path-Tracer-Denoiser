# Final development screen

Run from the GPU host:

```bash
bash /home/ubuntu/quality_workspace/configs/run-final-quality-screen.sh
```

Default: a **60-minute worker/monitor ceiling**, at most **four epochs per arm**,
and a new output directory at `runs/final-quality-screen-20261117`. Plan
inspection loads the small warm checkpoint and metadata, without training or
reading pixel arrays:

```bash
bash /home/ubuntu/quality_workspace/configs/run-final-quality-screen.sh --plan-only
```

Adjust the screen without editing files, for example:

```bash
bash /home/ubuntu/quality_workspace/configs/run-final-quality-screen.sh \
  --minutes 90 --epochs 6 \
  --output /home/ubuntu/quality_workspace/runs/final-quality-screen-90m
```

The script refuses to overwrite an existing output directory. Launch/preflight,
small source snapshot copying and final report overhead are outside the worker
ceiling. The watchdog can take up to its polling/monitor timeout plus five
seconds to terminate a worker. Incomplete coverage is explicitly reported and
returns a nonzero exit code. A completed negative experiment can return zero
while `decision.json` selects no candidate.

## Comparisons

Every arm starts from the latest readiness winner, the fixed-sampler epoch-5
checkpoint, using fresh Adam, seed 7, the same 24 BMW/Living Room training clips
and 12 Dragon/Bathroom development clips. Both networks remain LARGE/width48.
Batch size remains eight. No new data is downloaded; reserved test pixels remain
untouched. This screen is not the broader full-data training run.

| Arm | Crop | Training |
|---|---:|---|
| fixed-128 | 128 | Fixed useful sampler, denoiser LR 1e-4 |
| fixed-256 | 256 | Same training, larger context |
| joint-128 | 128 | Joint from the beginning; sampler LR 1e-5, denoiser 1e-4 |
| staged-128 | 128 | First third with sampler frozen, then joint |
| zero-curriculum-128 | 128 | Staged training, then protected forced holes at 1% and 5% |
| zero-relaxation-128 | 128 | Staged training, then reserve floor 1 → 0.5 → 0, with at most 1% then 5% eligible for zeros |

The forced-hole arm is a separate stress control: a mild pretrained sampler
may never produce zero counts even when the reserve floor is relaxed. The
relaxation arm allows zeros without forcing them. Reports distinguish policy
quota from observed zero fraction. Zero eligibility is restricted to valid
geometry with valid warped history, and is selected by lowest logits with
deterministic ties. The selector remains discrete and nondifferentiable. The
existing positive-count DASR surrogate, zero-count unit denominator, hard masks,
and exact-black zero encoding derivative remain explicit policies. This does
not claim to repair the sparse backward path; `backward-probe.json` exposes it.

All variants preserve **exactly 4*H*W paths per frame**, maximum 63, with
minimum one outside the permitted/forced hole set. Stable largest-remainder
rounding repairs integer budgets. Every arm keeps the 1e-3 backward-only
radiance floor. Full five-frame history gradients remain connected.

The crop comparison shares window order, random augmentations and crop centers:
128 crops are nested within the 256 crops. It matches batch size and maximum
optimizer updates, not pixel compute; 256 processes four times as many pixels.
Recorded hashes verify matching window/augmentation streams and same-size
crop streams. The 256 arm receives twice the wall-time share. Half the overall
budget is reserved for native evaluation; unused early-arm time is redistributed.

## Selection and evidence

Each epoch evaluates all 240 fixed validation windows. Best selection includes
initialization. Non-improvement halves learning rates; three consecutive
non-improvements stop an arm. Each phase transition restores its best weights
and Adam state, preserving the ongoing matched crop stream. Gradient norms above
100, missing outputs, or saturation above 1% stop the arm before the suspect
optimizer update. Failed/partial arms cannot qualify. This is a short screen;
four epochs cannot prove long-run convergence.

Native evaluation checks the warm checkpoint, every arm's best checkpoint,
and both sparse arms' final checkpoints on **all eight frames of all 12
development clips**, with global allocation and independent learned/uniform
histories. Every learned frame receives a spatial score. All temporal pairs are
recorded; the main temporal mean excludes the initialization pair 0→1, comparing
two learned outputs. Frame-zero diagnostics remain available separately.

Native selection requires at least 0.1% mean allocation improvement over the
same-denoiser uniform control, no per-clip spatial/temporal/total regression over
0.1%, no per-clip regression over 0.5% against warm-start, exact budgets, no
unavailable outputs and no excess unexpectedly black pixels above 0.1 percentage
points. Dense candidates also must beat shuffled placement in the crop audit.
Sparse shuffling is omitted because it would violate protected coverage. These
are engineering thresholds, not statistical significance or paper-equivalent
quality. If nobody passes, the decision selects no candidate.

Outputs include `report.md`, `decision.json`, `status.json`, `progress.json`,
`run.log`, source/input hashes, per-step gradients and crop hashes, epoch
histories, best/last checkpoints, native per-frame metrics and fixed-exposure
images/GIFs for one clip per validation scene. Other clips retain all metrics.
Checkpoints retain optimizer/RNG state for audit; this short screen does not
offer automatic resume. Historical outputs and data are preserved. Only small
Python source files are copied for provenance; data arrays are reused in place.

The allocator is capped at 75% or lower after accounting for existing device
usage and a 512-MiB margin. Whole-host and whole-device monitoring terminates at
79%, below the user's 80% limit. Loss of memory monitoring also terminates the
worker. No automatic full-training launch, publishing, or cleanup follows.
