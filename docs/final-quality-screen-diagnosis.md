# Final screen diagnosis, 2026-11-17

All six arms completed four epochs and 192 updates each. Native evaluation covered
all eight frames of all 12 development clips for the warm checkpoint, each best
checkpoint, and both sparse arms' final checkpoints. Both recorded crop-stream
matching checks passed. The decision remains **no candidate**. This is a completed
negative experiment, not an execution failure.

Percentages below are 100 * (candidate/control - 1); positive means worse.
Crop controls use fixed 256-pixel validation crops. Native controls use independent
full-frame recurrent histories with the same frozen denoiser and exactly equal
4-SPP path budgets. The warm comparison is against the shared initialization.

| Arm | Crop vs uniform | Native vs uniform | Native vs warm | Maximum training zero fraction |
|---|---:|---:|---:|---:|
| fixed-128 | -0.520% | +0.038% | -1.922% | 0.00000 |
| fixed-256 | -0.533% | +0.042% | -0.287% | 0.00000 |
| joint-128 | -0.093% | +0.121% | -1.884% | 0.00000 |
| staged-128 | -0.082% | +0.090% | -2.216% | 0.00000 |
| zero-curriculum-128 | +0.042% | +4.858% | +4.049% | 0.04928 |
| zero-relaxation-128 | -0.194% | +0.195% | -2.322% | 0.00000 |

The fixed sampler shows useful placement on crops but loses its small advantage
at native resolution. This is evidence of a crop/native discrepancy; it does not
establish its cause. Increasing training crops to 256 did not improve the result
in this four-epoch comparison. Larger crops process four times as many pixels.

Staged and relaxed arms reduce aggregate native loss versus warm initialization,
but none beats its same-denoiser uniform control or satisfies the per-clip gates.
The denoiser improving does not establish learned-allocation utility.

Forced holes worsen native loss by 4.858% versus uniform. Relaxation observed zero
training holes despite its 5% eligibility quota, so it did not establish sparse
behavior. The backward probe still measures zero gradients through exact-black
encoding at counts 0, 1, and 4 despite finite renderer-surrogate gradients.
Changing a learning rate or reserve floor does not repair that path.

All arms reached the epoch target without triggering the gradient stop. Monitored
peak whole-device usage was 34.66%; whole-host pressure was 3.69%, below the 80%
limit. Polling does not bound between-sample peaks. Test pixels remained reserved.

## Changes and next pilot discussion

Preserve the existing selection thresholds and no-candidate decision. The report
now spells out the delta formula and sign. Diagnostic failure-path tests now
use repository configs and avoid requiring host training data for CUDA refusal.
No training-policy correction is
justified by these data alone; gradient stability is separate from allocation
quality. Source, pilot configurations, host launch wrappers and small numerical
evidence are included in this branch. Arrays, checkpoints and credentials stay
outside Git. Original run artifacts remain unchanged.

For the next development pilot, prefer 128-pixel training crops and dense coverage.
Prioritize a uniform-allocation denoiser control and a matched fixed-sampler arm;
keep any joint arm secondary until it demonstrates native allocation benefit.
Continue checking per-clip temporal loss and native controls. A longer run may
improve denoising, but this screen gives no basis to promise adaptive improvement.
Agree the wall-time budget and whether to reuse the existing 24/12 clip split
before preparing the pilot. No new training was launched during this diagnosis.

The configurations and shell wrappers record the original GPU host layout:
`/home/ubuntu/quality_workspace`, with source under `Path-Tracer-Denoiser`,
configuration files in sibling `configs/`, and reused arrays under `runs/`.
A fresh Git clone alone does not provide the data, environment, warm checkpoint
or readiness inventory. Copy the versioned quality JSON configurations to that
sibling directory when reconstructing this layout; inspect each plan before use.

Machine-readable arm metrics, per-clip aggregate metrics, checkpoint hashes,
source hashes, rejection reasons and backward-probe values are retained in
[the evidence JSON](final-quality-screen-20261117.json). Full per-frame evidence,
PNGs and GIFs remain in the source run directory recorded there.
