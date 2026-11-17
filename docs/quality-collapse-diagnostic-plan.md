# Sampler-collapse diagnostic plan

The diagnostic is implemented in `Path-Tracer-Denoiser/tools/diagnose_quality_collapse.py`.
Its settings are in [quality-collapse-diagnostic.json](quality-collapse-diagnostic.json).
CUDA is unavailable in the authoring session; no additional GPU training has been
launched. Execute the command below on the original GPU host.

## Execute

```bash
bash /home/ubuntu/quality_workspace/run_quality_diagnostic.sh
```

This command reuses the verified local data and creates
`/home/ubuntu/quality_workspace/runs/quality-collapse-diagnostic/`. It refuses to
overwrite an existing output. To repeat the experiment, add
`--output /home/ubuntu/quality_workspace/runs/quality-collapse-diagnostic-repeat`.
No R2 credentials, downloads, captures, backfills, or installations are required.

Inspect the effective plan without CUDA, hashing the full dataset, or writing
any outputs:

```bash
bash /home/ubuntu/quality_workspace/run_quality_diagnostic.sh --plan-only
```

Progress appears in the terminal and `runs/quality-collapse-diagnostic/run.log`.
For a terminal that may disconnect, use:

```bash
nohup bash /home/ubuntu/quality_workspace/run_quality_diagnostic.sh > /home/ubuntu/quality_workspace/runs/quality-collapse-launch.log 2>&1 &
```

The result is `runs/quality-collapse-diagnostic/report.md`, with machine-readable
`summary.json`, `status.json`, and `resources.jsonl`. Each arm has independent
outputs. The launcher writes a partial report on a worker failure, memory stop,
deadline, or Ctrl-C. If the launcher itself is killed abruptly, rebuild the
report without CUDA using `--report-only` with the same `--output`.

## Existing evidence and reusable artifacts

Resource logs record approximately 27m06s preparation, 1h52m47s training invocation,
and 3m47s previews: about 2h23m40s across those stages. The training timer stopped
normally at an epoch boundary after 53 epochs / 2,544 updates. Best validation
was epoch 5. Sampler mean gradient norms jumped from about 0.6 at epoch 8 to
87,499 at epoch 9 and 147,436 at epoch 10, alongside rapidly increasing zero
allocations. This correlation does not establish the initiating cause.

Reuse `runs/bmw-living-quality-2h/data` read-only, without new downloads, scene
selection, copies, or changes to the split overlay. Preserve every existing
checkpoint and report. New outputs belong in a new directory under `runs/`.
Any changed training behavior must have its own configuration, source patch,
and experiment identity; it is not an exact continuation of the original run.

## Budget and controls

The parent enforces a **30-minute overall worker deadline**, including full data
hashing, initialization, all arms, evaluation, crop exports, and the transfer
probe. Lightweight preflight/source recording precedes this timer; termination
grace and report writing can add several seconds. Each arm gets at most eight
minutes, up to 12 new epochs. At least one minute of the remaining overall budget
is reserved before starting another arm. The unused portion of 30 minus 3×8
minutes covers setup and the probe. Late arms are explicitly skipped if setup
uses the available time. The deadline is enforced even if hashing or evaluation
stalls. A hard stop can interrupt the current update/checkpoint; partial reports
and prior completed checkpoint files remain available.

Within an arm, measured initial/scheduled validation time reserves room for final
validation and a uniform control. The loop can stop mid-epoch; `completed_epochs`
counts only full epochs, while `global_step` includes all completed updates.
Evaluations stop between batches at their deadline. Their `windows`,
`total_windows`, and `complete` fields distinguish complete and partial results;
the report excludes incomplete losses from quality comparisons. Compare matched
update counts as well as elapsed time.

Use the original full selection and crop/batch settings initially. Early baseline
epochs were fast enough to expose the failure, and retaining the data and random
plans avoids confounding a stability change with a smaller dataset.

1. Baseline: fresh original initialization/seed, original joint optimization.
   Reproduce the allocation transition with per-update instrumentation. If it
   does not reproduce, check source/config/sample-order differences before fixes.
2. Denoiser control: identical denoiser initialization and data plans, uniform
   4-SPP replay, train only the denoiser. Determine whether loss improves without
   a changing allocation. This is an independent experiment, not an exact resume.
3. Sampler control: freeze the epoch-5 denoiser and train the sampler starting
   from epoch-5 weights with a fresh, explicitly recorded optimizer. Keep gradients
   through the frozen denoiser to the sampler. Determine whether collapse persists
   without denoiser parameter updates. This arm has a different starting state;
   do not interpret it as a single-variable comparison against arm 1.

If baseline is already reproducible before consuming its full budget, prioritize
tracing the first abnormal update over accumulating more collapsed epochs.

Both fresh arms must match the original initialization weights exactly before
training. Baseline epoch loss, zero fraction, and both gradient-norm differences
against the original history are written to `baseline/summary.json`. Full data
hashes must reproduce the original inventory, including the experiment split.
The shared dataset read lock spans all arms. Source hashes, a small source copy,
Git patch, runtime versions, and source-checkpoint hashes are recorded in the
new output. Do not edit source/config files while execution is running.

## Instrumentation and follow-up

`ARM/steps.jsonl` records every update's loss components, both network gradient
norms, actual SPP, and zero/capped/unavailable fractions. Every 24 updates and at
the first anomaly it also records sampler logits, sampled allocation quantiles,
per-channel feature ranges, and recurrent-frame gradients. Gradient attribution
separates exact-black, positive near-black (≤1e-6), and bright (≥1) radiance.
Quantiles use at most 8192 deterministic strided values; no diagnostic RNG is
consumed. Detailed statistics add overhead, so diagnostic elapsed times are not
throughput benchmarks.

The first update with sampler gradient norm ≥100, zero fraction ≥0.5, or
unavailable fraction ≥0.5 captures `anomaly.json`, `anomaly-before.pt`,
`anomaly-after.pt`, and `anomaly-batch.pt`. Thresholds are configurable evidence
triggers, not an automatic diagnosis or stop. The snapshot includes the current
CPU batch, exact allocations, count-surrogate gradients, model/optimizer state,
and RNG state. At most one such batch is retained per arm. If baseline captures
an anomaly and the overall budget allows, `transfer-probe.json` measures up to
four one-sample transfers on its first window/final frame. Each transfer preserves
the exact integer path total, freezing the pre-update denoiser. This probes a
single fixed sample bank; discrete replay differences are not a smooth derivative
or an unbiased estimate of expected gradient quality. Compare gradient signs,
not magnitudes, because training used a full-batch mean.

Initial, scheduled, final, and uniform-control validation export small fixed
crops from one clip per validation scene under `ARM/previews/`. Their four
columns are uniform noisy, allocated noisy, denoised, and reference. The count
NPY and per-crop metrics accompany each image. Allocation/recurrence are local
to the validation crop; these are **not** native full-image previews. Images may
be absent if the evaluation deadline is reached before that scene.

`initial.pt`, `best.pt`, and `last.pt` use a separate diagnostic checkpoint format
containing the arm policy. Generic joint inference/resume rejects them, preventing
a uniform-control checkpoint from being mistaken for a learned-sampler model.
Mid-epoch snapshots support update replay; this workflow does not implement
exact interrupted-run resume. Re-run to a new output directory instead.

Only after identifying the trigger, test one correction at a time. Candidate
ablations include a lower sampler learning rate or gradient clipping, with
explicitly recorded thresholds. Neither repairs the documented zero-count
hard-mask gradient blind spot by itself. Any change to allocation, reconstruction,
or zero-count gradients needs separate mathematical review and experiment identity.

Optional implemented arms are `warm-joint` (epoch-5 weights, fresh Adam, both
networks trained), `sampler-lr-low` (fresh weights, sampler LR multiplied by 0.1),
and `sampler-clip` (fresh weights, sampler global gradient norm clipped to 1).
They are not enabled by default. To use one, copy the small plan JSON to a sibling
file under `configs/`, replace its `arms` list, and pass `--plan` and a new
`--output`. Keep the same relative `source_run` and `config` paths. A matched
`["warm-joint", "frozen"]` run isolates denoiser freezing at epoch 5 more cleanly
than comparing the warm frozen arm with the fresh baseline. A repair comparison
should pair `baseline` with just one repair arm, retaining the 30-minute total
budget and identical data/initialization. Logged gradient norms are measured
before any optional clipping.

Use the epoch-5 checkpoint for a later repair branch if justified. Avoid starting
the primary repair from the already collapsed epoch-53 sampler. Preserve a fresh
initialization control to distinguish prevention from recovery.

Keep the existing 75% GPU allocator cap and resource guard stopping at 79% host
or whole-device memory, below the user's 80% limit. Use `activate.sh` and verify
the active clone's imports. Require CUDA; do not silently switch to CPU training.
Generate only small diagnostic crops initially. Full-resolution previews and a
longer confirmation run follow only after stable allocation and improved losses.

The parent samples whole-host pressure (`MemTotal - MemAvailable`) and whole-device
GPU-0 usage every 0.5 seconds. A failed resource query also terminates the worker.
Polling cannot prevent arbitrary sub-poll spikes or allocations by unrelated
processes. This launcher targets the original single A100 / CUDA device 0 host.
At least 2 GB free disk is required for small logs, model snapshots, and crop
evidence; the 88 GB decoded payload is shared in place.

Success means preventing collapse while improving denoising and allocation
utility on the existing development validation data. Nonzero finite gradients,
more epochs, or visually interesting count maps are insufficient. No paper-quality
or untouched-test generalization claim follows from this diagnostic.

## Readiness validation

On 2026-11-16, 49 CPU tests passed across the diagnostic, existing joint trainer,
and existing quality-pilot checks. The diagnostic checks establish exact baseline
forward/gradient/Adam parity, identical full-epoch CPU results, uniform replay
parity, frozen-component behavior, exact before/after anomaly-update replay,
fixed-budget transfer bookkeeping, partial evaluation labels, and parent process
termination/report preservation at memory and time thresholds. Two existing
NetworkX/NumPy deprecation warnings were emitted.

The real source initialization (epoch 0) and best-trained checkpoint (epoch 5)
loaded successfully and matched the original config and inventory. A metadata-only
inventory audit matched the completed pilot; full selected-array checksum hashing
is intentionally performed by the final execution command. Plan-only, Python
compilation, shell syntax, and Git whitespace checks passed. CUDA is inaccessible
in this authoring session; GPU numerical parity, runtime, and memory peaks remain
to be measured. No new GPU experiment has been started.

Machine-readable readiness evidence is saved at
`runs/quality-collapse-readiness.json` outside the source checkout.
