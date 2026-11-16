# Validation cadence and grouped finite checks

The original trainer validates after each epoch, not each optimizer step.
Validation frequency is configurable independently of U-Net/CNN/RNN structure.
For example, Lightning supports `check_val_every_n_epoch`:
https://lightning.ai/docs/pytorch/stable/optional-api/api/lightning.pytorch.trainer.trainer.Trainer

Optional config fields (omission preserves historical behavior/config identity):

- `validation_every_epochs`: positive integer, default one.
- `validation_warmup_epochs`: nonnegative integer, default zero. Validate each
  epoch through this number, then use global epoch multiples of the interval.
- `finite_check_policy: grouped_v1`: check every gradient and weight while
  combining boolean results before host transfers. Omission uses legacy checks.

Initial and final validation always run, including on resumed invocations.
With interval five and warmup five, a twelve-epoch run evaluates epochs
1/2/3/4/5/10/12, plus initialization. `last.pt` is saved every epoch by default
(the new-run config now uses every five; see below). `model.pt`
is updated only when an actually evaluated loss beats the current invocation's
best (including its initial model). Skipped history rows have `validation: null`,
zero validation seconds and zero validation windows; no stale metrics are reused.
Reports show "not evaluated" and project validation cost using the actual
schedule for each target epoch count.

As before, best selection is scoped to this invocation; a resumed run has its
own output directory and starting-model evaluation. Preserve the previous
run's best checkpoint when comparing quality across resumed invocations.
Training/optimizer/RNG state resumes exactly, but an interrupted short run's
mandatory final/next-invocation initial validation adds observations absent
from an uninterrupted schedule. Consequently the best selected checkpoint need
not be the same across those invocations. Validation never updates the model.

These settings are pinned in checkpoint config; changing them is a new
experiment. Extending only total epochs keeps the global cadence. Validation
less often can miss an intermediate best model or delay detecting deterioration;
it is a throughput/monitoring tradeoff, not a quality-equivalence guarantee.

Grouped checks retain all per-tensor finite reductions, then transfer two
gradient-group flags together before the update, and one weight flag after
the update: two guard-related GPU-to-CPU synchronization points per step.
Failure still identifies sampler/denoiser gradients and blocks the optimizer;
nonfinite weights still fail immediately after an update. No gradients are
clipped, detached, skipped or replaced; norm and optimizer calculations remain.
Metric logging and other model checks still have their own synchronization.

Benchmark from clean source (outputs outside the checkout):

```bash
PYTHONPATH=src python tools/benchmark_joint.py \
  configs/train-ready-pilot-a100-throughput.json /ABSOLUTE/FRESH/OUTPUT \
  --sweep safety --warmup-epochs 2 --measured-epochs 10 \
  --max-memory-fraction .8 --discard-checkpoints
```

The three cases hold model, loader, data and precision fixed: legacy guards
with every-epoch validation; grouped guards with every-epoch validation; and
grouped guards with interval five/warmup five. Initial and final validation are
included in each case; warmup epochs are excluded from throughput calculations.
Validation windows/s measures only actual evaluation time, not skipped epochs.

## A100 results, 2026-11-16

Clean source `329916263804c96e4b33b924ef9c61b53281b47d`, direct crop loader,
batch 16/cache 16/depth 2, same seed, model, float32 and existing local data.
Each case used twelve epochs, first two excluded from throughput. All five
GPU cases produced exactly identical final model and optimizer/RNG state
hashes. Every epoch's training metrics, and validation metrics when evaluated,
also matched exactly. No held-out test pixels were opened.

First sweep:

| Checks/cadence | Train windows/s | Mean epoch seconds |
| --- | ---: | ---: |
| Legacy / every epoch | 37.10 | 4.147 |
| Grouped / every epoch | 36.80 | 4.504 |
| Grouped / first five, then every five | 27.84 | 4.059 |

A timing shift began at epoch eleven of the grouped dense case, before the
scheduled case began. Both training and validation slowed. The first sweep
therefore does not provide a reliable scheduling speed ratio. The finite-check
change showed no measurable training-phase speedup in this small pilot; no
standalone speed gain is claimed for it.

A reverse-order repeat with grouped checks in both cases:

| Validation cadence | Train windows/s | Mean epoch seconds |
| --- | ---: | ---: |
| First five, then every five | 27.261 | 3.987 |
| Every epoch | 27.265 | 6.965 |

Training throughput was virtually identical; the scheduled case used **42.75%
less epoch time** in this repeat. This is the observed short-pilot saving,
not a guaranteed full-dataset ratio. In twelve epochs, validation runs after
1/2/3/4/5/10/12 plus initialization. In 1,000 uninterrupted epochs the schedule
has 204 post-epoch evaluations instead of 1,000, plus initialization. Best
checkpoint selection is necessarily coarser. In the five-epoch warmup, validation
is still performed every epoch; the three-epoch smoke config skips none.

47 distinct relevant tests passed, including CUDA finite reductions with NaN
and both infinities at every position, empty gradient groups, injected bad
sampler/denoiser gradients blocking updates, post-update weight corruption,
last-checkpoint saves on skipped epochs, null metrics, final/global cadence,
CPU resume and report generation. Extended schedule/report coverage was rerun.
A separate real-data GPU check compared six uninterrupted epochs (interval 5,
warmup 0) against one epoch plus resume to six: weights, optimizer, all RNG,
execution flags, inventory and step count matched exactly. This explicitly
exercises resume between scheduled evaluations.

All cases stayed under the 80% host/GPU memory guard. Generated checkpoints
were discarded after recording hashes or completing exact-state comparisons;
there was no new data download or disk cache. The new-run A100 config enables
`validation_every_epochs: 5`, `validation_warmup_epochs: 5`, and
`finite_check_policy: grouped_v1`. Original config omissions still retain the
old behavior. Changing policy fields on an existing checkpoint is intentionally
rejected; use the original config to resume it.

Raw results under `/home/ubuntu/workspace/h100-ready-pilot/runs/`:

- `a100-validation-safety-20261116`
- `a100-safety-repeat-scheduled-20261116`
- `a100-safety-repeat-dense-20261116`
- `a100-scheduled-resume-20261116`

The repeated scheduled run includes a report with null-row handling and
schedule-aware full-inventory extrapolation. Compact evidence is in
`docs/a100-validation-safety-results-20261116.json`; the GPU resume diagnostic
is `/home/ubuntu/denoiser-handoff/gpu_scheduled_resume.py`. OS caching was
uncontrolled and observed throughput varied over time. Remeasure on the larger
dataset before committing the final presentation-time budget.

## Resume checkpoint interval

`checkpoint_every_epochs` is a positive integer, default one for historical
configs. The new-run A100 config selects **five**. Save `last.pt` at global
epoch multiples and always at the final epoch, including runs shorter than
the interval. Best-model saves remain independent and occur on any evaluated
improvement. The interval is pinned in checkpoint config; changing it requires
a new experiment. Histories report which resume/best files were actually saved.

Observed combined checkpoint cost was 0.10–0.14 seconds/epoch. Over 1,000
epochs that is only 100–140 seconds total. Treating all of it as resume-save
cost gives an optimistic upper-bound saving of 80–112 seconds at interval 5.
Actual savings are smaller because best-model saves are retained. Interval 100
would save only another roughly 19–27 seconds over interval 5, while exposing
up to 99 completed epochs to loss after interruption. At interval 5, up to four
completed epochs plus the in-progress epoch may need repeating. This is a
small I/O reduction, not a substantial full-training speedup.

Only focused checkpoint tests were run for this change, including invalid
intervals, global cadence, mandatory final save off cadence, unchanged best
model saves and exact final weights/RNG after resume. No full suite or new
GPU benchmark was run. Report projections use observed time per write and
the requested resume-save count, with observed best-save frequency as an
estimate; best improvements during future training cannot be predicted.
