# A100 batch measurements

The first optimization is optional validation batching. Set
`validation_batch_size` to a positive integer; omission keeps batch 1 and the
historical config representation. Metrics are weighted by actual window count,
including a partial final batch; replayed path/pixel totals remain sums.
References remain outside prediction. The option is pinned in checkpoint
configs: changing it requires a new experiment, including adding an explicit
value to a config that previously omitted it.

Use `tools/benchmark_joint.py` to compare training batches 4/8/16/32 or validation
batches 1/4/8 on existing local data. Run from a clean committed source tree on
the GPU host, using the setup in `git-training-workflow.md`:

```bash
python -c 'import torch; print(torch.ones(1, device="cuda"))'
python tools/benchmark_joint.py configs/train-ready-pilot-lambda.json \
  /home/ubuntu/workspace/h100-ready-pilot/runs/a100-train-batches \
  --sweep train
python tools/benchmark_joint.py configs/train-ready-pilot-lambda.json \
  /home/ubuntu/workspace/h100-ready-pilot/runs/a100-validation-batches \
  --sweep validation
```

Each command requires a new output directory outside the checkout. Nothing is
downloaded or moved. A case runs in a separate process, resetting model,
optimizer, loader RNG and seed. Widths, crop sizes, learning rate, gradient
policy, precision and worker count are unchanged. Validation uses fixed crops.
Training uses the existing seeded crop/augmentation schedule; the tool does
not implement a separate persisted crop schedule or change dataset loading.

Defaults are two warmup epochs followed by twenty measured epochs per case.
Use `--warmup-epochs`, `--measured-epochs` and `--batches` to select a longer or
shorter experiment. In this 32-window pilot, batch 32 has only one update per
epoch. Compare windows/s and memory; changing batch size changes optimizer
update count, so final losses do not rank learning quality fairly. Pixel rates
count all five frames per window, including initialization.

`manifest.json` records source commit/branch, exact base config, its input hash,
seed, case settings and cache caveats. Each case saves its resolved config,
verified dataset inventory, software/import path, execution flags, checkpoints,
history and timing summary. The benchmark also saves `pip-freeze.txt`,
`nvidia-smi.txt`, logs, `comparison.json` and `comparison.md`. Source changes
invalidate a run; differing inventory digests invalidate the comparison.
Failed cases retain their logs and a `failure.json` record; they are not skipped.

Throughput excludes warmup epochs and uses total windows divided by total
elapsed time. Training/validation phase timings synchronize CUDA in the existing
trainer. Startup and initial validation are separately available in each
summary. Memory peaks cover the entire case, including startup/warmup; process
RSS is isolated by process. Epoch timings include loading and host overhead,
so these are end-to-end phase rates, not isolated CUDA kernel measurements.

The OS page cache is uncontrolled and later cases can benefit from previous
reads. These are warm local-subset comparisons, not cold I/O or full-dataset
runtime measurements. CUDA event profiling, first-batch breakdown, persisted
crop schedules, loading changes, AMP and compilation remain future work.
Measured A100 results and the GPU parity limitation are recorded below.

## Measured on 2026-11-16

Source: `b3e2c1a`, clean `benchmark/a100` checkout. GPU: A100-SXM4-40GB;
driver 580.105.08; PyTorch 2.7.0, CUDA 12.8, cuDNN 90800. All cases used the
existing local 32-frame pilot, float32, seed 7, deterministic algorithms,
cuDNN deterministic enabled/benchmark disabled, and unchanged TF32 policies.
No data was added. Every case had the same verified inventory digest.

Training: two warmup and twenty measured epochs per case (640 measured
windows). Validation remained batch 1.

| Training batch | Windows/s | Gain over 4 | Measured updates | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: |
| 4 | 10.836 | baseline | 160 | 1.577 |
| 8 | 12.732 | 17.5% | 80 | 3.109 |
| 16 | 14.782 | 36.4% | 40 | 6.168 |
| 32 | 15.625 | 44.2% | 20 | 12.240 |

`configs/train-ready-pilot-a100-throughput.json` selects training batch 16,
which gives most of the measured throughput gain with half batch 32's memory.
The initial batch sweep changed only batch size. The config now also enables
the measured `prefetch_v1` loader; see [loading results](asynchronous-loading.md).
Validation stays at
the historical batch-1 default. This is a **new throughput experiment**, not
an exact resume or a convergence recommendation. On this 32-window subset,
batch 16 performs two updates/epoch versus eight at batch 4. Equal update
counts require four times as many epochs and four times as many windows;
do not interpret the throughput gain as faster convergence or scale the
learning rate automatically. Epochs remain three as a smoke-run default.

Training preparation remained about 1.54 seconds/epoch. It represented 52%
of training time at batch 4 and 71% at batch 16; profiling repeated metadata
verification, mmap opening and crop copying is the next performance task.
Training-phase gains do not imply the same gain in total epoch time because
batch-1 validation still costs roughly 7.5–8 seconds per epoch.

### Validation experiment: parity gate failed on CUDA

Training batch 4 was fixed. Validation batches 4 and 8 used two warmup and
eight measured epochs; batch 1 reuses the first eight measured epochs of the
training-batch-4 baseline, with identical initial state/training schedule.

| Validation batch | Windows/s | Status |
| --- | ---: | --- |
| 1 | 5.116 | retained in recommended config |
| 4 | 6.579 | experimental; strict CUDA parity failed |
| 8 | 6.645 | experimental; strict CUDA parity failed |

A separate frozen-checkpoint check used training-batch-4's epoch-22 `last.pt`
on all 40 fixed validation windows and on a 13-window subset (partial final
batches at sizes 4 and 8). It applied the CPU regression tolerance
`rtol=2e-5, atol=2e-7` to mean metrics and exact equality to integer totals.
Both larger batches failed this gate. For all windows, batch-4 loss changed
from 0.2394950702 to 0.2394837484, and replayed paths from 52,227,106 to
52,226,344. The 13-window batch-4 loss changed by -0.00003494. Pixel totals
were identical. These are small differences, but batching is not equivalent
under the existing acceptance criterion. Batch-dependent floating-point
kernels followed by discrete recurrent allocation are a plausible explanation,
not a verified root cause. Keep batch 1 until the difference is investigated
and a suitable numerical contract is explicitly established. The tolerance
was not relaxed to make the experiment pass.

### Validation and retained artifacts

117 relevant CPU regressions passed, including batching/partial batches,
exact resume, radiance policy, estimator, allocation, reconstruction,
streaming and preview checks. Two NetworkX/NumPy deprecation warnings appeared
in each test invocation. All six GPU benchmark cases completed with the
trainer's finite-gradient and finite-weight checks enabled. The additional
GPU parity check exited 1 as described above.

Raw configs, inventory, provenance, software, histories, logs, checkpoints and
comparisons remain outside Git under:

- `/home/ubuntu/workspace/h100-ready-pilot/runs/a100-train-batches-20261116`
- `/home/ubuntu/workspace/h100-ready-pilot/runs/a100-validation-batches-20261116`

The second directory also contains `gpu-parity.json`; its diagnostic script is
`/home/ubuntu/denoiser-handoff/gpu_validation_parity.py`. Compact machine-readable
results are retained in `docs/a100-batch-results-20261116.json`.

The OS page cache was uncontrolled; these measurements describe this warm
local subset. They establish neither full-dataset I/O throughput nor a quality
advantage for a larger training batch. Reserved test pixels were not loaded.
