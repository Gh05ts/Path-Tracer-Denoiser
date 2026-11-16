# Larger batches, decoded map caches and prefetch queues

`decoded_cache_frames` sets the LRU frame-map capacity **per dataset** (train
and validation have separate caches); omission preserves sixteen. It requires
`loader_policy`. `prefetch_batches` sets CPU batches prepared ahead of the
consumer; omission preserves two. It requires `prefetch_v1` or `prefetch_fused_v1`.
Both settings are positive integers and pinned in checkpoint config for exact
resume. Changing either means a new experiment. The original defaults remain.

Larger map capacities reuse existing read-only `.npy` files; no arrays are
copied to disk. Only frames actually accessed are mapped. A capacity of 64
does not populate 64 entries when a partition contains only sixteen frames.
Queued crops consume host RAM proportional to batch size, crop size and depth.

The size experiments use the existing pilot frames with sixteen crops/window
instead of four: 128 training windows per epoch. This makes full batches up to
64 possible, while keeping the selection, crop size and model fixed across
cases. It is a throughput experiment with more crops, not additional source
data. Changing batch size changes optimizer updates; loss does not rank learning
quality across batches. Final adoption must also consider the original short
pilot schedule and longer-epoch throughput.

Use a fresh directory outside the checkout for each command:

```bash
PYTHONPATH=src python tools/benchmark_joint.py /ABSOLUTE/size-config.json \
  /ABSOLUTE/NEW/batches --sweep train --batches 16 32 48 64 \
  --warmup-epochs 2 --measured-epochs 4 \
  --max-memory-fraction 0.8 --discard-checkpoints

PYTHONPATH=src python tools/benchmark_joint.py /ABSOLUTE/size-config.json \
  /ABSOLUTE/NEW/caches --sweep cache --cache-frames 8 16 32 64 \
  --prefetch-depths 2 --warmup-epochs 2 --measured-epochs 4 \
  --max-memory-fraction 0.8 --discard-checkpoints
```

The memory guard uses `MemTotal - MemAvailable` for whole-host pressure and
`nvidia-smi` whole-device usage at about 0.5-second intervals. It terminates a
case on a threshold violation and records the reason. The worker additionally
caps PyTorch's allocator below 80% of device total after subtracting initial
context/external use and a further 512 MiB margin. An allocator-cap OOM fails
the case; larger cases should not be attempted after a resource failure. This
margin covers expected nonallocator overhead; unrelated processes must not
start consuming GPU memory during the sweep. Polling is not an absolute
instantaneous guarantee for allocations outside PyTorch. PyTorch peak allocated
and reserved memory remain recorded in each completed case.

`--discard-checkpoints` removes only `model.pt` and `last.pt` generated inside
each successful fresh case. It retains exact model and optimizer/RNG tree
hashes, configs, inventory, software, metrics, logs and resource samples. A
failed case retains its artifacts. At most one case's checkpoints need persist
during the sweep; there are no additional dataset downloads or disk caches.

## Measured results, 2026-11-16

Source `7e8ef71`, clean `benchmark/a100`, same A100-SXM4-40GB and existing
decoded pilot. Two warmup and four measured epochs per case, separate worker
processes with identical initialization, seed, inventory and float32 execution
policies. Thirteen cases completed. Validation stayed batch 1 throughout.

### Batch sweep: 128 training windows/epoch

| Batch | Train windows/s | Mean epoch seconds | Peak whole-device GPU use |
| --- | ---: | ---: | ---: |
| 16 | 35.08 | 7.359 | 19.31% |
| 32 | 36.71 | 7.228 | 37.52% |
| 48 | 28.99 | 8.567 | 55.72% |
| 64 | 26.66 | 15.022 | 73.71% |

Batch 32 gained 4.65% training throughput over 16 in this sweep, but repeated
baseline timings varied more than that: the same batch-32/cache-16/depth-2
setting later measured 30.94 windows/s. This small apparent gain is not robust
enough to select a new default. Larger batches 48/64 were slower despite fitting
the memory limit. Batch 48 includes a partial 32-window final batch. Batch 64
also increased validation time substantially; the measurements do not establish
the cause, and no allocator/kernel root cause is claimed. There was no reason
to spend the remaining memory headroom on still larger batches.

### Cache capacity: batch 32, depth 2, 128 windows/epoch

| Cached frames per partition | Train windows/s | Mean epoch seconds |
| --- | ---: | ---: |
| 8 | 20.46 | 10.288 |
| 16 | 30.94 | 7.707 |
| 32 | 30.06 | 7.788 |
| 64 | 30.77 | 7.733 |

Eight frames caused heavy reopening/revalidation. Sixteen frames already cover
each selected partition's complete working set. Capacities 32/64 therefore
retain no extra maps on this dataset and deliver no measurable benefit. This
does not establish the optimal capacity for a future larger inventory.

### Prefetch depth: batch 16, cache 16, 128 windows/epoch

| CPU batches ahead | Train windows/s | Mean epoch seconds |
| --- | ---: | ---: |
| 2 | 32.89 | 7.545 |
| 4 | 30.56 | 8.065 |
| 8 | 32.06 | 7.593 |

Longer queues did not improve end-to-end throughput. Keep depth 2. There are
eight batches in these epochs, so depth 8 can plan the whole remaining epoch;
the configured queue capacity need not fill if the consumer keeps up.

### Original-pilot confirmation: 32 windows/epoch

| Batch | Train windows/s | Mean epoch seconds |
| --- | ---: | ---: |
| 16 | 27.99 | 4.719 |
| 32 | 22.11 | 5.070 |

Batch 32 reduces this tiny pilot to one training batch, losing preparation
overlap between training batches. It was slower both in training and overall.
**Keep the existing throughput config: batch 16, cache 16, prefetch depth 2.**
Its config file is unchanged; the new size controls are available for future
experiments. Resweep once more frames are available and epochs are longer.

### Memory, disk and correctness

- No memory-guard violations or allocator-cap OOMs. Highest sampled whole-device
  GPU usage: **73.71%**, under the 80% cap. Peak PyTorch reserved memory was also
  under 80% in every case.
- Highest process RSS: **10.90 GiB** on a host with roughly 216 GiB RAM
  (about 5.0%). Whole-host pressure, measured as `1 - MemAvailable/MemTotal`,
  peaked at **3.88%**. RSS includes reclaimable mapped file pages.
- Retained new run artifacts: **1,359,983 bytes (~1.30 MiB)**. All thirteen
  cases' generated checkpoints were removed after recording their hashes. No
  arrays were downloaded, duplicated or persistently cached on disk.
- All six epochs' train/validation metrics and final model plus optimizer/RNG
  hashes matched exactly within each cache/queue sweep. Every case used the
  same inventory digest. Changing batch size intentionally changes training
  updates and weights; their losses do not rank convergence.
- The default batch-16 pilot's epoch metrics also match the earlier prefetch
  implementation's GPU history exactly. Seventy-five relevant CPU tests passed,
  including expanded-depth sample/RNG parity, actual overlap and queue bounds,
  resume at nondefault sizes, configuration pinning, guard termination, checkpoint
  cleanup, map eviction and mutation detection. Existing deprecation warnings
  remain.

Artifacts under `/home/ubuntu/workspace/h100-ready-pilot/runs/`:

- `a100-size-batches-20261116`
- `a100-size-caches-20261116`
- `a100-size-queues-20261116`
- `a100-size-pilot-confirm-20261116`

Each includes configs, manifests, software, histories, summaries, sampled
resources and comparison reports. `docs/a100-size-results-20261116.json` keeps
compact results, resource peaks and parity assertions in Git. OS page caching
was uncontrolled; these are warm local-subset measurements, not full-dataset
I/O throughput, native-resolution inference, or convergence results.
