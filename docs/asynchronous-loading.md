# Local joint-loader optimization

Optional `loader_policy` settings preserve the historical default when omitted:

- `cached_v1`: synchronous preparation with a per-dataset LRU defaulting to sixteen
  read-only decoded frame map pairs and verified metadata.
- `prefetch_v1`: the same cache plus one CPU preparation thread and at most
  two batches ahead of the consumer by default. GPU transfers remain on the main thread.

`decoded_cache_frames` and `prefetch_batches` make these sizes configurable.
See [larger-size measurements](a100-size-sweeps.md) for memory-bounded sweeps;
the selected pilot settings remain 16 frames and two batches ahead.

These settings are pinned in checkpoint configuration; changing or adding one
requires a new experiment. R2 streaming currently rejects them. `num_workers`
stays zero. Local EXR data can use preparation prefetch, but map reuse applies
only to decoded arrays. Full SHA verification still runs at initial inventory
time; no reserved test pixels are opened.

Every cached access stats all three provenance locks, the base/layer/decoded
manifests, and both payload arrays. Device, inode, size, modification time and
change time must match. Changes fail the run; new metadata is never silently
adopted. Small file-identity stamps remain for previously accessed frames after
map eviction, so an evicted map cannot hide a same-size payload modification.
The stamps grow with the selected frames; open map pairs remain bounded at the
configured capacity (16 by default) per dataset. Maps and metadata are
process-local and never checkpointed.
This relies on the existing shared dataset read lock and ordinary filesystem
identity/change semantics, not cryptographic hashing of every crop.

Crop and augmentation plans are chosen on the main thread using the existing
CPU RNG and DataLoader's seeded order. The background thread reads/copies and
applies those plans without random draws. The current model consumes no CPU
randomness during forward/backward; adding such randomness in a future model
requires revisiting this versioned contract. Training tensors remain in FIFO
order, including a partial final batch. Both CPU batch contents and complete
training results are checked against synchronous loading.

The executor is joined on normal completion, early exit and error. No future
or queue survives an epoch boundary, validation phase, or checkpoint. Worker
errors propagate to the caller. At most the current batch plus the configured
lookahead (two by default) are retained by the iterator, in addition to
model/device tensors. The queue is
bounded by batch count; memory therefore scales with configured batch/crop size.
There is no mid-epoch resume guarantee.

The initial real-data profile prepared 32 training crop windows in 1.981 seconds
under cProfile: `verify_decoded_frame` accounted for 0.629 seconds, with 640
`np.load` calls (verification reopened headers/maps, then crop loading reopened
them). Copies and stacking also contributed materially. Caching and prefetching
are measured separately; asynchronous loading does not eliminate preparation.

Reproduce the end-to-end comparison from a clean committed checkout:

```bash
PYTHONPATH=src python tools/benchmark_joint.py \
  configs/train-ready-pilot-a100-throughput.json \
  /ABSOLUTE/NEW/OUTPUT --sweep loader --warmup-epochs 2 --measured-epochs 8
```

The tool fixes training batch 16 and validation batch 1 from that config, resets
model/optimizer/RNG in a separate process per case, and records source,
inventory, execution policies, phase timings, RSS and allocator peaks. These
are warm local-subset measurements with uncontrolled OS cache. `training_data_seconds`
measures consumer wait including plan/refill overhead, not total background CPU
work. Loader lookahead is not a disk cache or R2 download prefetcher.

## A100 measurements, 2026-11-16

Source `57e5b6059a2b808abaab6eb9ae680b9558bffff8`, clean `benchmark/a100`.
A100-SXM4-40GB, PyTorch 2.7.0 / CUDA 12.8 / cuDNN 90800. Each case had two
warmup and eight measured epochs, 256 measured training windows and sixteen
optimizer updates. Batch 16, widths 48, five-frame 128x128 training windows,
batch-1 256x256 validation, precision, gradient policy, LR and inventory were
fixed. No data was downloaded or added.

| Loader | Train windows/s | Validation windows/s | Mean epoch seconds | Peak process RSS GiB |
| --- | ---: | ---: | ---: | ---: |
| Synchronous | 13.698 | 5.046 | 10.363 | 2.554 |
| Maps/metadata reused | 20.034 | 6.111 | 8.244 | 8.343 |
| Reuse + CPU prefetch | 25.358 | 11.275 | 4.910 | 8.583 |

Mean training phase: 2.336 -> 1.597 -> 1.262 seconds. Mean validation phase:
7.926 -> 6.546 -> 3.548 seconds. Prefetch plus reuse gave **2.11x measured
epoch throughput** (52.6% less epoch time) versus the fresh synchronous
baseline. Training consumer data wait fell from 1.702 to 0.641 seconds/epoch;
background CPU work still exists. Peak GPU allocation remained about 6.17 GiB
in every case. Retained mappings increase resident host pages substantially;
RSS includes those file-backed pages and is not just the prefetch queue.

An isolated warm transfer measurement moved 193,986,560 bytes for one training
batch in a median 11.57 ms over ten trials (blocking pageable H2D, including
allocation and host overhead). Two such batches cost about 23 ms, under 2% of
the measured prefetched training phase. GPU transfer streams/pinning remain
deferred: the measured benefit here comes from preparation reuse and overlap.

All ten epochs' training and validation metrics, final model weights and
CPU/loader RNG state matched exactly across all three GPU cases. A separate
real-data GPU run compared uninterrupted three-epoch training with one epoch
plus a resumed two epochs: weights, optimizer, **all** RNG states, execution
flags, inventory and global step matched exactly. Validation batch remains 1;
the earlier validation-batching parity failure does not apply to this change.

153 relevant CPU regressions passed, including deterministic order and samples,
partial batches, exact training/resume, bounded lookahead with actual overlap,
worker failure/early-exit cleanup, same-size payload mutation after map eviction,
and changes to every metadata dependency. Existing deprecation warnings remain.

`configs/train-ready-pilot-a100-throughput.json` now selects `prefetch_fused_v1`
for new experiments; see [direct construction results](direct-crop-loading.md).
The initial prefetch measurements above used `prefetch_v1`.
The original Lambda pilot config is unchanged. Old checkpoints
must continue with their original loader policy; adding the new policy is a
new experiment rather than exact resume.

Artifacts outside Git:

- `/home/ubuntu/workspace/h100-ready-pilot/runs/a100-loaders-20261116`: manifest,
  configs, inventory, histories, software, logs, checkpoints, comparison,
  `cross-policy-parity.json`, and `h2d-profile.json`.
- `/home/ubuntu/workspace/h100-ready-pilot/runs/a100-loader-resume-20261116`:
  uninterrupted/first/resumed runs and `parity.json`. Diagnostic script:
  `/home/ubuntu/denoiser-handoff/gpu_loader_resume.py`.

Compact results are retained in `docs/a100-loader-results-20261116.json`.
These measurements use the warm pilot subset and uncontrolled OS cache. They
are not full-dataset I/O or convergence results, and do not establish throughput
when reading a working set larger than RAM.
