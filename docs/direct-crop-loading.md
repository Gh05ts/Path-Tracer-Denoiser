# Direct decoded crop construction

`loader_policy: prefetch_fused_v1` keeps the existing map cache and FIFO CPU
prefetch behavior, but copies decoded views directly into preallocated final
window tensors. Flip/rotation operations use NumPy views before the copy.
The decoded path avoids intermediate HWC crop buffers, per-frame CHW tensor
copies, time/layer stacks and spatial-transform clones. Motion-vector and
history-mask semantics share the existing Torch implementation. Final batch
collation and device transfers remain unchanged. EXR input uses the legacy
assembly path; R2 streaming still rejects these local loader policies.

All guide/reference/count buffers and six SPP layers are retained. Counts stay
int64, radiance/guides float32. Returned windows own contiguous storage; mapped
source arrays are never modified or exposed as mutable tensors. The map cache's
inventory/provenance and mutation checks are unchanged. CPU randomness stays
on the main thread, outside preparation. The new policy is pinned in checkpoint
configuration and is a new experiment; historical policies remain available.

Initial real-data CPU profile: 32 training windows at 128x128, width-independent
preparation, four CPU threads: 1.049 seconds for the existing prefetched loader's
window assembly, 0.556 seconds for direct construction under cProfile. Both
profiles included eleven first-time frame verifications. All sixteen spatial
augmentation combinations matched exactly on real data. This is preparation
profiling, not a claim about GPU end-to-end speed.

Reproduce the paired comparison from clean source:

```bash
PYTHONPATH=src python tools/benchmark_joint.py \
  configs/train-ready-pilot-a100-throughput.json /ABSOLUTE/FRESH/OUTPUT \
  --sweep crops --warmup-epochs 2 --measured-epochs 8 \
  --max-memory-fraction 0.8 --discard-checkpoints
```

The crop sweep holds cache/queue settings from the config fixed and compares
`prefetch_v1` against `prefetch_fused_v1`. It retains exact model/optimizer/RNG
state hashes, histories, provenance and resource samples. No new dataset or
persistent disk cache is created. Headroom is used for preallocated output
windows and asynchronous queued batches, rather than keeping redundant copies.

## A100 results, 2026-11-16

Clean source `875094696830c074bc26e0d79db23e1193a1ac1b`, batch 16, cache 16,
depth 2, unchanged model/precision/seed/crops and existing pilot data. Each
case used two warmup and eight measured epochs, 256 measured training windows
and sixteen updates. Validation stayed batch 1, 256x256, five frames/window.

| Path | Training windows/s | Validation windows/s | Mean epoch seconds |
| --- | ---: | ---: | ---: |
| Existing prefetch | 27.68 | 11.53 | 4.721 |
| Direct construction + prefetch | 36.80 | 12.35 | 4.204 |

Epoch time decreased **10.96%**. Training throughput increased **32.93%** and
validation throughput **7.08%**. Training consumer wait decreased from 0.545
to 0.236 seconds/epoch. Peak GPU allocation/reservation were unchanged; sampled
whole-device GPU use peaked at **19.31%**, and direct-path process RSS at
**8.59 GiB**, below the user's 80% memory caps. No extra dataset or persistent
disk cache was created. Output storage is allocated once per window, not reused
while the consumer or a queued batch may still hold it.

All ten epochs' training and validation metrics matched exactly. Final model
and optimizer/RNG state hashes matched exactly. A separate real-data CUDA
test compared three uninterrupted epochs with one epoch plus a two-epoch
resume: model weights, optimizer, all RNG states, execution flags, inventory
and global step matched exactly. Checkpoints were removed only after these
comparisons and result records were saved.

85 distinct relevant CPU tests passed, including all sixteen spatial
augmentations, output ownership/source-byte preservation, decoded-only full
training and resume with cache 32/depth 4, historical loader/resume behavior,
sequence transforms and the paired benchmark command. Two extended decoded
tests were rerun after adding resume/bounds coverage. Existing deprecation
warnings remain. Real pilot crops also passed all sixteen augmentation cases.

The A100 throughput config now selects `prefetch_fused_v1` for new runs.
Original historical policies/configs remain valid. Changing the policy on an
existing checkpoint is intentionally rejected by exact-resume validation.

Raw artifacts: `/home/ubuntu/workspace/h100-ready-pilot/runs/a100-direct-crops-20261116`
and `a100-direct-crop-resume-20261116` (the latter in the same parent directory).
The resume diagnostic is `/home/ubuntu/denoiser-handoff/gpu_direct_crop_resume.py`.
Compact machine results: `docs/a100-direct-crop-results-20261116.json`.
CPU profiles and the compact working handoff are outside Git in
`/home/ubuntu/denoiser-handoff/`. Measurements use a warm local subset with an
uncontrolled OS cache; they do not establish full-dataset I/O or convergence.
