# R2 streaming training plan

## Scope and current implementation

The production dataset is generated exclusively by the **v4 capture tools** and
streamed from `r2://denoiser/denoiser-v4` using `--layout decoded` (the preparation
CLI default). This reads the lossless arrays published by the sequential and
distributed generators. No v5 capture run or dataset export is required. The
explicit `inline` and `v5` EXR layouts remain supported for compatibility.
Internally the adapter reuses the joint reader's metadata schema and validators.

R2 remains the source of dataset pixels. Local storage holds a frozen metadata
snapshot, training outputs, and temporary image downloads. The first version is
synchronous and uses the existing joint sampler/denoiser trainer. It implements
the path needed to start correctness checks; the 20 GB cache and asynchronous
prefetch are subsequent work. No real BMW/cloud training result is claimed here.

The training path is:

```text
R2 metadata -> local frozen snapshot -> deterministic sequence/crop order
R2 float.npy + uint.npy -> verified temporary files -> memory-mapped shared crop
  -> sampler -> allocation/replay -> denoiser -> sequence loss -> both optimizers
```

For `decoded`, one frame's `float.npy` and `uint.npy` are downloaded in chunks,
checked against their exact byte sizes and SHA-256, and opened as read-only
memory maps. The existing decoded reader copies just the selected crop, including
all six sample layers, reference RGB, guides and integer counts. No EXR is
downloaded or decoded. Source EXR manifests remain part of the provenance audit.
Both temporary arrays are removed after reading, including on errors; only the
crop survives into the training window. At 1920x1080 their combined size is about
307 MB. Downloads still transfer the complete arrays, not a remote crop or byte
range. The minimal version has no persistent cache or prefetch buffer and will
download overlapping windows repeatedly. Removing EXR decoding therefore does
not establish production training throughput; measure transfer time separately.

The loader supports three remote layouts:

| Layout | Base image | Additional layers |
| --- | --- | --- |
| `decoded` (default; all buffers and layers) | `decoded/GROUP/frame_NNNNNN/float.npy` | `decoded/GROUP/frame_NNNNNN/uint.npy` |
| `inline` (current capture archive) | `GROUP/frame_NNNNNN/frame.exr` | `sample_layers/GROUP/frame_NNNNNN/layers.exr` |
| `v5` (standalone companion dataset) | `base/GROUP/frame_NNNNNN/frame.exr` | `layers/GROUP/frame_NNNNNN/layers.exr` |

The decoded layout keeps `capture_lock.json`, `sample_layers_lock.json`,
`decoded_cache_lock.json`, and the original base/layer/cache manifest paths.
The receipt must bind all three locks, selected manifests, and both arrays.
Array declarations, source bindings, channel layouts and listed byte sizes are
checked during preparation; every payload is hashed during its download and its
NPY header is validated before use. Missing decoded arrays are an error, with no
EXR fallback. EXR payloads need not be present in the training source.

The inline layout is converted to v5 **metadata** in the local snapshot. Neither
remote keys nor image bytes are changed. The existing 4-SPP radiance is reused;
the companion must contain genuine independent 1/2/8/16/32-SPP means. Missing
layers are an error. A complete scene's verified archive receipt must include
all selected image/manifest pairs. This version accepts receipts bound to the
current cloud capture lock; adopting historical receipts through capture
revisions requires a later metadata-only exporter. A stale receipt must not be
fixed by editing its hashes.

## Deferred BMW and early Bistro backfills

All **36 BMW clips and the first two Bistro clips** (38 clips / 304 frames) are
outside the original inline layer boundary. The streaming adapter accepts their
later 1/2/8/16/32-SPP backfills through the same v4 object layout. The existing
4-SPP image supplies the fourth layer; base images, guides and references stay
at their original keys. There is no synthetic replacement for missing layers.

The future backfill capture/publisher must produce this contract:

1. Keep the existing `sample_layers_lock.json` and its original capture boundary
   unchanged. Write `sample_layer_configs/GROUP.json` for each backfilled clip:
   its original base config plus `sample_layers` and `reserved_reference_samples`
   matching the established disjoint sample allocation.
2. Publish `sample_layers/GROUP/frame_NNNNNN/layers.exr` and its manifest for every
   frame. Match each base manifest's camera, scene/render fingerprints, integrator
   and render settings. BMW requires the original integrator-11 layer renderer;
   the first two Bistro clips require integrator 12. A matching config alone is
   insufficient if the layer manifest declares another integrator.
3. Verify and publish a current-lock scene archive receipt containing the original
   base image/manifest pairs, the new layer image/manifest pairs, and hashes of
   every backfilled per-clip config. The shared v4 archiver now treats separately
   configured backfills as required payloads: incomplete ones block scene archival.
   The streamer checks the receipt-bound config against the original source
   config and validates the layer manifests before allowing training.
4. Prepare a **new** local metadata snapshot after publication. Existing snapshots
   stay frozen; a run cannot silently adopt newly added backfill frames on resume.

The adapter and archival inclusion are implemented. Generating the deferred
layers and publishing their updated receipts remain future work. Simply uploading
unregistered EXRs, editing an old receipt, or relabelling an incomplete clip does
not make it eligible. Until completion, select only ready clips explicitly;
requesting the full training partition will reject missing backfills.

Decoded training also requires the separate **44-clip / 352-frame decoded
backfill**, recorded in `DECODED_BACKFILL.md`, including the stopped eighth
Bistro clip. The original decoded-cache boundary stays fixed. A backfilled
frame is eligible when it has a valid decoded manifest bound to that cache lock
and its original base/layer manifests, and a current verified scene receipt
includes both arrays and all metadata. It need not be added to the automatic
decode group list. Until those backfills are published, select only completed
decoded clips explicitly. This loader change does not execute backfills.

## Starting a technical pilot after assets are ready

Use an environment with the project and its `[train,test]` dependencies installed.
Credentials are read from environment variables, never from the capture launcher's
Python constants:

```bash
export R2_ACCOUNT_ID=YOUR_ACCOUNT_ID
export R2_ACCESS_KEY_ID=YOUR_ACCESS_KEY_ID
export R2_SECRET_ACCESS_KEY=YOUR_SECRET_ACCESS_KEY
# Alternatively set R2_ENDPOINT_URL for the account/jurisdiction endpoint.
```

Select completed clips from the registered training and validation scenes. Replace
the example IDs below with clips that have complete decoded arrays and verified
scene receipts:

```bash
python3 tools/prepare_r2_training.py r2://denoiser/denoiser-v4 \
  runs/r2-pilot-metadata --layout decoded \
  --train-groups COMPLETED_TRAIN_GROUP \
  --validation-groups COMPLETED_VALIDATION_GROUP

CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 -m wavefront_denoiser train-adaptive \
  runs/r2-pilot-metadata/train-streaming-pilot.json \
  runs/r2-streaming-pilot --device cuda
```

Preparation only lists objects and downloads JSON. It validates scene partitions,
locks, completion records, manifest hashes, declared sample ranges, and known
scene/content overlap. It writes a new metadata directory atomically and refuses
to overwrite an existing snapshot. No EXR/NPY GET, upload, or remote deletion occurs
during preparation. Omitting group selectors requests all train/validation clips;
this fails until all required clips are complete.

The generated pilot configuration uses one epoch, batch size 1, five-frame
windows, one 32x32 crop per window, one fixed validation crop, and widths of 8 in
both networks. These are technical-check settings. Increase crop sizes/widths
for the actual training experiment. Each selected eight-frame clip supplies four
five-frame windows; this pilot runs all of them, without a step limit.

An optional `spool_directory` inside `r2_streaming` places temporary arrays on a
chosen local disk. Relative paths resolve against the training config. The
default is the system temporary directory. Every run uses the same training CLI,
loss, augmentation, and checkpoint formats as local training. There are no new
training dependencies beyond the existing ones.

```json
"r2_streaming": {
  "source": "r2://denoiser/denoiser-v4",
  "layout": "decoded",
  "spool_directory": "../r2-spool"
}
```

`summary.json` reports downloaded bytes/objects, `decoded_cache_frames` (array
reads), and `decoded_frames` (EXR decodes; zero for decoded streaming). Configuration
and checkpoint provenance pin the metadata inventory, source, layout and account
endpoint. Each downloaded payload is SHA-256 verified before use. The up-front
inventory checks JSON; it does not claim to have verified every remote payload byte
before training. Local-only training retains its existing up-front byte checks.

Resume by increasing `epochs` in the same config, selecting a new run directory,
and passing `--resume PREVIOUS_RUN/last.pt`. The snapshot, selections, other
configuration, and device type must remain the same. An interrupted partial
epoch repeats from the previous completed epoch. Download retries reopen files
from byte zero; terminal failures stop the run rather than dropping a sample.

BMW's 36 clips represent one scene, not 36 independent scene identities. A BMW-only
experiment can check forward/backward paths and streaming, but cannot establish
held-out-scene quality. The strict joint trainer continues to require separate
registered train/validation scenes. It does not relabel BMW clips to bypass that
check. The production pilot command above therefore waits for at least one
completed clip in each of those partitions. A future BMW-only gradient smoke
must be labelled as such and kept outside checkpoint selection/quality reports.

## Pilot acceptance checks

1. Metadata preparation downloads JSON only and rejects missing arrays or layers,
   corrupt manifests, cross-partition identities and stale receipts.
2. The same window/crop/seed produces identical local and streamed tensors,
   including motion transforms, layer ordering and 4-SPP reuse.
3. Both networks receive finite, nonzero gradients and update their weights.
   Expected/actual SPP, zero/capacity fractions and unavailable-output rates are
   recorded. Reference images enter the estimator/loss only.
4. With deterministic CPU settings, local and streamed fixture training produce
   identical weights. Two uninterrupted epochs equal one epoch plus resume.
   Repeat the resume check on the actual CUDA host before production training.
5. Network interruptions retry from zero. Corrupt/truncated payloads fail before
   use. Temporary downloads disappear after success or failure. Test payloads are
   never requested for training or checkpoint selection.
6. After real assets arrive, record GPU model, config, source snapshot digest,
   losses/gradients, allocation statistics, bytes downloaded, wall time and peak
   disk/RAM/VRAM use in a pilot report. A passing small pilot establishes technical
   readiness, not learned-sampling effectiveness.

The automated fake-R2 tests use real generated arrays/EXRs and the actual trainer.
Decoded tests work with no remote EXRs and fail on any attempted EXR decode. They validate
the data and optimization path; live R2 credentials/network behavior, cloud
throughput and production CUDA memory/performance remain to be checked.

## Full training workflow

1. **Finish captures and freeze data.** Confirm the selected base/layer clips and
   all disjoint sample ranges, integrator settings and fixed-budget reference
   status. Publish a metadata snapshot under an immutable run/version identity.
   Do not expand the training inventory beneath a resumable run. Use a new run
   and explicit warm start if the inventory must change. Extend the metadata
   exporter to handle validated historical capture revisions before selecting
   older archived scenes through a newer lock.
2. **Run the technical pilot.** First exercise tiny fixture parity, then real
   streamed clips on the training GPU. Check gradients, resume, allocation bounds,
   missing-data behavior, I/O and memory. Keep this diagnostic separate from any
   quality comparison or production checkpoint selection.
3. **Add the bounded cache/prefetch implementation described below.** Establish
   uncached and cached throughput on identical windows. Enable it only after
   crop, optimization and resume parity pass. Keep synchronous mode as an oracle.
4. **Freeze the main configuration.** Start with the existing defaults: both
   networks width 48, batch size 4, five-frame windows, four 128x128 training
   crops per window, Adam at 0.001, five fixed 256x256 validation crops per window,
   average 4 SPP and capacity 63. Pin `capped_softmax_v1`, `unit_denominator`,
   `pixel_mean_v2` and `bilinear_gather_v1`; these include documented engineering
   choices. Use seed 7 initially and independent seeds for final comparisons.
   The existing 1000-epoch setting is a ceiling to assess, not evidence of a
   suitable convergence budget. Set the real budget using validation trends and
   measured time/bytes per epoch. Any new early-stop rule must be specified and
   checkpointed before use.
5. **Train and monitor.** Save `last.pt` each completed epoch and select `model.pt`
   by validation loss. Record sampler/denoiser gradient norms, spatial/temporal
   losses, expected/actual budgets, zero/saturated/unavailable fractions, data
   wait time, optimizer throughput, cache hit bytes, retries, downloaded bytes,
   and resource peaks. Keep test data reserved. Add explicit checkpoint backup
   to a separate R2 output prefix as a later feature; the minimal loader only
   reads R2 and keeps checkpoints locally.
6. **Validate quality after freezing selection.** Add adaptive evaluation support
   for fixed crop and full-resolution recurrent sequences. Compare uniform,
   seeded random, learned sampler with fixed denoiser, and jointly trained models
   at matched measured integer sample budgets and renderer settings. Report
   spatial/temporal metrics, runtime and failure cases, including missing outputs.
   Use the reserved test scenes only after all selection/tuning is frozen.
   Offline layer replay alone does not establish live adaptive renderer speed.

## Planned production cache and prefetch: 20 GB disk budget

Synchronous streaming is the current pilot implementation. Production v4
streaming will add the cache and prefetch below to this same adapter, preserving
the v4 keys, deferred-backfill contract, crop semantics and checkpoint provenance.

Use decimal byte limits: **16,000,000,000 bytes** for verified reusable objects and
**4,000,000,000 bytes** for prefetch/in-flight staging. These are disk budgets;
decoded RAM, pinned memory, GPU tensors and checkpoints need separate budgets.

| Component | Planned behavior |
| --- | --- |
| Object identity | Account endpoint + bucket/key + expected SHA-256; immutable content checked before reuse |
| Reuse cache (16 GB) | Byte-accounted LRU of verified decoded arrays (or EXRs in compatibility mode), with pinned active entries and atomic promotion |
| Prefetch (4 GB) | Bounded background queue for upcoming deterministic windows; includes reservations, partial files and completed prefetched files |
| Scheduling | Deduplicate shared frame requests; sample order and augmentation RNG remain owned by the trainer |
| Eviction | Evict only unpinned completed entries; apply backpressure before issuing a download if quota cannot be reserved |
| Oversized objects | Fail clearly or use an explicitly configured synchronous path with reserved space; never silently exceed the 20 GB total |
| Restart | Remove abandoned partials, validate cache index/file identity and rebuild accounting; cache warmth must not affect training RNG/order |
| Observability | Hit/miss objects and bytes, eviction bytes, queue occupancy, wait/decode/transfer time, retries and peak reserved/actual disk bytes |

Promoting an object from staging into the cache must transfer its byte reservation
atomically, without counting it in neither pool or keeping two unaccounted copies.
If the reader needs files outside the cache/prefetch directories, those temporary
bytes must be included in the same total. Start with one training process and one
prefetch worker. Shared multi-process locks, distributed ranks and multi-GPU
training follow only after single-process correctness is established.

Cache tests must exercise overlap, deduplication, cache-full active windows,
failed downloads, eviction races, restart cleanup and byte limits throughout
downloads/promotions. Compare uncached/cached weights and resume states using
identical seeds and windows. Shuffle windows deterministically; any later
scene-aware locality sampler must be versioned and evaluated for statistical
effects rather than being introduced as an invisible I/O optimization.
