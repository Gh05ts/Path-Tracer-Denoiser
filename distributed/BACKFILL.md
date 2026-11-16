# Deferred v4 backfills on both machines

Run from `Path-Tracer-Denoiser/`, after activating the same Python 3.10+
environment used for generation. Pull/copy the updated denoiser checkout to
both machines. Keep their existing local dataset directories and R2 settings.
No scene assets or original EXRs need to be copied between the machines: the
runner restores EXRs from their verified R2 scene receipts as needed.

Both machines use the existing persistent **integrator-12** renderer. BMW's
archived bases remain integrator 11, and its new layers declare integrator 12.
This is an explicit exception using the existing two-camera BMW comparison in
`capture-history/bmw-comparison.json`, SHA-256
`ee506a00532bb6c2ac9967ed88417bf7df2f7cf949afc40d34678c2f6896b10c`.
That comparison is limited evidence, not proof of identical pixels across all
views. The original bases, references, guides, and sample ranges are retained.
The layer validator requires the same scene identity, cameras, and render
settings, and a fingerprint differing only by its final integrator version.
Each BMW layer manifest includes this source compatibility binding, while
retaining the renderer's actual integrator-12 metadata and fingerprint.
Other scenes and version pairs receive no such exception.

## Initialize once, on either machine

This checks that every original generation task is complete, installs the
frozen generation metadata locally, validates the source archives, and creates
the separate shared backfill queue. It starts no rendering. Rerunning it reuses
the same frozen plan and preserves claims/progress.

```bash
python3 distributed/backfill.py init \
  --destination r2://denoiser/denoiser-v4 \
  --settings-file tools/generate_r2_dataset.py \
  --output ../dataset/denoiser-v4
```

The current v4 boundaries give **44 clips / 352 frames** to decode, including
**38 clips / 304 frames** requiring new 1/2/8/16/32-SPP layers. The remaining six
clips restore their existing layers and only decode. Initialization prints the
counts and required renderer checksum. It refuses incomplete generation,
unverified source receipts, or unrecognized BMW compatibility evidence.

## Check each machine

On the secondary machine, first export its local runtime libraries in the
same shell (adjust the path if its workspace differs):

```bash
export LD_LIBRARY_PATH="/home/myid/lv37348/pt/dataset/bin/capture-runtime${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Run this check on both machines. It downloads metadata and verifies the exact
integrator-12 binary and source scene/asset bytes without claiming a task:

```bash
python3 distributed/backfill.py check \
  --destination r2://denoiser/denoiser-v4 \
  --settings-file tools/generate_r2_dataset.py \
  --output ../dataset/denoiser-v4 --workspace .. \
  --renderer 12=../dataset/bin/dataset_capture-base-integrator12
```

## Start both workers

On the **primary machine**, after activating its environment:

```bash
DATASET_PYTHON=python3 \
  bash tools/run_dataset_background.sh backfill worker \
  --destination r2://denoiser/denoiser-v4 \
  --settings-file tools/generate_r2_dataset.py \
  --worker-id 3090-a --workspace .. \
  --output ../dataset/denoiser-v4 \
  --renderer 12=../dataset/bin/dataset_capture-base-integrator12
```

On the **secondary machine**, after activating its environment and exporting
the runtime library path:

```bash
DATASET_PYTHON=python3 \
  bash tools/run_dataset_background.sh backfill worker \
  --destination r2://denoiser/denoiser-v4 \
  --settings-file tools/generate_r2_dataset.py \
  --worker-id 3090-b --workspace .. \
  --output ../dataset/denoiser-v4 \
  --renderer 12=../dataset/bin/dataset_capture-base-integrator12
```

Each worker claims different whole clips from R2, restores their original
EXRs, captures only missing layers, decodes, uploads, and verifies the entire
clip before releasing local EXR/array storage. Each machine handles one clip
at a time, with two OpenEXR decoder threads. Reserve space for a complete clip's
base and layer EXRs plus decoded arrays (the uncompressed arrays alone are
about 2.5 GB per 1080p clip). Capture and decoding run sequentially per machine.
The launcher survives SSH disconnects and prints its log/PID paths.

Use `--max-clips 1` for a one-clip production check or `--scenes bmw` /
`--scenes bistro` to restrict a worker. Both machines may use the default full
queue; conditional claims prevent duplicate capture. Claiming never expires
automatically. Restart with the same worker ID and local output to resume.
Keep one worker per local output directory; do not restart generation workers
or the sequential generator against the backfill dataset.

## Status and recovery

```bash
python3 distributed/backfill.py status \
  --destination r2://denoiser/denoiser-v4 \
  --settings-file tools/generate_r2_dataset.py
```

An interrupted worker retains its claim and validated local work. Restart its
same command to resume without rerendering completed layers. A failed upload
or conflicting remote object retains local payloads. Conflicts are never
overwritten. After all backfill clips in a scene complete, the worker verifies
the complete scene inventory, retains its original receipt under
`_backfill/denoiser-v4-backfill/source-receipts/`, and conditionally publishes a
new current-lock receipt at `r2-archives/SCENE.json`. Concurrent receipt changes
are rejected. A restarted worker repairs interrupted receipt publication and
cleanup after checking retained payload hashes.

If a claim must move to another machine, stop the old worker and its renderer
first, then release its exact token from status:

```bash
python3 distributed/backfill.py release \
  --destination r2://denoiser/denoiser-v4 \
  --settings-file tools/generate_r2_dataset.py \
  --group CLIP_ID --claim-token TOKEN_FROM_STATUS
```

Already uploaded identical objects are reused. Local partial work on the old
machine is not migrated automatically. The new machine may need to recapture
unpublished layers, but never rerenders source base frames or references.

After all 44 backfill clips are complete, prepare a **new** training metadata
snapshot. Existing snapshots and checkpoints keep their original frozen
inventory; the runner does not silently expand a running experiment.
