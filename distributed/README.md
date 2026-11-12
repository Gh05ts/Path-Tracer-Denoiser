# Distributed v4 clip generation

One task is one entire eight-frame clip: the locked 4-SPP inputs and 1024-SPP
references, its required 1/2/8/16/32-SPP companion layers, and its required decoded
cache. Existing backfill boundaries remain in force. Earlier clips are not
silently rerendered or given different integrators/sample ranges.

The two machines have separate local disks and coordinate through the same R2
bucket/prefix. No inbound connection or always-on coordinator server is needed.
Each machine runs one GPU capture stream and one background CPU process that
decodes and uploads completed clips while its GPU captures the next clip.
At most two finish jobs may be pending on each machine.

R2 stores a frozen plan, per-clip claims and verified completion records under
`_distributed/denoiser-v4/`. Atomic conditional writes prevent two workers from
claiming the same clip. A shared active marker also prevents a second independent
queue from managing the same dataset prefix. Claims survive process/machine restarts; there is no
automatic expiry that could steal work from a slow or disconnected renderer.
See [R2 conditional operations](https://developers.cloudflare.com/r2/api/s3/conditional-operations/).

Each worker uploads its clip's EXRs, decoded arrays, manifests and logs directly
to their usual v4 paths. Writes are create-only: conflicting existing objects
are rejected, never overwritten. Complete clip verification and an R2
completion readback precede local EXR/array cleanup. Local manifests/logs stay.
Once every clip of a scene is verified, workers publish a deterministic scene
receipt under `r2-archives/`. Existing BMW receipts retain their original binding.
This distributed workflow uploads per clip, rather than waiting for all scene
payloads to be present on one local disk.

## Current handoff

Generation is stopped. The preparation captures all 162 v4 tasks: 36 BMW clips
already archived, eight Bistro clips with progress on the primary machine
(seven complete base clips and `bistro_view_05_forward_dolly_slow` at 7/8), and
118 untouched clips available to either worker. The eight local clips are
reserved for `3090-a` so another machine cannot regenerate their existing work.
Their stored frame hashes are checked on the primary worker.

Decoded backfill still covers the first 44 clips / 352 frames, including all of
the stopped clip. Additional-layer backfill still covers 38 clips / 304 frames.
This generator resumes capture; it does not perform those deferred backfills.

## Prepare and initialize on the primary machine

Commands below run from `Path-Tracer-Denoiser/` using a Python 3.10+ environment
with this checkout installed (`pip install -e .`). The primary environment is
currently `/tmp/ntasad-venv/bin/python`. Preparation is local and does not start
rendering or upload anything:

```bash
python distributed/generate.py prepare \
  --output ../dataset/denoiser-v4 \
  --workspace .. \
  --prepared ../dataset/denoiser-v4/distributed/prepared \
  --primary-worker 3090-a
```

Initialize once, after checking the preparation. This verifies existing archived
BMW objects and uploads the frozen metadata/queue, but starts no workers:

```bash
python distributed/generate.py init \
  --prepared ../dataset/denoiser-v4/distributed/prepared \
  --output ../dataset/denoiser-v4 \
  --destination r2://denoiser/denoiser-v4 \
  --settings-file tools/generate_r2_dataset.py
```

Keep the original primary output directory. Initialization marks it as managed
by distributed workers; the sequential generator then refuses to run against
it. Do not run the old sequential resume alongside distributed workers.

## Set up the second RTX 3090 machine

Copy both updated repositories and the complete scene assets, preserving their
relative directory layout. Copy the exact persistent renderer binary
`dataset/bin/dataset_capture-base-integrator12`. Install compatible CUDA/runtime
libraries and the same Python/OpenEXR versions. A rebuilt executable with a
different SHA-256 is rejected, even if its source appears identical.

The workspace's absolute path may differ. The worker verifies the renderer and
all referenced scene/geometry/texture bytes against v4. Only temporary runtime
configs have their scene paths rebased; the original capture/config/provenance
files and renderer fingerprints remain unchanged. No existing image dataset
needs to be copied to the second machine. It downloads the shared metadata.

Provide `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` and optionally
`R2_ENDPOINT_URL` through the local environment. Alternatively, pass a local
`--settings-file`; only its literal credential constants are read, without
executing the launcher. Environment values take precedence. Credentials and
launcher scripts are never included in the shared metadata bundle.

Check a machine without claiming/rendering a task:

```bash
python distributed/generate.py check \
  --destination r2://denoiser/denoiser-v4 \
  --output ../dataset/denoiser-v4 \
  --workspace .. \
  --renderer ../dataset/bin/dataset_capture-base-integrator12
```

## Start workers when ready

Primary machine, using its original dataset directory:

```bash
python distributed/generate.py worker \
  --destination r2://denoiser/denoiser-v4 \
  --settings-file tools/generate_r2_dataset.py \
  --worker-id 3090-a --workspace .. \
  --output ../dataset/denoiser-v4 \
  --renderer ../dataset/bin/dataset_capture-base-integrator12
```

Second machine, with its own local output directory and environment credentials:

```bash
python distributed/generate.py worker \
  --destination r2://denoiser/denoiser-v4 \
  --worker-id 3090-b --workspace .. \
  --output ../dataset/denoiser-v4 \
  --renderer ../dataset/bin/dataset_capture-base-integrator12
```

Use a unique, stable worker ID per machine. The output directory retains a
machine UUID and a local filesystem lock, preventing another local worker or a
different workspace with the same worker ID from taking its active claims.
Rerun the same worker command to resume interrupted tasks and reuse verified
completed frames, layers, decoded files and previously uploaded objects.
`--max-clips 1` limits a worker to one task for an initial production check.

To detach, use the existing log/PID launcher:

```bash
DATASET_PYTHON=/tmp/ntasad-venv/bin/python \
  bash tools/run_dataset_background.sh distributed worker \
  --destination r2://denoiser/denoiser-v4 \
  --settings-file tools/generate_r2_dataset.py \
  --worker-id 3090-a --workspace .. \
  --output ../dataset/denoiser-v4 \
  --renderer ../dataset/bin/dataset_capture-base-integrator12
```

Workers exit when no eligible pending or locally owned task remains. Clips
owned by another machine remain claimed; they are not timed out and rerendered.

## Status and recovery

```bash
python distributed/generate.py status \
  --destination r2://denoiser/denoiser-v4 \
  --settings-file tools/generate_r2_dataset.py
```

Status lists task counts, owning workers, phases and claim tokens. For a crashed
worker, restart it with the same worker ID and local output directory. If a
claim must instead be released, first stop that worker and its background
process, then explicitly release the current token:

```bash
python distributed/generate.py release \
  --destination r2://denoiser/denoiser-v4 \
  --group CLIP_ID --claim-token TOKEN_FROM_STATUS
```

Primary-affinity tasks still require their original local progress. Releasing a
claim does not migrate those files to another disk. Upload failures retain
payloads locally; an incompatible remote object stops the worker for inspection.
If a scene-receipt write was interrupted after task completion, the next worker
run repairs the receipt. Local payloads retained after such an interruption may
be removed after confirming their verified completion record.

The queue uses R2 conditional writes. Existing sequential B2/R2 generation is
still supported; B2 is not a distributed coordinator backend in this version.
