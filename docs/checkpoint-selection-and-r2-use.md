# Select final weights and use them from R2

Use this after the [joint-run handoff](../FINAL_RUN_HANDOFF.md). Commands run
from the repository with `source tools/activate.sh`, R2 credentials exported,
and `FINAL_ROOT` pointing to the prepared run, for example:

```bash
export FINAL_ROOT="$QUALITY_RUN_ROOT/final-01"
```

## Check completion and compare the saved checkpoints

Read `train.log.status.json`, `train/summary.json` and the last entries in
`train/history.jsonl`. A nonzero guard exit, deadline or memory stop is different
from a clean trainer stop. An interrupted run can still have useful completed
epochs; do not describe a partial epoch as completed. The 1000-epoch setting is
a ceiling, not an expected update count or a convergence guarantee.

| Checkpoint | Meaning in the fresh, time-budgeted final run |
| --- | --- |
| `initial.pt` | Fresh epoch-zero model, before any optimizer updates |
| `model.pt` | Lowest evaluated crop-validation loss, **including initialization** |
| `best-trained.pt` | Lowest evaluated crop-validation loss among trained epochs |
| `last.pt` | Latest fully completed epoch; preferred starting point for continuing training |

Lower validation `loss` is better. It is `spatial_l1 + temporal_l1` in the
encoded radiance domain, with equal weights. Compare validation losses, not
random training-crop losses. Validation runs in epochs 1–5, every fifth epoch,
and at the final stop; an unevaluated `last.pt` has no measured validation rank.
Two filenames can contain identical weights. A saved best is best among
evaluated states in this invocation, not necessarily among every training step.

Print a checkpoint table using the trainer's recorded fixed-crop measurements:

```bash
python - "$FINAL_ROOT/train" <<'PYCODE'
import json, pathlib, sys
from wavefront_denoiser.joint_checkpoint import load_joint_checkpoint

root = pathlib.Path(sys.argv[1])
summary = json.loads((root/'summary.json').read_text()) if (root/'summary.json').is_file() else {}
rows = [json.loads(line) for line in (root/'history.jsonl').read_text().splitlines()]
validation = {row['epoch']: row['validation'] for row in rows if row.get('validation') is not None}
if summary.get('initial_validation') is not None:
    validation[0] = summary['initial_validation']  # This guide assumes a fresh run.
print('checkpoint           epoch    updates          loss       spatial      temporal')
for name in ('initial.pt', 'model.pt', 'best-trained.pt', 'last.pt'):
    path = root/name
    if not path.is_file():
        print(f'{name:20s} missing')
        continue
    model, payload = load_joint_checkpoint(path)
    metrics = validation.get(payload['epoch'])
    numbers = (' '.join(f'{metrics[k]:13.7f}' for k in ('loss', 'spatial_l1', 'temporal_l1'))
               if metrics is not None else 'not recorded; evaluate before ranking')
    print(f"{name:20s} {payload['epoch']:5d} {payload['global_step']:10d} {numbers}")
    del model, payload
PYCODE
```

If a hard interruption leaves an incomplete final JSONL line, preserve the
original log and read only its complete JSON records. If `summary.json` is
missing, the initial metrics may be unavailable; native evaluation below can
still compare the saved weights. Do not invent missing scores.

Start with `model.pt` as the crop-selected default. If it is epoch zero, say
that training did not beat initialization on that selection metric. Keep
`best-trained.pt` as the trained candidate, rather than silently relabelling it
as an improvement. Keep `last.pt` for recovery even when its quality is worse.

## Compare native adaptive and uniform behavior

Crop selection and native-resolution performance can disagree, as the earlier
screen showed. Before choosing deployment weights, evaluate initialization and
the best trained candidate on the **same frozen validation clips**, native
resolution, full recurrent histories, exact 4-SPP budgets and shared exposure.
Include `model.pt` and `last.pt` when they represent different candidates. Do
not use reserved test pixels to choose weights.

Create local inference bundles without uploading them. These contain weights,
JSON metadata and small evidence, not a copy of the training image arrays:

```bash
python - "$FINAL_ROOT" <<'PYCODE'
import pathlib, sys
from wavefront_denoiser.model_artifacts import build_bundle

root = pathlib.Path(sys.argv[1])
for name in ('initial', 'model', 'best-trained', 'last'):
    checkpoint = root/'train'/(name+'.pt')
    if checkpoint.is_file():
        build_bundle(checkpoint, root/'data', root/'comparison-bundles'/name,
                     results=root/'train')
PYCODE
```

Each bundle/output directory must be new. Run the following for each candidate,
changing `CANDIDATE` but keeping the validation inventory and options fixed:

```bash
export CANDIDATE=best-trained
python tools/run_with_memory_guard.py --hours 2 \
  --log "$FINAL_ROOT/compare-$CANDIDATE.log" --script tools/generate_presentation.py \
  -- --bundle "$FINAL_ROOT/comparison-bundles/$CANDIDATE" \
  --output "$FINAL_ROOT/compare-$CANDIDATE" --device cuda --all-validation
```

The two-hour evaluation guard is an example cap, separate from the training
budget; size it from measured inference throughput. Full validation exports
PNGs, sample maps and GIFs and streams the selected image arrays from R2 again
for each candidate, so check disk and bandwidth first. For a small technical
preview omit `--all-validation`, but do not equate that representative subset
with full validation. An interrupted evaluation without `presentation.json`
is incomplete; use a new output/log name for a rerun. A `nohup` launch as in the
handoff also works for these guard commands if SSH may disconnect.

Once complete reports exist, print comparable native scores. Pass only
completed candidate output directories; the script rejects different clip sets:

```bash
python - "$FINAL_ROOT/compare-initial" "$FINAL_ROOT/compare-best-trained" <<'PYCODE'
import json, pathlib, sys

groups = None
print('candidate          epoch     adaptive       uniform    adaptive vs uniform')
for folder in map(pathlib.Path, sys.argv[1:]):
    report = json.loads((folder/'presentation.json').read_text())
    assert report['complete'] and report['clips'], 'Incomplete evaluation'
    current = sorted(report['groups'])
    assert groups is None or groups == current, 'Different validation clip sets'
    groups = current
    scores = {}
    for variant in ('adaptive', 'uniform'):
        values = [clip[variant]['spatial_encoded_l1'] + clip[variant]['temporal_encoded_l1']
                  for clip in report['clips']]
        scores[variant] = sum(values)/len(values)
    adaptive, uniform = scores['adaptive'], scores['uniform']
    delta = f'{100*(adaptive/uniform-1):+.3f}%' if uniform else 'undefined (zero control loss)'
    print(f"{folder.name:20s} {report['epoch']:5d} {adaptive:13.7f} {uniform:13.7f} {delta:>20s}")
PYCODE
```

These native scores equally weight clips and sum mean encoded spatial and
temporal errors; they are not the crop-validation score or paper PSNR. Spatial
means cover frames 1 onward; temporal means cover pairs 1→2 onward. Negative
`100*(adaptive/uniform - 1)` means adaptive is better. Also compare the trained
adaptive score to initialization, and inspect the spatial and temporal components
in each report. Retain per-scene/per-clip results and any regressions.

Inspect identical native crops of disocclusions, specular regions and penumbras
where present; choose coordinates before comparing candidate improvements and
use them for every candidate. The [presentation guide](post-run-presentation.md)
documents `--crops`, fixed exposure, full controls, EXR export and GIF limits.
The current metrics are full-frame/clip metrics; crop images do not automatically
produce semantic-region scores. Do not infer an aggregate win from selected crops.

The uniform inference control uses **the same jointly trained denoiser** with
independent recurrent history and uniform 4-SPP inputs. It measures allocation
utility for those weights. A denoiser optimized specifically for uniform
sampling requires a separate training run with matched data, architecture,
losses and reported training budget. This joint run does not supply that baseline.

Keep the original crop-selection result intact. If native validation leads to
a different deployment choice, record that as a separate development selection
in `selection.json`: checkpoint filename/SHA-256/epoch, selection metric, frozen
clip IDs, candidate scores, per-clip regressions and reason. Include that file
alongside retained reports; do not rename checkpoints to erase their provenance.
If no trained candidate beats initialization or uniform, report that outcome.

## Export the selected model and recovery state

Training does **not** upload automatically. Choose the checkpoint explicitly
after the comparison. The example below exports the trained candidate; change
`SELECTED_CHECKPOINT` if another checkpoint was selected:

```bash
export SELECTED_CHECKPOINT="$FINAL_ROOT/train/best-trained.pt"
python tools/model_r2.py export \
  --checkpoint "$SELECTED_CHECKPOINT" \
  --metadata "$FINAL_ROOT/data" --results "$FINAL_ROOT/train" \
  --include-resume --output "$FINAL_ROOT/export-selected" \
  --destination r2://denoiser/model-runs/final-01-selected
```

The bundle contains `weights.pt` (both networks, inference only), `resume.pt`
(the selected checkpoint including optimizer/RNG state), frozen `metadata/`,
training evidence and a checksum manifest. It does not contain image arrays.
The exporter currently copies only its fixed evidence file list; archive
`selection.json` and native comparison outputs separately. Their inclusion is
not implied by `--results`. Preserve the original source revision and environment
records along with those artifacts.

To preserve the latest training progress too, export `train/last.pt` to a
different local directory and `r2://denoiser/model-runs/final-01-last`. This is
necessary when selected weights come from an earlier epoch. Export initialization
or other candidates under distinct identities if they must also survive machine
retirement. Use a new identity for changed contents: existing differing remote
objects are rejected. A failed upload can be retried with:

```bash
python tools/model_r2.py upload "$FINAL_ROOT/export-selected" \
  r2://denoiser/model-runs/final-01-selected
```

## Restore and run inference on another machine

Bootstrap a compatible clone as in the handoff, source `tools/activate.sh`, and
export R2 list/read credentials. The original training directory is unnecessary
for inference. Verify recovery into a new directory **before retiring the GPU
machine**:

```bash
python tools/model_r2.py download r2://denoiser/model-runs/final-01-selected \
  "$QUALITY_RUN_ROOT/final-01-restored"
python - "$QUALITY_RUN_ROOT/final-01-restored" <<'PYCODE'
import pathlib, sys
from wavefront_denoiser.model_artifacts import verify_bundle
from wavefront_denoiser.joint_checkpoint import load_joint_checkpoint

root = pathlib.Path(sys.argv[1])
manifest = verify_bundle(root)
model, payload = load_joint_checkpoint(root/'weights.pt')
print('Verified epoch:', payload['epoch'], 'updates:', payload['global_step'])
print('Both-network model:', model.specification())
print('Resume payload:', manifest['resume_checkpoint'])
PYCODE
```

Generate native adaptive output, the matched uniform-denoiser control,
reference comparisons and GIFs from the restored bundle:

```bash
python tools/run_with_memory_guard.py --hours 2 \
  --log "$QUALITY_RUN_ROOT/final-01-inference.log" --script tools/generate_presentation.py \
  -- --bundle "$QUALITY_RUN_ROOT/final-01-restored" \
  --output "$QUALITY_RUN_ROOT/final-01-inference" --device cuda --hdr
```

This defaults to one reproducible validation clip per scene. Use
`--all-validation` for the full selected validation inventory, or
`--groups EXACT_VALIDATION_CLIP_ID` for an explicit clip. Read `index.md`,
`presentation.json` and each clip's `metrics.json`; the presentation guide lists
all PNG/EXR/GIF products. The generator fetches hash-verified frame arrays from
the dataset's original R2 location and releases temporary frame files as it goes.
The model bundle alone does not replace access to that dataset.

Alternatively, replace `--bundle ...` with
`--model-r2 r2://denoiser/model-runs/final-01-selected` to download and infer in
one command; it creates a new sibling `OUTPUT-model` directory. All output,
download and log names must be unused. For CPU inference use `--cpu` on the
guard **and** `--device cpu` on the generator; expect different throughput.

The supported final-model inference path is `generate_presentation.py` and the
joint Python API, not the older uniform-only `denoise`/`denoise-sequence` CLI.
These weights support offline independent-layer replay with guides and temporal
history. Live adaptive path-tracer scheduling/integration and arbitrary-scene
capture are separate work; the R2 bundle is not a live renderer plugin.

## Continue training from recovered weights

`weights.pt` cannot resume training. Use `resume.pt`, preferably from the
separately exported `final-01-last` bundle when the goal is the latest progress.
Resume requires the same resolved data path, frozen inventory, configuration
(only the total epoch ceiling may increase), execution policies and device type.
Restore/materialize the frozen metadata/arrays at the original `data_root` and
use the saved `evidence/config.json`; the small bundle contains no arrays.
Do not change paths or learning rates and describe that as exact resume.

After restoring those prerequisites and increasing the total epoch target if
needed, use the [handoff's SSH-safe guard launch](../FINAL_RUN_HANDOFF.md) with
the inner command:

```bash
python -m wavefront_denoiser train-adaptive ORIGINAL_COMPATIBLE_CONFIG.json \
  NEW_RESUME_OUTPUT_DIRECTORY --device cuda \
  --resume "$QUALITY_RUN_ROOT/final-01-last-restored/resume.pt"
```

An interrupted partial epoch is repeated from the last saved completed epoch.
The time budget starts afresh for each invocation, so continuing training needs
its own chosen wall-time budget. Exact new-machine GPU parity still needs a
technical check; the repository does not establish it from CPU tests alone.
