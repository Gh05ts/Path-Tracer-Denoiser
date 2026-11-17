# Final run on a new machine

This is the entry point after retiring the old machine. Clone `main`; the final
workflow starts **fresh weights** and downloads verified data from
`r2://denoiser/denoiser-v4`. No old home directory, venv, symlink, checkpoint,
renderer executable, or `/tmp` checkout is required. All new data and results go
in a sibling `runs/` directory. Keep R2 credentials in your password manager or
secret store: Git and this handoff do not contain them.

## What the evidence supports

Read [the paper fidelity audit](docs/ntasad-fidelity-audit.md) and
[the completed screen diagnosis](docs/final-quality-screen-diagnosis.md).
The 24-train/12-development-clip screen selected **no candidate**: every adaptive
arm lost to its same-denoiser uniform control at native resolution. Denoising
improvement alone does not establish adaptive-sampling utility. Forced holes
were harmful; relaxed zeros did not exercise sparse training. Final training is
an experiment, not an established reproduction of NTASAD paper quality.

`main` includes `minimum_one_exact_v1`: minimum one path/pixel, capped allocation,
and stable largest-remainder rounding to exactly `4*H*W` paths. Replay and the
DASR backward denominator use the same integer counts. New final configs pin
LARGE/48-channel networks, five frames, 128 training crops, four crops/window,
256 validation crops, five fixed validation crops, batch eight, Adam denoiser
LR 1e-4 / sampler LR 1e-5, backward radiance floor 1e-3, seed seven, float32,
full recurrence, and deterministic execution. These are documented engineering
choices; the minimum-one policy cannot demonstrate learned zero-sample behavior.

## Intended run: joint sampler and denoiser training

The final run **co-trains both networks from fresh weights**. The
`train-adaptive` command calls `train_joint`, whose Adam optimizer includes the
sampler at LR `1e-5` and the denoiser at LR `1e-4`. Both update on every training
batch through the recurrent reconstruction loss; neither network is frozen.
The uniform first frame initializes history, and the remaining four frames use
learned allocation. It is not a denoiser-only run or a staged warm start.

Keep joint training as the chosen experiment despite the earlier screen's
no-candidate result. Evaluate matched-budget native adaptive and same-denoiser
uniform inference, including fixed detail crops of disocclusions, specular
regions and penumbras where present in the development clips. Report regional
and temporal behavior alongside aggregate quality; neither stability nor a few
favorable crops establishes an overall sampling advantage. This run does not
restrict learning to those regions or assume adaptive sampling helps only there.

The technical smoke must show finite, nonzero gradient norms **and changed
weights in both networks**, exact 4-SPP allocation, and no unavailable outputs
before the long run. Minimum-one allocation, the smaller sampler LR and the
radiance backward floor reduce known risks; fresh-machine CUDA stability still
needs verification. If the smoke fails, diagnose it before launching rather than
silently switching to denoiser-only training.

The approved wall-time ceiling for this run is **36 hours**, allowing the
estimated sub-24-hour run to overrun. Use `--hours 35` for preparation's soft
training budget and `--hours 36` for the outer training guard, reserving one hour
for startup and overrun protection. The commands below use these approved budgets.

## 1. Bootstrap

Use a Linux CUDA machine with an NVIDIA driver compatible with PyTorch 2.7.0
(the previous environment used CUDA 12.8). Install Git and Python 3.10 or newer.
The previous tested Python libraries were Torch 2.7.0, NumPy 1.21.5, Pillow 9.0.1,
and OpenEXR 3.5.2; newer Python versions may need a newer NumPy. Record actual
versions and GPU before timing or training. The repository's train/test extras
install the dependencies; choose a CUDA-enabled Torch installation for the host.

```bash
git clone --branch main https://github.com/Gh05ts/Path-Tracer-Denoiser.git
cd Path-Tracer-Denoiser
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'torch==2.7.0'
python -m pip install -e '.[train,test]'
source tools/activate.sh
mkdir -p "$QUALITY_RUN_ROOT"
python -c 'import torch,wavefront_denoiser; print(wavefront_denoiser.__file__); print(torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0))'
git rev-parse HEAD
nvidia-smi
```

`tools/activate.sh` is the portable replacement for the old workspace activation
script: it selects this clone's venv and puts this clone's `src` first. Source it
in each new shell, from the repository. Set `QUALITY_RUN_ROOT` before sourcing
if a larger local disk should hold the sibling runs directory.

Set these environment variables privately (do not paste tokens into committed
files or commands recorded in shared logs): `R2_ACCESS_KEY_ID`,
`R2_SECRET_ACCESS_KEY`, and `R2_ACCOUNT_ID` or `R2_ENDPOINT_URL`.
Data preparation needs list/read permission. Model export also needs write
permission on the **same `denoiser` bucket**, under `model-runs/`. The native R2
client reads only these environment variables; it does not read `~/.aws` or old
capture-launcher constants.

Enter the S3 access-key pair directly in the GPU server's Bash terminal. Hidden
prompts keep the credential values out of shell history; do not paste them into
chat. The account ID is the 32-character Cloudflare account ID:

```bash
set +x
read -rsp 'R2 Access Key ID: ' R2_ACCESS_KEY_ID
printf '\n'
read -rsp 'R2 Secret Access Key: ' R2_SECRET_ACCESS_KEY
printf '\n'
read -rp 'Cloudflare Account ID: ' R2_ACCOUNT_ID
export R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_ACCOUNT_ID
unset R2_ENDPOINT_URL
```

These exports last for this shell and its children. Re-enter them in a new shell
when downloading or uploading; the repository does not persist credentials.

## 2. Freeze the data and actual budget

Preparation requires `--hours`; the command below uses the approved soft budget
for the intended joint run. The default requests every original training and
validation clip.
Preparation fails if any selected decoded backfill or current-lock receipt is
missing. It never silently drops clips or reads reserved test pixels.

```bash
export FINAL_ROOT="$QUALITY_RUN_ROOT/final-01"
python tools/prepare_final_run.py "$FINAL_ROOT" --hours 35 --epochs 1000
cat "$FINAL_ROOT/plan.json"
```

This lists R2 and downloads **JSON only**, validates provenance and split identity,
then writes portable `train.json` and a frozen `data/` metadata snapshot. Inspect
scene lists, counts, `payload_bytes`, disk free space, and reference status. Full
decoded train+validation was estimated near **354 GB**; the printed audited total
is authoritative. If the full archive is incomplete or disk is insufficient,
choose an explicit completed subset via `--selection selection.json` and a **new**
preparation directory. The file has exactly these keys:

```json
{
  "train": ["COMPLETED_ORIGINAL_TRAIN_GROUP"],
  "validation": ["COMPLETED_ORIGINAL_VALIDATION_GROUP"]
}
```

Use registered original partitions, distinct scenes, and real complete arrays.
Do not copy the old BMW/Living/Dragon/Bathroom experiment's lists blindly: it
moved Dragon to development validation. The full final workflow uses original
R2 partitions and records a new inventory. Dragon and other development scenes
have previous experimental use; their results cannot be described as untouched
test performance. Reserved test scenes remain excluded. A different selection
is a different experiment and requires fresh preparation, not a resume.

Download the selected arrays once, with at least the printed payload total plus
8 GB free for outputs (allow more for repeated checkpoint exports):

```bash
python tools/run_with_memory_guard.py --cpu --hours 12 \
  --log "$FINAL_ROOT/materialize.log" --script tools/materialize_r2_training.py \
  -- "$FINAL_ROOT/data"
```

The materializer reuses only size/SHA-256-matching local arrays and verifies every
new download. No EXR decoding or large second disk copy is needed. Interrupted
materialization can rerun with a **new log name** against the same metadata.
R2 snapshots stay frozen. If remote locks have advanced incompatibly, prepare a
new snapshot; never edit receipt hashes. `prepare_final_run.py --materialize`
combines preparation/download for a new output directory, with a disk check.

## 3. Verify the machine, then launch explicitly

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python -m pytest -q tests/test_final_handoff.py tests/test_sampler_stability.py \
  tests/test_joint_checkpoint.py tests/test_checkpoint_cadence.py
```

First use a separate technical smoke config/run (same selection, one epoch,
one crop/window, batch one), with the guard below. Check finite gradients in both
networks, actual 4 SPP, zero unavailable outputs, spatial/temporal losses, and
memory. A full-data epoch may be slow: measure it before committing the final
wall budget. The smoke's changes are a separate experiment, not a resume. Preserve
its config and timings. CPU tests do not establish fresh-machine CUDA throughput,
convergence, or GPU resume parity.

To make that separate smoke configuration without changing the final config:

```bash
python - "$FINAL_ROOT" <<'PYCODE'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
c = json.loads((root/'train.json').read_text())
c['data_root'] = str((root/'data').resolve())
c.update(epochs=1, batch_size=1, patches_per_window=1, max_training_seconds=1800)
(root/'train-smoke.json').write_text(json.dumps(c, indent=2)+'\n')
PYCODE
python tools/run_with_memory_guard.py --hours 1 \
  --log "$FINAL_ROOT/smoke.log" --module wavefront_denoiser \
  -- train-adaptive "$FINAL_ROOT/train-smoke.json" "$FINAL_ROOT/smoke" --device cuda
```

The smoke processes the full selected inventory for one epoch; its soft budget
can be exceeded by that first epoch. If the guard expires, use the measured
partial throughput to adjust the budget or explicitly prepare a smaller
technical subset, while retaining the larger final selection separately.

Once the data, machine smoke, and chosen budget are satisfactory, explicitly
launch from an interactive Bash shell. This survives an ordinary SSH disconnect:

```bash
nohup python -u tools/run_with_memory_guard.py --hours 36 --gpu-id 0 \
  --log "$FINAL_ROOT/train.log" --module wavefront_denoiser \
  -- train-adaptive "$FINAL_ROOT/train.json" "$FINAL_ROOT/train" --device cuda \
  > "$FINAL_ROOT/launcher.log" 2>&1 < /dev/null &
TRAIN_GUARD_PID=$!
printf '%s\n' "$TRAIN_GUARD_PID" > "$FINAL_ROOT/train-guard.pid"
disown
```

The machine must remain running. Do not launch this twice for the same output;
training/log directories must be new. Monitor with:

```bash
tail -f "$FINAL_ROOT/train.log"
```

`Ctrl+C` exits `tail`, leaving training running. After reconnecting, source
`tools/activate.sh` again and restore `FINAL_ROOT="$QUALITY_RUN_ROOT/final-01"`.
The guard writes its status only when it exits:

```bash
cat "$FINAL_ROOT/train.log.status.json"
```

The trainer requests an epoch-boundary stop with measured validation/checkpoint
reserve inside 35 hours; the outer **36-hour** ceiling includes startup hashing
and overrun protection. The trainer may finish an epoch beyond its soft budget;
the guard is the outer deadline and can interrupt an epoch. It saves `last.pt` every completed
epoch. Validation runs in epochs 1–5, every fifth epoch, and at final stop;
`model.pt` selects the best evaluated state including initialization.
`best-trained.pt` selects the best evaluated trained state separately. Do not
assume last is best or that training beat initialization.

The guard makes the chosen physical GPU visible as `cuda:0` (`--gpu-id` accepts
an index or UUID), sets the allocator to at most 75% with existing-use/headroom
allowance, and stops at **79% whole-host or whole-device usage**. Loss of monitoring
also stops work. It records observed peaks and stop reason in `.log.status.json`, and source
commit/tree, Python/package versions and Python source hashes in
`.log.provenance.json`.
One-second polling plus query time cannot bound instantaneous peaks. Leave
headroom and stop other memory-heavy processes. No AMP, compile, large RAM cache,
extra workers, or larger batch is enabled by this handoff.

Preserve config, source commit/tree, package versions, hardware, R2 metadata,
initial/best/last checkpoints, full history, guard log/status and actual update/
epoch counts. A 1000-epoch ceiling is not a convergence claim. If run with a
new duration/selection/hardware, report those differences. Do not tune against
reserved test pixels. The generated presentation includes a same-denoiser uniform
control; a separately trained uniform denoiser is an additional experiment.

## 4. Export weights before retiring any training machine

Follow [checkpoint selection and R2 use](docs/checkpoint-selection-and-r2-use.md)
to compare initialization, crop-selected best, best trained and last weights,
check native adaptive/uniform behavior, choose an export, and restore it for
inference or continued training. Upload is a separate command, not an automatic
post-training action.

Choose the validation-selected checkpoint after comparing it to initialization
and the trained best. The following exports a compact loadable inference model,
the exact JSON metadata needed to fetch its data again, optional full optimizer/
RNG checkpoint, and small run evidence. No image arrays are copied into the bundle.

```bash
python tools/model_r2.py export \
  --checkpoint "$FINAL_ROOT/train/model.pt" \
  --metadata "$FINAL_ROOT/data" --results "$FINAL_ROOT/train" \
  --include-resume --output "$FINAL_ROOT/export-selected" \
  --destination r2://denoiser/model-runs/final-01-selected
```

Export `best-trained.pt` and `last.pt` under different identities if needed.
The script rejects bucket mismatch, dataset-prefix overlap, changed inventory,
local corruption and remote collisions. It uploads objects with create-only PUTs,
publishes the manifest last, verifies sizes/MD5 ETags, and writes an upload receipt.
Downloads recheck every payload's SHA-256. A failed upload can be retried without
rebuilding using `python tools/model_r2.py upload LOCAL_BUNDLE R2_DESTINATION`; use a new output path for a new CLI
export. Existing identical remote objects are reused; differing ones fail.

Verify recovery into a **different local directory** before removing a machine:

```bash
python tools/model_r2.py download r2://denoiser/model-runs/final-01-selected \
  "$QUALITY_RUN_ROOT/recovery-check"
python -c 'import sys; from wavefront_denoiser.joint_checkpoint import load_joint_checkpoint; m,p=load_joint_checkpoint(sys.argv[1]); print(p["epoch"],p["global_step"],m.specification())' \
  "$QUALITY_RUN_ROOT/recovery-check/weights.pt"
```

The bundle supports independent presentation inference on another machine.
Inference weights alone do not resume training. `resume.pt` preserves optimizer/
RNG state, but exact resume also requires the same resolved configuration/data
path, inventory, execution policies and device type: reconstruct the original
path on the new machine or treat relocation as a new experiment. Old diagnostic
`wavefront-training-readiness`/screen checkpoints are deliberately not accepted
by this standard final-model exporter. The fresh final workflow needs none of them.

Existing old-machine PNGs/GIFs, optimizer checkpoints and logs are **not already
backed up by writing this guide**. This source change does not delete anything or
upload those old artifacts. Preserve any history you want separately before
you remove local directories. The source repository and existing R2 dataset are
sufficient for the fresh final run; old trained weights require their own export.

## 5. Generate presentation material

Follow [post-run presentation instructions](docs/post-run-presentation.md).
The generator restores the model from R2 and downloads the selected native
4-SPP image, guides, independent layers and reference through the existing
receipt-bound decoded loader. It runs actual inference, never image synthesis.

For the next agent: read this file first, inspect `plan.json` and all run evidence,
retain the 80% resource ceiling, keep artifacts outside source, preserve dataset
receipts/splits, and distinguish denoiser quality from adaptive allocation utility.
Do not launch a long run merely because this handoff exists.
