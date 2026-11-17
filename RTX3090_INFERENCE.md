# RTX 3090 inference previews

Use a frozen **best-so-far** checkpoint while the final training run continues.
This generates visual comparisons from one eight-frame clip per requested scene,
with full native-resolution recurrent history. It does not run full validation
or evaluation, train either network, or require the training image archive on
the inference machine. Both the sampler and denoiser are included.

## 1. Export on the training server

Use an updated, separate inference clone/worktree of `main` for these tools;
leave the running trainer's checkout and environment alone. Activate that clone's
environment as in [the handoff](FINAL_RUN_HANDOFF.md). `FINAL_ROOT` below is the
existing prepared run, containing `train/` and `data/`.

```bash
source tools/activate.sh
export FINAL_ROOT=/home/ubuntu/runs/final-01
export SNAPSHOT_ID="final-01-preview-$(date -u +%Y%m%dT%H%M%SZ)"
export EXPORT_DIR="$HOME/denoiser-exports/$SNAPSHOT_ID"

python tools/model_r2.py export-best \
  --run "$FINAL_ROOT" --output "$EXPORT_DIR"
cat "$EXPORT_DIR/selection.json"
```

`export-best` snapshots `train/model.pt`: the lowest measured crop-validation
loss saved so far, including initialization. It pins one complete checkpoint
before loading, hashing and exporting it, even if training replaces the best
file during export. `selection.json` records epoch, updates and available
validation metrics. If `initialization_selected` is true, training has not yet
beaten initialization on that selection metric. A missing score is recorded as
`null`; trainer completion and `summary.json` are not required.

Export uses CPU and copies weights, frozen JSON metadata and small training
evidence. It excludes image arrays and optimizer/resume state. Training keeps
its original checkpoints and continues normally.

Set R2 credentials privately in the export shell if they are not already exported:

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

The exporter needs write permission under `denoiser/model-runs/`. Upload the
frozen bundle under its own identity:

```bash
python tools/model_r2.py upload "$EXPORT_DIR" \
  "r2://denoiser/model-runs/$SNAPSHOT_ID"
```

Keep the printed R2 URI for the 3090. You can also pass `--destination` directly
to `export-best` to export and upload in one invocation. Repeat with a **new**
snapshot ID when you want newer weights. Existing differing remote objects are
rejected; retry an interrupted upload with the same local bundle and URI.

If you prefer direct transfer, copy the entire exported bundle to the 3090
with `scp -r` or a downloaded archive and use `--bundle /path/to/unpacked/bundle`
instead of `--model-r2`. This bypasses model upload/download; the 3090 still
needs R2 read credentials for the selected dataset frames. Keep that bundle
outside both the preview output and the new disposable download directory.

## 2. Prepare the RTX 3090 server

Use a separate Linux clone with a working NVIDIA driver. The 3090's 24 GB is
expected to be suitable for frame-at-a-time inference, but native-resolution
VRAM use and throughput must be checked on this machine. Start with Bathroom.
The first run also verifies data access before the presentation deadline.

```bash
git clone --branch main https://github.com/Gh05ts/Path-Tracer-Denoiser.git Path-Tracer-Denoiser-inference
cd Path-Tracer-Denoiser-inference
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip 'setuptools>=64' wheel
python -m pip install 'torch==2.7.0'
python -m pip install --no-build-isolation -e '.[train]'
source tools/activate.sh
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'

export INFERENCE_ROOT="$HOME/denoiser-inference"
mkdir -p "$INFERENCE_ROOT/logs"
```

Use a CUDA-enabled Torch installation compatible with the installed driver;
the CUDA check above must succeed. No renderer executable is needed. Set R2
credentials using the hidden prompts above, with **list/read** access to both
the model URI and the original `denoiser/denoiser-v4` dataset. Credentials stay
in the shell environment and must be re-entered after reconnecting.

## 3. Download and generate selected previews

Set `MODEL_URI` to the URI actually uploaded in step 1:

```bash
export MODEL_URI=r2://denoiser/model-runs/YOUR_SNAPSHOT_ID
python tools/run_with_memory_guard.py --hours 6 \
  --log "$INFERENCE_ROOT/logs/bathroom-001.log" \
  --script tools/preview_inference.py -- \
  --model-r2 "$MODEL_URI" \
  --download-dir "$INFERENCE_ROOT/downloads/bathroom-001" \
  --output "$INFERENCE_ROOT/previews/bathroom-001" \
  --scenes bathroom --device cuda
```

The six-hour value is a configurable deadline, **not** a runtime estimate. The
guard caps the PyTorch allocator at up to 75% of VRAM and stops if whole-device
or host memory reaches 79%. Frame progress includes elapsed time; per-frame
load/inference/export timings, total clip times and peak allocated GPU memory
are retained in `metrics.json`, `presentation.json` and `preview.json`.
These are offline replay timings, not live renderer FPS. `--fps` only controls
GIF playback. No RTX 3090 benchmark has been established yet.

`--scenes` accepts one or more of:

| Argument | Preview selection |
| --- | --- |
| `bathroom` | First lexicographic Bathroom clip in the checkpoint's frozen validation inventory |
| `classroom` | First lexicographic Classroom clip in that inventory |
| `sponza` | Fixed held-out `sponza_view_00_circle_slow` clip |

For all three, use a fresh pair of directories and `--scenes bathroom classroom
sponza`. This is 24 frames, roughly 7–8 GB of streamed input at 1080p. One clip
is roughly 2.5 GB of network transfer. Files are hash-checked and released after
each frame loads; disk holds approximately one frame's payloads at a time rather
than the entire clip. The model alone does not replace dataset access.

To reuse the downloaded weights for Classroom and Sponza without downloading
the model again:

```bash
python tools/run_with_memory_guard.py --hours 6 \
  --log "$INFERENCE_ROOT/logs/remaining-001.log" \
  --script tools/preview_inference.py -- \
  --bundle "$INFERENCE_ROOT/downloads/bathroom-001/model" \
  --download-dir "$INFERENCE_ROOT/downloads/remaining-001" \
  --output "$INFERENCE_ROOT/previews/remaining-001" \
  --scenes classroom sponza --device cuda
```

Scene selection controls which data is downloaded. Training clips are excluded.
Requesting Sponza explicitly enables held-out access for that frozen snapshot;
use validation to choose weights and treat these selected images as previews,
not a full evaluation or evidence for choosing a later checkpoint from Sponza.

Add `--hdr` for final-frame EXRs. Add `--crop-size 160` to change automatic crop
size, or `--crops /path/to/crops.json` for explicit native `[x,y,width,height]`
boxes keyed by the selected exact clip IDs. Automatic crops use center plus
reference-gradient regions. All methods use the same crops and exposure
(`--exposure 1` by default). Cropping happens after full-frame inference and
does not change allocation or history.

Use `nohup`/`tmux` if SSH may disconnect. Pick new log, output and download
directory names on every retry; partial outputs are retained for diagnosis.
The original [presentation guide](docs/post-run-presentation.md) covers the
separate full-validation/evaluation workflow.

## 4. Inspect outputs and rotate disk space

Directories are kept separate and may not overlap or nest, including through
symlinks. For the first command:

```text
denoiser-inference/
  downloads/bathroom-001/
    model/                  verified weights, frozen metadata, selection evidence
    work/spool/             temporary frame downloads; empty after normal completion
  previews/bathroom-001/
    index.md                scene comparisons and links
    preview.json            completion, epoch, checkpoint hash, timing and directories
    presentation.json       per-clip evidence; explicitly labelled visual_preview
    evaluation-protocol.json
    CLIP_ID/
      comparison-native.png
      controls.png
      comparison.gif
      sample-count.npy      exact final-frame integer SPP, not normalized
      sample-count.png      grayscale display, fixed linear 0–63 SPP
      allocation-heatmap.png white to blue; darker means more samples
      allocation-legend.png fixed labelled scale
      frame_000000/...      five method PNGs, raw counts and both allocation maps
      crop_01/...           matching method/map crops, raw counts and 3x displays
      metrics.json
  logs/bathroom-001.log
```

The comparison shows uniform noisy 4-SPP input, adaptive denoised output and
the reference. `controls.png` additionally shows adaptive noisy input and the
uniform input processed by the **same trained denoiser**. The heat map uses
actual integer allocation, a fixed `log2(1+SPP)/log2(64)` scale and a labelled
white-to-blue palette for every frame and scene. This is an NTASAD-style
allocation visualization, not a claim to match the paper's exact palette.
Frame zero initializes uniformly at 4 SPP, so its map is constant. The current
minimum-one policy cannot show learned zero-sample pixels.

Check completion before keeping images or removing downloads:

```bash
python - "$INFERENCE_ROOT/previews/bathroom-001" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
preview = json.loads((root/'preview.json').read_text())
report = json.loads((root/'presentation.json').read_text())
assert preview['complete'] and report['complete']
assert preview['checkpoint_sha256'] == report['checkpoint_sha256']
print('Complete:', preview['scenes'], 'epoch:', preview['epoch'])
print('Seconds including model download:', preview['elapsed_seconds'])
print('Peak allocated GPU GB:', (preview['peak_gpu_allocated_bytes'] or 0)/1e9)
PY
du -sh "$INFERENCE_ROOT/downloads/bathroom-001" "$INFERENCE_ROOT/previews/bathroom-001"
```

After confirming completion and finishing any reuse of `model/`, remove just
that disposable download directory:

```bash
rm -rf -- "$INFERENCE_ROOT/downloads/bathroom-001"
```

Preview PNGs, maps, GIFs and reports remain usable without downloads. Retain
the R2 URI and reports to reproduce the snapshot. Rotate preview directories
separately after copying the chosen presentation assets. No command
automatically deletes old models, training state or preview outputs.
