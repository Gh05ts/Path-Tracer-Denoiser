# Run the completed-data pilot on an H100

This pilot uses completed v4 decoded captures and excludes all pending sample-layer
and decoded backfills. The clip selection was checked against local distributed
verification receipts. Remote metadata preparation rechecks the current scene
receipts, manifests and array inventories before allowing the download.

| Partition | Clip | Frames |
| --- | --- | --- |
| Training | `breakfast_room_view_00_right_orbit_slow` | 8 |
| Training | `dragon_view_01_left_dolly_slow` | 8 |
| Validation | `bathroom_view_00_dolly_medium` | 8 |
| Validation | `classroom_view_02_linear_fast` | 8 |

The 64 arrays total **9,820,577,792 bytes (9.82 GB)**. Metadata and training outputs
need additional space. There are no test pixels in this selection. This is a
technical diagnostic, not the final full-dataset experiment or a convergence study.

Both networks use width 48. Training uses batch 4, five-frame windows, four
128x128 crops/window; validation uses five fixed 256x256 crops/window, batch 1.
Three epochs give **24 optimizer steps** and **160 validation windows**, including
initial validation. Training windows start from independent uniform initialization
and perform four learned sampler/replay/denoiser steps with recurrent gradients.

The pilot explicitly sets `radiance_gradient_floor=1e-6`. Extremely small positive
capture values (down to float32 subnormals) otherwise produce enormous gamma
derivatives that can overflow through recurrent allocation gradients on the first
batch. This opt-in policy leaves forward image, allocation and loss values
unchanged and bounds the gamma derivative at the stated linear-radiance floor;
it is a training-gradient surrogate for values below that floor. Exact black
retains its zero subgradient. Existing configs that omit the field retain their
legacy derivative behavior. The config/checkpoints pin the policy, so changing
it during an exact resume is rejected. This is an engineering stabilization and
must be disclosed in experimental results; it is not the exact paper derivative
in the near-black interval. No NaNs are replaced and nonfinite checks remain on.

## 1. Rent and connect

Use one H100 80 GB with a CUDA/PyTorch Linux image. For this pilot, 100 GB of
usable disk provides headroom for the environment, 10 GB data and checkpoints.
If the same instance will hold the entire dataset afterward, provision the
planned 500 GB–1 TB, preferably local NVMe. Use the provider's SSH connection
instructions and save its host/port. The pilot does not deploy or rent anything.

The portable `h100-ready-pilot.tar.gz` contains source, the pilot config and the
tools below. It excludes capture credentials, EXRs, arrays, existing runs and Git
history. Transfer it using the provider's upload feature or SCP from the machine
where the bundle is stored, substituting the host, port and user shown by the
provider:

```bash
scp -P PORT /tmp/h100-ready-pilot.tar.gz USER@HOST:/workspace/
```

In the rented instance's terminal:

```bash
cd /workspace
tar -xzf h100-ready-pilot.tar.gz
cd h100-ready-pilot
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip 'setuptools>=64' wheel
python -m pip install '.[train]'
python -c 'import torch; print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
nvidia-smi
df -h .
```

The environment should use the image's GPU-enabled PyTorch installation. If the
CUDA assertion fails, fix the image/driver setup before proceeding. Do not use a
CPU run to estimate H100 performance. Keep the environment and package versions
unchanged between this pilot and the subsequent training benchmark.

## 2. Supply read-only R2 access

Use a token with object read and bucket listing access. These prompts avoid
putting the secret in shell history; credentials are used only on the rented
instance and are not written to the report:

```bash
read -rp 'R2 account ID: ' R2_ACCOUNT_ID
read -rp 'R2 access key ID: ' R2_ACCESS_KEY_ID
read -rsp 'R2 secret access key: ' R2_SECRET_ACCESS_KEY
printf '\n'
export R2_ACCOUNT_ID R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY
```

If the bucket uses a jurisdiction endpoint, also set `R2_ENDPOINT_URL` to its
HTTPS S3 endpoint without a bucket/path.

## 3. Freeze metadata and download just the selected data

Run from the extracted project directory:

```bash
python tools/prepare_r2_training.py r2://denoiser/denoiser-v4 runs/ready-pilot-data \
  --layout decoded \
  --train-groups breakfast_room_view_00_right_orbit_slow dragon_view_01_left_dolly_slow \
  --validation-groups bathroom_view_00_dolly_medium classroom_view_02_linear_fast

python tools/materialize_r2_training.py runs/ready-pilot-data
du -sh runs/ready-pilot-data
```

Preparation fetches JSON only. Materialization downloads and hashes only the
selected arrays, then validates their NPY headers. It never fetches test arrays
or EXRs. Rerun materialization after an interrupted download; existing matching
files are verified and reused. If metadata preparation is interrupted or rejects
a receipt, it removes its staging directory. If a completed snapshot already
exists, reuse it instead of rerunning preparation into that directory.

Training is **local**, using `configs/train-ready-pilot.json` without an
`r2_streaming` option. The small config generated by metadata preparation is not
used: this benchmark deliberately keeps the production model and crop sizes.

## 4. Run the three-epoch pilot

```bash
bash tools/run_ready_pilot.sh
```

For a run that survives closing SSH, use this command instead of the foreground
command above (launch only once):

```bash
mkdir -p runs
nohup bash tools/run_ready_pilot.sh > runs/ready-pilot-launch.log 2>&1 &
```

Check progress from another terminal:

```bash
tail -f runs/ready-pilot-h100.log
```

Setup hashes the selected local arrays and runs initial validation before creating
the run directory. The trainer prints a record after each completed epoch;
silence during this setup is not proof of a stalled GPU. `nvidia-smi` can be run
in another terminal to inspect GPU utilization. The runner refuses existing run
or log paths. For a new attempt, choose a new path, for example:

```bash
bash tools/run_ready_pilot.sh runs/ready-pilot-h100-attempt2
```

An out-of-memory failure at these settings is a useful finding. Preserve the log
and hardware file and report it before reducing batch/crop sizes. If a network
download fails, training has not started; resume the materialization first.

## 5. Read and bring back the results

On successful completion the runner automatically writes and prints:

```text
runs/ready-pilot-h100/report.md
```

The report includes training/data-wait/validation/checkpoint times per epoch,
setup and initial validation time, peak process RAM, peak PyTorch allocated and
reserved VRAM, software/GPU versions, losses, both networks' gradient norms,
sample allocation statistics, and a full-dataset timing extrapolation. It does
not promise convergence. PyTorch memory peaks exclude memory outside its allocator;
process RAM is lifetime peak RSS. The 10 GB subset may fit in the OS page cache,
so its local I/O can be faster than the full dataset's I/O.

To regenerate the report:

```bash
python tools/report_joint_pilot.py runs/ready-pilot-h100
cat runs/ready-pilot-h100/report.md
```

Paste `report.md` back into the chat, or download a compact diagnostics archive:

```bash
tar -czf ready-pilot-diagnostics.tar.gz \
  runs/ready-pilot-h100/report.md \
  runs/ready-pilot-h100/summary.json \
  runs/ready-pilot-h100/history.jsonl \
  runs/ready-pilot-h100/config.json \
  runs/ready-pilot-h100/inventory.json \
  runs/ready-pilot-h100.log \
  runs/ready-pilot-h100.hardware.txt
```

Use the provider's file download or SCP from your own computer:

```bash
scp -P PORT USER@HOST:/workspace/h100-ready-pilot/ready-pilot-diagnostics.tar.gz .
```

Also retain/download `model.pt` and `last.pt` before deleting the instance or its
disk. They are deliberately excluded from the compact diagnostics archive.
If training fails before a report is written, bring back the `.log` and
`.hardware.txt` files instead. Do not put R2 credentials in the diagnostics archive.

## View a denoised validation crop

After training, run the optional preview exporter (available in
`ready-pilot-preview.tar.gz` for an older pilot bundle):

```bash
python tools/preview_joint_pilot.py runs/ready-pilot-h100 --device cuda
```

If the gradient-fix rerun used `runs/ready-pilot-a100-fixed`, substitute that run
path. The tool defaults to the trained `last.pt`, because validation-selected
`model.pt` can still be the initial model after a short pilot. Its output JSON
records the checkpoint epoch/step, provenance and crop location. A 256x256 center
crop from the first selected validation clip is replayed through five recurrent
frames; the final frame is exported. No reference enters prediction. This is a
crop diagnostic, not native-resolution inference or a quality/convergence claim.

Open `RUN/preview/comparison.png` for uniform 4-SPP input, adaptive noisy input,
joint denoised output and reference at the same display exposure. Individual PNGs,
`denoised.exr`, `sample-count.png`, `availability.png` and `preview.json` are also
written. Download the PNG to your computer to view it. The exporter refuses to
overwrite outputs; use `--output NEW_DIRECTORY` for another preview. Use
`--sequence-id classroom_view_02_linear_fast` to select the other validation
scene, or `--window 3` to display the final frame of the eight-frame clip. A
larger `--patch-size` is optional and has not been benchmarked for memory/runtime.

## Interpreting this pilot

Finite nonzero gradients in both networks, finite losses/allocation statistics
and completed checkpoints establish that the joint training path operates on
real completed captures. Three epochs on four clips cannot establish final
quality or generalization. The report estimates full-epoch time from per-batch
training and per-window validation measurements, normally excluding the first
epoch. Validate that estimate on a larger selection before committing the final
epoch budget. Full-resolution recurrent evaluation requires its own benchmark.

When backfills become available, use a new frozen inventory for the final run.
The strict resume interface rejects changed dataset inventory; continuing this
pilot with newly added scenes is not an exact checkpoint resume. Plan the final
initialization/warm-start choice explicitly after reviewing the pilot results.
