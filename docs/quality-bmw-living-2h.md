# BMW / Living Room two-hour quality pilot

Start from a terminal with the existing GPU environment available and the R2
read credentials exported (`R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, and either
`R2_ACCOUNT_ID` or `R2_ENDPOINT_URL`). Credentials are never written to artifacts.

```bash
bash /home/ubuntu/quality_workspace/run_quality_2h.sh
```

The launcher prepares and downloads the selected completed data, trains, and
exports previews. It does not start training until all selected arrays pass
the existing receipt, manifest, sample-range, NPY-header, and checksum checks.
Missing BMW sample-layer/decoded backfills fail preparation before any pixel
download. No capture/backfill job is launched and no remote objects are changed.

To inspect the plan without downloads or training:

```bash
bash /home/ubuntu/quality_workspace/run_quality_2h.sh --plan-only
```

To download first and train later, use `--prepare-only`, then run the normal
command. Interrupted downloads can be retried with the same command and output
directory; existing matching arrays are verified and reused. Once training has
started, the launcher refuses to overwrite it. Choose a new `--output` for a new
experiment. To finish/retry post-training previews, use `--preview-only`.

For an SSH session that may disconnect, launch once with:

```bash
mkdir -p /home/ubuntu/quality_workspace/runs
nohup bash /home/ubuntu/quality_workspace/run_quality_2h.sh \
  > /home/ubuntu/quality_workspace/runs/bmw-living-quality-launch.log 2>&1 &
```

The experiment uses 12 clips each from BMW and Living Room for training, and six
each from Dragon and Bathroom for development validation: 288 frames, about
88.39 GB of arrays. Training covers all six camera positions in each scene.
Validation covers all camera positions, all six motion types, and two clips at
each motion-speed category per scene. Exact clip IDs are in
`quality-bmw-living-2h.json`. All selected arrays are downloaded once under the
new run; matching existing pilot arrays are shared through verified symlinks.
The experiment data view shares those files and adds a small, hash-pinned split
overlay. Original capture locks, receipts, partitions, data, and archived results
remain intact. Reserved test scenes cannot move or supply training pixels.

Both models start fresh with LARGE width 48, five-frame recurrence, 128×128
random training crops (four/window), 256×256 fixed validation crops (five/window),
batch eight, Adam at 0.0003, seed seven, deterministic float32 execution, and the
existing backward-only near-black gradient floor of 1e-6. The lower learning rate
and batch eight are explicit quality-pilot choices; they differ from the paper's
0.001 learning rate and original batch four and have not been tuned on these
scenes. Allocation remains capped softmax at nominal 4 SPP / capacity 63, with
unit-denominator zero-count gradients and pixel-mean-v2 sparse reconstruction.
These are the existing audited engineering policies, not a claim of exact paper
reproduction. See `Path-Tracer-Denoiser/docs/ntasad-fidelity-audit.md`.

`max_training_seconds=7200` starts after inventory hashing/model setup and covers
initial validation, training, scheduled/final validation, and checkpoint writes.
Validation runs in the first three epochs, every five epochs, and at the final
epoch. The high epoch ceiling leaves the timer in control. After each complete
epoch, measured maximum training/evaluation/checkpoint times determine whether
there is enough reserve for another epoch. This is a soft two-hour target: the
first epoch or an unexpected later I/O slowdown can overrun. Downloading,
metadata preparation, integrity checks, and post-training full-image evidence
take additional wall time. Every completed epoch saves resumable `last.pt`.

The GPU allocator cap is 75% of device memory. A 0.5-second polling guard stops
at 79% whole-host pressure (`MemTotal - MemAvailable`) or whole-device GPU usage,
leaving headroom below the user's 80% limit. Polling cannot bound arbitrary
sub-poll spikes or allocations by unrelated processes. Resource samples for
preparation, training, and previews are recorded separately. CUDA is required
for training and automated previews; there is no silent CPU training fallback.

Outputs are under `/home/ubuntu/quality_workspace/runs/bmw-living-quality-2h/`:

- `train.log` and `train/history.jsonl`: progress, losses, allocation statistics,
  and gradient norms for both networks.
- `train/initial.pt`: initialization for this invocation.
- `train/model.pt`: best validation checkpoint, including initialization.
- `train/best-trained.pt`: best evaluated checkpoint after an optimizer epoch.
- `train/last.pt`: resumable final completed epoch.
- `train/report.md`: training timing/resource report.
- `quality-report.md` and `quality-evidence.json`: initial/best/final full-frame
  quality comparisons and component diagnostics.
- `preview.html`: local viewer with full-frame comparison PNGs and GIF links.
- `previews/`: native rectangular images and HDR denoised EXRs, loss metrics,
  count maps (PNG and exact integer NPY), availability maps, small temporal GIFs,
  checkpoint identities, and actual path totals.

Each checkpoint is previewed on one Dragon clip and one Bathroom clip, using
five native-resolution frames with global image allocation and full recurrent
history. No tiling, guide resizing, or reference input enters predictions.
PNG/EXR outputs are native resolution; GIFs use smaller presentation thumbnails.
The uniform comparison uses the same frozen denoiser with an independent uniform
history. This diagnoses denoiser quality and learned-allocation utility at
nominal 4 SPP; independent rounding means realized integer budgets are reported
rather than exactly matched. These two preview clips do not replace validation
over the twelve selected clips.

After training, open `preview.html`, or inspect `previews/*/comparison.png` with
the image preview tool. A single preview can also be regenerated explicitly:

```bash
source /home/ubuntu/quality_workspace/activate.sh
python tools/preview_joint_pilot.py \
  /home/ubuntu/quality_workspace/runs/bmw-living-quality-2h/train \
  --full-frame --checkpoint best-trained.pt \
  --sequence-id dragon_view_01_left_dolly_slow --device cuda \
  --output /home/ubuntu/quality_workspace/runs/bmw-living-quality-2h/manual-dragon
```

Two hours does not guarantee convergence or sampler improvement. A sampling map
and nonzero gradients alone do not establish either network's quality. Dragon
was previously used in throughput diagnostics, so it is development validation
for this fresh run, not an untouched test. References have fixed SPP and are not
convergence-qualified. Review trained versus initial images, uniform-denoiser
controls, temporal behavior, actual budgets, missing outputs, and held-out losses
before describing either model as performing well.

Preparation and CPU fixture checks were performed without a CUDA device or R2
downloads in the authoring session. A100 runtime, current remote completion, and
full-resolution GPU peak memory must be checked by the launcher on the GPU host.
The focused regression run passed 80 cases with one CUDA-only case skipped;
the additional end-to-end metadata/materialization reuse check passed separately.
These cover time-budget stopping/final saves, CPU resume, split provenance/test
exclusion, native rectangular preview parity, legacy preview behavior, and the
existing joint-training/R2/validation/checkpoint paths. The launcher plan check,
shell syntax check, and Git whitespace check also passed. CUDA preflight correctly
refused to start a real run in this GPU-inaccessible session.
