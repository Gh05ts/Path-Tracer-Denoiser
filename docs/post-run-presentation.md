# Post-run presentation images, crops and GIFs

First [compare checkpoints and select/export weights](checkpoint-selection-and-r2-use.md).
That guide also covers verified R2 recovery and the distinction between
same-denoiser uniform inference and a separately trained uniform baseline.
Finish checkpoint selection using validation results before including the
reserved test scene. The command below includes validation **and Sponza test**
images from the same frozen checkpoint. It performs inference only; it does not
train or fine-tune either network.

Start from [the final-run handoff](../FINAL_RUN_HANDOFF.md) on the inference
machine. The trained bundle must already be exported to R2. Activate the new
clone's venv with `source tools/activate.sh` and supply the R2 read credentials.
No paths or pixels from the training machine are needed for inference.

```bash
python tools/run_with_memory_guard.py --hours 2 \
  --log "$QUALITY_RUN_ROOT/slides-01.log" --script tools/generate_presentation.py \
  -- --model-r2 r2://denoiser/model-runs/final-01-selected \
  --output "$QUALITY_RUN_ROOT/slides-01" --device cuda --hdr --include-test
```

This creates `slides-01-model/` for verified weights and frozen metadata, and
`slides-01/` for outputs. Each name must be new. A subsequent rendering can use
`--bundle "$QUALITY_RUN_ROOT/slides-01-model"` with another output name, saving
the model download. Data arrays are streamed from R2 per frame, SHA-256 checked,
and removed immediately after loading; the spool holds only one frame's files.
The existing decoded pair contains 4-SPP RGB, guides, reference and all replay
layers together (about 307 MB at 1080p). The generator does not claim to download
only RGB channels or request remote crops. Test pixels are downloaded only when
`--include-test` explicitly enables the held-out evaluation path.

Without `--include-test`, default clip selection is the lexicographically first
validation clip per scene (Bathroom and Classroom in the full final run).
With `--include-test`, the first clip per reserved test scene is added (Sponza).
Training scenes are excluded in both cases. Selection is reproducible and does
not search for the largest quality gain. Select particular clips using
`--groups CLIP_A CLIP_B`; test clip IDs also require `--include-test`.
Use `--all-validation` for complete selected-validation coverage and
`--include-test --all-test` for complete reserved-test coverage.
All eight frames (or the recorded clip length) are
processed in order; frame zero initializes independent uniform/adaptive histories.
Full images determine allocation and history; exported crops are taken **after
native inference**, so a crop never changes the sampler's global budget.

Every clip directory contains:

| File | Use |
| --- | --- |
| `frame_NNNNNN/uniform-4spp.png` | Captured/replayed uniform 4-SPP noisy baseline |
| `frame_NNNNNN/adaptive-input.png` | Adaptive noisy replay at exactly mean 4 SPP |
| `frame_NNNNNN/adaptive-denoised.png` | Learned allocation plus trained denoiser |
| `frame_NNNNNN/uniform-denoised.png` | Same trained denoiser, independent uniform history |
| `frame_NNNNNN/reference.png` | Actual captured high-SPP reference |
| `frame_NNNNNN/sample-count.npy` | Exact native integer path map |
| `comparison-native.png` | Final-frame 3-column 4-SPP / adaptive+denoised / reference with crop boxes |
| `controls.png` | Five-column overview including adaptive input and uniform-denoiser control |
| `crop_01/`, etc. | Identical native coordinates for all five methods, comparison strips and 3x nearest-neighbor PNGs |
| `comparison.gif` | Temporal 3-column animation using one palette across time |
| `*.exr` with `--hdr` | Final-frame linear HDR images for all five methods |
| `metrics.json` | Frame budgets, encoded spatial/temporal errors, availability, crop coordinates and display settings |

`presentation.json` records checkpoint SHA-256, epoch, method policies, R2 download
statistics, chosen clips and all clip metrics. Each clip records its partition,
and `groups_by_partition` separates validation from test. Do not combine the
two splits into a single reported performance average. `index.md` labels each
clip's split and provides image/GIF links.
The generator requires the final `minimum_one_exact_v1` 4-SPP checkpoint and
rejects mismatched inventory/policies. References enter metrics and reference-only
crop ranking; **they never enter prediction**. The first GIF frame is labelled
uniform initialization. The default six-FPS playback is a presentation setting,
not a measurement of renderer frame rate. GIF quantization is lossy; use native
PNGs or EXRs to assess detail, not the animation's colors.

## Sponza final test

For the full original split, Sponza has 18 held-out test clips. To render all 36
validation clips and all 18 Sponza clips using the already restored bundle:

```bash
python tools/run_with_memory_guard.py --hours 2 \
  --log "$QUALITY_RUN_ROOT/slides-full-01.log" --script tools/generate_presentation.py \
  -- --bundle "$QUALITY_RUN_ROOT/slides-01-model" \
  --output "$QUALITY_RUN_ROOT/slides-full-01" --device cuda --hdr \
  --all-validation --include-test --all-test
```

This is a larger workload than the representative three-clip command; size the
inference guard from measured throughput and available resources. The active
training materialization does not need to be restarted or extended for Sponza.

The generator writes `evaluation-protocol.json` with the chosen checkpoint hash,
clip selection and display/crop settings **before reading any test metadata or
pixels**. It then creates a separate `test-metadata/` snapshot under the output
directory, verifies the original capture/sample-layer/decoded locks against the
training bundle, and requires current, verified scene receipts and complete
decoded arrays. Test scene identities and image hashes must not overlap training
or validation. Frozen training metadata and `train.json` are never expanded.
Missing or changed test data causes an error; do not edit locks or receipts to
bypass it. The test snapshot is rejected by the training materializer and
requires explicit inference access in the streaming loader.

For a Sponza-only presentation, pass `--include-test --groups
sponza_view_00_circle_slow` to the generator with a new output/log name. This
processes one complete clip; the representative default also includes Bathroom
and Classroom. Keep test results separate from validation and retain negative
results. After inspecting Sponza, do not use its scores or images to choose a
different checkpoint or tune training/settings while describing it as untouched
final test evidence. A saved protocol records this run's choice; it cannot prove
the scene was never inspected in an earlier experiment.

## Choose informative crops and consistent images

Default crops include the center and up to two nonoverlapping regions ranked by
**reference image gradients alone**. This highlights edges/texture without ranking
methods by their improvement. Inspect all scenes, include a typical case and a
failure case, and keep the uniform-denoiser control visible in the appendix.
Do not imply that a selected attractive crop represents aggregate quality.

For deliberate crops, save a JSON file mapping exact clip IDs to native pixel
`[x, y, width, height]` lists. Coordinates have top-left origin. Example coordinates
must be adjusted to the actual resolution:

```json
{
  "YOUR_VALIDATION_CLIP": [[650, 300, 160, 160], [1000, 500, 160, 160]]
}
```

Rerender using the already restored model:

```bash
python tools/run_with_memory_guard.py --hours 2 \
  --log "$QUALITY_RUN_ROOT/slides-crops.log" --script tools/generate_presentation.py \
  -- --bundle "$QUALITY_RUN_ROOT/slides-01-model" \
  --output "$QUALITY_RUN_ROOT/slides-crops" --device cuda \
  --groups YOUR_VALIDATION_CLIP --crops crops.json --exposure 1 --hdr
```

Use the same exposure for every method and every frame of a clip. `--exposure`
is a positive linear multiplier, followed by `x/(1+x)` and display gamma 1/2.2.
There is no method-specific normalization, sharpening, registration or retouching.
Prefer detail crops containing thin geometry, specular edges, shadows and moving/
disoccluded regions. Keep crop coordinates and exposure in the slide notes.
To make a PowerPoint-friendly still, use `comparison-native.png` or native PNGs;
for enlarged details use the `-3x.png` files. Insert `comparison.gif` directly in
slides that support animated GIFs; playback includes initialization to expose the
history warm-up. The earlier screen's GIFs cannot be recovered from Git: regenerate
from exported compatible final weights, or separately retain old run artifacts.

## Captions and evidence

Suggested caption: “Offline 4-SPP independent-layer replay, LARGE NTASAD-style
model; native global exact-budget allocation. All methods share exposure and crop
coordinates. Reference is the captured high-SPP target. Clip: …; frame: …;
checkpoint/epoch: ….” Use the recorded reference sample settings/status; never
rename a fixed-budget reference “ground truth” or claim convergence without its
qualification. BMW base/layer integrator compatibility remains a dataset caveat.

The primary spatial average covers frames 1 onward; the temporal average covers
pairs 1→2 onward. Seed-frame and 0→1 diagnostics remain in per-frame metrics.
These are encoded-domain L1 diagnostics, not paper PSNR. Compare adaptive against
uniform+same-denoiser at **equal actual integer paths**, show per-scene/per-clip
results and unavailable-output rates, and retain negative results. Offline replay
quality does not establish live adaptive renderer speed or paper-equivalent
quality. Validation presentation uses the frozen development inventory. Reserved
test presentation uses the separate, explicitly enabled path above after
checkpoint selection is frozen.
