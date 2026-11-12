# Joint adaptive training

The v5 trainer updates the sampler and sparse denoiser together through five
recurrent frames. It supports bounded allocations, crop-aware layer replay and
epoch-boundary resume. The uniform trainer and its checkpoints retain their
existing interfaces.

Production inputs come from the v4 generator's inline layers in R2, including
later BMW/early Bistro backfills. See the [streaming training plan and minimal
loader](r2-streaming-training.md). No v5 generation is required; the adapter
reuses the joint reader's local metadata schema. A 16 GB cache and 4 GB prefetch
buffer are planned after the synchronous pilot. The optional standalone v5
layout remains compatible with the training instructions below.

## Run and resume

From the denoiser checkout, after the selected v5 train/validation clips have
completed:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 -m wavefront_denoiser train-adaptive \
  ../dataset/denoiser-v5/train-joint.json runs/adaptive --device cuda
```

The v5 planner now writes `train-joint.json`; rerunning the plan stage adds it
to existing plans without overwriting an edited configuration. The older
`train-joint-plan.json` remains a historical experiment description. A standalone
template is available at [`../configs/train-adaptive.json`](../configs/train-adaptive.json).
Relative data paths resolve against the configuration file.

The inline production dataset also provides `train-joint-decoded.json` with
`data_root` set to that dataset directory. Once its selected caches and backfill
are complete, the same `train-adaptive` command accepts this configuration.
It needs `decoded/` arrays, source/companion manifests, `capture_lock.json`,
`sample_layers_lock.json` and `decoded_cache_lock.json`, but no EXR payloads.
The original train/validation/test assignments and provenance checks are retained.

For a pilot, set `train_groups` and `validation_groups` to explicit completed
group IDs in their registered partitions. Null selects the entire partition.
Use smaller widths/crops for a technical check, and record those differences.
The full defaults use 48 channels in each network, five-frame windows, batch
size 4, four random 128-pixel training crops per window, Adam at 0.001, and
five fixed 256-pixel validation crops per window. Both networks start fresh.
The template specifies 1000 epochs; validation should determine a suitable
production convergence budget.

To resume, increase the total `epochs` in the same configuration and choose a
new output directory:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 -m wavefront_denoiser train-adaptive \
  ../dataset/denoiser-v5/train-joint.json runs/adaptive-resumed --device cuda \
  --resume runs/adaptive/last.pt
```

Only the total epoch target may change. Resume checks the model specification,
all other configuration fields and dataset inventory. The current exact-resume
implementation requires `num_workers=0` and the saved device type. It resumes
completed epochs; an interrupted partial epoch is repeated from `last.pt`.

## Data, recurrence and loss

The trainer verifies hashes, manifests and completion records for selected
train/validation clips before creating the run directory. It rejects shared
scene fingerprints or image payloads across those partitions. Known test scene
identity is checked from metadata when available; test EXRs are never decoded
or hashed for model selection. A shared dataset lock allows concurrent readers
and rejects a capture writer.

For decoded inline data, `JointPatchDataset` memory maps float32/uint32 arrays
and copies only the requested crop. Inventory verification hashes the arrays
once; crop access checks metadata/headers without opening EXRs or hashing whole
arrays. The legacy v5 path decodes one full frame at a time using batched EXR
channel reads, copies the crop, then releases the full frame.
Both paths apply the same crop, flip and rotation to guides, all six RGB layers
and references, including existing motion/normal transforms. Validation uses
fixed center/corner crops without consuming augmentation RNG.

Each window starts with dense uniform replay. Four later frames run sampler ->
allocation -> replay -> sparse denoiser, preserving gradients through history
for both networks. Training references enter only the replay backward estimator
and loss. Evaluation prediction never receives a reference. The loss is the
existing transformed-domain final-frame spatial L1 plus final-pair temporal L1
with equal weights. Unavailable predictions remain in the loss.

## Allocation and paper differences

The six independent layer means represent integer counts 0..63. The paper's
unbounded softmax can request more samples than this bank contains. Two explicit
policies are provided:

| Policy | Expected allocation | Capacity behavior |
| --- | --- | --- |
| `paper_softmax_v1` | Paper softmax normalization followed by independent rounding | Rejects a rounded count above capacity |
| `capped_softmax_v1` (default) | `min(C, lambda * exp(logit))`, with lambda chosen to preserve the per-image continuous budget | Active-set redistribution enforces capacity |

The bounded policy is an engineering variant, not an exact reproduction of the
paper's allocation equation. Its unsaturated pixels retain ordinary softmax
gradients; saturated pixels have zero local allocation derivative. Neither
policy imposes a minimum count. Independent round-to-even means actual integer
path totals can differ from the continuous budget. No hidden budget repair is
performed. History records both expected/actual SPP, zero/capacity fractions,
unavailable outputs and total replayed paths/pixels. The SPP means cover learned
frames; path totals also include uniform initialization.

The default `zero_policy=unit_denominator` extends the DASR count-gradient
surrogate to zero counts. `zero` is an alternative ablation. Both are explicit
extensions beyond the cited positive-count expression. See
[`adaptive-reconstruction.md`](adaptive-reconstruction.md) for the derivation
and sparse reconstruction choices. These policies and their versions are
pinned by configuration and checkpoints.

## Determinism and checkpoints

The joint model uses `bilinear_gather_v1` history warping and strict PyTorch
deterministic algorithms. Four pixel-space bilinear gathers match grid-sample
values and interior derivatives; the derivative at an exactly clamped boundary
uses the gather implementation's explicit convention. CUDA grid-sample backward
was insufficient for exact resume, so joint checkpoints pin the gather backend.
Standalone adaptive APIs retain their existing default warp. Set
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before launching CUDA training. The trainer
restores process-wide deterministic/cuDNN flags when it exits.

The separate `wavefront-ntasad-joint-checkpoint` v2 format records both networks,
two named Adam parameter groups and moments, Python/NumPy/CPU/CUDA/loader RNG,
execution flags, epoch/global step, configuration, verified layer inventory and
architecture/feature/gradient/warp versions. Saves use atomic replacement;
nonfinite model or optimizer tensors are rejected. Uniform checkpoints cannot
be misidentified as joint checkpoints.

The default `reconstruction_version=pixel_mean_v2` uses an image pyramid of
equally weighted available pixel means; it agrees with ordinary box downsampling
when every pixel is sampled, even at unequal SPP. The older `sample_mass_v1`
weighted colors by sample counts. Original v1 checkpoints still load and run
with that original policy. To continue their training, explicitly select
`"reconstruction_version": "sample_mass_v1"`; switching an old optimizer/RNG
checkpoint to v2 reconstruction is rejected. Saving an old model uses the v2
envelope while retaining its v1 reconstruction policy. See the
[fidelity audit](ntasad-fidelity-audit.md) for specified and inferred details.

Each run writes:

* `model.pt`: the best validation model in this invocation, including its
  initial model. A resumed invocation does not compare against unloaded best
  weights from the preceding run.
* `last.pt`: resumable state at the most recently completed epoch.
* `config.json` and `inventory.json`: exact experiment/data provenance.
* `history.jsonl`: losses, allocation statistics and both gradient norms.
* `summary.json`: initial/best validation metrics and selection scope.

The Python API is `JointNTASAD`, `joint_optimizer`, `save_joint_checkpoint`,
`load_joint_checkpoint`, and `restore_joint_training` in their respective
`joint`/`joint_checkpoint` modules. A weights-only save is usable for inference
but cannot resume training.

## Validation and remaining work

Before the fidelity correction, the integrated suite passed **278 tests with no skips**, including allocation
gradchecks, sparse reconstruction, provenance/crop alignment, deterministic
warp derivatives, two-network checkpoint validation and exact CPU resume.
A separate RTX 3090 pilot captured 16 real 17x13 Demo/Prism frames with 16-SPP
references and trained reduced-width networks for two epochs. Both networks
received nonzero gradients. Two uninterrupted epochs and one epoch plus resume
produced bitwise-identical parameters after eight optimizer steps. Training
loss decreased from 0.06858 to 0.06395; validation loss decreased from 0.13927
after epoch one to 0.13803 after epoch two (initial 0.14068).

These are small-fixture crop losses, not paper PSNR results or evidence that
adaptive sampling beats uniform sampling. Production v4 data is expected at
`../dataset/denoiser-v4`; production reuse, native-resolution validation,
convergence training, matched-budget sampler comparisons and live CUDA adaptive
primary-ray scheduling remain to be completed.

The subsequent [fidelity audit](ntasad-fidelity-audit.md) verifies the v2
pixel-mean reconstruction and v1 compatibility: 266 model/data/checkpoint tests
passed on the GPU-enabled run, including EXR interoperability. The existing
renderer capture test stalled on its first frame, so both capture modules were
excluded from that completed run; a fresh full-suite pass is not claimed.
