# Final handoff validation, 2026-11-17

The focused integration selection contains 121 cases covering model artifacts,
allocation stability, training/resume/checkpoints, previews, decoded R2 transport,
and historical pilot/screen tools. The first combined run passed 116 and exposed
five integration failures: four preview fixtures mixed the new training-config
policy with a historical model policy; the older diagnostic optimizer ignored
the new separate sampler LR. Fixtures now construct the configured model, and
the diagnostic optimizer honors the configured sampler rate before applying its
explicit arm multiplier. All 35 cases in the affected handoff/preview/pilot/
diagnostic selection then passed. After adding local bundle checksum validation,
all 11 handoff cases passed again. This is coverage across focused runs, not a
claim that a full repository/GPU suite was executed.

New end-to-end coverage uses real tiny decoded arrays and models with a fake R2
HTTP transport. It exports/validates a model bundle, uploads idempotently, removes
the original local metadata/checkpoint/bundle, restores solely from fake R2, and
runs native inference plus PNG/EXR/crop/GIF export. It verifies identical recovered
weights, no reference buffers reaching either network, equal actual adaptive and
uniform path budgets, exact crop pixels, temporal GIF frame count, and empty
spool directories. It rejects changed inventories, local weight corruption,
remote object collisions, bucket/prefix mistakes and traversal paths. Preparation
tests verify JSON-only R2 requests and no image-array downloads.

A real CPU child process successfully ran through the wall/memory guard; status,
source commit/tree, source-file hashes and package-version provenance were
checked. Python compilation, CLI help, shell syntax and `git diff --check` passed.
The available environment used Torch 2.7.0, NumPy 1.21.5, Pillow 9.0.1 and
OpenEXR 3.5.2; the presentation code supports Pillow 9 enum naming.

No final training, production-size inference, live R2 write/readback, data
regeneration, deletion, or new quality claim was made. A fresh-machine CUDA smoke,
actual completed-data receipt audit, disk/throughput measurement, live model
upload/readback and final native quality evaluation remain run-time steps in the
handoff. Existing experimental quality evidence and its no-candidate result are
preserved separately.

## Joint-run verification, 2026-11-17

The intended final experiment remains fresh joint sampler/denoiser training,
with a 35-hour soft budget and the user's approved 36-hour outer ceiling.
The focused handoff/stability/checkpoint/cadence selection now passes all
**36 tests** on CPU (Torch 2.7.0, NumPy 1.21.5, Pillow 9.0.1, OpenEXR 3.5.2).
The new integration test prepares and materializes receipt-bound tiny decoded
arrays through fake R2, then invokes `train-adaptive` with the generated final
profile's 48-channel networks, learning rates, backward floor and fused loader.
Only workload dimensions and CPU threads are reduced for the technical smoke.
It confirms finite nonzero gradients and changed weights in both networks,
finite spatial/temporal losses, exact 4-SPP path counts, and zero unavailable
outputs. This establishes CPU execution of joint updates, not long-run quality
or GPU stability. Existing NumPy deprecation warnings were non-failing.

This workspace has no usable NVIDIA driver/device and no configured R2
environment credentials. No live dataset audit/download, CUDA smoke, final
training or model upload was performed. Those runtime steps remain required
before treating the run as launched or stable on the training machine.

The checkpoint-selection/R2-use guide's ten Bash blocks pass `bash -n`, all
four embedded Python snippets parse and execute against retained tiny fixture
artifacts, and local Markdown links resolve. These checks exercise checkpoint
score lookup, local bundle creation for all four saved states, native report
aggregation and recovered-bundle validation. The handoff/README/presentation
Bash blocks also pass syntax checks, and `git diff --check` passes. The native
aggregation check uses fixture reports; it is not a new production evaluation.

## Held-out presentation verification, 2026-11-17

The presentation generator now supports explicit `--include-test` inference
alongside validation, plus `--all-test` and test-only clip selection. Original
training selections and bundle metadata remain unchanged. Separate test metadata
must match the training locks and recorded reserved manifests, pass scene/content
separation checks, and bind complete decoded arrays to current scene receipts.
The selected checkpoint and protocol are saved before remote test metadata or
pixels are read. Output metrics and the presentation index label each split.

The handoff/R2-streaming/sampler-stability/joint-checkpoint/checkpoint-cadence
regression selection passed **66 tests** on CPU. After expanding representative,
all-clip and test-only coverage and adding a changed-reserved-manifest case, all
**22 handoff tests** passed. These focused runs use real tiny decoded frames and
a fake R2 transport. They verify inference with gradients disabled, reference-free
prediction, unchanged model/bundle checksums, equal adaptive/uniform integer
budgets, image/GIF outputs, empty spools, no training-pixel requests, explicit
test access, and rejection of incomplete/corrupt provenance before pixel reads.
The updated handoff's 11 Bash blocks and presentation guide's three Bash blocks
pass syntax checks; local Markdown links resolve and `git diff --check` passes.
No production Sponza inference or live R2 test download was performed for this
change; data availability and GPU throughput remain runtime checks.
