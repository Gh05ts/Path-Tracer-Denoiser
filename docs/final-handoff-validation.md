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
