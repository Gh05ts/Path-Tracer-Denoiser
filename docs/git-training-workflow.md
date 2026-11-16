# Git source workflow for Lambda

Use `https://github.com/Gh05ts/Path-Tracer-Denoiser.git` as the shared origin.
The active editing branch on Lambda is `benchmark/a100`. Keep source edits in
this checkout; preserve the older `h100-ready-pilot` directory for its existing
environment, data, checkpoints and reports. Do not edit the same branch in two
clones at once.

From this checkout's root, prepare each terminal:

```bash
source /home/ubuntu/workspace/h100-ready-pilot/.venv/bin/activate
export PYTHONPATH="$PWD/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -c 'import wavefront_denoiser; print(wavefront_denoiser.__file__)'
git rev-parse HEAD
git status --short --branch
```

The import must resolve inside this checkout. Reuse the installed GPU PyTorch;
do not reinstall it to change the source location.

`configs/train-ready-pilot-lambda.json` uses the existing absolute decoded-data
path. All other settings match the pilot, including the explicit `1e-6`
radiance-gradient floor. This config starts a new experiment: changing the
config/data paths is not an exact resume of an older run.

Before a run, choose a fresh absolute output directory outside Git. Save the
commit, branch, status, tracked/untracked source changes, exact config, dataset
inventory digest, `pip freeze`, `nvidia-smi`, random seed and execution policies
alongside the result. Use the existing trainer's inventory/config records as
well. Review untracked files explicitly; `git diff` alone does not capture them.
For example, after recording provenance:

```bash
python -m wavefront_denoiser train-adaptive \
  configs/train-ready-pilot-lambda.json \
  /home/ubuntu/workspace/h100-ready-pilot/runs/git-pilot-001 --device cuda
python tools/report_joint_pilot.py \
  /home/ubuntu/workspace/h100-ready-pilot/runs/git-pilot-001
```

Check GPU availability first. Do not edit or pull the source during a run; use
a separate worktree for concurrent development. The sample output path must be
changed if it already exists.

After local review/tests, commit on Lambda. Push only when explicitly requested.
Once published, inspect the original project clone's working tree, preserve any
local edits, fetch `origin`, and check out `benchmark/a100` tracking the same
remote branch. Update a clean tracking checkout with `git pull --ff-only`.
Review and merge through the project's usual workflow; never reset or force
push to conceal divergence. Remote SSH edits the remote checkout directly and
does not synchronize Git branches automatically.

The initial Lambda checkout was imported through the connected GitHub service
because shell DNS was unavailable. Every blob, tree and the original commit
were verified against Git object SHA-1s; the repository is shallow at
`435c3331b189b41ddc42c9e1077ac727148b466b`. Its normal Git origin is retained.
When shell networking is available, `git fetch --unshallow origin` retrieves
the earlier history. No push is needed for integration or local runs.
