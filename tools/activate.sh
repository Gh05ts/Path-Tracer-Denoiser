# Source from bash after creating .venv in a fresh clone.
_quality_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$_quality_repo/.venv/bin/activate"
export PYTHONPATH="$_quality_repo/src${PYTHONPATH:+:$PYTHONPATH}"
export QUALITY_REPO_ROOT="$_quality_repo"
export QUALITY_RUN_ROOT="${QUALITY_RUN_ROOT:-$(dirname "$_quality_repo")/runs}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
unset _quality_repo
