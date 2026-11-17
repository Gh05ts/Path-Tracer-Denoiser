#!/usr/bin/env bash
set -euo pipefail
source /home/ubuntu/quality_workspace/activate.sh
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
exec python -u tools/run_quality_pilot.py "$@"
