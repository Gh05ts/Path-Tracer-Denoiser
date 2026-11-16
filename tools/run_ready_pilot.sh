#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
pilot_python="${PILOT_PYTHON:-python3}"
pilot_output="${1:-runs/ready-pilot-h100}"
if [[ -e "$pilot_output" || -e "$pilot_output.log" ]]; then
  printf 'Choose a new output path; existing run or log: %s\n' "$pilot_output" >&2
  exit 1
fi
mkdir -p "$(dirname "$pilot_output")"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
nvidia-smi > "$pilot_output.hardware.txt"
"$pilot_python" -u -m wavefront_denoiser train-adaptive configs/train-ready-pilot.json "$pilot_output" --device cuda 2>&1 | tee "$pilot_output.log"
"$pilot_python" tools/report_joint_pilot.py "$pilot_output"
