#!/usr/bin/env bash
# Run on the SSH host; nohup survives a disconnected terminal, not a host reboot.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
entry="$here/generate_dataset.py"
if [[ "${1:-}" == "r2" ]]; then
    entry="$here/generate_r2_dataset.py"
    shift
elif [[ "${1:-}" == "reports" ]]; then
    entry="$here/generate_report_references.py"
    shift
elif [[ "${1:-}" == "train" ]]; then
    entry="$here/train_dataset.py"
    shift
fi
python_bin="${DATASET_PYTHON:-python3}"
dataset_workspace="$(cd -- "$here/../.." && pwd)"
log_dir="$dataset_workspace/dataset/jobs"
mkdir -p "$log_dir"
job_name="$(date -u +%Y%m%dT%H%M%SZ)-$$"
log_file="$log_dir/$job_name.log"
# A separate session also survives an execution service cleaning up the
# launching shell's process group. nohup alone only handles SIGHUP.
detach=()
if command -v setsid > /dev/null 2>&1; then
    detach=(setsid)
fi
nohup "${detach[@]}" "$python_bin" -u "$entry" "$@" >"$log_file" 2>&1 < /dev/null &
job_pid=$!
printf '%s\n' "$job_pid" > "$log_dir/$job_name.pid"
printf 'Started PID %s\nLog: %s\nMonitor: tail -f "%s"\n' "$job_pid" "$log_file" "$log_file"
