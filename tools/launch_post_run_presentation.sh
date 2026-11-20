#!/usr/bin/env bash
# Run from an interactive terminal; credentials stay in process memory.
set -euo pipefail
set +x
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
source tools/activate.sh
if [[ "${1:-}" == --help ]]; then
    python tools/run_post_run_presentation.py --help
    exit 0
fi
if [[ -z "${R2_ACCESS_KEY_ID:-}" ]]; then
    read -rsp 'R2 Access Key ID: ' R2_ACCESS_KEY_ID
    printf '\n'
fi
if [[ -z "${R2_SECRET_ACCESS_KEY:-}" ]]; then
    read -rsp 'R2 Secret Access Key: ' R2_SECRET_ACCESS_KEY
    printf '\n'
fi
if [[ -z "${R2_ENDPOINT_URL:-}" && -z "${R2_ACCOUNT_ID:-}" ]]; then
    read -rp 'Cloudflare Account ID: ' R2_ACCOUNT_ID
fi
export R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY
[[ -z "${R2_ACCOUNT_ID:-}" ]] || export R2_ACCOUNT_ID
[[ -z "${R2_ENDPOINT_URL:-}" ]] || export R2_ENDPOINT_URL
mkdir -p "$QUALITY_RUN_ROOT"
launcher_log="$(mktemp "$QUALITY_RUN_ROOT/presentation-launcher.XXXXXX.log")"
nohup python -u tools/run_post_run_presentation.py "$@" \
    > "$launcher_log" 2>&1 < /dev/null &
presentation_pid=$!
disown "$presentation_pid"
printf 'Presentation PID: %s\nLauncher log: %s\n' "$presentation_pid" "$launcher_log"
printf 'Monitor: tail -f %q\n' "$launcher_log"
