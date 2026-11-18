#!/usr/bin/env bash
# First rating run, used in the report as the replication (different seeds, overlapping tasks). One GPU.
# usage: bash run_replication.sh gptoss|qwen3|llama33     env: PY, REPLICATION_DIR (default ./results_replication)
set -euo pipefail
FAM=${1:?usage: run_replication.sh gptoss|qwen3|llama33}
HERE=$(cd "$(dirname "$0")" && pwd); source "$HERE/../../serve_vllm.sh"
PY=${PY:-python}; REPLICATION_DIR=${REPLICATION_DIR:-$PWD/results_replication}
OUT=$REPLICATION_DIR/results/$FAM; mkdir -p "$OUT"
case $FAM in gptoss) NAME=gpt-oss-120b ;; qwen3) NAME=qwen3-32b ;; llama33) NAME=llama-3.3-70b ;; esac
PORT=${PORT:-8700}; BASE=http://127.0.0.1:$PORT/v1
trap stop_vllm EXIT
start_vllm "$FAM" "$PORT" "$NAME" --enable-prefix-caching
"$PY" -u "$HERE/run_agents.py" --base-url "$BASE" --config "$FAM" --phase main --outdir "$OUT/main" --workers "${WORKERS:-32}"
"$PY" -I "$HERE/build_items.py" "$OUT/main"
"$PY" -u "$HERE/run_detached.py" --base-url "$BASE" --config "$FAM" --run-dir "$OUT/main" --workers "${WORKERS:-32}"
# afterwards: REPLICATION_DIR=$REPLICATION_DIR "$PY" -I destination_rule_gap.py <out dir>  (writes repairs/REPAIR_tables.md)
