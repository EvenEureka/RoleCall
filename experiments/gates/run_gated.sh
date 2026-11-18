#!/usr/bin/env bash
# Executed gates with the informed simulated user, for one acting model: complete AgentDojo episodes under no gate, the
# self-label gate, the reviewer gate and the rule on five outbound functions (same episode keys and seeds). One GPU.
# usage: bash run_gated.sh gptoss|qwen3|llama33      env: PY, GATES_RESULTS (default ./results_gates), VLLM, HF_HOME, LOGDIR
set -euo pipefail
FAM=${1:?usage: run_gated.sh gptoss|qwen3|llama33}
HERE=$(cd "$(dirname "$0")" && pwd); source "$HERE/../serve_vllm.sh"
PY=${PY:-python}; GATES_RESULTS=${GATES_RESULTS:-$PWD/results_gates}
case $FAM in gptoss) DW=64 ;; qwen3) DW=32 ;; llama33) DW=24 ;; esac
PORT=${PORT:-8600}; export ROLECALL_SERVED=rolecall-$FAM; BASE=http://127.0.0.1:$PORT/v1
trap stop_vllm EXIT
start_vllm "$FAM" "$PORT" "$ROLECALL_SERVED" --enable-prefix-caching
"$PY" -u "$HERE/run_gated.py" --base-url "$BASE" --config "$FAM" --outdir "$GATES_RESULTS/_smoke/informed/$FAM" --phase smoke --workers 16
"$PY" -u "$HERE/run_gated.py" --base-url "$BASE" --config "$FAM" --outdir "$GATES_RESULTS/informed/$FAM" --phase main --workers "$DW"
