#!/usr/bin/env bash
# Identical-history branching for one model: run to the first injected tool output, then continue once per
# (injection voice x header of that one message). One GPU.
# usage: bash run_branching.sh llama33|gptoss
# env:   PY (python of the AgentDojo environment), BRANCH_DIR (default ./results_branching), VLLM, HF_HOME, LOGDIR
set -euo pipefail
FAM=${1:?usage: run_branching.sh llama33|gptoss}
HERE=$(cd "$(dirname "$0")" && pwd); source "$HERE/../serve_vllm.sh"
PY=${PY:-python}; BRANCH_DIR=${BRANCH_DIR:-$PWD/results_branching}
OUT=$BRANCH_DIR/results/$FAM; mkdir -p "$OUT"
case $FAM in llama33) NAME=llama-3.3-70b; MAXTOK=2048 ;; gptoss) NAME=gpt-oss-120b; MAXTOK=4096 ;; esac
PORT=${PORT:-8100}
trap stop_vllm EXIT
start_vllm "$FAM" "$PORT" "$NAME"
COMMON=(--base-url "http://127.0.0.1:$PORT/v1" --model "$NAME" --family "$FAM" --workers "${WORKERS:-32}" --max-tokens "$MAXTOK"
        --heartbeat "$OUT/heartbeat" --watchdog-min 20)
"$PY" -u "$HERE/branch_run.py" "${COMMON[@]}" --phase smoke --outdir "$OUT/smoke/branch" --samples 1
"$PY" -u "$HERE/branch_run.py" "${COMMON[@]}" --phase main --outdir "$OUT/main/branch" --stages prefix,branch
"$PY" -u "$HERE/branch_run.py" "${COMMON[@]}" --phase main --outdir "$OUT/main/branch" --stages samples --samples 3
# analysis (after both models): python3 -I branch_analyze.py "$BRANCH_DIR" branching.md branching.json
