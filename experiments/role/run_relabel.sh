#!/usr/bin/env bash
# Relabelling experiment for one model: every tool output delivered in the tool role or relabelled as a user message,
# under three AgentDojo attack templates. One GPU.
# usage: bash run_relabel.sh llama33|qwen3|gptoss
# env:   PY (python of the AgentDojo environment), RELABEL_DIR (default ./results_relabel), VLLM, HF_HOME, LOGDIR
set -euo pipefail
FAM=${1:?usage: run_relabel.sh llama33|qwen3|gptoss}
HERE=$(cd "$(dirname "$0")" && pwd); source "$HERE/../serve_vllm.sh"
PY=${PY:-python}; export RELABEL_DIR=${RELABEL_DIR:-$PWD/results_relabel}
OUT=$RELABEL_DIR/results/$FAM; mkdir -p "$OUT"
case $FAM in llama33) NAME=llama-3.3-70b; MAXTOK=2048 ;; qwen3) NAME=qwen3-32b; MAXTOK=2048 ;; gptoss) NAME=gpt-oss-120b; MAXTOK=4096 ;; esac
PORT=${PORT:-8100}
trap stop_vllm EXIT
start_vllm "$FAM" "$PORT" "$NAME"
"$PY" "$HERE/relabel_plans.py"
COMMON=(--base-url "http://127.0.0.1:$PORT/v1" --model "$NAME" --family "$FAM" --plan "$HERE/plans/$FAM.json"
        --workers "${WORKERS:-32}" --max-tokens "$MAXTOK")
"$PY" -u "$HERE/relabel_run.py" "${COMMON[@]}" --phase smoke --outdir "$OUT/smoke"
"$PY" -u "$HERE/relabel_run.py" "${COMMON[@]}" --phase main --outdir "$OUT/main"
# analysis (after all three models): python3 -I relabel_analyze.py relabel.md relabel.json
