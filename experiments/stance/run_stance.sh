#!/usr/bin/env bash
# Stance experiment for one model: AgentDojo episodes with the OpenHands-style security_risk label written while acting,
# then the same calls rated when asked in the same conversation (L2), by a fresh reviewer (L4) and by a reviewer that
# does not see tool outputs (L4blind). One GPU.
# usage: bash run_stance.sh gptoss|qwen3|llama33        (RW=1 also runs the reviewer-wording actor, not in the report)
# env:   PY (python of the AgentDojo environment), STANCE_DIR (default ./results_stance), VLLM, HF_HOME, LOGDIR
set -euo pipefail
FAM=${1:?usage: run_stance.sh gptoss|qwen3|llama33}
HERE=$(cd "$(dirname "$0")" && pwd); source "$HERE/../serve_vllm.sh"
PY=${PY:-python}; STANCE_DIR=${STANCE_DIR:-$PWD/results_stance}
OUT=$STANCE_DIR/results/$FAM; mkdir -p "$OUT"
case $FAM in gptoss) W=48; DW=64 ;; qwen3) W=32; DW=32 ;; llama33) W=24; DW=24 ;; esac
PORT=${PORT:-8600}; export ROLECALL_SERVED=rolecall-$FAM; BASE=http://127.0.0.1:$PORT/v1
trap stop_vllm EXIT
start_vllm "$FAM" "$PORT" "$ROLECALL_SERVED" --enable-prefix-caching
# smoke: a few episodes through every stage, checked for parse health only
"$PY" -u "$HERE/run_agents.py" --base-url "$BASE" --config "$FAM" --phase smoke --outdir "$OUT/smoke" --workers 4
"$PY" -I "$HERE/build_items.py" "$OUT/smoke"
"$PY" -u "$HERE/run_detached.py" --base-url "$BASE" --config "$FAM" --run-dir "$OUT/smoke" --workers 16
"$PY" -I "$HERE/smoke_check.py" "$OUT/smoke"
# main: 310 injected + 310 authorised-twin + 97 benign episodes, then the ratings of their calls
"$PY" -u "$HERE/run_agents.py" --base-url "$BASE" --config "$FAM" --phase main --outdir "$OUT/main" --workers "$W"
"$PY" -I "$HERE/build_items.py" "$OUT/main"
"$PY" -u "$HERE/run_detached.py" --base-url "$BASE" --config "$FAM" --run-dir "$OUT/main" --workers "$DW"
if [ "${RW:-0}" = 1 ]; then
  "$PY" -u "$HERE/run_agents.py" --base-url "$BASE" --config "$FAM" --phase rw --outdir "$OUT/rw" --workers "$W"
  "$PY" -I "$HERE/build_items.py" "$OUT/rw"
  "$PY" -u "$HERE/run_detached.py" --base-url "$BASE" --config "$FAM" --run-dir "$OUT/rw" --arms L4,L2 --workers "$DW"
fi
# rating completion: the reviewer (L4) rating of every benign call the main pass left unrated
COMP=$STANCE_DIR/complete/$FAM
"$PY" -I "$HERE/build_complete_items.py" "$OUT/main" "$COMP"
for pass in 1 2; do   # pass 2 retries anything that failed in pass 1 (the runner skips done items)
  "$PY" -u "$HERE/run_detached.py" --base-url "$BASE" --config "$FAM" --run-dir "$COMP" --arms L4 --workers "$DW"
done
