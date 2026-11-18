#!/usr/bin/env bash
# Installed LlamaIndex ReAct and smolagents ToolCallingAgent agents for one model, framework default (tool outputs as
# user messages) vs the framework's tool-role option. One GPU.
# usage: bash run_installed.sh llama33|qwen3|gptoss
# env:   PY (python of the environment with AgentDojo + llama-index + smolagents), INSTALLED_RESULTS (default
#        ./results_installed), VLLM, HF_HOME, LOGDIR
set -euo pipefail
FAM=${1:?usage: run_installed.sh llama33|qwen3|gptoss}
HERE=$(cd "$(dirname "$0")" && pwd); source "$HERE/../serve_vllm.sh"
PY=${PY:-python}; export INSTALLED_RESULTS=${INSTALLED_RESULTS:-$PWD/results_installed}
OUT=$INSTALLED_RESULTS/$FAM; mkdir -p "$OUT"
export SMOLAGENTS_TELEMETRY=false HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1
MAXTOK=2048; ARMS=NLIu,NLIt,NSAu,NSAt
# gpt-oss-120b: the LlamaIndex ReAct configuration gave no usable runs (registered contingency), so smolagents only
[ "$FAM" = gptoss ] && { MAXTOK=4096; ARMS=NSAu,NSAt; }
PORT=${PORT:-8100}; NAME=installed-$FAM
trap stop_vllm EXIT
start_vllm "$FAM" "$PORT" "$NAME"
COMMON=(--base-url "http://127.0.0.1:$PORT/v1" --model "$NAME" --family "$FAM" --arms "$ARMS" --workers "${WORKERS:-32}"
        --max-tokens "$MAXTOK" --heartbeat "$OUT/heartbeat" --watchdog-min 30)
"$PY" -u "$HERE/installed_run.py" "${COMMON[@]}" --pairs smoke --outdir "$OUT/smoke"
python3 -I "$HERE/installed_smoke_check.py" "$OUT/smoke"
"$PY" -u "$HERE/installed_run.py" "${COMMON[@]}" --pairs all --outdir "$OUT/main"
# one resume pass: AgentDojo reuses the logs of finished episodes, so only episodes that ended in an exception rerun
"$PY" -u "$HERE/installed_run.py" "${COMMON[@]}" --pairs all --outdir "$OUT/main"
NATIVE_FAMILY=$FAM python3 -I "$HERE/installed_analyze.py"
