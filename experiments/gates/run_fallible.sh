#!/usr/bin/env bash
# Executed gates with a fallible LLM user that sees only the request and the proposed call. Two GPUs: the acting model
# on GPU 0, the user model on GPU 1.
# usage: bash run_fallible.sh gptoss|qwen3|llama33 gemma3|phi4 standard|cautious
#   the report's users: gemma3 standard (Gemma-3-27B), gemma3 cautious (stricter approval sentence), phi4 standard
# env:   PY, GATES_RESULTS (default ./results_gates), VLLM, HF_HOME, LOGDIR
set -euo pipefail
FAM=${1:?acting model}; UMODEL=${2:?gemma3|phi4}; UPROMPT=${3:-standard}
HERE=$(cd "$(dirname "$0")" && pwd); source "$HERE/../serve_vllm.sh"
PY=${PY:-python}; GATES_RESULTS=${GATES_RESULTS:-$PWD/results_gates}
case $UMODEL in gemma3) USER_LABEL=gemma_$UPROMPT ;; phi4) USER_LABEL=phi4_$UPROMPT ;; *) echo "unknown user model"; exit 1 ;; esac
case $FAM in gptoss) DW=64 ;; qwen3) DW=32 ;; llama33) DW=24 ;; esac
PORT=${PORT:-8600}; UPORT=$((PORT + 700)); export ROLECALL_SERVED=rolecall-$FAM; UNAME=rolecall-user
trap stop_vllm EXIT
export CUDA_VISIBLE_DEVICES=0; start_vllm "$FAM" "$PORT" "$ROLECALL_SERVED" --enable-prefix-caching   # acting model
export CUDA_VISIBLE_DEVICES=1; start_vllm "$UMODEL" "$UPORT" "$UNAME"                                   # user model
unset CUDA_VISIBLE_DEVICES
USERARGS=(--user-base-url "http://127.0.0.1:$UPORT/v1" --user-model "$UNAME")
if [ "$USER_LABEL" = gemma_standard ]; then
  RUN=("$HERE/run_fallible.py")                        # first fallible-user round
else
  RUN=("$HERE/run_fallible2.py" --user-prompt "$UPROMPT")   # second round (same code path, prompt chosen by flag)
fi
"$PY" -u "${RUN[@]}" --base-url "http://127.0.0.1:$PORT/v1" "${USERARGS[@]}" --config "$FAM" \
  --outdir "$GATES_RESULTS/_smoke/$USER_LABEL/$FAM" --phase smoke --workers 16
"$PY" -u "${RUN[@]}" --base-url "http://127.0.0.1:$PORT/v1" "${USERARGS[@]}" --config "$FAM" \
  --outdir "$GATES_RESULTS/$USER_LABEL/$FAM" --phase main --workers "$DW"
