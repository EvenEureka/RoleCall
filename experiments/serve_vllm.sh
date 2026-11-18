#!/usr/bin/env bash
# Helper sourced by the run_*.sh scripts: start a vLLM (0.24) OpenAI-compatible server for one model and wait until it
# answers. One model per GPU; the original runs used one 96 GB GPU per model.
#
#   source experiments/serve_vllm.sh
#   start_vllm <model> <port> <served-name> [extra vllm serve arguments ...]
#   stop_vllm                       # stops every server started by this shell
#
# <model>: llama33 | gptoss | qwen3 | gemma3 | phi4. Weights come from the Hugging Face hub at the revisions used in the
# report (set HF_HOME / HF_HUB_OFFLINE=1 to use a local cache). VLLM can point to a specific vllm executable.
# Server logs go to $LOGDIR (default ./logs).

VLLM_PIDS=()

start_vllm() {
  local fam=$1 port=$2 name=$3; shift 3
  local model rev util=0.92 maxlen=32768 dtype=auto args=()
  case "$fam" in
    llama33) model=RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic; rev=f50dbad2c84590ca17dc51e207c34321b65ff14b; util=0.95
             args=(--enable-auto-tool-choice --tool-call-parser llama3_json) ;;          # bf16 KV cache (no fp8 KV)
    gptoss)  model=openai/gpt-oss-120b; rev=b5c939de8f754692c1647ca79fbf85e8c1e70f8a
             args=(--enable-auto-tool-choice --tool-call-parser openai --reasoning-parser openai_gptoss) ;;
    qwen3)   model=Qwen/Qwen3-32B; rev=9216db5781bf21249d130ec9da846c4624c16137
             args=(--enable-auto-tool-choice --tool-call-parser hermes) ;;
    gemma3)  model=unsloth/gemma-3-27b-it; rev=7a5a3053dbd5d1d58e48159e87b9df2fc545a49a; util=0.90; maxlen=8192; dtype=bfloat16
             args=(--enable-prefix-caching --limit-mm-per-prompt '{"image":0}') ;;     # simulated user
    phi4)    model=microsoft/phi-4; rev=2db69c1c3e91a05d2c64a3185acfbaf36f744e25; util=0.90; maxlen=8192; dtype=bfloat16
             args=(--enable-prefix-caching) ;;                                          # simulated user
    *) echo "unknown model $fam" >&2; return 1 ;;
  esac
  mkdir -p "${LOGDIR:-logs}"
  "${VLLM:-vllm}" serve "$model" --revision "$rev" --served-model-name "$name" --host 127.0.0.1 --port "$port" \
    --tensor-parallel-size 1 --gpu-memory-utilization "$util" --max-model-len "$maxlen" --dtype "$dtype" \
    "${args[@]}" "$@" > "${LOGDIR:-logs}/vllm.$name.log" 2>&1 &
  local pid=$!
  VLLM_PIDS+=("$pid")
  for _ in $(seq 1 720); do                       # up to one hour for the first download / compile
    curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && { echo "vLLM $name ready on port $port"; return 0; }
    kill -0 "$pid" 2>/dev/null || { echo "vLLM $name exited; see ${LOGDIR:-logs}/vllm.$name.log" >&2; return 1; }
    sleep 5
  done
  echo "vLLM $name not ready after one hour" >&2
  return 1
}

stop_vllm() {
  for p in "${VLLM_PIDS[@]}"; do kill "$p" 2>/dev/null || true; done
  VLLM_PIDS=()
}
