#!/usr/bin/env bash
# Stance analyses after run_stance.sh has finished for all three models (CPU only).
#   1. frozen per-run tables (first attacker-directed call by stance, matched workloads)
#   2. merge the completion ratings, then twin separation and gate operating points on common populations
#   3. the post hoc crossover contrasts (needs the gates workload analysis, ../gates/analyze_all.sh, for its readout counts)
# usage: bash analyze_all.sh            env: PY, STANCE_DIR (default ./results_stance), GATES_RESULTS (default ./results_gates)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); PY=${PY:-python}
STANCE_DIR=${STANCE_DIR:-$PWD/results_stance}; GATES_RESULTS=${GATES_RESULTS:-$PWD/results_gates}
"$PY" -I "$HERE/analyze_stance.py" "$STANCE_DIR"                 # -> $STANCE_DIR/results/RESULT_tables.md, metrics.json
M=$STANCE_DIR/merged; rm -rf "$M"
for c in gptoss qwen3 llama33; do
  mkdir -p "$M/results/$c/main"
  cp "$STANCE_DIR/results/$c/main/episodes.jsonl" "$STANCE_DIR/results/$c/main/calls.jsonl" "$M/results/$c/main/"
  cat "$STANCE_DIR/results/$c/main/detached.jsonl" "$STANCE_DIR/complete/$c/detached.jsonl" > "$M/results/$c/main/detached.jsonl"
done
"$PY" -I "$HERE/recompute_gates.py" "$M" "$STANCE_DIR/recompute_gates_complete"
[ -f "$GATES_RESULTS/workload.json" ] && \
  "$PY" -I "$HERE/analyze_crossover.py" "$M" "$STANCE_DIR/recompute_gates_complete.json" "$GATES_RESULTS/workload.json" "$STANCE_DIR/crossover"
# replication gaps (first rating run; see replication/README in ../README.md):
#   REPLICATION_DIR=<dir> "$PY" -I replication/destination_rule_gap.py <out dir>
