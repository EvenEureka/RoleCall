#!/usr/bin/env bash
# Gate analyses after all runs (CPU only). $GATES_RESULTS/<user>/<model>/<gate>/episodes.jsonl for the four users.
# usage: bash analyze_all.sh        env: PY, GATES_RESULTS (default ./results_gates)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); PY=${PY:-python}; G=${GATES_RESULTS:-$PWD/results_gates}
"$PY" -I "$HERE/analyze_gated.py" "$G/informed" "$G/gated"                                   # Table 3, top
"$PY" -I "$HERE/analyze_fallible.py" "$G/informed" "$G/informed_reference" --informed
for U in gemma_standard phi4_standard gemma_cautious; do
  "$PY" -I "$HERE/analyze_fallible.py" "$G/$U" "$G/$U"                                        # fallible users
done
USERS=(informed="$G/informed" gemma_standard="$G/gemma_standard" phi4_standard="$G/phi4_standard" gemma_cautious="$G/gemma_cautious")
"$PY" -I "$HERE/analyze_workload.py" "$G/workload" "${USERS[@]}"                              # Table 3, middle
"$PY" -I "$HERE/analyze_matched_cost.py" "$G/matched_cost" "$G/workload.json" "${USERS[@]}"   # cost at the equal rate
GATES_RESULTS=$G "$PY" -I "$HERE/diag_user_answers.py" "$G/user_answer_check.md"
