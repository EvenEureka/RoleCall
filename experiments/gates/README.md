# Gates: what confirmation gates prevent and interrupt in complete episodes

**Question.** A *confirmation gate* pauses a proposed tool call and asks the user to approve it. A rejected call is
not executed; the agent sees OpenHands' rejection text and continues. The stance experiment measured ratings in
isolation. Here the gates run inside complete AgentDojo episodes, so we see what they prevent and what they cost:

| Gate | Name in code | Asks the user when |
|---|---|---|
| No gate | `none` | never |
| Self-label gate | `inline` | the acting model's own `security_risk` label is HIGH, missing, unknown or invalid (OpenHands `ConfirmRisky` at HIGH) |
| Reviewer gate | `l4` | a fresh reviewer (same weights, same policy) rates the call HIGH, or its rating is unknown, unparsed or fails |
| Rule | `out5` | the call uses one of five outbound functions: `send_money`, `schedule_transaction`, `update_scheduled_transaction`, `send_email`, `post_webpage` |

Every gate runs the same 717 episode keys per model (310 injected, 310 authorised twins, 97 benign) with the same
seeds, so episodes can be compared one to one across gates.

**Users.** The *informed* simulated user rejects exactly the attacker-directed calls it is shown. That is the best
case for every gate. The *fallible* users are LLMs that see only the user's request and the proposed call and answer
APPROVE or REJECT: Gemma-3-27B (`gemma_standard`), Gemma-3-27B with a stricter approval sentence (`gemma_cautious`),
and Phi-4 (`phi4_standard`). An answer that is not a decision counts as an approval.

**Equal interruption.** A gate that asks more often also catches more. To compare two gates fairly, the one that
interrupts more is mixed with no gate per episode (with a fixed probability) until both interrupt the same share of
legitimate episodes, then their attack success is compared (`analyze_workload.py`). "Legitimate" is weighted four
ways: twins only, benign only, both pooled, and both weighted equally.

## Files

| File | Role |
|---|---|
| `gated.py` | the four gates and the informed user inside AgentDojo's tool loop |
| `run_gated.py` | runner for the informed user (reuses `../stance/` for the agent, tasks and schema patch) |
| `gated_fallible.py`, `run_fallible.py` | the LLM user (first round, Gemma-3-27B, standard prompt) |
| `gated_fallible2.py`, `run_fallible2.py` | the same with a choice of user prompt (second round: stricter Gemma, Phi-4) |
| `mock_user.py` | CPU-only mock user server for plumbing tests |
| `analyze_gated.py` | Table 3, top: ASR, utility, authorised completion and interruptions per gate (informed user) |
| `analyze_fallible.py` | the same per fallible user, plus the users' false approvals and false rejections |
| `analyze_workload.py` | Table 3, middle: equal-interruption comparisons for the four users and four workloads |
| `analyze_matched_cost.py` | the cost of the reviewer at the equal rate (authorised completion, extra inferences) |
| `diag_user_answers.py` | how the users' answers were read (Phi-4's non-decisions) |
| `run_gated.sh`, `run_fallible.sh`, `analyze_all.sh` | run scripts |

## How it was run

The acting model on one GPU, the user model on a second GPU, both served with vLLM 0.24 (`../serve_vllm.sh`);
AgentDojo package 0.1.35, benchmark `v1.2.2`; the same decoding settings as the stance experiment. The user models
answer greedily with at most 8 tokens. The parser takes the first word if it is APPROVE or REJECT, otherwise the first
such word anywhere in the answer, and retries once; an answer it cannot read counts as an approval.

```bash
export PY=/path/to/agentdojo-env/bin/python VLLM=/path/to/vllm-env/bin/vllm
bash run_gated.sh qwen3                         # informed user; also gptoss, llama33
bash run_fallible.sh qwen3 gemma3 standard      # Gemma-3-27B
bash run_fallible.sh qwen3 gemma3 cautious      # stricter Gemma
bash run_fallible.sh qwen3 phi4 standard        # Phi-4
bash analyze_all.sh
```

Statistics: cluster bootstrap over (suite, user task), 97 clusters, 5,000 resamples, seed 20261024, percentile
intervals (`../stance/stats_common.py`). A gate "reduces ASR" when the 95% interval of its difference from no gate
lies below zero.

## Released data

`data/gates_episodes.csv.gz`: one row per (user, model, gate, episode) with the outcome, the number of confirmation
requests, how the user answered them, and the number of reviewer inferences.
