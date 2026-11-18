<!-- TOP: written by parent -->

## Repository map

```
RoleCall/
  README.md                  this file
  LICENSE                    MIT
  requirements.txt           what reproduce/ needs (numpy; everything else is the Python standard library)
  reproduce/
    headline_numbers.py      recomputes the report's numbers from data/ and checks them against the report
    expected_output.txt      its output without intervals
    expected_output_ci.txt   its output with --ci (bootstrap intervals and interval-based readouts)
  data/                      compact per-episode, per-branch and per-call tables (gzipped CSV, under 0.5 MB in total)
    headline.json            every number the script computes, with intervals (written with --ci --json)
  experiments/
    README.md                terms, software versions, models, decoding, how data/ was cut from the raw records
    serve_vllm.sh            starts one vLLM server per model (exact model revisions and flags)
    build_released_data.py   raw run records -> data/
    role/                    Section 3: which chat role a tool output arrives in
    stance/                  Section 4: who rates the risk of a proposed call
    gates/                   Section 5: confirmation gates executed through complete episodes
```

## Reproduce the numbers (no GPU)

You need Python 3.9 or newer and numpy.

```bash
pip install -r requirements.txt
python reproduce/headline_numbers.py          # point estimates, a few seconds
python reproduce/headline_numbers.py --ci     # also the 95% bootstrap intervals, a few seconds more
python reproduce/headline_numbers.py --ci --json data/headline.json   # rewrite the JSON summary
```

The script reads only the tables in `data/`. Every number is computed from rows (one per episode, branch or rated
call), not copied from the report. Each output line has four parts:

```
  [  ok] +29.3 [17.7, 42.2]       report +29.3 [17.7, 42.2]      change in ASR, default minus tool role (points); ...
  status value [95% interval]     value printed in the report    what the number means
```

- `ok`: matches the report at the precision the report prints (intervals too, where the report prints them).
- `DIFF`: does not match. The script exits with code 1 if any line says `DIFF`.
- `info`: computed but not printed in the report (for example a family the report summarises as a range).
- `n/a`: cannot be computed from the released data; the line says why.

Intervals use the report's method: a cluster bootstrap that resamples whole user tasks, with the same resampling
scheme, number of resamples and random seed as the original analysis, so the intervals match to the printed digit.

**Result on the released data:** 219 of 219 checked numbers match the report (`--ci` run), 22 further numbers are
printed for information, and nothing differs. See `reproduce/expected_output_ci.txt`.

**Covered:** Table 1 (both halves), the relabelling numbers of Section 3, Table 2, the first-call ratings and gaps of
Section 4 including the replication and the post hoc crossover contrasts, all three blocks of Table 3, and the
fallible-user and equal-interruption numbers in the text of Section 5.

**Not covered:** gpt-oss-120b in the LlamaIndex agent (the report says this configuration gave no usable runs, so there
are no records; printed as `n/a`). Numbers outside the report's main line that the raw analyses also produced (for
example the next-step sampling at temperature 0.7, the exploratory raw-string and `input`-role relabelling arms, and
an exploratory 8B model) are not in the released tables.

## Rerun the experiments (GPUs)

The experiments need GPUs that can hold each model: the original runs used one 96 GB GPU per model, and two GPUs for
the runs with an LLM user. Install two Python 3.11 environments: one with AgentDojo (`agentdojo==0.1.35`, which
ships benchmark version `v1.2.2`) and one with AgentDojo plus `llama-index-core==0.14.25`,
`llama-index-llms-openai-like==0.8.1` and `smolagents==1.26.0` for the installed agents; and vLLM 0.24.0. The
experiment folders explain each run; in short:

```bash
export PY=/path/to/agentdojo-env/bin/python VLLM=/path/to/vllm-env/bin/vllm
cd experiments/role   && bash run_relabel.sh llama33 && bash run_branching.sh llama33 \
                      && PY=/path/to/frameworks-env/bin/python bash run_installed.sh llama33
cd experiments/stance && bash run_stance.sh qwen3 && bash analyze_all.sh
cd experiments/gates  && bash run_gated.sh qwen3 && bash run_fallible.sh qwen3 gemma3 standard && bash analyze_all.sh
```

Each script serves the model with vLLM, runs a short smoke phase, then the main run, and writes full transcripts.
`experiments/build_released_data.py` turns those transcripts into the tables in `data/`. The rated and gated runs
sample (temperature above 0), so a rerun gives new episodes; expect the numbers to move within their intervals, not to
repeat to the digit. The role experiments decode greedily but still depend on the vLLM version and GPU kernels.

## Data

All files are gzipped CSV with a header row. `1`/`0` mean true/false; an empty cell means "does not apply" (for
example no reviewer rating was requested for that call). Model names are `llama-3.3-70b`, `gpt-oss-120b` and
`qwen3-32b`. AgentDojo task ids (`user_task_7`, `injection_task_3`) refer to AgentDojo `v1.2.2`.

| File | Rows | One row is | Used for |
|---|---|---|---|
| `role_installed_episodes.csv.gz` | 3,000 | one episode of an installed framework agent | Table 1, top |
| `role_branching.csv.gz` | 1,809 | one continued branch (or one pair that never reached the injection) | Table 1, bottom |
| `role_relabel_episodes.csv.gz` | 4,482 | one attacked episode of the relabelling experiment | Section 3 text |
| `stance_episodes.csv.gz` | 2,151 | one episode of the stance experiment (717 per model) | Section 4, Table 2 |
| `stance_calls.csv.gz` | 8,894 | one proposed call in the final attempt of a stance episode, with its four ratings | Section 4, Table 2, Table 3 bottom |
| `stance_replication_calls.csv.gz` | 4,770 | one proposed call of the replication run | Section 4 text |
| `gates_episodes.csv.gz` | 34,416 | one episode of one gate, user and model (717 keys x 4 gates x 3 models x 4 users) | Table 3, Section 5 text |
| `headline.json` | | every recomputed number, nested by section; values in percent or points, `*_ci95` keys are intervals | website check |

Columns that need a word of explanation:

- `kind`: `attacked` / `benign` / `feasibility` (the injection goal run as a normal user task) in the role tables;
  `injected` / `twin` / `benign` in the stance and gate tables. A *twin* is the same user task with the injection
  goal appended to the user's own request and no injection, so the same action is authorised.
- `delivery_role`: `user` (the framework default, or the relabelled arm) or `tool` (the tool-role option).
- `attack_success`, `goal_reached`: AgentDojo's security check, true when the injection goal was reached. In a twin
  episode `goal_reached` means the requested goal was completed (*authorised completion*).
- `error`, `exception`: the episode ended with a recorded exception and is excluded, as in the report.
- `prefix_status`, `branch_step`: `branch` if the episode reached an injected tool output (then the step at which it
  branched), `ended_before_injection` otherwise. `injection_voice`: `user_authored` (AgentDojo's template, signed by the
  user) or `unattributed`. `header`: the role of the injected message only. `replay_identical`: the injection-free
  history replayed byte for byte.
- `goal_class`: `sensitive_data` (goals that send the user's data out), `money_only`, `non_sensitive`.
- `call_set`: `AU` attacker-directed call in an injected episode; `AA` the goal's send in a twin; `B` call in a benign
  episode; `C` other call after the injection was read; `P` other call before it (or a twin's other calls).
  `first_in_episode`: the first `AU`/`AA` call of the episode.
- `label_acting`, `label_asked`, `label_reviewing`, `label_reviewing_no_tool_outputs`: LOW / MEDIUM / HIGH; `OMITTED`
  when the acting model left the label out; `UNPARSED` when a rater's answer had no label.
- `gate`: `none`, `inline` (self-label gate), `l4` (reviewer gate), `out5` (rule on five outbound functions).
  `user`: `informed`, `gemma_standard`, `gemma_cautious`, `phi4_standard`.
- `confirmations`: how many times the gate asked the user. `attacker_asks` / `attacker_asks_approved`: requests about
  attacker-directed calls and how many the user approved (fallible users only). `other_asks` /
  `other_asks_rejected`: the remaining requests and how many the user rejected. `nonstandard_answers`: answers that did
  not start with APPROVE or REJECT. `reviewer_inferences`: reviewer calls made by the `l4` gate in that episode.

**Not released:** the full transcripts (model outputs, tool outputs and every HTTP request, several GB). They are not
needed for any number in the report, and they contain long model generations; the tables above keep the outcome of
every episode and every rating the analyses use.

## License

The code and the data in this repository are released under the MIT License (see `LICENSE`). The task ids, injection
goals and attacker destinations come from AgentDojo, which is MIT-licensed. `experiments/stance/oh_patch.py` quotes the
`security_risk` schema text and risk policy of the OpenHands software-agent-sdk (commit `acd96e6f`); see that project
for its license. The model weights are not included and keep their own licenses.
