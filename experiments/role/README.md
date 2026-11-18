# Role: which chat role a tool output arrives in

**Question.** A chat model sees a conversation as a list of messages, and each message has a *role*: `system`,
`user`, `assistant`, or `tool` (the result of a function the agent called). When the agent reads an e-mail that
contains a planted instruction, the harness decides which role that text arrives in. Some frameworks send tool
results as `user` messages by default. Does that choice change how often the same model follows the planted
instruction?

Three experiments answer it, from most realistic to most controlled. All use AgentDojo's banking and Slack suites:
249 attacked pairs (16 banking user tasks x 9 injection goals, 21 Slack user tasks x 5 goals) from 37 user tasks.

## 1. Installed frameworks (report Table 1, top)

The real LlamaIndex `ReActAgent` and smolagents `ToolCallingAgent`, installed from PyPI, drive the episode. Each
AgentDojo function is wrapped as a framework tool. Each framework runs twice: once with its default (tool results go
in as `user` messages) and once with its documented tool-role option.

| Arm | Framework | Where tool results go |
|---|---|---|
| `NLIu` | LlamaIndex 0.14.25 ReAct | `user` role (default `observation_role`) |
| `NLIt` | LlamaIndex 0.14.25 ReAct | `tool` role (`observation_role=TOOL`) |
| `NSAu` | smolagents 1.26.0 ToolCallingAgent | `user` role (default `TOOL_RESPONSE -> user` conversion) |
| `NSAt` | smolagents 1.26.0 ToolCallingAgent | `tool` role (`custom_role_conversions`) |

Files: `installed_agents.py` (the framework agents as one AgentDojo pipeline element), `installed_run.py` (runner),
`installed_smoke_check.py` (checks on the wire that every observation carries the arm's role),
`installed_analyze.py` (frozen analysis), `run_installed.sh` (one model, one GPU).
Models: Llama-3.3-70B, Qwen3-32B and gpt-oss-120b. For gpt-oss-120b the LlamaIndex ReAct configuration on vLLM gave
no usable runs, so only the smolagents arms were run (a registered contingency).
Released data: `data/role_installed_episodes.csv.gz`.

## 2. Branching at the injected message (report Table 1, bottom)

A framework switch relabels *every* tool result, so the conversation can already differ before the injection is
read. Branching removes that difference. Each attacked pair first runs with the model's native tool channel until the
first tool result that contains the injection (the *prefix*). The run stops before the model reads it. Then the
pair continues four times from the same saved history: the injected message gets a `tool` or a `user` header, and the
injection is either *user-authored* (AgentDojo's `important_instructions` text, signed by the user) or *unattributed*
(the same text with the first-person authorship removed). Every other message and byte is unchanged; the replayed
injection-free tool results are checked to be byte-identical.

Files: `branch_run.py` (prefix, branch and sample stages), `branch_llms.py` (the two branching model wrappers, which
change only the header of the one injected message), `branch_voice.py` (the two injection voices and the
attacker-action detector), `branch_fake_client.py` (a CPU-only scripted client for plumbing tests),
`branch_analyze.py` (frozen analysis), `run_branching.sh`. Models: Llama-3.3-70B (text tool calls, AgentDojo's
`local` prompt) and gpt-oss-120b (native tool calls).
Released data: `data/role_branching.csv.gz`. Main endpoint: *continued ASR*, AgentDojo's attack success after the
continued episode; also the next action alone.

## 3. Relabelling every observation under three templates (report Section 3, last paragraph)

The same episode runs with every tool output either in the `tool` role or relabelled as a `user` message, with the
rendered bytes unchanged where the chat template allows it. This runs under three AgentDojo attack templates:
`important_instructions`, `ignore_previous` and `direct`.

| Arm (in `llms.py`) | Tool calls | Tool output |
|---|---|---|
| `A` / `Au` | native (`tools=`) | `tool` role / `user` role (gpt-oss-120b) |
| `B` / `C` | text, AgentDojo's `local` prompt | `tool` role / `user` role, same rendered bytes (Llama-3.3-70B, Qwen3-32B) |
| `Cr`, `Di` | text | raw-string `user` and `input` roles (exploratory, not in the report) |

Files: `llms.py` (all arms), `relabel_run.py` (plan-driven runner), `relabel_plans.py` (writes `plans/<model>.json`),
`relabel_load.py` (final-record-per-episode loader), `relabel_analyze.py` (paired contrasts), `run_relabel.sh`,
`pairs_manifest.json` (the 249 pairs). Released data: `data/role_relabel_episodes.csv.gz`.

## Shared

`stats.py` holds the statistics of all three experiments: paired differences with a cluster bootstrap over user tasks
(10,000 resamples, seed 1, percentile intervals), plus pair, goal and two-way schemes reported as robustness checks.

## How it was run

One model per GPU, served with vLLM 0.24 (`../serve_vllm.sh`), greedy decoding, Qwen3 thinking off, at most 2,048
generated tokens per call (4,096 for gpt-oss-120b). AgentDojo package 0.1.35, benchmark `v1.2.2`. Example:

```bash
export PY=/path/to/agentdojo-env/bin/python VLLM=/path/to/vllm-env/bin/vllm
bash run_relabel.sh llama33          # also: qwen3, gptoss
bash run_branching.sh llama33        # also: gptoss
PY=/path/to/frameworks-env/bin/python bash run_installed.sh llama33   # also: qwen3, gptoss
python3 -I relabel_analyze.py relabel.md relabel.json
python3 -I branch_analyze.py results_branching branching.md branching.json
```

Each runner first runs a short smoke phase. The original runs paused after it until a person had read the smoke
output (pipeline checks only, never outcomes); the scripts here continue directly.
