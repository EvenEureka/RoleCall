# Experiment code

This folder holds the code that produced every number in the report, cleaned for release. The logic is the code that
ran. What changed: cluster paths became command-line arguments or environment variables, cluster job files became the
plain `run_*.sh` scripts, a few modules were renamed, and parts that no reported number uses were removed (each
removal is noted in the file it came from).

| Folder | Question | README |
|---|---|---|
| `role/` | Does the chat role in which a tool output arrives change what the agent obeys? | [role/README.md](role/README.md) |
| `stance/` | Does the model rate its own proposed call differently when acting, when asked, and when reviewing? | [stance/README.md](stance/README.md) |
| `gates/` | What do confirmation gates built on those ratings, or a simple rule, prevent and interrupt in complete episodes? | [gates/README.md](gates/README.md) |
| `build_released_data.py` | Cuts the compact tables in `../data/` out of the raw run records. | below |
| `serve_vllm.sh` | Starts one vLLM server for one model; sourced by every `run_*.sh`. | below |

## Terms used in all three READMEs

- **Prompt injection.** Text planted in data that an agent reads (an e-mail, a web page, a file) that tries to make
  the agent do something its user did not ask for.
- **Harness.** Everything around the model weights: the prompt format, the chat role each message gets, which tools
  exist, and whether a call needs confirmation. RoleCall keeps the weights fixed and changes the harness.
- **AgentDojo.** The benchmark ([Debenedetti et al., 2024](https://github.com/ethz-spylab/agentdojo)). It pairs a
  *user task* (what the user wants) with an *injection task* (the attacker's goal), runs the agent in a simulated
  environment (bank account, Slack workspace, travel agency, office workspace) and checks the final state.
- **Attack success rate (ASR).** The share of attacked episodes in which AgentDojo's check says the attacker's goal
  was reached. *Utility under attack* (UA) is the share in which the user's task was still done.
- **vLLM.** An open-source server for running language models on a GPU behind an OpenAI-compatible HTTP API. Every
  model here is served by vLLM on the same machine as the agent code.

## Software versions

| Component | Version |
|---|---|
| Python | 3.11 for all GPU runs (the analysis scripts also run on 3.10+) |
| AgentDojo | package `agentdojo` 0.1.35, benchmark version `v1.2.2` |
| vLLM | 0.24.0 (with transformers 5.13.0) |
| LlamaIndex (installed agents only) | `llama-index-core` 0.14.25, `llama-index-llms-openai-like` 0.8.1, `llama-index-llms-openai` 0.8.3 |
| smolagents (installed agents only) | 1.26.0 |
| Other | `openai` 3.37.0 (2.54.0 in the installed-agents environment), `pydantic` 2.14.1, `httpx` 0.28.1, `numpy` 2.4.6 |

Two Python environments were used: one with AgentDojo (every experiment except the installed agents) and one with
AgentDojo plus LlamaIndex and smolagents (`role/run_installed.sh`). vLLM ran in its own environment; point `VLLM` at its
`vllm` executable.

## Models

All models are served with vLLM from fixed Hugging Face revisions (`serve_vllm.sh` has the exact revisions and flags):
Llama-3.3-70B-Instruct (RedHatAI FP8-dynamic build), gpt-oss-120b and Qwen3-32B as agents; Gemma-3-27B-it and Phi-4
as simulated users in `gates/`. One model fits on one 96 GB GPU. Decoding:

- `role/`: greedy (temperature 0); Qwen3 thinking off.
- `stance/`, `gates/`: gpt-oss temperature 1.0 and medium reasoning effort; Qwen3 thinking off, temperature 0.7,
  top-p 0.8, top-k 20; Llama temperature 0.6, top-p 0.9 (`stance/agent_llm.py`, `CONFIGS`).

## From raw records to `data/`

The runs write full transcripts (several GB). `build_released_data.py` keeps one row per episode, branch or rated call
and only the fields the reproduce script needs:

```bash
python build_released_data.py --out ../data \
  --installed  results_installed      --branching results_branching/results  --relabel results_relabel/results \
  --stance     results_stance/merged/results     --stance-rep results_replication/results \
  --gates      informed=results_gates/informed gemma_standard=results_gates/gemma_standard \
               phi4_standard=results_gates/phi4_standard gemma_cautious=results_gates/gemma_cautious
```

The directory names are the defaults of the `run_*.sh` scripts. The column meanings are in the top-level README,
section "Data".
