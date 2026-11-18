"""First rating run (replication): on-policy AgentDojo runner (adapted from the role experiments' runner): AgentDojo's own benchmark functions
(`run_task_with_injection_tasks` / `run_task_without_injection_tasks`, suites v1.2.2, attack important_instructions),
parallel episodes, one local vLLM server, the OpenHands field patched in (confirm_llm.OHNativeLLM).

Task set (pre-registration §3, fixed by tasks.py): attacked = gold-HIGH injection tasks; benign = every user task, no injection.
Per episode the raw record (one JSON line) holds: kind, suite, user_task, injection_task, utility, security,
exception, the injection strings, every LLM call (content, reasoning, tool calls with the inline label), and the final
message list (AgentDojo messages; tool-call args as executed).

usage: python run_agents.py --base-url URL --config gptoss|qwen3|qwen3think|llama33
       --phase smoke|main --outdir DIR [--kinds attacked,benign] [--workers N] [--deadline EPOCH]
"""
import argparse
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openai  # noqa: E402
from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, load_system_message  # noqa: E402
from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage  # noqa: E402
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor, tool_result_to_str  # noqa: E402
from agentdojo.attacks.attack_registry import load_attack  # noqa: E402
from agentdojo.attacks.base_attacks import get_model_name_from_pipeline  # noqa: E402
from agentdojo.benchmark import run_task_with_injection_tasks, run_task_without_injection_tasks  # noqa: E402
from agentdojo.logging import LOGGER_STACK, NullLogger  # noqa: E402
from agentdojo.task_suite.load_suites import get_suite  # noqa: E402

import confirm_llm  # noqa: E402
import tasks  # noqa: E402

BENCH = "v1.2.2"


class DirLogger(NullLogger):
    def __init__(self, logdir):
        self.logdir = logdir
        self.messages = []

    def __enter__(self):
        LOGGER_STACK.get().append(self)
        return self

    def __exit__(self, *a):
        LOGGER_STACK.get().pop()
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--config", required=True, choices=list(confirm_llm.CONFIGS))
    ap.add_argument("--cfg", default="main", choices=["main"])
    ap.add_argument("--phase", default="main", choices=["smoke", "main"])
    ap.add_argument("--kinds", default="attacked,benign")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--deadline", type=float, default=0.0, help="epoch secs: no new episodes after this")
    a = ap.parse_args()

    client = openai.OpenAI(base_url=a.base_url, api_key="EMPTY", timeout=1800, max_retries=1)
    sysmsg = load_system_message(None)
    suites = {s: get_suite(BENCH, s) for s in tasks.SUITES}
    os.makedirs(a.outdir, exist_ok=True)
    lock = threading.Lock()
    llm = confirm_llm.OHNativeLLM(client, a.config, a.cfg)
    pipe = AgentPipeline([SystemMessage(sysmsg), InitQuery(), llm,
                          ToolsExecutionLoop([ToolsExecutor(tool_result_to_str), llm])])
    pipe.name = f"local-oh-{a.config}-{a.cfg}"
    assert get_model_name_from_pipeline(pipe) == "Local model", get_model_name_from_pipeline(pipe)
    attacks = {}
    for sname, su in suites.items():
        att = load_attack("important_instructions", su, pipe)
        for ut in su.user_tasks.values():  # pre-warm lru_cache in the main thread
            att.get_injection_candidates(ut)
        attacks[sname] = att

    eps = tasks.smoke_episodes() if a.phase == "smoke" else tasks.main_episodes(suites)
    kinds = set(a.kinds.split(","))
    eps = [e for e in eps if e["kind"] in kinds]
    outp = os.path.join(a.outdir, "episodes.jsonl")
    done = set()
    if os.path.exists(outp):  # resumable
        for line in open(outp):
            r = json.loads(line)
            if r.get("exception") is None:
                done.add(r["key"])
    eps = [e for e in eps if e["key"] not in done]
    with open(os.path.join(a.outdir, f"manifest.{a.phase}.json"), "w") as f:
        json.dump({"config": a.config, "cfg": a.cfg, "bench": BENCH, "episodes": eps, "already_done": len(done)}, f, indent=0)
    logdir = Path(a.outdir) / "adojo_logs"

    def run_one(ep):
        su = suites[ep["suite"]]
        LOGGER_STACK.set([])
        confirm_llm.tls_reset(ep["key"])
        t0 = time.time()
        res = dict(ep)
        res.update({"config": a.config, "cfg": a.cfg})
        try:
            with DirLogger(logdir):
                ut = su.get_user_task_by_id(ep["user_task"])
                if ep["kind"] == "attacked":
                    it = su.get_injection_task_by_id(ep["injection_task"])
                    res["injections"] = attacks[ep["suite"]].attack(ut, it)
                    ur, sr = run_task_with_injection_tasks(su, pipe, ut, attacks[ep["suite"]], logdir, False,
                                                           [ep["injection_task"]], BENCH)
                    k = (ep["user_task"], ep["injection_task"])
                    res["utility"], res["security"] = ur[k], sr[k]
                else:
                    res["injections"] = {}
                    res["utility"], res["security"] = run_task_without_injection_tasks(su, pipe, ut, logdir, False, BENCH)
            res["exception"] = None
        except Exception as e:  # recorded; excluded as error in the analysis
            res["exception"] = repr(e)[:3000]
            res["traceback"] = traceback.format_exc()[-3000:]
        res["wall"] = round(time.time() - t0, 2)
        log, last = confirm_llm.tls_get()
        res["llm_calls"] = log
        res["messages"] = confirm_llm.serialize_messages(last) if last is not None else None
        with lock:
            with open(outp, "a") as f:
                f.write(json.dumps(res, default=str, ensure_ascii=False) + "\n")
        return res

    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] {a.config}/{a.cfg} phase={a.phase}: {len(eps)} episodes "
          f"({len(done)} already done), workers={a.workers}", flush=True)
    n = n_exc = 0
    skipped = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(lambda e=ep: None if (a.deadline and time.time() > a.deadline) else run_one(e)) for ep in eps]
        for fu in as_completed(futs):
            r = fu.result()
            if r is None:
                skipped += 1
                continue
            n += 1
            n_exc += r["exception"] is not None
            if n % 25 == 0 or n == len(eps):
                print(f"[{time.strftime('%H:%M:%S')}] {n}/{len(eps)} done, exceptions={n_exc}, "
                      f"elapsed={(time.time()-t_start)/60:.1f} min", flush=True)
    print(f"AGENTS_DONE config={a.config} cfg={a.cfg} phase={a.phase} done={n} exceptions={n_exc} "
          f"skipped_deadline={skipped} minutes={(time.time()-t_start)/60:.1f}", flush=True)


if __name__ == "__main__":
    main()
