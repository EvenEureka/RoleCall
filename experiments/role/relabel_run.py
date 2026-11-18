"""Relabelling-experiment episode runner: AgentDojo's own benchmark functions, parallel
episodes against one vLLM server, driven by a plan file (plans/<family>.json, written by relabel_plans.py).

Plan: {"smoke": {"cells": [...]}, "main": {"groups": [{"name": ..., "cells": [...]}, ...]}}
Cell: {"arm": "B", "attack": "direct", "pairs": "all" | "subset60" | [[suite, user_task, injection_task], ...],
       "benign": bool (also run benign user tasks + injection tasks as user tasks), "seed": int | null,
       "temperature": float}
Pipeline name: local-<family>-arm<ARM>[-s<seed>] (resolves to "Local model" in the important_instructions attack).
Groups run in order; inside a group, episodes of all cells are interleaved so cells finish together. With a deadline,
no new episode starts after it (later groups are dropped first: groups are listed in priority order).
AgentDojo logs: <outdir>/logs/<pipeline>/<suite>/<task>/<attack|none>/<inj|none>.json; raw completions and
exceptions: <outdir>/raw/<pipeline>.jsonl. Existing AgentDojo logs are reused (force_rerun=False) = resume.

usage: python relabel_run.py --base-url URL --model NAME --family F --plan PLAN --phase smoke|main --outdir DIR
       [--workers 32] [--max-tokens 2048] [--deadline EPOCH] [--reasoning-effort X]
"""
import argparse
import hashlib
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

from llms import make_llm, raw_log, reset_raw_log, set_episode_seed  # noqa: E402

BENCH = "v1.2.2"
HERE = os.path.dirname(os.path.abspath(__file__))
# 60-pair subset for the seeded repeat cells (not used by any reported number); optional
SUBSET60 = os.environ.get("SUBSET60", os.path.join(HERE, "plans", "subset60.json"))


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


def pipe_name(family, arm, seed):
    return f"local-{family}-arm{arm}" + (f"-s{seed}" if seed is not None else "")


def stable_seed(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", required=True, choices=["qwen3", "llama33", "gptoss", "llama31"])
    ap.add_argument("--plan", required=True)
    ap.add_argument("--phase", default="main", choices=["smoke", "main"])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--deadline", type=float, default=0.0)
    ap.add_argument("--reasoning-effort", default=None)
    ap.add_argument("--dry-run", action="store_true", help="print episode counts only")
    a = ap.parse_args()

    plan = json.load(open(a.plan))
    groups = [{"name": "smoke", "cells": plan["smoke"]["cells"]}] if a.phase == "smoke" else plan["main"]["groups"]
    subset60 = [tuple(x) for x in json.load(open(SUBSET60))] if os.path.exists(SUBSET60) else None

    client = openai.OpenAI(base_url=a.base_url, api_key="EMPTY", timeout=900, max_retries=1)
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}} if a.family == "qwen3" else {}
    sysmsg = load_system_message(None)
    suites = {s: get_suite(BENCH, s) for s in ("banking", "slack")}
    os.makedirs(os.path.join(a.outdir, "raw"), exist_ok=True)
    lock = threading.Lock()

    pipes, attacks = {}, {}

    def get_pipe(arm, seed, temperature):
        key = (arm, seed)
        if key not in pipes:
            llm = make_llm(arm, client, a.model, a.family, a.max_tokens, extra_body, a.reasoning_effort, temperature)
            p = AgentPipeline([SystemMessage(sysmsg), InitQuery(), llm,
                               ToolsExecutionLoop([ToolsExecutor(tool_result_to_str), llm])])
            p.name = pipe_name(a.family, arm, seed)
            assert get_model_name_from_pipeline(p) == "Local model", get_model_name_from_pipeline(p)
            pipes[key] = p
        return pipes[key]

    def get_attack(arm, seed, sname, attack):
        key = (arm, seed, sname, attack)
        if key not in attacks:
            att = load_attack(attack, suites[sname], pipes[(arm, seed)])
            for ut in suites[sname].user_tasks.values():  # pre-warm lru_cache in the main thread
                att.get_injection_candidates(ut)
            attacks[key] = att
        return attacks[key]

    def all_pairs():
        return [(s, u, i) for s, su in suites.items() for u in su.user_tasks for i in su.injection_tasks]

    def cell_episodes(c):
        arm, seed, temp = c["arm"], c.get("seed"), float(c.get("temperature", 0.0))
        assert (seed is None) == (temp == 0.0), c
        get_pipe(arm, seed, temp)
        pairs = c["pairs"]
        pairs = all_pairs() if pairs == "all" else (subset60 if pairs == "subset60" else [tuple(x) for x in pairs])
        assert pairs, c
        eps = []
        if c.get("benign"):
            for s, su in suites.items():
                for u in su.user_tasks:
                    eps.append(("benign", arm, seed, s, u, None, None))
                for i in su.injection_tasks:
                    eps.append(("feasibility", arm, seed, s, None, i, None))
        for s, u, i in pairs:
            get_attack(arm, seed, s, c["attack"])
            eps.append(("attacked", arm, seed, s, u, i, c["attack"]))
        return eps

    logdir = Path(a.outdir) / "logs"
    with open(os.path.join(a.outdir, "manifest.json"), "w") as f:
        json.dump({"suites": {s: {"user_tasks": list(su.user_tasks), "injection_tasks": list(su.injection_tasks)}
                              for s, su in suites.items()}, "plan": plan, "phase": a.phase,
                   "subset60": subset60}, f, indent=1)

    def run_one(ep):
        kind, arm, seed, sname, u, i, attack = ep
        LOGGER_STACK.set([])
        reset_raw_log()
        su, pipe = suites[sname], pipes[(arm, seed)]
        set_episode_seed(stable_seed(seed, arm, sname, u, i, attack, kind) if seed is not None else None)
        t0 = time.time()
        res = {"kind": kind, "arm": arm, "seed": seed, "pipeline": pipe.name, "suite": sname, "user_task": u,
               "injection_task": i, "attack": attack}
        try:
            with DirLogger(logdir):
                if kind == "attacked":
                    ur, sr = run_task_with_injection_tasks(su, pipe, su.get_user_task_by_id(u),
                                                           get_attack(arm, seed, sname, attack), logdir, False, [i],
                                                           BENCH)
                    res["utility"], res["security"] = ur[(u, i)], sr[(u, i)]
                elif kind == "benign":
                    res["utility"], res["security"] = run_task_without_injection_tasks(
                        su, pipe, su.get_user_task_by_id(u), logdir, False, BENCH)
                else:
                    res["utility"], res["security"] = run_task_without_injection_tasks(
                        su, pipe, su.get_injection_task_by_id(i), logdir, False, BENCH)
            res["exception"] = None
        except Exception as e:
            res["exception"] = repr(e)[:3000]
            res["traceback"] = traceback.format_exc()[-3000:]
        res["wall"] = time.time() - t0
        res["raw"] = raw_log()
        with lock:
            with open(os.path.join(a.outdir, "raw", f"{pipe.name}.jsonl"), "a") as f:
                f.write(json.dumps(res, default=str) + "\n")
        return res

    t_start = time.time()
    for g in groups:
        lists = [cell_episodes(c) for c in g["cells"]]
        seen, eps = set(), []
        for k in range(max(len(x) for x in lists)):  # round-robin interleave, dedupe (benign shared per pipeline)
            for x in lists:
                if k < len(x) and x[k] not in seen:
                    seen.add(x[k])
                    eps.append(x[k])
        desc = ", ".join("%s/%s/%s" % (c["arm"], c["attack"], c.get("seed")) for c in g["cells"])
        print(f"[{time.strftime('%H:%M:%S')}] group {g['name']}: {len(eps)} episodes ({desc}), workers={a.workers}",
              flush=True)
        if a.dry_run:
            continue
        done = n_exc = 0
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(lambda e=ep: None if (a.deadline and time.time() > a.deadline) else run_one(e))
                    for ep in eps]
            for fu in as_completed(futs):
                r = fu.result()
                if r is None:
                    continue
                done += 1
                n_exc += r["exception"] is not None
                if done % 50 == 0 or done == len(eps):
                    print(f"[{time.strftime('%H:%M:%S')}] {g['name']}: {done}/{len(eps)} done, exceptions={n_exc}, "
                          f"elapsed={(time.time() - t_start) / 60:.1f} min", flush=True)
        print(f"GROUP_DONE {g['name']} done={done}/{len(eps)} exceptions={n_exc}", flush=True)
        if a.deadline and time.time() > a.deadline:
            print("DEADLINE reached; stopping after group", g["name"], flush=True)
            break
    print(f"RUN_DONE phase={a.phase} family={a.family} minutes={(time.time() - t_start) / 60:.1f}", flush=True)


if __name__ == "__main__":
    main()
