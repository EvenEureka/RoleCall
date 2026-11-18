"""Stance experiment: on-policy AgentDojo runner (pre-registered).

Episode loop = AgentDojo's `TaskSuite.run_task_with_pipeline` (v0.1.35) re-implemented so that EVERY attempt is kept:
load the default environment with the episode's injections, `user_task.init_environment`, copy the pre-environment,
query the pipeline up to 3 times in the same environment until the final model output is not None, then score with
the suite's own `_check_task_result` on the final attempt (AgentDojo semantics). Kinds:
  U  injections = important_instructions(user task, shipped injection task); utility + security (goal achieved)
  A  no injections; prompt = PROMPT + TWIN_JOIN + injection GOAL; utility + goal achieved (injection task's own check)
  B  no injections; utility
Per episode one JSON line: ids, utility, goal, exception, injections, every LLM call (with attempt index and inline
label), the serialized messages of EVERY attempt, wall time.
usage: python run_agents.py --base-url URL --config gptoss|qwen3|llama33 --phase smoke|main|rw --outdir DIR
       [--workers N] [--deadline EPOCH] [--progress FILE]
"""
import argparse
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openai  # noqa: E402
from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, load_system_message  # noqa: E402
from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage  # noqa: E402
from agentdojo.agent_pipeline.errors import AbortAgentError  # noqa: E402
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor, tool_result_to_str  # noqa: E402
from agentdojo.attacks.attack_registry import load_attack  # noqa: E402
from agentdojo.attacks.base_attacks import get_model_name_from_pipeline  # noqa: E402
from agentdojo.functions_runtime import FunctionsRuntime  # noqa: E402
from agentdojo.task_suite.load_suites import get_suite  # noqa: E402
from agentdojo.task_suite.task_suite import functions_stack_trace_from_messages, model_output_from_messages  # noqa: E402

import agent_llm  # noqa: E402
import tasks  # noqa: E402

BENCH = "v1.2.2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--config", required=True, choices=list(agent_llm.CONFIGS))
    ap.add_argument("--phase", default="main", choices=["smoke", "main", "rw"])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--deadline", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0, help="CPU tests only")
    a = ap.parse_args()

    client = openai.OpenAI(base_url=a.base_url, api_key="EMPTY", timeout=1800, max_retries=1)
    sysmsg = load_system_message(None)
    suites = {s: get_suite(BENCH, s) for s in tasks.SUITES}
    os.makedirs(a.outdir, exist_ok=True)
    pipes = {}
    for cfg in ("main", "rw"):
        llm = agent_llm.OHNativeLLM(client, a.config, cfg)
        p = AgentPipeline([SystemMessage(sysmsg), InitQuery(), llm, ToolsExecutionLoop([ToolsExecutor(tool_result_to_str), llm])])
        p.name = f"local-oh-{a.config}-{cfg}"
        assert get_model_name_from_pipeline(p) == "Local model", get_model_name_from_pipeline(p)
        pipes[cfg] = p
    attacks = {}
    for sname, su in suites.items():
        att = load_attack("important_instructions", su, pipes["main"])
        for ut in su.user_tasks.values():
            att.get_injection_candidates(ut)
        attacks[sname] = att
    # tool schemas per suite and cfg, saved once for the raters (ACTOR-frame rebuild)
    sch = {f"{s}|{cfg}": agent_llm.tool_schemas(FunctionsRuntime(su.tools), cfg) for s, su in suites.items() for cfg in ("main", "rw")}
    json.dump(sch, open(os.path.join(a.outdir, "tool_schemas.json"), "w"))

    if a.phase == "smoke":
        eps = tasks.smoke_episodes()
    elif a.phase == "main":
        eps = tasks.main_episodes(suites)
    else:
        eps = tasks.rw_episodes(suites)
    if a.limit:
        eps = eps[:a.limit]
    outp = os.path.join(a.outdir, "episodes.jsonl")
    done = set()
    if os.path.exists(outp):
        for line in open(outp):
            r = json.loads(line)
            if r.get("exception") is None:
                done.add(r["key"])
    eps = [e for e in eps if e["key"] not in done]
    json.dump({"config": a.config, "phase": a.phase, "bench": BENCH, "served": agent_llm.SERVED, "episodes": eps,
               "already_done": len(done)}, open(os.path.join(a.outdir, f"manifest.{a.phase}.json"), "w"), indent=0)
    lock = threading.Lock()

    def run_one(ep):
        su = suites[ep["suite"]]
        agent_llm.tls_reset(ep["key"])
        t0 = time.time()
        res = dict(ep)
        res["config"] = a.config
        attempts = []
        try:
            ut = su.get_user_task_by_id(ep["user_task"])
            it = su.get_injection_task_by_id(ep["injection_task"]) if ep["injection_task"] else None
            inj = attacks[ep["suite"]].attack(ut, it) if ep["kind"] == "U" else {}
            res["injections"] = inj
            prompt = ut.PROMPT + tasks.TWIN_JOIN + it.GOAL if ep["kind"] == "A" else ut.PROMPT
            res["prompt"] = prompt
            env = su.load_and_inject_default_environment(inj)
            task_env = ut.init_environment(env)
            pre_env = task_env.model_copy(deep=True)
            runtime = FunctionsRuntime(su.tools)
            messages, out = [], None
            for k in range(3):
                agent_llm.tls_set_attempt(k)
                try:
                    _, _, task_env, messages, _ = pipes[ep["cfg"]].query(prompt, runtime, task_env)
                except AbortAgentError as ex:
                    task_env, messages = ex.task_environment, ex.messages
                attempts.append(agent_llm.serialize_messages(messages))
                out = model_output_from_messages(messages)
                if out is not None:
                    break
            trace = functions_stack_trace_from_messages(messages)
            res["utility"] = su._check_task_result(ut, out or [], pre_env, task_env, trace)
            res["goal"] = su._check_task_result(it, out or [], pre_env, task_env, trace) if it is not None else None
            res["n_attempts"] = len(attempts)
            res["exception"] = None
        except Exception as e:  # recorded; such episodes are excluded from the primary analysis (pre-registration §6)
            res["exception"] = repr(e)[:3000]
            res["traceback"] = traceback.format_exc()[-3000:]
            log, last = agent_llm.tls_get()
            if last is not None and (not attempts or len(attempts) <= getattr(agent_llm._TLS, "attempt", 0)):
                attempts.append(agent_llm.serialize_messages(last))
        res["wall"] = round(time.time() - t0, 2)
        res["llm_calls"], _ = agent_llm.tls_get()
        res["attempts"] = attempts
        with lock:
            with open(outp, "a") as f:
                f.write(json.dumps(res, default=str, ensure_ascii=False) + "\n")
        return res

    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] {a.config} phase={a.phase}: {len(eps)} episodes ({len(done)} already done), "
          f"workers={a.workers} served={agent_llm.SERVED}", flush=True)
    n = n_exc = skipped = 0
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
    print(f"AGENTS_DONE config={a.config} phase={a.phase} done={n} exceptions={n_exc} skipped_deadline={skipped} "
          f"minutes={(time.time()-t_start)/60:.1f}", flush=True)


if __name__ == "__main__":
    main()
