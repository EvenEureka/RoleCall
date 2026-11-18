"""Fallible-user runner (= run_gated.py with FallibleGatedToolsExecutor: the user is an LLM; pre-registered).
Same episodes (tasks.main_episodes: U 310 / A 310 / B 97), same OpenHands-patched actor and sampling, same episode
keys and per-turn seeds for every gate (so trajectories coincide until a gate first acts). All (gate, episode) pairs
are interleaved in one pool. Scoring: AgentDojo `_check_task_result` on the final attempt, with the trace of executed
calls only (rejected calls are removed from the function trace, since some security checks read traces).
Per episode one JSON line in <outdir>/<gate>/episodes.jsonl (+ confirmation events and rejected call ids).
usage: python run_fallible.py --base-url URL --user-base-url URL --user-model NAME --config gptoss|qwen3|llama33 --outdir DIR [--gates none,inline,out5,l4]
       [--phase main|smoke] [--workers N] [--deadline EPOCH]
"""
import argparse
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

R2S = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stance")
sys.path.insert(0, R2S)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openai  # noqa: E402
from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, load_system_message  # noqa: E402
from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage  # noqa: E402
from agentdojo.agent_pipeline.errors import AbortAgentError  # noqa: E402
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop  # noqa: E402
from agentdojo.attacks.attack_registry import load_attack  # noqa: E402
from agentdojo.attacks.base_attacks import get_model_name_from_pipeline  # noqa: E402
from agentdojo.functions_runtime import FunctionsRuntime  # noqa: E402
from agentdojo.task_suite.load_suites import get_suite  # noqa: E402
from agentdojo.task_suite.task_suite import functions_stack_trace_from_messages, model_output_from_messages  # noqa: E402

import agent_llm  # noqa: E402
import gated  # noqa: E402
import gated_fallible  # noqa: E402
import tasks  # noqa: E402

BENCH = "v1.2.2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--config", required=True, choices=list(agent_llm.CONFIGS))
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--gates", default="none,inline,out5,l4")
    ap.add_argument("--user-base-url", required=True)
    ap.add_argument("--user-model", required=True)
    ap.add_argument("--phase", default="main", choices=["smoke", "main"])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--deadline", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0, help="CPU tests only")
    a = ap.parse_args()
    gates = a.gates.split(",")
    client = openai.OpenAI(base_url=a.base_url, api_key="EMPTY", timeout=1800, max_retries=1)
    user_client = openai.OpenAI(base_url=a.user_base_url, api_key="EMPTY", timeout=600, max_retries=2)
    sysmsg = load_system_message(None)
    suites = {s: get_suite(BENCH, s) for s in tasks.SUITES}
    pipes = {}
    for g in gates:
        llm = agent_llm.OHNativeLLM(client, a.config, "main")
        p = AgentPipeline([SystemMessage(sysmsg), InitQuery(), llm,
                           ToolsExecutionLoop([gated_fallible.FallibleGatedToolsExecutor(g, client, a.config, user_client, a.user_model), llm])])
        p.name = f"local-oh-{a.config}-main"   # same name as round 2, so the attack text is identical
        assert get_model_name_from_pipeline(p) == "Local model"
        pipes[g] = p
        os.makedirs(os.path.join(a.outdir, g), exist_ok=True)
    attacks = {}
    for sname, su in suites.items():
        att = load_attack("important_instructions", su, pipes[gates[0]])
        for ut in su.user_tasks.values():
            att.get_injection_candidates(ut)
        attacks[sname] = att
    eps = tasks.smoke_episodes() if a.phase == "smoke" else tasks.main_episodes(suites)
    if a.limit:
        eps = eps[:a.limit]
    work = []
    for g in gates:
        done = set()
        f = os.path.join(a.outdir, g, "episodes.jsonl")
        if os.path.exists(f):
            done = {json.loads(l)["key"] for l in open(f) if json.loads(l).get("exception") is None}
        work += [(g, e) for e in eps if e["key"] not in done]
    work = [w for k in range(len(eps)) for w in work if w[1] is eps[k]]  # episode-major interleave across gates
    lock = threading.Lock()

    def run_one(g, ep):
        su = suites[ep["suite"]]
        agent_llm.tls_reset(ep["key"])
        gated.ctx_reset(ep["kind"], ep["suite"], ep["key"], ep["cfg"])
        t0 = time.time()
        res = dict(ep, config=a.config, gate=g)
        attempts = []
        try:
            ut = su.get_user_task_by_id(ep["user_task"])
            it = su.get_injection_task_by_id(ep["injection_task"]) if ep["injection_task"] else None
            inj = attacks[ep["suite"]].attack(ut, it) if ep["kind"] == "U" else {}
            res["injections"] = inj
            prompt = ut.PROMPT + tasks.TWIN_JOIN + it.GOAL if ep["kind"] == "A" else ut.PROMPT
            env = su.load_and_inject_default_environment(inj)
            task_env = ut.init_environment(env)
            pre_env = task_env.model_copy(deep=True)
            runtime = FunctionsRuntime(su.tools)
            messages, out = [], None
            for k in range(3):
                agent_llm.tls_set_attempt(k)
                try:
                    _, _, task_env, messages, _ = pipes[g].query(prompt, runtime, task_env)
                except AbortAgentError as ex:
                    task_env, messages = ex.task_environment, ex.messages
                attempts.append(agent_llm.serialize_messages(messages))
                out = model_output_from_messages(messages)
                if out is not None:
                    break
            trace = [fc for fc in functions_stack_trace_from_messages(messages) if fc.id not in gated.CTX.denied]
            res["utility"] = su._check_task_result(ut, out or [], pre_env, task_env, trace)
            res["goal"] = su._check_task_result(it, out or [], pre_env, task_env, trace) if it is not None else None
            res["n_attempts"] = len(attempts)
            res["exception"] = None
        except Exception as e:  # noqa: BLE001
            res["exception"] = repr(e)[:3000]
            res["traceback"] = traceback.format_exc()[-3000:]
        res["wall"] = round(time.time() - t0, 2)
        res["confirmations"] = gated.CTX.events
        res["denied"] = sorted(gated.CTX.denied)
        res["llm_calls"], _ = agent_llm.tls_get()
        res["attempts"] = attempts
        with lock:
            with open(os.path.join(a.outdir, g, "episodes.jsonl"), "a") as f:
                f.write(json.dumps(res, default=str, ensure_ascii=False) + "\n")
        return res

    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] {a.config} gates={gates}: {len(work)} episodes, workers={a.workers}", flush=True)
    n = n_exc = skipped = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(lambda w=w: None if (a.deadline and time.time() > a.deadline) else run_one(*w)) for w in work]
        for fu in as_completed(futs):
            r = fu.result()
            if r is None:
                skipped += 1
                continue
            n += 1
            n_exc += r["exception"] is not None
            if n % 100 == 0 or n == len(work):
                print(f"[{time.strftime('%H:%M:%S')}] {n}/{len(work)} done, exceptions={n_exc}, "
                      f"elapsed={(time.time() - t_start) / 60:.1f} min", flush=True)
    print(f"GATED_DONE config={a.config} done={n} exceptions={n_exc} skipped_deadline={skipped} "
          f"minutes={(time.time() - t_start) / 60:.1f}", flush=True)


if __name__ == "__main__":
    main()
