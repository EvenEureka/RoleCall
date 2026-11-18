"""Installed-framework episode runner (the adapter-round runner with the installed-framework element of
installed_agents.py instead of the source-reimplemented adapters). AgentDojo v1.2.2 banking + slack, attack
`important_instructions`; per arm the 249 attacked pairs, the 37 benign user tasks and the 14 injection tasks run as user
tasks (feasibility), i.e. 300 episodes, as in round-3 experiment (b).

AgentDojo logs: <outdir>/logs/<pipeline>/<suite>/<task>/<attack|none>/<inj|none>.json (force_rerun=False = resume);
per-episode record incl. every HTTP request/response to the model server: <outdir>/raw/<pipeline>.jsonl.

usage: python installed_run.py --base-url URL --model NAME --arms NLIu,NLIt,NSAu,NSAt --outdir DIR
       [--pairs all|smoke] [--workers 32] [--max-tokens 2048] [--deadline EPOCH] [--heartbeat FILE]
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
from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, load_system_message  # noqa: E402
from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage  # noqa: E402
from agentdojo.attacks.attack_registry import load_attack  # noqa: E402
from agentdojo.attacks.base_attacks import get_model_name_from_pipeline  # noqa: E402
from agentdojo.benchmark import run_task_with_injection_tasks, run_task_without_injection_tasks  # noqa: E402
from agentdojo.logging import LOGGER_STACK, NullLogger  # noqa: E402
from agentdojo.task_suite.load_suites import get_suite  # noqa: E402

from installed_agents import ARMS, NativeAgent, episode_log, reset_episode_log  # noqa: E402

BENCH = "v1.2.2"
ATTACK = "important_instructions"
# smoke = the three pairs of the round-3 smoke plan (one banking, two slack)
SMOKE_PAIRS = [("banking", "user_task_0", "injection_task_0"), ("slack", "user_task_1", "injection_task_1"),
               ("slack", "user_task_4", "injection_task_3")]


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
    ap.add_argument("--model", required=True)
    ap.add_argument("--arms", default="NLIu,NLIt,NSAu,NSAt")
    ap.add_argument("--family", default="llama33", choices=["llama33", "gptoss", "qwen3"])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--pairs", default="all", choices=["all", "smoke"])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--deadline", type=float, default=0.0)
    ap.add_argument("--heartbeat", default=None)
    ap.add_argument("--watchdog-min", type=float, default=30.0)
    a = ap.parse_args()
    arms = a.arms.split(",")
    assert all(x in ARMS for x in arms), arms

    sysmsg = load_system_message(None)
    suites = {s: get_suite(BENCH, s) for s in ("banking", "slack")}
    os.makedirs(os.path.join(a.outdir, "raw"), exist_ok=True)
    lock = threading.Lock()
    last_done = {"t": time.time()}
    inflight = {"n": 0}

    def watchdog():
        while True:
            time.sleep(30)
            idle = time.time() - last_done["t"]
            if inflight["n"] > 0 and idle > a.watchdog_min * 60:
                print(f"WATCHDOG: no episode finished for {idle / 60:.1f} min with {inflight['n']} in flight; exiting",
                      flush=True)
                os._exit(3)
    threading.Thread(target=watchdog, daemon=True).start()

    pipes, attacks = {}, {}
    for arm in arms:
        extra = {"chat_template_kwargs": {"enable_thinking": False}} if a.family == "qwen3" else None  # as in the earlier runner
        p = AgentPipeline([SystemMessage(sysmsg), InitQuery(), NativeAgent(arm, a.base_url, a.model, a.max_tokens, extra)])
        p.name = f"local-{a.family}-arm{arm}"
        assert get_model_name_from_pipeline(p) == "Local model", get_model_name_from_pipeline(p)
        pipes[arm] = p
        for s, su in suites.items():
            att = load_attack(ATTACK, su, p)
            for ut in su.user_tasks.values():  # pre-warm lru_cache in the main thread
                att.get_injection_candidates(ut)
            attacks[(arm, s)] = att

    pairs = ([(s, u, i) for s, su in suites.items() for u in su.user_tasks for i in su.injection_tasks]
             if a.pairs == "all" else SMOKE_PAIRS)
    per_arm = {}
    for arm in arms:
        eps = []
        if a.pairs == "all":
            for s, su in suites.items():
                eps += [("benign", arm, s, u, None) for u in su.user_tasks]
                eps += [("feasibility", arm, s, None, i) for i in su.injection_tasks]
        else:
            eps += [("benign", arm, "banking", "user_task_0", None)]
        eps += [("attacked", arm, s, u, i) for s, u, i in pairs]
        per_arm[arm] = eps
    eps = [x for k in range(max(map(len, per_arm.values()))) for arm in arms for x in per_arm[arm][k:k + 1]]
    logdir = Path(a.outdir) / "logs"
    json.dump({"arms": arms, "pairs": a.pairs, "n_episodes": len(eps), "bench": BENCH, "attack": ATTACK},
              open(os.path.join(a.outdir, "manifest.json"), "w"), indent=1)
    print(f"[{time.strftime('%H:%M:%S')}] {len(eps)} episodes, arms={arms}, workers={a.workers}", flush=True)

    def run_one(ep):
        kind, arm, sname, u, i = ep
        LOGGER_STACK.set([])
        reset_episode_log()
        su, pipe = suites[sname], pipes[arm]
        t0 = time.time()
        res = {"kind": kind, "arm": arm, "pipeline": pipe.name, "suite": sname, "user_task": u, "injection_task": i,
               "attack": ATTACK if kind == "attacked" else None}
        try:
            with DirLogger(logdir):
                if kind == "attacked":
                    ur, sr = run_task_with_injection_tasks(su, pipe, su.get_user_task_by_id(u), attacks[(arm, sname)],
                                                           logdir, False, [i], BENCH)
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
        res["native"] = episode_log()
        # AgentDojo reused a finished log (resume): the framework was not run, so there are no native diagnostics
        res["resumed_from_log"] = res["native"] is None and res["exception"] is None
        with lock:
            with open(os.path.join(a.outdir, "raw", f"{pipe.name}.jsonl"), "a") as f:
                f.write(json.dumps(res, default=str) + "\n")
            last_done["t"] = time.time()
            if a.heartbeat:
                Path(a.heartbeat).touch()
        return res

    def guarded(e):
        if a.deadline and time.time() > a.deadline:
            return None
        with lock:
            inflight["n"] += 1
        try:
            return run_one(e)
        finally:
            with lock:
                inflight["n"] -= 1

    t_start = time.time()
    done = n_exc = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for fu in as_completed([ex.submit(guarded, ep) for ep in eps]):
            r = fu.result()
            if r is None:
                continue
            done += 1
            n_exc += r["exception"] is not None
            if done % 50 == 0 or done == len(eps):
                print(f"[{time.strftime('%H:%M:%S')}] {done}/{len(eps)} done, exceptions={n_exc}, "
                      f"elapsed={(time.time() - t_start) / 60:.1f} min", flush=True)
    print(f"RUN_DONE done={done}/{len(eps)} exceptions={n_exc} minutes={(time.time() - t_start) / 60:.1f}", flush=True)


if __name__ == "__main__":
    main()
