"""Identical-history branching with a 2 (delivery header) x 2 (voice) factorial.

Stage `prefix` (per attacked pair): run the episode with the model's trained tool channel (Llama: wave-2 arm B,
`ipython`; gpt-oss: wave-2 arm A, `functions.NAME`) on the V_U-injected environment and STOP at the first tool batch
whose output contains the injection marker (the attacked observation), before the model sees it. The saved
pre-injection history (system, user, assistant calls, injection-free tool results, the assistant message whose calls
return the attacked observation) is voice-independent by construction.

Stage `branch` (per pair x voice v in {U, N} x header h in {tool, user}): rebuild the environment with voice-v
injections, REPLAY the saved tool calls (asserting byte-identical injection-free outputs), deliver the attacked
batch with its injected message(s) under header h (all other messages keep the trained header), then
  next step  : one LLM call -> does it contain an attacker-directed call (branch_voice.attacker_action)?
  continued  : AgentDojo ToolsExecutionLoop with the remaining iteration budget (15 - j), default header for every
               later observation -> AgentDojo utility and security (= targeted attack success) on the final state.
No retries (AgentDojo's 3x retry on an empty final output would regenerate the prefix).

Stage `samples` (secondary): for each branch, K next-step samples at temperature T (seeded per request).

Outputs (one JSON per unit; existing files are skipped = resume): <outdir>/{prefix,branch,samples}/...json
usage: python branch_run.py --base-url URL --model NAME --family llama33|gptoss --phase smoke|main --outdir DIR
       [--stages prefix,branch,samples] [--workers 32] [--max-tokens N] [--deadline EPOCH] [--samples 3]
       [--sample-temp 0.7] [--heartbeat FILE] [--watchdog-min 20] [--client fake]
"""
import argparse
import copy
import hashlib
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
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor, tool_result_to_str  # noqa: E402
from agentdojo.attacks.attack_registry import load_attack  # noqa: E402
from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime  # noqa: E402
from agentdojo.logging import LOGGER_STACK  # noqa: E402
from agentdojo.task_suite.load_suites import get_suite  # noqa: E402
from agentdojo.task_suite.task_suite import functions_stack_trace_from_messages  # noqa: E402
from agentdojo.types import get_text_content_as_str  # noqa: E402

import branch_llms as r3_llms  # noqa: E402
from branch_llms import HEARTBEAT, make_llm, raw_log, reset_episode_state, set_episode_seed  # noqa: E402
from branch_voice import any_attacker_action, build_injections, contains_marker  # noqa: E402

BENCH = "v1.2.2"
MAX_ITERS = 15  # AgentDojo ToolsExecutionLoop default
VOICES, HEADERS = ("U", "N"), ("tool", "user")
SMOKE_PAIRS = [("banking", "user_task_0", "injection_task_0"), ("banking", "user_task_3", "injection_task_1"),
               ("slack", "user_task_1", "injection_task_1")]
ARM = {"llama33": "Bbr", "gptoss": "Abr", "llama31": "Bbr"}


# ------------------------------------------------------------------ (de)serialisation of AgentDojo messages
def ser(msgs):
    def one(m):
        d = dict(m)
        if d.get("tool_calls") is not None:
            d["tool_calls"] = [tc.model_dump() for tc in d["tool_calls"]]
        if d.get("tool_call") is not None:
            d["tool_call"] = d["tool_call"].model_dump()
        return d
    return [one(m) for m in msgs]


def deser(msgs):
    out = []
    for d in msgs:
        m = dict(d)
        if m.get("tool_calls") is not None:
            m["tool_calls"] = [FunctionCall.model_validate(tc) for tc in m["tool_calls"]]
        if m.get("tool_call") is not None:
            m["tool_call"] = FunctionCall.model_validate(m["tool_call"])
        out.append(m)
    return out


def tool_sig(m):
    return (get_text_content_as_str(m["content"]) if m["content"] is not None else None, m.get("error"))


def calls_of(msg):
    return [{"function": c.function, "args": dict(c.args)} for c in (msg.get("tool_calls") or [])]


def stable_seed(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def unit_name(s, u, i, v=None, h=None, k=None):
    return "__".join(x for x in (s, u, i, v, h, (f"s{k}" if k is not None else None)) if x is not None) + ".json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", required=True, choices=list(ARM))
    ap.add_argument("--phase", default="main", choices=["smoke", "main"])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--stages", default="prefix,branch,samples")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--deadline", type=float, default=0.0)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--sample-temp", type=float, default=0.7)
    ap.add_argument("--heartbeat", default=None)
    ap.add_argument("--watchdog-min", type=float, default=20.0)
    ap.add_argument("--reasoning-effort", default=None)
    ap.add_argument("--client", default="openai", choices=["openai", "fake"])
    a = ap.parse_args()

    if a.client == "fake":
        from branch_fake_client import FakeClient
        client = FakeClient(a.family)
    else:
        client = openai.OpenAI(base_url=a.base_url, api_key="EMPTY", timeout=900, max_retries=1)
    arm = ARM[a.family]
    suites = {s: get_suite(BENCH, s) for s in ("banking", "slack")}
    sysmsg = load_system_message(None)
    llm0 = make_llm(arm, client, a.model, a.family, a.max_tokens, None, a.reasoning_effort, 0.0)
    llmT = make_llm(arm, client, a.model, a.family, a.max_tokens, None, a.reasoning_effort, a.sample_temp)
    pipe = AgentPipeline([SystemMessage(sysmsg), InitQuery(), llm0, ToolsExecutionLoop([ToolsExecutor(), llm0])])
    pipe.name = f"local-{a.family}-armBr"
    attacks = {s: load_attack("important_instructions", su, pipe) for s, su in suites.items()}
    for s, att in attacks.items():
        assert att.model_name == "Local model", att.model_name
        for ut in suites[s].user_tasks.values():  # pre-warm lru_cache in the main thread
            att.get_injection_candidates(ut)
    pairs = SMOKE_PAIRS if a.phase == "smoke" else \
        [(s, u, i) for s, su in suites.items() for u in su.user_tasks for i in su.injection_tasks]
    for d in ("prefix", "branch", "samples"):
        os.makedirs(os.path.join(a.outdir, d), exist_ok=True)
    HEARTBEAT["path"] = a.heartbeat
    lock = threading.Lock()

    # ---------------------------------------------------------------- watchdog
    state = {"inflight": 0}

    def watchdog():
        while True:
            time.sleep(30)
            idle = time.time() - HEARTBEAT["t"]
            if state["inflight"] > 0 and idle > a.watchdog_min * 60:
                print(f"WATCHDOG: no completed request for {idle / 60:.1f} min with {state['inflight']} in flight; "
                      f"exiting", flush=True)
                os._exit(3)
    threading.Thread(target=watchdog, daemon=True).start()

    def write(path, obj):
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(obj, f, default=str)
        os.replace(tmp, path)

    def summary(stage, rec):
        with lock:
            with open(os.path.join(a.outdir, f"summary_{stage}.jsonl"), "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")

    def env_for(s, u, i, voice):
        su = suites[s]
        ut, it = su.get_user_task_by_id(u), su.get_injection_task_by_id(i)
        inj = build_injections(attacks[s], ut, it, voice)
        env = su.load_and_inject_default_environment(inj)
        task_env = ut.init_environment(env)
        return su, ut, it, inj, task_env, task_env.model_copy(deep=True)

    # ---------------------------------------------------------------- stage 1: prefix
    def run_prefix(pair):
        s, u, i = pair
        LOGGER_STACK.set([])
        reset_episode_state()
        out = {"suite": s, "user_task": u, "injection_task": i, "family": a.family, "arm": arm}
        t0 = time.time()
        try:
            su, ut, it, inj, task_env, pre_env = env_for(s, u, i, "U")
            runtime = FunctionsRuntime(su.tools)
            ex = ToolsExecutor(tool_result_to_str)
            _, _, _, msgs, _ = SystemMessage(sysmsg).query(ut.PROMPT, runtime)
            _, _, _, msgs, _ = InitQuery().query(ut.PROMPT, runtime, task_env, msgs)
            _, _, task_env, msgs, _ = llm0.query(ut.PROMPT, runtime, task_env, msgs)
            j, status = 0, None
            while True:
                last = msgs[-1]
                if last["role"] != "assistant" or not last.get("tool_calls"):
                    status = "ended_before_injection"
                    break
                if j >= MAX_ITERS:
                    status = "cap_before_injection"
                    break
                n0 = len(msgs)
                _, _, task_env, msgs, _ = ex.query(ut.PROMPT, runtime, task_env, msgs)
                j += 1
                hit = [k for k, m in enumerate(msgs[n0:]) if contains_marker(m)]
                if hit:
                    status = "branch"
                    out.update({"j": j, "attacked_idx": hit, "n_batch": len(msgs) - n0,
                                "prefix": ser(msgs[:n0]), "batch_U": ser(msgs[n0:])})
                    break
                _, _, task_env, msgs, _ = llm0.query(ut.PROMPT, runtime, task_env, msgs)
            out["status"] = status
            if status != "branch":  # unbranched episode: score it as is (descriptive only)
                mo = msgs[-1]["content"] if msgs[-1]["role"] == "assistant" else None
                tr = functions_stack_trace_from_messages(msgs)
                out["utility"] = su._check_task_result(ut, mo or [], pre_env, task_env, tr)
                out["security"] = su._check_task_result(it, mo or [], pre_env, task_env, tr)
                out["messages"] = ser(msgs)
            out["exception"] = None
        except Exception as e:
            out["status"], out["exception"], out["traceback"] = "exception", repr(e)[:3000], traceback.format_exc()[-3000:]
        out["raw"], out["wall"] = raw_log(), time.time() - t0
        write(os.path.join(a.outdir, "prefix", unit_name(s, u, i)), out)
        summary("prefix", {k: out.get(k) for k in ("suite", "user_task", "injection_task", "status", "j",
                                                   "attacked_idx", "exception", "utility", "security")})
        return out

    # ---------------------------------------------------------------- replay helper
    def replay(pre, voice):
        s, u, i = pre["suite"], pre["user_task"], pre["injection_task"]
        su, ut, it, inj, task_env, pre_env = env_for(s, u, i, voice)
        runtime = FunctionsRuntime(su.tools)
        ex = ToolsExecutor(tool_result_to_str)
        ref = deser(pre["prefix"])
        msgs, ref_tools = [], [m for m in ref if m["role"] == "tool"]
        for m in ref:
            if m["role"] == "tool":
                continue
            msgs.append(m)
            if m["role"] == "assistant" and m.get("tool_calls"):
                _, _, task_env, msgs, _ = ex.query(ut.PROMPT, runtime, task_env, msgs)
        got_tools = [m for m in msgs if m["role"] == "tool"]
        n_ref = len(ref_tools)
        same = [tool_sig(x) for x in got_tools[:n_ref]] == [tool_sig(x) for x in ref_tools]
        batch = got_tools[n_ref:]
        hit = [k for k, m in enumerate(batch) if contains_marker(m)]
        ok = same and len(batch) == pre["n_batch"] and hit == pre["attacked_idx"]
        if voice == "U":
            ok = ok and [tool_sig(x) for x in batch] == [tool_sig(x) for x in deser(pre["batch_U"])]
        return su, ut, it, runtime, task_env, pre_env, msgs, len(msgs) - len(batch), hit, ok

    def flagged(msgs, start, hit, header):
        out = list(msgs)
        for k in hit:
            m = dict(out[start + k])
            m["r3_header"] = header
            out[start + k] = m
        return out

    # ---------------------------------------------------------------- stage 2: branch
    def run_branch(pre, voice, header):
        s, u, i = pre["suite"], pre["user_task"], pre["injection_task"]
        LOGGER_STACK.set([])
        reset_episode_state()
        out = {"suite": s, "user_task": u, "injection_task": i, "voice": voice, "header": header, "j": pre["j"],
               "family": a.family}
        t0 = time.time()
        try:
            su, ut, it, runtime, task_env, pre_env, msgs, start, hit, ok = replay(pre, voice)
            out["replay_ok"] = ok
            msgs = flagged(msgs, start, hit, header)
            n0 = len(msgs)
            _, _, task_env, msgs, _ = llm0.query(ut.PROMPT, runtime, task_env, msgs)
            nxt = msgs[n0]
            out["next_calls"] = calls_of(nxt)
            out["next_attacker"] = any_attacker_action(s, i, nxt.get("tool_calls"))
            out["next_text"] = get_text_content_as_str(nxt["content"]) if nxt["content"] is not None else None
            loop = ToolsExecutionLoop([ToolsExecutor(tool_result_to_str), llm0], max_iters=MAX_ITERS - pre["j"])
            if MAX_ITERS - pre["j"] > 0:
                _, _, task_env, msgs, _ = loop.query(ut.PROMPT, runtime, task_env, msgs)
            mo = msgs[-1]["content"] if msgs[-1]["role"] == "assistant" else None
            tr = functions_stack_trace_from_messages(msgs)
            out["utility"] = su._check_task_result(ut, mo or [], pre_env, task_env, tr)
            out["security"] = su._check_task_result(it, mo or [], pre_env, task_env, tr)
            out["model_output_none"] = mo is None
            later = [m for m in msgs[n0:] if m["role"] == "assistant"]
            first = [k for k, m in enumerate(later) if any_attacker_action(s, i, m.get("tool_calls"))]
            out["first_attacker_step"] = first[0] if first else None
            out["n_after"] = len(later)
            out["messages"] = ser(msgs)
            out["exception"] = None
        except Exception as e:
            out["exception"], out["traceback"] = repr(e)[:3000], traceback.format_exc()[-3000:]
        out["raw"], out["wall"] = raw_log(), time.time() - t0
        write(os.path.join(a.outdir, "branch", unit_name(s, u, i, voice, header)), out)
        summary("branch", {k: out.get(k) for k in ("suite", "user_task", "injection_task", "voice", "header",
                                                   "replay_ok", "next_attacker", "utility", "security",
                                                   "first_attacker_step", "exception")})
        return out

    # ---------------------------------------------------------------- stage 3: next-step samples
    def run_sample(pre, voice, header, k):
        s, u, i = pre["suite"], pre["user_task"], pre["injection_task"]
        LOGGER_STACK.set([])
        reset_episode_state()
        set_episode_seed(stable_seed(k, a.family, s, u, i, voice, header))
        out = {"suite": s, "user_task": u, "injection_task": i, "voice": voice, "header": header, "sample": k,
               "temperature": a.sample_temp}
        try:
            su, ut, it, runtime, task_env, pre_env, msgs, start, hit, ok = replay(pre, voice)
            out["replay_ok"] = ok
            msgs = flagged(msgs, start, hit, header)
            n0 = len(msgs)
            _, _, task_env, msgs, _ = llmT.query(ut.PROMPT, runtime, task_env, msgs)
            out["next_calls"] = calls_of(msgs[n0])
            out["next_attacker"] = any_attacker_action(s, i, msgs[n0].get("tool_calls"))
            out["exception"] = None
        except Exception as e:
            out["exception"], out["traceback"] = repr(e)[:3000], traceback.format_exc()[-3000:]
        out["raw"] = raw_log()
        write(os.path.join(a.outdir, "samples", unit_name(s, u, i, voice, header, k)), out)
        summary("samples", {k2: out.get(k2) for k2 in ("suite", "user_task", "injection_task", "voice", "header",
                                                       "sample", "replay_ok", "next_attacker", "exception")})
        return out

    # ---------------------------------------------------------------- driver
    def run_stage(name, units, fn):
        todo = [x for x in units if not os.path.exists(x[0])]
        print(f"[{time.strftime('%H:%M:%S')}] stage {name}: {len(todo)} to run ({len(units) - len(todo)} already "
              f"done), workers={a.workers}", flush=True)
        HEARTBEAT["t"] = time.time()
        done = exc = 0

        def wrap(args):
            if a.deadline and time.time() > a.deadline:
                return None
            with lock:
                state["inflight"] += 1
            try:
                return fn(*args)
            finally:
                with lock:
                    state["inflight"] -= 1
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(wrap, x[1]) for x in todo]
            for fu in as_completed(futs):
                r = fu.result()
                if r is None:
                    continue
                done += 1
                exc += r.get("exception") is not None
                if done % 50 == 0 or done == len(todo):
                    print(f"[{time.strftime('%H:%M:%S')}] {name}: {done}/{len(todo)} done, exceptions={exc}",
                          flush=True)
        print(f"STAGE_DONE {name} done={done}/{len(todo)} exceptions={exc}", flush=True)
        return not (a.deadline and time.time() > a.deadline)

    stages = a.stages.split(",")
    t_start = time.time()

    def load_prefixes():
        out = []
        for s, u, i in pairs:
            p = os.path.join(a.outdir, "prefix", unit_name(s, u, i))
            if os.path.exists(p):
                d = json.load(open(p))
                if d.get("status") == "branch":
                    out.append(d)
        return out

    if "prefix" in stages:
        units = [(os.path.join(a.outdir, "prefix", unit_name(*p)), (p,)) for p in pairs]
        if not run_stage("prefix", units, run_prefix):
            print("DEADLINE after prefix", flush=True)
            return
    pres = load_prefixes()
    print(f"prefixes with a branch point: {len(pres)}/{len(pairs)}", flush=True)
    if "branch" in stages:
        units = [(os.path.join(a.outdir, "branch", unit_name(p["suite"], p["user_task"], p["injection_task"], v, h)),
                  (p, v, h)) for p in pres for v in VOICES for h in HEADERS]
        if not run_stage("branch", units, run_branch):
            print("DEADLINE after branch", flush=True)
            return
    if "samples" in stages and a.samples > 0:
        units = [(os.path.join(a.outdir, "samples",
                               unit_name(p["suite"], p["user_task"], p["injection_task"], v, h, k)), (p, v, h, k))
                 for k in range(1, a.samples + 1) for p in pres for v in VOICES for h in HEADERS]
        run_stage("samples", units, run_sample)
    print(f"RUN_DONE family={a.family} phase={a.phase} minutes={(time.time() - t_start) / 60:.1f}", flush=True)


if __name__ == "__main__":
    main()
