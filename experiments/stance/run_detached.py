"""Detached + on-policy-L2 raters for the stance experiment (pre-registered). Same weights/mode/sampling as the actor, via the
job's vLLM server. Every (item, arm) is appended to <run_dir>/detached.jsonl (resumable). Work order by set then arm,
so a deadline keeps the confirmatory primary (AU_first/AA_goal L4, L2) complete first.
usage: python run_detached.py --base-url URL --config C --run-dir DIR [--arms ...] [--deadline EPOCH] [--workers N]
"""
import argparse
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import openai  # noqa: E402

import agent_llm  # noqa: E402
import detached  # noqa: E402

SET_PRIO = {"AU_first": 0, "AA_goal": 1, "B_low": 2, "B_out": 3, "C_low": 4}
ARM_PRIO = {"L4": 0, "L2": 1, "L4blind": 2, "L4labels": 3, "TS": 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--config", required=True, choices=list(agent_llm.CONFIGS))
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--arms", default="L4,L2,L4blind,L4labels,TS")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--deadline", type=float, default=0.0)
    a = ap.parse_args()
    c = agent_llm.CONFIGS[a.config]
    client = openai.OpenAI(base_url=a.base_url, api_key="EMPTY", timeout=1800, max_retries=1)
    items = [json.loads(line) for line in open(os.path.join(a.run_dir, "items.jsonl"))]
    want = set(a.arms.split(","))
    outp = os.path.join(a.run_dir, "detached.jsonl")
    done = set()
    if os.path.exists(outp):
        for line in open(outp):
            r = json.loads(line)
            if r.get("error") is None:
                done.add((r["id"], r["arm"]))
    work = []
    for it in items:
        for arm in it["arms"]:
            if arm in want and (it["id"], arm) not in done:
                work.append((SET_PRIO.get(it["rset"], 9), ARM_PRIO.get(arm, 9), arm, it))
    work.sort(key=lambda w: (w[1], w[0], w[3]["id"]))  # arm-major so L4 of every set finishes before L2 of any set
    lock = threading.Lock()
    stats = {"n": 0, "err": 0, "skip": 0}
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] detached_r2 {a.config}: {len(work)} requests ({len(done)} done) arms={a.arms}", flush=True)

    def one(w):
        _, _, arm, it = w
        if a.deadline and time.time() > a.deadline:
            with lock:
                stats["skip"] += 1
            return
        msgs = it["prompts"][arm]
        kw = dict(model=agent_llm.SERVED, messages=msgs, max_tokens=c["max_tokens"],
                  seed=agent_llm.seed_of(it["id"], arm), extra_body=dict(c["extra"]), **c["sampling"])
        if arm == "L2":  # the follow-up turn is a tool-less chat; keep the actor's own tools out of it
            pass
        if c["reasoning_effort"]:
            kw["reasoning_effort"] = c["reasoning_effort"]
        rec = {"id": it["id"], "arm": arm, "config": a.config, "set": it["set"], "rset": it["rset"]}
        err = None
        for k in range(2):
            try:
                resp = client.chat.completions.create(**kw)
                ch = resp.choices[0]
                content = ch.message.content
                rec.update({"content": content, "finish_reason": ch.finish_reason,
                            "reasoning": getattr(ch.message, "reasoning_content", None) or getattr(ch.message, "reasoning", None),
                            "usage": resp.usage.model_dump() if resp.usage else None})
                rec["label"] = detached.parse_ts(content) if arm == "TS" else detached.parse_l4(content)
                err = None
                break
            except Exception as e:  # noqa: BLE001
                err = repr(e)[:2000]
                time.sleep(5 * (k + 1))
        rec["error"] = err
        if err is not None:
            rec["label"] = None
            rec["traceback"] = traceback.format_exc()[-1500:]
        rec["t"] = round(time.time() - t0, 1)
        with lock:
            with open(outp, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats["n"] += 1
            stats["err"] += err is not None
            if stats["n"] % 100 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] {stats['n']}/{len(work)} err={stats['err']} "
                      f"{(time.time()-t0)/60:.1f} min", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(one, work))
    print(f"DETACHED_R2_DONE config={a.config} done={stats['n']} errors={stats['err']} skipped={stats['skip']} "
          f"minutes={(time.time()-t0)/60:.1f}", flush=True)


if __name__ == "__main__":
    main()
