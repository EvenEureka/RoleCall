"""Detached raters on the same weights as the actor (pre-registration §5), via the OpenAI-compatible chat API of the job's
vLLM server (server-side chat template, same mode as the actor: Qwen thinking on/off, gpt-oss medium reasoning).
Work order is fixed (pre-registration §7): arm L4 first, then TS, then L4blind; within an arm A-first-attacker calls, then
B policy-LOW calls, then C policy-LOW calls, then the remaining A, B, C calls. A deadline stops new requests.
Output: <run_dir>/detached.jsonl (append, resumable on (id, arm)).
usage: python run_detached.py --base-url URL --config C --run-dir DIR [--arms L4,TS,L4blind] [--sets A,B,C]
       [--workers N] [--deadline EPOCH]
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

import confirm_llm  # noqa: E402
import detached  # noqa: E402


def prio(it):
    if it.get("from_error_episode"):  # Addendum B: sensitivity-only items go after every primary-analysis item
        return 10 + prio({**it, "from_error_episode": False})
    if it["set"] == "A" and it["first_attacker"]:
        return 0
    if it["set"] == "B" and it["policy_low"]:
        return 1
    if it["set"] == "C" and it["policy_low"]:
        return 2
    return {"A": 3, "B": 4, "C": 5}[it["set"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--config", required=True, choices=list(confirm_llm.CONFIGS))
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--arms", default="L4,TS,L4blind")
    ap.add_argument("--sets", default="A,B,C")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--deadline", type=float, default=0.0)
    a = ap.parse_args()
    c = confirm_llm.CONFIGS[a.config]
    client = openai.OpenAI(base_url=a.base_url, api_key="EMPTY", timeout=1800, max_retries=1)
    items = [json.loads(line) for line in open(os.path.join(a.run_dir, "items.jsonl"))]
    sets = set(a.sets.split(","))
    items = [it for it in items if it["set"] in sets]
    outp = os.path.join(a.run_dir, "detached.jsonl")
    done = set()
    if os.path.exists(outp):
        for line in open(outp):
            r = json.loads(line)
            if r.get("error") is None:
                done.add((r["id"], r["arm"]))
    work = []
    for ai, arm in enumerate(a.arms.split(",")):
        for it in sorted(items, key=lambda x: (prio(x), x["id"])):
            if (it["id"], arm) not in done:
                work.append((ai, prio(it), arm, it))
    work.sort(key=lambda w: (w[0], w[1]))
    lock = threading.Lock()
    stats = {"n": 0, "err": 0, "skipped": 0}
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] detached {a.config}: {len(work)} requests ({len(done)} already done), "
          f"arms={a.arms} workers={a.workers}", flush=True)

    def one(w):
        _, _, arm, it = w
        if a.deadline and time.time() > a.deadline:
            with lock:
                stats["skipped"] += 1
            return
        msgs = it["prompts"][arm]
        kw = dict(model=c["served"], messages=msgs, max_tokens=c["max_tokens"],
                  seed=confirm_llm.seed_of(it["id"], arm), extra_body=dict(c["extra"]), **c["sampling"])
        if c["reasoning_effort"]:
            kw["reasoning_effort"] = c["reasoning_effort"]
        rec = {"id": it["id"], "arm": arm, "config": a.config, "set": it["set"]}
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
                print(f"[{time.strftime('%H:%M:%S')}] detached {stats['n']}/{len(work)} errors={stats['err']} "
                      f"elapsed={(time.time()-t0)/60:.1f} min", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(one, work))
    print(f"DETACHED_DONE config={a.config} done={stats['n']} errors={stats['err']} skipped_deadline={stats['skipped']} "
          f"minutes={(time.time()-t0)/60:.1f}", flush=True)


if __name__ == "__main__":
    main()
