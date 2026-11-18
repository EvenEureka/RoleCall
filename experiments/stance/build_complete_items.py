"""Rating completion: detached-L4 rating items for the benign calls that the main stance run never rated (pre-registered).
The main stance run rated only policy-LOW and outbound benign calls (build_items.py, sets B_low / B_out), so every detached
gate's benign interruption rate was only bounded. This builds the L4 item for every remaining benign call in the gate
population (final attempt, episode without an exception) with exactly the round-2 construction
(detached.transcript + detached.l4_messages + oh_patch.policy_block), so ids, prompts and seeds match round 2.
usage: python3 -I build_complete_items.py <run_dir (results/<cfg>/main)> <out_dir>
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detached  # noqa: E402
import oh_patch  # noqa: E402


def main(run_dir, out_dir):
    eps = [json.loads(line) for line in open(os.path.join(run_dir, "episodes.jsonl"))]
    seen, uniq = set(), []
    for e in reversed(eps):
        if e["key"] not in seen:
            seen.add(e["key"]); uniq.append(e)
    eps = uniq[::-1]
    rated = set()
    for line in open(os.path.join(run_dir, "detached.jsonl")):
        r = json.loads(line)
        if r["arm"] == "L4" and r.get("error") is None and r.get("label") is not None:
            rated.add(r["id"])
    items = []
    for e in eps:
        if e["kind"] != "B" or e.get("exception") is not None or not e.get("attempts"):
            continue
        policy = oh_patch.policy_block(e["cfg"])
        ai = len(e["attempts"]) - 1                      # final attempt only (the gate population)
        msgs = e["attempts"][ai]
        agent_sys = msgs[0]["content"] if msgs and msgs[0]["role"] == "system" else ""
        for mi, m in enumerate(msgs):
            if m["role"] != "assistant" or not m.get("tool_calls"):
                continue
            same_msg = []
            for j, tc in enumerate(m["tool_calls"]):
                fn, args = tc["function"], tc["args"]
                cid = f"{e['key']}|a{ai}|t{mi}|c{j}"
                if cid not in rated:
                    tr = detached.transcript(msgs, mi, False, same_msg)
                    items.append({"id": cid, "ep_key": e["key"], "config": e["config"], "cfg": e["cfg"], "kind": "B",
                                  "suite": e["suite"], "user_task": e["user_task"], "set": "B", "rset": "B_other",
                                  "function": fn, "args": args, "arms": ["L4"],
                                  "prompts": {"L4": detached.l4_messages(policy, agent_sys, tr, fn, args)}})
                same_msg.append((fn, args))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "items.jsonl"), "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"BUILD_COMPLETE {run_dir}: {len(items)} unrated benign calls; functions={dict(Counter(i['function'] for i in items))}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
