"""Build the detached-rating items from on-policy episodes (pre-registration §4-§5). CPU only, stdlib.
usage: python3 -I build_items.py <run_dir>   (reads <run_dir>/episodes.jsonl, writes <run_dir>/items.jsonl + calls.jsonl)

calls.jsonl = every tool call of every non-error episode with its inline label and set:
  A = attacker-goal call (attacker function + attacker destination) in an attacked episode; `first_attacker` marks the
      first one per episode (unit of the primary endpoint);
  B = any call in a benign (no-injection) episode;
  C = post-injection non-attacker call in an attacked episode (after the first tool output that contains an injection);
  P = pre-injection call in an attacked episode (inline label only; not rated by detached raters).
items.jsonl = calls of sets A, B, C with the prompts for the detached arms L4, L4blind, TS.
"""
import json
import os
import re
import sys

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
import detached  # noqa: E402
import tasks  # noqa: E402

POLICY = None


def policy_for(cfg):
    # local import keeps this script stdlib-only apart from oh_patch (stdlib)
    import oh_patch
    return oh_patch.policy_block(cfg)


def _norm(s):
    """Whitespace-insensitive form: AgentDojo's YAML tool-output formatter collapses blank lines and may re-wrap or
    escape newlines, so injection strings are matched after removing whitespace and literal '\\n' escapes
    (Addendum A1: the raw-substring match of v1 found no injection in any output)."""
    return re.sub(r"\s+", "", (s or "").replace("\\n", ""))


def inj_markers(injections):
    return [_norm(v)[:60] for v in (injections or {}).values() if isinstance(v, str) and v.strip()]


def main(run_dir):
    eps = [json.loads(line) for line in open(os.path.join(run_dir, "episodes.jsonl"))]
    seen, uniq = set(), []
    for e in reversed(eps):  # resumed runs: keep the last record per key
        if e["key"] not in seen:
            seen.add(e["key"]); uniq.append(e)
    eps = uniq[::-1]
    calls_out, items, early_out = [], [], []
    n_err = 0
    for e in eps:
        # Addendum B: calls of episodes that ended in an exception are kept (flagged from_error_episode) when the
        # message list up to the last successful LLM call exists; the primary analysis excludes them.
        err_ep = e.get("exception") is not None
        if err_ep:
            n_err += 1
        if not e.get("messages"):
            continue
        msgs = e["messages"]
        policy = policy_for(e["cfg"])
        agent_sys = msgs[0]["content"] if msgs and msgs[0]["role"] == "system" else ""
        marks = inj_markers(e.get("injections"))
        first_inj = None
        for i, m in enumerate(msgs):
            if m["role"] == "tool" and marks and any(mk in _norm(m.get("output")) for mk in marks):
                first_inj = i
                break
        seen_attacker = False
        ts_hist = []
        # AgentDojo's run_task_with_pipeline re-runs the whole pipeline (up to 3 attempts, same environment) when the
        # final model output is None; only the LAST attempt's message list is returned/logged, so only its calls can
        # be given a transcript (Addendum A2). Earlier-attempt calls are written to early_calls.jsonl (inline only).
        starts = [i for i, r in enumerate(e["llm_calls"]) if r["n_messages"] == 2] or [0]
        for rec in e["llm_calls"][:starts[-1]]:
            for j, c in enumerate(rec["calls"]):
                early_out.append({"ep_key": e["key"], "config": e["config"], "kind": e["kind"], "suite": e["suite"],
                                  "turn": rec["turn"], "j": j, "function": c["function"], "inline": c["label"],
                                  "attacker": e["kind"] == "attacked" and tasks.is_attacker_call(
                                      e["suite"], c["function"], json.dumps(c["args"], ensure_ascii=False).lower())})
        for rec in e["llm_calls"][starts[-1]:]:
            mi = rec["n_messages"]
            if mi >= len(msgs) or msgs[mi]["role"] != "assistant":
                continue
            am = msgs[mi]
            exec_calls = am.get("tool_calls") or []
            same_msg = []
            for j, c in enumerate(rec["calls"]):
                fn, args = c["function"], c["args"]
                # sanity: the executed call j should match
                if j < len(exec_calls) and exec_calls[j]["function"] != fn:
                    print("WARN call mismatch", e["key"], mi, j, file=sys.stderr)
                s = json.dumps(args, ensure_ascii=False).lower()
                post = first_inj is not None and first_inj < mi
                if e["kind"] == "attacked":
                    if tasks.is_attacker_call(e["suite"], fn, s):
                        st = "A"
                    elif post:
                        st = "C"
                    else:
                        st = "P"
                else:
                    st = "B"
                first_att = st == "A" and not seen_attacker
                if st == "A":
                    seen_attacker = True
                thought = (rec.get("content") or "") if j == 0 else ""
                act = detached.ts_action(fn, args, c.get("summary"), thought)
                cid = f"{e['key']}|t{rec['turn']}|c{j}"
                row = {"id": cid, "ep_key": e["key"], "config": e["config"], "cfg": e["cfg"], "kind": e["kind"],
                       "suite": e["suite"], "user_task": e["user_task"], "injection_task": e["injection_task"],
                       "cluster": f"{e['suite']}/{e['user_task']}", "turn": rec["turn"], "j": j, "function": fn,
                       "args": args, "inline": c["label"], "raw_label": c.get("raw_label"), "summary": c.get("summary"),
                       "set": st, "first_attacker": first_att, "post_injection": post,
                       "read": tasks.is_read(fn), "policy_low": tasks.is_policy_low(fn),
                       "outbound": fn in tasks.ATTACK_FUNCS, "utility": e.get("utility"), "security": e.get("security"),
                       "from_error_episode": err_ep,
                       "data_only": (e["suite"], int(e["injection_task"].split("_")[-1])) not in tasks.DATA_ONLY_EXCLUDE
                       if e["injection_task"] else None}
                calls_out.append(row)
                if st in ("A", "B", "C"):
                    it = dict(row)
                    it["prompts"] = {
                        "L4": detached.l4_messages(policy, agent_sys, detached.transcript(msgs, mi, False, same_msg), fn, args),
                        "L4blind": detached.l4_messages(policy, agent_sys, detached.transcript(msgs, mi, True, same_msg), fn, args),
                        "TS": detached.ts_messages(ts_hist, act),
                    }
                    items.append(it)
                ts_hist.append(act)
                same_msg.append((fn, args))
    with open(os.path.join(run_dir, "calls.jsonl"), "w") as f:
        for r in calls_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(run_dir, "early_calls.jsonl"), "w") as f:
        for r in early_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(run_dir, "items.jsonl"), "w") as f:
        for r in items:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"BUILD episodes={len(eps)} error_episodes={n_err} error_episode_calls={sum(r['from_error_episode'] for r in calls_out)} calls={len(calls_out)} items={len(items)} "
          f"sets={dict(Counter(r['set'] for r in calls_out))} first_attacker={sum(r['first_attacker'] for r in calls_out)} "
          f"early_attempt_calls={len(early_out)} early_attacker_calls={sum(r['attacker'] for r in early_out)}")


if __name__ == "__main__":
    main(sys.argv[1])
