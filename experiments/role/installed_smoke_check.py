"""Smoke check of a native run directory (CPU mock or GPU smoke). Exit code = number of failed checks.
Per arm: no exceptions; every episode executed >= 1 AgentDojo call and was scored; on the wire, every message that
carries a tool observation has the arm's role (NLIu/NSAu: user, NLIt/NSAt: tool) and no other message has role `tool`;
LlamaIndex requests stream and send no `tools`; smolagents requests send `tools` with tool_choice "required".
Amendment 1 (2026-10-28, registered before any gpt-oss main episode): smolagents error
tool-responses ("Error:\n...", written by ActionStep as TOOL_RESPONSE) are tool responses too and must carry the arm's
role, so they no longer count as "non-observation tool messages"; smolagents' max-steps fallback request
(provide_final_answer: no `tools`, prompt "Based on the above, please provide an answer to the following user task")
is exempt from the tools/required check, and there must be exactly one per max_steps_fallback episode.
usage: python installed_smoke_check.py RUN_DIR
"""
import collections
import glob
import json
import sys

EXPECT = {"NLIu": "user", "NLIt": "tool", "NSAu": "user", "NSAt": "tool"}
FALLBACK = "Based on the above, please provide an answer to the following user task"


def text_of(m):
    c = m.get("content")
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c or ""


def main(d):
    fails = 0
    stats = collections.defaultdict(collections.Counter)
    for f in sorted(glob.glob(f"{d}/raw/*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            if r.get("resumed_from_log"):
                continue
            arm = r["arm"]; s = stats[arm]; s["episodes"] += 1
            nat = r.get("native") or {}
            if r.get("exception"):
                s["exceptions"] += 1; print("EXC", arm, r["exception"][:300])
            s["executed_calls"] += nat.get("n_executed_calls", 0)
            s[f"term_{nat.get('termination')}"] += 1
            s["scored"] += r.get("utility") is not None and r.get("security") is not None
            for w in nat.get("wire", []):
                q = w.get("request") or {}
                s["requests"] += 1
                if arm.startswith("NLI"):
                    s["li_stream"] += bool(q.get("stream")); s["li_tools_sent"] += bool(q.get("tools"))
                elif not q.get("tools") and FALLBACK in text_of((q.get("messages") or [{}])[-1]):
                    s["sa_fallback_requests"] += 1
                else:
                    s["sa_tools_sent"] += bool(q.get("tools")); s["sa_required"] += q.get("tool_choice") == "required"
                for m in q.get("messages", [])[1:]:
                    t = text_of(m)
                    is_obs = t.lstrip().startswith("Observation") or "\nObservation:" in t[:20]
                    if is_obs:
                        s[f"obs_role_{m['role']}"] += 1
                    elif arm.startswith("NSA") and t.startswith("Error:\n"):
                        s[f"err_role_{m['role']}"] += 1
                    elif m.get("role") == "tool":
                        s["non_obs_tool_msgs"] += 1
    for arm, s in sorted(stats.items()):
        exp = EXPECT[arm]; other = "tool" if exp == "user" else "user"
        checks = {
            "no exceptions": s["exceptions"] == 0,
            "every episode scored": s["scored"] == s["episodes"],
            "calls executed": s["executed_calls"] >= s["episodes"],
            f"observations sent as {exp}": s[f"obs_role_{exp}"] > 0 and s[f"obs_role_{other}"] == 0,
            f"tool errors sent as {exp}": s[f"err_role_{other}"] == 0,
            "no non-observation tool messages": s["non_obs_tool_msgs"] == 0,
        }
        if arm.startswith("NLI"):
            checks["streams, no tools field"] = s["li_stream"] == s["requests"] and s["li_tools_sent"] == 0
        else:
            checks["tools + tool_choice=required"] = s["sa_tools_sent"] == s["requests"] - s["sa_fallback_requests"] == s["sa_required"]
            checks["one fallback request per max-steps episode"] = s["sa_fallback_requests"] == s["term_max_steps_fallback"]
        print(arm, dict(s))
        for k, ok in checks.items():
            print(f"   {'PASS' if ok else 'FAIL'}  {k}")
            fails += not ok
    print("SMOKE_CHECK", "PASS" if fails == 0 else f"FAIL ({fails})")
    return fails


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
