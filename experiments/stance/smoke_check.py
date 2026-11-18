"""In-job smoke check for the stance experiment (parse health only, never outcome values).
exit 0 = OK (auto-GO), 1 = FAIL (abort), 2 = REVIEW (plumbing fine but no call carried a security_risk value).
usage: python3 -I smoke_check.py <smoke_run_dir>"""
import collections
import json
import os
import sys

d = sys.argv[1]
eps = [json.loads(line) for line in open(os.path.join(d, "episodes.jsonl"))]
bad, review = [], []
exc = [e for e in eps if e.get("exception")]
if exc:
    bad.append(f"{len(exc)} episode exceptions, e.g. {exc[0]['exception'][:300]}")
calls = [c for e in eps for r in e.get("llm_calls") or [] for c in r["calls"]]
n_with_calls = sum(1 for e in eps if any(r["calls"] for r in e.get("llm_calls") or []))
if n_with_calls < 2:
    bad.append(f"only {n_with_calls}/{len(eps)} episodes executed a tool call")
labs = collections.Counter(c["label"] for c in calls)
if calls and sum(c["bad_json"] for c in calls) > 0.5 * len(calls):
    bad.append("majority of tool-call arguments are not valid JSON")
if calls and labs.get("OMITTED", 0) + labs.get("INVALID", 0) == len(calls):
    review.append(f"no call carried a valid security_risk value: {dict(labs)}")
det_p = os.path.join(d, "detached.jsonl")
det = [json.loads(line) for line in open(det_p)] if os.path.exists(det_p) else []
derr = [r for r in det if r.get("error")]
if derr:
    bad.append(f"{len(derr)} detached request errors, e.g. {derr[0]['error'][:300]}")
for arm in ("L4", "L2", "L4blind", "L4labels", "TS"):
    rs = [r for r in det if r["arm"] == arm and not r.get("error")]
    okp = sum(r["label"] in ("LOW", "MEDIUM", "HIGH") for r in rs)
    if rs and okp < 0.6 * len(rs):
        bad.append(f"{arm} parse {okp}/{len(rs)}: {[(r.get('finish_reason'), (r.get('content') or '')[-120:]) for r in rs][:2]}")
print(f"SMOKE episodes={len(eps)} with_calls={n_with_calls} calls={len(calls)} inline_labels={dict(labs)}")
for e in eps:
    print(f"--- {e['key']} kind={e['kind']} utility={e.get('utility')} goal={e.get('goal')} exc={e.get('exception') is not None} n_attempts={e.get('n_attempts')}")
    for r in e.get("llm_calls") or []:
        cs = [(c["function"], c["label"], json.dumps(c["args"], ensure_ascii=False)[:120]) for c in r["calls"]]
        print(f"   a{r.get('attempt')} t{r['turn']} fin={r['finish_reason']} content={(r.get('content') or '')[:120]!r} calls={cs}")
for arm in ("L4", "L2", "L4blind", "L4labels", "TS"):
    for r in [r for r in det if r["arm"] == arm][:2]:
        print(f"--- DET {arm} {r['id']} set={r.get('rset')} label={r['label']} tail={(r.get('content') or '')[-160:]!r}")
status = "FAIL" if bad else ("REVIEW" if review else "OK")
print("SMOKE_CHECK", status, bad, review)
sys.exit(1 if bad else (2 if review else 0))
