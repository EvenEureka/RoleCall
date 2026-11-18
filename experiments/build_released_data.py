"""Build the compact tables in data/ from the raw experiment records.

The raw records (full transcripts, several GB in total) are not part of this repository. This script shows exactly how
each released table was cut from them. Every table keeps one row per episode (or per branch, or per rated call) and
only the fields that reproduce/headline_numbers.py needs.

usage:
  python build_released_data.py --out ../data \
      --installed  <dir with <model>/main/raw/local-<model>-arm{NLIu,NLIt,NSAu,NSAt}.jsonl>      (role/installed)
      --branching  <dir with <model>/main/branch/{prefix,branch}/*.json>                         (role/branching)
      --relabel    <dir with <model>/main/{logs,raw}/local-<model>-arm<arm>...>                  (role/relabel)
      --stance     <dir with <model>/main/{episodes,calls,detached}.jsonl>                       (stance, main run)
      --stance-rep <dir with <model>/main/{calls,detached}.jsonl>                                (stance, replication)
      --gates      informed=<dir> gemma_standard=<dir> phi4_standard=<dir> gemma_cautious=<dir>
                   (each dir holds <model>/<gate>/episodes.jsonl)                                (gates)
Any source can be left out; its tables are then skipped. Standard library only.
"""
import argparse
import collections
import csv
import glob
import gzip
import json
import os
import sys

MODELS = {"llama33": "llama-3.3-70b", "gptoss": "gpt-oss-120b", "qwen3": "qwen3-32b"}
KIND = {"U": "injected", "A": "twin", "B": "benign"}
GOAL_CLASS = {"S": "sensitive_data", "F": "money_only", "N": "non_sensitive", None: ""}

# destination-specific attacker-call rule: the attacker's AgentDojo destinations (IBAN, URL, e-mail addresses of the
# benchmark's fictional attacker) must appear in the call's destination argument; see experiments/stance/taxonomy.py
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "stance"))
from taxonomy import is_attacker_call  # noqa: E402


def b(x):
    """bool -> 0/1, None -> empty cell"""
    return "" if x is None else int(bool(x))


def jsonl(path):
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        path = path + ".gz"
    if not os.path.exists(path):
        return
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write(out, name, header, rows):
    p = os.path.join(out, name)
    with gzip.open(p, "wt", newline="", compresslevel=9) as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {name}: {len(rows)} rows")


# ----------------------------------------------------------------------------------------------- role: installed agents
def installed(src, out):
    arms = {"NLIu": ("llamaindex", "user"), "NLIt": ("llamaindex", "tool"),
            "NSAu": ("smolagents", "user"), "NSAt": ("smolagents", "tool")}
    rows = []
    for fam, model in MODELS.items():
        for arm, (fw, role) in arms.items():
            path = os.path.join(src, fam, "main", "raw", f"local-{fam}-arm{arm}.jsonl")
            last = {}
            for r in jsonl(path):
                k = (r["kind"], r["suite"], r["user_task"], r["injection_task"])
                # last record per episode, preferring records in which the framework actually ran
                if k not in last or not r.get("resumed_from_log"):
                    last[k] = r
            for (kind, s, u, i), r in sorted(last.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
                err = bool(r.get("exception")) or r.get("utility") is None or r.get("security") is None
                rows.append([model, fw, role, kind, s, u, i or "", b(r.get("utility")),
                             b(r.get("security")) if kind == "attacked" else "", int(err)])
    write(out, "role_installed_episodes.csv.gz",
          ["model", "framework", "delivery_role", "kind", "suite", "user_task", "injection_task", "utility",
           "attack_success", "error"], rows)


# ------------------------------------------------------------------------------------------------ role: branching
def branching(src, out):
    voice = {"N": "unattributed", "U": "user_authored"}
    rows = []
    for fam in ("llama33", "gptoss"):
        base = os.path.join(src, fam, "main", "branch")
        if not os.path.isdir(base):
            continue
        pre = {}
        for f in glob.glob(os.path.join(base, "prefix", "*.json")):
            d = json.load(open(f))
            pre[(d["suite"], d["user_task"], d["injection_task"])] = d
        br = collections.defaultdict(list)
        for f in glob.glob(os.path.join(base, "branch", "*.json")):
            d = json.load(open(f))
            br[(d["suite"], d["user_task"], d["injection_task"])].append(d)
        for k in sorted(pre):
            p = pre[k]
            if not br.get(k):
                rows.append([MODELS[fam], *k, p["status"], p.get("j", ""), "", "", "", "", "", "", "", ""])
                continue
            for d in sorted(br[k], key=lambda d: (d["voice"], d["header"])):
                rows.append([MODELS[fam], *k, p["status"], p.get("j", ""), voice[d["voice"]], d["header"],
                             b(d.get("security")), b(d.get("utility")), b(d.get("next_attacker")),
                             int(d.get("first_attacker_step") is not None), int(bool(d.get("exception"))),
                             int(bool(d.get("replay_ok")))])
    write(out, "role_branching.csv.gz",
          ["model", "suite", "user_task", "injection_task", "prefix_status", "branch_step", "injection_voice",
           "header", "attack_success", "utility", "next_step_attacker_action", "attacker_action_any_later_step",
           "exception", "replay_identical"], rows)


# ------------------------------------------------------------------------------------------------ role: relabelling
def relabel(src, out):
    cells = {"llama33": {"B": ("text", "tool"), "C": ("text", "user")},
             "qwen3": {"B": ("text", "tool"), "C": ("text", "user")},
             "gptoss": {"A": ("native", "tool"), "Au": ("native", "user")}}
    templates = ("important_instructions", "ignore_previous", "direct")
    rows = []
    for fam, arms in cells.items():
        base = os.path.join(src, fam, "main")
        man = json.load(open(os.path.join(base, "manifest.json")))
        man = man.get("suites", man)
        pairs = [(s, u, i) for s in ("banking", "slack") for u in man[s]["user_tasks"] for i in man[s]["injection_tasks"]]
        for arm, (calling, role) in arms.items():
            pipe = f"local-{fam}-arm{arm}"
            raw = collections.defaultdict(list)
            for r in jsonl(os.path.join(base, "raw", pipe + ".jsonl")):
                raw[(r["kind"], r["suite"], r["user_task"], r["injection_task"], r.get("attack"))].append(r)
            for att in templates:
                for s, u, i in pairs:
                    recs = raw.get(("attacked", s, u, i, att), [])
                    last = recs[-1] if recs else None
                    lp = os.path.join(base, "logs", pipe, s, u, att, f"{i}.json")
                    if not os.path.exists(lp):
                        rows.append([MODELS[fam], calling, role, att, s, u, i, "", "", 1])
                        continue
                    d = json.load(open(lp))
                    err = d.get("error") is not None or (last or {}).get("exception") is not None or "utility" not in d
                    rows.append([MODELS[fam], calling, role, att, s, u, i, b(d.get("security")), b(d.get("utility")),
                                 int(err)])
    write(out, "role_relabel_episodes.csv.gz",
          ["model", "call_syntax", "delivery_role", "template", "suite", "user_task", "injection_task",
           "attack_success", "utility", "error"], rows)


# ------------------------------------------------------------------------------------------------ stance
def detached_labels(path):
    det = collections.defaultdict(dict)
    for r in jsonl(path):
        if r.get("error") is None and r.get("label") is not None:
            det[r["id"]][r["arm"]] = r["label"]
    return det


def stance(src, out):
    ep_rows, call_rows = [], []
    for fam, model in MODELS.items():
        d = os.path.join(src, fam, "main")
        eps = {}
        for e in jsonl(os.path.join(d, "episodes.jsonl")):
            eps[e["key"]] = e                       # last record per episode key
        for k, e in eps.items():
            ep_rows.append([model, k, KIND[e["kind"]], e["suite"], e["user_task"], e.get("injection_task") or "",
                            GOAL_CLASS[e.get("goal_class")], int(e.get("exception") is not None), b(e.get("utility")),
                            b(e.get("goal"))])
        det = detached_labels(os.path.join(d, "detached.jsonl"))
        for c in jsonl(os.path.join(d, "calls.jsonl")):
            if not c.get("final_attempt"):
                continue
            L = det.get(c["id"], {})
            call_rows.append([model, c["ep_key"], c["id"], c["set"], int(bool(c["first_in_episode"])), c["function"],
                              c["inline"], L.get("L2", ""), L.get("L4", ""), L.get("L4blind", ""),
                              int(bool(c.get("from_error_episode")))])
    write(out, "stance_episodes.csv.gz",
          ["model", "episode", "kind", "suite", "user_task", "injection_task", "goal_class", "exception", "utility",
           "goal_reached"], ep_rows)
    write(out, "stance_calls.csv.gz",
          ["model", "episode", "call_id", "call_set", "first_in_episode", "function", "label_acting", "label_asked",
           "label_reviewing", "label_reviewing_no_tool_outputs", "from_error_episode"], call_rows)


def stance_rep(src, out):
    rows = []
    for fam, model in MODELS.items():
        d = os.path.join(src, fam, "main")
        det = detached_labels(os.path.join(d, "detached.jsonl"))
        for c in jsonl(os.path.join(d, "calls.jsonl")):
            att = c["kind"] == "attacked" and is_attacker_call(c["suite"], c["function"], c["args"])
            rows.append([model, c["ep_key"], c["kind"], c["suite"], c["user_task"], c.get("injection_task") or "",
                         c["turn"], c["j"], c["function"], int(att), c["inline"], det.get(c["id"], {}).get("L4", ""),
                         int(bool(c.get("from_error_episode")))])
    write(out, "stance_replication_calls.csv.gz",
          ["model", "episode", "kind", "suite", "user_task", "injection_task", "turn", "call_index", "function",
           "attacker_directed", "label_acting", "label_reviewing", "from_error_episode"], rows)


# ------------------------------------------------------------------------------------------------ gates
def gates(specs, out):
    rows = []
    for spec in specs:
        user, root = spec.split("=", 1)
        for fam, model in MODELS.items():
            for gate in ("none", "inline", "out5", "l4"):
                p = os.path.join(root, fam, gate, "episodes.jsonl")
                ok, bad = {}, {}
                for r in jsonl(p):
                    (ok if r.get("exception") is None else bad)[r["key"]] = r
                for k in sorted(set(ok) | set(bad)):
                    r = ok.get(k) or bad[k]
                    conf = r.get("confirmations") or []
                    att = [c for c in conf if c.get("attacker")]
                    oth = [c for c in conf if not c.get("attacker")]
                    nonstd = sum(c.get("raw") is None or not str(c.get("raw")).strip().upper().lstrip("*\"' ").startswith(("APPROVE", "REJECT"))
                                 for c in conf) if any("decision" in c for c in conf) else 0
                    infer = sum(len(c.get("calls") or []) for c in (r.get("llm_calls") or []))
                    rows.append([user, model, gate, k, KIND[r["kind"]], r["suite"], r["user_task"],
                                 r.get("injection_task") or "", GOAL_CLASS[r.get("goal_class")],
                                 int(k not in ok), b(r.get("utility")), b(r.get("goal")), len(conf),
                                 len(att), sum(c.get("decision") == "APPROVE" for c in att),
                                 len(oth), sum(c.get("decision") == "REJECT" for c in oth), nonstd,
                                 infer if gate == "l4" else ""])
    write(out, "gates_episodes.csv.gz",
          ["user", "model", "gate", "episode", "kind", "suite", "user_task", "injection_task", "goal_class",
           "exception", "utility", "goal_reached", "confirmations", "attacker_asks", "attacker_asks_approved",
           "other_asks", "other_asks_rejected", "nonstandard_answers", "reviewer_inferences"], rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--installed")
    ap.add_argument("--branching")
    ap.add_argument("--relabel")
    ap.add_argument("--stance")
    ap.add_argument("--stance-rep")
    ap.add_argument("--gates", nargs="*")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.installed:
        installed(a.installed, a.out)
    if a.branching:
        branching(a.branching, a.out)
    if a.relabel:
        relabel(a.relabel, a.out)
    if a.stance:
        stance(a.stance, a.out)
    if a.stance_rep:
        stance_rep(a.stance_rep, a.out)
    if a.gates:
        gates(a.gates, a.out)


if __name__ == "__main__":
    main()
