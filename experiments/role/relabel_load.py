"""Load wave-1 / wave-2 relabelling episodes (read-only) with explicit final-record-per-episode semantics.

An episode's outcome comes from AgentDojo's own log JSON (utility, security); its error status from the LAST raw
record for that episode in raw/<pipeline>.jsonl (records are appended; a whole-cell re-run appends a second record).
The gpt-oss G2 cells were re-run after the first job was stopped externally (a documented deviation); every G2 episode
therefore has two raw records, and only the last one is used. `audit_final_records` cross-checks the final raw record against the log JSON.
stdlib only.
"""
import json
import os

# directories that hold results/<family>/main/{logs,raw} of the first and the second relabelling wave
W1 = os.environ.get("RELABEL_WAVE1_DIR", "results_relabel_wave1")   # only used as a fallback for benign utility
W2 = os.environ.get("RELABEL_DIR", "results_relabel")
II, DI, IG = "important_instructions", "direct", "ignore_previous"
SUITES = ("banking", "slack")

# the 249 attacked pairs (16 banking x 9 goals + 21 Slack x 5 goals); same list as AgentDojo v1.2.2's suites
_mp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pairs_manifest.json")
_man = json.load(open(_mp))
_man = _man.get("suites", _man)
PAIRS = [(s, u, i) for s in SUITES for u in _man[s]["user_tasks"] for i in _man[s]["injection_tasks"]]
BEN = [(s, u) for s in SUITES for u in _man[s]["user_tasks"]]
FEA = [(s, i) for s in SUITES for i in _man[s]["injection_tasks"]]
assert len(PAIRS) == 249 and len(BEN) == 37 and len(FEA) == 14


def raw_records(path, wave):
    """-> {episode key: [records in file order]}"""
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        r = json.loads(line)
        att = r.get("attack") if wave == 2 else (II if r["kind"] == "attacked" else None)
        out.setdefault((r["kind"], r["suite"], r["user_task"], r["injection_task"], att), []).append(r)
    return out


def _rec(path, raw_last):
    if not os.path.exists(path):
        return {"missing": True, "err": True}
    d = json.load(open(path))
    exc = raw_last.get("exception") if raw_last else None
    err = d.get("error") is not None or exc is not None or "utility" not in d
    calls = (raw_last or {}).get("raw", [])
    return {"missing": False, "err": err, "utility": bool(d.get("utility")), "security": bool(d.get("security")),
            "n_calls": len(calls), "length": sum(1 for c in calls if c.get("finish_reason") == "length"),
            "raw_utility": (raw_last or {}).get("utility"), "raw_security": (raw_last or {}).get("security"),
            "log_mtime": os.path.getmtime(path)}


def _paths(fam, arm, wave, seed=None):
    pipe = f"local-{fam}-arm{arm}" + (f"-s{seed}" if seed is not None else "")
    base = os.path.join(W2 if wave == 2 else W1, "results", fam, "main")
    return os.path.join(base, "logs", pipe), os.path.join(base, "raw", (pipe if wave == 2 else arm) + ".jsonl")


def load_attacked(fam, arm, attack, wave=2, seed=None, pairs=None):
    ld, rp = _paths(fam, arm, wave, seed)
    if not os.path.isdir(ld):
        return None
    raw = raw_records(rp, wave)
    out = {}
    for s, u, i in (pairs or PAIRS):
        recs = raw.get(("attacked", s, u, i, attack), [])
        r = _rec(os.path.join(ld, s, u, attack, f"{i}.json"), recs[-1] if recs else None)
        r["n_raw_records"] = len(recs)
        out[(s, u, i)] = r
    if all(v["missing"] for v in out.values()):
        return None
    return out


def load_benign(fam, arm, wave):
    ld, rp = _paths(fam, arm, wave)
    if not os.path.isdir(ld):
        return None
    raw = raw_records(rp, wave)

    def last(k):
        v = raw.get(k, [])
        return v[-1] if v else None
    ben = {(s, u): _rec(os.path.join(ld, s, u, "none", "none.json"), last(("benign", s, u, None, None))) for s, u in BEN}
    fea = {(s, i): _rec(os.path.join(ld, s, i, "none", "none.json"), last(("feasibility", s, None, i, None)))
           for s, i in FEA}
    if all(v["missing"] for v in ben.values()):
        return None
    return {"ben": ben, "fea": fea, "wave": wave}


def load_cell_at(main_dir, pipe, attack, pairs=None):
    """Attacked cell of a pipeline in an arbitrary results/<family>/main dir (round-3 adapters)."""
    ld, rp = os.path.join(main_dir, "logs", pipe), os.path.join(main_dir, "raw", pipe + ".jsonl")
    if not os.path.isdir(ld):
        return None
    raw = raw_records(rp, 2)
    out = {}
    for s, u, i in (pairs or PAIRS):
        recs = raw.get(("attacked", s, u, i, attack), [])
        r = _rec(os.path.join(ld, s, u, attack, f"{i}.json"), recs[-1] if recs else None)
        r["n_raw_records"] = len(recs)
        r["raw_last"] = recs[-1] if recs else None
        out[(s, u, i)] = r
    return out


def load_benign_at(main_dir, pipe):
    ld, rp = os.path.join(main_dir, "logs", pipe), os.path.join(main_dir, "raw", pipe + ".jsonl")
    if not os.path.isdir(ld):
        return None
    raw = raw_records(rp, 2)

    def last(k):
        v = raw.get(k, [])
        return v[-1] if v else None
    ben = {(s, u): _rec(os.path.join(ld, s, u, "none", "none.json"), last(("benign", s, u, None, None))) for s, u in BEN}
    fea = {(s, i): _rec(os.path.join(ld, s, i, "none", "none.json"), last(("feasibility", s, None, i, None)))
           for s, i in FEA}
    return {"ben": ben, "fea": fea, "wave": 3}


def benign_source(fam, arm):
    """pre-registered rule: wave-2 benign runs where they exist, else wave 1."""
    return load_benign(fam, arm, 2) or load_benign(fam, arm, 1)
