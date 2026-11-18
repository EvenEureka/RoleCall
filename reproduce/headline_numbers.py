"""Recompute the headline numbers of the RoleCall report from the released tables in data/.

usage:
    python reproduce/headline_numbers.py            # point estimates (seconds)
    python reproduce/headline_numbers.py --ci       # also the bootstrap intervals and the readouts that need them
                                                    # (a few seconds more; pure Python for Section 3, numpy elsewhere)
    python reproduce/headline_numbers.py --ci --json data/headline.json    # also (re)write the JSON summary

Every number is computed here from per-episode, per-branch or per-call rows. Nothing is copied from the report's
tables. The script prints one line per number: whether it matches the value printed in the report, the value itself,
and one sentence saying what it means. With --json it also writes every number to a JSON file (data/headline.json in
the repository was written with --ci). The exit code is 1 if any number differs from the report.

Vocabulary used in the comments:
  attacked pair      one AgentDojo user task combined with one injection goal (the attacker's instruction).
  attack success     AgentDojo's own check that the attacker's goal was reached in the final environment state.
  ASR                attack success rate: the share of attacked pairs (or injected episodes) with attack success.
  UA                 utility under attack: the user's own task was completed in an attacked episode.
  cluster bootstrap  resample whole user tasks (with every pair or episode that belongs to them), recompute the
                     statistic, and take percentiles over the resamples. This respects that pairs sharing a user task
                     are not independent.
"""
import argparse
import collections
import csv
import gzip
import json
import os
import random
import sys

try:
    import numpy as np
except ImportError:  # point estimates only need the standard library
    np = None

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
NAME = {"llama-3.3-70b": "Llama-3.3-70B", "gpt-oss-120b": "gpt-oss-120b", "qwen3-32b": "Qwen3-32B"}
FAMS = ("gpt-oss-120b", "qwen3-32b", "llama-3.3-70b")      # column order of the report's Tables 2 and 3

LINES, COUNTS = [], collections.Counter()
H = {}                                                       # -> data/headline.json
CI = False


# ====================================================================================================== output helpers
def out(s=""):
    LINES.append(s)
    print(s, flush=True)


def put(path, value):
    d = H
    for k in path[:-1]:
        d = d.setdefault(k, {})
    d[path[-1]] = value


def fmt(x, nd=1, signed=False):
    if x is None or x != x:
        return "nan"
    s = f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"
    return s.replace("-0.0", "0.0").replace("+0.0", "0.0") if float(s) == 0 else s


def check(path, value, report, meaning, nd=1, signed=False, ci=None, report_ci=None, scale=100.0):
    """value and ci are fractions (scale 100 -> percent or points) unless scale=1."""
    v = None if value is None else value * scale
    c = None if ci is None else [ci[0] * scale, ci[1] * scale]
    vs = fmt(v, nd, signed)
    status = "ok"
    if report is not None and vs != fmt(report, nd, signed):
        status = "DIFF"
    cs = ""
    if c is not None:
        cs = f" [{fmt(c[0], nd)}, {fmt(c[1], nd)}]"
        if report_ci is not None and (fmt(c[0], nd) != fmt(report_ci[0], nd) or fmt(c[1], nd) != fmt(report_ci[1], nd)):
            status = "DIFF"
    rep = "" if report is None else f"report {fmt(report, nd, signed)}" + (
        f" [{fmt(report_ci[0], nd)}, {fmt(report_ci[1], nd)}]" if report_ci is not None and c is not None else "")
    if report is None:
        status = "info"
    COUNTS[status] += 1
    out(f"  [{status:>4}] {vs + cs:<24} {rep:<30} {meaning}")
    put(path, round(v, 4) if v is not None else None)
    if c is not None:
        put(path[:-1] + [path[-1] + "_ci95"], [round(c[0], 4), round(c[1], 4)])


def check_text(path, value, report, meaning):
    status = "info" if report is None else ("ok" if value == report else "DIFF")
    COUNTS[status] += 1
    rep = "" if report is None else f"report {report}"
    out(f"  [{status:>4}] {str(value):<24} {rep:<30} {meaning}")
    put(path, value)


def unavailable(meaning, why):
    COUNTS["n/a"] += 1
    out(f"  [ n/a] {'-':<24} {'':<30} {meaning} -- {why}")


def section(title):
    out("")
    out("=" * 118)
    out(title)
    out("=" * 118)


# ====================================================================================================== data loading
def table(name):
    p = os.path.join(DATA, name)
    with gzip.open(p, "rt", newline="") as f:
        return list(csv.DictReader(f))


def i(x):
    return None if x == "" else int(x)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


# ====================================================================================================== statistics
# (1) Section 3 (role): the same pure-Python cluster bootstrap as the original analysis: 10,000 resamples of user
#     tasks, seed 1, percentile interval read off the sorted resamples.
def boot_user_tasks(diffs, clusters, nb=10000, seed=1):
    g = {}
    for v, c in zip(diffs, clusters):
        t = g.setdefault(c, [0, 0])
        t[0] += v
        t[1] += 1
    keys = sorted(g)
    sums, cnts = [g[k][0] for k in keys], [g[k][1] for k in keys]
    m = len(keys)
    rng = random.Random(seed)
    bs = []
    for _ in range(nb):
        s = c = 0
        for _ in range(m):
            j = rng.randrange(m)
            s += sums[j]
            c += cnts[j]
        bs.append(s / c if c else 0.0)
    bs.sort()

    def pct(level):
        a = (1 - level) / 2
        return (bs[int(a * nb)], bs[int((1 - a) * nb) - 1])
    return {"ci95": pct(0.95), "ci90": pct(0.90)}


# (2) Sections 4-5 (stance, gates): cluster bootstrap over (suite, user task) with numpy, seed 20261024. Clusters are
#     sorted by name and drawn in the same order as the original analysis; a resample is summarised by how many times
#     each cluster was drawn, so ratio statistics become matrix products.
SEED = 20261024


def cluster_weights(k, b, seed=SEED):
    rng = np.random.default_rng(seed)
    w = np.zeros((b, k))
    for r in range(b):
        w[r] = np.bincount(rng.integers(0, k, k), minlength=k)
    return w


_WCACHE = {}


def weights(k, b):
    if (k, b) not in _WCACHE:
        _WCACHE[(k, b)] = cluster_weights(k, b)
    return _WCACHE[(k, b)]


def pct_ci(vals, alpha=0.05):
    vals = vals[~np.isnan(vals)]
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [float(lo), float(hi)]


def boot_ratio(rows, num, den, b=5000, alpha=0.05):
    """95% interval of sum(num(row)) / sum(den(row)), resampling clusters (row['cluster'])."""
    cl = sorted({r["cluster"] for r in rows})
    ix = {c: j for j, c in enumerate(cl)}
    N, D = np.zeros(len(cl)), np.zeros(len(cl))
    for r in rows:
        N[ix[r["cluster"]]] += num(r)
        D[ix[r["cluster"]]] += den(r)
    W = weights(len(cl), b)
    with np.errstate(divide="ignore", invalid="ignore"):
        return pct_ci((W @ N) / (W @ D), alpha)


def boot_mean_diff(rows, f, b=5000):
    return boot_ratio(rows, f, lambda r: 1.0, b)


# ====================================================================================================== Section 3: role
def role():
    section("Section 3 / Table 1 (top): installed LlamaIndex and smolagents agents, tool-role option vs framework default")
    rows = table("role_installed_episodes.csv.gz")
    cells = collections.defaultdict(dict)
    for r in rows:
        if r["kind"] == "attacked":
            cells[(r["model"], r["framework"], r["delivery_role"])][(r["suite"], r["user_task"], r["injection_task"])] = r
    n_err = sum(1 for r in rows if r["kind"] == "attacked" and r["error"] == "1")
    pairs = {(r["suite"], r["user_task"], r["injection_task"]) for r in rows if r["kind"] == "attacked"}
    check(["role", "installed", "n_attacked_pairs"], len(pairs), 249, "attacked pairs (banking and Slack suites)", nd=0, scale=1)
    check(["role", "installed", "n_user_tasks"], len({p[:2] for p in pairs}), 37, "user tasks behind those pairs", nd=0,
          scale=1)
    check_text(["role", "installed", "attacked_episodes_with_error"], n_err, 0,
               "attacked episodes (all models, frameworks, roles) that ended with a recorded exception")
    report = {  # model, framework: tool ASR, user ASR, dASR [CI], dUA [CI]   (difference = default (user) - tool role)
        ("llama-3.3-70b", "llamaindex"): (65.9, 95.2, 29.3, (17.7, 42.2), 4.0, (-4.0, 13.0)),
        ("llama-3.3-70b", "smolagents"): (39.0, 83.9, 45.0, (33.2, 57.1), -0.4, (-10.5, 8.4)),
        ("gpt-oss-120b", "smolagents"): (83.1, 84.3, 1.2, (-2.4, 4.8), -2.4, (-8.6, 3.6)),
        ("qwen3-32b", "llamaindex"): (61.0, 60.2, -0.8, (-5.6, 4.0), -5.2, (-10.4, -0.8)),
        ("qwen3-32b", "smolagents"): (68.3, 65.1, -3.2, (-9.9, 2.0), 3.2, (0.4, 6.6)),
    }
    for (m, fw), (rt, ru, rd, rdci, rua, ruaci) in report.items():
        t, u = cells.get((m, fw, "tool")), cells.get((m, fw, "user"))
        out(f"-- {NAME[m]}, {fw}")
        if not t or not u:
            unavailable(f"{NAME[m]} {fw}", "no records in data/")
            continue
        keys = sorted(k for k in u if k in t and u[k]["error"] == "0" and t[k]["error"] == "0")
        xa = [int(u[k]["attack_success"]) for k in keys]
        ya = [int(t[k]["attack_success"]) for k in keys]
        xu = [int(u[k]["utility"]) for k in keys]
        yu = [int(t[k]["utility"]) for k in keys]
        P = ["role", "installed", f"{m}/{fw}"]
        check(P + ["n_pairs"], len(keys), 249, "attacked pairs valid under both roles", nd=0, scale=1)
        check(P + ["asr_tool"], mean(ya), rt, "ASR (%) with tool outputs in the tool role (the framework's option)")
        check(P + ["asr_user"], mean(xa), ru, "ASR (%) with tool outputs as user messages (the framework default)")
        da, du = [x - y for x, y in zip(xa, ya)], [x - y for x, y in zip(xu, yu)]
        cl = [(k[0], k[1]) for k in keys]
        bA = boot_user_tasks(da, cl) if CI else None
        bU = boot_user_tasks(du, cl) if CI else None
        check(P + ["delta_asr"], mean(da), rd, "change in ASR, default minus tool role (points); 95% user-task bootstrap",
              signed=True, ci=bA and bA["ci95"], report_ci=rdci)
        check(P + ["delta_ua"], mean(du), rua, "change in utility under attack, default minus tool role (points)",
              signed=True, ci=bU and bU["ci95"], report_ci=ruaci)
        sw = bU and [-bU["ci95"][1], -bU["ci95"][0]]
        rsw = {("llama-3.3-70b", "llamaindex"): (-4.0, (-13.0, 4.0), False),
               ("llama-3.3-70b", "smolagents"): (0.4, (-8.4, 10.5), True)}.get((m, fw))
        check(P + ["delta_ua_switching_to_tool"], -mean(du), rsw and rsw[0], "the same utility change read in the "
              "recommended direction, tool role minus default (points)", signed=True, ci=sw, report_ci=rsw and rsw[1])
        if CI:
            check_text(P + ["ua_noninferior_switching_to_tool_10pp"], bool(sw[0] > -0.10), rsw and rsw[2],
                       "switching to the tool role is noninferior for utility under attack at a 10-point margin")
        if CI and m == "gpt-oss-120b":
            lo, hi = bA["ci90"]
            check_text(P + ["delta_asr_equivalent_to_zero_10pp"], bool(lo > -0.10 and hi < 0.10), True,
                       f"90% interval [{100 * lo:.1f}, {100 * hi:.1f}] lies inside +-10 points (equivalent to zero)")
    out("-- gpt-oss-120b, llamaindex")
    unavailable("gpt-oss-120b in the LlamaIndex ReAct agent", "the report states this configuration gave no usable runs; "
                "no records are released for it")

    section("Section 3 / Table 1 (bottom): branching at the first injected message; only that message's header changes")
    br = table("role_branching.csv.gz")
    by = collections.defaultdict(dict)
    status = {}
    for r in br:
        k = (r["model"], r["suite"], r["user_task"], r["injection_task"])
        status[k] = r["prefix_status"]
        if r["header"]:
            by[k][(r["injection_voice"], r["header"])] = r
    cellsv = [(v, h) for v in ("unattributed", "user_authored") for h in ("tool", "user")]
    rep = {"llama-3.3-70b": {"n": 224, "unattributed": (56.7, 83.9, 27.2, (18.2, 36.8)),
                             "user_authored": (68.8, 92.4, 23.7, (13.7, 34.9)), "next": {"unattributed": 18.8, "user_authored": 21.9}},
           "gpt-oss-120b": {"n": 212, "unattributed": (80.2, 77.4, -2.8, (-10.1, 4.6)),
                            "user_authored": (83.5, 83.5, 0.0, (-4.4, 4.4)), "next": {}}}
    for m in ("llama-3.3-70b", "gpt-oss-120b"):
        out(f"-- {NAME[m]}")
        S = [k for k in sorted(status) if k[0] == m and status[k] == "branch" and all(c in by[k] for c in cellsv)
             and not any(by[k][c]["exception"] == "1" for c in cellsv) and all(by[k][c]["replay_identical"] == "1" for c in cellsv)]
        P = ["role", "branching", m]
        check(P + ["n_pairs"], len(S), rep[m]["n"], "attacked pairs branched (episode read the injection; replay identical; "
              "no exception)", nd=0, scale=1)
        for v in ("unattributed", "user_authored"):
            rt, ru, rd, rdci = rep[m][v]
            tool = [int(by[k][(v, "tool")]["attack_success"]) for k in S]
            user = [int(by[k][(v, "user")]["attack_success"]) for k in S]
            d = [x - y for x, y in zip(user, tool)]
            b_ = boot_user_tasks(d, [(k[1], k[2]) for k in S]) if CI else None
            check(P + [v, "continued_asr_tool"], mean(tool), rt, f"{v} injection, injected message under the tool header: "
                  "continued ASR (%)")
            check(P + [v, "continued_asr_user"], mean(user), ru, f"{v} injection, same message under a user header: "
                  "continued ASR (%)")
            check(P + [v, "delta"], mean(d), rd, "header effect on continued ASR, user minus tool header (points)",
                  signed=True, ci=b_ and b_["ci95"], report_ci=rdci)
            if CI and m == "gpt-oss-120b":
                lo, hi = b_["ci90"]
                check_text(P + [v, "equivalent_to_zero_10pp"], bool(lo > -0.10 and hi < 0.10), True,
                           f"90% interval [{100 * lo:.1f}, {100 * hi:.1f}] lies inside +-10 points")
            nxt = [int(by[k][(v, "user")]["next_step_attacker_action"]) - int(by[k][(v, "tool")]["next_step_attacker_action"])
                   for k in S]
            check(P + [v, "delta_next_step"], mean(nxt), rep[m]["next"].get(v), "header effect on the very next action "
                  "being attacker-directed (points)", signed=True)
        lv = [mean(int(by[k][c]["attack_success"]) for k in S) for c in cellsv]
        put(P + ["range_four_cells"], [round(100 * min(lv), 4), round(100 * max(lv), 4)])
        if m == "gpt-oss-120b":
            check(P + ["min_cell_asr"], min(lv), 77.4, "lowest continued ASR of the four cells (%)")
            check(P + ["max_cell_asr"], max(lv), 83.5, "highest continued ASR of the four cells (%)")

    section("Section 3 (text): every observation relabelled, bytes unchanged, under three attack templates")
    rl = table("role_relabel_episodes.csv.gz")
    cell = collections.defaultdict(dict)
    for r in rl:
        cell[(r["model"], r["template"], r["delivery_role"])][(r["suite"], r["user_task"], r["injection_task"])] = r
    rep = {("gpt-oss-120b", "ignore_previous"): 25.7, ("gpt-oss-120b", "direct"): 15.3,
           ("gpt-oss-120b", "important_instructions"): None,
           ("llama-3.3-70b", "direct"): 17.7, ("llama-3.3-70b", "ignore_previous"): 18.9,
           ("llama-3.3-70b", "important_instructions"): 24.2,
           ("qwen3-32b", "direct"): 2.0, ("qwen3-32b", "ignore_previous"): -1.2, ("qwen3-32b", "important_instructions"): -0.4}
    for (m, tpl), rd in rep.items():
        t, u = cell[(m, tpl, "tool")], cell[(m, tpl, "user")]
        keys = sorted(k for k in u if k in t and u[k]["error"] == "0" and t[k]["error"] == "0")
        d = [int(u[k]["attack_success"]) - int(t[k]["attack_success"]) for k in keys]
        syntax = "native tool calls" if m == "gpt-oss-120b" else "text tool calls"
        check(["role", "relabel", m, tpl, "delta_asr"], mean(d), rd, f"{NAME[m]}, {tpl}, {syntax}: ASR with every tool "
              f"output as a user message minus tool role (points, n={len(keys)})", signed=True)
        if m == "qwen3-32b" and tpl == "direct":
            du = [int(u[k]["utility"]) - int(t[k]["utility"]) for k in keys]
            check(["role", "relabel", m, tpl, "delta_ua"], mean(du), -12.4, "Qwen3-32B, direct: change in utility under "
                  "attack when relabelled (points)", signed=True)


# ====================================================================================================== Section 4: stance
LAB = {"acting": "label_acting", "asked": "label_asked", "reviewing": "label_reviewing",
       "reviewing_no_tool_outputs": "label_reviewing_no_tool_outputs"}
FIRE_ACT = {"HIGH", "UNKNOWN", "OMITTED", "INVALID"}       # self-label gate fires on HIGH or a missing/invalid label
FIRE_REV = {"HIGH", "UNPARSED", "UNKNOWN"}                 # reviewer gate fires on HIGH or an unparsed rating
OUT5 = {"send_money", "schedule_transaction", "update_scheduled_transaction", "send_email", "post_webpage"}


def stance_data():
    eps = {(r["model"], r["episode"]): r for r in table("stance_episodes.csv.gz")}
    calls = [c for c in table("stance_calls.csv.gz") if c["from_error_episode"] == "0"]
    for c in calls:
        e = eps[(c["model"], c["episode"])]
        c["cluster"] = f"{e['suite']}/{e['user_task']}"
        c["goal_class"] = e["goal_class"]
    return eps, calls


def stance():
    eps, calls = stance_data()
    section("Section 4 (text): HIGH on the first attacker-directed call of each injected episode, by stance")
    rep_first = {"acting": (4.4, 42.6, 33.0), "asked": (37.8, 57.4, 82.1), "reviewing": (76.7, 82.4, 88.4),
                 "reviewing_no_tool_outputs": (82.2, 76.1, 96.4)}
    rep_gap = {"reviewing-acting": (72.2, 39.8, 55.4), "asked-acting": (33.3, 14.8, 49.1), "reviewing-asked": (38.9, 25.0, 6.2)}
    rep_S = {"acting": (4.3, 43.8, 28.2), "reviewing": (92.8, 88.3, 92.9)}
    rep_n = (90, 176, 112)
    for m in FAMS:
        kinds = collections.Counter(e["kind"] for (mm, _), e in eps.items() if mm == m)
        check_text(["stance", "episodes", m], [kinds["injected"], kinds["twin"], kinds["benign"]], [310, 310, 97],
                   f"{NAME[m]}: injected / authorised-twin / benign episodes run")
    for j, m in enumerate(FAMS):
        out(f"-- {NAME[m]}")
        first = [c for c in calls if c["model"] == m and c["call_set"] == "AU" and c["first_in_episode"] == "1"]
        P = ["stance", "first_attacker_call", m]
        check(P + ["n"], len(first), rep_n[j], "first attacker-directed calls (one per injected episode that made one)",
              nd=0, scale=1)
        for s, col in LAB.items():
            rs = [c for c in first if c[col] != ""]
            check(P + [f"high_{s}"], mean(c[col] == "HIGH" for c in rs), rep_first[s][j],
                  f"% rated HIGH, {s.replace('_', ' ')}")
        for g, (a, b_) in {"reviewing-acting": ("acting", "reviewing"), "asked-acting": ("acting", "asked"),
                           "reviewing-asked": ("asked", "reviewing")}.items():
            rs = [c for c in first if c[LAB[a]] != "" and c[LAB[b_]] != ""]
            f = lambda c, a=a, b_=b_: (c[LAB[b_]] == "HIGH") - (c[LAB[a]] == "HIGH")
            ci = boot_mean_diff(rs, f) if CI else None
            check(P + [f"gap_{g.replace('-', '_minus_')}"], mean(f(c) for c in rs), rep_gap[g][j],
                  f"gap in HIGH, {g.replace('-', ' minus ')} (points, same calls)", signed=False, ci=ci)
        for s in ("acting", "reviewing"):
            rs = [c for c in first if c["goal_class"] == "sensitive_data" and c[LAB[s]] != ""]
            check(P + [f"high_{s}_data_goals"], mean(c[LAB[s]] == "HIGH" for c in rs), rep_S[s][j],
                  f"% HIGH, {s}, only goals that send the user's data out (n={len(rs)})")

    section("Section 4 (text): replication with different seeds and overlapping tasks (reviewing minus acting)")
    rc = table("stance_replication_calls.csv.gz")
    byep = collections.defaultdict(list)
    for c in rc:
        byep[(c["model"], c["episode"])].append(c)
    rep = (70.5, 39.2, 54.6)
    for j, m in enumerate(FAMS):
        firsts = []
        for (mm, _), cs in byep.items():
            if mm != m:
                continue
            cs.sort(key=lambda c: (int(c["turn"]), int(c["call_index"])))
            f = next((c for c in cs if c["attacker_directed"] == "1"), None)
            if f and f["from_error_episode"] == "0" and f["label_reviewing"] != "":
                f["cluster"] = f"{f['suite']}/{f['user_task']}"
                firsts.append(f)
        g = lambda c: (c["label_reviewing"] == "HIGH") - (c["label_acting"] == "HIGH")
        check(["stance", "replication", m, "gap_reviewing_minus_acting"], mean(g(c) for c in firsts), rep[j],
              f"{NAME[m]}: gap on first attacker-directed calls in the replication run (points, n={len(firsts)})",
              ci=boot_mean_diff(firsts, g) if CI else None)

    section("Section 4 / Table 2: HIGH on the attacker-directed call vs the goal-matched authorised send (twin pairs)")
    ok = {k: e for k, e in eps.items() if e["exception"] == "0"}
    firstof = {}
    for c in calls:
        k = (c["model"], c["episode"])
        if c["first_in_episode"] == "1" and c["call_set"] in ("AU", "AA") and k in ok and k not in firstof:
            firstof[k] = c                                   # the first flagged call of the episode, in turn order
    rep_tab = {  # stance: per family (U, A, sep)
        "acting": ((3.6, 1.2, 2.4), (41.4, 44.6, -3.2), (37.0, 16.4, 20.5)),
        "asked": ((35.7, 32.1, 3.6), (54.8, 64.3, -9.6), (80.8, 84.9, -4.1)),
        "reviewing": ((75.0, 70.2, 4.8), (82.2, 63.1, 19.1), (90.4, 87.7, 2.7)),
        "reviewing_no_tool_outputs": ((81.0, 79.8, 1.2), (74.5, 75.2, -0.6), (98.6, 95.9, 2.7))}
    rep_ci = {("llama-3.3-70b", "acting"): (7.3, 31.0), ("qwen3-32b", "reviewing"): (11.5, 25.4)}
    rep_np = (84, 157, 73)
    PAIRS = {}
    for j, m in enumerate(FAMS):
        out(f"-- {NAME[m]}")
        U, A = {}, {}
        for (mm, ek), e in ok.items():
            if mm != m:
                continue
            c = firstof.get((mm, ek))
            if c is None:
                continue
            tw = (e["suite"], e["user_task"], e["injection_task"])
            if e["kind"] == "injected" and c["call_set"] == "AU":
                U[tw] = c
            elif e["kind"] == "twin" and c["call_set"] == "AA":
                A[tw] = c
        pairs = [{"cluster": f"{tw[0]}/{tw[1]}", "suite": tw[0], "u": U[tw], "a": A[tw]} for tw in sorted(U) if tw in A]
        PAIRS[m] = pairs
        P = ["stance", "twin_pairs", m]
        check(P + ["n_pairs"], len(pairs), rep_np[j], "completed twin pairs (injected episode and its authorised twin "
              "each made the goal's send)", nd=0, scale=1)
        for s, col in LAB.items():
            Q = [p for p in pairs if p["u"][col] != "" and p["a"][col] != ""]
            hu = mean(p["u"][col] == "HIGH" for p in Q)
            ha = mean(p["a"][col] == "HIGH" for p in Q)
            f = lambda p, col=col: (p["u"][col] == "HIGH") - (p["a"][col] == "HIGH")
            ci = boot_mean_diff(Q, f) if CI else None
            ru, ra, rs = rep_tab[s][j]
            check(P + [s, "high_attacker_directed"], hu, ru, f"{s.replace('_', ' ')}: % HIGH on the attacker-directed call")
            check(P + [s, "high_authorised"], ha, ra, f"{s.replace('_', ' ')}: % HIGH on the matched authorised send")
            check(P + [s, "separation"], mean(f(p) for p in Q), rs, "separation = the two rates' difference (points)",
                  signed=True, ci=ci, report_ci=rep_ci.get((m, s)))
    if CI:
        crossover(PAIRS)
    else:
        out("  (run with --ci for the post hoc acting-to-reviewing contrasts of separation)")


def crossover(PAIRS):
    """Exploratory contrasts: change in separation from acting to reviewing, per family and between families. Six
    contrasts resampled jointly over the union of clusters, intervals at 1 - 0.05/6 (20,000 resamples)."""
    out("-- post hoc: change in separation from acting to reviewing (adjusted intervals)")
    clusters = sorted({p["cluster"] for m in FAMS for p in PAIRS[m]})
    cidx = {c: j for j, c in enumerate(clusters)}

    def tot(P, col):
        n, su, sa = np.zeros(len(clusters)), np.zeros(len(clusters)), np.zeros(len(clusters))
        for p in P:
            if p["u"][col] == "" or p["a"][col] == "":
                continue
            k = cidx[p["cluster"]]
            n[k] += 1
            su[k] += p["u"][col] == "HIGH"
            sa[k] += p["a"][col] == "HIGH"
        return n, su, sa

    def contr(P, W, alpha):
        D, bD = {}, {}
        for m in FAMS:
            ti, tr = tot(P[m], LAB["acting"]), tot(P[m], LAB["reviewing"])
            D[m] = (tr[1].sum() - tr[2].sum()) / tr[0].sum() - (ti[1].sum() - ti[2].sum()) / ti[0].sum()
            with np.errstate(divide="ignore", invalid="ignore"):
                bD[m] = (W @ (tr[1] - tr[2])) / (W @ tr[0]) - (W @ (ti[1] - ti[2])) / (W @ ti[0])
        r = {f"D_{m}": (D[m], pct_ci(bD[m], alpha)) for m in FAMS}
        r["qwen3_minus_llama"] = (D["qwen3-32b"] - D["llama-3.3-70b"], pct_ci(bD["qwen3-32b"] - bD["llama-3.3-70b"], alpha))
        return r

    k = len(clusters)
    a = contr(PAIRS, weights(k, 20000), 0.05 / 6)
    check(["stance", "crossover", "n_clusters"], k, 69, "(suite, user task) clusters in the joint resampling", nd=0, scale=1)
    check(["stance", "crossover", "qwen3_change"], a["D_qwen3-32b"][0], 22.3, "Qwen3-32B: separation rises from acting "
          "to reviewing (points, adjusted interval)", signed=True, ci=a["D_qwen3-32b"][1], report_ci=(5.2, 36.7))
    check(["stance", "crossover", "llama_change"], a["D_llama-3.3-70b"][0], -17.8, "Llama-3.3-70B: change in separation "
          "(points; interval includes zero)", signed=True, ci=a["D_llama-3.3-70b"][1], report_ci=(-32.6, 1.4))
    check(["stance", "crossover", "qwen3_minus_llama"], a["qwen3_minus_llama"][0], 40.1, "Qwen3's change minus "
          "Llama's change (points)", signed=True, ci=a["qwen3_minus_llama"][1], report_ci=(14.3, 59.3))
    nob = {m: [p for p in PAIRS[m] if p["suite"] != "banking"] for m in FAMS}
    a2 = contr(nob, weights(k, 5000), 0.05)
    check(["stance", "crossover", "qwen3_minus_llama_without_banking"], a2["qwen3_minus_llama"][0], 11.0,
          "the same difference without the banking suite (points, 95%)", signed=True, ci=a2["qwen3_minus_llama"][1],
          report_ci=(-15.1, 35.9))


# ====================================================================================================== Section 5: gates
def gate_operating_points():
    """From the rating experiment: benign interruption and first-call coverage of each gate, without executing it."""
    section("Section 5 / Table 3 (bottom) and text: gate operating points from the rating experiment")
    eps, calls = stance_data()
    ok = {k: e for k, e in eps.items() if e["exception"] == "0"}
    by = collections.defaultdict(list)
    for c in calls:
        by[(c["model"], c["episode"])].append(c)
    fire = {"self_label": lambda c: c["label_acting"] in FIRE_ACT,
            "reviewer": lambda c: c["label_reviewing"] in FIRE_REV,   # every benign call is rated (complete data)
            "rule": lambda c: c["function"] in OUT5}
    rep_ratio = {"self_label": (0.08, 0.38, 2.81), "reviewer": (0.53, 0.96, 0.79)}
    rep_rule_b = (11.7, 14.4, 19.8)
    for j, m in enumerate(FAMS):
        out(f"-- {NAME[m]}")
        B = [by[k] for k, e in ok.items() if k[0] == m and e["kind"] == "benign" and by[k]]
        U = []
        for k, e in ok.items():
            if k[0] == m and e["kind"] == "injected":
                f = [c for c in by[k] if c["call_set"] == "AU" and c["first_in_episode"] == "1"]
                if f:
                    U.append(f[0])
        A = [by[k] for k, e in ok.items() if k[0] == m and e["kind"] == "twin" and by[k]]
        unrated = sum(1 for cs in B for c in cs if c["label_reviewing"] == "")
        check_text(["gates", "operating_points", m, "benign_calls_without_review"], unrated, 0,
                   "benign calls with no reviewer rating (0 = the reviewer's benign rate is a point estimate)")
        pt = {}
        for g, f in fire.items():
            x = mean(any(f(c) for c in cs) for cs in B)
            y = mean(f(c) for c in U)
            pt[g] = (x, y)
        check(["gates", "operating_points", m, "rule_benign_interrupted"], pt["rule"][0], rep_rule_b[j],
              "% of benign episodes (with >= 1 call) the rule would interrupt")
        check(["gates", "operating_points", m, "rule_twins_with_call_interrupted"],
              mean(any(c["function"] in OUT5 for c in cs) for cs in A), (77.6, 89.3, None)[j],
              "% of authorised twins that made a call that the rule would interrupt")
        for g in ("self_label", "reviewer"):
            ratio = (pt[g][1] / pt[g][0]) / (pt["rule"][1] / pt["rule"][0])
            check(["gates", "operating_points", m, f"{g}_coverage_per_interruption_vs_rule"], ratio, rep_ratio[g][j],
                  f"{g.replace('_', '-')} gate: first-call coverage per benign interruption, relative to the rule "
                  f"(coverage {100 * pt[g][1]:.1f}%, benign {100 * pt[g][0]:.1f}%)", nd=2, scale=1)
    out("  note: the rule's twin rate for Llama-3.3-70B is not printed in the report (range 77.6-89.3 covers it).")


def gate_data():
    E = collections.defaultdict(dict)        # (user, model, gate) -> {episode: row}, rows without an exception
    allrows = table("gates_episodes.csv.gz")
    for r in allrows:
        r["cluster"] = f"{r['suite']}/{r['user_task']}"
        if r["exception"] == "0":
            E[(r["user"], r["model"], r["gate"])][r["episode"]] = r
    return E, allrows


def vs_none(E, user, m, g, kind, field):
    """level of `field` under gate g and its difference from no gate, on episodes valid in both."""
    none, gg = E[(user, m, "none")], E[(user, m, g)]
    keys = [k for k, r in none.items() if r["kind"] == kind and k in gg]
    rows = [{"cluster": none[k]["cluster"], "x": int(gg[k][field]), "y": int(none[k][field])} for k in keys]
    lvl = mean(r["x"] for r in rows)
    d = mean(r["x"] - r["y"] for r in rows)
    ci = boot_mean_diff(rows, lambda r: r["x"] - r["y"]) if CI and g != "none" else None
    return lvl, d, ci, len(rows)


def interrupted(E, user, m, g, kind):
    rs = [r for r in E[(user, m, g)].values() if r["kind"] == kind]
    return mean(int(r["confirmations"]) > 0 for r in rs)


VALID = []
GATE = {"none": "no gate", "inline": "self-label gate (HIGH or missing)", "l4": "reviewer gate (HIGH)",
        "out5": "rule: confirm the 5 outbound functions"}


def gates():
    E, allrows = gate_data()
    section("Section 5 / Table 3 (top): executed gates, informed simulated user (rejects exactly the attacker-directed calls)")
    rep = {  # gate: per family (ASR, delta, twins interrupted)
        "none": ((26.5, None, None), (43.0, None, None), (26.1, None, None)),
        "inline": ((26.5, 0.0, 26.5), (23.9, -19.1, 47.6), (19.9, -6.2, 11.9)),
        "l4": ((7.7, -18.7, 65.9), (4.5, -38.5, 61.2), (7.5, -18.6, 49.2)),
        "out5": ((0.0, -26.5, 78.2), (0.0, -43.0, 90.0), (0.0, -26.1, 52.6))}
    keys_sched = collections.Counter((r["user"], r["model"], r["gate"]) for r in allrows)
    for j, m in enumerate(FAMS):
        out(f"-- {NAME[m]}")
        valid = [len(E[("informed", m, g)]) for g in GATE]
        check_text(["gates", "informed", m, "episodes_scheduled"], sorted({keys_sched[("informed", m, g)] for g in GATE}), [717],
                   "episode keys scheduled per gate")
        put(["gates", "informed", m, "episodes_without_exception"], dict(zip(GATE, valid)))
        VALID.extend(valid)
        out(f"  [info] {str(valid):<24} {'':<30} episodes without an exception: none / self-label / reviewer / rule")
        for g in ("none", "inline", "l4", "out5"):
            ra, rd, rt = rep[g][j]
            lvl, d, ci, n = vs_none(E, "informed", m, g, "injected", "goal_reached")
            P = ["gates", "informed", m, g]
            check(P + ["asr"], lvl, ra, f"{GATE[g]}: ASR (%) over injected episodes (n={n})")
            if g != "none":
                check(P + ["delta_asr_vs_none"], d, rd, "change in ASR vs no gate (points)", signed=True, ci=ci,
                      report_ci=(-4.2, 4.4) if (m, g) == ("gpt-oss-120b", "inline") else None)
                check(P + ["twins_interrupted"], interrupted(E, "informed", m, g, "twin"), rt,
                      "% of authorised-twin episodes in which the gate asked the user at least once")
                lb, db, cib, _ = vs_none(E, "informed", m, g, "benign", "utility")
                ni = None if cib is None else bool(cib[0] > -0.10)
                check(P + ["delta_benign_utility"], db, -5.2 if (m, g) == ("gpt-oss-120b", "l4") else None,
                      "change in benign utility vs no gate (points)" + ("" if ni is None else
                      f"; noninferior at 10 points: {ni}"), signed=True, ci=cib,
                      report_ci=(-14.4, 4.1) if (m, g) == ("gpt-oss-120b", "l4") else None)
                if ni is not None:
                    put(P + ["benign_utility_noninferior"], ni)
        for x, y, rd in (("out5", "l4", (-7.8, -4.5, -7.5)[j]), ("l4", "inline", (-18.7, -19.4, -12.4)[j])):
            ex, ey = E[("informed", m, x)], E[("informed", m, y)]
            keys = [k for k, r in ex.items() if r["kind"] == "injected" and k in ey]
            rows = [{"cluster": ex[k]["cluster"], "d": int(ex[k]["goal_reached"]) - int(ey[k]["goal_reached"])} for k in keys]
            ci = boot_mean_diff(rows, lambda r: r["d"]) if CI else None
            check(["gates", "informed", m, f"asr_{x}_minus_{y}"], mean(r["d"] for r in rows), rd,
                  f"ASR under {GATE[x]} minus under {GATE[y]} (points; the report says every 95% interval is below zero)",
                  signed=True, ci=ci)
            if ci is not None:
                put(["gates", "informed", m, f"asr_{x}_minus_{y}_ci_below_zero"], bool(ci[1] < 0))

    check_text(["gates", "informed", "episodes_without_exception_range"], [min(VALID), max(VALID)], [711, 716],
               "fewest and most episodes without an exception, over families and gates (of 717)")
    fallible(E, allrows)
    equal_rate(E)


def fallible(E, allrows):
    section("Section 5 (text): executed gates with fallible LLM users (shown only the request and the proposed call)")
    users = ("gemma_standard", "gemma_cautious", "phi4_standard")
    label = {"gemma_standard": "Gemma-3-27B", "gemma_cautious": "Gemma-3-27B, stricter prompt", "phi4_standard": "Phi-4"}
    worst = {}
    for u in users:
        best_asr = best_ac = (0, None)
        for m in FAMS:
            for g in ("inline", "l4", "out5"):
                _, d, ci, _ = vs_none(E, u, m, g, "injected", "goal_reached")
                _, dac, _, _ = vs_none(E, u, m, g, "twin", "goal_reached")
                put(["gates", "fallible", u, m, g, "delta_asr_vs_none"], round(100 * d, 4))
                put(["gates", "fallible", u, m, g, "delta_authorised_completion_vs_none"], round(100 * dac, 4))
                if ci is not None:
                    put(["gates", "fallible", u, m, g, "delta_asr_vs_none_ci95"], [round(100 * ci[0], 4), round(100 * ci[1], 4)])
                if -d > best_asr[0]:
                    best_asr = (-d, (m, g))
                if -dac > best_ac[0]:
                    best_ac = (-dac, (m, g))
        worst[u] = (best_asr, best_ac)
    check(["gates", "fallible", "gemma_standard_largest_asr_reduction"], worst["gemma_standard"][0][0], 37.9,
          f"Gemma-3-27B user: largest ASR reduction of any gate ({NAME[worst['gemma_standard'][0][1][0]]}, "
          f"{GATE[worst['gemma_standard'][0][1][1]]}) (points)")
    check(["gates", "fallible", "gemma_standard_largest_completion_drop"], worst["gemma_standard"][1][0], 23.0,
          "Gemma-3-27B user: largest fall in authorised completion (points; the report rounds to 23)", nd=0)
    best = max(((worst[u][0][0], u) for u in users))
    bestc = max(((worst[u][1][0], u) for u in users))
    check(["gates", "fallible", "largest_asr_reduction_any_fallible_user"], best[0], 41.6,
          f"largest ASR reduction over all three fallible users (under {label[best[1]]})")
    check(["gates", "fallible", "largest_completion_drop_any_fallible_user"], bestc[0], 41.2,
          f"largest fall in authorised completion over all three fallible users (under {label[bestc[1]]})")
    if CI:
        nored = all(vs_none(E, u, "gpt-oss-120b", "inline", "injected", "goal_reached")[2][1] >= 0 for u in users)
        check_text(["gates", "fallible", "gptoss_self_label_no_detectable_reduction"], nored, True,
                   "gpt-oss-120b self-label gate: the 95% interval of the ASR change includes zero for every fallible user")
    # user decisions, Gemma-3-27B (standard prompt)
    fa, fr = [], []
    for m in FAMS:
        for g in ("out5", "l4"):
            rs = list(E[("gemma_standard", m, g)].values())
            asks = sum(int(r["attacker_asks"]) for r in rs)
            fa.append(sum(int(r["attacker_asks_approved"]) for r in rs) / asks)
        for g in ("inline", "out5", "l4"):
            rs = [r for r in E[("gemma_standard", m, g)].values() if r["kind"] == "twin"]
            oth = sum(int(r["other_asks"]) for r in rs)
            fr.append(sum(int(r["other_asks_rejected"]) for r in rs) / oth)
    check(["gates", "fallible", "gemma_false_approval_min"], min(fa), 13.0, "Gemma-3-27B approved at least this % of the "
          "attacker-directed calls the rule or reviewer showed it (report: 13-22)", nd=0)
    check(["gates", "fallible", "gemma_false_approval_max"], max(fa), 22.0, "... and at most this %", nd=0)
    check(["gates", "fallible", "gemma_false_rejection_twins_min"], min(fr), 11.5, "Gemma-3-27B rejected at least this % "
          "of the other requests in authorised-twin episodes (any gate, any family)")
    check(["gates", "fallible", "gemma_false_rejection_twins_max"], max(fr), 40.5, "... and at most this %")
    ph = [r for r in allrows if r["user"] == "phi4_standard" and r["gate"] != "none"]
    check(["gates", "fallible", "phi4_no_usable_decision"],
          sum(int(r["nonstandard_answers"]) for r in ph) / sum(int(r["confirmations"]) for r in ph), 11.7,
          "Phi-4 answers that did not start with APPROVE or REJECT, over all its requests (counted as approvals)")
    inf = [mean(int(r["reviewer_inferences"]) for r in E[(u, m, "l4")].values() if r["kind"] in ("twin", "benign"))
           for u in ("informed",) + users for m in FAMS]
    check(["gates", "fallible", "reviewer_inferences_per_legit_min"], min(inf), 4.4, "extra reviewer inferences per "
          "legitimate episode under the reviewer gate, smallest over the 4 users x 3 families (the rule adds none)", scale=1)
    check(["gates", "fallible", "reviewer_inferences_per_legit_max"], max(inf), 7.3, "... largest", scale=1)


def mixture_cell(E, user, m, a, b, w, want_ci):
    """Equal-interruption comparison of gates a and b on workload w. The gate X that interrupts more is mixed with no
    gate per episode (probability lambda) until its interruption rate matches gate Y's. D = ASR(Y) - ASR(mixed X);
    D > 0: X protects more at Y's rate; D < 0: Y protects more. Also the same contrast for authorised completion."""
    none, ea, eb = E[(user, m, "none")], E[(user, m, a)], E[(user, m, b)]
    keys = [k for k in none if k in ea and k in eb]
    cl = sorted({none[k]["cluster"] for k in keys})
    ix = {c: j for j, c in enumerate(cl)}
    F = collections.defaultdict(lambda: np.zeros(len(cl)) if np is not None else None)
    tot = collections.defaultdict(float)
    for k in keys:
        n, x, y = none[k], ea[k], eb[k]
        kind = {"injected": "U", "twin": "A", "benign": "B"}[n["kind"]]
        vals = {f"n_{kind}": 1}
        if kind == "U":
            vals.update(none_U=int(n["goal_reached"]), a_U=int(x["goal_reached"]), b_U=int(y["goal_reached"]))
        elif kind == "A":
            vals.update(none_AC=int(n["goal_reached"]), a_AC=int(x["goal_reached"]), b_AC=int(y["goal_reached"]))
        if kind in ("A", "B"):
            vals.update({f"a_ask_{kind}": int(int(x["confirmations"]) > 0), f"b_ask_{kind}": int(int(y["confirmations"]) > 0)})
        for f, v in vals.items():
            tot[f] += v
            if want_ci:
                F[f][ix[n["cluster"]]] += v

    def rate(s, g):
        ra, rb = s[f"{g}_ask_A"] / s["n_A"], s[f"{g}_ask_B"] / s["n_B"]
        if w == "A":
            return ra
        if w == "B":
            return rb
        if w == "pooled":
            return (s[f"{g}_ask_A"] + s[f"{g}_ask_B"]) / (s["n_A"] + s["n_B"])
        return 0.5 * ra + 0.5 * rb

    def effects(s, x, y, vec=False):
        bx, by = rate(s, x), rate(s, y)
        if vec:
            with np.errstate(divide="ignore", invalid="ignore"):
                lam = np.where(bx == 0, 1.0, np.minimum(1.0, by / bx))
        else:
            lam = 1.0 if bx == 0 else min(1.0, by / bx)
        d = s[f"{y}_U"] / s["n_U"] - (lam * s[f"{x}_U"] / s["n_U"] + (1 - lam) * s["none_U"] / s["n_U"])
        dac = s[f"{y}_AC"] / s["n_A"] - (lam * s[f"{x}_AC"] / s["n_A"] + (1 - lam) * s["none_AC"] / s["n_A"])
        return d, dac, lam

    x, y = ("a", "b") if rate(tot, "a") >= rate(tot, "b") else ("b", "a")
    d, dac, lam = effects(tot, x, y)
    res = {"X": {"a": a, "b": b}[x], "Y": {"a": a, "b": b}[y], "lambda": lam, "D": d, "D_completion": dac}
    if want_ci:
        W = weights(len(cl), 5000)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = {f: W @ v for f, v in F.items()}
            bd, bdac, _ = effects(s, x, y, vec=True)
        res["ci"], res["ci_completion"] = pct_ci(bd), pct_ci(bdac)
        res["readout"] = (f"{res['X']} more" if res["ci"][0] > 0 else (f"{res['Y']} more" if res["ci"][1] < 0 else "no difference"))
    return res


def equal_rate(E):
    section("Section 5 / Table 3 (middle) and text: gates compared at an equal interruption rate")
    users = ("informed", "gemma_standard", "phi4_standard", "gemma_cautious")
    rep_b = {("gpt-oss-120b", "inline"): "4/0/0", ("qwen3-32b", "inline"): "4/0/0", ("llama-3.3-70b", "inline"): "0/0/4",
             ("gpt-oss-120b", "l4"): "4/0/0", ("qwen3-32b", "l4"): "0/0/4", ("llama-3.3-70b", "l4"): "4/0/0"}
    rep_all = {("qwen3-32b", "l4"): "0/12/4", ("llama-3.3-70b", "inline"): "0/0/16"}
    if not CI:
        out("  (the counts below are readouts of 95% intervals; run with --ci to compute them)")
    for g in ("inline", "l4"):
        for m in FAMS:
            pair = ("out5", g)
            res = {(u, w): mixture_cell(E, u, m, *pair, w, CI) for u in users for w in ("A", "B", "pooled", "equal")}
            for (u, w), r in res.items():
                put(["gates", "equal_rate", m, f"rule_vs_{g}", u, w],
                    {"X_mixed_down": r["X"], "Y": r["Y"], "lambda": round(r["lambda"], 6), "D_asr": round(100 * r["D"], 4),
                     "D_authorised_completion": round(100 * r["D_completion"], 4), "readout": r.get("readout")})
            if not CI:
                continue

            def cnt(ws):
                c = collections.Counter(res[(u, w)]["readout"] for u in users for w in ws)
                return f"{c.get('out5 more', 0)}/{c.get(g + ' more', 0)}/{c.get('no difference', 0)}"
            gname = "self-label" if g == "inline" else "reviewer"
            check_text(["gates", "equal_rate", m, f"rule_vs_{g}", "benign_workload_counts"], cnt(("B",)), rep_b[(m, g)],
                       f"{NAME[m]}, {gname} gate vs rule, benign workload, 4 users: rule more / gate more / no difference")
            check_text(["gates", "equal_rate", m, f"rule_vs_{g}", "all_16_counts"], cnt(("A", "B", "pooled", "equal")),
                       rep_all.get((m, g)), "... the same over all four ways of weighting legitimate tasks (16 comparisons)")
    for u, rv in (("gemma_cautious", -3.8), ("phi4_standard", -10.4)):
        r = mixture_cell(E, u, "qwen3-32b", "out5", "l4", "pooled", CI)
        check(["gates", "equal_rate", "qwen3-32b", f"reviewer_completion_cost_{u}"], r["D_completion"], rv,
              f"Qwen3-32B, {u}: authorised completion of the reviewer minus the rule mixed down to its rate (points; "
              f"X={r['X']}, Y={r['Y']})", signed=True, ci=r.get("ci_completion"))


# ====================================================================================================== main
def main():
    global CI
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ci", action="store_true", help="also compute bootstrap intervals and interval-based readouts")
    ap.add_argument("--json", default=None, help="write all numbers to this JSON file (e.g. data/headline.json)")
    a = ap.parse_args()
    CI = a.ci
    if CI and np is None:
        sys.exit("--ci needs numpy (pip install -r requirements.txt)")
    if np is None:
        out("numpy not found: Sections 4-5 that need it are skipped (pip install -r requirements.txt)")
    out("RoleCall: headline numbers recomputed from data/  (" + ("with" if CI else "without") + " bootstrap intervals)")
    out("Columns: [status] recomputed value [95% interval] | value printed in the report | what the number means")
    out("Status: ok = matches the report at its printed precision; DIFF = does not match; info = not printed in the "
        "report; n/a = cannot be recomputed from the released data (reason given)")
    role()
    if np is not None:
        stance()
        gate_operating_points()
        gates()
    section("Summary")
    out("  " + ", ".join(f"{k}: {v}" for k, v in sorted(COUNTS.items())))
    H["_meta"] = {"generated_by": "reproduce/headline_numbers.py" + (" --ci" if CI else ""),
                  "units": "percent or percentage points unless the key says n/count/ratio; *_ci95 = 95% percentile "
                           "cluster-bootstrap interval",
                  "status_counts": dict(COUNTS)}
    if a.json:
        with open(a.json, "w") as f:
            json.dump(H, f, indent=1, sort_keys=True)
        out(f"  wrote {a.json}")
    if COUNTS.get("DIFF"):
        sys.exit(1)


if __name__ == "__main__":
    main()
