"""Second fallible-user round: cost at the equal interruption rate and direct contrasts (pre-registered).
usage: python -I analyze_matched_cost.py <out prefix> <frozen workload.json> <label>=<root> [<label>=<root> ...]
       each root holds <family>/<gate>/episodes.jsonl (an analysis view), as for analyze_workload.py
Every cell keeps analyze_workload.py's pairing and orientation. Per-cluster totals make the bootstrap a matrix product, so
one draw of clusters serves all conditions (needed for contrasts across users and families).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stance"))
from analyze_fallible import CFGS, PAIRS, cl, load  # noqa: E402

SEED = 20261024
B_CELL, B_ADJ = 5000, 20000
N_CONTRASTS = 33
WORKLOADS = ("A", "B", "pooled", "equal")
USERS_VS = ("informed", "phi4_standard", "gemma_cautious")
REF_USER = "gemma_standard"
MODEL_CONTRASTS = (("qwen3", "gptoss"), ("qwen3", "llama33"), ("gptoss", "llama33"))
FIELDS = ("n_U", "n_A", "n_B", "none_U", "a_U", "b_U", "none_AC", "a_AC", "b_AC", "none_BU", "a_BU", "b_BU",
          "a_ask_A", "b_ask_A", "a_ask_B", "b_ask_B", "a_conf_A", "b_conf_A", "a_conf_B", "b_conf_B")


def draws(n_clusters, b):
    """Cluster multiplicities per resample, in the order stats_common.cluster_boot draws them."""
    rng = np.random.default_rng(SEED)
    w = np.zeros((b, n_clusters), dtype=np.int32)
    for i in range(b):
        w[i] = np.bincount(rng.integers(0, n_clusters, n_clusters), minlength=n_clusters)
    return w


def totals(E, a, b, cidx):
    """Per-cluster totals of one cell. Clusters absent from the cell stay zero."""
    keys = [k for k in E["none"] if k in E[a] and k in E[b]]
    T = {f: np.zeros(len(cidx)) for f in FIELDS}
    for k in keys:
        n, x, y, c = E["none"][k], E[a][k], E[b][k], cidx[cl(E["none"][k])]
        kind = n["kind"]
        T[f"n_{kind}"][c] += 1
        if kind == "U":
            T["none_U"][c] += bool(n["goal"]); T["a_U"][c] += bool(x["goal"]); T["b_U"][c] += bool(y["goal"])
        elif kind == "A":
            T["none_AC"][c] += bool(n["goal"]); T["a_AC"][c] += bool(x["goal"]); T["b_AC"][c] += bool(y["goal"])
        else:
            T["none_BU"][c] += bool(n["utility"]); T["a_BU"][c] += bool(x["utility"]); T["b_BU"][c] += bool(y["utility"])
        if kind in ("A", "B"):
            T[f"a_ask_{kind}"][c] += bool(x["confirmations"]); T[f"b_ask_{kind}"][c] += bool(y["confirmations"])
            T[f"a_conf_{kind}"][c] += len(x["confirmations"]); T[f"b_conf_{kind}"][c] += len(y["confirmations"])
    return T, len(keys)


def rates(s, g, w):
    """Interruption rate and confirmations per legitimate episode of gate slot g on workload w. s maps field -> sums."""
    ra, rb = s[f"{g}_ask_A"] / s["n_A"], s[f"{g}_ask_B"] / s["n_B"]
    ca, cb = s[f"{g}_conf_A"] / s["n_A"], s[f"{g}_conf_B"] / s["n_B"]
    if w == "A":
        return ra, ca
    if w == "B":
        return rb, cb
    if w == "pooled":
        n = s["n_A"] + s["n_B"]
        return (s[f"{g}_ask_A"] + s[f"{g}_ask_B"]) / n, (s[f"{g}_conf_A"] + s[f"{g}_conf_B"]) / n
    return 0.5 * ra + 0.5 * rb, 0.5 * ca + 0.5 * cb


def effects(s, x, y, w):
    """x, y are slots 'a'/'b'. Returns arrays (D, dAC, dBU, dConf, lambda)."""
    bx, cx = rates(s, x, w)
    by, cy = rates(s, y, w)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where(bx == 0, 1.0, np.minimum(1.0, by / bx))
    mix = lambda f, n: s[f"{y}_{f}"] / s[n] - (lam * s[f"{x}_{f}"] / s[n] + (1 - lam) * s[f"none_{f}"] / s[n])
    return mix("U", "n_U"), mix("AC", "n_A"), mix("BU", "n_B"), cy - lam * cx, lam


def pct_ci(v, alpha):
    v = v[~np.isnan(v)]
    return [float(x) for x in np.percentile(v, [100 * alpha / 2, 100 * (1 - alpha / 2)])]


def gate_cost(E):
    out = {}
    for g in ("inline", "out5", "l4"):
        lg = [r for r in E[g].values() if r["kind"] in ("A", "B")]
        conf = np.array([len(r["confirmations"]) for r in lg], dtype=float)
        row = {"conf_per_legit": float(conf.mean()), "share_two_or_more": float((conf >= 2).mean()), "n_legit": len(lg)}
        for kind in ("A", "B"):
            c = [len(r["confirmations"]) for r in lg if r["kind"] == kind]
            row[f"conf_per_{kind}"] = float(np.mean(c))
        if g == "l4":
            row["reviewer_inferences_per_legit"] = float(np.mean([sum(len(c.get("calls") or []) for c in r["llm_calls"]) for r in lg]))
        out[g] = row
    return out


def main(outp, frozen_path, roots):
    frozen = json.load(open(frozen_path))
    data = {lab: {f: {g: load(os.path.join(root, f, g, "episodes.jsonl")) for g in ("none", "inline", "out5", "l4")} for f in CFGS}
            for lab, root in roots}
    clusters = sorted({cl(r) for lab in data for f in data[lab] for g in data[lab][f] for r in data[lab][f][g].values()})
    cidx = {c: i for i, c in enumerate(clusters)}
    print(f"clusters {len(clusters)}", flush=True)
    W_cell, W_adj = draws(len(clusters), B_CELL), draws(len(clusters), B_ADJ)
    J = {"n_clusters": len(clusters), "clusters_by_suite": {}, "cells": {}, "gate_cost": {}, "contrasts": {}}
    for c in clusters:
        s = c.split("/")[0]
        J["clusters_by_suite"][s] = J["clusters_by_suite"].get(s, 0) + 1

    # regression check against the frozen analysis, then cell-level cost
    worst_point, worst_ci, adjS = 0.0, 0.0, {}
    for lab, _ in roots:
        J["cells"][lab], J["gate_cost"][lab] = {}, {}
        for f in CFGS:
            E = data[lab][f]
            J["gate_cost"][lab][f] = gate_cost(E)
            J["cells"][lab][f] = {}
            for a, b in PAIRS:
                T, n = totals(E, a, b, cidx)
                full = {k: v.sum() for k, v in T.items()}
                boot = {k: W_cell @ v for k, v in T.items()}
                bootA = {k: W_adj @ v for k, v in T.items()}
                J["cells"][lab][f][f"{a}-{b}"] = {}
                for w in WORKLOADS:
                    frz = frozen[lab][f][f"{a}-{b}"][w]
                    ra, rb = float(rates(full, "a", w)[0]), float(rates(full, "b", w)[0])
                    X, Y = ("a", "b") if ra >= rb else ("b", "a")
                    name = {"a": a, "b": b}
                    if (name[X], name[Y]) != (frz["X"], frz["Y"]):
                        raise SystemExit(f"REGRESSION orientation {lab} {f} {a}-{b} {w}: {name[X]},{name[Y]} vs {frz['X']},{frz['Y']}")
                    D, dAC, dBU, dConf, lam = (float(v) for v in effects(full, X, Y, w))
                    bD, bAC, bBU, bConf, _ = effects(boot, X, Y, w)
                    ci = pct_ci(bD, 0.05)
                    worst_point = max(worst_point, abs(D - frz["D"]), abs(lam - frz["lambda"]))
                    worst_ci = max(worst_ci, abs(ci[0] - frz["ci"][0]), abs(ci[1] - frz["ci"][1]))
                    J["cells"][lab][f][f"{a}-{b}"][w] = {
                        "X": name[X], "Y": name[Y], "n": n, "rate_X": ra if X == "a" else rb, "rate_Y": rb if X == "a" else ra,
                        "lambda": lam, "D": D, "ci": ci, "readout": frz["readout"], "dAC": dAC, "ci_AC": pct_ci(bAC, 0.05),
                        "dBU": dBU, "ci_BU": pct_ci(bBU, 0.05), "dConf": dConf, "ci_Conf": pct_ci(bConf, 0.05)}
                    if (a, b) == ("out5", "l4"):
                        sign = 1.0 if name[X] == "out5" else -1.0
                        adjS[(lab, f, w)] = sign * effects(bootA, X, Y, w)[0]
                        J["cells"][lab][f][f"{a}-{b}"][w]["S"] = sign * D
    J["regression"] = {"max_abs_point_and_lambda_diff": worst_point, "max_abs_ci_diff": worst_ci}
    print("regression", J["regression"], flush=True)
    if worst_point > 1e-12 or worst_ci > 0.0005:
        json.dump(J, open(outp + ".FAILED.json", "w"), indent=1)
        raise SystemExit("REGRESSION CHECK FAILED: no results written")

    # direct contrasts of the rule-reviewer difference
    alpha = 0.05 / N_CONTRASTS
    pair = "out5-l4"
    cell_S = lambda u, f, w: J["cells"][u][f][pair][w]["S"]
    side = lambda u, f, w: (lambda c: 1 if c["ci"][0] > 0 else (-1 if c["ci"][1] < 0 else 0))(J["cells"][u][f][pair][w]) * (
        1 if J["cells"][u][f][pair][w]["X"] == "out5" else -1)

    def contrast(name, factor, c1, c2):
        est = cell_S(*c1) - cell_S(*c2)
        ci = pct_ci(adjS[c1] - adjS[c2], alpha)
        changes = ci[0] > 0 or ci[1] < 0
        reverses = changes and side(*c1) * side(*c2) == -1
        J["contrasts"][name] = {"factor": factor, "estimate": est, "ci_adjusted": ci, "changes": bool(changes), "reverses": bool(reverses)}

    labels = [lab for lab, _ in roots]
    for u in labels:
        for f in CFGS:
            contrast(f"workload|{u}|{f}|A-B", "workload", (u, f, "A"), (u, f, "B"))
    for f in CFGS:
        for u in USERS_VS:
            contrast(f"user|{f}|{u}-{REF_USER}", "user", (u, f, "pooled"), (REF_USER, f, "pooled"))
    for u in labels:
        for f1, f2 in MODEL_CONTRASTS:
            contrast(f"model|{u}|{f1}-{f2}", "model", (u, f1, "pooled"), (u, f2, "pooled"))
    if len(J["contrasts"]) != N_CONTRASTS:
        raise SystemExit(f"expected {N_CONTRASTS} contrasts, got {len(J['contrasts'])}")

    J["factor_summary"] = {}
    for fac in ("workload", "user", "model"):
        cs = [c for c in J["contrasts"].values() if c["factor"] == fac]
        big = max(cs, key=lambda c: abs(c["estimate"]) if c["changes"] else -1)
        J["factor_summary"][fac] = {"n": len(cs), "n_changes": sum(c["changes"] for c in cs), "n_reverses": sum(c["reverses"] for c in cs),
                                    "largest_changing": big["estimate"] if big["changes"] else None}
    json.dump(J, open(outp + ".json", "w"), indent=1)

    p = lambda v: f"{100 * v:+.1f}"
    iv = lambda c: f"[{100 * c[0]:+.1f}, {100 * c[1]:+.1f}]"
    L = ["# Cost at the equal rate and direct contrasts", "",
         f"Clusters {J['n_clusters']} {J['clusters_by_suite']}. Regression: {J['regression']}.", "",
         "## Rule vs detached reviewer, pooled workload, cost at the equal rate", "",
         "| user | family | X vs Y | ΔASR | Δ authorised completion | Δ benign utility | Δ confirmations per legit episode |", "|---|---|---|---|---|---|---|"]
    for u in labels:
        for f in CFGS:
            c = J["cells"][u][f][pair]["pooled"]
            L.append(f"| {u} | {f} | {c['X']} vs {c['Y']} | {p(c['D'])} {iv(c['ci'])} | {p(c['dAC'])} {iv(c['ci_AC'])} | {p(c['dBU'])} {iv(c['ci_BU'])} | {c['dConf']:+.2f} [{c['ci_Conf'][0]:+.2f}, {c['ci_Conf'][1]:+.2f}] |")
    L += ["", "## Cost per gate, unmixed", "", "| user | family | gate | conf per legit | conf per A | conf per B | share ≥2 | reviewer inferences per legit |", "|---|---|---|---|---|---|---|---|"]
    for u in labels:
        for f in CFGS:
            for g, r in J["gate_cost"][u][f].items():
                L.append(f"| {u} | {f} | {g} | {r['conf_per_legit']:.2f} | {r['conf_per_A']:.2f} | {r['conf_per_B']:.2f} | {100 * r['share_two_or_more']:.1f} | {r.get('reviewer_inferences_per_legit', float('nan')):.2f} |")
    L += ["", f"## Direct contrasts of S (positive: rule protects more), Bonferroni 1−0.05/{N_CONTRASTS}, {B_ADJ} resamples", "", "| contrast | estimate pp | adjusted interval | changes | reverses |", "|---|---|---|---|---|"]
    for k, c in J["contrasts"].items():
        L.append(f"| {k} | {p(c['estimate'])} | {iv(c['ci_adjusted'])} | {'yes' if c['changes'] else ''} | {'yes' if c['reverses'] else ''} |")
    L += ["", f"Factor summary: {J['factor_summary']}", ""]
    open(outp + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L[-3:]))
    print("ANALYSIS_DONE")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], [tuple(x.split("=", 1)) for x in sys.argv[3:]])
