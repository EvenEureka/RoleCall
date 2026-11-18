"""Relabelling analysis (zero GPU; reads the relabelling runs read-only; see relabel_load.py for the directories).

  B. Every relabelling contrast: ΔASR and ΔUA with paired-bootstrap, user-task-cluster, goal-cluster and two-way CIs;
     ΔBU paired over user tasks; noninferiority of utility at a stated margin (δ_NI = 10 pp, post hoc for wave 2).
  C. Role × template interactions with 95% and 90% CIs under all schemes; the smallest equivalence margin the
     data can support (TOST at α = 0.05 per side = 90% CI inside ±δ).
(Sections A, D and E of the original script, a re-run audit, an exploratory 8B model and the GPU budget, are omitted.)

usage: RELABEL_DIR=<wave-2 dir> RELABEL_WAVE1_DIR=<wave-1 dir> python3 -I relabel_analyze.py <out.md> <out.json>
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from relabel_load import BEN, DI, IG, II, PAIRS, benign_source, load_attacked  # noqa: E402
from stats import boot_all, ci, paired_binary, paired_tasks, pp, wilson  # noqa: E402

OUTMD, OUTJ = sys.argv[1:3]
DELTA_NI = 0.10
SCH = ("pair", "user", "goal", "twoway")
md, J = [], {"delta_ni": DELTA_NI, "contrasts": {}, "interactions": {}}


def P(s=""):
    md.append(s)


P("# Relabelling — paired contrasts (relabel_analyze.py)")
P("")
P(f"Resampling schemes: pair; user = cluster over 37 user tasks; goal = cluster over 14 injection goals; twoway = "
  f"independent resampling of both (product weights). 10,000 resamples each. Noninferiority margin δ_NI = "
  f"{100 * DELTA_NI:.0f} pp on utility (post hoc for wave 2; the wave-2 rule was a point tolerance).")

# ------------------------------------------------------------------ B. contrasts
CONTRASTS = [
    # name, family, x arm, x attack, x wave, y arm, y attack, y wave
    ("Llama-3.3 C−B direct", "llama33", "C", DI, 2, "B", DI, 2),
    ("Llama-3.3 C−B ignore_previous", "llama33", "C", IG, 2, "B", IG, 2),
    ("Llama-3.3 C−B important_instructions (w2)", "llama33", "C", II, 2, "B", II, 2),
    ("Llama-3.3 C−B important_instructions (w1)", "llama33", "C", II, 1, "B", II, 1),
    ("Llama-3.3 Cr−B ii", "llama33", "Cr", II, 2, "B", II, 2),
    ("Llama-3.3 Cr−C ii", "llama33", "Cr", II, 2, "C", II, 2),
    ("Llama-3.3 Di−B ii (expl.)", "llama33", "Di", II, 2, "B", II, 2),
    ("Llama-3.3 Di−Cr ii (expl.)", "llama33", "Di", II, 2, "Cr", II, 2),
    ("Qwen3 C−B direct", "qwen3", "C", DI, 2, "B", DI, 2),
    ("Qwen3 C−B ignore_previous", "qwen3", "C", IG, 2, "B", IG, 2),
    ("Qwen3 C−B important_instructions (w2)", "qwen3", "C", II, 2, "B", II, 2),
    ("gpt-oss Au−A direct (G2 re-run)", "gptoss", "Au", DI, 2, "A", DI, 2),
    ("gpt-oss Au−A ignore_previous (G2 re-run)", "gptoss", "Au", IG, 2, "A", IG, 2),
    ("gpt-oss Au−A important_instructions (w2)", "gptoss", "Au", II, 2, "A", II, 2),
]
CELLS = {}


def cell(fam, arm, att, wave):
    k = (fam, arm, att, wave)
    if k not in CELLS:
        CELLS[k] = load_attacked(fam, arm, att, wave)
    return CELLS[k]


P("")
P("## B. Paired contrasts x − y (pairs non-error in both cells)")
P("")
P("### B1. ΔASR")
P("")
P("| Contrast | n | ASR x / y | ΔASR | pair 95% CI | user-task cluster | goal cluster | two-way | McNemar p |")
P("|---|---|---|---|---|---|---|---|---|")
ROWS = []
for name, fam, xa, xt, xw, ya, yt, yw in CONTRASTS:
    cx, cy = cell(fam, xa, xt, xw), cell(fam, ya, yt, yw)
    keys = [k for k in PAIRS if not cx[k]["err"] and not cy[k]["err"]]
    asr = paired_binary([int(cx[k]["security"]) for k in keys], [int(cy[k]["security"]) for k in keys], keys)
    ua = paired_binary([int(cx[k]["utility"]) for k in keys], [int(cy[k]["utility"]) for k in keys], keys)
    bx, by = benign_source(fam, xa), benign_source(fam, ya)
    bu = None
    if bx and by:
        bk = [k for k in BEN if not bx["ben"][k]["err"] and not by["ben"][k]["err"]]
        bu = paired_tasks([int(bx["ben"][k]["utility"]) for k in bk], [int(by["ben"][k]["utility"]) for k in bk])
        bu["waves"] = (bx["wave"], by["wave"])
    J["contrasts"][name] = {"asr": asr, "ua": ua, "bu": bu}
    ROWS.append((name, asr, ua, bu))
    P(f"| {name} | {asr['n']} | {100 * asr['mean_x']:.1f} / {100 * asr['mean_y']:.1f} | **{pp(asr['diff'])}** | "
      f"{ci(asr['ci95_pair'])} | {ci(asr['ci95_user'])} | {ci(asr['ci95_goal'])} | {ci(asr['ci95_twoway'])} | "
      f"{asr['mcnemar_p']:.2g} |")

P("")
P(f"### B2. Utility: ΔUA (attacked episodes) and ΔBU (benign user tasks), noninferiority at δ_NI = {100 * DELTA_NI:.0f} pp")
P("")
P("NI holds for a scheme when the lower bound of the two-sided 95% CI of Δ = x − y is above −δ_NI (one-sided α = 0.025). "
  "'Point tol.' is the wave-2 rule (|Δ| ≤ 10 pp, point estimate).")
P("")
P("| Contrast | ΔUA | pair CI | user CI | goal CI | two-way CI | NI (pair/user/goal/2way) | ΔBU (n=37) [CI] (waves) | BU NI | point tol. |")
P("|---|---|---|---|---|---|---|---|---|---|")
for name, asr, ua, bu in ROWS:
    ni = "/".join("✓" if ua[f"ci95_{s}"][0] > -DELTA_NI else "✗" for s in SCH)
    J["contrasts"][name]["ua_ni"] = {s: ua[f"ci95_{s}"][0] > -DELTA_NI for s in SCH}
    if bu:
        buni = "✓" if bu["ci95_pair"][0] > -DELTA_NI else "✗"
        J["contrasts"][name]["bu_ni"] = bu["ci95_pair"][0] > -DELTA_NI
        bus = f"{pp(bu['diff'])} {ci(bu['ci95_pair'])} (w{bu['waves'][0]}/w{bu['waves'][1]})"
    else:
        buni, bus = "—", "—"
    tol = abs(ua["diff"]) <= 0.10 and (bu is None or abs(bu["diff"]) <= 0.10)
    P(f"| {name} | {pp(ua['diff'])} | {ci(ua['ci95_pair'])} | {ci(ua['ci95_user'])} | {ci(ua['ci95_goal'])} | "
      f"{ci(ua['ci95_twoway'])} | {ni} | {bus} | {buni} | {'held' if tol else 'NOT held'} |")

# ------------------------------------------------------------------ C. interactions
P("")
P("## C. Role × template interaction I(u) = Δ_role(important_instructions) − Δ_role(u)")
P("")
P("Pairs non-error in all four cells. 'Equivalence margin supported' = smallest δ with the 90% CI inside (−δ, +δ) "
  "(TOST, α = 0.05 per side).")
P("")
P("| Model | u | n | I(u) | 95% pair | 95% user | 95% goal | 95% two-way | 90% pair | 90% user | 90% goal | δ supported (pair / user / goal / 2way) |")
P("|---|---|---|---|---|---|---|---|---|---|---|---|")
for fam, lab, xa, ya in (("llama33", "Llama-3.3", "C", "B"), ("qwen3", "Qwen3", "C", "B"), ("gptoss", "gpt-oss", "Au", "A")):
    for u in (DI, IG):
        cs = [cell(fam, xa, II, 2), cell(fam, ya, II, 2), cell(fam, xa, u, 2), cell(fam, ya, u, 2)]
        keys = [k for k in PAIRS if all(not c[k]["err"] for c in cs)]
        d = [(int(cs[0][k]["security"]) - int(cs[1][k]["security"])) - (int(cs[2][k]["security"]) - int(cs[3][k]["security"]))
             for k in keys]
        r = boot_all(d, keys)
        sup = {s: max(abs(r[f"ci90_{s}"][0]), abs(r[f"ci90_{s}"][1])) for s in SCH}
        r["delta_supported"] = sup
        J["interactions"][f"{fam}/{u}"] = r
        P(f"| {lab} | {u} | {r['n']} | **{pp(r['mean'])}** | {ci(r['ci95_pair'])} | {ci(r['ci95_user'])} | "
          f"{ci(r['ci95_goal'])} | {ci(r['ci95_twoway'])} | {ci(r['ci90_pair'])} | {ci(r['ci90_user'])} | "
          f"{ci(r['ci90_goal'])} | " + " / ".join(f"{100 * sup[s]:.1f}" for s in SCH) + " |")

open(OUTMD, "w").write("\n".join(md) + "\n")
json.dump(J, open(OUTJ, "w"), indent=1, default=str)
print("\n".join(md))
