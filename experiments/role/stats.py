"""Shared statistics for the role experiments (stdlib only; run with python3 -I).

Paired contrasts on attacked pairs k = (suite, user_task, injection_task), with four resampling schemes:
  pair   : bootstrap over pairs (wave-1/2 default)
  user   : cluster bootstrap over user tasks (suite, user_task)      -- 37 clusters on banking+slack
  goal   : cluster bootstrap over injection goals (suite, injection_task) -- 14 clusters
  twoway : independent resampling of user tasks and goals; a pair's weight is mult(user) * mult(goal)
           (pigeonhole / product bootstrap; the most conservative of the four)
CIs are percentile intervals, 10,000 resamples, seed 0 (pair), 1 (user), 2 (goal), 3 (twoway).
Exact two-sided McNemar for binary paired outcomes.
"""
import math
import random

NB = 10000


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n))


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (p, max(0.0, c - h), min(1.0, c + h))


def _pct(sorted_vals, level):
    a = (1 - level) / 2
    nb = len(sorted_vals)
    return (sorted_vals[int(a * nb)], sorted_vals[int((1 - a) * nb) - 1])


def _boot_pairs(d, nb, seed):
    n = len(d)
    rng = random.Random(seed)
    out = []
    for _ in range(nb):
        s = 0
        for _ in range(n):
            s += d[rng.randrange(n)]
        out.append(s / n)
    out.sort()
    return out


def _boot_cluster(d, cl, nb, seed):
    g = {}
    for v, c in zip(d, cl):
        t = g.setdefault(c, [0, 0])
        t[0] += v
        t[1] += 1
    keys = sorted(g)
    sums = [g[k][0] for k in keys]
    cnts = [g[k][1] for k in keys]
    m = len(keys)
    rng = random.Random(seed)
    out = []
    for _ in range(nb):
        s = c = 0
        for _ in range(m):
            j = rng.randrange(m)
            s += sums[j]
            c += cnts[j]
        out.append(s / c if c else 0.0)
    out.sort()
    return out


def _boot_twoway(d, ucl, gcl, nb, seed):
    us = sorted(set(ucl))
    gs = sorted(set(gcl))
    ui = {u: i for i, u in enumerate(us)}
    gi = {g: i for i, g in enumerate(gs)}
    idx = [(ui[u], gi[g]) for u, g in zip(ucl, gcl)]
    rng = random.Random(seed)
    out = []
    for _ in range(nb):
        mu = [0] * len(us)
        mg = [0] * len(gs)
        for _ in range(len(us)):
            mu[rng.randrange(len(us))] += 1
        for _ in range(len(gs)):
            mg[rng.randrange(len(gs))] += 1
        s = w = 0
        for v, (a, b) in zip(d, idx):
            ww = mu[a] * mg[b]
            if ww:
                s += ww * v
                w += ww
        out.append(s / w if w else 0.0)
    out.sort()
    return out


def boot_all(d, keys, nb=NB, schemes=("pair", "user", "goal", "twoway")):
    """d: per-pair values (differences or interaction contrasts); keys: (suite, user_task, injection_task)."""
    ucl = [(k[0], k[1]) for k in keys]
    gcl = [(k[0], k[2]) for k in keys]
    res = {"n": len(d), "mean": sum(d) / len(d) if d else float("nan"),
           "n_user_clusters": len(set(ucl)), "n_goal_clusters": len(set(gcl))}
    if not d:
        return res
    for sch in schemes:
        if sch == "pair":
            bs = _boot_pairs(d, nb, 0)
        elif sch == "user":
            bs = _boot_cluster(d, ucl, nb, 1)
        elif sch == "goal":
            bs = _boot_cluster(d, gcl, nb, 2)
        else:
            bs = _boot_twoway(d, ucl, gcl, nb, 3)
        res[f"ci95_{sch}"] = _pct(bs, 0.95)
        res[f"ci90_{sch}"] = _pct(bs, 0.90)
        res[f"p_{sch}"] = min(1.0, 2 * min(sum(b <= 0 for b in bs) / nb, sum(b >= 0 for b in bs) / nb))
    return res


def paired_binary(xs, ys, keys, nb=NB, schemes=("pair", "user", "goal", "twoway")):
    d = [x - y for x, y in zip(xs, ys)]
    r = boot_all(d, keys, nb, schemes)
    b = sum(1 for x, y in zip(xs, ys) if x == 1 and y == 0)
    c = sum(1 for x, y in zip(xs, ys) if x == 0 and y == 1)
    r.update({"diff": r["mean"], "mean_x": sum(xs) / len(xs) if xs else float("nan"),
              "mean_y": sum(ys) / len(ys) if ys else float("nan"), "x1y0": b, "x0y1": c,
              "mcnemar_p": mcnemar_exact(b, c)})
    return r


def paired_tasks(xs, ys, nb=NB, seed=0):
    """Benign utility: paired over user tasks (each task is its own cluster)."""
    d = [x - y for x, y in zip(xs, ys)]
    if not d:
        return {"n": 0}
    bs = _boot_pairs(d, nb, seed)
    b = sum(1 for x, y in zip(xs, ys) if x == 1 and y == 0)
    c = sum(1 for x, y in zip(xs, ys) if x == 0 and y == 1)
    return {"n": len(d), "diff": sum(d) / len(d), "ci95_pair": _pct(bs, 0.95), "ci90_pair": _pct(bs, 0.90),
            "x1y0": b, "x0y1": c, "mcnemar_p": mcnemar_exact(b, c)}


def lnor(xs, ys):
    b = sum(1 for x, y in zip(xs, ys) if x == 1 and y == 0)
    c = sum(1 for x, y in zip(xs, ys) if x == 0 and y == 1)
    return math.log((b + 0.5) / (c + 0.5))


def holm(pvals):
    """pvals: list of (name, p). Returns {name: adjusted p}."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i][1])
    m = len(pvals)
    out, run = {}, 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, max(run, (m - rank) * pvals[i][1]))
        run = adj
        out[pvals[i][0]] = adj
    return out


def pp(x):
    return "nan" if x != x else f"{100 * x:+.1f}"


def ci(t):
    return "—" if t is None else f"[{pp(t[0])}, {pp(t[1])}]"
