"""Shared statistics for the stance and gate experiments (stdlib + numpy). Used by destination_rule_gap.py and analyze_stance.py.
- wilson(k, n): Wilson 95 % interval.
- cluster_boot(rows, fn): percentile CI over (suite, user_task) cluster resamples (paired: every arm of a resampled
  cluster comes along). Same defaults as the first rating run (5,000 resamples, seed 20261024).
- hull_catch(points, b): ROC convex hull (upper concave envelope through (0,0) and (1,1)) evaluated at budget b.
  A point is (interruption budget, catch rate) of one deterministic gate; any point on the hull is reachable by
  randomising per episode between two gates, so this is "catch rate at a matched interruption budget".
"""
import collections
import math

import numpy as np

SEED = 20261024
B = 5000


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def mean(xs):
    xs = list(xs)
    return float(np.mean(xs)) if len(xs) else float("nan")


def cluster_boot(rows, fn, b=B, seed=SEED, alpha=0.05):
    cl = collections.defaultdict(list)
    for r in rows:
        cl[r["cluster"]].append(r)
    keys = sorted(cl)
    if not keys:
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    est = fn(rows)
    vals = []
    for _ in range(b):
        pick = rng.integers(0, len(keys), len(keys))
        rs = [r for i in pick for r in cl[keys[i]]]
        v = fn(rs)
        if v == v:
            vals.append(v)
    if not vals:
        return est, (float("nan"), float("nan"))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return est, (float(lo), float(hi))


def upper_hull(points):
    """points: iterable of (x, y) in [0,1]^2. Returns the upper concave envelope through (0,0) and (1,1), sorted by x."""
    pts = sorted(set([(0.0, 0.0), (1.0, 1.0)] + [(float(x), float(y)) for x, y in points if x == x and y == y]))
    # keep the best y for each x
    best = {}
    for x, y in pts:
        best[x] = max(best.get(x, -1.0), y)
    pts = sorted(best.items())
    hull = []
    for p in pts:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            # remove hull[-1] if it lies on or below the segment hull[-2] -> p
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) >= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    # the envelope must be non-decreasing: a gate can always be randomised with "always fire"
    out = []
    for x, y in hull:
        if out and y < out[-1][1]:
            y = out[-1][1]
        out.append((x, y))
    return out


def hull_catch(points, b):
    h = upper_hull(points)
    if b <= 0:
        return max([y for x, y in h if x <= 0] + [0.0])
    for (x1, y1), (x2, y2) in zip(h, h[1:]):
        if x1 <= b <= x2:
            return y1 if x2 == x1 else y1 + (y2 - y1) * (b - x1) / (x2 - x1)
    return 1.0


def zq(p):
    from statistics import NormalDist
    return NormalDist().inv_cdf(p)


def dprime(hit_k, hit_n, fa_k, fa_n):
    h = (hit_k + 0.5) / (hit_n + 1)
    f = (fa_k + 0.5) / (fa_n + 1)
    return zq(h) - zq(f), -(zq(h) + zq(f)) / 2


def pct(x):
    return "n/a" if x is None or x != x else f"{100 * x:.1f}"


def pp(x):
    return "n/a" if x is None or x != x else f"{100 * x:+.1f}"


def ci_pct(ci):
    return f"[{pct(ci[0])}, {pct(ci[1])}]"


def ci_pp(ci):
    return f"[{pp(ci[0])}, {pp(ci[1])}]"
