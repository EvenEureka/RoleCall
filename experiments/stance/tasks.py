"""Task set of the stance and gate experiments (pre-registered). Only AgentDojo v1.2.2's shipped injection tasks are used; no new injection
text is written. Every unauthorised episode (kind U) has an authorised twin (kind A): the same user task and the same
goal, but the goal is stated by the user in the prompt and no injection is placed in the environment.

Goal classes (fixed before any round-2 output; payload is then adjudicated per call by the blind labellers):
  S  sensitive-data goals: banking 0,1,2,3,8; slack 2,4; travel 1,3,5; workspace 3-12
  F  money-only goals:      banking 4,5,6
  N  non-sensitive goal:    workspace 0 (a fixed greeting e-mailed to the attacker address)
Episodes:
  U  = user task x goal, goal injected with `important_instructions` (as in the first rating run)
  A  = same pairs, prompt = user task PROMPT + TWIN_JOIN + goal, no injection
  B  = every user task, no injection (97)
  RW = U and A pairs of class S run with the reviewer-wording actor (cfg "rw"; pre-registration §5.3)
Workspace has 40 user tasks; each gets ONE class-S goal (balanced, seeded; each of the 10 goals used 4 times) plus the
class-N goal. All other suites: every user task x every goal of its classes.
"""
import random

SUITES = ["banking", "slack", "travel", "workspace"]
SEED = 20261025
GOALS = {"S": {"banking": [0, 1, 2, 3, 8], "slack": [2, 4], "travel": [1, 3, 5], "workspace": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]},
         "F": {"banking": [4, 5, 6]},
         "N": {"workspace": [0]}}
WS_S_PER_USER_TASK = 1
TWIN_JOIN = "\n\nAlso, please do the following: "


def key_of(kind, suite, ut, it, cfg="main"):
    return f"{cfg}:{kind}:{suite}:{ut}:{it}"


def goal_class(suite, inj_no):
    for g, m in GOALS.items():
        if inj_no in m.get(suite, []):
            return g
    return None


def _ep(kind, suite, ut, it, cfg="main"):
    gc = goal_class(suite, int(it.split("_")[-1])) if it else None
    return {"key": key_of(kind, suite, ut, it, cfg), "kind": kind, "suite": suite, "user_task": ut, "injection_task": it,
            "goal_class": gc, "cfg": cfg}


def pairs(suites):
    """(suite, user_task, injection_task) for the U/A pairs, in a fixed order."""
    rng = random.Random(SEED)
    out = []
    for s in SUITES:
        uts = sorted(suites[s].user_tasks, key=lambda x: int(x.split("_")[-1]))
        s_goals = [f"injection_task_{i}" for i in GOALS["S"].get(s, [])]
        other = [f"injection_task_{i}" for g in ("F", "N") for i in GOALS[g].get(s, [])]
        for i in s_goals + other:
            assert i in suites[s].injection_tasks, (s, i)
        if s == "workspace":
            order = uts[:]
            rng.shuffle(order)
            pool = s_goals[:]
            rng.shuffle(pool)
            for k, ut in enumerate(order):
                for j in range(WS_S_PER_USER_TASK):
                    out.append((s, ut, pool[(k * WS_S_PER_USER_TASK + j) % len(pool)]))
        else:
            for ut in uts:
                for i in s_goals:
                    out.append((s, ut, i))
        for ut in uts:
            for i in other:
                out.append((s, ut, i))
    return out


def main_episodes(suites):
    eps = []
    for s, ut, it in pairs(suites):
        eps.append(_ep("U", s, ut, it))
        eps.append(_ep("A", s, ut, it))
    for s in SUITES:
        for ut in sorted(suites[s].user_tasks, key=lambda x: int(x.split("_")[-1])):
            eps.append(_ep("B", s, ut, None))
    random.Random(SEED + 1).shuffle(eps)
    return eps


def rw_episodes(suites):
    eps = []
    for s, ut, it in pairs(suites):
        if goal_class(s, int(it.split("_")[-1])) == "S":
            eps.append(_ep("U", s, ut, it, "rw"))
            eps.append(_ep("A", s, ut, it, "rw"))
    random.Random(SEED + 2).shuffle(eps)
    return eps


SMOKE = [("U", "banking", "user_task_0", "injection_task_0"), ("A", "banking", "user_task_0", "injection_task_0"),
         ("U", "slack", "user_task_1", "injection_task_2"), ("B", "workspace", "user_task_0", None)]


def smoke_episodes():
    return [_ep(*x) for x in SMOKE] + [_ep("U", "banking", "user_task_0", "injection_task_0", "rw")]
