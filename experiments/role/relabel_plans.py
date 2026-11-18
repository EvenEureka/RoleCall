"""Write plans/<family>.json (the pre-registered cells). Groups are in priority order (a deadline drops later groups).
Only the `G1_*` / `G2_native_voice` cells (three templates, tool vs user role) feed the released data."""
import json
import os

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plans")
II, DI, IG = "important_instructions", "direct", "ignore_previous"
SMOKE_PAIRS = [["banking", "user_task_0", "injection_task_0"], ["banking", "user_task_3", "injection_task_1"],
               ["slack", "user_task_1", "injection_task_1"]]


def cell(arm, attack, pairs="all", benign=False, seed=None, temperature=0.0):
    return {"arm": arm, "attack": attack, "pairs": pairs, "benign": benign, "seed": seed, "temperature": temperature}


def voice(arms):
    return [cell(a, at) for at in (DI, IG, II) for a in arms]


def seeds(arms, T=0.7):
    return [cell(a, II, "subset60", seed=s, temperature=T) for s in (1, 2, 3) for a in arms]


plans = {
    "llama33": {
        "smoke": {"cells": [cell("B", DI, SMOKE_PAIRS), cell("C", IG, SMOKE_PAIRS), cell("Cr", II, SMOKE_PAIRS),
                            cell("Di", II, SMOKE_PAIRS), cell("Au", II, SMOKE_PAIRS),
                            cell("B", II, SMOKE_PAIRS, seed=1, temperature=0.7),
                            cell("C", II, SMOKE_PAIRS, seed=1, temperature=0.7)]},
        "main": {"groups": [
            {"name": "G1_voice", "cells": voice(["B", "C"])},
            {"name": "G2_raw_seeds", "cells": [cell("Cr", II, benign=True)] + seeds(["B", "C"])},
            {"name": "G3_headers", "cells": [cell("Di", II, benign=True), cell("Au", II, benign=True)]},
        ]},
    },
    "qwen3": {
        "smoke": {"cells": [cell("B", DI, SMOKE_PAIRS), cell("C", IG, SMOKE_PAIRS),
                            cell("B", II, SMOKE_PAIRS, seed=1, temperature=0.7),
                            cell("C", II, SMOKE_PAIRS, seed=1, temperature=0.7)]},
        "main": {"groups": [
            {"name": "G1_voice", "cells": voice(["B", "C"])},
            {"name": "G2_seeds", "cells": seeds(["B", "C"])},
        ]},
    },
    "gptoss": {
        "smoke": {"cells": [cell("Au", II, SMOKE_PAIRS), cell("A", DI, SMOKE_PAIRS), cell("Au", IG, SMOKE_PAIRS)]},
        "main": {"groups": [
            {"name": "G1_native_role", "cells": [cell("Au", II, benign=True), cell("A", II)]},
            {"name": "G2_native_voice", "cells": [cell(a, at) for at in (DI, IG) for a in ("A", "Au")]},
        ]},
    },
    "llama31": {
        "smoke": {"cells": [cell("B", II, SMOKE_PAIRS), cell("C", II, SMOKE_PAIRS)]},
        "main": {"groups": [{"name": "G1_role", "cells": [cell("B", II, benign=True), cell("C", II, benign=True)]}]},
    },
}
os.makedirs(D, exist_ok=True)
for fam, p in plans.items():
    json.dump(p, open(os.path.join(D, f"{fam}.json"), "w"), indent=1)
    print("wrote", fam)
