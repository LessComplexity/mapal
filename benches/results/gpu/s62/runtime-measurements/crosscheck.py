#!/usr/bin/env python3
"""Value cross-checks between programs (arm_ratio checks arms within one program): a request-clock
probe prints the values of its steady-clock twin (pt_rq_X == pt_X) and the regenerated loop prints
the retained loop's (pt_rg_X == pt_rt_X), per face, in every session.
usage: crosscheck.py <results>..."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../gpu'))
import arm_ratio as ar  # noqa: E402

bad = n = 0
for f in sys.argv[1:]:
    d = ar.parse(f)
    for (prog, face), arms in d.items():
        for a, b in (('pt_rq_', 'pt_'), ('pt_rg_', 'pt_rt_')):
            if not prog.startswith(a):
                continue
            twin = (b + prog[len(a):], face)
            if twin not in d:
                continue
            x = {o for v in arms.values() for o in v['out']}
            y = {o for v in d[twin].values() for o in v['out']}
            n += 1
            if x != y:
                bad += 1
                print(f"MISMATCH {os.path.basename(f)} {prog} {face}: {x} vs {y}")
print(f"{n} program pairs compared, {bad} mismatches")
sys.exit(1 if bad else 0)
