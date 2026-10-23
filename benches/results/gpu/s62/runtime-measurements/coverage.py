#!/usr/bin/env python3
"""coverage.py <results>...: per program, the reps each (face, arm) reached; complete = all at max."""
import re
import sys
from collections import defaultdict

for f in sys.argv[1:]:
    c = defaultdict(lambda: defaultdict(int))
    for l in open(f):
        m = re.match(r'probe prog=(\S+) face=(\S+) arm=(\S+) rep=(\d+)', l)
        if m:
            c[m[1]][(m[2], m[3])] = max(c[m[1]][(m[2], m[3])], int(m[4]))
    print(f)
    for p, v in c.items():
        reps = set(v.values())
        print(f"  {p}: arms={len(v)} reps={sorted(reps)}{'' if reps == {12} else '  INCOMPLETE'}")
