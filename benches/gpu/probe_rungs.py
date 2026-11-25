#!/usr/bin/env python3
"""The S54 rule for a placement rung over one part's sessions (box_run probe rows, different boxes).

Per rung and face, per probe and size: s = (OFF iter ms) / (ON iter ms), medians within one
session (`probe prog=… arm=on|off-<rung>`, the program's own clock around every computation).
A size is stable+ when s > 1.01 in every session, stable- when s < 0.99 in every session; `*` marks
disjoint min-max ranges. The rung ships ON for a face iff some probe size is stable+ and none is
stable- with disjoint ranges in every session (a placement that loses on one program is not shipped
on the strength of another). An OFF arm is read against the ON text's arm that launched the same site
picks (picks.py: `on`, else `rules`, else `pick<i>`; the cell names it when not `on`): a tuned box's
`on` runs its wisdom pick, the OFF text the rules' candidate 0. No such arm: `pick≠`, out of the verdict. The traced kernel ms (`probe_kernels`, summed per run) is shown beside
it, with the same rule on it: the verdict for a rung that changes only kernels (a store order),
whose effect the host's timing noise swamps. A placement rung that moves host work or transfers
(rowfold) adds a kernel by construction: read its iter verdict.
usage: python3 benches/gpu/probe_rungs.py <results-session-1> [<results-session-2>...]"""
import re
import sys
from collections import defaultdict

from picks import launched, partner

EPS = 0.01
med = lambda xs: sorted(xs)[len(xs) // 2]


def runs(path):
    it, kern, pk = defaultdict(list), defaultdict(list), defaultdict(dict)
    for l in open(path):
        if l.startswith('probe prog='):
            f = dict(re.findall(r'(\w+)=([^\s\[\]]+)', l))
            m = re.search(r'iter ms=\s*([-\d.eE+]+)', l)
            if m:
                it[(f['prog'], f['face'], f['arm'])].append(float(m[1]))
            s = launched(l)
            if s is not None:
                pk[(f['prog'], f['face'])].setdefault(f['arm'], s)
        elif l.startswith('probe_kernels '):
            f = dict(re.findall(r'(\w+)=([^\s\[\]]+)', l))
            kern[(f['prog'], f['face'], f['arm'])].append(sum(map(float, re.findall(r' ms=([\d.]+)', l))))
    return it, kern, pk


def main(files):
    sess = [runs(f) for f in files]
    cls = defaultdict(lambda: defaultdict(dict))  # (rung, face) -> (prog) -> session -> (s, disj, ks)
    for i, (it, kern, pk) in enumerate(sess):
        for (prog, face, arm), off in it.items():
            if not arm.startswith('off-') or (prog, face, 'on') not in it:
                continue
            base = partner(pk.get((prog, face), {}), arm)
            if base is None or (prog, face, base) not in it:
                cls[(arm[4:], face)][prog][i] = None  # pick-confounded: no ON-text arm at the OFF arm's picks
                continue
            on = it[(prog, face, base)]
            disj = max(on) < min(off) or max(off) < min(on)
            ko, kf = kern.get((prog, face, base)), kern.get((prog, face, arm))
            ks = med(kf) / med(ko) if ko and kf else float('nan')
            cls[(arm[4:], face)][prog][i] = (med(off) / med(on), disj, ks, med(on), med(off), base)
    print(f"| rung | face | rule on iter ({len(files)} session{'s' if len(files) > 1 else ''}) | rule on device ms | "
          "per probe: OFF/ON iter (`*` disjoint) [OFF/ON traced kernel ms], per session |")
    print('|---|---|---|---|---|')
    for (rung, face), progs in sorted(cls.items()):
        full = {p: s for p, s in progs.items() if len(s) == len(files) and None not in s.values()}
        pos = [p for p, s in full.items() if all(v[0] > 1 + EPS for v in s.values())]
        neg = [p for p, s in full.items() if all(v[0] < 1 - EPS and v[1] for v in s.values())]
        verdict = 'ON' if pos and not neg else 'OFF'
        # A rung that changes only kernels (a store order) is read on the device clock too: the
        # same rule over the traced kernel sums, which the host's timing noise does not reach.
        kpos = [p for p, s in full.items() if all(v[2] > 1 + EPS for v in s.values())]
        kneg = [p for p, s in full.items() if all(v[2] < 1 - EPS for v in s.values())]
        kverdict = 'ON' if kpos and not kneg else 'OFF'
        cell = lambda v: ('pick≠' if v is None else f"{v[0]:.3f}{'*' if v[1] else ''} [{v[2]:.3f}]"
                          + ('' if v[5] == 'on' else f' (vs {v[5]})'))
        cells = '; '.join(f"{p[3:]}: " + ' '.join(cell(s[i]) for i in sorted(s)) for p, s in sorted(progs.items()))
        print(f"| {rung} | {face} | {verdict} | {kverdict} | {cells} |")


if __name__ == '__main__':
    main(sys.argv[1:])
