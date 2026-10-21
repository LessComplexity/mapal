#!/usr/bin/env python3
"""The S54 rule for a placement rung over one part's sessions (box_run probe rows, different boxes).

Per rung and face, per probe and size: s = (OFF iter ms) / (ON iter ms), medians within one
session (`probe prog=… arm=on|off-<rung>`, the program's own clock around every computation).
A size is stable+ when s > 1.01 in every session, stable- when s < 0.99 in every session; `*` marks
disjoint min-max ranges. The rung ships ON for a face iff some probe size is stable+ and none is
stable- with disjoint ranges in every session (a placement that loses on one program is not shipped
on the strength of another). The traced kernel ms (`probe_kernels`, summed per run) is shown beside
it: the device share of the same comparison.
usage: python3 benches/gpu/probe_rungs.py <results-session-1> [<results-session-2>...]"""
import re
import sys
from collections import defaultdict

EPS = 0.01
med = lambda xs: sorted(xs)[len(xs) // 2]


def runs(path):
    it, kern = defaultdict(list), defaultdict(list)
    for l in open(path):
        if l.startswith('probe prog='):
            f = dict(re.findall(r'(\w+)=([^\s\[\]]+)', l))
            m = re.search(r'iter ms=\s*([-\d.eE+]+)', l)
            if m:
                it[(f['prog'], f['face'], f['arm'])].append(float(m[1]))
        elif l.startswith('probe_kernels '):
            f = dict(re.findall(r'(\w+)=([^\s\[\]]+)', l))
            kern[(f['prog'], f['face'], f['arm'])].append(sum(map(float, re.findall(r' ms=([\d.]+)', l))))
    return it, kern


def main(files):
    sess = [runs(f) for f in files]
    cls = defaultdict(lambda: defaultdict(dict))  # (rung, face) -> (prog) -> session -> (s, disj, ks)
    for i, (it, kern) in enumerate(sess):
        for (prog, face, arm), off in it.items():
            if not arm.startswith('off-') or (prog, face, 'on') not in it:
                continue
            on = it[(prog, face, 'on')]
            disj = max(on) < min(off) or max(off) < min(on)
            ko, kf = kern.get((prog, face, 'on')), kern.get((prog, face, arm))
            ks = med(kf) / med(ko) if ko and kf else float('nan')
            cls[(arm[4:], face)][prog][i] = (med(off) / med(on), disj, ks, med(on), med(off))
    print(f"| rung | face | rule ({len(files)} session{'s' if len(files) > 1 else ''}) | per probe: OFF/ON iter "
          "(`*` disjoint) [OFF/ON traced kernel ms], per session |")
    print('|---|---|---|---|')
    for (rung, face), progs in sorted(cls.items()):
        full = {p: s for p, s in progs.items() if len(s) == len(files)}
        pos = [p for p, s in full.items() if all(v[0] > 1 + EPS for v in s.values())]
        neg = [p for p, s in full.items() if all(v[0] < 1 - EPS and v[1] for v in s.values())]
        verdict = 'ON' if pos and not neg else 'OFF'
        cells = '; '.join(
            f"{p[3:]}: " + ' '.join(f"{s[i][0]:.3f}{'*' if s[i][1] else ''} [{s[i][2]:.3f}]" for i in sorted(s))
            for p, s in sorted(progs.items()))
        print(f"| {rung} | {face} | {verdict} | {cells} |")


if __name__ == '__main__':
    main(sys.argv[1:])
