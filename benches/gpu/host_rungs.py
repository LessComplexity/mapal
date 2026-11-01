#!/usr/bin/env python3
"""A host-placement rung's ON vs OFF arm over one part's box_run sessions (different boxes),
read from the `e2e prog=` and `probe prog=` rows (E2E_ARMS / PROBE_ARMS = "on off-<rung>").

Per session and (program, face): the process wall, the program's own `iter ms`, the trace's
alloc+free ms and alloc count, medians (min-max); OFF/ON > 1 = ON faster; `disjoint` = the
ranges do not overlap; the cuBLAS whole-program floor (e2e) or the best library leg (probes).
Then the rule, per (program, face) and metric: stable+ when OFF/ON > 1.01 in every session,
stable- when < 0.99 in every session (`*` = disjoint in every session). The rung ships ON on
the part iff some row is stable+ and none is stable-*.

usage: python3 benches/gpu/host_rungs.py <rung> <part> <results-session>..."""
import re
import sys

med = lambda xs: sorted(xs)[len(xs) // 2]
EPS = 0.01  # the S54 rule's no-effect band (rungs.py)


def fields(l):
    return dict(kv.split('=', 1) for kv in re.findall(r'(\w+=[^\s\[\]]+)', l))


def read(path):
    runs, floor, legs = {}, {}, {}
    for l in open(path):
        if l.startswith(('e2e prog=', 'probe prog=')):
            f = fields(l)
            it = re.search(r'iter ms=\s*([-\d.eE+]+)', l)
            r = runs.setdefault((f['prog'], f['face']), {}).setdefault(f['arm'], [])
            r.append({'wall': int(f['wall_us']) / 1e3, 'iter': float(it[1]) if it else None,
                      'af': float(f.get('alloc_ms', 'nan')) + float(f.get('free_ms', 'nan')),
                      'n': f.get('allocs', '?'), 'out': l.split('out=[')[1].split(']')[0].split('iter')[0].strip()})
        elif l.startswith('floor prog='):
            f = fields(l)
            floor.setdefault(f['prog'], []).append(float(f['total_ms']))
        elif l.startswith('chain probe='):
            f = fields(l)
            legs.setdefault(f"pt_{f['probe']}_{f['n']}", {})[f['leg']] = float(f['med_ms'])
    return runs, floor, legs


def vals(rs, key):
    return [r[key] for r in rs if r[key] is not None]


def session(path, off):
    runs, floor, legs = read(path)
    print(f"\n{path}\n")
    print("| program | face | reps | wall ms on | wall ms off | wall off/on | iter ms on | iter ms off | iter off/on "
          "| alloc+free ms on / off | allocs on / off | same out | library |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for (p, face), arms in runs.items():
        on, of = arms.get('on'), arms.get(off)
        if not on or not of:
            continue
        cells = []
        for key in ('wall', 'iter'):
            a, b = vals(on, key), vals(of, key)
            if not a or not b:
                cells.append(' | | ')
                continue
            dis = ' disjoint' if max(a) < min(b) or max(b) < min(a) else ''
            cells.append(f"{med(a):.2f} ({min(a):.1f}–{max(a):.1f}) | {med(b):.2f} ({min(b):.1f}–{max(b):.1f}) "
                         f"| {med(b) / med(a):.3f}{dis}")
        outs = {r['out'] for r in on} | {r['out'] for r in of}
        lib = f"cuBLAS floor {med(floor[p]):.1f} ms" if p in floor else ''
        if p in legs:
            k, v = min(legs[p].items(), key=lambda kv: kv[1])
            lib = f"best {k} {v:.3f} ms"
        print(f"| {p} | {face} | {len(on)}/{len(of)} | {cells[0]} | {cells[1]} "
              f"| {med(vals(on, 'af')):.2f} / {med(vals(of, 'af')):.2f} | {on[0]['n']} / {of[0]['n']} "
              f"| {'yes' if len(outs) == 1 else 'NO'} | {lib} |")
    lines = open(path).readlines()
    diff = ' '.join(l.strip() for l in lines if l.startswith('differential:'))
    print(f"\n{diff}; trap {sum(l.startswith('TRAP-PASS') for l in lines)} pass / "
          f"{sum(l.startswith('TRAP-FAIL') for l in lines)} fail")
    return runs


def decide(part, paths, sessions, off):
    keys = [k for k in sessions[0] if all(off in s.get(k, {}) and 'on' in s.get(k, {}) for s in sessions)]
    total = {'wall': [0, 0, 0, 0], 'iter': [0, 0, 0, 0]}
    print(f"\n{part}: OFF/ON per session ({', '.join(paths)})\n")
    print("| program | face | wall OFF/ON per session | wall | iter OFF/ON per session | iter |")
    print("|---|---|---|---|---|---|")
    for k in keys:
        cells = []
        for m in ('wall', 'iter'):
            rs, dis = [], True
            for s in sessions:
                a, b = vals(s[k]['on'], m), vals(s[k][off], m)
                if not a or not b:
                    rs = None
                    break
                rs.append(med(b) / med(a))
                dis &= max(a) < min(b) or max(b) < min(a)
            if rs is None:
                cells += ['', '']
                continue
            sign = '+' if all(r > 1 + EPS for r in rs) else '-' if all(r < 1 - EPS for r in rs) else '0'
            total[m]['+-0'.index(sign)] += 1
            total[m][3] += sign == '-' and dis
            cells += [' / '.join(f'{r:.3f}' for r in rs), sign + ('*' if dis and sign != '0' else '')]
        print(f"| {k[0]} | {k[1]} | {' | '.join(cells)} |")
    on = any(t[0] for t in total.values()) and not any(t[3] for t in total.values())
    for m, (p, n, z, nd) in total.items():
        print(f"\n{m}: {p} stable+, {n} stable- ({nd} disjoint), {z} no stable sign")
    print(f"\n{part}: ships {'ON' if on else 'OFF'}")


if __name__ == '__main__':
    rung, part, paths = sys.argv[1], sys.argv[2], sys.argv[3:]
    off = f'off-{rung}'
    decide(part, paths, [session(p, off) for p in paths], off)
