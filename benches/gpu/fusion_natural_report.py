#!/usr/bin/env python3
"""fusion-natural-forms tables from the static evidence (`evidence.py facts`): per suite program and arch, the
sprint's base build (`base/<arch>`: 27b4f99 on its own programs), the merged build with the three new rungs off
(`.off-fuse+window+range-map`: "before" in one build) and shipped — launches, device buffers, what the fuse
units keep and store, host<->device bytes — intended (the emitter's `; mapal.fuse.host` / `; mapal.fuse` /
`; mapal.move` lines) against realized (host IR call sites and bytes). Then the intent check per rung.

  launches  host call sites of mapal_gpu_launch / launchv / launch_l2 / tile (a call in a loop body runs every
            trip); `gen` (devgen operand builds) apart
  buffers   device objects the host asks the arena / allocator for
  kept      fuse members no outside reader needs: never stored (intended) — realized: buffers(off-fuse) − buffers
  copies    whole-program h2d / d2h bytes (realized, host IR); per trip: the facts' `; mapal.move` → realized

usage: fusion_natural_report.py <raw dir>   (<raw>/static/<arch>/facts.json[.gz], <raw>/base/<arch>/facts.json[.gz])
"""
import gzip
import json
import os
import sys
from collections import Counter

ARCHS = ('rtx-4090', 'rtx-5090', 'h100-pcie', 'rtx-3090')
NEW = '.off-fuse+window+range-map'


def load(raw, sub, a):
    p = os.path.join(raw, sub, a, 'facts.json')
    if os.path.exists(p + '.gz'):
        return json.load(gzip.open(p + '.gz'))
    return json.load(open(p)) if os.path.exists(p) else {}


def arm(fx, stem, suffix):
    """An arm the bundle dropped equals the shipped build."""
    return fx.get(stem + suffix) or fx.get(stem)


def launches(e):
    c = e['host'].get('calls', {})
    return sum(c.get(k, 0) for k in ('launch', 'launchv', 'launch_l2', 'tile'))


def bufs(e):
    c = e['host'].get('calls', {})
    return c.get('arena', 0) + c.get('alloc', 0)


def units(e):
    return [k['fuse'] for k in e['kernels'].values() if k.get('plan', {}).get('kind') == 'fuse']


def kinds(e):
    k = Counter(x.get('plan', {}).get('kind', '?') for x in e['kernels'].values())
    k.pop('gen', None)
    return ' '.join(f'{n}{"×" + str(v) if v > 1 else ""}' for n, v in sorted(k.items())) or 'host'


def size(b):
    return f'{b / 2 ** 20:.1f} M' if b >= 2 ** 20 else f'{b / 2 ** 10:.1f} K' if b >= 1024 else str(b)


def trip(e):
    """Per trip, every program loop: h2d / d2h / host copy, the facts' → realized."""
    lp = e['host'].get('move', {}).get('per_loop', [])
    if not lp:
        return '-'
    f = lambda k, r: (f'{size(sum(int(x["intent"].get(k, 0)) for x in lp))}→'
                      f'{size(sum(x[r] for x in lp))}')
    return f'{f("h2d", "h2d_b")} / {f("d2h", "d2h_b")} / {f("host_copy", "copy_b")}'


def whole(e):
    m = e['host'].get('move', {})
    if 'runtime_h2d_b' not in m:
        return '-'
    return f'{size(m["runtime_h2d_b"])} / {size(m["runtime_d2h_b"])}'


def program_rows(raw):
    rows = {}
    for a in ARCHS:
        fx, bx = load(raw, 'static', a), load(raw, 'base', a)
        for stem in sorted(s for s in fx if '.' not in s and s.startswith('pt_') and not s.endswith(('_c', '_x'))):
            on, before, base = fx[stem], arm(fx, stem, NEW), bx.get(stem)
            off_fuse = arm(fx, stem, '.off-fuse')
            u = units(on)
            kept = sorted({k for x in u for k in x.get('keep', [])})
            stored = sorted({k for x in u for k in x.get('store', [])})
            fh = on['host'].get('fuse', {})
            want = fh.get('launches')
            got = launches(on) + on['host']['calls'].get('gen', 0)
            lch = 'ok' if not want or want == got else f'MISMATCH {want}≠{got}'
            nk = sum(x.get('n_keep', 0) for x in u)
            kch = 'ok' if bufs(off_fuse) - bufs(on) == nk else f'MISMATCH {bufs(off_fuse)}−{bufs(on)}≠{nk}'
            b = lambda e, f: str(f(e)) if e else '-'
            cells = (f'{b(base, launches)} → {launches(before)} → {launches(on)} | {lch} | {kinds(on)} | '
                     f'{b(base, bufs)} → {bufs(before)} → {bufs(on)} | {",".join(kept) or "-"} | '
                     f'{",".join(stored) or "-"} | {kch} | {whole(base) if base else "-"} → {whole(on)} | '
                     f'{trip(base) if base else "-"} ⇒ {trip(on)}')
            rows.setdefault((stem, cells), []).append(a)
    return rows


def cpu_rows(raw):
    rows = {}
    for a in ARCHS:
        fx, bx = load(raw, 'static', a), load(raw, 'base', a)
        for stem in sorted(s for s in fx if s.endswith('.cpu') and s.startswith('pt_')):
            p = stem[:-4]
            on, base = fx[stem], bx.get(stem)
            if not on['host'].get('move', {}).get('per_loop'):
                continue
            ins = lambda e: e['host'].get('obj', {}).get('insns', '-') if e else '-'
            cells = f'{trip(base) if base else "-"} ⇒ {trip(on)} | {ins(base)} → {ins(on)}'
            rows.setdefault((p, cells), []).append(a)
    return rows


def fusion_rows(raw):
    """Every unit kernel (shipped): its members (`name:kind:1x<cells per row>[+halo]`), what it keeps and
    stores; the host line's refusals (`producer->consumer:reason`)."""
    rows = {}
    for a in ARCHS:
        fx = load(raw, 'static', a)
        for stem in sorted(s for s in fx if '.' not in s and s.startswith('pt_')):
            e = fx[stem]
            for u in units(e):
                rows.setdefault((stem, u.get('members', '?'), ','.join(u.get('keep', [])) or '-',
                                 ','.join(u.get('store', [])) or '-'), []).append(a)
    return rows


RUNG_ARMS = ('.off-fuse', '.off-window', '.off-range-map', '.off-lane-stride')


def arm_rows(raw):
    """Each new rung's OFF arm (and `lane-stride`, which gates the strided 2-D sites), where the bundle kept it
    (it differs from shipped): launches, kernels, device buffers, whole-program h2d / d2h, per trip, host insns."""
    rows = {}
    ins = lambda e: e['host'].get('obj', {}).get('insns', '-')
    for a in ARCHS:
        fx = load(raw, 'static', a)
        for suffix in RUNG_ARMS:
            for stem in sorted(s for s in fx if s.endswith(suffix) and s.startswith('pt_')):
                p = stem[:-len(suffix)]
                off, on = fx[stem], fx[p]
                cells = (f'{launches(off)} → {launches(on)} | {kinds(off)} → {kinds(on)} | {bufs(off)} → {bufs(on)} | '
                         f'{whole(off)} → {whole(on)} | {trip(off)} ⇒ {trip(on)} | {ins(off)} → {ins(on)}')
                rows.setdefault((suffix[5:], p, cells), []).append(a)
    return rows


def intents(raw):
    """Per arch and rung: checks ok / mismatch / n-a; a mismatch's (rung, check) and whether `.off-*` of a new
    rung (or the base) shows it too."""
    out = []
    for a in ARCHS:
        p = os.path.join(raw, 'static', a, 'evidence.tsv')
        if not os.path.exists(p):
            continue
        st = Counter()
        mm = Counter()
        for line in open(p):
            f = line.rstrip('\n').split('\t')
            if len(f) < 8 or f[0] == 'arch':
                continue
            rung = f[3].split(':')[0]
            st[(rung, f[7])] += 1
            if f[7] == 'MISMATCH':
                mm[(rung, f[4], f[1].split('.')[1] if '.' in f[1] else 'shipped')] += 1
        out.append((a, st, mm))
    return out


def prev_ratios(readme, heading='### Workload suite: iter ms (median of 30 reps), Mapal vs the best library'):
    """(workload, arch) -> `Mapal / best other` of the README's S65 suite table."""
    out, on = {}, False
    for line in open(readme):
        if line.startswith('###'):
            on = line.startswith(heading)
        elif on and line.startswith('| ') and not line.startswith('| workload'):
            c = [x.strip() for x in line.strip().strip('|').split('|')]
            if len(c) >= 8:
                out[(c[0], c[1])] = c[7]
    return out


def timing(raw):
    """The box runs (<raw>/<arch>/results-<arch>.txt): the suite against the best other leg, S65's ratio beside
    it; then each new rung's OFF arm against shipped per program — traced launches, kernel ms, iter ms."""
    import re
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import evidence_report as er
    readme = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../docs/performance/evidence/README.md')
    prev = prev_ratios(readme)
    # `<raw>/confirm/<arch>`: a second box of the arch (another host); its own ratio beside the first's.
    conf = {(r['key'], r['arch']): r for r in er.suite_rows(os.path.join(raw, 'confirm'))}
    print('\n### Suite: iter ms (median of the reps), Mapal vs the best other leg on the same box\n')
    print('| workload | arch | CPU-only | CPU+GPU | null | best other (ms) | Mapal / best other | confirm box: '
          'CPU+GPU, best other, ratio | S65 ratio |')
    print('|---|---|---|---|---|---|---|---|---|')
    for r in sorted(er.suite_rows(raw), key=lambda r: (r['key'], ARCHS.index(r['arch']))):
        b = f'{r["best"][0]} {r["best"][1]:.4f}' if r['best'] else '-'
        c = conf.get((r['key'], r['arch']))
        cb = (f'{er.f4(c["on"])}, {c["best"][0] + " " + er.f4(c["best"][1]) if c["best"] else "-"}, '
              f'{er.f3(c["ratio"])}') if c else '-'
        print(f'| {r["key"]} | {r["arch"]} | {er.f4(r["cpu"])} | {er.f4(r["on"])} | {er.f3(r["null"])} | {b} | '
              f'{er.f3(r["ratio"])} | {cb} | {prev.get((r["key"], r["arch"]), "-")} |')
    print('\n### Each new rung ON vs its OFF arm (traced rep 1; iter ms median of the reps; OFF/ON per rep, '
          'median; null = |null/shipped − 1| max)\n')
    print('| program | arch | box | arm | launches ON / OFF | device buffers ON / OFF | kernel ms ON / OFF | '
          'iter ms ON / OFF | OFF/ON | null | sign |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    for a, box, rr in [(a, b, r) for a in ARCHS for b, r in (('main', raw), ('confirm', os.path.join(raw, 'confirm')))]:
        lines = er.results(rr, a)
        lg = er.legs(lines)
        tr = {}
        for l in lines:
            if l.startswith('probe prog=') and 'mapal_gpu_trace_total' in l and ' face=_default ' in l:
                f = dict(re.findall(r'(\w+)=([^\s\[\]]+)', l))
                tr.setdefault((f['prog'], f['arm']), (int(f.get('launches', 0)), float(f.get('kernel_ms', 0))))
        fx = load(raw, 'static', a)
        for leg in sorted(lg):
            m = re.match(r'(pt_\w+)\.(off-(?:fuse|window|range-map|lane-stride)[\w+-]*)$', leg)
            if not m or m[1] not in lg:
                continue
            p, arm_ = m[1], m[2]
            on, off = lg[p], lg[leg]
            ks = [k for k in off if k in on]
            if not ks:
                continue
            ratio = er.med([off[k] / on[k] for k in ks])
            nl = lg.get(p + '.null', {})
            xs = [abs(nl[k] / on[k] - 1) for k in nl if k in on]
            nul = max(xs) if xs else None
            rs = [off[k] / on[k] for k in ks]
            sign = ('ON faster' if nul is not None and all(x > 1 + nul for x in rs) else
                    'ON slower' if nul is not None and all(x < 1 - nul for x in rs) else 'inside null')
            t_on, t_off = tr.get((p, 'on')), tr.get((p, arm_))
            e_on, e_off = fx.get(p), fx.get(f'{p}.{arm_}')
            lb = f'{t_on[0]} / {t_off[0]}' if t_on and t_off else '-'
            kb = f'{t_on[1]:.3f} / {t_off[1]:.3f}' if t_on and t_off else '-'
            bb = f'{bufs(e_on)} / {bufs(e_off)}' if e_on and e_off else '-'
            print(f'| {p.removeprefix("pt_")} | {a} | {box} | {arm_} | {lb} | {bb} | {kb} | {er.f4(er.med(list(on.values())))} / '
                  f'{er.f4(er.med(list(off.values())))} | {ratio:.3f} | {er.f3(nul)} | {sign} |')


def main(raw):
    if any(os.path.exists(os.path.join(raw, a, f'results-{a}.txt')) for a in ARCHS):
        timing(raw)
    rows = program_rows(raw)
    print('| program | archs | launches: base → new rungs off → shipped | launches: `fuse.host` intent = realized | '
          'kernels (shipped) | device buffers: base → off → shipped | kept (never stored) | stored fuse members | '
          'kept: buffers(off-fuse) − buffers = kept | h2d / d2h bytes, whole program: base → shipped | '
          'per trip h2d / d2h / host copy, facts→realized: base ⇒ shipped |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    for (stem, cells), archs in sorted(rows.items()):
        print(f'| {stem} | {"all four" if len(archs) == len(ARCHS) else " ".join(archs)} | {cells} |')
    rows = cpu_rows(raw)
    if rows:
        print('\nCPU-only builds (loops):\n')
        print('| program | archs | per trip h2d / d2h / host copy, facts→realized: base ⇒ shipped | host insns: base → shipped |')
        print('|---|---|---|---|')
        for (stem, cells), archs in sorted(rows.items()):
            print(f'| {stem} | {"all four" if len(archs) == len(ARCHS) else " ".join(archs)} | {cells} |')
    print('\nUnit kernels (shipped):\n')
    print('| program | archs | members | kept | stored |')
    print('|---|---|---|---|---|')
    for (stem, m, k, st), archs in sorted(fusion_rows(raw).items()):
        print(f'| {stem} | {"all four" if len(archs) == len(ARCHS) else " ".join(archs)} | {m} | {k} | {st} |')
    print('\nEach rung OFF → shipped (arms the bundle kept: they differ from shipped):\n')
    print('| rung off | program | archs | launches | kernels | device buffers | h2d / d2h bytes, whole program | '
          'per trip h2d / d2h / host copy, facts→realized | host insns |')
    print('|---|---|---|---|---|---|---|---|---|')
    for (rung, stem, cells), archs in sorted(arm_rows(raw).items()):
        print(f'| {rung} | {stem} | {"all four" if len(archs) == len(ARCHS) else " ".join(archs)} | {cells} |')
    print('\nIntent checks (evidence.py check), ok / mismatch / n-a:\n')
    res = intents(raw)
    rungs = sorted({r for _, st, _ in res for r, _ in st})
    print('| rung | ' + ' | '.join(a for a, _, _ in res) + ' |')
    print('|---|' + '---|' * len(res))
    for r in rungs:
        print(f'| {r} | ' + ' | '.join(f'{st[(r, "ok")]} / {st[(r, "MISMATCH")]} / {st[(r, "n/a")]}'
                                       for _, st, _ in res) + ' |')
    print('\nMismatches by rung, check and arm (shipped = the merged build as it ships):\n')
    print('| arch | rung | check | arm | count |')
    print('|---|---|---|---|---|')
    for a, _, mm in res:
        for (r, c, armname), n in sorted(mm.items()):
            print(f'| {a} | {r} | {c} | {armname} | {n} |')


if __name__ == '__main__':
    main(sys.argv[1])
