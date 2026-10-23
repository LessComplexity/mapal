#!/usr/bin/env python3
"""S64 evidence pass, C: re-judge every GPU_RUNGS_OFF row on its static evidence.

usage:
  rejudge.py arms [profile...]          the ARMS spec (integrated_bundle.sh) of every row's cited cells:
                                        `<profile>:<stem>:<face>:<rung>:on` — the rung forced ON where the
                                        part ships it OFF (A100 rows: no box in scope, skipped)
  rejudge.py judge <bundle> [results-<arch>.txt...]   one TSV row per GPU_RUNGS_OFF row:
      row profile rung fill face cells status delta  (+ re-measured ON/OFF where a results file has them)

A row's cells come from its cite: `n=a/b/c` / `n=a..b` (square cells, `mm_<n>`), `MxNxK`, `pt_*`,
`decode_1024_256`; its faces are the cite's face tokens the row's Face covers (`_exact` → `_x`, `_c`,
`_tf`, `_r`, `_rc`; an S63 cell without a suffix is the default face; an S62 one, exact).
Per cell, `evidence.py diff` of the shipped emission vs the ON arm (dropped by the bundle when equal):
  no-op at cited cells  the ON arm is the shipped program at every in-class cited cell (the row is kept:
                  another cell of its class may differ; S64 records it, does not remove it)
  explained       an in-class ON arm carries a cost the cited timing sign agrees with (ON not faster):
                  more spills, fewer warps/SM, more instructions per k-step, more barriers per k-step,
                  fewer FFMA per shared load, another kernel kind, more launches / copies / allocations
  open            deltas, none of them a cost
  unverified      no in-class cell could be built (out of class everywhere, or no cell named)
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evidence import KERNEL_RUNGS, diff, rows_off  # noqa: E402

PROFILES = ('rtx-4090', 'rtx-5090', 'h100-pcie', 'rtx-3090')
FACE_TOK = {'_exact': '_x', '_c': '_c', '_tf': '_tf', '_r': '_r', '_rc': '_rc'}
COVERS = {'Exact': {'_x', '_r'}, 'Contract': {'_c', '', '_rc', '_tf'}, 'Tf32': {'_tf'},
          'Any': {'_x', '_c', '', '_r', '_rc', '_tf'}}
# Host-rung rows name probes without their `pt_` (l2-window, streams, pdl): the programs they cite.
PROBE = re.compile(r'\b(?:pt_)?((?:rm_m\d+_d\d+_h\d+)|(?:(?:twin|diamond|mlp2|mlp|loop|rowsum|softmax_rows|indep|'
                   r'fanout|epi|bias|ew|residual|rmlp\d+x\d+|rmlp)_\d+(?:_\d+)?))(_c|_exact)?\b')


def cells(cite, face):
    """[(stem, face suffix)] the cite names, on the faces the row covers."""
    out = []
    faces = {FACE_TOK[m] for m in re.findall(r'(?<![\w.])(_exact|_c|_tf|_rc|_r)\b', cite)}
    s63 = 's63/' in cite
    s62 = 's62/' in cite or 's62 v2' in cite
    want = COVERS[face]
    for m in re.finditer(r'(?<![\w])(?:mm_)?(\d+x\d+x\d+)(_rc|_r|_c|_x)?\b', cite):
        # S62 files: a bare stem is the exact face, `_r` / `_rc` the fixed faces of the same cell.
        f = ({m.group(2)} if m.group(2) else {''} if s63 else
             ({'_x'} | (faces & {'_r', '_rc'}) if s62 else faces or {''}))
        out += [(f'mm_{m.group(1)}', x) for x in f & want or (f if face == 'Any' else set())]
    for m in re.finditer(r'n=([\d/.]+)', cite):
        v = m.group(1).rstrip('.')
        ns = ([1024, 2048, 4096, 8192] if '..' in v else [int(x) for x in v.split('/') if x.isdigit()])
        f = (faces & want) or (want & {'_x', '_c'})
        out += [(f'mm_{n}', x) for n in ns for x in f]
    for m in re.finditer(r'mm_(\d+)(?!\d|x)', cite):
        out += [(f'mm_{m.group(1)}', x) for x in (faces & want) or (want & {'_x', '_c'})]
    if re.search(r'\bdecode\b|decode_1024_256', cite):
        out += [('decode_1024_256', x) for x in want & {'_x', '_c'}]
    for m in PROBE.finditer(cite):
        stem = 'pt_' + m.group(1)
        if s62 and not m.group(2) and stem.startswith('pt_rm_'):
            f = {'_x'} & want
        else:
            f = {FACE_TOK[m.group(2)]} if m.group(2) else (faces & want) or (want & {'_x', '_c'})
        out += [(stem, x) for x in f & want]
    # Rows that name a probe and its sizes apart (`pt_rowsum … 4096 … 2048`, `contract 2048: diamond, mlp2`).
    sizes = sorted(set(re.findall(r'(?<![\w.x/])(512|1024|2048|4096)(?![\w.x/])', cite)))
    for name in re.findall(r'\b(?:pt_)?(rowsum|diamond|mlp2|mlp)\b(?!_\d)', cite):
        out += [(f'pt_{name}_{n}', x) for n in sizes for x in (faces & want) or (want & {'_x', '_c'})]
    return sorted(set(out))


def arms(profiles):
    out = []
    for p, rung, fill, face, cite in rows_off():
        if p not in profiles:
            continue
        for stem, f in cells(cite, face):
            out.append(f'{p}:{stem}:{f}:{rung}:on')
    return sorted(set(out))


# The plan's pre-registered list (S64 design C.2): registers count only through the warps they cost,
# instruction counts only per k-step (`span`); a whole-function count is not an explanation.
COST = [  # (fact, sign): +1 = ON larger is a cost, -1 = ON smaller is a cost
    ('ptxas.spill_st', 1), ('occ.warps_per_sm', -1),
    ('sass.hot.span', 1), ('sass.hot.bar', 1), ('sass.hot.ffma_per_lds', -1), ('sass.hot.ffma_per_ldg', -1),
    ('host.calls.launch', 1), ('host.calls.launchv', 1), ('host.calls.launch_l2', 1), ('host.calls.h2d', 1),
    ('host.calls.d2h', 1), ('host.calls.alloc', 1), ('host.calls.tile', 1), ('host.calls.zero', 1),
    ('host.calls.event', 1), ('host.calls.pin', 1), ('launch.S', 1)]


def classify(delta):
    """delta: [(fact, kernel, off, on)] -> (costs, benefits)"""
    cost, gain = [], []
    for fact, k, off, on in delta:
        for name, sign in COST:
            if fact == name or fact.endswith('.' + name) or fact == name.split('.', 1)[1] and name.startswith('host'):
                if isinstance(off, (int, float)) and isinstance(on, (int, float)) and off != on:
                    (cost if (on - off) * sign > 0 else gain).append(f'{fact} {off}->{on}')
    return cost, gain


def judge(bundle, results):
    rows = rows_off()
    facts = {}
    for p in PROFILES:
        fp = os.path.join(bundle, p, 'evidence', 'facts.json')
        if os.path.exists(fp):
            facts[p] = json.load(open(fp))
    remeasured = remeasure(results)
    print('row\tprofile\trung\tfill\tface\tcells\tstatus\tevidence\tremeasured\tchange')
    for i, (p, rung, fill, face, cite) in enumerate(rows):
        if p == 'a100':
            print(f'{i}\t{p}\t{rung}\t{fill}\t{face}\t-\tunverified (no box)\t-\t-\tno')
            continue
        fx = facts.get(p, {})
        notes, costs, gains, inclass, noop = [], [], [], 0, 0
        for stem, f in cells(cite, face):
            base, arm = stem + f, f'{stem}{f}.on-{rung}'
            if base not in fx:
                notes.append(f'{base}: not built')
                continue
            if arm not in fx:  # dropped: equal to the shipped emission
                on_now = rung_on(fx[base], rung)
                if on_now:
                    notes.append(f'{base}: rung already ON here (other class)')
                else:
                    inclass += 1
                    noop += 1
                    notes.append(f'{base}: ON arm = shipped')
                continue
            inclass += 1
            d = diff(os.path.join(bundle, p), base, arm)
            c, g = classify(d)
            costs += [f'{base} {x}' for x in c]
            gains += [f'{base} {x}' for x in g]
            if not c and not g:
                notes.append(f'{base}: {len(d)} other deltas ({", ".join(x[0] for x in d[:4])})')
        if inclass == 0:
            st = 'unverified (no in-class cell)'
        elif noop == inclass:
            st = 'no-op at cited cells'
        elif costs:
            st = 'explained'
        else:
            st = 'open'
        ev = '; '.join(costs[:6] + [f'+{x}' for x in gains[:3]] + notes[:4])
        mine = {k[1]: v for k, v in remeasured.items() if k[0] == p and k[2] == rung
                and k[1] in {s + f for s, f in cells(cite, face)}}
        rm = '; '.join(fmt_remeasured(b, v) for b, v in sorted(mine.items())) or '-'
        # The pre-registered change rule (S64 design C.4): ON faster beyond null in every process at every
        # re-measured cell AND a static gain that explains it; otherwise the row stands.
        change = ('remove: ON faster, static gain ' + '; '.join(gains[:2])
                  if mine and gains and all(v['sign'] == 'faster' for v in mine.values()) else 'no')
        print(f'{i}\t{p}\t{rung}\t{fill}\t{face}\t{" ".join(s + f for s, f in cells(cite, face))}\t{st}\t{ev}\t{rm}'
              f'\t{change}')


def rung_on(e, rung):
    if rung in KERNEL_RUNGS:
        return any(rung in k.get('plan', {}).get('rungs_on', []) for k in e['kernels'].values()
                   if k.get('plan', {}).get('kind') in ('tile', 'naive', 'mma', 'wgmma'))
    return rung in e.get('host_table', [])


def remeasure(results):
    """(profile, base stem, rung) -> dict(on, off, ratios, null, sign) from a box's results: tile_bench /
    mlp_bench `time leg=` medians per process (pass=1, 2), probe `iter ms` per repetition (each its own
    process, ABBA). `sign`: slower / faster when every process's ON/OFF is beyond the part's null
    (max |null / shipped - 1| of that cell or program), else inside."""
    out = {}
    for path in results:
        prof = re.search(r'results-([\w-]+)\.txt', path).group(1)
        med, probe = {}, {}
        for line in open(path, errors='replace'):
            if m := re.search(r'^pass=(\d) (?:cell|mlp)=\S+ time leg=(\S+) .*?(?:event_med_ms|med_ms)=([\d.]+)', line):
                med.setdefault((m.group(2), m.group(1)), float(m.group(3)))
            if line.startswith('probe prog=') and (it := re.search(r'iter ms=\s*([-\d.eE+]+)', line)):
                f = dict(re.findall(r'(\w+)=([^\s\[\]]+)', line))
                face = '' if f['face'] == '_default' else f['face']
                leg = f['prog'] + face + ('' if f['arm'] == 'on' else '.' + f['arm'])
                probe.setdefault(leg, {})[f['rep']] = float(it.group(1))
        for (leg, ps), v in med.items():
            probe.setdefault(leg, {})[f'p{ps}'] = v
        for leg, on in probe.items():
            m = re.match(r'(.+)\.on-([\w+-]+)$', leg)
            if not m or m.group(1) not in probe:
                continue
            base, off = m.group(1), probe[m.group(1)]
            ks = sorted(k for k in on if k in off)
            if not ks:
                continue
            r = [on[k] / off[k] for k in ks]
            # the null: the default face's `.null` arm (cells `mm_X.null`, MLP `_c.null`, programs `.null`)
            root = re.sub(r'_(x|c|r|rc|tf)$', '', base)
            nl = probe.get(root + '.null') or probe.get(root + '_c.null') or {}
            sh = probe.get(root) or probe.get(root + '_c') or {}
            nul = [abs(nl[k] / sh[k] - 1) for k in nl if k in sh]
            null = max(nul) if nul else None
            sign = ('slower' if null is not None and all(x > 1 + null for x in r) else
                    'faster' if null is not None and all(x < 1 - null for x in r) else 'inside')
            out[(prof, base, m.group(2))] = dict(on=[on[k] for k in ks], off=[off[k] for k in ks], ratios=r,
                                                 null=null, sign=sign, keys=ks)
    return out


def fmt_remeasured(base, d):
    r = '/'.join(f'{x:.3f}' for x in d['ratios'])
    n = '-' if d['null'] is None else f"{d['null']:.3f}"
    return f"{base} ON/OFF {r} null {n} {d['sign']}"


if __name__ == '__main__':
    if sys.argv[1] == 'arms':
        print(' '.join(arms(sys.argv[2:] or PROFILES)))
    elif sys.argv[1] == 'cells':
        for i, (p, rung, fill, face, cite) in enumerate(rows_off()):
            print(i, p, rung, fill, face, cells(cite, face))
    else:
        judge(sys.argv[2], sys.argv[3:])
