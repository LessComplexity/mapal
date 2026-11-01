#!/usr/bin/env python3
"""Tables for the H100 PCIe disassembly report from the box results + local static analysis.

usage: report.py <results.txt> <trace dir> <static dir> <bundle/h100-pcie> > tables.md   (box trace)
       report.py <raw dir> > tables.md   (committed evidence; tar xJf raw/static/mapal-sass.tar.xz -C raw/static)
(also writes <static dir>/report-data.json)."""
import json
import os
import re
import statistics
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from hw import peaks, usage  # noqa: E402
from nsys_parse import api_trace, gpu_trace, kernel_summary  # noqa: E402
from occ import occupancy, waves  # noqa: E402
import mapal_static  # noqa: E402
import timing  # noqa: E402

CELLS = ['16x2048x7168', '16x1024x4096', '16x18432x7168', '128x2048x7168', '4096x4096x4096']
FACES = [('E', ''), ('C', '_c'), ('R', '_r'), ('RC', '_rc')]
LIBS = ['cublas', 'cutlass', 'cutlass_tuned']
LIBNAME = dict(cublas='cuBLAS', cutlass='CUTLASS stock', cutlass_tuned='CUTLASS tuned')


def dims(c):
    return [int(x) for x in c.split('x')]


def short(name):
    if name.startswith('mapal_'):
        return name
    m = re.search(r'GemmShapeILi(\d+)ELi(\d+)ELi(\d+)E', name)
    if 'cutlass6Kernel' in name and m:
        return f'cutlass::Kernel<SIMT {m.group(1)}x{m.group(2)}x{m.group(3)}>'
    n = re.sub(r'^_Z\d+', '', name)
    return n[:70]


def lib_of(name):
    if name.startswith('mapal_'):
        return 'mapal'
    if 'cutlass6Kernel' in name:
        return 'cutlass_tuned' if 'GemmShapeILi16ELi64ELi16E' in name else 'cutlass'
    if 'gen_operand' in name or 'gen_op' in name:
        return 'gen'
    return 'cublas'


def verdict(r1, r2, null):
    """r = Mapal / other, per pass; better/worse only when both passes clear the null band."""
    if r1 is None or r2 is None:
        return 'n/a'
    if r1 < 1 - null and r2 < 1 - null:
        return 'better'
    if r1 > 1 + null and r2 > 1 + null:
        return 'worse'
    return 'on-par'


def main(res, trace, static, bdir):
    T = timing.main(res)
    pk = peaks(T['facts'])
    sms = pk['sms']
    ms_json = os.path.join(static, 'mapal_static.json')
    MS = json.load(open(ms_json)) if bdir is None and os.path.exists(ms_json) else mapal_static.main(bdir, static, sms)
    out, data = [], dict(peaks=pk, facts=T['facts'])
    P = out.append

    # ---------- 1. per-cell timing and verdicts
    P('### Per-cell timing (ms, median of 2 processes x 30 CUDA events, L2 flushed) and verdicts\n')
    P('Ratio = Mapal / other (>1: Mapal slower). null = largest of |on - null copy| and pass-to-pass drift of any leg in the cell; '
      'a verdict needs both processes outside max(null, 1%).\n')
    P('| cell | face | Mapal | null | cuBLAS | CUTLASS stock | CUTLASS tuned | vs cuBLAS | vs stock | vs tuned |')
    P('|---|---|---:|---:|---:|---:|---:|---|---|---|')
    data['cells'] = {}
    for cell in CELLS:
        C = T['cells'].get(cell)
        if not C:
            continue
        st = f'mm_{cell}'
        nulls = []
        for fs in ('', '_c'):
            on, nu = C.get(st + fs, {}).get('pass', {}), C.get(st + fs + '.null', {}).get('pass', {})
            for p in (1, 2):
                if p in on and p in nu:
                    nulls.append(abs(on[p]['med'] - nu[p]['med']) / on[p]['med'])
        # pass-to-pass drift of every leg also widens the band
        for leg, d in C.items():
            ps = d.get('pass', {})
            if 1 in ps and 2 in ps:
                nulls.append(abs(ps[1]['med'] - ps[2]['med']) / ps[1]['med'])
        measured = max(nulls) if nulls else None
        null = max(nulls + [0.01]) if nulls else 0.02  # floor 1%
        row = dict(null=null, null_measured=measured, faces={})
        for f, fs in FACES:
            m = C.get(st + fs)
            if not m or not m.get('pooled_med'):
                continue
            cells_ = [f'{m["pooled_med"]:.4f}']
            vs = []
            fr = dict(ms=m['pooled_med'], pass_meds=[m['pass'][p]['med'] for p in (1, 2) if p in m['pass']])
            for L in LIBS:
                l = C.get(L)
                if not l:
                    vs.append('n/a'); continue
                r = m['pooled_med'] / l['pooled_med']
                rp = [m['pass'][p]['med'] / l['pass'][p]['med'] if p in m['pass'] and p in l['pass'] else None for p in (1, 2)]
                v = verdict(rp[0], rp[1], null)
                fr[L] = dict(ratio=r, pass_ratios=rp, verdict=v)
                vs.append(f'{v} {r:.2f}')
            row['faces'][f] = fr
            libs = [f'{C[L]["pooled_med"]:.4f}' if L in C else 'n/a' for L in LIBS]
            P(f'| {cell} | {f} | {cells_[0]} | {100 * measured:.2f}% | ' + ' | '.join(libs) + ' | ' + ' | '.join(vs) + ' |')
        row['libs'] = {L: C[L]['pooled_med'] for L in LIBS if L in C}
        data['cells'][cell] = row

    # ---------- 2. kernels executed per cell x implementation (nsys) + resources/occupancy
    P('\n### Kernels executed (nsys trace; regs/smem from the trace, occupancy computed for sm_90)\n')
    P('| cell | impl | kernel | per call | grid | block | regs | spill B | smem st/dyn | occ | warps/SM | waves | µs |')
    P('|---|---|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|')
    data['kernels'] = {}
    for cell in CELLS:
        base = os.path.join(trace, f'nsys_cell_{cell}')
        rows = gpu_trace(base)
        if not rows:
            continue
        ks = kernel_summary(rows)
        # calls per leg in that process: 3 warmups + 3 reps (+ determinism for Mapal arms)
        launch_of = {}
        for f, fs in FACES:
            e = MS.get(f'mm_{cell}{fs}')
            if e:
                launch_of[e['main']['kernel']] = (f, e)
                if 'reduce' in e:
                    launch_of[e['reduce']['kernel']] = (f, e)
        calls = dict(cublas=6, cutlass=6, cutlass_tuned=6)
        kl = []
        for name, d in sorted(ks.items(), key=lambda x: (lib_of(x[0]), x[0])):
            lib = lib_of(name)
            if lib == 'gen':
                continue
            impl = f'Mapal {launch_of[name][0]}' if name in launch_of else LIBNAME.get(lib, lib)
            per = d['n'] / 7 if lib == 'mapal' else d['n'] / calls.get(lib, 6)
            smem = d['static_smem'] + d['dyn_smem']
            threads = d['block'][0] * d['block'][1] * d['block'][2]
            spill = d.get('local')
            b, w, occ, lim = occupancy(threads, d['regs'], smem)
            nb = d['grid'][0] * d['grid'][1] * d['grid'][2]
            wv = waves(nb, b, sms)
            rw = min(nb, b * sms) * ((threads + 31) // 32) / sms
            kl.append(dict(impl=impl, name=name, per_call=per, grid=d['grid'], grids=d['grids'], block=d['block'],
                           regs=d['regs'], static_smem=d['static_smem'], dyn_smem=d['dyn_smem'], occ=occ, lim=lim,
                           warps_per_sm=rw, waves=wv, us=d['med_us'], spill=spill))
            P(f'| {cell} | {impl} | `{short(name)}` | {per:.2f} | {"x".join(map(str, d["grid"]))} | {threads} | '
              f'{d["regs"]} | {spill if spill is not None else "-"} | {d["static_smem"]}/{d["dyn_smem"]} | {occ:.2f} ({lim}) | '
              f'{rw:.1f} | {wv:.2f} | {d["med_us"]:.1f} |')
        data['kernels'][cell] = kl

    # ---------- 3. executed SASS (observer-captured images: the driver's JIT of Mapal's PTX, the library cubins)
    LS = {c['capture'].split('/')[-1]: c for c in json.load(open(os.path.join(static, 'libsass.json')))}
    data['sass'] = {}
    P('\n### Executed SASS, hot loop (static counts in the K-loop body of the image the observer saw execute)\n')
    P('MAC = FFMA, or FMUL+FADD pairs on the Exact face. acc = accumulation chains per thread; k/iter = MAC per chain per '
      'loop trip; MAC/gword = MAC per 4-byte global word loaded; MAC/LDS = MAC per shared-load instruction.\n')
    P('load cover = MAC instructions a warp issues between a register-staged global load and the first instruction '
      'that reads its data (min / median over the loop\'s loads; cyclic over the back-edge); cp.async = LDGSTS into a '
      'multi-stage shared ring, waited on by DEPBAR.\n')
    P('| cell | impl | kernel | loop insns | MAC | LDG (count x bits) | LDGSTS | LDS | STS | BAR+DEPBAR | SHFL | ISETP+BRA | MAC/gword | MAC/LDS | acc | k/iter | MAC share | load cover (MAC) |')
    P('|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|')
    for cell in CELLS:
        cap = LS.get(f'cap_cell_{cell}')
        if not cap:
            continue
        impl_of = {k['name']: k['impl'] for k in data['kernels'].get(cell, [])}
        rows = sorted((k for k in cap['kernels'] if k.get('in_region') and k['name'] in impl_of),
                      key=lambda k: (impl_of[k['name']], k['name']))
        for k in rows:
            l = k.get('sass', {}).get('loop')
            data['sass'].setdefault(cell, {})[k['name']] = dict(impl=impl_of[k['name']], **(k.get('sass') or {}))
            if not l:
                P(f'| {cell} | {impl_of[k["name"]]} | `{short(k["name"])}` | no loop ({k.get("sass", {}).get("insns")} insns) |' + ' |' * 14)
                continue
            w, mx = l['widths'], l['mix']
            ldg = ', '.join(f'{v}x{x.split(".")[1]}' for x, v in w.items() if x.startswith('LDG.'))
            P(f'| {cell} | {impl_of[k["name"]]} | `{short(k["name"])}` | {l["insns"]} | {l["mac"]} | {ldg or "-"} | '
              f'{mx.get("LDGSTS", 0)} | {mx.get("LDS", 0)} | {mx.get("STS", 0)} | {mx.get("BAR", 0) + mx.get("DEPBAR", 0) + mx.get("LDGDEPBAR", 0)} | '
              f'{mx.get("SHFL", 0)} | {mx.get("ISETP", 0) + mx.get("BRA", 0)} | {l["mac_per_global_word"]} | {l["mac_per_lds"]} | '
              f'{l["accumulators"]} | {l["unroll_k"]} | {l["mac_share"]} | '
              f'{"cp.async" if mx.get("LDGSTS") else ("%d / %d" % tuple(l["ldg_cover_mac"]) if l.get("ldg_cover_mac") else "-")} |')

    # ---------- 4. hardware usage
    P(f'\n### Hardware usage (peaks from the box: FP32 {pk["flops"] / 1e12:.1f} TFLOP/s, DRAM {pk["bw"] / 1e9:.0f} GB/s; bytes = A+B+C)\n')
    P('| cell | impl | ms | GB/s | % DRAM peak | TFLOP/s | % FP32 peak | AI (FLOP/B) | roof time ms | % of roof | class |')
    P('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|')
    data['usage'] = {}
    for cell in CELLS:
        if cell not in data['cells']:
            continue
        M, N, K = dims(cell)
        row = data['cells'][cell]
        impls = [(f'Mapal {f}', v['ms']) for f, v in row['faces'].items()] + [(LIBNAME[L], v) for L, v in row['libs'].items()]
        data['usage'][cell] = {}
        for name, ms in impls:
            u = usage(M, N, K, ms, pk)
            if name in ('Mapal E', 'Mapal R'):  # separate FMUL+FADD per MAC (SASS): half the FFMA issue ceiling
                u = usage(M, N, K, ms, dict(pk, flops=pk['flops'] / 2))
                u['fp32_pct'] /= 2; u['tflops'] = 2.0 * M * N * K / (ms * 1e-3) / 1e12
                name = name + ' (ceiling: FMUL+FADD)'
            data['usage'][cell][name] = u
            P(f'| {cell} | {name} | {ms:.4f} | {u["gbs"]:.0f} | {u["bw_pct"]:.1f} | {u["tflops"]:.2f} | {u["fp32_pct"]:.1f} | '
              f'{u["ai"]:.1f} | {u["roof_ms"]:.4f} | {100 * u["roof_frac"]:.0f}% | {u["cls"]} |')

    # ---------- 5. MLP whole programs
    from mlp import library_legs, mapal_region
    P('\n### Residual MLP whole programs (Mapal `iter ms`: one region per process, median of 30 processes; '
      'library: host wall per call / CUDA-event ms, median of 2 x 30 calls, operands L2-resident as in the program)\n')
    P('| program | face | Mapal on | null | base | on, CUDA_MODULE_LOADING=EAGER | traced kernel ms (k1, k2) | cuBLAS wall / event | cuBLASLt fused wall / event | CUTLASS stock fused | CUTLASS tuned fused |')
    P('|---|---|---:|---:|---:|---:|---|---|---|---|---|')
    data['mlp'] = {}
    for prog in ('pt_rm_m128_d256_h512', 'pt_rm_m128_d1024_h2048'):
        lib = {}
        for r in T['rmlib'].get(prog, {}).get('rm', []):
            lib.setdefault(r['leg'], []).append(r)
        L = {k: dict(wall=statistics.median(float(x['wall_med_ms']) for x in v),
                     event=statistics.median(float(x['event_med_ms']) for x in v),
                     first=statistics.median(float(x['first_wall_ms']) for x in v),
                     passes=[(float(x['wall_med_ms']), float(x['event_med_ms'])) for x in v]) for k, v in lib.items()}
        for f, fs in (('E', ''), ('C', '_c')):
            A = T['mlp'].get(prog + fs, {})
            g = lambda a: A.get(a, {}).get('iter_med')
            tr = A.get('on:trace', {}).get('kernel_ms', [])
            cells = [f'{g(a):.4f}' if g(a) is not None else '-' for a in ('on', 'null', 'base', 'eager')]
            libs = [f'{L[k]["wall"]:.4f} / {L[k]["event"]:.4f}' if k in L else '-' for k in ('cublas', 'lt', 'cutlass', 'cutlass_tuned')]
            P(f'| {prog} | {f} | ' + ' | '.join(cells) + f' | {", ".join(f"{x:.4f}" for x in tr)} | ' + ' | '.join(libs) + ' |')
            data['mlp'].setdefault(prog, {})['lib'] = L
            data['mlp'][prog + fs] = dict(arms={a: A.get(a, {}) for a in ('on', 'null', 'base', 'eager')},
                                          trace={a: A.get(a + ':trace', {}).get('kernel_ms') for a in ('on', 'null', 'base', 'eager')},
                                          lib=L)
    P('\n### MLP hotspots (nsys, one traced process per arm). Mapal: the program\'s timed region; library: one steady-state call\n')
    P('| program | face | arm | region / harness wall ms | host before 1st launch ms | kernels | kernel ms | GPU idle between kernels µs | wait after last kernel ms | top API in span (ms) |')
    P('|---|---|---|---:|---:|---:|---:|---|---:|---|')
    data['hot'] = {}
    for prog in ('pt_rm_m128_d256_h512', 'pt_rm_m128_d1024_h2048'):
        for f, fs in (('E', ''), ('C', '_c')):
            for arm in ('on', 'base', 'eager'):
                r = mapal_region(os.path.join(trace, f'nsys_mlp_{prog}{fs}.{arm}'))
                data['hot'][f'{prog}{fs}.{arm}'] = r
                if not r:
                    continue
                api = ', '.join(f'{k} {v:.4f}' for k, v in list(r['api_ms'].items())[:3])
                P(f'| {prog} | {f} | Mapal {arm} | {r["span_ms"]:.4f} | {r["first_launch_api_after_ms"]:.4f} | {len(r["kernels"])} | '
                  f'{r["kernel_ms"]:.4f} | {", ".join(f"{g:.1f}" for g in r["gaps_us"])} | {r["tail_after_last_kernel_ms"]:.4f} | {api} |')
        lb = library_legs(os.path.join(trace, f'nsys_rmlib_{prog}'))
        data['hot'][f'{prog}.lib'] = lb
        L = data['mlp'][prog]['lib']
        for leg, r in lb.items():
            api = ', '.join(f'{k} {v:.4f}' for k, v in list(r['api_ms'].items())[:3])
            P(f'| {prog} | lib | {leg} | {L[leg]["wall"]:.4f} | enqueue {r["span_ms"]:.4f} | {len(r["kernels"])} | {r["kernel_ms"]:.4f} | '
              f'{", ".join(f"{g:.1f}" for g in r["gaps_us"])} | - | {api} |')

    P('\n### MLP verdicts (ratio = Mapal / library; band = max(|on - null|, |on - identical-code base|, 1%))\n')
    P('| program | face | band | Mapal on (lazy, shipped) vs cuBLAS / Lt / CUTLASS stock / tuned (host wall) | Mapal eager-loaded vs same | Mapal kernels vs library kernels (nsys GPU ms) |')
    P('|---|---|---:|---|---|---|')
    data['mlp_verdicts'] = {}
    for prog in ('pt_rm_m128_d256_h512', 'pt_rm_m128_d1024_h2048'):
        L = data['mlp'][prog]['lib']
        lk = {l: r['kernel_ms'] for l, r in data['hot'][f'{prog}.lib'].items()}
        for f, fs in (('E', ''), ('C', '_c')):
            A = data['mlp'][prog + fs]['arms']
            on, nu, ba, ea = (A[a].get('iter_med') for a in ('on', 'null', 'base', 'eager'))
            band = max([abs(on - nu) / on] + ([abs(on - ba) / on] if f == 'E' else []) + [0.01])
            def v(r):
                return f'{"better" if r < 1 - band else "worse" if r > 1 + band else "on-par"} {r:.2f}'
            libs = ('cublas', 'lt', 'cutlass', 'cutlass_tuned')
            mk = data['hot'][f'{prog}{fs}.on']['kernel_ms']
            row = dict(band=band, on={l: on / L[l]['wall'] for l in libs}, eager={l: ea / L[l]['wall'] for l in libs},
                       kernels={l: mk / lk[l] for l in libs})
            data['mlp_verdicts'][prog + fs] = row
            P(f'| {prog} | {f} | {100 * band:.1f}% | ' + ' / '.join(v(row['on'][l]) for l in libs) + ' | '
              + ' / '.join(v(row['eager'][l]) for l in libs) + ' | ' + ' / '.join(v(row['kernels'][l]) for l in libs) + ' |')
    P('\n### MLP hardware usage (FLOPs = 2 GEMMs; operands L2-resident, so no DRAM roof applies)\n')
    P('| program | impl | GPU kernel ms | TFLOP/s on kernels | % FP32 peak | wall ms | kernel share of wall |')
    P('|---|---|---:|---:|---:|---:|---:|')
    for prog in ('pt_rm_m128_d256_h512', 'pt_rm_m128_d1024_h2048'):
        d_, h_ = (256, 512) if 'd256' in prog else (1024, 2048)
        fl = 2.0 * 128 * d_ * h_ * 2
        rows_ = [(f'Mapal {f}', data['hot'][f'{prog}{fs}.on']['kernel_ms'], data['mlp'][prog + fs]['arms']['on']['iter_med'])
                 for f, fs in (('E', ''), ('C', '_c'))]
        rows_ += [(LIBNAME.get(l, 'cuBLASLt fused'), r['kernel_ms'], data['mlp'][prog]['lib'][l]['wall'])
                  for l, r in data['hot'][f'{prog}.lib'].items()]
        for name, km, wall in rows_:
            P(f'| {prog} | {name} | {km:.4f} | {fl / (km * 1e-3) / 1e12:.2f} | {100 * fl / (km * 1e-3) / pk["flops"]:.1f} | {wall:.4f} | {100 * km / wall:.0f}% |')

    # lazy (first-use) kernel loading inside the timed region vs the kernel's SASS image size
    from sass_stats import functions
    P('\n### MLP: host time before each launch inside the region vs the kernel image size (nsys)\n')
    P('| program | face | arm | SASS bytes k1, k2 | host from region start to launch 1 µs | host between launch 1 and launch 2 µs | GPU idle k1→k2 µs |')
    P('|---|---|---|---|---:|---:|---:|')
    data['lazy'] = {}
    for prog in ('pt_rm_m128_d256_h512', 'pt_rm_m128_d1024_h2048'):
        for f, fs in (('E', ''), ('C', '_c')):
            for arm in ('on', 'base', 'eager'):
                stem = prog + fs + ('.base' if arm == 'base' else '')
                fn = functions(open(os.path.join(static, stem + '.sass'), errors='replace').read())
                base_ = os.path.join(trace, f'nsys_mlp_{prog}{fs}.{arm}')
                apis = sorted(api_trace(base_), key=lambda a: a['start'])
                tiles = {k['corr']: k for k in gpu_trace(base_) if k['kind'] == 'kernel' and k['name'].startswith('mapal_tile')}
                L_ = [a for a in apis if a['corr'] in tiles and 'Launch' in a['name']]
                sy = [a for a in apis if 'Synchronize' in a['name'] and a['start'] + a['dur'] <= L_[0]['start']][-1]
                lead = (L_[0]['start'] - sy['start'] - sy['dur']) / 1e3
                between = (L_[1]['start'] - L_[0]['start'] - L_[0]['dur']) / 1e3
                sizes = [len(fn[tiles[a['corr']]['name']]) * 16 for a in L_]
                k0, k1 = tiles[L_[0]['corr']], tiles[L_[1]['corr']]
                idle = (k1['start'] - k0['start'] - k0['dur']) / 1e3
                data['lazy'][f'{prog}{fs}.{arm}'] = dict(sizes=sizes, lead_us=lead, between_us=between, idle_us=idle)
                P(f'| {prog} | {f} | {arm} | {sizes[0]}, {sizes[1]} | {lead:.1f} | {between:.1f} | {idle:.1f} |')

    # ---------- 6. MLP kernels: geometry, resources, executed SASS
    P('\n### MLP kernels (nsys geometry/time; SASS from the observer-captured images)\n')
    P('| program | impl | kernel | grid | block | regs | smem B | occ | warps/SM | µs | loop insns | MAC | global loads | LDS | acc | k/iter | MAC/LDS |')
    P('|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|')
    sass_by = {}
    for cname, cap in LS.items():
        if cname.startswith(('cap_mlp_', 'cap_rmlib_')):
            for k in cap['kernels']:
                if k.get('sass'):
                    sass_by.setdefault((cname, k['name']), k['sass'])
    for prog in ('pt_rm_m128_d256_h512', 'pt_rm_m128_d1024_h2048'):
        groups = [(f'Mapal {f}', data['hot'].get(f'{prog}{fs}.on'), f'cap_mlp_{prog}{fs}') for f, fs in (('E', ''), ('C', '_c'))]
        groups += [(LIBNAME.get(l, 'cuBLASLt fused' if l == 'lt' else l), data['hot'][f'{prog}.lib'].get(l), f'cap_rmlib_{prog}')
                   for l in ('cublas', 'lt', 'cutlass', 'cutlass_tuned')]
        for impl, r, cname in groups:
            if not r:
                continue
            for k in r['kernels']:
                th = k['block'][0] * k['block'][1] * k['block'][2]
                b_, w_, occ, lim = occupancy(th, k['regs'], k['smem'])
                nb = k['grid'][0] * k['grid'][1] * k['grid'][2]
                rw = min(nb, b_ * sms) * ((th + 31) // 32) / sms
                st = sass_by.get((cname, k['name']), {})
                l = st.get('loop') or {}
                gl = ', '.join(f'{v}x{x.split(".")[0]}.{x.split(".")[1]}' for x, v in (l.get('widths') or {}).items() if x.startswith(('LDG.', 'LDGSTS')))
                P(f'| {prog} | {impl} | `{short(k["name"])}` | {"x".join(map(str, k["grid"]))} | {th} | {k["regs"]} | {k["smem"]} | '
                  f'{occ:.2f} | {rw:.1f} | {k["us"]:.1f} | {l.get("insns", "-")} | {l.get("mac", "-")} | {gl or "-"} | '
                  f'{(l.get("mix") or {}).get("LDS", "-")} | {l.get("accumulators", "-")} | {l.get("unroll_k", "-")} | {l.get("mac_per_lds", "-")} |')

    json.dump(data, open(os.path.join(static, 'report-data.json'), 'w'), indent=1, default=str)
    return '\n'.join(out)


if __name__ == '__main__':
    if len(sys.argv) == 2:  # the committed raw/ dir (tar xJf raw/static/mapal-sass.tar.xz -C raw/static first)
        r = sys.argv[1]
        print(main(os.path.join(r, 'results-h100-pcie.txt'), os.path.join(r, 'nsys'), os.path.join(r, 'static'), None))
    else:
        print(main(*sys.argv[1:5]))
