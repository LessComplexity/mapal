"""disasm-rtx-5090: tables from the box's raw evidence (results.txt, static/, nsys/, obs/).
usage: python3 analyze.py <evidence_dir>  -> markdown on stdout"""
import collections, csv, glob, json, os, re, statistics, sys

E = sys.argv[1]
R = open(f'{E}/results.txt').read().splitlines()
PEAK_BW = 1792e9          # B/s: 512-bit GDDR7 at 28 Gb/s (RTX 5090 spec)
PEAK_FP = 170 * 128 * 2 * 2.407e9  # FLOP/s: 170 SMs x 128 FP32 lanes x FMA x 2.407 GHz boost = 104.8 T
SMS, REGS_SM, THR_SM, BLK_SM = 170, 65536, 1536, 24
CELLS = ['16x2048x7168', '16x1024x4096', '16x18432x7168', '128x2048x7168', '4096x4096x4096']
LEGS = {'': 'Mapal E', '_c': 'Mapal C', '_r': 'Mapal R', '_rc': 'Mapal RC'}
LIBS = ['cublas', 'cutlass', 'cutlass_tuned']
facts = {}
for l in R:
    if ' facts ' in l and 'sms=' in l:
        facts = dict(kv.split('=') for kv in l.split(' facts ')[1].split())
        break
SMEM_SM = int(facts.get('smem_per_sm', 102400))


def kv(line):
    return dict(re.findall(r'(\w+)=(\[[^\]]*\]|\S+)', line))


def med(v):
    v = sorted(v)
    return v[len(v) // 2] if v else float('nan')


# ---------- per-cell timing (dis_bench, two processes)
cell = collections.defaultdict(lambda: collections.defaultdict(dict))   # cell -> leg -> pass -> samples
arm = collections.defaultdict(dict)                                     # cell -> leg -> arm line
lib = collections.defaultdict(dict)
vals = collections.defaultdict(dict)
for l in R:
    m = re.match(r'pass=(\d) cell=(\S+) (.*)', l)
    if not m: continue
    p, c, rest = int(m[1]), m[2], m[3]
    d = kv(rest)
    if rest.startswith('samples '):
        leg = d['leg']
        s = rest.split(' ', 2)[2] if len(rest.split(' ', 2)) > 2 else ''
        cell[c][leg][p] = [float(x) for x in s.split(',') if x]
    elif rest.startswith('arm '):
        arm[c][d['stem']] = d
    elif rest.startswith('libfacts '):
        lib[c][d['leg']] = d
    elif rest.startswith('values '):
        vals[c][d['leg']] = d


def leg_of(c, suffix):
    return f'mm_{c}{suffix}'


def stats(c, leg):
    s = cell[c][leg]
    return [med(s.get(1, [])), med(s.get(2, [])), med(s.get(1, []) + s.get(2, []))]


def null_var(c):
    """max |null/orig - 1| over the Mapal null copies (E, C) in both processes, and the
    process-to-process drift of every leg's median."""
    w = []
    for suf in ('', '_c'):
        a, b = cell[c][leg_of(c, suf)], cell[c][leg_of(c, suf) + '.null']
        for p in (1, 2):
            if a.get(p) and b.get(p): w.append(abs(med(b[p]) / med(a[p]) - 1))
    drift = []
    for leg, s in cell[c].items():
        if s.get(1) and s.get(2): drift.append(abs(med(s[2]) / med(s[1]) - 1))
    return max(w) if w else float('nan'), max(drift) if drift else float('nan')


TICK = 0.00205  # ms: the flushed-L2 event samples land on a 2.05 us grid on this box


def verdict(a, b, null, tick=TICK):
    """a/b with a band of max(null, one timer tick) around 1."""
    r = a / b
    band = max(null, tick / min(a, b))
    if r < 1 - band: return f'better {r:.2f}'
    if r > 1 + band: return f'worse {r:.2f}'
    return f'on-par {r:.2f}'


def nbytes(c, S=1, separate=True):
    M, N, K = map(int, c.split('x'))
    b = 4 * (M * K + K * N + M * N)
    if S > 1:  # partials: S written + read by a reduce kernel, or (S-1) read-modify-writes in-kernel
        b += 2 * 4 * (S if separate else S - 1) * M * N
    return b, 2 * M * N * K


out = []
P = out.append
P('## Same-box timing and verdicts (kernel ms, CUDA events, L2 flushed, medians of 30 per process; `a / b` = process 1 / process 2, `all` = 60 samples)\n')
P('| cell | impl | p1 / p2 med ms | all med ms | null (copy) / drift | vs cuBLAS | vs CUTLASS stock | vs CUTLASS tuned |')
P('|---|---|---|---|---|---|---|---|')
timing = {}
for c in CELLS:
    if not cell[c]: continue
    nv, drift = null_var(c)
    null = max(nv, 0.01)
    lt = {x: stats(c, x)[2] for x in LIBS}
    for suf, name in list(LEGS.items()) + [(x, x) for x in LIBS]:
        leg = leg_of(c, suf) if suf in LEGS else suf
        if leg not in cell[c]: continue
        a, b, al = stats(c, leg)
        timing[(c, name)] = al
        vs = [verdict(al, lt[x], null) if x != leg else '—' for x in LIBS]
        P(f'| {c} | {name} | {a:.4f} / {b:.4f} | {al:.4f} | {nv * 100:.1f}% / {drift * 100:.1f}% | ' + ' | '.join(vs) + ' |')
P('')
P('Verdict rule: ratio = this impl / that impl (all-sample medians); better/worse only outside the cell\'s band: '
  'max(Mapal identical-copy variation in both processes, 1%, one 2.05 us event tick / the faster time).\n')

# ---------- launch geometry: nsys gpu trace (every implementation) + arm lines
def nsys_rows(path):
    if not os.path.exists(path): return []
    rows = list(csv.DictReader(open(path)))
    return rows


def kind(n):
    if n.startswith('mapal_'): return 'Mapal'
    if 'cutlass::Kernel<' in n and 'GemmShape<(int)16, (int)64, (int)16>' in n: return 'CUTLASS tuned'
    if 'cutlass::Kernel<' in n: return 'CUTLASS stock'
    if 'bias_relu' in n or 'residual(' in n: return 'cuBLAS (harness elementwise)'
    return 'cuBLAS'


def obs_smem(case):
    """(grid, block) -> exact dynamic smem bytes from the observer's admitted launches."""
    m = collections.defaultdict(set)
    path = f'{E}/obs/{case}/launches.tsv'
    if os.path.exists(path):
        for l in open(path):
            f = l.rstrip('\n').split('\t')
            m[(f[1], f[2])].add(int(f[3]))
    return m


def occ(threads, regs, smem):
    if not threads: return 0, 0
    wpb = -(-threads // 32)
    rpw = -(-max(regs, 1) * 32 // 256) * 256
    b_regs = (REGS_SM // rpw) // wpb if rpw else BLK_SM
    b_smem = SMEM_SM // (smem + 1024) if smem else BLK_SM
    b = max(0, min(BLK_SM, THR_SM // (wpb * 32), b_regs, b_smem))
    return b, b * wpb / 48


P('## Kernels actually launched (nsys `cuda_gpu_trace`, one process per cell; regs/smem as nsys reports them)\n')
P('| cell | impl | kernel | grid | block | regs | smem B/block | blocks/SM (theory) | theor. occupancy | waves | resident warps/SM | nsys median us |')
P('|---|---|---|---|---|---|---|---|---|---|---|---|')
geo = {}
for c in CELLS:
    rows = nsys_rows(f'{E}/nsys/cell_{c}_cuda_gpu_trace.csv')
    osm = obs_smem(f'cell_{c}')
    by = collections.defaultdict(list)
    for r in rows:
        n = r.get('Name', '')
        if not r.get('GrdX'): continue
        by[(n, r['GrdX'], r['GrdY'], r['GrdZ'], r['BlkX'], r['BlkY'], r['BlkZ'], r.get('Reg/Trd', ''), r.get('StcSMem (MB)', '0'), r.get('DymSMem (MB)', '0'))].append(float(r['Duration (ns)']))
    for k, ds in sorted(by.items(), key=lambda kv: kv[0][0]):
        n, gx, gy, gz, bx, by_, bz, regs, ss, ds_ = k
        if n.startswith('gen_') or 'memset' in n.lower(): continue
        thr = int(bx) * int(by_) * int(bz)
        dyn = osm.get((f'{gx}x{gy}x{gz}', f'{bx}x{by_}x{bz}'), set())
        dn = float(ds_ or 0) * 1e6  # nsys prints MB with 3 decimals: take the exact observed value nearest it
        smem = int(round(float(ss or 0) * 1e6)) + (min(dyn, key=lambda v: abs(v - dn)) if dyn else int(round(dn)))
        blocks = int(gx) * int(gy) * int(gz)
        b, o = occ(thr, int(regs or 0), smem)
        waves = blocks / (b * SMS) if b else float('nan')
        impl = kind(n)
        short = n if len(n) < 70 else n[:67] + '...'
        geo[(c, n)] = dict(grid=(gx, gy, gz), thr=thr, regs=regs, smem=smem, b=b, occ=o, waves=waves, us=med(ds) / 1e3)
        rw = min(blocks, b * SMS) * (-(-thr // 32)) / SMS if b else 0
        P(f'| {c} | {impl} | `{short}` | {gx}x{gy}x{gz} | {thr} | {regs} | {smem} | {b} | {o * 100:.0f}% | {waves:.2f} | {rw:.1f} | {med(ds) / 1e3:.1f} |')
P('')
json.dump({f'{k[0]}|{k[1]}': v for k, v in timing.items()}, open(f'{E}/timing.json', 'w'), indent=1)

# ---------- hardware usage per cell x implementation
def lib_split(c, which):
    """(S, separate_reduce) of the cuBLAS GEMM launch (grid z) from the cell's nsys trace."""
    S, red = 1, False
    for (cc, n), g in geo.items():
        if cc != c: continue
        if kind(n) == 'cuBLAS' and 'splitKreduce' in n: red = True
        elif kind(n) == 'cuBLAS' and int(g['grid'][2]) > 1: S = int(g['grid'][2])
    return (S, red) if which == 'cublas' else (1, False)
P('## Hardware usage (bytes = A+B+C compulsory + split-K partials: 2·S·M·N·4 with a reduce kernel, 2·(S-1)·M·N·4 for in-kernel serial split; FLOP = 2MNK)\n')
P(f'Peaks: DRAM {PEAK_BW / 1e9:.0f} GB/s (512-bit GDDR7, 28 Gb/s), FP32 {PEAK_FP / 1e12:.1f} TFLOP/s (170 SM x 128 lanes x 2 x 2.407 GHz boost); ridge {PEAK_FP / PEAK_BW:.0f} FLOP/B. '
  'Class: memory if >= 50% of DRAM peak, compute if >= 50% of FP32 peak, launch if < 10 us, else latency (neither unit near its peak).\n')
P('| cell | impl | ms | S | MB moved | GB/s | % DRAM | GFLOP/s | % FP32 | FLOP/B | bound |')
P('|---|---|---|---|---|---|---|---|---|---|---|')
usage = {}
for c in CELLS:
    for suf, name in list(LEGS.items()) + [(x, x) for x in LIBS]:
        t = timing.get((c, name))
        if t is None or t != t: continue
        if suf in LEGS:
            a = arm[c].get(leg_of(c, suf), {})
            S, red = int(a.get('S', 1)), True
        else:
            S, red = lib_split(c, suf)
        b, f = nbytes(c, S, red)
        bw, fl = b / (t * 1e-3), f / (t * 1e-3)
        cls = 'memory' if bw >= 0.5 * PEAK_BW else 'compute' if fl >= 0.5 * PEAK_FP else 'launch' if t < 0.01 else 'latency'
        usage[(c, name)] = (bw / PEAK_BW, fl / PEAK_FP, cls)
        P(f'| {c} | {name} | {t:.4f} | {S} | {b / 1e6:.1f} | {bw / 1e9:.0f} | {100 * bw / PEAK_BW:.0f}% | {fl / 1e9:.0f} | {100 * fl / PEAK_FP:.1f}% | {f / b:.1f} | {cls} |')
P('')

# ---------- MLP: in-process steady state + cold, and the whole programs
mlp = collections.defaultdict(lambda: collections.defaultdict(dict))  # p -> leg -> pass -> dict(event, wall)
cold = collections.defaultdict(lambda: collections.defaultdict(list))
mlpvals = collections.defaultdict(dict)
mlparms = collections.defaultdict(dict)
for l in R:
    m = re.match(r'pass=(\d) mlp=(\S+) (.*)', l)
    if not m: continue
    p, prog, rest = int(m[1]), m[2], m[3]
    d = kv(rest)
    if rest.startswith('samples '):
        ev = [float(x) for x in d['event'].split(',')]
        wl = [float(x) for x in d['wall'].split(',')]
        mlp[prog][d['leg']][p] = dict(event=ev, wall=wl)
    elif rest.startswith('cold '):
        cold[prog][d['leg']].append(float(d['wall_ms']))
    elif rest.startswith('values '):
        mlpvals[prog][d['leg']] = d
    elif rest.startswith('arm ') or rest.startswith('lt ') or rest.startswith('libfacts '):
        mlparms[prog].setdefault(d.get('stem') or d.get('leg') or 'lt', []).append(d)
progs = collections.defaultdict(lambda: collections.defaultdict(list))  # p+v -> loading -> [iter ms]
pwall = collections.defaultdict(lambda: collections.defaultdict(list))
for l in R:
    m = re.match(r'prog prog=(\S+) loading=(\S+) rep=\d+ wall_us=(\d+) out=\[(.*)\]', l)
    if m:
        it = re.search(r'iter ms=([0-9.e-]+)', m[4])
        if it: progs[m[1]][m[2]].append(float(it[1]))
        pwall[m[1]][m[2]].append(int(m[3]) / 1e3)

MLPN = {'pt_rm_m128_d256_h512': 'MLP 128/256/512', 'pt_rm_m128_d1024_h2048': 'MLP 128/1024/2048'}
P('## Residual MLP (y = relu(x·W1 + b)·W2 + x), one process per pass, all implementations in it\n')
P('Steady state: 5 warmups, 30 iterations round-robin ABBA, L2 warm (as in the program, which generates its operands just before `t0`). '
  '`event` = CUDA events around the iteration\'s launches; `wall` = host clock around launches + stream sync. '
  '`cold` = the first execution of that implementation in the process (lazy kernel loading / library first call included).\n')
P('| MLP | impl | event med ms p1 / p2 | wall med ms (all) | cold first-call ms p1 / p2 | gate |')
P('|---|---|---|---|---|---|')
mlpt = {}
for prog, name in MLPN.items():
    for leg in sorted(mlp[prog], key=lambda x: (not x.startswith('pt_'), x)):
        s = mlp[prog][leg]
        ev = [med(s[p]['event']) if p in s else float('nan') for p in (1, 2)]
        wl = med(s.get(1, {}).get('wall', []) + s.get(2, {}).get('wall', []))
        evall = med(s.get(1, {}).get('event', []) + s.get(2, {}).get('event', []))
        mlpt[(prog, leg)] = (evall, wl)
        c = cold[prog][leg]
        v = mlpvals[prog].get(leg, {})
        gate = ('bit-exact' if v.get('bit_exact_vs_ordered') == 'yes' else f"maxrel {v.get('maxrel_vs_f64', '?')}") + (' FAIL' if 'GATE-FAILED' in str(v) else '')
        P(f'| {name} | {leg.replace(prog, "Mapal")} | {ev[0]:.4f} / {ev[1]:.4f} | {wl:.4f} | ' + ' / '.join(f'{x:.4f}' for x in c) + f' | {gate} |')
P('')
P('### MLP verdicts (steady-state event medians, all 60 samples; band = max(Mapal C vs its identical copy, process drift, 1%))\n')
P('| MLP | Mapal face | ms | vs cuBLAS | vs cuBLASLt fused | vs CUTLASS stock fused | vs CUTLASS tuned fused | vs Mapal 4e1102c (base) |')
P('|---|---|---|---|---|---|---|---|')
for prog, name in MLPN.items():
    t = {leg: mlpt[(prog, leg)][0] for (pp, leg) in mlpt if pp == prog}
    a, b = t.get(prog + '_c'), t.get(prog + '_c.null')
    nulls = [abs(b / a - 1)] if a and b else []
    for leg, sd in mlp[prog].items():
        if 1 in sd and 2 in sd: nulls.append(abs(med(sd[2]['event']) / med(sd[1]['event']) - 1))
    band = max(nulls + [0.01])
    for f, fl in (('', 'E'), ('_c', 'C'), ('_r', 'R'), ('_rc', 'RC')):
        x = t.get(prog + f)
        if x is None: continue
        base = t.get(prog + ('_c' if f in ('_c', '_rc') else '') + '.base')
        P(f'| {name} | {fl} | {x:.4f} | ' + ' | '.join(verdict(x, t[k], band, 0) for k in ('cublas', 'lt', 'cutlass', 'cutlass_tuned')) +
          f' | {verdict(x, base, band, 0)} |')
    P(f'| {name} | band | {band * 100:.1f}% | | | | | |')
P('')
P('### MLP hardware usage (steady-state event ms; FLOP = 2·M·h·d·2; bytes = x, W1, W2, bias, y + hh written and re-read: L2-resident, < 96 MB)\n')
P('| MLP | impl | ms | MB touched | GB/s | GFLOP/s | % FP32 | bound |')
P('|---|---|---|---|---|---|---|---|')
for prog, name in MLPN.items():
    M, d, h = map(int, re.findall(r'\d+', prog)[1:]) if False else (128, *map(int, re.search(r'd(\d+)_h(\d+)', prog).groups()))
    fl = 2 * M * h * d * 2
    by = 4 * (M * d + d * h + h * d + h + M * d + 2 * M * h)
    for leg in [prog, prog + '_c', prog + '_r', prog + '_rc', prog + '.base', prog + '_c.base', 'cublas', 'lt', 'cutlass', 'cutlass_tuned']:
        t = mlpt.get((prog, leg), (None,))[0]
        if t is None: continue
        P(f"| {name} | {leg.replace(prog, 'Mapal')} | {t:.4f} | {by / 1e6:.2f} | {by / t / 1e6:.0f} | {fl / t / 1e6:.0f} | {100 * fl / t / 1e-3 / PEAK_FP:.1f}% | {'launch' if t < 0.01 else 'latency'} |")
P('')
P('### Whole Mapal programs: `iter ms` (the program\'s own `time` pair: two launches + sync), 30 processes per arm, ABBA\n')
P('| program | arm | LAZY (default) iter ms med | EAGER iter ms med | LAZY/EAGER | process wall ms med (LAZY) |')
P('|---|---|---|---|---|---|')
for prog in MLPN:
    for v in ['', '_c', '.base', '_c.base']:
        k = prog + v
        if k not in progs: continue
        la, ea = med(progs[k]['LAZY']), med(progs[k]['EAGER'])
        P(f'| {MLPN[prog]} | {v or "on"} | {la:.4f} (n={len(progs[k]["LAZY"])}) | {ea:.4f} | {la / ea:.2f} | {med(pwall[k]["LAZY"]):.1f} |')
P('')

print('\n'.join(out))

# ---------- whole-program hotspots (nsys -t cuda,osrt, one LAZY process per arm)
def prog_hot(stem):
    api = nsys_rows(f'{E}/nsys/prog_{stem}_cuda_api_trace.csv')
    gpu = [r for r in nsys_rows(f'{E}/nsys/prog_{stem}_cuda_gpu_trace.csv') if r.get('GrdX')]
    if not api: return None
    st = lambda r: int(r['Start (ns)']); du = lambda r: int(r['Duration (ns)']); en = lambda r: st(r) + du(r)
    syncs = [r for r in api if r['Name'] == 'cuCtxSynchronize']
    launches = [r for r in api if r['Name'].startswith('cuLaunchKernel')]
    w0, w1 = en(syncs[0]), en(syncs[1])
    l1, l2 = [r for r in launches if w0 < st(r) < w1][:2]
    tiles = [r for r in gpu if r['Name'].startswith('mapal_tile')]
    k = [int(r['Duration (ns)']) for r in tiles]
    out_ = open(f'{E}/nsys/prog_{stem}.out').read()
    it = re.search(r'iter ms=([0-9.]+)', out_)
    sum_ = lambda name: sum(du(r) for r in api if r['Name'].startswith(name)) / 1e3
    return dict(window=(w1 - w0) / 1e3, pre1=(st(l1) - w0) / 1e3, launch1=du(l1) / 1e3, pre2=(st(l2) - en(l1)) / 1e3,
                launch2=du(l2) / 1e3, sync=du(syncs[1]) / 1e3, k1=k[0] / 1e3, k2=k[1] / 1e3,
                idle=(w1 - w0 - sum(k)) / 1e3, iter=float(it[1]) * 1e3 if it else float('nan'),
                module=sum_('cuModuleLoad'), alloc=sum_('cuMemAlloc'), d2h=sum_('cuMemcpyDtoH'), free=sum_('cuMemFree'))


P('## Whole-MLP-program hotspots (nsys `-t cuda,osrt`, one process per arm, CUDA lazy loading = default)\n')
P('Window = end of the sync after input generation to the end of the sync after the two tile launches (the program\'s `t0`..`t1`). '
  '`host before launch N` = the host time between the previous API call and `cuLaunchKernel` N: the program\'s `time()` plus '
  '`mapal_gpu.c`\'s `cuModuleGetFunction`, which under lazy loading loads that kernel at its first call (nsys does not list it). '
  'All us.\n')
P('| program | arm | window | host before launch 1 | launch 1 | host before launch 2 | launch 2 | sync | kernel 1 GPU | kernel 2 GPU | GPU idle in window | kernels / window | program iter ms (under nsys) | module load ms | alloc ms | D2H us | free us |')
P('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for prog in MLPN:
    for v in ['', '.base', '_c', '_c.base']:
        h = prog_hot(prog + v)
        if not h: continue
        P(f"| {MLPN[prog]} | {v or 'on'} | {h['window']:.1f} | {h['pre1']:.1f} | {h['launch1']:.1f} | {h['pre2']:.1f} | {h['launch2']:.1f} | {h['sync']:.1f} | "
          f"{h['k1']:.1f} | {h['k2']:.1f} | {h['idle']:.1f} | {100 * (h['k1'] + h['k2']) / h['window']:.0f}% | {h['iter'] / 1e3:.4f} | {h['module'] / 1e3:.2f} | {h['alloc'] / 1e3:.2f} | {h['d2h']:.1f} | {h['free']:.1f} |")
P('')
print('\n'.join(out[out.index('## Whole-MLP-program hotspots (nsys `-t cuda,osrt`, one process per arm, CUDA lazy loading = default)\n'):]))

P('## MLP: kernels per iteration and their GPU time (nsys, MLP harness, median over every launch in the process)\n')
P('| MLP | impl | kernel | grid | block | regs | smem B | resident warps/SM | median us |')
P('|---|---|---|---|---|---|---|---|---|')
mlp_kern = {}
for prog in MLPN:
    rows = nsys_rows(f'{E}/nsys/mlp_{prog}_cuda_gpu_trace.csv')
    osm = obs_smem(f'mlp_{prog}')
    by = collections.defaultdict(list)
    order = []
    for r in rows:
        if not r.get('GrdX'): continue
        k = (r['Name'], r['GrdX'], r['GrdY'], r['GrdZ'], r['BlkX'], r['BlkY'], r['BlkZ'], r.get('Reg/Trd', ''), r.get('StcSMem (MB)', '0'), r.get('DymSMem (MB)', '0'))
        if k not in by: order.append(k)
        by[k].append(float(r['Duration (ns)']))
    for k in order:
        n, gx, gy, gz, bx, by_, bz, regs, ss, ds_ = k
        thr = int(bx) * int(by_) * int(bz)
        dyn = osm.get((f'{gx}x{gy}x{gz}', f'{bx}x{by_}x{bz}'), set()); dn = float(ds_ or 0) * 1e6
        smem = int(round(float(ss or 0) * 1e6)) + (min(dyn, key=lambda v: abs(v - dn)) if dyn else int(round(dn)))
        blocks = int(gx) * int(gy) * int(gz)
        b, _ = occ(thr, int(regs or 0), smem)
        rw = min(blocks, b * SMS) * (-(-thr // 32)) / SMS if b else 0
        short = n if len(n) < 70 else n[:67] + '...'
        P(f'| {MLPN[prog]} | {kind(n)} | `{short}` | {gx}x{gy}x{gz} | {thr} | {regs} | {smem} | {rw:.2f} | {med(by[k]) / 1e3:.1f} |')
P('')
print('\n'.join(out[out.index('## MLP: kernels per iteration and their GPU time (nsys, MLP harness, median over every launch in the process)\n'):]))

# ---------- executed SASS (observer-admitted images, nvdisasm of the exact launched entry)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sass_static as SS


def sass_facts(case):
    res = {}
    for f in sorted(glob.glob(f'{E}/obs/{case}/sass/*.sass')):
        for name, ins in SS.split_functions(open(f, errors='replace').read()):
            c, loops, h = SS.facts(ins, name)
            res[name] = (c, h)
    return res


def lib_name(n):
    if n.startswith('mapal_'): return n
    m = re.search(r'cutlass_80_simt_sgemm_\w+?_nn', n)
    if m: return 'cuBLAS ' + m[0][:-3]
    if 'splitKreduce' in n: return 'cuBLAS splitKreduce'
    m = re.search(r'GemmShapeILi(\d+)ELi(\d+)ELi(\d+)E', n)
    if m: return f'CUTLASS {m[1]}x{m[2]}x{m[3]}'
    return n[:40]


P('## Executed SASS (observer-admitted images; hot loop = the backward-branch loop with the most FFMA/FADD)\n')
P('Columns per thread. `acc` = accumulator chains started per trip (FFMA/FADD whose accumulator input was not produced by an FFMA/FADD earlier in the trip); '
  '`k/trip` = loop MACs / acc; `MAC/gld` = loop FFMA(or FADD) per global-load instruction (LDG + LDGSTS); `B/trip` = global bytes requested per thread per trip.\n')
P('| cell | impl | entry | instr | FFMA | FMUL | FADD | LDG (.128) | LDGSTS | LDS (.128) | STS | BAR | loop MAC | loop gld (B/trip) | loop LDS | acc | k/trip | MAC/gld | regs | local B |')
P('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
regs_of = {}
for c in CELLS:
    for st, a in arm[c].items():
        regs_of[a['kernel']] = (a.get('regs'), a.get('local_bytes'))
    for leg, d in lib[c].items():
        regs_of[leg] = (d.get('regs'), d.get('local_bytes'))
for c in CELLS:
    sf = sass_facts(f'cell_{c}')
    names = {}
    for suf, nm in LEGS.items():
        a = arm[c].get(leg_of(c, suf))
        if a: names[a['kernel']] = nm
        if a and a.get('reduce'): names[a['reduce']] = nm + ' reduce'
    for n, (cnt, h) in sf.items():
        impl = names.get(n)
        if impl is None:
            ln = lib_name(n)
            impl = 'cuBLAS' if ln.startswith('cuBLAS') else ('CUTLASS tuned' if '16x64x16' in ln else 'CUTLASS stock')
        h = h or dict(span=0, ffma=0, fadd=0, fmul=0, glb=0, glb_bytes=0, lds=0, bar=0, accs=0)
        mac = h['ffma'] + h['fadd']
        r = regs_of.get(n) or (regs_of.get('cutlass_tuned') if impl == 'CUTLASS tuned' else regs_of.get('cutlass') if impl == 'CUTLASS stock' else (None, None))
        if impl.startswith('cuBLAS'):
            key = re.search(r'cutlass_80_simt_sgemm_\w+?_nn|splitKreduce', n)[0]
            g = [v for (cc, nn), v in geo.items() if cc == c and key in nn]
            r = (g[0]['regs'], 'n/a') if g else (None, None)
        if impl.endswith(' reduce'):
            a = arm[c].get(leg_of(c, {'Mapal R reduce': '_r', 'Mapal RC reduce': '_rc'}[impl]), {})
            r = (a.get('rregs'), 'n/a')
        P(f"| {c} | {impl} | `{lib_name(n)}` | {cnt['total']} | {cnt['FFMA']} | {cnt['FMUL']} | {cnt['FADD']} | {cnt['LDG']} ({cnt['LDG.128']}) | {cnt['LDGSTS']} | {cnt['LDS']} ({cnt['LDS.128']}) | {cnt['STS']} | {cnt['BAR']} | "
          f"{mac} | {h['glb']} ({h['glb_bytes']}) | {h['lds']} | {h['accs']} | {(mac / h['accs']) if h['accs'] else 0:.0f} | {(mac / h['glb']) if h['glb'] else 0:.1f} | {r[0]} | {r[1]} |")
P('')
print('\n'.join(out[out.index('## Executed SASS (observer-admitted images; hot loop = the backward-branch loop with the most FFMA/FADD)\n'):]))
