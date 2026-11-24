#!/usr/bin/env python3
"""S64 evidence pass: per kernel and arch, what the emitter intended (`; mapal.plan`, `; mapal.host`,
the measured rung table) against what LLVM / NVPTX / ptxas realized (optimization remarks, PTX, SASS,
`ptxas -v`, occupancy, the box's JIT image). A mismatch is a finding, never a pass (CLAUDE.md).

usage:
  evidence.py static <dir> <sm>              ptxas -v + cuobjdump -sass per PTX (cudatool.sh), llvm-objdump
                                             per host object, into <dir>/evidence/
  evidence.py facts <dir> <sm>               -> <dir>/evidence/facts.json (one entry per stem x kernel)
  evidence.py check <dir> [results.txt]      -> TSV on stdout: arch stem kernel rung check expected realized status
  evidence.py diff <dir> <stemA> <stemB>     every numeric fact that differs, per kernel
  evidence.py jit <capture dir> <sass>...    the box's JIT image vs ptxas, opcode sequence per mapal kernel
  evidence.py launch <stem>                  per-site-choice: <stem>.launch (the first launch; a table's candidate 0),
                                             <stem>.sites (one line per candidate) and <stem>.pick<i>.launch
                                             (+ .ptx link) per candidate of the first table, off <stem>.host.ll
  evidence.py covered <arm> <shipped> <rungs> exit 0 when the shipped stem has site tables, the arm's rungs are all
                                             CHOICE_AXES rungs and every kernel of the arm's device IR is one of the
                                             shipped module's kernels (body equal up to its name)
<dir> is one profile's bundle directory (`<bundle>/<profile>`); its name is the arch's profile.
Static counts only: no claim about dynamic instructions, stalls or traffic follows from them.
"""
import glob
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from occ import occupancy  # noqa: E402
from ptx_static import CLS, entries  # noqa: E402
from sass_stats import TENSOR_MACS, accumulators, functions, split, stats, width  # noqa: E402

# Per part: the `facts` line its box prints (cudaDeviceProp); `occ.py`'s limits.
PARTS = {
    'rtx-4090': dict(sm='sm_89', sms=128, threads=1536, blocks=24, smem=102400),
    'rtx-5090': dict(sm='sm_120', sms=170, threads=1536, blocks=24, smem=102400),
    'h100-pcie': dict(sm='sm_90a', sms=114, threads=2048, blocks=32, smem=233472),
    'rtx-3090': dict(sm='sm_86', sms=82, threads=1536, blocks=16, smem=102400),
    'rtx-5080': dict(sm='sm_120', sms=84, threads=1536, blocks=24, smem=102400),
}
# The rungs a kernel's text reads (func/gpu.rs KERNEL_RUNGS); the rest are host rungs.
KERNEL_RUNGS = ('arch-rungs interleave unroll min-blocks warp prefetch mma async wide ldmatrix warp-fill idx32 affine '
                'pad wide-fma wgmma-hopper raster l2-gate pdl row-fit wave-gate grid-share panel-regs fill-rows '
                'cell-fill asyncstage splitk splitk-atomic streamk row-split fill-wave fit-first tf32-radix kmajor-a full-tile '
                'async-depth bank-plan streamk-fill k-fill vec-cells stream-fly bulk-stage wgmma-ws ws-regs tma-digits kmajor-b').split()
GPU_SYMS = ('alloc arena copy2d d2h_on d2h event free gen h2d h2d_async h2d_on init launch_l2 launchv launch launch_coop '
            'pin site sync tile unpin zero').split()
PROFILE_RS = os.path.join(HERE, '..', '..', 'crates', 'backends', 'llvm', 'src', 'profile.rs')


# ---- the rung rules' table (hardware-properties: `emit --rungs --target=<part>`, written by the bundle) -------
def rows_off(d):
    """{(wps class 0/1/2, face): set of rungs on} from `<part dir>/rungs.tsv`; {} without it."""
    path = os.path.join(d, 'rungs.tsv')
    if not os.path.exists(path):
        return {}
    out = {}
    for line in open(path):
        wps, face, on = (line.rstrip('\n').split('\t') + [''])[:3]
        out[(int(wps), face)] = set(r for r in on.split(',') if r)
    return out


def face_of(arith):
    return {'SeparateMulAdd': 'Exact', 'OrderedFma': 'Contract'}.get(arith, 'Tf32')


def table_rungs(rows, prof, wps, face, arm):
    """The rungs `rungs_at` gives: the rules' rungs at (fill, face) — every rung under `arch-rungs` off —
    then the arm."""
    on, off = arm
    every = set().union(*rows.values()) | set(KERNEL_RUNGS)
    r = every if 'arch-rungs' in off else set(rows.get((min(wps, 2), face), set()))
    r = (r | on) - off
    # per-site-choice D1: the kmajor-a rows are bank-plan's OFF arm (func/gpu.rs `rungs_at`).
    if 'bank-plan' in r and 'kmajor-a' not in off:
        r.add('kmajor-a')
    return sorted(r)


def arm_of(stem):
    """`.off-a+b` / `.on-a`: the forced rungs of a bundle arm."""
    on, off = set(), set()
    for part in stem.split('.')[1:]:
        # S65: `cpu-off-<rung>`: the CPU-only build's arm.
        part = part[4:] if part.startswith('cpu-') else part
        if part.startswith('off-'):
            off |= set(part[4:].split('+'))
        elif part.startswith('on-'):
            on |= set(part[3:].split('+'))
    return on, off


# ---- PTX -------------------------------------------------------------------------------------------
I64 = re.compile(r'^(add|sub|mul\.lo|mul\.wide|mad\.lo|mad\.wide|shl|shr)\.[usb]?(64|32)\b|^cvt\.[us]64\.[us]32')
EXTRA = [('atom_add', r'^atom\.(?:\w+\.)*?add\.f'), ('red_add', r'^red\.(?:\w+\.)*?add\.f'), ('mma', r'^mma\.sync'),  # (newer llc: .relaxed.gpu.global)
         ('ldmatrix', r'^ldmatrix'), ('wgmma', r'^wgmma\.mma_async'), ('wait_n', r'^cp\.async\.wait_group'),
         ('div_rem', r'^(div|rem)\.[us]\d+'), ('call', r'^call'),  # integer only (a float `div.rn` is the program's)
         # kernel-merging `calls`: a grid barrier's release arrival; a non-coherent (read-only path) load.
         ('gsync_arrive', r'^atom\.add\.release\.gpu'), ('ld_nc', r'^ld\.global\.nc'),
         # sweep-attention-2 `kmajor-b`: a vector load from global memory (2 or 4 elements).
         ('ld_global_vec', r'^ld\.global(\.nc)?\.v[24]\.'),
         # exp (ADR-0042) `exp-hw`: the hardware form's `ex2.approx.ftz.f32`.
         ('ex2', r'^ex2\.approx'),
         # math (ADR-0046) `math-hw`: the admitted approximate units.
         ('math_approx', r'^(?:sqrt|sin|cos)\.approx'),
         # sweep-runtime-2 `streamk-epi`: a tile's ticket (an acq_rel integer add on the tile's counter).
         ('ticket', r'^atom\.acq_rel\.(?:\w+\.)*?add\.u32')]


PTX_KEYS = [k.replace('.', '_') for k, _ in CLS] + [k for k, _ in EXTRA] + ['i64_arith']


def ptx_counts(lines):
    c = Counter({k: 0 for k in PTX_KEYS})
    for ins in lines:
        ins = re.sub(r'^@!?%p\d+\s+', '', ins)
        for k, rx in CLS:
            if re.match(rx, ins):
                c[k.replace('.', '_')] += 1
                break
        for k, rx in EXTRA:
            if re.match(rx, ins):
                c[k] += 1
        m = I64.match(ins)
        if m and m.group(2) == '64' or ins.startswith(('mul.wide', 'mad.wide', 'cvt.u64', 'cvt.s64')):
            c['i64_arith'] += 1
    return c


def ptx_facts(text):
    """{entry: {...counts, maxntid, minnctapersm, hot: {...}}}"""
    out = {}
    for name, params, dirs, body in entries(text):
        lines = [x.strip() for x in body.splitlines()]
        f = dict(ptx_counts(lines))
        f['params'] = params.count('.param')
        f['loops'] = 0  # backward branches
        for d in ('maxntid', 'minnctapersm'):
            m = re.search(r'\.' + d + r'\s+(\d+)', dirs)
            f[d] = int(m.group(1)) if m else 0
        waits = [int(m.group(1)) for x in lines if (m := re.match(r'cp\.async\.wait_group\s+(\d+)', x))]
        f['wait_n_max'] = max(waits, default=-1)
        labels = {m[1]: i for i, x in enumerate(lines) if (m := re.match(r'^(\$L__\w+):', x))}
        best, spans = None, []
        for i, x in enumerate(lines):
            m = re.search(r'bra(?:\.uni)?\s+(\$L__\w+);', x)
            if m and m[1] in labels and labels[m[1]] < i:
                f['loops'] += 1
                seg = lines[labels[m[1]]:i + 1]
                c = ptx_counts(seg)
                key = (c['fma_f32'] + c['mul_f32'] + MMA_WEIGHT * (c['mma'] + c['wgmma']), -len(seg))
                spans.append((labels[m[1]], i, key, seg, c))
                if best is None or key > best[0]:
                    best = (key, seg, c, labels[m[1]], i)
        if best is not None:
            best = innermost(best, spans)[:3]
        if best is None:  # no loop (a fully unrolled fold): the whole body is the hot region
            seg = lines
            best = (None, seg, ptx_counts(seg))
        if best:
            seg, c = best[1], best[2]
            bases = {m.group(1) for x in seg if re.match(r'^(@!?%p\d+\s+)?(ld\.global|cp\.async\.c)', x)
                     for m in [re.search(r'\[(%rd\d+)', x)] if m}
            f['hot'] = dict(c, lines=len(seg), distinct_load_bases=len(bases),
                            fma=c['fma_f32'], mul=c['mul_f32'], add=c['add_f32'])
        out[name] = f
    return out


# ---- ptxas -v, SASS --------------------------------------------------------------------------------
def ptxas_facts(text):
    """S62 mapal_static.py:ptxas — {entry: regs spill_st spill_ld stack static_smem}"""
    out, cur = {}, None
    for line in text.splitlines():
        m = re.search(r"Compiling entry function '(\S+)'", line)
        if m:
            cur = m.group(1)
            out[cur] = {}
        m = re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads', line)
        if m and cur:
            out[cur].update(stack=int(m.group(1)), spill_st=int(m.group(2)), spill_ld=int(m.group(3)))
        m = re.search(r'Used (\d+) registers', line)
        if m and cur:
            out[cur]['regs'] = int(m.group(1))
            s = re.search(r'(\d+) bytes smem', line)
            out[cur]['static_smem'] = int(s.group(1)) if s else 0
    return out


MAC = ('FFMA', 'FMUL', 'HMMA', 'HGMMA')
# tensor-cores-2: a matrix instruction's work against an FFMA's — an m16n8k8 HMMA is 1024 multiply-adds a warp, an
# FFMA 32. Counted one for one, a split tile's cold path (its warp's tile recomputed by FFMA, unrolled: 128 FFMA a
# trip) out-weighed the 96-HMMA k loop, and the A100's `split-tile` intents read the cold loop's 40 LDG as the hot
# loop's (5fcb401b finding 2): the k loop held 8 LDGSTS and no LDG. One weight with the SASS key (bugs-properties'
# `TENSOR_MACS`, sass_stats.py).
MMA_WEIGHT = TENSOR_MACS


def innermost(best, spans):
    """The hot loop: from the loop with the most multiply work (`best`: key, body, …, first, last), descend into a
    loop nested inside it that holds more than half that work. It runs at least once a trip of the outer one, so it
    holds most of the executed products; the outer loop's other products are its own non-loop code — a stream-K
    piece loop's epilogue store around its k loop (integration 3: the `streamk-epi` ticket kernels, whose member
    multiplies made the piece loop out-count the k loop)."""
    while True:
        inner = [s for s in spans if best[-2] <= s[0] and s[1] <= best[-1] and (s[0], s[1]) != (best[-2], best[-1])
                 and 2 * s[2][0] > best[0][0]]
        if not inner:
            return best
        s = max(inner, key=lambda s: s[2])
        best = (s[2], *s[3:], s[0], s[1])


def sass_facts(body):
    """Whole-function opcode families + the hot loop (the backward-branch body with the most MAC/MMA)."""
    c, w = stats(body)
    f = {k: 0 for k in ('ffma', 'fmul', 'fadd', 'hmma', 'hgmma', 'ldg', 'ldgsts', 'lds', 'ldsm', 'sts', 'stg', 'bar',
                        'red', 'atom', 'atomg', 'redg', 'ldl', 'stl', 'vote')}
    f.update({k.lower(): v for k, v in c.items()})
    f.update({k.lower().replace('.', '_'): v for k, v in w.items()})
    f['insns'] = len(body)
    # exp (ADR-0042) `exp-hw`: the SFU's `ex2` (the hardware form's MUFU.EX2).
    f['mufu_ex2'] = sum(1 for _, ins in body if split(ins)[1].startswith('MUFU.EX2'))
    # math (ADR-0046) `math-hw`: the SFU's square root and cosine (sine is never admitted on the measured parts).
    f['mufu_math'] = sum(1 for _, ins in body if split(ins)[1].startswith(('MUFU.SQRT', 'MUFU.COS', 'MUFU.SIN')))
    addr = [a for a, _ in body]
    best, spans = None, []
    for i, (a, ins) in enumerate(body):
        _, op, rest = split(ins)
        m = re.search(r'0x([0-9a-f]+)', rest) if op.split('.')[0] == 'BRA' else None
        if not m or int(m.group(1), 16) >= a or int(m.group(1), 16) not in addr:
            continue
        j = addr.index(int(m.group(1), 16))
        loop = body[j:i + 1]
        ops = Counter(split(x)[1].split('.')[0] for _, x in loop)
        # bugs-properties: a matrix instruction weighs its multiply-accumulates a thread (`TENSOR_MACS`), so a
        # tensor kernel's hot loop is its HMMA loop, not the FFMA guard path beside it (tf32x3's cold loop, 128
        # FFMA + 32 LDG, out-counted the 96-HMMA k loop: cross-arch 5fcb401b's "40 LDG, 0 LDGSTS").
        key = (ops['FFMA'] + ops['FMUL'] + TENSOR_MACS * (ops['HMMA'] + ops['HGMMA']), -len(loop))
        spans.append((j, i, key, loop))
        if best is None or key > best[0]:
            best = (key, loop, j, i)
    if best is not None:
        best = innermost(best, spans)[:2]
    if best is None or best[0][0] == 0:  # no MAC loop (a fully unrolled fold): the whole body
        best = ((0, 0), body)
    if best:
        loop = best[1]
        lc, lw = stats(loop)
        ops = [split(x)[1] for _, x in loop]
        base = [o.split('.')[0] for o in ops]
        mac = lc['FFMA'] + min(lc['FMUL'], lc['FADD'])
        acc = accumulators(loop)
        ismac = [b in ('FFMA', 'FMUL', 'HMMA', 'HGMMA') for b in base]
        fm = ismac.index(True) if any(ismac) else 0
        lm = len(base) - 1 - ismac[::-1].index(True) if any(ismac) else len(base) - 1
        isg = [b in ('LDG', 'LDGSTS') for b in base]
        # MAC between each register-staged LDG and its first reader (cyclic): the math covering it.
        cover = []
        for i, (_, ins) in enumerate(loop):
            _, op, rest = split(ins)
            if op.split('.')[0] != 'LDG':
                continue
            rs = re.findall(r'\bR\d+\b', rest)
            if not rs:
                continue
            dst = {f'R{int(rs[0][1:]) + j}' for j in range(max(1, width(op) // 32))}
            n = 0
            for d in range(1, len(loop) + 1):
                _, op2, rest2 = split(loop[(i + d) % len(loop)][1])
                src = re.findall(r'\bR\d+\b', rest2)
                if dst & set(src if op2.startswith(('STS', 'STG', 'ST')) else src[1:]):
                    break
                n += op2.split('.')[0] in ('FFMA', 'FMUL')
            cover.append(n)
        f['hot'] = dict(
            insns=len(loop), ffma=lc['FFMA'], fmul=lc['FMUL'], fadd=lc['FADD'], hmma=lc['HMMA'], hgmma=lc['HGMMA'], f2fp=sum(b == 'F2FP' for b in base),
            ldg=lc['LDG'], ldgsts=lc['LDGSTS'], lds=lc['LDS'], ldsm=lc['LDSM'], sts=lc['STS'], bar=lc['BAR'],
            bra=lc['BRA'], ldg128=lw.get('LDG.128', 0), lds128=lw.get('LDS.128', 0),
            # tile-efficiency: copy widths and the LSU mix (T2 / T4)
            ldgsts128=sum(o.startswith('LDGSTS') and o.endswith('.128') for o in ops),
            sts128=sum(o.startswith('STS') and o.endswith('.128') for o in ops),
            lsu=lc['LDS'] + lc['STS'] + lc['LDG'] + lc['LDGSTS'],
            ffma_per_lsu=round(mac / (lc['LDS'] + lc['STS'] + lc['LDG'] + lc['LDGSTS']), 2)
            if lc['LDS'] + lc['STS'] + lc['LDG'] + lc['LDGSTS'] else 0,
            imad_wide=sum(o.startswith('IMAD.WIDE') for o in ops), mac=mac, acc=acc,
            # half-precision (ADR-0043): 2-byte global loads (a narrow operand read in its format)
            ldg16=sum(o.startswith('LDG') and ('.U16' in o or '.S16' in o) for o in ops),
            ldg64=sum(o.startswith('LDG') and '64' in o.split('.') for o in ops),
            # fp8-leftovers: 1-byte global loads (an fp8 operand read one element at a time)
            ldg8=sum(o.startswith('LDG') and ('.U8' in o or '.S8' in o) for o in ops),
            ksteps=round(mac / acc, 2) if acc else 0,
            span=round(len(loop) / (mac / acc), 2) if acc and mac else 0,
            ffma_per_lds=round(mac / lc['LDS'], 2) if lc['LDS'] else 0,
            ffma_per_ldg=round(mac / (lc['LDG'] + lc['LDGSTS']), 2) if lc['LDG'] + lc['LDGSTS'] else 0,
            ldg_before_math=sum(isg[:fm]), ldg_after_math=sum(isg[lm + 1:]),
            ldg_cover_min=min(cover) if cover else -1)
    return f


# ---- LLVM optimization remarks ---------------------------------------------------------------------
def remarks(path):
    """[(kind, pass, name, function, {arg: value})] from a remark YAML (stdlib regex parse; .gz too)."""
    if os.path.exists(path + '.gz'):
        path += '.gz'
    if not os.path.exists(path):
        return None
    text = (gzip.open if path.endswith('.gz') else open)(path, 'rt', errors='replace').read()
    out = []
    for doc in re.split(r'^--- !', text, flags=re.M)[1:]:
        kind = doc.split('\n', 1)[0].strip()
        g = lambda k: (re.search(r'^' + k + r':\s+(\S+)', doc, re.M) or [None, ''])[1]
        args = dict(re.findall(r'^\s+- (\w+):\s+\'?([^\'\n]*)\'?', doc, re.M))
        out.append((kind, g('Pass'), g('Name'), g('Function'), args))
    return out


def remark_facts(recs, fn=None):
    if recs is None:
        return {}
    rs = [r for r in recs if fn is None or r[3] == fn]
    f = Counter(f'{k.lower()}_{p}_{n}'.replace('-', '_') for k, p, n, _, _ in rs)
    f = dict(f)
    f['full_unroll'] = max((int(a.get('UnrollCount', 0)) for k, p, n, _, a in rs if n == 'FullyUnrolled'), default=0)
    # Every FullyUnrolled count: a kernel may unroll more loops than its bk-step one (a stream-K ticket kernel's
    # epilogue store over its rows, integration 3), so the bk-step check asks for its own count among them.
    f['full_unrolls'] = sorted({int(a.get('UnrollCount', 0)) for k, p, n, _, a in rs if n == 'FullyUnrolled'})
    f['partial_unroll'] = max((int(a.get('UnrollCount', 0)) for k, p, n, _, a in rs
                               if p == 'loop-unroll' and k == 'Passed' and n != 'FullyUnrolled'), default=0)
    f['vectorized'] = sum(1 for k, p, n, _, _ in rs if p == 'loop-vectorize' and k == 'Passed')
    f['vf'] = max((int(a.get('VectorizationFactor', 0) or 0) for k, p, n, _, a in rs
                   if p == 'loop-vectorize' and k == 'Passed' and str(a.get('VectorizationFactor', '')).isdigit()),
                  default=0)
    f['slp'] = sum(1 for k, p, n, _, _ in rs if p == 'slp-vectorizer' and k == 'Passed')
    f['licm_hoisted'] = sum(1 for k, p, n, _, _ in rs if p == 'licm' and n == 'Hoisted')
    f['sroa'] = sum(1 for k, p, n, _, _ in rs if p == 'sroa' and k == 'Passed')
    f['insns'] = max((int(a.get('NumInstructions', 0)) for k, p, n, _, a in rs if n == 'InstructionCount'), default=0)
    return f


# ---- host --------------------------------------------------------------------------------------------
def host_facts(ll, asm):
    f = {}
    if ll:
        m = re.search(r'^; mapal\.host (.*)$', ll, re.M)
        if m:
            f.update(kv(m.group(1)))
        f['calls'] = {s: len(re.findall(r'call [^@\n]*@mapal_gpu_' + s + r'\(', ll)) for s in GPU_SYMS}
        # cpu-gpu-boundary: the runtime flags (`preload` 8, `stage` 16) and the arenas made at
        # function entry (`entry-arena`: ahead of the function's every other device call and clock read).
        m = re.search(r'@mapal_gpu_init\(i32 (\d+)\)', ll)
        f['init_flags'] = int(m.group(1)) if m else 0
        f['arena_entry'] = sum(len(re.findall(r'@mapal_gpu_arena\(', re.split(
            r'\n[^\n]*(?:@mapal_time_ms\(|@mapal_gpu_(?!arena\()\w+\()', d)[0])) for d in ll.split('\ndefine ')[1:])
        # `entry-arena` ON only: a member's buffer is the entry arena's base loaded (OFF asks the arena).
        f['arena_load'] = len(re.findall(r'= load i64, ptr %darena\b', ll))
        f['par'] = len(re.findall(r'@mapal_par_begin\(', ll))
        # recurrence-precision `trip-batch`: loop entries at a batched interval's last trip.
        f['batch'] = len(re.findall(r'; mapal\.batch enter=', ll))
        f['sme'] = len(re.findall(r'@mapal_sme_panel', ll))
        # generic-fusion: every function's `; mapal.flow.host` line, summed, and the unit launches.
        fu = Counter()
        for m in re.finditer(r'; mapal\.flow\.host (.*)$', ll, re.M):
            k = kv(m.group(1))
            for x in ('groups', 'units', 'launches'):
                fu[x] += k.get(x, 0) if isinstance(k.get(x), int) else 0
            fu['refused'] += len([r for r in str(k.get('refused', '')).split(';') if r])
        fu['unit_calls'] = len(re.findall(r'@mapal_gpu_launchv\([^\n]*; unit$', ll, re.M))
        f['flow'] = dict(fu)
        # kernel-merging `zero-fold`: every function's `; mapal.zero.host` line summed, the kernels taking a zero
        # tail (`; mapal.zero sites=`), and the site calls flagged ZEROED (flags bit 1).
        zf = Counter()
        for m in re.finditer(r'; mapal\.zero\.host (.*)$', ll, re.M):
            k = kv(m.group(1))
            for x in ('sites', 'folded', 'kernels', 'gated', 'stopped', 'none'):
                zf[x] += k.get(x, 0) if isinstance(k.get(x), int) else 0
        zf['tails'] = len(re.findall(r'; mapal\.zero sites=', ll))
        # (flags bit 1 = ZEROED; fp8-leftovers' COLD (8) and COOP (4) ride beside it)
        zf['flagged'] = sum(int(x) & 2 != 0 for x in re.findall(r'@mapal_gpu_site\(i64 -?\d+, i32 (\d+),', ll))
        # A direct atomic / stream-K launch (no table: `--cands=1`) folded: no zero call right before it.
        lines = ll.splitlines()
        zf['direct'] = sum(1 for i, x in enumerate(lines)
                           if re.search(r'@mapal_gpu_launch\(.*; (splitk=\d+ atomic|streamk=)', x)
                           and not (i and '@mapal_gpu_zero(' in lines[i - 1]))
        f['zero'] = dict(zf)
        # kernel-merging `horizontal`: every function's `; mapal.hz.host` line summed, the merged kernels the host
        # names (`; mapal.hz kernel=`), and the launches of one (a merged table's or a single merged kernel's).
        hz = Counter()
        for m in re.finditer(r'; mapal\.hz\.host (.*)$', ll, re.M):
            k = kv(m.group(1))
            for x in ('launches', 'groups', 'members', 'cells', 'units', 'tiles', 'branchy'):
                hz[x] += k.get(x, 0) if isinstance(k.get(x), int) else 0
        # (`calls`' persistent kernels, `grid=`, are counted under `chain`.)
        merged = set(re.findall(r'; mapal\.hz kernel=mapal_tile_([0-9a-f]{16}) (?![^\n]*grid=)', ll))
        hz['specs'] = len(merged)
        hz['calls'] = sum(1 for m in re.finditer(r'call void @mapal_gpu_(?:site|launchv)\(i64 (-?\d+)', ll)
                          if f'{int(m.group(1)) & (2**64 - 1):016x}' in merged)
        f['hz'] = dict(hz)
        # kernel-merging `calls`: every function's `; mapal.calls.host` line summed, the persistent kernels the host
        # names (`; mapal.hz kernel=… grid=`), and the cooperative launches of one (a table flagged COOP, bit 2, or
        # one `launch_coop`).
        ch = Counter()
        for m in re.finditer(r'; mapal\.calls\.host (.*)$', ll, re.M):
            k = kv(m.group(1))
            for x in ('chains', 'formed', 'units', 'barriers', 'zeros'):
                ch[x] += k.get(x, 0) if isinstance(k.get(x), int) else 0
        pers = set(re.findall(r'; mapal\.hz kernel=mapal_tile_([0-9a-f]{16}) [^\n]*grid=', ll))
        ch['specs'] = len(pers)
        ch['coop'] = sum(1 for m in re.finditer(r'call void @mapal_gpu_(site|launch_coop)\(i64 (-?\d+), i32 (\d+)', ll)
                         if f'{int(m.group(2)) & (2**64 - 1):016x}' in pers
                         and (m.group(1) == 'launch_coop' or int(m.group(3)) & 4))
        f['chain'] = dict(ch)
    if asm:
        ins = [x.split('\t')[-1].strip() for x in asm.splitlines() if re.match(r'^\s+[0-9a-f]+:\s', x)]
        f['obj'] = dict(insns=len(ins), ymm_ops=sum('%ymm' in x for x in ins),
                        xmm_ops=sum('%xmm' in x and '%ymm' not in x for x in ins),
                        ymm_fma=sum(x.startswith('vfmadd') and '%ymm' in x for x in ins),
                        fma=sum(x.startswith(('vfmadd', 'vfmsub', 'vfnmadd')) for x in ins))
    return f


# ---- S65 movement: what the facts require to cross per trip vs what the host IR does ------------------
def llt_size(t):
    """Bytes of an LLVM type text (the emitter's closed grammar: iN, float, double, ptr, [n x T], { … })."""
    t = t.strip()
    if t.startswith('['):
        n, e = t[1:-1].split(' x ', 1)
        return int(n) * llt_size(e)
    if t.startswith('{'):
        fs, depth, cur = [], 0, ''
        for ch in t[1:-1]:
            depth += ch in '{['
            depth -= ch in '}]'
            if ch == ',' and depth == 0:
                fs.append(cur)
                cur = ''
            else:
                cur += ch
        fs.append(cur)
        size, al = 0, 1
        for f in fs:
            a = llt_align(f)
            size = -(-size // a) * a + llt_size(f)
            al = max(al, a)
        return -(-size // al) * al
    return {'ptr': 8, 'double': 8, 'float': 4}.get(t, int(t[1:]) // 8 if re.fullmatch(r'i\d+', t) else 0)


def llt_align(t):
    t = t.strip()
    if t.startswith('['):
        return llt_align(t[1:-1].split(' x ', 1)[1])
    if t.startswith('{'):
        return max(llt_align(f) for f in re.split(r',(?![^{\[]*[}\]])', t[1:-1]))
    return max(1, llt_size(t))


def copy_bytes(line):
    """A host `llvm.memcpy`'s byte count (the gep-null sizeof or a literal); None when not constant."""
    m = re.search(r'getelementptr \((.*), ptr null, i64 1\) to i64\)', line)
    if m:
        return llt_size(m.group(1))
    m = re.search(r'llvm\.memcpy[^(]*\(ptr [^,]+, ptr [^,]+, i64 (\d+)', line)
    return int(m.group(1)) if m else None


def block_list(fn):
    out, label, cur = [], '', []
    for line in fn.splitlines():
        m = re.match(r'^(\w[\w.]*):', line)
        if m:
            out.append((label, cur))
            label, cur = m[1], []
        else:
            cur.append(line)
    out.append((label, cur))
    return out


def block_moves(lines):
    """memcpy / h2d / d2h / alloc call counts and bytes of one block."""
    c = Counter()
    for x in lines:
        if 'call void @llvm.memcpy' in x:
            c['copy_n'] += 1
            b = copy_bytes(x)
            c['copy_b'] += b or 0
            c['copy_unknown'] += b is None
        m = re.search(r'call \w+ @mapal_gpu_(h2d_async|h2d_on|h2d|d2h|d2h_on|alloc|arena|tile)\((.*)\)', x)
        if not m:
            continue
        op, args = m.group(1), [a.split()[-1] for a in m.group(2).split(',')]
        last = int(args[-1]) if re.fullmatch(r'\d+', args[-1]) else None
        # cpu-gpu-boundary: `sync_n` counts the calls that wait for the device (a synchronous
        # upload, every copy-out); `async-h2d`'s upload is the same bytes without the wait.
        if op in ('h2d', 'h2d_async', 'h2d_on'):
            c['h2d_n'] += 1
            c['h2d_b'] += last or 0
            c['sync_n'] += op == 'h2d'
        elif op in ('d2h', 'd2h_on'):
            c['d2h_n'] += 1
            c['d2h_b'] += last or 0
            c['sync_n'] += 1
        elif op == 'tile':  # the S52 round trip: two copies in, one out
            c['h2d_n'] += 2
            c['d2h_n'] += 1
        elif op == 'arena':  # one allocation per function, on its first ask
            c['arena_n'] += 1
        else:
            c['alloc_n'] += 1
    return c


MOVE_KEYS = ('copy_n', 'copy_b', 'h2d_n', 'h2d_b', 'd2h_n', 'd2h_b', 'alloc_n', 'sync_n')


def move_facts(ll, trips=()):
    """Per program loop the emitter headed with `; mapal.move` (its intent: the facts' derivation), what
    the host IR realizes per trip — every block from the header to the latch, less a first pass's
    (`store i1 true` to a pass flag: once per loop entry) — and, with the source's trip counts, the
    run's totals the trace must show (`runtime_*`: straight-line once, the decide cone trips + 1 times,
    the advance cone trips times, a first-pass branch once and its sibling one trip fewer)."""
    loops, total, once = [], Counter(), Counter()
    li = arena = 0
    for fn in re.split(r'\n(?=define )', ll):
        if not fn.startswith('define'):
            continue
        # A function that makes an arena calls it (a `declare` between two functions is not one).
        arena += 'call i64 @mapal_gpu_arena(' in fn
        bbs = block_list(fn)
        # `entry-arena`: a slot the entry block seeds from the arena never allocates (its
        # `mapal_gpu_alloc` finds the buffer): those call sites move nothing.
        entry = next((ls for lab, ls in bbs if lab == 'entry'), [])
        seeded = set(re.findall(r'store i64 %[\w.]+, ptr (%d\d+)$', '\n'.join(entry), re.M))
        if seeded:
            fn = '\n'.join(x for x in fn.split('\n')
                           if not ((m := re.search(r'@mapal_gpu_alloc\(ptr (%d\d+),', x)) and m.group(1) in seeded))
            bbs = block_list(fn)
        index = {lab: i for i, (lab, _) in enumerate(bbs)}
        inside, last = set(), 0
        for h, (lab, lines) in enumerate(bbs):
            mv = next((x for x in lines if x.startswith('; mapal.move ')), None)
            if not mv:
                continue
            intent = {k: v for k, v in kv(mv[len('; mapal.move '):]).items() if k != 'rungs_on'}
            latch = max(i for i, (_, ls) in enumerate(bbs) if any(re.search(r'br label %' + lab + r'$', x)
                                                                  for x in ls))
            exit_l = bbs[latch + 1][0] if latch + 1 < len(bbs) else None
            adv = h
            for i in range(h, latch + 1):
                for x in bbs[i][1]:
                    m = re.search(r'br i1 \S+, label %([\w.]+), label %' + str(exit_l) + r'$', x)
                    if m:
                        adv = index.get(m.group(1), h)
            first = {i for i in range(h, latch + 1) if any(re.search(r'store i1 true, ptr %s\d+', x)
                                                           for x in bbs[i][1])}
            sib = set()
            for i in range(h, latch + 1):
                for x in bbs[i][1]:
                    m = re.search(r'br i1 \S+, label %([\w.]+), label %([\w.]+)$', x)
                    if m and index.get(m.group(2)) in first:
                        sib.add(index.get(m.group(1)))
                    elif m and index.get(m.group(1)) in first:
                        sib.add(index.get(m.group(2)))
            per, nested = Counter(), 0
            n = trips[li] if li < len(trips) else None
            # `trip-batch`: the loop enters at trip E (`; mapal.batch enter=E` before its header), so it runs
            # n − E trips; the batch launch before it is straight-line.
            pre = [x.strip() for _, ls in bbs[last:h] for x in ls if x.strip().startswith('; mapal.batch enter=')]
            if n is not None and pre:
                n -= int(kv(pre[-1])['enter'])
            run = Counter()
            for i in range(h, latch + 1):
                inside.add((id(fn), i))
                c = block_moves(bbs[i][1])
                if i not in first:
                    per += c
                if n is not None:
                    mult = n + 1 if i < adv else n
                    mult = 1 if i in first else mult - 1 if i in sib else mult
                    for k in MOVE_KEYS:
                        run[k] += c[k] * mult
                for x in bbs[i][1]:
                    m = re.search(r'br (?:i1 \S+, )?label %([\w.]+)', x)
                    if m and h < index.get(m.group(1), -1) <= i and c['copy_n']:
                        nested += 1
            # a slot allocates once (`mapal_gpu_alloc` caches it): every call site counts once at runtime
            run['alloc_n'] = sum(block_moves(bbs[i][1])['alloc_n'] for i in range(h, latch + 1))
            loops.append(dict(intent=intent, trips=n, nested_copy_blocks=nested,
                              **{k: per[k] for k in MOVE_KEYS}, run=dict(run)))
            total += run
            li += 1
            last = latch + 1
        for i, (_, lines) in enumerate(bbs):
            if (id(fn), i) not in inside:
                once += block_moves(lines)
    f = dict(loops=len(loops), per_loop=loops, arena=arena)
    for k in MOVE_KEYS:
        f[k] = sum(x[k] for x in loops)
    for k in ('h2d', 'd2h', 'host_copy', 'same'):
        f['intent_' + k] = sum(int(x['intent'].get(k, 0)) for x in loops)
    carried = [c for x in loops for c in str(x['intent'].get('carried', '-')).split(',') if c != '-']
    f['carried'] = len(carried)
    f['by_value'] = sum(c.endswith(':value') for c in carried)
    f['carried_b'] = sum(int(c.split(':')[1]) for c in carried)
    if all(x['trips'] is not None for x in loops):
        for k in MOVE_KEYS:
            f['runtime_' + k] = once[k] + total[k]
    return f


def kid(kernel):
    """A kernel's host id: its name's 64-bit hash as the signed i64 operand the host passes."""
    m = re.search(r'_([0-9a-f]{16})$', kernel)
    if not m:
        return None
    v = int(m.group(1), 16)
    return v - ((v >= 1 << 63) << 64)


def launched(kernel, host_ll):
    """How many host calls name the kernel."""
    v = kid(kernel)
    return -1 if v is None else len(re.findall(r'\bi64 ' + str(v) + r'\b', host_ll))


def launches(host_ll):
    """id -> the kernel's first `mapal_gpu_launch[_l2]` call: gx gy block smem, combine and S off the
    emitter's comment (`; splitk=S [atomic]`, `; streamk=CxI`)."""
    out = {}
    for m in re.finditer(r'call void @mapal_gpu_(launch|launch_l2)\(i64 (-?\d+),([^\n]*)', host_ll):
        if int(m.group(2)) in out:
            continue
        i32 = [int(x) for x in re.findall(r'i32 (\d+)', m.group(3))][-4:]
        if len(i32) < 4:
            continue
        f = dict(gx=i32[0], gy=i32[1], block=i32[2], smem=i32[3], S=1, combine='none')
        if c := re.search(r'; splitk=(\d+)( atomic)?', m.group(3)):
            f.update(S=int(c.group(1)), combine='atomic' if c.group(2) else 'splitk')
        elif re.search(r'; streamk=', m.group(3)):
            f['combine'] = 'streamk'
        out[int(m.group(2))] = f
    # per-site-choice: a table's candidates (S and the combine come from the plan line: `facts`)
    for t in site_tables(host_ll):
        for i, c in enumerate(t['cands']):
            out.setdefault(kid(c['kernel']), dict(gx=c['gx'], gy=c['gy'], block=c['block'], smem=c['smem'],
                                                  site=t['site'], cand=i))
    return out


MASK = (1 << 64) - 1
SITE_CALL = re.compile(r'call void @mapal_gpu_site\(i64 (-?\d+), i32 (\d+), i32 (\d+), i32 (\d+), ptr (@[\w.]+)')
CAND_FIELDS = ('id', 'rid', 'zero', 'ws', 'gx', 'gy', 'block', 'smem', 'rgx', 'rblock')
ENV_ARM = re.compile(r'^(rules|pick\d+)$')  # per-site-choice B5: arms of one binary chosen by MAPAL_GPU_PICK


def kname(v):
    return f'mapal_tile_{v & MASK:016x}'


def site_tables(host_ll):
    """per-site-choice B2: each `mapal_gpu_site` call in host order: site (candidate 0's kernel), flags, n and each
    candidate's recipe (the `%mapal_cand` constant: kernel, reduce kernel, zero / workspace bytes, grids)."""
    consts = dict(re.findall(r'^(@[\w.]+) = .*?constant \[\d+ x %mapal_cand\] \[(.*)\]$', host_ll, re.M))
    out = []
    for m in SITE_CALL.finditer(host_ll):
        rows = [dict(zip(CAND_FIELDS, (int(x.split()[1]) for x in r.split(', '))))
                for r in re.findall(r'%mapal_cand \{ ([^}]*) \}', consts.get(m[5], ''))]
        for r in rows:
            r['kernel'], r['rkernel'] = kname(r['id']), kname(r['rid']) if r['rid'] else None
        out.append(dict(site=kname(int(m[1])), flags=int(m[2]), n=int(m[4]), cands=rows, at=m.start()))
    return out


def choice_axes(path=PROFILE_RS):
    """The rungs of `CHOICE_AXES` (profile.rs): what a site table measures instead of a rung arm."""
    t = read(path) or ''
    m = re.search(r'pub\(crate\) const CHOICE_AXES: &\[ChoiceAxis\] = &\[(.*?)\n\];', t, re.S)
    return sorted(set(re.findall(r'rung: "([^"]+)"', m[1]))) if m else []


def plan_lines(dev):
    return {p['kernel']: p for p in (kv(m) for m in re.findall(r'^; mapal\.plan (.*)$', dev or '', re.M))}


def recipe(c, plan):
    """A candidate as a `.launch` line: `kernel gx gy block smem [S rkernel rgx rgy rblock rsmem | S atomic |
    1 streamk]` (S and the combine off the kernel's plan line)."""
    head = [c['kernel'], c['gx'], c['gy'], c['block'], c['smem']]
    S = plan.get('S', 1)
    if c['ws']:
        return head + [S, c['rkernel'], c['rgx'], 1, c['rblock'], 0]
    if c['zero']:
        return head + ([1, 'streamk'] if plan.get('combine') == 'streamk' else [S, 'atomic'])
    return head


def launch_file(stem):
    """`<stem>.launch`: the first of the host's launches and site tables (a table: its candidate 0); `<stem>.sites`:
    `site i n kernel gx gy block smem [tail]` per candidate of every table; `<stem>.pick<i>.launch` + `.ptx` (a
    link to the shipped module) per candidate i >= 1 of the first table (tile_bench's per-candidate legs)."""
    host = read(stem + '.host.ll') or ''
    plans = plan_lines(read(stem + '.dev.ll'))
    ts = site_tables(host)
    first = re.search(r'call void @mapal_gpu_(tile|launch|launch_l2)\(i64 (-?\d+)', host)
    for f in glob.glob(stem + '.pick*.launch') + glob.glob(stem + '.pick*.ptx'):
        if re.fullmatch(r'\.pick\d+\.(launch|ptx)', f[len(stem):]):
            os.remove(f)
    line = ''
    if ts and (not first or ts[0]['at'] < first.start()):
        t = ts[0]
        line = ' '.join(map(str, recipe(t['cands'][0], plans.get(t['cands'][0]['kernel'], {}))))
        for i, c in enumerate(t['cands'][1:], 1):
            open(f'{stem}.pick{i}.launch', 'w').write(' '.join(map(str, recipe(c, plans.get(c['kernel'], {})))) + '\n')
            if os.path.exists(stem + '.ptx'):
                os.symlink(os.path.basename(stem) + '.ptx', f'{stem}.pick{i}.ptx')
    elif first:  # the pre-sprint rule: the first launch, the split's second (reduce) launch off the next call
        ls = re.findall(r'call void @mapal_gpu_(?:tile|launch|launch_l2)\(i64 (-?\d+)(.*)', host)[:2]
        n = lambda l: re.findall(r'i32 (\d+)', l.split(';')[0])
        s = re.search(r'; splitk=(\d+)( atomic)?', ls[0][1])
        tail = ([s[1], 'atomic'] if s[2] else [s[1], kname(int(ls[1][0])), *n(ls[1][1])]) if s else \
            (['1', 'streamk'] if re.search(r'; streamk=', ls[0][1]) else [])
        line = ' '.join([kname(int(ls[0][0])), *n(ls[0][1]), *tail])
    open(stem + '.launch', 'w').write(line + '\n' if line else '')
    if ts:
        open(stem + '.sites', 'w').write(''.join(
            f"{t['site']} {i} {t['n']} {' '.join(map(str, recipe(c, plans.get(c['kernel'], {}))))}\n"
            for t in ts for i, c in enumerate(t['cands'])))
    elif os.path.exists(stem + '.sites'):
        os.remove(stem + '.sites')


def kernel_bodies(dev):
    """Each kernel's body in a device unit, its own name and metadata numbers taken out."""
    out = set()
    for m in re.finditer(r'^define ptx_kernel void @(mapal_\w+)\((.*?)^}$', dev or '', re.M | re.S):
        out.add(re.sub(r'!\d+', '!', m[2]).replace(m[1], '@'))
    return out


def covered(arm, shipped, rungs):
    """An arm a site table measures: every rung a CHOICE_AXES one, every kernel already a candidate's."""
    host = read(shipped + '.host.ll') or ''
    if not site_tables(host) or not set(rungs.replace('+', ',').split(',')) <= set(choice_axes()):
        return False
    return kernel_bodies(read(arm + '.dev.ll')) <= kernel_bodies(read(shipped + '.dev.ll'))


def source_text(d, stem):
    base = stem.split('.')[0]
    for cand in [base] + [base[:-len(f)] for f in ('_rc', '_r', '_x', '_c', '_tf') if base.endswith(f)]:
        t = read(os.path.join(d, '..', 'tiles', cand + '.mapal'))
        if t is not None:
            return t
    return None


def source_trips(d, stem):
    """Each `loop {`'s trip count, in source order: its guard `(<i> < <N>)` with the counter from 0 by 1
    (every timed loop of the suite); () when a guard is not of that form."""
    t = source_text(d, stem) or ''
    out = []
    for m in re.finditer(r'loop \{\s*\((\w+) < (\d+)\)', t):
        v = m.group(1)
        if not re.search(r'mut ' + v + r': i32 <- 0;', t) or not re.search(v + r' \+ 1 -> ' + v + ';', t):
            return ()
        out.append(int(m.group(2)))
    return tuple(out) if len(out) == t.count('loop {') else ()


def source_loops(d, stem):
    """`loop {` blocks in the stem's program (`<bundle>/tiles/<program>.mapal`); -1: source not found."""
    base = stem.split('.')[0]
    for cand in [base] + [base[:-len(f)] for f in ('_rc', '_r', '_x', '_c', '_tf') if base.endswith(f)]:
        t = read(os.path.join(d, '..', 'tiles', cand + '.mapal'))
        if t is not None:
            return t.count('loop {')
    return -1


def kv(s):
    out = {}
    for t in s.split():
        if '=' in t:
            k, v = t.split('=', 1)
            out[k] = int(v) if re.fullmatch(r'-?\d+', v) else v
    out['rungs_on'] = out['rungs_on'].split(',') if isinstance(out.get('rungs_on'), str) else out.get('rungs_on', [])
    return out


def launch_facts(path):
    """`kernel gx gy block smem [S rkernel rgx rgy rblock rsmem | S atomic | 1 streamk]`"""
    if not os.path.exists(path):
        return {}
    t = open(path).read().split()
    if not t:
        return {}
    f = dict(kernel=t[0], gx=int(t[1]), gy=int(t[2]), block=int(t[3]), smem=int(t[4]), S=1, combine='none',
             reduce=None)
    if len(t) > 6 and t[6] == 'atomic':
        f.update(S=int(t[5]), combine='atomic')
    elif len(t) > 6 and t[6] == 'streamk':
        f['combine'] = 'streamk'
    elif len(t) > 6:
        f.update(S=int(t[5]), combine='splitk', reduce=t[6], rgx=int(t[7]), rgy=int(t[8]), rblock=int(t[9]))
    return f


def exp_calls(dev, sym=r'@mapal\.exp\.f32'):
    """exp (ADR-0042): {kernel: calls of `@mapal.exp.f32` in its body or the internal functions it reaches} — the
    device unit's call graph (a member body is a function the kernel calls; the exp definition is inlined later).
    `sym`: another builtin's symbol pattern (ADR-0046: sqrt, sin, cos)."""
    fns = {}
    for chunk in dev.split('\ndefine ')[1:]:
        m = re.match(r'[^@]*@([\w.$]+)\(', chunk)
        if m:
            fns[m.group(1)] = chunk.split('\n}\n', 1)[0]
    out = {}
    for k, body in fns.items():
        if not body.startswith('ptx_kernel'):
            continue
        seen, todo, n = {k}, [k], 0
        while todo:
            b = fns[todo.pop()]
            n += len(re.findall(r'call float ' + sym + r'\(', b))
            for c in re.findall(r'call [^@\n]*@([\w.$]+)\(', b):
                if c in fns and c not in seen and not re.fullmatch(sym.lstrip('@'), c):
                    seen.add(c)
                    todo.append(c)
        out[k] = n
    return out


def flow_facts(f, dev):
    """A unit kernel's `; mapal.flow` line (func/flow.rs): how many members it holds, keeps (no store, no
    pointer), stores and reads from Gmem, and its `define`'s pointer parameters in the device unit."""
    n = lambda key: len([x for x in str(f.get(key, '')).split(';') if x])
    d = re.search(r'^define ptx_kernel void @' + re.escape(f['kernel']) + r'\((.*)\)', dev, re.M)
    return dict(n_members=n('members'), n_keep=n('keep'), n_store=n('store'), n_gmem=n('gmem'),
                n_law=n('law'), n_halo=str(f.get('members', '')).count('+halo'), rows=f.get('rows', 0),
                smem=f.get('smem', 0), lanes=f.get('lanes', 1), tree=f.get('tree', 0), onchip=f.get('onchip', 0),
                ll_params=d.group(1).count('ptr addrspace(1)') if d else -1,
                # kernel-merging `zero-fold`: the zero tail's jobs (`%zp<j>`) and its parameters (`%zt`, then a
                # pointer and a byte count a job).
                zero=d.group(1).count('%zp') if d else 0,
                zero_params=(1 + 2 * d.group(1).count('%zp')) if d and '%zp' in d.group(1) else 0,
                # compile-time-graph A: a row-block unit's rows per block, tiles, colors (0 on a per-row unit).
                rows_per_unit=f.get('rows_per_unit', 0), tiles=f.get('tiles', 0), colors=f.get('colors', 0),
                # long-rows `row-slices`: blocks a row, the phases its laws leave (grid barriers between), and the
                # rescale laws (ADR-0042 D7: a sum folded beside its shift).
                slices=f.get('slices', 1) or 1, phases=f.get('phases', 1) or 1,
                # long-prefill `slice-launches`: the phases at kernel boundaries (one launch a phase).
                launches=f.get('launches', 0) or 0,
                n_rescale=str(f.get('laws', '')).split(',').count('rescale'),
                # flash-attention `stream`: one pass over windows of the axis — its windows, whether a wholly
                # masked window is skipped (`on`, else why / `none`), the axis-wide members never stored.
                stream=1 if f.get('stream') else 0, windows=f.get('windows', 0) or 0,
                # sweep-attention-2 `stream-mma`: the products on the warp matrix product.
                stream_mma=1 if ':mma' in str(f.get('stream', '')) else 0,
                skip=1 if str(f.get('skip', '')) == 'on' else 0,
                n_dissolved=len([x for x in str(f.get('dissolved', '')).split(',') if x]),
                # sweep-attention `unit-uniform`: the operands the unit's launch moves by a captured number.
                n_moved=len([x for x in str(f.get('moved', '')).split(';') if x]),
                members=str(f.get('members', '')),
                keep=[x.split(':')[0] for x in str(f.get('keep', '')).split(';') if x],
                store=[x for x in str(f.get('store', '')).split(';') if x])


# ---- static: ptxas / cuobjdump / objdump ------------------------------------------------------------
def static(d, sm):
    ev = os.path.join(d, 'evidence')
    os.makedirs(ev, exist_ok=True)
    seen, todo = {}, []
    for p in sorted(glob.glob(os.path.join(d, '*.ptx'))):
        stem = os.path.basename(p)[:-4]
        h = hashlib.sha1(open(p, 'rb').read()).hexdigest()
        if h in seen:  # an identical PTX (a null copy, an arm equal in PTX): its twin's evidence
            json.dump({'same_as': seen[h]}, open(os.path.join(ev, stem + '.alias'), 'w'))
            continue
        seen[h] = stem
        if not os.path.exists(os.path.join(ev, stem + '.sass')):
            todo.append(stem)
    # Incremental bundles (integrated_bundle.sh): EVIDENCE_CACHE holds each PTX's ptxas -v and SASS by the PTX's
    # bytes, the SM and the ptxas release — an unchanged PTX takes the outputs ptxas / cuobjdump made for it before.
    cache, ver = os.environ.get('EVIDENCE_CACHE'), None
    if cache and todo:
        ver = (subprocess.run([os.path.join(HERE, 'cudatool.sh'), 'ptxas', '--version'], capture_output=True,
                              text=True).stdout.strip().splitlines() or [''])[-1]
        if not os.path.exists(os.path.join(ev, 'ptxas.version')):
            open(os.path.join(ev, 'ptxas.version'), 'w').write(ver + '\n')
        key = {t: hashlib.sha1(open(os.path.join(d, t + '.ptx'), 'rb').read() + f'|{sm}|{ver}'.encode()).hexdigest()
               for t in todo}
        hit = [t for t in todo if all(os.path.exists(os.path.join(cache, key[t] + x)) for x in ('.ptxas', '.sass'))]
        for t in hit:
            for x in ('.ptxas', '.sass'):
                shutil.copyfile(os.path.join(cache, key[t] + x), os.path.join(ev, t + x))
        todo = [t for t in todo if t not in hit]
        if hit:
            print(f'static: {len(hit)} PTX from the evidence cache', file=sys.stderr)
    if todo:
        lst = os.path.join(ev, 'todo.txt')
        open(lst, 'w').write('\n'.join(todo) + '\n')
        script = (f'cd "{os.path.abspath(ev)}" && ptxas --version | tail -1 > ptxas.version && '
                  f'xargs -P 8 -I@ sh -c \'ptxas -arch={sm} -O3 -v ../@.ptx -o @.cubin 2> @.ptxas; '
                  f'cuobjdump -sass @.cubin > @.sass 2>&1; rm -f @.cubin\' < todo.txt')
        subprocess.run([os.path.join(HERE, 'cudatool.sh'), 'bash', '-c', script], check=True)
        os.remove(lst)
        if cache:
            os.makedirs(cache, exist_ok=True)
            for t in todo:
                for x in ('.ptxas', '.sass'):
                    tmp = os.path.join(cache, f'{key[t]}{x}.{os.getpid()}')
                    shutil.copyfile(os.path.join(ev, t + x), tmp)
                    os.replace(tmp, os.path.join(cache, key[t] + x))
    objdump = next((x for x in (os.environ.get('LLVM', '/opt/homebrew/opt/llvm/bin') + '/llvm-objdump',
                                'llvm-objdump', 'objdump') if subprocess.run(['sh', '-c', f'command -v {x}'],
                                                                             capture_output=True).returncode == 0),
                   None)
    for o in sorted(glob.glob(os.path.join(d, '*.host.o'))):
        stem = os.path.basename(o)[:-7]
        a = os.path.join(ev, stem + '.host.asm')
        if objdump and not os.path.exists(a):
            with open(a, 'w') as fh:
                subprocess.run([objdump, '-d', '--no-show-raw-insn', o], stdout=fh)
    print(f'static: {len(todo)} PTX through ptxas/cuobjdump ({len(seen)} distinct)', file=sys.stderr)
    # bugs-properties `ptx-isa`: the preflight — every distinct PTX of the part must pass `ptxas` for its target
    # (an error, a fatal, or no SASS fails the bundle loudly; cross-arch 5fcb401b finding 1: 25 `_f8w` modules
    # rejected on sm_89 were logged here and nothing read them, the box failed at `cuModuleLoadData`).
    bad = []
    for t in sorted(seen.values()):
        log = read(os.path.join(ev, t + '.ptxas')) or ''
        errs = [l for l in log.splitlines() if re.search(r'\b(error|fatal)\s*:', l)]
        if errs or not (read(os.path.join(ev, t + '.sass')) or '').strip():
            bad.append((t, errs or ['no SASS']))
    if bad:
        for t, errs in bad:
            print(f'ptxas preflight FAILED {sm} {t}.ptx: ' + ' | '.join(errs[:3]), file=sys.stderr)
        sys.exit(f'ptxas preflight: {len(bad)} of {len(seen)} PTX rejected for {sm} in {d}')


# ---- facts -------------------------------------------------------------------------------------------
def read(path):
    if os.path.exists(path):
        return open(path, errors='replace').read()
    if os.path.exists(path + '.gz'):
        return gzip.open(path + '.gz', 'rt', errors='replace').read()
    return None


def profile_part(prof, path=PROFILE_RS):
    """The part's own geometry from its `TargetProfile` (`name: "<prof>"` in profile.rs: `sms`, threads / blocks /
    shared memory an SM) — the facts the rules read — where PARTS has no row (cross-arch 3a8e654f: the A100 / H100
    had none, so every `occ.sms` intent held vacuously there). {} where the profile spells none."""
    src = read(path) or ''
    m = re.search(r'name: "%s",(.*?)(?:\npub const|\Z)' % re.escape(prof), src, re.S)
    if not m:
        return {}
    body, out = m.group(1), {}
    for key, field in (('sms', 'sms'), ('threads', 'max_threads_per_sm'), ('blocks', 'max_blocks_per_sm'),
                       ('smem', 'smem_per_sm')):
        f = re.search(r'\b%s: ([0-9_ *]+),' % field, body)
        if f:
            out[key] = eval(f.group(1).replace('_', ''), {})
    return out


def facts(d, sm):
    prof = os.path.basename(os.path.normpath(d))
    part = PARTS.get(prof) or profile_part(prof)
    lim = dict(threads=part.get('threads', 1536), blocks=part.get('blocks', 16), regs=65536, reg_unit=256,
               max_warps=part.get('threads', 1536) // 32, smem=part.get('smem', 102400), smem_reserved=1024)
    ev = os.path.join(d, 'evidence')
    rows = rows_off(d)
    # (a `.pick<i>` PTX is the shipped module linked under a tile_bench leg's name: its stem's facts)
    stems = sorted({os.path.basename(p).split('.host.o')[0] for p in glob.glob(os.path.join(d, '*.host.o'))}
                   | {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(d, '*.ptx'))
                      if not re.search(r'\.pick\d+\.ptx$', p)})
    out = {}
    for stem in stems:
        base = os.path.join(d, stem)
        src = stem
        al = os.path.join(ev, stem + '.alias')
        if os.path.exists(al):
            src = json.load(open(al))['same_as']
        dev = read(base + '.dev.ll') or read(os.path.join(d, src + '.dev.ll')) or ''
        plans = {p['kernel']: p for p in (kv(m) for m in re.findall(r'^; mapal\.plan (.*)$', dev, re.M))}
        # per-site-choice: a table's candidate i flips its axes' rungs against candidate 0 (`; mapal.site … axes=`)
        flips, names = {}, '|'.join(sorted(KERNEL_RUNGS + ['plan-cost'], key=len, reverse=True))
        for ids, axes in re.findall(r'^; mapal\.site site=\w+ n=\d+ cands=(\S+) axes=(\S+)', dev, re.M):
            for k, ax in zip(ids.split(','), axes.split(',')):
                flips[f'mapal_tile_{k}'] = re.findall(rf'([+-])({names})(?![a-z])', ax)
        flows = {f['kernel']: flow_facts(f, dev) for f in (kv(m) for m in re.findall(r'^; mapal\.flow (.*)$', dev, re.M))}
        exps = exp_calls(dev)
        hw_exp = int('@llvm.nvvm.ex2.approx.ftz.f(' in dev)
        # math (ADR-0046) `math-hw`: f32 sqrt / sin / cos calls a kernel reaches; whether the module's definitions
        # take an approximate unit (the default face's admitted `sqrt.approx` / `cos.approx`).
        maths = exp_calls(dev, r'@mapal\.(?:sqrt|sin|cos)\.f32')
        hw_math = int(bool(re.search(r'@llvm\.nvvm\.(?:sqrt|sin|cos)\.approx\.f\(', dev)))
        # kernel-merging `calls`: each persistent kernel's grid barriers as emitted (before opt / llc copy any).
        gsyncs = {m.group(1): len(re.findall(r'call void @mapal_gsync\(', m.group(2)))
                  for m in re.finditer(r'^define ptx_kernel void @([\w.$]+)\((.*?)^}', dev, re.M | re.S)}
        devm = re.search(r'^; mapal\.device (.*)$', dev, re.M)
        ptx = read(os.path.join(d, src + '.ptx'))
        pf = ptx_facts(ptx) if ptx else {}
        pa = ptxas_facts(read(os.path.join(ev, src + '.ptxas')) or '')
        sass_txt = read(os.path.join(ev, src + '.sass'))
        sf = functions(sass_txt) if sass_txt else {}
        rem_opt = remarks(base + '.dev.opt.remarks.yaml')
        rem_llc = remarks(base + '.dev.llc.remarks.yaml')
        rem_host = remarks(base + '.host.remarks.yaml')
        host = host_facts(read(base + '.host.ll'), read(os.path.join(ev, stem + '.host.asm')))
        host['remarks'] = remark_facts(rem_host)
        launch = launch_facts(base + '.launch')
        arm = arm_of(stem)
        e = dict(stem=stem, arch=prof, sm=sm, arm=[sorted(arm[0]), sorted(arm[1])], host=host, launch=launch,
                 device=kv(devm.group(1)) if devm else {}, kernels={})
        host['cpu'] = int(not devm and 'profile' not in host)
        host['loop'] = source_loops(d, stem)
        host['move'] = move_facts(read(base + '.host.ll') or '', source_trips(d, stem))
        # long-prefill `carry-take`: loop carries that start in their host-built init's device buffer, and its bytes.
        takes = [kv(m.group(1)) for m in re.finditer(r'^\s*; mapal\.take (.*)$', read(base + '.host.ll') or '', re.M)]
        host['take'], host['take_bytes'] = len(takes), sum(int(t.get('bytes', 0)) for t in takes)
        if rows:  # the host rungs the rules ship at their fill (GPU_RESIDENT_BLOCKS: Filled) and face
            from intent import INTENT
            hr = set(INTENT) - set(KERNEL_RUNGS)
            face = 'Contract' if e['device'].get('fma', host.get('fma', 1)) == 1 else 'Exact'
            e['host_rungs'] = sorted(hr)
            e['host_table'] = [r for r in table_rungs(rows, prof, 2, face, arm) if r not in KERNEL_RUNGS]
        hll = read(base + '.host.ll') or ''
        lau = launches(hll)
        cands = {c['kernel'] for t in site_tables(hll) for c in t['cands']}
        for k in sorted(set(plans) | set(pf)):
            p = plans.get(k, {})
            la = lau.get(kid(k), {})
            if 'site' in la:
                la = dict(la, S=p.get('S', 1), combine=p.get('combine', 'none'))
            # D2: an alternative's axes as its launch realized them (async: a cp.async panel; kmajor-a: a k-major
            # panel; plan-cost: the plan line's `pick`; full-tile: a pick under full-tile; else the rung on its plan line).
            ron = set(p.get('rungs_on', []))
            real = lambda r: (p.get('stages', 1) >= 2 if r == 'async' else p.get('kmajor', 0) == 1 if r == 'kmajor-a'
                              else 'pick' in p if r == 'plan-cost' else 'pick' in p and r in ron if r == 'full-tile'
                              else r in ron)
            axes = [[sg, r, real(r)] for sg, r in flips.get(k, [])]
            kf = dict(plan=p, flow=flows.get(k, {}), ptx=pf.get(k, {}), ptxas=pa.get(k, {}),
                      mod=dict(launched=launched(k, hll), cand=int(k in cands), alt=int(bool(axes)), axes=axes,
                               exp32=exps.get(k, 0), hw_exp=hw_exp, math32=maths.get(k, 0), hw_math=hw_math,
                               gsync=gsyncs.get(k, 0)),
                      launch=la,
                      sass=sass_facts(sf[k]) if k in sf else {},
                      remarks=dict(opt=remark_facts(rem_opt, k), llc=remark_facts(rem_llc, k)))
            if 'face' in p and rows:
                tr = set(table_rungs(rows, prof, p['wps'], face_of(p['face']), arm))
                for sign, r in flips.get(k, []):
                    tr = tr | {r} if sign == '+' else tr - {r}
                kf['table'] = dict(rungs_on=[r for r in tr if r in KERNEL_RUNGS])
            r = kf['ptxas']
            blk = p.get('block') or kf['ptx'].get('maxntid')
            if blk and 'regs' in r:
                smem = (p.get('smem') or 0) + r.get('static_smem', 0)
                b, w, o, why = occupancy(blk, r['regs'], smem, lim)
                grid = (p.get('gx') or 0) * (p.get('gy') or 1)
                if p.get('combine') == 'streamk':
                    grid = p.get('ctas', grid)
                elif p.get('combine') in ('splitk', 'atomic'):  # the plan's grid is one partial fold's
                    grid *= p.get('S', 1)
                kf['occ'] = dict(blocks_per_sm=b, warps_per_sm=w, occupancy=round(o, 3), limiter=why,
                                 sms=part.get('sms', 0), blocks=grid,
                                 waves=round(grid / (b * part['sms']), 3) if b and part.get('sms') else 0,
                                 resident_warps_per_sm=round(min(grid, b * part.get('sms', 1)) * -(-blk // 32)
                                                             / part.get('sms', 1), 2))
            e['kernels'][k] = kf
        out[stem] = e
    os.makedirs(ev, exist_ok=True)
    json.dump(out, open(os.path.join(ev, 'facts.json'), 'w'), indent=None, separators=(',', ':'))
    print(f'facts: {len(out)} stems, {sum(len(e["kernels"]) for e in out.values())} kernels', file=sys.stderr)
    return out


# ---- check -------------------------------------------------------------------------------------------
class Missing(Exception):
    pass


class D(dict):
    def __getattr__(self, k):
        if k not in self:
            raise Missing(k)
        v = self[k]
        return D(v) if isinstance(v, dict) else v


class Scope(dict):
    def __missing__(self, k):
        if k in SAFE:  # eval looks a builtin up in the locals first: fall through to the globals
            raise KeyError(k)
        raise Missing(k)


SAFE = dict(ceil=math.ceil, set=set, max=max, min=min, len=len, sum=sum, any=any, all=all, abs=abs, float=float, str=str)
NAMES = re.compile(r'\b(plan|flow|host|launch|ptx|ptxas|sass|occ|remarks|exec|jit|table|device|mod|dev)(\.\w+)+')


def ev(expr, ns):
    try:
        return eval(expr, {'__builtins__': {}, **SAFE}, ns)  # noqa: S307 — our own intent table
    except Missing:
        return None
    except (TypeError, ZeroDivisionError):
        return None


def realized(expr, ns):
    vals = []
    for m in NAMES.finditer(expr):
        n = m.group(0)
        v = ev(n, ns)
        vals.append(f'{n}={"n/a" if v is None else v}')
    return ' '.join(dict.fromkeys(vals))


def exec_facts(results):
    """(stem, kernel) -> regs local smem of the JIT image (tile_bench / mlp_bench `arm stem=` lines)."""
    out = {}
    for line in open(results, errors='replace'):
        m = re.search(r' arm stem=(\S+) .*kernel=(\S+) .*smem=(\d+) regs=(\d+) local_bytes=(\d+)', line)
        if m:
            out[(re.sub(r'\.pick\d+$', '', m.group(1)), m.group(2))] = dict(smem=int(m.group(3)), regs=int(m.group(4)), local=int(m.group(5)))
    return out


def run_intent(it, state, ns):
    """The rows one rung contributes at `state` (`on` / `off`): [(label, expr, realized, status)]."""
    gate = it.get(f'applies_{state}', it.get('applies', 'True'))
    if not it.get(state) or not ev(gate, ns):
        return []
    out = []
    for label, expr in it[state]:
        r = ev(expr, ns)
        out.append((label, expr, realized(expr, ns), 'n/a' if r is None else 'ok' if r else 'MISMATCH'))
    return out


KINDS = ('tile', 'naive', 'mma', 'wgmma', 'reduce', 'rowfold', 'gen', 'cell', 'flow', 'hz', 'calls')


def build_face(host, stem):
    """`contract` / `free` of a build: the emitter's host line (`fma=`, `order=`) where it has one, else the stem's
    face suffix (`_x` / `_r` exact, `_c` / `_r` / `_rc` ordered)."""
    base = stem.split('.')[0]
    contract = host['fma'] == 1 if 'fma' in host else not re.search(r'_(x|r)$', base)
    free = host['order'] == 'Free' if 'order' in host else not re.search(r'_(x|c|r|rc)$', base)
    return dict(contract=contract, free=free)


def check(d, results=None, out=sys.stdout):
    from intent import INTENT
    prof = os.path.basename(os.path.normpath(d))
    fx = json.load(open(os.path.join(d, 'evidence', 'facts.json')))
    ex = exec_facts(results) if results and os.path.exists(results) else {}
    rows = []
    for stem, e in sorted(fx.items()):
        if '.null' in stem:
            continue
        on, off = (set(x) for x in e['arm'])
        for k, kf in sorted(e['kernels'].items()):
            ns = Scope({x: D(kf.get(x, {})) for x in ('plan', 'flow', 'ptx', 'ptxas', 'sass', 'occ', 'table',
                                                       'mod')})
            ns.update(host=D(e['host']), device=D(e['device']), remarks=D(kf.get('remarks', {})),
                      launch=D(kf.get('launch') or (e['launch'] if e['launch'].get('kernel') == k else {})),
                      exec=D(ex.get((stem, k), {})), arch=prof, sm=e['sm'],
                      # fold-algebra: the build's face contracts (default, `_c`, `_rc`), not `_x` / `_r`;
                      # exp (ADR-0042): its order is free (default, `_tf`) — where `exp` may take the hardware form.
                      # Read from the host line (`; mapal.host fma= order=`): the `.xref` builds and the px
                      # differentials are exact-face builds whatever their stem says.
                      **build_face(e['host'], stem))
            kon = set(kf.get('plan', {}).get('rungs_on', [])) | on
            alt = kf.get('mod', {}).get('alt', 0)
            for rung, it in INTENT.items():
                if it.get('host') or (alt and it.get('rule')):
                    continue
                # exp-hw: a face whose order is not free takes the oracle's steps (its OFF realization).
                state = 'off' if rung in off or (rung == 'exp-hw' and not ns['free']) else 'on' if rung in kon or rung in (
                    'row-slices', 'exp-hw', 'exp-rescale', 'slice-launches') or rung.startswith(('face', 'exec', 'module', 'window', 'radix-tile', 'tf32-tile', 'plan-cost', 'linear', 'choice', 'fma-pairs', 'horizontal-', 'calls-', 'narrow-tile', 'half-tile', 'narrow-async', 'staged-regs', 'split-tile', 'mma-combine', 'round-store', 'streamk-price', 'streamk-epi')) else 'off'  # plan-cost / linear / the site-priced split-tile, mma-combine, the module's round-store, streamk-price's walk, the group's streamk-epi ticket: gated by the plan line itself (`applies`), not rungs_on
                rows += [(prof, stem, k, f'{rung}:{state}', *r) for r in run_intent(it, state, ns)]
        # host rungs: one row per stem (kernel `-`)
        if not e['host'].get('calls'):
            continue
        kinds = Counter(kf.get('plan', {}).get('kind', '?') for kf in e['kernels'].values())
        dev = {x: kinds.get(x, 0) for x in KINDS}
        dev['epi'] = sum(kf.get('plan', {}).get('epi', 0) == 1 for kf in e['kernels'].values())
        cells = [kf for kf in e['kernels'].values() if kf.get('plan', {}).get('kind') == 'cell']
        dev['cell_u'] = sum(kf['plan'].get('uniform', 0) > 0 for kf in cells)
        nest = [kf for kf in cells if kf['plan'].get('nested', 0) > 0]
        dev['cell_nest'] = len(nest)
        # A nest cell walks its fold on the device: a loop, or the walk unrolled (≥ work-1 FP ops).
        fp = lambda p: sum(p.get(k, 0) for k in ('fma_f32', 'mul_f32', 'add_f32'))
        dev['cell_nest_walk'] = sum(kf.get('ptx', {}).get('loops', 0) > 0
                                    or fp(kf.get('ptx', {})) >= kf['plan'].get('work', 1) - 1 for kf in nest)
        # recurrence-precision: range cells, cells reading a device number, batch variants, lane cells (and those
        # combining by >= 5 butterfly shuffles: log2 of a 32-lane warp).
        has = lambda kf, key: str(kf['plan'].get(key, '')) not in ('', '0')
        dev['cell_range'] = sum(has(kf, 'range') for kf in cells)
        dev['cell_dev'] = sum(has(kf, 'dev') for kf in cells)
        dev['cell_batch'] = sum(has(kf, 'batch') for kf in cells)
        lanes = [kf for kf in cells if int(kf['plan'].get('lanes', 1) or 1) > 1]
        dev['cell_lanes'] = len(lanes)
        dev['cell_lanes_shfl'] = sum(kf.get('ptx', {}).get('shfl', 0) >= 5 for kf in lanes)
        # device-geometry: post sites (`post=1`: tile kernels and their split reduce), row groups (`group=`; the
        # block's rows inside one group: one `b` panel a group), piece cells, lane cells beside local folds.
        tiles = [kf for kf in e['kernels'].values() if kf.get('plan', {}).get('kind') in ('tile', 'mma', 'naive')]
        dev['post'] = sum(kf['plan'].get('post', 0) == 1 for kf in tiles)
        dev['post_epi'] = sum(kf['plan'].get('post', 0) == 1 and kf['plan'].get('epi', 0) == 1 for kf in tiles)
        dev['post_split'] = sum(kf['plan'].get('post', 0) == 1 and kf['plan'].get('combine') == 'splitk' for kf in tiles)
        dev['post_red'] = sum(kf.get('plan', {}).get('kind') == 'reduce' and kf['plan'].get('epi', 0) == 1
                              for kf in e['kernels'].values())
        # long-prefill `uniform-base`: tile kernels whose launch moves an operand by a captured number (`ubase=`).
        dev['ubase'] = sum(str(kf['plan'].get('ubase', '')) not in ('', '0') for kf in tiles)
        dev['ubase_staged'] = sum(str(kf['plan'].get('ubase', '')) not in ('', '0')
                                  and kf['plan'].get('kind') in ('tile', 'mma') for kf in tiles)
        grp = [kf for kf in tiles if int(kf['plan'].get('group', 0) or 0) > 0]
        dev['group'] = len(grp)
        dev['group_whole'] = sum(kf['plan'].get('kind') == 'tile'
                                 and int(kf['plan']['group']) % (kf['plan'].get('ty', 1) * kf['plan'].get('tm', 1)) == 0
                                 for kf in grp)
        dev['cell_piece'] = sum(has(kf, 'piece') for kf in cells)
        # caching-memory `retain-cell`: a cell's delta kernel (`retain=<step>x<w>`).
        dev['cell_retain'] = sum(has(kf, 'retain') for kf in cells)
        multi = [kf for kf in lanes if kf['plan'].get('nested', 0) > 1]
        dev['cell_lanes_multi'] = len(multi)
        dev['cell_lanes_multi_shfl'] = sum(kf.get('ptx', {}).get('shfl', 0) >= 5 for kf in multi)
        flanes = [kf for kf in e['kernels'].values()
                  if kf.get('plan', {}).get('kind') == 'rowfold' and int(kf['plan'].get('lanes', 1) or 1) > 1]
        dev['rowfold_lanes'] = len(flanes)
        dev['rowfold_lanes_shfl'] = sum(kf.get('ptx', {}).get('shfl', 0) >= 5 for kf in flanes)
        ns = Scope(host=D(e['host']), device=D(e['device']), launch=D(e['launch']), dev=D(dev), arch=prof,
                   sm=e['sm'])
        hon = set(e.get('host_table', [])) | on
        for rung, it in INTENT.items():
            if not it.get('host'):
                continue
            state = 'off' if rung in off else 'on' if rung in hon or rung not in e.get('host_rungs', [rung]) else 'off'
            rows += [(prof, stem, '-', f'{rung}:{state}', *r) for r in run_intent(it, state, ns)]
    rows += runtime_moves(prof, fx, results)
    rows += choice_rows(prof, d, fx, results)
    print('arch\tstem\tkernel\trung\tcheck\texpected\trealized\tstatus', file=out)
    for r in rows:
        print('\t'.join(map(str, r)), file=out)
    st = Counter(r[-1] for r in rows)
    print(f"evidence: {len(rows)} checks, {st['MISMATCH']} mismatch, {st['n/a']} n/a", file=out)
    return rows


TRACE_KEYS = (('h2d', 'h2d_n'), ('bytes_in', 'h2d_b'), ('d2h', 'd2h_n'), ('bytes_out', 'd2h_b'),
              ('allocs', 'alloc_n'))


def traces(results):
    """stem -> the first `mapal_gpu_trace_total` of its probe runs (box_run.sh: `probe prog= face= arm=`)."""
    out = {}
    if not results or not os.path.exists(results):
        return out
    for line in open(results, errors='replace'):
        m = re.match(r'probe prog=(\S+) face=(\S+) arm=(\S+) .*mapal_gpu_trace_total (.*?) rc=', line)
        if not m:
            continue
        face = '' if m.group(2) in ('_default', '_exact') else m.group(2)
        stem = m.group(1) + face + ('' if m.group(3) == 'on' else '.' + m.group(3))
        out.setdefault(stem, kv(m.group(4)))
    return out


def runtime_moves(prof, fx, results):
    """S65 movement at runtime: the traced run's copies and allocations against the run the host IR and
    the source's trip counts predict (`host.move.runtime_*`). One more crossing than predicted is a copy
    no instruction in the host IR states — a finding."""
    rows = []
    for stem, t in sorted(traces(results).items()):
        # The null arm is the shipped build's object copied: the shipped stem's facts predict it.
        # (and an environment arm: the shipped binary)
        mv = fx.get(re.sub(r'\.(null|rules|pick\d+)$', '', stem), {}).get('host', {}).get('move', {})
        if 'runtime_h2d_n' not in mv:
            continue
        for tk, mk in TRACE_KEYS:
            want = mv['runtime_' + mk] + (mv.get('arena', 0) if tk == 'allocs' else 0)
            got = t.get(tk)
            st = 'n/a' if got is None else 'ok' if got == want else 'MISMATCH'
            rows.append((prof, stem, '-', 'by-ref:runtime', f'trace {tk} == predicted', f'{want}', f'{tk}={got}', st))
    return rows


# ---- diff (the re-judge instrument) -------------------------------------------------------------------
def margin(t):
    """`CHOICE_MARGIN` (mapal_choice.h): candidate 0 stays unless beaten by more than max(3 %, 1 us)."""
    return max(t * 3 // 100, 1000)


def wisdom_lines(path):
    """A wisdom file's decisions (`key site n pick reps ns.. kernels..`), the last line of a key winning."""
    out = {}
    for l in open(path, errors='replace'):
        f = l.rstrip('\n').split('\t')
        try:
            n = int(f[2])
            w = dict(key=f[0], site=kname(int(f[1], 16)), n=n, pick=int(f[3]), reps=int(f[4]),
                     ns=[int(x) for x in f[5:5 + n]], kernels=tuple(f[5 + n:5 + 2 * n]))
        except (IndexError, ValueError):
            continue
        if len(w['ns']) == n == len(w['kernels']) and w['pick'] < n:
            out[w['key']] = w
    return out


def argmin_pick(ns):
    """The decision the trial's minima make: the argmin when it beats candidate 0 beyond the margin, else 0."""
    w = min(range(len(ns)), key=ns.__getitem__)
    return w if w and ns[w] + margin(ns[0]) < ns[0] else 0


def choice_rows(prof, d, fx, results):
    """per-site-choice intent `choice` (intent.py), measured: every stored pick is the argmin of its minima beyond
    the margin; every AUTO run launched the stored pick (`op=site … src=wisdom`, box_run.sh `sites=`); AUTO <=
    RULES x (1 + null) per program and face (`on` = AUTO, `rules`; null: the `null` arm against `on`)."""
    rows = []
    if not results or not os.path.exists(results):
        return rows
    wis = {}
    for f in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(results)), 'wisdom', 'wisdom-*.tsv'))):
        wis.update(wisdom_lines(f))
    owner = defaultdict(list)  # (site, kernels) -> stems whose host IR holds that table
    for stem in fx:
        for t in site_tables(read(os.path.join(d, stem + '.host.ll')) or ''):
            owner[(t['site'], tuple(c['kernel'] for c in t['cands']))].append(stem)
    for w in wis.values():
        want = argmin_pick(w['ns'])
        for stem in owner.get((w['site'], w['kernels']), ['?']):
            rows.append((prof, stem, w['site'], 'choice:on', 'stored pick = measured argmin beyond max(3 %, 1 us)',
                         f'pick={want}', f"pick={w['pick']} ns={','.join(map(str, w['ns']))}",
                         'ok' if w['pick'] == want else 'MISMATCH'))
    ms = defaultdict(list)
    for line in open(results, errors='replace'):
        m = re.match(r'probe prog=(\S+) face=(\S+) arm=(\S+) rep=\d+ .*?out=\[[^\]]*iter ms=([\d.]+)', line)
        if not m:
            continue
        stem = m[1] + ('' if m[2] in ('_default', '_exact') else m[2])
        ms[(stem, m[3])].append(float(m[4]))
        sites = line.split(' sites=[', 1)[1].split(']')[0] if m[3] == 'on' and ' sites=[' in line else ''
        for key, pick, src in re.findall(r'([0-9a-f]{16}):(\d+):(\w+)', sites):
            stored = wis.get(key)
            rows.append((prof, stem, key, 'choice:on', 'AUTO launches the stored pick (src=wisdom)',
                         f"pick={stored['pick'] if stored else '?'} src=wisdom", f'pick={pick} src={src}',
                         'n/a' if not stored else 'ok' if (src, int(pick)) == ('wisdom', stored['pick']) else 'MISMATCH'))
    med = {k: sorted(v)[len(v) // 2] for k, v in ms.items()}
    for (stem, arm), rules in sorted(med.items()):
        if arm != 'rules' or (stem, 'on') not in med:
            continue
        auto, prog = med[(stem, 'on')], re.sub(r'_[cx]$', '', stem)
        null = abs(med[(prog, 'null')] / med[(prog, 'on')] - 1) if (prog, 'null') in med and (prog, 'on') in med \
            else None
        rows.append((prof, stem, '-', 'choice:on', 'AUTO <= RULES x (1 + null)', 'auto <= rules*(1+null)',
                     f'auto={auto:.4g} rules={rules:.4g} null={"n/a" if null is None else f"{null:.4f}"}',
                     'n/a' if null is None else 'ok' if auto <= rules * (1 + null) else 'MISMATCH'))
    return rows


def flat(x, pre=''):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from flat(v, f'{pre}.{k}' if pre else k)
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield pre, x


def diff(d, a, b):
    fx = json.load(open(os.path.join(d, 'evidence', 'facts.json')))
    ea, eb = fx[a], fx[b]
    out = []
    for scope in ('host', 'launch'):
        fa, fb = dict(flat(ea[scope])), dict(flat(eb[scope]))
        for k in sorted(set(fa) | set(fb)):
            if fa.get(k) != fb.get(k):
                out.append((f'{scope}.{k}', '-', fa.get(k), fb.get(k)))
    ka, kb = ea['kernels'], eb['kernels']
    # kernels pair by role: the launched kernel with the launched kernel (a module may carry an unlaunched
    # twin under the shipped name), then same name, then in plan order
    la, lb = ea['launch'].get('kernel'), eb['launch'].get('kernel')
    first = [(la, lb)] if la in ka and lb in kb and la != lb else []
    used_a, used_b = {x for x, _ in first}, {y for _, y in first}
    pairs = first + [(k, k) for k in ka if k in kb and k not in used_a and k not in used_b]
    rest_a = [k for k in ka if k not in kb and k not in used_a]
    rest_b = [k for k in kb if k not in ka and k not in used_b]
    for x, y in zip(rest_a, rest_b):
        pairs.append((x, y))
    for x, y in pairs:
        fa = dict(flat({s: ka[x].get(s, {}) for s in ('plan', 'ptx', 'ptxas', 'sass', 'occ')}))
        fb = dict(flat({s: kb[y].get(s, {}) for s in ('plan', 'ptx', 'ptxas', 'sass', 'occ')}))
        for k in sorted(set(fa) | set(fb)):
            if fa.get(k) != fb.get(k):
                out.append((k, x if x == y else f'{x}->{y}', fa.get(k), fb.get(k)))
    for k in rest_a[len(rest_b):]:
        out.append(('kernel', k, 'present', 'absent'))
    for k in rest_b[len(rest_a):]:
        out.append(('kernel', k, 'absent', 'present'))
    return out


# ---- jit (the box's driver image vs ptxas) --------------------------------------------------------------
def jit(cap, sass_files):
    local = {}
    for f in sass_files:
        for n, body in functions(open(f, errors='replace').read()).items():
            if n.startswith('mapal_'):
                local.setdefault(n, []).append([split(i)[1] for _, i in body])
    same = differ = 0
    for c in sorted(glob.glob(os.path.join(cap, 'module-*.sass'))):
        for n, body in functions(open(c, errors='replace').read()).items():
            if not n.startswith('mapal_') or n not in local:
                continue
            ok = [split(i)[1] for _, i in body] in local[n]
            same += ok
            differ += not ok
            if not ok:
                print(f'jit DIFF {n} {len(body)} insns')
    print(f'jit: {same} identical opcode sequences, {differ} differ')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'static':
        static(sys.argv[2], sys.argv[3])
    elif cmd == 'facts':
        facts(sys.argv[2], sys.argv[3])
    elif cmd == 'check':
        check(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    elif cmd == 'diff':
        for r in diff(sys.argv[2], sys.argv[3], sys.argv[4]):
            print('\t'.join(map(str, r)))
    elif cmd == 'jit':
        jit(sys.argv[2], sys.argv[3:])
    elif cmd == 'launch':
        launch_file(sys.argv[2])
    elif cmd == 'covered':
        sys.exit(0 if covered(sys.argv[2], sys.argv[3], sys.argv[4]) else 1)
    else:
        print(__doc__)
        sys.exit(2)
