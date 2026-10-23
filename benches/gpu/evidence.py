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
import subprocess
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from occ import occupancy  # noqa: E402
from ptx_static import CLS, entries  # noqa: E402
from sass_stats import accumulators, functions, split, stats, width  # noqa: E402

# Per part: the `facts` line its box prints (cudaDeviceProp); `occ.py`'s limits.
PARTS = {
    'rtx-4090': dict(sm='sm_89', sms=128, threads=1536, blocks=24, smem=102400),
    'rtx-5090': dict(sm='sm_120', sms=170, threads=1536, blocks=24, smem=102400),
    'h100-pcie': dict(sm='sm_90a', sms=114, threads=2048, blocks=32, smem=233472),
    'rtx-3090': dict(sm='sm_86', sms=82, threads=1536, blocks=16, smem=102400),
}
# The rungs a kernel's text reads (func/gpu.rs KERNEL_RUNGS); the rest are host rungs.
KERNEL_RUNGS = ('arch-rungs interleave unroll min-blocks warp prefetch mma async wide ldmatrix warp-fill idx32 affine '
                'pad wide-fma wgmma-hopper raster l2-gate pdl row-fit wave-gate grid-share panel-regs fill-rows '
                'cell-fill asyncstage splitk splitk-atomic streamk').split()
GPU_SYMS = ('alloc arena d2h_on d2h event free gen h2d init launch_l2 launchv launch pin sync tile unpin '
            'zero').split()
PROFILE_RS = os.path.join(HERE, '..', '..', 'crates', 'backends', 'llvm', 'src', 'profile.rs')


# ---- the measured rung table (profile.rs GPU_RUNGS_OFF) --------------------------------------------
def rows_off(path=PROFILE_RS):
    """[(profile, rung, fill, face, cite)] in table order."""
    t = open(path).read()
    t = t[t.index('pub const GPU_RUNGS_OFF'):]
    t = t[:t.index('\n];\n')]
    out = []
    for b in t.split('RungOff {')[1:]:
        g = lambda k: re.search(k + r':\s*(?:Fill::|Face::)?"?([\w-]+)"?', b).group(1)
        c = re.sub(r'\\\n\s*', '', re.search(r'cite:\s*"(.*)"\s*,\s*\}', b, re.S).group(1))
        out.append((g('profile'), g('rung'), g('fill'), g('face'), c))
    return out


def face_of(arith):
    return {'SeparateMulAdd': 'Exact', 'OrderedFma': 'Contract'}.get(arith, 'Tf32')


def covers(row_face, face):
    return row_face in ('Any', face) or (row_face == 'Contract' and face == 'Tf32')


def table_rungs(rows, prof, wps, face, arm):
    """The rungs `rungs_at` gives: every rung minus the part's rows at (fill, face), then the arm."""
    on, off = arm
    if 'arch-rungs' in off:
        r = set(KERNEL_RUNGS) | {n for p, n, *_ in rows}
    else:
        fill = 'Under' if wps < 2 else 'Filled'
        dead = {n for p, n, fl, fa, _ in rows if p == prof and covers(fa, face)
                and (fl in ('Any', fill) or (fl == 'Partial' and 1 <= wps < 2))}
        r = (set(KERNEL_RUNGS) | {n for p, n, *_ in rows}) - dead
    return sorted((r | on) - off)


def arm_of(stem):
    """`.off-a+b` / `.on-a`: the forced rungs of a bundle arm."""
    on, off = set(), set()
    for part in stem.split('.')[1:]:
        if part.startswith('off-'):
            off |= set(part[4:].split('+'))
        elif part.startswith('on-'):
            on |= set(part[3:].split('+'))
    return on, off


# ---- PTX -------------------------------------------------------------------------------------------
I64 = re.compile(r'^(add|sub|mul\.lo|mul\.wide|mad\.lo|mad\.wide|shl|shr)\.[usb]?(64|32)\b|^cvt\.[us]64\.[us]32')
EXTRA = [('atom_add', r'^atom\.(global\.)?add\.f'), ('red_add', r'^red\.(global\.)?add\.f'), ('mma', r'^mma\.sync'),
         ('ldmatrix', r'^ldmatrix'), ('wgmma', r'^wgmma\.mma_async'), ('wait_n', r'^cp\.async\.wait_group'),
         ('div_rem', r'^(div|rem)\.')]


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
    for name, _params, dirs, body in entries(text):
        lines = [x.strip() for x in body.splitlines()]
        f = dict(ptx_counts(lines))
        for d in ('maxntid', 'minnctapersm'):
            m = re.search(r'\.' + d + r'\s+(\d+)', dirs)
            f[d] = int(m.group(1)) if m else 0
        waits = [int(m.group(1)) for x in lines if (m := re.match(r'cp\.async\.wait_group\s+(\d+)', x))]
        f['wait_n_max'] = max(waits, default=-1)
        labels = {m[1]: i for i, x in enumerate(lines) if (m := re.match(r'^(\$L__\w+):', x))}
        best = None
        for i, x in enumerate(lines):
            m = re.search(r'bra(?:\.uni)?\s+(\$L__\w+);', x)
            if m and m[1] in labels and labels[m[1]] < i:
                seg = lines[labels[m[1]]:i + 1]
                c = ptx_counts(seg)
                key = (c['fma_f32'] + c['mul_f32'] + c['mma'] + c['wgmma'], -len(seg))
                if best is None or key > best[0]:
                    best = (key, seg, c)
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


def sass_facts(body):
    """Whole-function opcode families + the hot loop (the backward-branch body with the most MAC/MMA)."""
    c, w = stats(body)
    f = {k: 0 for k in ('ffma', 'fmul', 'fadd', 'hmma', 'hgmma', 'ldg', 'ldgsts', 'lds', 'ldsm', 'sts', 'stg', 'bar',
                        'red', 'atom', 'atomg', 'redg', 'ldl', 'stl')}
    f.update({k.lower(): v for k, v in c.items()})
    f.update({k.lower().replace('.', '_'): v for k, v in w.items()})
    f['insns'] = len(body)
    addr = [a for a, _ in body]
    best = None
    for i, (a, ins) in enumerate(body):
        _, op, rest = split(ins)
        m = re.search(r'0x([0-9a-f]+)', rest) if op.split('.')[0] == 'BRA' else None
        if not m or int(m.group(1), 16) >= a or int(m.group(1), 16) not in addr:
            continue
        loop = body[addr.index(int(m.group(1), 16)):i + 1]
        ops = Counter(split(x)[1].split('.')[0] for _, x in loop)
        key = (sum(ops[k] for k in MAC), -len(loop))
        if best is None or key > best[0]:
            best = (key, loop)
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
            insns=len(loop), ffma=lc['FFMA'], fmul=lc['FMUL'], fadd=lc['FADD'], hmma=lc['HMMA'], hgmma=lc['HGMMA'],
            ldg=lc['LDG'], ldgsts=lc['LDGSTS'], lds=lc['LDS'], ldsm=lc['LDSM'], sts=lc['STS'], bar=lc['BAR'],
            bra=lc['BRA'], ldg128=lw.get('LDG.128', 0), lds128=lw.get('LDS.128', 0),
            imad_wide=sum(o.startswith('IMAD.WIDE') for o in ops), mac=mac, acc=acc,
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
        f['par'] = len(re.findall(r'@mapal_par_begin\(', ll))
        f['sme'] = len(re.findall(r'@mapal_sme_panel', ll))
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
        m = re.search(r'call \w+ @mapal_gpu_(h2d|d2h|d2h_on|alloc|arena|tile)\((.*)\)', x)
        if not m:
            continue
        op, args = m.group(1), [a.split()[-1] for a in m.group(2).split(',')]
        last = int(args[-1]) if re.fullmatch(r'\d+', args[-1]) else None
        if op == 'h2d':
            c['h2d_n'] += 1
            c['h2d_b'] += last or 0
        elif op in ('d2h', 'd2h_on'):
            c['d2h_n'] += 1
            c['d2h_b'] += last or 0
        elif op == 'tile':  # the S52 round trip: two copies in, one out
            c['h2d_n'] += 2
            c['d2h_n'] += 1
        elif op == 'arena':  # one allocation per function, on its first ask
            c['arena_n'] += 1
        else:
            c['alloc_n'] += 1
    return c


MOVE_KEYS = ('copy_n', 'copy_b', 'h2d_n', 'h2d_b', 'd2h_n', 'd2h_b', 'alloc_n')


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
        arena += '@mapal_gpu_arena(' in fn
        bbs = block_list(fn)
        index = {lab: i for i, (lab, _) in enumerate(bbs)}
        inside = set()
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
        for i, (_, lines) in enumerate(bbs):
            if (id(fn), i) not in inside:
                once += block_moves(lines)
    f = dict(loops=len(loops), per_loop=loops, arena=arena)
    for k in MOVE_KEYS:
        f[k] = sum(x[k] for x in loops)
    for k in ('h2d', 'd2h', 'host_copy'):
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
    return out


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
    if todo:
        lst = os.path.join(ev, 'todo.txt')
        open(lst, 'w').write('\n'.join(todo) + '\n')
        script = (f'cd "{os.path.abspath(ev)}" && ptxas --version | tail -1 > ptxas.version && '
                  f'xargs -P 8 -I@ sh -c \'ptxas -arch={sm} -O3 -v ../@.ptx -o @.cubin 2> @.ptxas; '
                  f'cuobjdump -sass @.cubin > @.sass 2>&1; rm -f @.cubin\' < todo.txt')
        subprocess.run([os.path.join(HERE, 'cudatool.sh'), 'bash', '-c', script], check=True)
        os.remove(lst)
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


# ---- facts -------------------------------------------------------------------------------------------
def read(path):
    if os.path.exists(path):
        return open(path, errors='replace').read()
    if os.path.exists(path + '.gz'):
        return gzip.open(path + '.gz', 'rt', errors='replace').read()
    return None


def facts(d, sm):
    prof = os.path.basename(os.path.normpath(d))
    part = PARTS.get(prof, {})
    lim = dict(threads=part.get('threads', 1536), blocks=part.get('blocks', 16), regs=65536, reg_unit=256,
               max_warps=part.get('threads', 1536) // 32, smem=part.get('smem', 102400), smem_reserved=1024)
    ev = os.path.join(d, 'evidence')
    rows = rows_off() if os.path.exists(PROFILE_RS) else []
    stems = sorted({os.path.basename(p).split('.host.o')[0] for p in glob.glob(os.path.join(d, '*.host.o'))}
                   | {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(d, '*.ptx'))})
    out = {}
    for stem in stems:
        base = os.path.join(d, stem)
        src = stem
        al = os.path.join(ev, stem + '.alias')
        if os.path.exists(al):
            src = json.load(open(al))['same_as']
        dev = read(base + '.dev.ll') or read(os.path.join(d, src + '.dev.ll')) or ''
        plans = {p['kernel']: p for p in (kv(m) for m in re.findall(r'^; mapal\.plan (.*)$', dev, re.M))}
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
        if rows:  # the host rungs the table ships at their fill (GPU_RESIDENT_BLOCKS: Filled) and face
            hr = {n for _, n, *_ in rows} - set(KERNEL_RUNGS)
            face = 'Contract' if e['device'].get('fma', host.get('fma', 1)) == 1 else 'Exact'
            e['host_rungs'] = sorted(hr)
            e['host_table'] = [r for r in table_rungs(rows, prof, 2, face, arm) if r not in KERNEL_RUNGS]
        hll = read(base + '.host.ll') or ''
        lau = launches(hll)
        for k in sorted(set(plans) | set(pf)):
            p = plans.get(k, {})
            kf = dict(plan=p, ptx=pf.get(k, {}), ptxas=pa.get(k, {}), mod=dict(launched=launched(k, hll)),
                      launch=lau.get(kid(k), {}),
                      sass=sass_facts(sf[k]) if k in sf else {},
                      remarks=dict(opt=remark_facts(rem_opt, k), llc=remark_facts(rem_llc, k)))
            if 'face' in p and rows:
                kf['table'] = dict(rungs_on=[r for r in table_rungs(rows, prof, p['wps'], face_of(p['face']), arm)
                                             if r in KERNEL_RUNGS])
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


SAFE = dict(ceil=math.ceil, set=set, max=max, min=min, len=len, sum=sum, any=any, all=all, abs=abs)
NAMES = re.compile(r'\b(plan|host|launch|ptx|ptxas|sass|occ|remarks|exec|jit|table|device|mod)(\.\w+)+')


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
            out[(m.group(1), m.group(2))] = dict(smem=int(m.group(3)), regs=int(m.group(4)), local=int(m.group(5)))
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


KINDS = ('tile', 'naive', 'mma', 'wgmma', 'reduce', 'rowfold', 'gen', 'cell')


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
            ns = Scope({x: D(kf.get(x, {})) for x in ('plan', 'ptx', 'ptxas', 'sass', 'occ', 'table', 'mod')})
            ns.update(host=D(e['host']), device=D(e['device']), remarks=D(kf.get('remarks', {})),
                      launch=D(kf.get('launch') or (e['launch'] if e['launch'].get('kernel') == k else {})),
                      exec=D(ex.get((stem, k), {})), arch=prof, sm=e['sm'])
            kon = set(kf.get('plan', {}).get('rungs_on', [])) | on
            for rung, it in INTENT.items():
                if it.get('host'):
                    continue
                state = 'off' if rung in off else 'on' if rung in kon or rung.startswith(('face', 'exec', 'module')) else 'off'
                rows += [(prof, stem, k, f'{rung}:{state}', *r) for r in run_intent(it, state, ns)]
        # host rungs: one row per stem (kernel `-`)
        if not e['host'].get('calls'):
            continue
        kinds = Counter(kf.get('plan', {}).get('kind', '?') for kf in e['kernels'].values())
        dev = {x: kinds.get(x, 0) for x in KINDS}
        dev['epi'] = sum(kf.get('plan', {}).get('epi', 0) == 1 for kf in e['kernels'].values())
        ns = Scope(host=D(e['host']), device=D(e['device']), launch=D(e['launch']), dev=D(dev), arch=prof,
                   sm=e['sm'])
        hon = set(e.get('host_table', [])) | on
        for rung, it in INTENT.items():
            if not it.get('host'):
                continue
            state = 'off' if rung in off else 'on' if rung in hon or rung not in e.get('host_rungs', [rung]) else 'off'
            rows += [(prof, stem, '-', f'{rung}:{state}', *r) for r in run_intent(it, state, ns)]
    rows += runtime_moves(prof, fx, results)
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
        mv = fx.get(stem, {}).get('host', {}).get('move', {})
        if 'runtime_h2d_n' not in mv:
            continue
        for tk, mk in TRACE_KEYS:
            want = mv['runtime_' + mk] + (mv.get('arena', 0) if tk == 'allocs' else 0)
            got = t.get(tk)
            st = 'n/a' if got is None else 'ok' if got == want else 'MISMATCH'
            rows.append((prof, stem, '-', 'by-ref:runtime', f'trace {tk} == predicted', f'{want}', f'{tk}={got}', st))
    return rows


# ---- diff (the re-judge instrument) -------------------------------------------------------------------
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
    else:
        print(__doc__)
        sys.exit(2)
