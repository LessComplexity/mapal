#!/usr/bin/env python3
"""Static SASS mix of one function in a `cuobjdump -sass` listing (sm_90).

usage: sass_stats.py <listing.sass> <function-name-substring> [--json]
Prints, for the whole function and for its hot loop (the backward-branch body with the most
FFMA), opcode-family counts, global-load widths, FFMA per global load, unroll depth and
accumulators (FFMA chains whose addend is live into the loop body). Static counts only: no
dynamic instruction, traffic or stall claim follows from them.
"""
import json
import re
import sys
from collections import Counter

INSN = re.compile(r'^\s*/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;')
FUNC = re.compile(r'Function : (\S+)')


NVSEC = re.compile(r'^//-+ \.text\.(\S+) -+')
LABEL = re.compile(r'^(\.L_x_\d+):')


def functions(text):
    """cuobjdump -sass (`Function : name`) or nvdisasm (`.text.name` sections, `.L_x_N` labels)."""
    out, name, body, labels, pending = {}, None, [], {}, []

    def close():
        if name:
            fix = lambda ins: re.sub(r'`?\((\.L_x_\d+)\)', lambda m: hex(labels.get(m.group(1), 0)), ins)
            out[name] = [(a, fix(i)) for a, i in body]

    for line in text.splitlines():
        m = FUNC.search(line) or NVSEC.match(line)
        if m:
            close()
            name, body, labels, pending = m.group(1), [], {}, []
            continue
        m = LABEL.match(line)
        if m:
            pending.append(m.group(1))
            continue
        m = INSN.match(line)
        if m and name:
            a = int(m.group(1), 16)
            for l in pending:
                labels[l] = a
            pending = []
            body.append((a, m.group(2)))
    close()
    return out


def split(ins):
    pred = ''
    if ins.startswith('@'):
        pred, ins = ins.split(None, 1)
    parts = ins.split(None, 1)
    return pred, parts[0], (parts[1] if len(parts) > 1 else '')


def width(op):
    for w in ('128', '64', 'U16', 'S16', 'U8', 'S8'):
        if '.' + w in op:
            return {'128': 128, '64': 64}.get(w, 16 if '16' in w else 8)
    return 32


def family(op):
    b = op.split('.')[0]
    if b in ('FFMA', 'FADD', 'FMUL', 'FMNMX', 'FSETP', 'FSEL', 'FCHK', 'MUFU', 'HFMA2'):
        return 'HFMA2_MOV' if op.startswith('HFMA2.MMA') else b
    if b in ('LDG', 'LDGSTS', 'LDS', 'LDSM', 'STS', 'STG', 'LDL', 'STL', 'LDC', 'ULDC', 'LD', 'ST', 'RED', 'ATOM', 'ATOMG',
             'REDG', 'UBLKCP', 'UTMALDG', 'LDGDEPBAR', 'DEPBAR', 'BAR', 'SHFL', 'BRA', 'EXIT', 'ISETP', 'PLOP3', 'BSSY',
             'BSYNC', 'WARPSYNC', 'NOP', 'S2R', 'S2UR', 'CS2R', 'MOV', 'UMOV', 'IMAD', 'IADD3', 'LEA', 'LOP3', 'SHF', 'SEL',
             'HMMA', 'IMMA', 'DMMA', 'HGMMA', 'MEMBAR', 'ERRBAR', 'CCTL', 'SYNCS', 'ELECT', 'R2UR', 'VOTE', 'P2R', 'R2P'):
        return b
    if b.startswith('U'):
        return 'UNIFORM'
    return 'OTHER'


def stats(body):
    c, w = Counter(), Counter()
    for _, ins in body:
        _, op, _ = split(ins)
        f = family(op)
        c[f] += 1
        if f in ('LDG', 'LDGSTS', 'LDS', 'STS', 'STG'):
            w[f'{f}.{width(op)}'] += 1
    return c, w


def regs(s):
    return re.findall(r'\bR\d+\b', s)


def hot_loop(body):
    addr = [a for a, _ in body]
    best = None
    for i, (a, ins) in enumerate(body):
        _, op, rest = split(ins)
        if op.split('.')[0] != 'BRA':
            continue
        m = re.search(r'0x([0-9a-f]+)', rest)
        if not m:
            continue
        t = int(m.group(1), 16)
        if t >= a or t not in addr:
            continue
        j = addr.index(t)
        loop = body[j:i + 1]
        nf = mac(loop)
        key = (nf, -len(loop))
        if best is None or key > best[0]:
            best = (key, loop, t, a)
    return best


def mac(loop):
    """Multiply-accumulate steps: FFMA, plus FMUL+FADD pairs (the Exact face's separate rounding)."""
    c = Counter(split(x)[1].split('.')[0] for _, x in loop)
    return c['FFMA'] + min(c['FMUL'], c['FADD'])


def accumulators(loop):
    """Accumulation chains: FFMA addends / FADD sources read before any write in the body (live-in)."""
    written, heads = set(), set()
    for _, ins in loop:
        _, op, rest = split(ins)
        rs = regs(rest)
        if op.startswith('FFMA') and len(rs) >= 4 and rs[3] not in written:
            heads.add(rs[3])
        if op.startswith('FADD') and len(rs) >= 3:
            heads.update(r for r in rs[1:3] if r not in written)
        if rs:
            written.add(rs[0])
    return len(heads)


def analyse(listing, want):
    fns = functions(open(listing, errors='replace').read())
    names = [n for n in fns if want in n]
    if not names:
        raise SystemExit(f'no function matching {want} in {listing}')
    name = min(names, key=len)
    body = fns[name]
    c, w = stats(body)
    r = {'function': name, 'insns': len(body), 'mix': dict(c), 'widths': dict(w)}
    hl = hot_loop(body)
    if hl:
        (_, loop, t, a) = hl
        lc, lw = stats(loop)
        gl = lc['LDG'] + lc['LDGSTS']
        gbytes = sum(int(k.split('.')[1]) // 8 * v for k, v in lw.items() if k.split('.')[0] in ('LDG', 'LDGSTS'))
        acc = accumulators(loop)
        m = mac(loop)
        ops = [split(i)[1].split('.')[0] for _, i in loop]
        ismac = [o in ('FFMA', 'FMUL', 'FADD') for o in ops]
        fm = ismac.index(True) if any(ismac) else len(ops)
        lm = len(ops) - 1 - ismac[::-1].index(True) if any(ismac) else -1
        isg = [o in ('LDG', 'LDGSTS') for o in ops]
        # where the next tile's global loads sit relative to this tile's math: before the first MAC (prefetch
        # distance = one tile of math), interleaved, or after the last MAC (consumed right after the back-edge)
        g_pos = dict(before_math=sum(isg[:fm]), within_math=sum(isg[fm:lm + 1]), after_math=sum(isg[lm + 1:]))
        # register-staged global loads: MAC instructions issued between each LDG and the first reader of its
        # destination (cyclic over the back-edge) = the math available to hide that load's latency in one warp
        cover = []
        n = len(loop)
        for i, (_, ins) in enumerate(loop):
            _, op, rest = split(ins)
            if not op.startswith('LDG.') and op != 'LDG':
                continue
            rs = regs(rest)
            if not rs:
                continue
            base_r = int(rs[0][1:]); w = max(1, width(op) // 32)
            dst = {f'R{base_r + j}' for j in range(w)}
            macs = 0
            for d in range(1, n + 1):
                _, op2, rest2 = split(loop[(i + d) % n][1])
                srcs = regs(rest2)[1:] if not op2.startswith(('STS', 'STG', 'ST')) else regs(rest2)
                if dst & set(srcs):
                    break
                macs += op2.split('.')[0] in ('FFMA', 'FMUL')
            cover.append(macs)
        r['loop'] = {'global_load_position': g_pos,
                     'ldg_cover_mac': (min(cover), sorted(cover)[len(cover) // 2]) if cover else None,'range': f'{t:#x}-{a:#x}', 'insns': len(loop), 'mix': dict(lc), 'widths': dict(lw),
                     'ffma': lc['FFMA'], 'mac': m, 'global_loads': gl, 'global_bytes': gbytes,
                     'mac_per_global_load': round(m / gl, 2) if gl else None,
                     'mac_per_global_word': round(m / (gbytes / 4), 2) if gbytes else None,
                     'mac_per_lds': round(m / lc['LDS'], 2) if lc['LDS'] else None,
                     'accumulators': acc, 'unroll_k': round(m / acc, 2) if acc else None,
                     'mac_share': round(m / len(loop), 3)}
    return r


if __name__ == '__main__':
    r = analyse(sys.argv[1], sys.argv[2])
    print(json.dumps(r, indent=None if '--json' in sys.argv else 1))
