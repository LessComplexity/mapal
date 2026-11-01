#!/usr/bin/env python3
"""Static SASS mix of every function in `cuobjdump -sass` output.

usage: sass_mix.py <file.sass> [function-substring]
Prints one JSON object per function: whole-function opcode counts and, for the
hot loop (the backward-branch region with the most FFMA), its counts, FFMA per
global load, global-load bytes per iteration, and the distinct FFMA destination
registers (accumulators per thread). Static counts only: no dynamic weighting.
"""
import json, re, sys
from collections import Counter

INS = re.compile(r'/\*([0-9a-f]{4,})\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)\s*([^;]*);')
FUN = re.compile(r'^\s*Function\s*:\s*(\S+)')
WIDTH = {'.128': 16, '.64': 8, '.U8': 1, '.S8': 1, '.U16': 2, '.S16': 2}


def width(op):
    for k, v in WIDTH.items():
        if k in op:
            return v
    return 4


def fam(op):
    base = op.split('.')[0]
    return base


def counts(ins):
    c = Counter()
    ldg_bytes = 0
    lds_bytes = 0
    for _, pred, op, args in ins:
        b = fam(op)
        c[b] += 1
        if b == 'LDG':
            ldg_bytes += width(op)
            c['LDG' + ('.128' if '.128' in op else '.64' if '.64' in op else '.32')] += 1
        if b == 'LDS':
            lds_bytes += width(op)
            c['LDS' + ('.128' if '.128' in op else '.64' if '.64' in op else '.32')] += 1
        if b == 'STG':
            c['STG' + ('.128' if '.128' in op else '.64' if '.64' in op else '.32')] += 1
        if pred:
            c['predicated'] += 1
    return c, ldg_bytes, lds_bytes


def summarize(name, ins):
    c, gb, sb = counts(ins)
    addr = [int(a, 16) for a, *_ in ins]
    loops = []
    for i, (a, pred, op, args) in enumerate(ins):
        if fam(op) != 'BRA':
            continue
        m = re.search(r'0x([0-9a-f]+)', args)
        if not m:
            continue
        t = int(m.group(1), 16)
        if t <= addr[i]:
            body = [x for x in ins if t <= int(x[0], 16) <= addr[i]]
            loops.append((t, addr[i], body))
    hot = None
    if loops:
        t, e, body = max(loops, key=lambda l: (sum(Counter(fam(x[2]) for x in l[2])[k] for k in ('FFMA', 'FMUL', 'FADD')), -len(l[2])))
        lc, lgb, lsb = counts(body)
        # Independent FFMA chains: an FFMA whose addend was not last written by an
        # FFMA in this body starts a chain (register renaming across unrolled
        # steps keeps one chain one chain).
        acc = 0
        writer = {}
        for _, _, op, args in body:
            regs = [r.strip().replace('.reuse', '') for r in args.split(',')]
            if fam(op) == 'FFMA' and len(regs) == 4:
                if writer.get(regs[3]) != 'FFMA':
                    acc += 1
            if fam(op) == 'FADD' and len(regs) == 3:  # Exact: acc = acc + p, p from an FMUL
                a, b = regs[1].lstrip('-|'), regs[2].lstrip('-|')
                if writer.get(a) == 'FMUL' and writer.get(b) not in ('FADD',):
                    acc += 1
                elif writer.get(b) == 'FMUL' and writer.get(a) not in ('FADD',):
                    acc += 1
            if regs and re.match(r'^R\d+$', regs[0]) and not fam(op).startswith(('ST', 'BAR', 'BRA', 'RED', 'ATOM')):
                writer[regs[0]] = fam(op)
        ffma = lc['FFMA'] + lc['FMUL']  # multiply-accumulate steps: FFMA, or FMUL+FADD on Exact
        loads = lc['LDG'] + lc['LDGSTS']
        hot = {
            'start': hex(t), 'end': hex(e), 'instructions': len(body),
            'FFMA': ffma, 'FADD': lc['FADD'], 'FMUL': lc['FMUL'],
            'LDG': lc['LDG'], 'LDG.128': lc['LDG.128'], 'LDG.64': lc['LDG.64'], 'LDG.32': lc['LDG.32'],
            'LDGSTS': lc['LDGSTS'], 'LDS': lc['LDS'], 'LDS.128': lc['LDS.128'], 'LDS.64': lc['LDS.64'],
            'STS': lc['STS'], 'BAR': lc['BAR'], 'SHFL': lc['SHFL'], 'DEPBAR': lc['DEPBAR'],
            'BRA': lc['BRA'], 'ISETP': lc['ISETP'], 'LDL': lc['LDL'], 'STL': lc['STL'], 'predicated': lc['predicated'],
            'IMAD+IADD3+LEA': lc['IMAD'] + lc['IADD3'] + lc['LEA'],
            'ldg_bytes_per_iter_thread': lgb, 'lds_bytes_per_iter_thread': lsb,
            'ffma_per_global_load': round(ffma / loads, 2) if loads else None,
            'ffma_per_global_byte': round(ffma / lgb, 3) if lgb else None,
            'ffma_per_shared_load': round(ffma / lc['LDS'], 2) if lc['LDS'] else None,
            'accumulator_regs': acc,
            'mac_share': round(ffma / len(body), 3), 'mac_steps': ffma,
        }
    keys = ['FFMA', 'FADD', 'FMUL', 'LDG', 'LDG.128', 'LDG.64', 'LDG.32', 'LDGSTS', 'LDS', 'STS', 'STG',
            'BAR', 'SHFL', 'DEPBAR', 'BRA', 'ISETP', 'predicated', 'RED', 'ATOM', 'MEMBAR', 'LDL', 'STL']
    return {'function': name, 'instructions': len(ins), 'code_bytes': 16 * len(ins),
            **{k: c[k] for k in keys}, 'backward_branches': len(loops), 'hot_loop': hot}


def main():
    text = open(sys.argv[1], errors='replace').read()
    want = sys.argv[2] if len(sys.argv) > 2 else ''
    funs = {}
    cur = None
    for line in text.splitlines():
        m = FUN.match(line)
        if m:
            cur = m.group(1)
            funs[cur] = []
            continue
        if cur is None:
            continue
        m = INS.search(line)
        if m:
            funs[cur].append(m.groups())
    for name, ins in funs.items():
        if want in name and ins:
            print(json.dumps(summarize(name, ins)))


if __name__ == '__main__':
    main()
