"""Static SASS facts per function (cuobjdump -sass or nvdisasm --print-code text):
opcode mix, the hottest backward-branch loop (most FFMA+FADD+FMUL), its loads, accumulators
(distinct FFMA/FADD destination registers in that loop) and k-steps per trip.
usage: python3 sass_static.py file.sass...  -> TSV on stdout"""
import collections, re, sys

INS = re.compile(r'/\*([0-9a-f]{4,})\*/\s+(@!?U?P[T0-9]+\s+)?([A-Z][A-Z0-9_.]*)\s*([^;]*);')
FUN = re.compile(r'(?:Function\s*:\s*|\.text\.)(\S+?)(?::|\s*$)', re.M)
KEYS = ['FFMA', 'FMUL', 'FADD', 'LDG', 'LDG.128', 'LDG.64', 'LDGSTS', 'LDS', 'LDS.128', 'LDS.64', 'LDSM', 'STS', 'STG',
        'BAR', 'SHFL', 'BRA', 'ISETP', 'DEPBAR', 'HMMA', 'IMMA', 'total']


def width(op):
    m = re.search(r'\.(128|64)\b', op)
    return m[1] if m else '32'


LABELS = {}


def split_functions(text):
    """(name, [(addr, pred, op, args)]) per function in the listing; nvdisasm `.L_x_N` branch
    labels are resolved to the address of the instruction that follows them."""
    out, name, cur, pend = [], None, [], []
    for line in text.splitlines():
        f = re.search(r'Function\s*:\s*(\S+)', line) or re.match(r'\s*\.text\.(\S+?):', line)
        if f:
            if name and cur: out.append((name, cur))
            name, cur = f[1], []
            continue
        lab = re.match(r'\s*(\.L_x_\d+):', line)
        if lab: pend.append(lab[1])
        m = INS.search(line)
        if m and name:
            for x in pend: LABELS[(name, x)] = int(m[1], 16)
            pend = []
            args = re.sub(r'`?\((\.L_x_\d+)\)`?', lambda q: '@' + q[1], m[4])
            cur.append((int(m[1], 16), m[2] or '', m[3], args))
    if name and cur: out.append((name, cur))
    return out


def facts(ins, name=None):
    c = collections.Counter()
    for _, _, op, _ in ins:
        base = op.split('.')[0]
        c['total'] += 1
        if base in ('FFMA', 'FMUL', 'FADD', 'LDGSTS', 'STS', 'STG', 'BAR', 'SHFL', 'BRA', 'ISETP', 'DEPBAR', 'HMMA', 'IMMA', 'LDSM'):
            c[base] += 1
        if base in ('LDG', 'LDS'):
            c[base] += 1
            if width(op) != '32': c[f'{base}.{width(op)}'] += 1
    loops = []
    addr = [a for a, *_ in ins]
    for i, (a, pred, op, args) in enumerate(ins):
        if op.startswith('BRA'):
            t = re.search(r'0x([0-9a-f]+)', args)
            lab = re.search(r'@(\.L_x_\d+)', args)
            tgt = int(t[1], 16) if t else LABELS.get((name, lab[1])) if lab else None
            if tgt is not None and tgt < a:
                lo = tgt
                body = [x for x in ins if lo <= x[0] <= a]
                mac = [x for x in body if x[2].split('.')[0] in ('FFMA', 'FADD')]
                # accumulator chains: an FFMA/FADD whose accumulator input (FFMA c, FADD either
                # source) was not written by an FFMA/FADD earlier in this trip starts a chain.
                writer, starts = {}, 0
                for x in body:
                    ops = [re.sub(r'\.reuse|^[-|]+', '', o.strip()) for o in re.split(r'[&?]', x[3])[0].split(',')]
                    b = x[2].split('.')[0]
                    if b in ('FFMA', 'FADD'):
                        src = ops[3:4] if b == 'FFMA' else ops[1:3]
                        if not any(writer.get(r) == 'mac' for r in src): starts += 1
                    if ops and ops[0].startswith('R'):
                        writer[ops[0]] = 'mac' if b in ('FFMA', 'FADD') else 'other'
                accs = range(starts)
                g = sum(x[2].split('.')[0] in ('LDG', 'LDGSTS') for x in body)
                gb = sum(int(width(x[2])) // 8 for x in body if x[2].split('.')[0] in ('LDG', 'LDGSTS'))
                s = sum(x[2].split('.')[0] in ('LDS', 'LDSM') for x in body)
                loops.append(dict(span=len(body), ffma=sum(x[2].startswith('FFMA') for x in body),
                                  fadd=sum(x[2].startswith('FADD') for x in body),
                                  fmul=sum(x[2].startswith('FMUL') for x in body), glb=g, glb_bytes=gb, lds=s,
                                  bar=sum(x[2].startswith('BAR') for x in body), accs=len(accs)))
    hot = max(loops, key=lambda l: l['ffma'] + l['fadd'] + l['fmul'], default=None)
    return c, loops, hot


if __name__ == '__main__':
    print('file\tfunction\t' + '\t'.join(KEYS) + '\tloops\thot_span\thot_ffma\thot_fadd\thot_fmul\thot_glb\thot_glb_bytes'
          '\thot_lds\thot_bar\thot_accs\tmac_per_glb\tksteps_per_trip')
    for path in sys.argv[1:]:
        for name, ins in split_functions(open(path, errors='replace').read()):
            c, loops, h = facts(ins, name)
            h = h or dict(span=0, ffma=0, fadd=0, fmul=0, glb=0, glb_bytes=0, lds=0, bar=0, accs=0)
            mac = h['ffma'] + h['fadd']
            print(f"{path.split('/')[-1]}\t{name}\t" + '\t'.join(str(c[k]) for k in KEYS) +
                  f"\t{len(loops)}\t{h['span']}\t{h['ffma']}\t{h['fadd']}\t{h['fmul']}\t{h['glb']}\t{h['glb_bytes']}\t{h['lds']}"
                  f"\t{h['bar']}\t{h['accs']}\t{(mac / h['glb']) if h['glb'] else 0:.1f}\t{(mac / h['accs']) if h['accs'] else 0:.1f}")
