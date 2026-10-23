"""S62 rtx-5090/tools/ptx_static.py (S64: importable; evidence.py reuses CLS and entries). Static PTX facts per entry: directives, shared decls, instruction classes, loop shape."""
import re, sys, collections
CLS = [('fma.f32', r'^fma\.rn\.f32'), ('mul.f32', r'^mul\.rn\.f32|^mul\.f32'), ('add.f32', r'^add\.rn\.f32|^add\.f32'),
       ('ld.global.v4', r'^ld\.global(\.nc)?(\.[a-z]+)*\.v4'), ('ld.global.v2', r'^ld\.global(\.nc)?(\.[a-z]+)*\.v2'),
       ('ld.global', r'^ld\.global'), ('st.global', r'^st\.global'), ('ld.shared.v4', r'^ld\.shared\.v4'),
       ('ld.shared', r'^ld\.shared'), ('st.shared', r'^st\.shared'), ('cp.async', r'^cp\.async\.c[ag]'),
       ('cp.async.wait', r'^cp\.async\.(wait|commit)'), ('bar', r'^bar\.|^barrier'), ('shfl', r'^shfl'),
       ('setp', r'^setp'), ('bra', r'^bra'), ('griddep', r'^griddepcontrol')]
def entries(text):
    for m in re.finditer(r'\.visible \.entry (\w+)\((.*?)\)\s*((?:\.\w+[^{]*)?)\{(.*?)\n\}', text, re.S):
        yield m[1], m[2], m[3], m[4]
def main(paths):
    for path in paths:
        text = open(path).read()
        for name, params, dirs, body in entries(text):
            c = collections.Counter()
            lines = [l.strip() for l in body.splitlines()]
            for l in lines:
                ins = re.sub(r'^@!?%p\d+\s+', '', l)
                for k, rx in CLS:
                    if re.match(rx, ins):
                        c[k] += 1; break
            labels = {m[1]: i for i, l in enumerate(lines) if (m := re.match(r'^(\$L__\w+):', l))}
            loops = []
            for i, l in enumerate(lines):
                m = re.search(r'bra(?:\.uni)?\s+(\$L__\w+);', l)
                if m and m[1] in labels and labels[m[1]] < i:
                    seg = lines[labels[m[1]]:i]
                    f = sum(bool(re.match(r'^(@!?%p\d+\s+)?fma\.rn\.f32', s)) for s in seg)
                    a = sum(bool(re.match(r'^(@!?%p\d+\s+)?add\.rn\.f32', s)) for s in seg)
                    g = sum(bool(re.match(r'^(@!?%p\d+\s+)?(ld\.global|cp\.async\.c)', s)) for s in seg)
                    sh = sum(bool(re.match(r'^(@!?%p\d+\s+)?ld\.shared', s)) for s in seg)
                    loops.append((f + a, f, a, g, sh, len(seg)))
            hot = max(loops, default=(0,) * 6)
            smem = re.findall(r'\.shared \.align \d+ \.b8 (\w+)\[(\d+)\]', text) + re.findall(r'\.extern \.shared[^;]*', text)
            d = ' '.join(re.findall(r'\.(maxntid|reqntid|minnctapersm|maxnreg)\s+([\d, ]+)', dirs) and
                         [f'{a}={b.strip()}' for a, b in re.findall(r'\.(maxntid|reqntid|minnctapersm|maxnreg)\s+([\d, ]+)', dirs)])
            nparams = len([p for p in params.split(',') if p.strip()])
            print(f"{path.split('/')[-1]}\t{name}\tparams={nparams}\t{d or '-'}\tshared={smem or '-'}\t" +
                  ' '.join(f'{k}={c[k]}' for k, _ in CLS if c[k]) +
                  f"\tloops={len(loops)} hot_loop: mac={hot[0]} (fma={hot[1]} add={hot[2]}) glb_loads={hot[3]} lds={hot[4]} lines={hot[5]}")


if __name__ == "__main__":
    main(sys.argv[1:])
