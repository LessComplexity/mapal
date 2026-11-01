"""S63 (from the S62 disassembly tools): <stem>.k for mlp_bench.cu: each mapal_gpu_launchv of a pt_rm program's host IR, with its
argument roles (gens in source order are x w1 w2 bias; other slots hh then y)."""
import re, sys
for host in sys.argv[1:]:
    src = open(host).read().split('define internal void @mapal_main()')[1].split('\n}\n')[0]
    val, slot, role, arr, out = {}, {}, {}, {}, []
    gens = iter(['x', 'w1', 'w2', 'bias']); fresh = iter(['hh', 'y'])
    for m in re.finditer(r'call void @mapal_gpu_gen\(i64 -?\d+, i64 (%t\d+), i64 \d+\)', src):
        role[m[1]] = next(gens)
    for line in src.splitlines():
        line = line.strip()
        if m := re.match(r'store i64 (%t\d+), ptr (%d\d+)$', line):
            if m[1] not in role:
                role[m[1]] = next(fresh)
            slot[m[2]] = role[m[1]]
        elif m := re.match(r'(%t\d+) = load i64, ptr (%d\d+)$', line):
            val[m[1]] = slot[m[2]]
        elif m := re.match(r'(%t\d+) = getelementptr \[5 x i64\], ptr (%p\d+), i64 0, i64 (\d)$', line):
            arr[m[1]] = (m[2], int(m[3]))
        elif m := re.match(r'store i64 (%t\d+|0), ptr (%t\d+)$', line):
            p, i = arr[m[2]]
            v = '0' if m[1] == '0' else (val.get(m[1]) or slot.get(m[1]) or role.get(m[1]))
            if v is None:  # an output slot stored straight into the array (fresh arena offset)
                v = [s for d, s in slot.items() if d][-1]
            arr.setdefault(p, [None] * 5)[i] = v
        elif m := re.match(r'call void @mapal_gpu_launchv\(i64 (-?\d+), i32 0, i32 5, ptr (%p\d+), i32 (\d+), i32 (\d+), i32 (\d+), i32 (\d+)\)', line):
            k = 'mapal_tile_%016x' % (int(m[1]) & (2**64 - 1))
            out.append(' '.join([k, m[3], m[4], m[5], m[6], *arr[m[2]]]))
    open(host.replace('.host.ll', '.k'), 'w').write('\n'.join(out) + '\n')
    print(host.split('/')[-1], '|', ' / '.join(out))
