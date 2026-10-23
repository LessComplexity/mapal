#!/usr/bin/env python3
"""merge.py <out> <file>:<regex-of-progs-to-take> ...: one session file from several boxes, each
program's rows from exactly one box (the first file whose regex matches it). Header lines of every
part file are kept with a `# from <file>` marker."""
import re
import sys

out, specs = sys.argv[1], sys.argv[2:]
pats = [(f, re.compile(r)) for f, r in (s.split(':', 1) for s in specs)]
owner = {}
with open(out, 'w') as o:
    for f, rx in pats:
        o.write(f"# from {f.split('/')[-1]} programs matching {rx.pattern}\n")
        for l in open(f):
            m = re.search(r'\bprog=(pt_\w+?)(?:\s|$)', l)
            if not m:
                if l.startswith(('==', 'name,', 'NVIDIA', 'differential', 'PASS', 'FAIL', 'traps')):
                    o.write(l)
                continue
            p = m[1]
            if not rx.fullmatch(p):
                continue
            if owner.setdefault(p, f) != f:
                continue
            o.write(l)
print(out, len(owner), 'programs')
