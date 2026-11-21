#!/usr/bin/env python3
"""Drop the re-judge arms whose build equals the shipped one up to kernel names (integrated_bundle.sh).

A kernel's name is a hash that also covers the rung set it was emitted under, so an arm whose rung changes no kernel
still renames its site tables' candidates (and llc numbers their blocks in another order): the bundle's byte compare
keeps it, and the box times the shipped build twice (cross-arch 7a2b67bf: 535 of 1 038 built arms over four parts).
Here each function is reduced to its body with hashed names, block labels and comments canonical; every kernel hash
in the host module, launch and site files is replaced by its body's digest; an arm is dropped (its files removed, as
the bundle's `drop`) when the host module, launch, sites, the PTX outside functions and the multiset of bodies all
equal the shipped build's. usage: armdedup.py <part dir> [--dry]   (arms: `<stem>.<on|off>-<rungs>.ptx`)"""
import hashlib
import os
import re
import sys
from collections import Counter

NAME = re.compile(r'mapal_[a-z0-9]+_([0-9a-f]{16})')
ENTRY = re.compile(r'^(?:\.visible |\.weak )?\.(?:entry|func)\s')
ARM = re.compile(r'^(.+)\.((?:on|off)-[^.]+)\.ptx$')


def kernels(ptx):
    """(top-level PTX lines outside functions as a multiset, {hash: body digest}, multiset of body digests)."""
    canon, bodies, outside = {}, Counter(), Counter()
    cur, name = None, None
    for line in open(ptx, errors='replace'):
        line = re.sub(r'[ \t]*//.*', '', line.rstrip('\n'))
        if cur is None:
            if ENTRY.match(line):
                m = NAME.search(line)
                cur, name = [line], m[1] if m else line
            elif line.strip():
                outside[NAME.sub('K', line.strip())] += 1
            continue
        cur.append(line)
        if line == '}':
            b = NAME.sub('K', '\n'.join(cur))
            b = re.sub(r'\$L__BB\d+_', '$L__BB_', b)
            b = re.sub(r'\$L__tmp\d+', '$L__tmp', b)
            h = hashlib.sha1(b.encode()).hexdigest()[:16]
            canon[name] = h
            bodies[h] += 1
            cur = None
    return outside, canon, bodies


def norm(path, canon):
    if not os.path.exists(path):
        return None
    s = ''.join(l for l in open(path, errors='replace') if not l.startswith('; mapal.'))
    for n, h in canon.items():
        if not re.fullmatch(r'[0-9a-f]{16}', n):
            continue
        v = int(n, 16)
        if v >= 1 << 63:
            v -= 1 << 64
        s = re.sub(r'(?<![\d-])' + str(v) + r'(?!\d)', 'C' + h, s)
        s = re.sub(r'mapal_[a-z0-9]+_' + n, 'C' + h, s)
    return s


def view(d, stem):
    ptx = os.path.join(d, stem + '.ptx')
    if not os.path.exists(ptx):
        return None
    outside, canon, bodies = kernels(ptx)
    return (outside, bodies, *(norm(os.path.join(d, stem + x), canon) for x in ('.host.ll', '.launch', '.sites')))


def main():
    d = sys.argv[1]
    dry = '--dry' in sys.argv
    kept, dropped = Counter(), Counter()
    for f in sorted(os.listdir(d)):
        m = ARM.match(f)
        if not m or '.pick' in f:
            continue
        stem, tag = m[1], m[2]
        a, b = view(d, f'{stem}.{tag}'), view(d, stem)
        if a is None or b is None:
            continue
        if a != b:
            kept[tag] += 1
            continue
        dropped[tag] += 1
        if dry:
            continue
        arm = f'{stem}.{tag}.'
        for x in os.listdir(d):
            rest = x[len(arm):]
            if x.startswith(arm) and (rest.split('.')[0] in ('host', 'dev', 'ptx', 'launch', 'k', 'sites') or rest.startswith('pick')):
                os.remove(os.path.join(d, x))
        print(f'rejudge {os.path.basename(d.rstrip("/"))} {stem} {tag} = shipped (up to kernel names)')
    print(f'armdedup {d}: dropped {sum(dropped.values())}, kept {sum(kept.values())}; '
          + ', '.join(f'{t} {kept[t]}/{kept[t] + dropped[t]}' for t in sorted(set(kept) | set(dropped))))


if __name__ == '__main__':
    main()
