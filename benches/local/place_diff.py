#!/usr/bin/env python3
"""ADR-0050 C4 R4: the suite's emission with `machine-place` ON against OFF, per GPU profile.

Every timed-suite program (benches/local/suite.txt, the default face) is emitted with the emitter
(target/release/examples/emit) on both arms for each profile; host and device IR less the `; mapal.` evidence
lines are compared (changed.py's `norm`). Of the ON arm, each `; mapal.place` line (one a placed site and order)
gives the search's schedule, its price, the rules' (the incumbent's) price, whether the rules' launch was kept
within the band, and the open flips.

  place_diff.py <out> [--profiles a,b,..] [--jobs N] [--stems ..]
-> <out>/place.tsv (profile stem changed kept pick_ns rules_ns sched best open; changed 2: a launched kernel is new or
   differs from OFF's of its name, 1: the host and the dropped table candidates alone — a placed site's table is its
   pick —, 0: equal) and a summary per profile.
"""
import argparse
import concurrent.futures as cf
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EMIT = ROOT / 'target/release/examples/emit'
PROFILES = ['cuda-ada', 'rtx-4090', 'h100', 'h100-pcie', 'rtx-5090', 'rtx-5080', 'rtx-3090', 'a100']


def norm(text):
    return ''.join(l for l in text.splitlines(True) if not l.startswith('; mapal.'))


def emit(src, arch, on):
    with tempfile.TemporaryDirectory() as t:
        dev = Path(t) / 'd.ll'
        cmd = [str(EMIT), str(src), '-', '--host=x86-64-v3', f'--device-out={dev}', '--rewrite', f'--target={arch}']
        if on:
            cmd.append('--gpu-on=machine-place')
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            return None, ''
        d = dev.read_text() if dev.exists() else ''
        return (norm(r.stdout), norm(d)), d


def kernels(dev):
    """A device module's functions by name (`define ... @name(`), each its text."""
    out, name, buf = {}, None, []
    for l in dev.splitlines(True):
        if l.startswith('define '):
            if name:
                out[name] = ''.join(buf)
            name, buf = l.split('@', 1)[1].split('(', 1)[0], []
        if name and not l.startswith('!'):
            buf.append(l)
    if name:
        out[name] = ''.join(buf)
    # Loop metadata is numbered by the kernel's place in the module: not a difference.
    return {n: re.sub(r'!\d+', '!#', t) for n, t in out.items()}


def field(line, k):
    m = re.search(rf' {k}=(\S+)', line)
    return m[1] if m else ''


def one(job):
    arch, stem = job
    src = ROOT / 'benches/algorithms' / f'{stem}.mapal'
    off, _ = emit(src, arch, False)
    on, dev = emit(src, arch, True)
    if off is None or on is None:
        return [(arch, stem, 'err', '', '', '', '', '', '')]
    # 2: a kernel the ON arm launches is new or differs from OFF's of its name; 1: the host alone and kernels
    # OFF had that ON no longer emits (a placed site's table is its pick: the other candidates' kernels go); 0: equal.
    ko, kn = kernels(off[1]), kernels(on[1])
    moved = [n for n, t in kn.items() if ko.get(n) != t]
    changed = 2 if moved else int(off != on)
    rows = []
    seen = set()
    for l in dev.splitlines():
        if not l.startswith('; mapal.place '):
            continue
        key = (field(l, 'site'), field(l, 'order'))
        if key in seen:
            continue
        seen.add(key)
        opn = l.split(' open=[', 1)[1].rstrip(']') if ' open=[' in l else ''
        rows.append((arch, stem, changed, field(l, 'kept'), field(l, 'ns'), field(l, 'rules'),
                     l.split(' sched=', 1)[1].split(' ns=', 1)[0], field(l, 'best'), opn.replace('\t', ' ')))
    return rows or [(arch, stem, changed, '', '', '', '', '', '')]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('out')
    ap.add_argument('--profiles', default=','.join(PROFILES))
    ap.add_argument('--jobs', type=int, default=10)
    ap.add_argument('--stems', nargs='*')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stems = a.stems or (ROOT / 'benches/local/suite.txt').read_text().split()
    jobs = [(p, s) for p in a.profiles.split(',') for s in stems]
    rows = []
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        for r in ex.map(one, jobs):
            rows.extend(r)
    with open(out / 'place.tsv', 'w') as f:
        f.write('profile\tstem\tchanged\tkept\tpick_ns\trules_ns\tsched\tbest\topen\n')
        for r in rows:
            f.write('\t'.join(map(str, r)) + '\n')
    for p in a.profiles.split(','):
        pr = [r for r in rows if r[0] == p]
        progs = {r[1]: r[2] for r in pr}
        placed = [r for r in pr if r[3] != '']
        kept = [r for r in placed if r[3] == '1']
        moved = [r for r in placed if r[3] == '0']
        opn = [r for r in placed if r[8]]
        gain = sorted(((float(r[5]) / float(r[4]), r) for r in moved if r[5] not in ('', '-') and float(r[4]) > 0),
                      key=lambda x: -x[0])
        print(f'{p}: {len(progs)} programs, {sum(1 for v in progs.values() if v == 2)} with a moved kernel, '
              f'{sum(1 for v in progs.values() if v == 1)} the table alone, '
              f'{sum(1 for v in progs.values() if v == "err")} err; {len(placed)} placed sites: {len(kept)} kept, '
              f'{len(moved)} replaced, {len(opn)} open')
        for g, r in gain[:3]:
            print(f'    x{g:.2f} {r[1]}: rules {r[5]} ns -> {r[4]} ns  {r[6]}')


if __name__ == '__main__':
    sys.exit(main())
