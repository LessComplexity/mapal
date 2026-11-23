#!/usr/bin/env python3
"""sweep-runtime-2: a tuned pick confirmed in its program (the tune's in-context step, fp8-leftovers M3).

The trial (`mapal_choice.h`) times a site's candidates alone: back to back, the L2 evicted where the site's operand
lives in DRAM. Its pick can still lose inside the program — the kernels around the site, the clock the whole
program holds — and two sessions' trials can pick differently on equal code (integration: `llm_dec_b32_l2048`
4.7 % slower on one site's candidate 3 against 0, three such rows). This step runs the program itself with the
pick against the same wisdom with that site's candidate 0 (one appended line: the last line of a key wins), three
alternating pairs (0, w), (w, 0), (0, w) of whole processes, and keeps the pick only where the program is faster
with it: the median of the pairs' `iter ms` ratios w / 0 under `1 - MARGIN` and most pairs agree (the median, not
the mean: one disturbed process — a 40 us program once ran 4.2x in a pair on the shared machine — must not decide).
Otherwise candidate 0 — the rules' plan, the same in every session — is appended to the wisdom.

usage: confirm_pick.py <exe> <wisdom dir> <key:pick:src>...   (box_run.sh, after a stem's tune run; CONFIRM=0: off)
env:   the program's own (MAPAL_PTX, ...); CONFIRM_MARGIN (default 0.01), CONFIRM_PAIRS (default 3)
prints one line per tuned key: `confirm key=<k> pick=<w> pairs=<r1,r2,..> med=<m> keep|revert`
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ITER = re.compile(r'iter ms=([0-9.]+)')


def wisdom_line(wdir, key):
    """The file holding `key` and its last line's fields (the decision the program reads)."""
    found = None
    for f in sorted(Path(wdir).glob('wisdom-*.tsv')):
        for line in f.read_text().splitlines():
            x = line.split('\t')
            if len(x) >= 4 and x[0] == key:
                found = (f, x)
    return found


def run(exe, wdir):
    env = dict(os.environ, MAPAL_WISDOM=str(wdir), MAPAL_GPU_TUNE='0', MAPAL_GPU_PICK='auto')
    env.pop('MAPAL_GPU_TRACE', None)
    out = subprocess.run([exe], env=env, capture_output=True, text=True).stdout
    m = ITER.search(out)
    if not m:
        raise SystemExit(f'confirm: no `iter ms=` from {exe}')
    return float(m.group(1))


def confirm(exe, wdir, key, margin, pairs):
    hit = wisdom_line(wdir, key)
    if not hit:
        return f'confirm key={key} absent'
    f, x = hit
    pick = int(x[3])
    if pick == 0:
        return f'confirm key={key} pick=0 keep'
    zero = '\t'.join([x[0], x[1], x[2], '0', '0']) + '\n'
    with tempfile.TemporaryDirectory() as tmp:
        for g in Path(wdir).glob('wisdom-*.tsv'):
            shutil.copy(g, Path(tmp) / g.name)
        with open(Path(tmp) / f.name, 'a') as o:
            o.write(zero)
        ratios = []
        for r in range(pairs):
            order = [('0', tmp), ('w', wdir)] if r % 2 == 0 else [('w', wdir), ('0', tmp)]
            t = {arm: run(exe, d) for arm, d in order}
            ratios.append(t['w'] / t['0'])
    med = sorted(ratios)[len(ratios) // 2]
    keep = med < 1 - margin and sum(r < 1 for r in ratios) * 2 > len(ratios)
    if not keep:
        with open(f, 'a') as o:
            o.write(zero)
    rs = ','.join(f'{r:.3f}' for r in ratios)
    return f'confirm key={key} pick={pick} pairs={rs} med={med:.3f} {"keep" if keep else "revert"}'


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    exe, wdir = argv[0], argv[1]
    margin = float(os.environ.get('CONFIRM_MARGIN', '0.01'))
    pairs = int(os.environ.get('CONFIRM_PAIRS', '3'))
    for item in argv[2:]:
        key, pick, src = (item.split(':') + ['', ''])[:3]
        if src != 'trial' or not pick.isdigit() or int(pick) == 0:
            continue
        print(confirm(exe, wdir, key, margin, pairs), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
