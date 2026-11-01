#!/usr/bin/env python3
"""flow-sweep: per losing workload x arch, Mapal's kernels vs the best leg's, from the nsys traces of
box_profile.sh SWEEP_PROF (prof/sweep/{mapal,lib,torch}_<stem>.trace.csv.gz).
usage: sweep_report.py <raw dir> <workload:arch:...>   (raw/<arch>/prof-<arch>.tgz; a workload list as stems)
Counters are refused in vast containers: no DRAM GB/s / FP32 % here, only nsys times, grid/block, regs, smem."""
import csv, gzip, io, statistics, sys, tarfile
from collections import defaultdict

def load(tf, name):
    try:
        f = tf.extractfile(f'prof/sweep/{name}.trace.csv.gz')
    except KeyError:
        return None
    if f is None:
        return None
    txt = gzip.decompress(f.read()).decode(errors='replace')
    i = txt.find('Start (ns)')
    return [r for r in csv.DictReader(io.StringIO(txt[i:])) if r.get('GrdX')] if i >= 0 else None

def summ(rows, tag):
    if not rows:
        return f'{tag}: no trace'
    rows.sort(key=lambda r: int(r['Start (ns)']))
    by = defaultdict(list)
    for r in rows:
        by[r['Name']].append(r)
    gaps = [int(b['Start (ns)']) - int(a['Start (ns)']) - int(a['Duration (ns)']) for a, b in zip(rows, rows[1:])]
    gaps = [g for g in gaps if g < 5_000_000]  # within one iteration, not the process's host phases
    top = sorted(by.items(), key=lambda kv: -sum(int(r['Duration (ns)']) for r in kv[1]))[:3]
    out = [f'{tag}: {len(rows)} launches, {len(by)} distinct, median gap {statistics.median(gaps)/1e3 if gaps else 0:.1f} us']
    for n, rs in top:
        r = rs[0]
        d = statistics.median(int(x['Duration (ns)']) for x in rs) / 1e3
        out.append(f"  {n[:60]} x{len(rs)} med {d:.1f} us grid {r['GrdX']}x{r['GrdY']}x{r['GrdZ']} blk {r['BlkX']}x{r['BlkY']} "
                   f"regs {r['Reg/Trd']} smem {float(r['StcSMem (MB)'])*1024+float(r['DymSMem (MB)'])*1024:.1f} KB")
    return '\n'.join(out)

if __name__ == '__main__':
    raw, stems = sys.argv[1], sys.argv[2:]
    for arch in ('rtx-4090', 'rtx-5090', 'h100-pcie', 'rtx-3090'):
        try:
            tf = tarfile.open(f'{raw}/{arch}/prof-{arch}.tgz')
            tf.getmembers()
        except Exception as e:  # a truncated archive keeps the members read so far
            print(f'## {arch}: {e}')
            tf = None
        if tf is None:
            continue
        print(f'## {arch}')
        for s in stems:
            print(f'### {s}')
            for tag in ('mapal', 'lib', 'torch'):
                try:
                    print(summ(load(tf, f'{tag}_pt_{s}'), tag))
                except Exception as e:
                    print(f'{tag}: {e}')
