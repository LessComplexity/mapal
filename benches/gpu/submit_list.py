#!/usr/bin/env python3
"""The launches of shipped probe programs, for submit_bench.c (S62 submission stage 2).

usage: submit_list.py <bundle-part-dir> <prog>... > submit-list-<part>
One line per distinct tile launch in <prog>.host.ll: prog entry id nptr bytes gx gy block smem
(bytes = n·n·4 from the probe's size: every buffer at least as large as the program's).
"""
import re
import sys

d, progs = sys.argv[1], sys.argv[2:]
seen = set()
for p in progs:
    n = int(re.findall(r'_(\d+)', p)[-1])
    for l in open(f"{d}/{p}.host.ll"):
        m = re.search(r'@mapal_gpu_launchv\(i64 (-?\d+), i32 \d+, i32 (\d+), ptr %\w+, i32 (\d+), i32 (\d+), '
                      r'i32 (\d+), i32 (\d+)\)', l)
        if m:
            e, (i, k, gx, gy, b, s) = 'launchv', m.groups()
        else:
            m = re.search(r'@mapal_gpu_launch\(i64 (-?\d+), i64 %\w+, i64 %\w+, i64 %\w+, i32 (\d+), i32 (\d+), '
                          r'i32 (\d+), i32 (\d+)\)', l)
            if not m:
                continue
            e, k, (i, gx, gy, b, s) = 'launch', '3', m.groups()
        key = (p, i)
        if key in seen:
            continue
        seen.add(key)
        print(p, e, hex(int(i) & (2**64 - 1)), k, n * n * 4, gx, gy, b, s)
