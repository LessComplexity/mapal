#!/usr/bin/env python3
"""ADR-0050 cost model: a `place_calib` run's output as the calibration file's `place.*` values and as the Rust
`PlaceFacts` fields (profile.rs), every value measured by that one run.

usage: place_calib_facts.py <place_calib.txt> [--rust]
"""
import re
import sys

text = open(sys.argv[1]).read()
kv = lambda line: dict(re.findall(r'(\w+)=("[^"]*"|\S+)', line))
lines = text.splitlines()
out = {}
start = [kv(l) for l in lines if l.startswith('start ') and 'verdict=' in l]
if start:
    out['start'] = 'stagger' if any(s['verdict'] == 'stagger' for s in start) else 'lockstep'
keeps = {kv(l)['pattern']: kv(l) for l in lines if l.startswith('keep ') and 'keep_bytes=' in l}
if keeps:
    out['keep_rotated'] = keeps['rotated']['keep_bytes']
    out['keep_strided'] = keeps['strided']['keep_bytes']
    out['keep_hi'] = max(int(k['keep_hi_bytes']) for k in keeps.values())
    hit = [kv(l) for l in lines if l.startswith('keep pattern=cyclic ws_l2_pm=')]
    out['hit'] = ','.join(f"{h['ws_l2_pm']}:{h['hit_pm']}" for h in hit)
eff = [kv(l) for l in lines if l.startswith('eff ')]
if eff:
    out['eff'] = ','.join(f"{e['op']}:{e['chains']}:{round(float(e['wps']) * 1000)}:{e['eff_pm']}" for e in eff)
lat = [kv(l) for l in lines if l.startswith('lat op=')]
if lat:
    out['latency'] = ','.join(f"{l['op']}:{round(float(l['clk']) * 10)}" for l in lat)
clk = [kv(l) for l in lines if l.startswith('clock sustained_khz=')]
if clk:
    out['clock_khz'] = clk[0]['sustained_khz']
prim = [l for l in lines if l.startswith('calib place.prim=')]
if prim:
    out['prim'] = prim[-1].split('place.prim=')[1].split()[0]

if '--rust' not in sys.argv:
    print(' '.join(f'place.{k}={v}' for k, v in out.items()))
    sys.exit()
r = []
if 'start' in out:
    r.append(f"start: Some((mapal_place::Start::{out['start'].title()}, CITE)),")
for k in ('keep_rotated', 'keep_strided', 'keep_hi'):
    if k in out:
        r.append(f"{k}: {int(out[k]):_},".replace('_', '_'))
if 'hit' in out:
    r.append('hit_curve: &[' + ', '.join(f'({x}, {y})' for x, y in (p.split(':') for p in out['hit'].split(','))) + '],')
if 'eff' in out:
    pts = [p.rsplit(':', 3) for p in out['eff'].split(',')]
    r.append('eff: &[' + ', '.join(f'("{o}", {c}, {g}, {pm})' for o, c, g, pm in pts) + '],')
if 'latency' in out:
    r.append('latency: &[' + ', '.join(f'("{o}", {x})' for o, x in (p.rsplit(':', 1) for p in out['latency'].split(','))) + '],')
if 'clock_khz' in out:
    r.append(f"clock_khz: {out['clock_khz']},")
if 'prim' in out:
    rows = [p.split(':') for p in out['prim'].split(',')]
    r.append('prim: &[' + ', '.join(f'("{f[0]}", [{", ".join(f[1:])}])' for f in rows) + '],')
print('\n'.join(r))
