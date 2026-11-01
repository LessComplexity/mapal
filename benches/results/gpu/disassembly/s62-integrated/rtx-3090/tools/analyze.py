#!/usr/bin/env python3
"""Tables for the RTX 3090 disassembly session from results.txt (+ static/nsys/observer JSON).

usage: analyze.py <results.txt> <mapal-static.json> <library-static.json> <nsys-dir> <clocks.csv> > tables.json
Every number is computed from the raw files named; nothing is predicted.
"""
import csv, json, re, statistics, sys
from collections import defaultdict
from pathlib import Path

res, mstatic, lstatic, nsdir, clocks = sys.argv[1:6]
lines = Path(res).read_text(errors='replace').splitlines()

SMS, FP32_LANES = 82, 128
# sm_86 limits (CUDA occupancy calculator): per SM.
REGS_SM, THREADS_SM, BLOCKS_SM, SMEM_SM, SMEM_RES = 65536, 1536, 16, 102400, 1024


def occupancy(regs, threads, smem):
    if not regs or not threads:
        return None
    warps = -(-threads // 32)
    regs_warp = -(-(max(regs, 1) * 32) // 256) * 256
    by_regs = (REGS_SM // regs_warp) // warps if regs_warp else BLOCKS_SM
    by_thr = THREADS_SM // (warps * 32)
    by_smem = SMEM_SM // (smem + SMEM_RES) if smem is not None else BLOCKS_SM
    b = max(0, min(by_regs, by_thr, by_smem, BLOCKS_SM))
    return {'blocks_per_sm': b, 'warps_per_sm': b * warps, 'occupancy': round(b * warps / 48, 3),
            'limit': min((by_regs, 'regs'), (by_thr, 'threads'), (by_smem, 'smem'), (BLOCKS_SM, 'blocks'))[1]}


# --- achieved clocks during the timed section
sm_clk, mem_clk = [], []
with open(clocks) as f:
    for row in csv.reader(f):
        if len(row) < 3 or 'MHz' not in row[1]:
            continue
        try:
            s, m = float(row[1].split()[0]), float(row[2].split()[0])
        except ValueError:
            continue
        if s > 1000:  # under load
            sm_clk.append(s), mem_clk.append(m)
clk = statistics.median(sm_clk) if sm_clk else None
mclk = statistics.median(mem_clk) if mem_clk else None
peak_flops = SMS * FP32_LANES * 2 * clk * 1e6 if clk else None
peak_bw = mclk * 1e6 * 2 * 384 / 8 if mclk else None  # GDDR6X: reported clock x2 x bus bytes

# --- cell timings: per process (pass) medians
cell = defaultdict(lambda: defaultdict(dict))  # cell -> leg -> pass -> med
samples = defaultdict(lambda: defaultdict(list))
values = defaultdict(dict)
arm_info = {}
lib_info = defaultdict(dict)
for l in lines:
    m = re.match(r'pass=(\d) cell=(\S+) time leg=(\S+) M=\d+ N=\d+ K=\d+ n=\d+ min_ms=([\d.]+) med_ms=([\d.]+)', l)
    if m:
        cell[m[2]][m[3]][int(m[1])] = float(m[5])
    m = re.match(r'pass=(\d) cell=(\S+) samples leg=(\S+) (.*)', l)
    if m and m[4]:
        samples[m[2]][m[3]] += [float(x) for x in m[4].split(',') if x]
    m = re.match(r'pass=(\d) cell=(\S+) values leg=(\S+) (.*)', l)
    if m:
        values[m[2]][m[3]] = m[4]
    m = re.match(r'pass=1 cell=(\S+) arm stem=(\S+) .* kernel=(\S+) grid=(\d+)x(\d+) block=(\d+) smem=(\d+) regs=(\d+) local_bytes=(\d+) S=(\d+)(.*)', l)
    if m:
        arm_info[(m[1], m[2])] = dict(kernel=m[3], grid=[int(m[4]), int(m[5])], block=int(m[6]), smem=int(m[7]),
                                     regs_driver=int(m[8]), local=int(m[9]), S=int(m[10]), extra=m[11].strip())
    m = re.match(r'pass=1 cell=(\S+) library leg=(\S+) tb=(\S+) warp=(\S+) threads=(\d+) regs=(\d+) local_bytes=(\d+) static_smem=(\d+) dyn_smem=(\d+)', l)
    if m:
        lib_info[m[1]][m[2]] = dict(tb=m[3], warp=m[4], threads=int(m[5]), regs=int(m[6]), local=int(m[7]),
                                   static_smem=int(m[8]), dyn_smem=int(m[9]))


def dims(c):
    return [int(x) for x in c.split('x')]


LEGNAME = lambda c, leg: {f'mm_{c}': 'Mapal E', f'mm_{c}_c': 'Mapal C', f'mm_{c}_rc': 'Mapal reassoc+C',
                          f'mm_{c}_c.null': 'Mapal C (null copy)'}.get(leg, leg)
out = {'clock_sm_mhz_median_under_load': clk, 'clock_mem_mhz_median_under_load': mclk,
       'peak_fp32_tflops_at_clock': peak_flops and round(peak_flops / 1e12, 2),
       'peak_dram_gbs': peak_bw and round(peak_bw / 1e9, 1), 'cells': {}}
for c, legs in cell.items():
    M, N, K = dims(c)
    flops = 2 * M * N * K
    base_bytes = 4 * (M * K + K * N + M * N)
    rows = {}
    for leg, per in legs.items():
        meds = [per[p] for p in sorted(per)]
        both = statistics.median(samples[c][leg]) if samples[c][leg] else None
        S = arm_info.get((c, leg), {}).get('S', 1)
        bytes_ = base_bytes + (2 * S * M * N * 4 if S > 1 else 0)
        t = both / 1e3 if both else None
        rows[LEGNAME(c, leg)] = {
            'leg': leg, 'pass_medians_ms': meds, 'median_ms_all': both,
            'pass_spread': round(abs(meds[0] - meds[-1]) / min(meds), 4) if len(meds) == 2 else None,
            'bytes_min': bytes_, 'gbs': t and round(bytes_ / t / 1e9, 1),
            'pct_dram_peak': t and peak_bw and round(100 * bytes_ / t / peak_bw, 1),
            'tflops': t and round(flops / t / 1e12, 3),
            'pct_fp32_peak': t and peak_flops and round(100 * flops / t / peak_flops, 1),
            'ai_flop_per_byte': round(flops / bytes_, 2),
            't_dram_floor_ms': peak_bw and round(bytes_ / peak_bw * 1e3, 4),
            't_fp32_floor_ms': peak_flops and round(flops / peak_flops * 1e3, 4),
            'values': values[c].get(leg),
        }
    nul, con = rows.get('Mapal C (null copy)'), rows.get('Mapal C')
    null_var = None
    if nul and con:
        null_var = [round(abs(a - b) / b, 4) for a, b in zip(nul['pass_medians_ms'], con['pass_medians_ms'])]
    out['cells'][c] = {'rows': rows, 'null_variation_per_pass': null_var, 'arms': {k[1]: v for k, v in arm_info.items() if k[0] == c},
                       'library_cutlass': lib_info.get(c)}

# --- MLP programs
mlp = defaultdict(lambda: defaultdict(list))
mtrace = {}
for l in lines:
    m = re.match(r'mlp arm=(\S+) mode=(\S+) rep=\d+ out=\[.*iter ms=([\d.e-]+)', l)
    if m:
        mlp[m[1]][m[2]].append(float(m[3]))
    m = re.match(r'mlp_trace arm=(\S+) mode=(\S+) (.*)', l)
    if m:
        kv = dict(re.findall(r'(\w+)=([\d.]+)', m[3]))
        mtrace[(m[1], m[2])] = kv
out['mlp_programs'] = {a: {mode: {'n': len(v), 'median_ms': statistics.median(v), 'min_ms': min(v)} for mode, v in d.items()}
                       for a, d in mlp.items()}
out['mlp_traces'] = {f'{a}|{m}': v for (a, m), v in mtrace.items()}
rm = defaultdict(dict)
for l in lines:
    m = re.match(r'pass=(\d) rm_time M=128 d=(\d+) h=\d+ leg=(\S+) launches=(\d+) reps=\d+ first_ms=([\d.]+) event_med_ms=([\d.]+) event_min_ms=([\d.]+) wall_med_ms=([\d.]+)', l)
    if m:
        rm[(int(m[2]), m[3])][int(m[1])] = dict(launches=int(m[4]), first_ms=float(m[5]), event_med_ms=float(m[6]),
                                               wall_med_ms=float(m[8]))
    m = re.match(r'pass=(\d) rm_values M=128 d=(\d+) h=\d+ leg=(\S+) (.*)', l)
    if m:
        rm[(int(m[2]), m[3])].setdefault('values', m[4])
out['mlp_library'] = {f'd{d}|{leg}': v for (d, leg), v in rm.items()}

# --- nsys gpu traces: per kernel name, grid/block/regs/smem/duration
ns = {}
for f in sorted(Path(nsdir).glob('*.gpu_trace.csv')):
    rows = []
    txt = f.read_text(errors='replace').splitlines()
    hdr = next((i for i, x in enumerate(txt) if x.startswith('"Start') or x.startswith('Start')), None)
    if hdr is None:
        continue
    for r in csv.DictReader(txt[hdr:]):
        name = r.get('Name') or ''
        if not r.get('GrdX'):
            continue
        rows.append(dict(name=name, grid=[r.get('GrdX'), r.get('GrdY'), r.get('GrdZ')], block=[r.get('BlkX'), r.get('BlkY'), r.get('BlkZ')],
                         regs=r.get('Reg/Trd'), static_smem=r.get('StcSMem (MB)') or r.get('StcSMem'),
                         dyn_smem=r.get('DymSMem (MB)') or r.get('DymSMem'), dur_ns=r.get('Duration (ns)')))
    agg = defaultdict(list)
    for r in rows:
        key = (r['name'], tuple(r['grid']), tuple(r['block']), r['regs'], r['static_smem'], r['dyn_smem'])
        if r['dur_ns']:
            agg[key].append(float(r['dur_ns']))
    ns[f.name.replace('.gpu_trace.csv', '')] = [
        dict(name=k[0][:160], grid=k[1], block=k[2], regs=k[3], static_smem=k[4], dyn_smem=k[5], n=len(v),
             median_us=round(statistics.median(v) / 1e3, 2)) for k, v in agg.items()]
out['nsys'] = ns
out['occupancy_fn'] = 'sm_86: 64K regs, 1536 thr, 16 blk, 100KB smem (+1KB/blk reserved) per SM'
json.dump(out, sys.stdout, indent=1, default=str)
