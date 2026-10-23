"""MLP whole-program hotspots from nsys (cuda_api_trace + cuda_gpu_trace CSVs).

Mapal: the program's timed region = end of the cuCtxSynchronize before its first tile launch ..
end of the first cuCtxSynchronize after its last tile launch (mapal_gpu_sync around `time`).
Library (rm_bench): each timed leg call sits between two cudaEventRecord calls; legs run
round-robin in the harness order, reversed on odd reps (ABBA).
"""
import statistics

from nsys_parse import api_trace, gpu_trace

LEGS = ['cublas', 'lt', 'cutlass', 'cutlass_tuned']


def _segment(apis, kernels, t0, t1):
    inside = [a for a in apis if a['start'] >= t0 and a['start'] + a['dur'] <= t1 + 1]
    corrs = {a['corr'] for a in inside}  # kernels launched by the segment's own API calls (GPU time lags host time)
    ks = sorted((k for k in kernels if k['kind'] == 'kernel' and k['corr'] in corrs), key=lambda k: k['start'])
    by = {}
    for a in inside:
        by[a['name']] = by.get(a['name'], 0) + a['dur']
    busy = sum(k['dur'] for k in ks)
    gaps = [ks[i + 1]['start'] - (ks[i]['start'] + ks[i]['dur']) for i in range(len(ks) - 1)]
    return dict(span_ms=(t1 - t0) / 1e6, kernel_ms=busy / 1e6,
                kernels=[dict(name=k['name'], us=k['dur'] / 1e3, grid=k['grid'], block=k['block'], regs=k['regs'],
                              smem=k['static_smem'] + k['dyn_smem'], local=k['local']) for k in ks],
                first_kernel_after_ms=(ks[0]['start'] - t0) / 1e6 if ks else None,
                gpu_span_ms=((ks[-1]['start'] + ks[-1]['dur']) - ks[0]['start']) / 1e6 if ks else None,
                tail_after_last_kernel_ms=(t1 - (ks[-1]['start'] + ks[-1]['dur'])) / 1e6 if ks else None,
                first_launch_api_after_ms=next(((a['start'] - t0) / 1e6 for a in inside if 'Launch' in a['name']), None),
                gaps_us=[g / 1e3 for g in gaps], api_ms={k: v / 1e6 for k, v in sorted(by.items(), key=lambda x: -x[1])})


def mapal_region(base):
    apis = sorted(api_trace(base), key=lambda a: a['start'])
    ker = gpu_trace(base)
    tiles = {k['corr'] for k in ker if k['kind'] == 'kernel' and k['name'].startswith('mapal_tile_')}
    launches = [a for a in apis if a['corr'] in tiles and 'Launch' in a['name']]
    if not launches:
        return None
    first, last = launches[0], launches[-1]
    syncs = [a for a in apis if 'Synchronize' in a['name']]
    before = [s for s in syncs if s['start'] + s['dur'] <= first['start']]
    after = [s for s in syncs if s['start'] >= last['start']]
    if not before or not after:
        return None
    t0, t1 = before[-1]['start'] + before[-1]['dur'], after[0]['start'] + after[0]['dur']
    r = _segment(apis, ker, t0, t1)
    # whole process: where the rest goes (context, module, gens, d2h)
    tot = {}
    for a in apis:
        tot[a['name']] = tot.get(a['name'], 0) + a['dur']
    r['process_api_ms'] = {k: v / 1e6 for k, v in sorted(tot.items(), key=lambda x: -x[1])[:8]}
    r['process_mem'] = [(k['name'], k['dur'] / 1e3) for k in ker if k['kind'] == 'mem']
    return r


def library_legs(base, legs=LEGS):
    apis = sorted(api_trace(base), key=lambda a: a['start'])
    ker = gpu_trace(base)
    recs = [a for a in apis if a['name'].startswith('cudaEventRecord')]
    calls = []
    for i in range(0, len(recs) - 1, 2):
        t0, t1 = recs[i]['start'] + recs[i]['dur'], recs[i + 1]['start']
        calls.append(_segment(apis, ker, t0, t1))
    out = {}
    n = len(legs)
    for j, c in enumerate(calls):
        r, q = divmod(j, n)
        leg = legs[n - 1 - q] if r & 1 else legs[q]
        out.setdefault(leg, []).append(c)
    summ = {}
    for leg, cs in out.items():
        summ[leg] = dict(calls=len(cs), span_ms=statistics.median(c['span_ms'] for c in cs),
                         kernel_ms=statistics.median(c['kernel_ms'] for c in cs),
                         gpu_span_ms=statistics.median(c['gpu_span_ms'] for c in cs if c['gpu_span_ms'] is not None),
                         kernels=cs[-1]['kernels'], gaps_us=cs[-1]['gaps_us'], api_ms=cs[-1]['api_ms'])
    return summ
