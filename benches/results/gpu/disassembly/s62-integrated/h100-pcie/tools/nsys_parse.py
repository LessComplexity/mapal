"""nsys sqlite readers (the box exported `.sqlite` beside every `.nsys-rep`).

gpu_trace(base): kernel rows (name, start/dur ns, corr, grid, block, regs, static/dynamic smem,
local bytes/thread) and memcpy/memset rows; api_trace(base): CUDA runtime + driver API rows."""
import os
import sqlite3
import statistics


def _db(base):
    p = base + '.sqlite'
    return sqlite3.connect(p) if os.path.exists(p) else None


def _tables(c):
    return {r[0] for r in c.execute("select name from sqlite_master where type='table'")}


def _csv(path):
    import csv
    return list(csv.DictReader(open(path, newline=''))) if os.path.exists(path) else []


def gpu_trace(base):
    c = _db(base)
    if not c:  # the committed export (collect.py): <base>_kernels.csv
        return [dict(kind='kernel', name=r['name'], start=int(r['start_ns']), dur=int(r['end_ns']) - int(r['start_ns']),
                     corr=int(r['corr']), grid=[int(x) for x in r['grid'].split('x')],
                     block=[int(x) for x in r['block'].split('x')], regs=int(r['regs']), static_smem=int(r['static_smem']),
                     dyn_smem=int(r['dyn_smem']), local=int(r['local_per_thread']), strm=None)
                for r in _csv(base + '_kernels.csv')]
    S = dict(c.execute('select id, value from StringIds'))
    t = _tables(c)
    out = []
    for r in c.execute('select start, end, correlationId, mangledName, gridX, gridY, gridZ, blockX, blockY, blockZ, '
                       'registersPerThread, staticSharedMemory, dynamicSharedMemory, localMemoryPerThread, streamId '
                       'from CUPTI_ACTIVITY_KIND_KERNEL'):
        out.append(dict(kind='kernel', name=S.get(r[3], ''), start=r[0], dur=r[1] - r[0], corr=r[2], grid=list(r[4:7]),
                        block=list(r[7:10]), regs=r[10], static_smem=r[11], dyn_smem=r[12], local=r[13], strm=r[14]))
    for tb, nm in (('CUPTI_ACTIVITY_KIND_MEMCPY', 'memcpy'), ('CUPTI_ACTIVITY_KIND_MEMSET', 'memset')):
        if tb in t:
            for r in c.execute(f'select start, end, correlationId, bytes from {tb}'):
                out.append(dict(kind='mem', name=nm, start=r[0], dur=r[1] - r[0], corr=r[2], bytes=r[3]))
    return out


def api_trace(base):
    c = _db(base)
    if not c:  # the committed export: <base>_api.csv
        return [dict(name=r['name'], start=int(r['start_ns']), dur=int(r['end_ns']) - int(r['start_ns']), corr=int(r['corr']))
                for r in _csv(base + '_api.csv')]
    S = dict(c.execute('select id, value from StringIds'))
    return [dict(name=S.get(r[2], ''), start=r[0], dur=r[1] - r[0], corr=r[3])
            for r in c.execute('select start, end, nameId, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME')]


def kernel_summary(rows):
    """name -> {n, med_us, grid, block, regs, static_smem, dyn_smem, local} (geometry of the first row)."""
    by = {}
    for r in rows:
        if r['kind'] != 'kernel':
            continue
        d = by.setdefault(r['name'], dict(n=0, durs=[], grid=r['grid'], block=r['block'], regs=r['regs'], local=r['local'],
                                          static_smem=r['static_smem'], dyn_smem=r['dyn_smem'], grids=set()))
        d['n'] += 1; d['durs'].append(r['dur']); d['grids'].add(tuple(r['grid']))
    for d in by.values():
        d['med_us'] = statistics.median(d['durs']) / 1e3
        d['grids'] = sorted(d['grids'])
        del d['durs']
    return by
