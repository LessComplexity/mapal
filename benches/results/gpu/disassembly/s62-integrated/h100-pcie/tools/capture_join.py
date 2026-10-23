#!/usr/bin/env python3
"""Join one observer run (events.tsv, schema library-image-callback-v2) into the kernels it executed.

usage: capture_join.py <capture dir>  ->  JSON list, one row per executed kernel activity:
  name, grid, block, dyn_smem, in_region (its launch callback fired inside cudaProfilerStart/Stop),
  module_id (FUNCTION activity), image (the callback-saved module-N.cubin holding it), join kind.
Joins: kernel activity correlationId -> driver launch callback (region flag); kernel name ->
FUNCTION activity (contextId, name) -> moduleId -> MODULE_LOADED image. A name in several
images without a FUNCTION record is reported `ambiguous`, never guessed.
"""
import json
import os
import subprocess
import sys
from collections import defaultdict


def unhex(h):
    return bytes.fromhex(h).decode(errors='replace') if h else ''


def main(cap):
    images, kernels, funcs, launches, errors, region = {}, [], defaultdict(set), {}, [], []
    for line in open(os.path.join(cap, 'events.tsv'), errors='replace'):
        f = line.rstrip('\n').split('\t')
        k = f[0]
        if k == 'image':
            images[int(f[3])] = dict(event=int(f[1]), ctx=f[2], module_id=int(f[3]), size=int(f[4]), file=f[5])
        elif k == 'kernel':
            kernels.append(dict(event=int(f[1]), corr=int(f[2]), ctx_id=int(f[3]), stream=int(f[4]),
                                grid=[int(x) for x in f[6].split(',')], block=[int(x) for x in f[7].split(',')],
                                dyn_smem=int(f[8]), name=unhex(f[12]) if len(f) > 12 else ''))
        elif k == 'function-activity':
            funcs[(int(f[3]), unhex(f[6]))].add(int(f[4]))
        elif k == 'launch' and f[3] == '0':  # exit record: [4] corr, [13] region flag, [14] rc, [15] symbol
            la = dict(active=f[13] == '1', rc=int(f[14]), symbol=unhex(f[15]))
            launches[int(f[4])] = la
            if f[5] != '0':  # a runtime launch: kernel activity carries the runtime API's correlation id
                launches[int(f[5])] = la
        elif k == 'error':
            errors.append(f[1])
        elif k in ('region-start', 'region-stop', 'finish'):
            region.append(line.strip())
    # function names inside each saved image (for the fallback and for ambiguity reporting)
    names_in = {}
    for mid, im in images.items():
        p = os.path.join(cap, im['file'])
        try:
            out = subprocess.run([os.environ.get('CUOBJDUMP', 'cuobjdump'), '-elf', p], capture_output=True, text=True,
                                 timeout=300).stdout
        except Exception:  # noqa: BLE001
            out = ''
        names_in[mid] = {l.split()[-1] for l in out.splitlines() if 'STT_FUNC' in l and 'STB_GLOBAL' in l}
    rows = []
    for kr in kernels:
        mids = funcs.get((kr['ctx_id'], kr['name']), set())
        join = 'function-activity'
        if len(mids) != 1:
            hits = [m for m, ns in names_in.items() if kr['name'] in ns]
            mids, join = set(hits), ('image-symbol' if len(hits) == 1 else 'ambiguous' if hits else 'absent')
        mid = next(iter(mids)) if len(mids) == 1 else None
        la = launches.get(kr['corr'], {})
        rows.append(dict(name=kr['name'], grid=kr['grid'], block=kr['block'], dyn_smem=kr['dyn_smem'],
                         corr=kr['corr'], in_region=la.get('active'), module_id=mid,
                         image=images[mid]['file'] if mid in images else None, join=join))
    return dict(capture=cap, images=len(images), image_bytes=sum(i['size'] for i in images.values()),
                errors=errors, region=region, kernels=rows)


if __name__ == '__main__':
    print(json.dumps(main(sys.argv[1]), indent=1))
