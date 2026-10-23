#!/usr/bin/env python3
"""Run the repaired observer's admission (admit.py, unmodified) on every capture dir.

usage: python3 -I admit_all.py <observer_dir_with_admit.py> <cap_root> <out.json>
Writes {capture: {pass, error | selected launches}} and prints the cubin/symbol pairs to decode.
"""
import json, sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from admit import admit  # noqa: E402

root, out = Path(sys.argv[2]), Path(sys.argv[3])
report = {}
for ev in sorted(root.rglob('events.tsv')):
    d = ev.parent
    key = str(d.relative_to(root))
    try:
        r = admit(d)
        report[key] = {
            'pass': True, 'cupti_version': r['cupti_version'], 'images': len(r['images']),
            'all_kernel_records': r['all_kernel_records'], 'events_sha256': r['events_sha256'],
            'selected': [{k: x[k] for k in ('symbol', 'grid', 'block', 'smem', 'join', 'module_sha256', 'text_sha256',
                                            'elf_symbol_index', 'correlation')}
                         | {'image': next(i['path'] for i in r['images'] if i['event'] == x['image_event'])}
                         for x in r['selected_launches']],
        }
    except Exception as e:  # non-admission is a result, recorded as such
        report[key] = {'pass': False, 'error': f'{type(e).__name__}: {e}'}
out.write_text(json.dumps(report, indent=1) + '\n')
for k, v in report.items():
    print(k, 'ADMITTED' if v['pass'] else 'REJECTED ' + v['error'],
          *[f"{s['symbol'][:60]} grid={s['grid']} block={s['block']} smem={s['smem']}" for s in v.get('selected', [])], sep=' | ')
