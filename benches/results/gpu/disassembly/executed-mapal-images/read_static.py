#!/usr/bin/env python3
"""Read frozen inputs only; emits static facts, never performance conclusions."""
import collections, hashlib, json, pathlib, re
BASE = pathlib.Path('/private/tmp/mapal-s60-kernel')
OUT = pathlib.Path('/private/tmp/mapal-s61-gap-analysis')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
packets = {}
for name in ['bcomparison', 'diagnostics']:
    root = BASE / name
    sealed = json.loads((root / 'sealed.json').read_text())
    for path, expected in sealed.items():
        assert sha(root / path) == expected, (name, path)
    cases = json.loads((root / 'cases.json').read_text())
    rows = []
    for case in cases:
        variants = case.get('variants', {case.get('arm'): case})
        for arm, v in variants.items():
            path = root / v['ptx']
            text = path.read_text()
            assert sha(path) == v['ptx_sha256']
            if name == 'bcomparison':
                assert sha(root / v['null_ptx']) == v['ptx_sha256']
            l = v['launch']
            start = text.index('.visible .entry ' + l['entry'] + '(')
            end = text.find('.visible .entry ', start + 1)
            kernel = text[start:end if end >= 0 else len(text)]
            ins = re.findall(r'^\s*(?:@!?%p\d+\s+)?([a-z][a-z0-9_.]*)\s', kernel, re.M)
            row = {'case': case['id'], 'shape': [case[k] for k in ['M','N','K']], 'face': case['face'], 'arm': arm,
                   'entry': l['entry'], 'ptx_sha256': sha(path), 'grid': [l['gx'], l['gy']],
                   'total_blocks': l['gx'] * l['gy'], 'threads_per_block': l['block'], 'shared_bytes': l['smem'],
                   'ptx_kernel_static_mnemonics': dict(collections.Counter(ins)),
                   'null_verified': name == 'bcomparison'}
            step = re.findall(r'add\.s64\s+%rd\d+,\s*%rd\d+,\s*(\d+);', kernel)
            row['literal_add_s64_immediates'] = [int(x) for x in step]
            rows.append(row)
    packets[name] = {'sealed_files_verified': len(sealed), 'seal_sha256': sha(root/'sealed.json'), 'cases_sha256': sha(root/'cases.json'), 'rows': rows}
report = {'scope': 'frozen packet seal/PTX/null/geometry audit; no compilation, execution, or new performance result', 'packets': packets}
manifest = json.loads((BASE / 'evidence-manifest.json').read_text())
evidence = ['profile-B.rs', 'evidence/launch-context-summary.json', 'evidence/B-launch-context.log']
for name in evidence:
    assert sha(BASE/name) == manifest[name]
report['additional_evidence_hashes'] = {n: sha(BASE/n) for n in evidence}
contexts = json.loads((BASE / 'evidence/launch-context-summary.json').read_text())
report['wide_route_contexts'] = [r for r in contexts if r['N'] == 18432 and r['arm'] in ['old','S58','B']]
(OUT / 'static-facts.json').write_text(json.dumps(report, indent=2)+'\n')
for name, p in packets.items():
    print(name, 'sealed_files_verified', p['sealed_files_verified'])
for r in packets['bcomparison']['rows']:
    if '__timing__' in r['case']:
        print(r['shape'], r['face'], r['arm'], 'grid', r['grid'], 'blocks', r['total_blocks'], 'threads',r['threads_per_block'], 'shared',r['shared_bytes'])
