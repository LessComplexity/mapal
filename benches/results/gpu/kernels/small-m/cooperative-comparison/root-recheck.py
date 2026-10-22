import argparse
import hashlib
import json
import math
import re
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--trace', type=Path, required=True)
p.add_argument('--analysis', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
trace = a.trace.resolve()
outer = json.loads((trace / 'summary.json').read_text())
inner_dir = trace / 'smallm-values'
inner = json.loads((inner_dir / 'summary.json').read_text())
analysis = json.loads(a.analysis.read_text())
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
for name, expected in outer['files'].items():
    path = (trace / name).resolve()
    assert path.is_relative_to(trace) and path.is_file()
    assert sha(path) == expected, name
assert len(inner['records']) == 234 and inner['inventory_complete']
assert len({r['id'] for r in inner['records']}) == 234
records = {r['id']: r for r in inner['records']}
medians = {}
events = 0
processes = 0
for r in inner['records']:
    if 'process' not in r:
        assert r['outcome'] == 'quarantined_skipped' and not r['pass']
        continue
    process = r['process']
    status = json.loads((inner_dir / (r['id'] + '.status.json')).read_text())
    assert status == process
    for suffix, key in [('.stdout', 'stdout_sha256'), ('.stderr', 'stderr_sha256'), ('.resources.txt', 'resources_sha256')]:
        assert sha(inner_dir / (r['id'] + suffix)) == process[key]
    if not r['pass']:
        assert r['outcome'] == 'numeric_not_admitted' and process['exit'] != 0
        continue
    assert process['exit'] == 0
    raw = (inner_dir / (r['id'] + '.stdout')).read_text()
    assert 'NONFINITE' not in raw and 'VALUES WRONG' not in raw
    assert 'bit_exact_vs_in_order_fold=yes' in raw or 'within_4x_cublas_tf32=yes' in raw
    if r['stage'] != 'timing':
        continue
    samples = next(line for line in raw.splitlines() if line.startswith('samples '))
    row = {}
    for leg in ['mapal', 'cublas', 'cutlass']:
        values = [float(x) for x in re.search(r'\b' + leg + r'=([^ ]+)', samples).group(1).split(',')]
        assert len(values) == 30 and all(math.isfinite(x) and x > 0 for x in values)
        median = sorted(values)[len(values) // 2]
        line = next(line for line in raw.splitlines() if line.startswith('time leg=' + leg + ' '))
        reported = float(re.search(r' med_ms=([^ ]+)', line).group(1))
        assert abs(reported - median) <= 0.000101
        row[leg] = reported
        events += len(values)
    medians[r['id']] = row
    processes += 1
for row in analysis['rows']:
    if not row['admitted']:
        assert row['id'] not in inner['comparable_timing_cells']
        continue
    assert row['id'] in inner['comparable_timing_cells']
    for arm in ['old', 'candidate', 'candidate_null', 'old_null']:
        measured = medians[row['id'] + '__timing__' + arm]
        for leg in measured:
            assert abs(measured[leg] - row['processes'][arm]['times'][leg]['median_ms']) < 1e-9
    for i, arm in enumerate(['candidate', 'candidate_null']):
        measured = medians[row['id'] + '__timing__' + arm]
        for leg in ['cublas', 'cutlass']:
            ratio = measured[leg] / measured['mapal']
            assert abs(ratio - row['candidate_library_over_mapal_pairs'][leg][i]) < 1e-10
result = {
    'pass': True,
    'outer_files_rehashed': len(outer['files']),
    'scheduled_records': len(records),
    'passing_records': sum(r['pass'] for r in records.values()),
    'numeric_failures': sum(r['outcome'] == 'numeric_not_admitted' for r in records.values()),
    'quarantined_skips': sum(r['outcome'] == 'quarantined_skipped' for r in records.values()),
    'timing_processes_reparsed': processes,
    'raw_cuda_event_samples_reparsed': events,
    'admitted_timing_cells': sum(r['admitted'] for r in analysis['rows']),
    'analysis_sha256': sha(a.analysis),
    'outer_summary_sha256': sha(trace / 'summary.json'),
    'method': 'Independent root implementation: hashes, statuses, numerical pass markers, 30-event upper medians and same-process library ratios; no imported runner or analyst parser.',
}
a.out.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
