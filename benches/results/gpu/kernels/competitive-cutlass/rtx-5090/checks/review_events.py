from pathlib import Path
from decimal import Decimal
from collections import defaultdict
import hashlib
import json
import re

root = Path('/private/tmp/mapal-s61-gpu/attempt-09/retrieved/trace/competitive-ordinary')
out = Path('/private/tmp/mapal-competitive-timing-review')
summary = json.loads((root / 'summary.json').read_text())
assert summary['pass'] is True and summary['instrumented'] is False
assert summary['mapal_variant'] == 'old'
assert not summary['failures']
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
rows, tuning, confirmation = [], defaultdict(list), defaultdict(dict)
sample_count = 0
for stage, count in [('tuning', 9), ('confirmations', 30)]:
    for record in summary[stage]:
        case, candidate, family = (record[k] for k in ('case', 'candidate', 'family'))
        path = root / record['path']
        stdout = path / 'stdout'
        status_path = path / 'status.json'
        for p in (stdout, status_path, path / 'stderr'):
            assert sha(p) == summary['files'][str(p.relative_to(root))]
        status = json.loads(status_path.read_text())
        assert status['pass'] and status['exit'] == 0
        assert status['argv'] == record['process']['argv']
        assert status['environment_overrides']['NVIDIA_TF32_OVERRIDE'] == '0'
        assert '--profile-leg' not in status['argv']
        assert '--values-only' not in status['argv']
        assert ('--tuning' in status['argv']) == (stage == 'tuning')
        assert int(status['argv'][10]) == count
        shape = tuple(map(int, status['argv'][1:4]))
        assert shape in ((16, 2048, 7168), (16, 1024, 4096))
        face = case.rsplit('__', 1)[1]
        assert ('--contract' in status['argv']) == (face == 'contract')
        build = summary['candidate_builds'][candidate]
        assert build['pass']
        assert record['binary_sha256'] == build['binary_sha256'] == sha(root / build['binary'])
        lines = stdout.read_text().splitlines()
        samples = [line for line in lines if line.startswith('samples ')]
        assert len(samples) == 1
        fields = dict(item.split('=', 1) for item in samples[0].split()[1:])
        assert set(fields) == {'mapal', 'cublas', 'cutlass'}
        events = {}
        for arm, text in fields.items():
            values = [Decimal(v) for v in text.split(',')]
            assert len(values) == count and all(v.is_finite() and v > 0 for v in values)
            ordered = sorted(values)
            median = ordered[count // 2]
            reported = record['events'][arm]
            assert [float(v) for v in values] == reported['raw_event_ms']
            assert median == Decimal(str(reported['upper_median_ms']))
            assert ordered[(count - 1) // 10] == Decimal(str(reported['p10_ms']))
            assert ordered[(count * 9 - 1) // 10] == Decimal(str(reported['p90_ms']))
            assert reported['n'] == count
            events[arm] = median
            sample_count += count
        row = dict(stage=stage, case=case, family=family, candidate=candidate,
                   shape=shape, face=face, medians_ms={k: str(v) for k, v in events.items()},
                   stdout_sha256=sha(stdout), binary_sha256=build['binary_sha256'])
        rows.append(row)
        if stage == 'tuning':
            assert face == 'contract'
            tuning[family].append((events['cutlass'], candidate))
        else:
            key = (shape, face)
            assert candidate not in confirmation[key]
            confirmation[key][candidate] = events

assert len(rows) == 20 and sample_count == 1044
selected = {}
for family, observations in tuning.items():
    assert len(observations) == 6 and len({x[1] for x in observations}) == 6
    median, winner = min(observations)
    declared = summary['selection'][family]
    assert declared['candidate'] == winner
    assert Decimal(str(declared['tuning_upper_median_event_ms'])) == median
    assert declared['selection_uses_confirmation_data'] is False
    selected[family] = winner
assert set(selected.values()) == {'m16-n64-k16'}

table = []
for (shape, face), arms in sorted(confirmation.items()):
    assert set(arms) == {'default-control', 'm16-n64-k16'}
    default, tuned = arms['default-control'], arms['m16-n64-k16']
    table.append(dict(shape=shape, face=face,
        mapal_ms=str(tuned['mapal']), cublas_ms=str(tuned['cublas']),
        cutlass_default_ms=str(default['cutlass']), cutlass_tuned_ms=str(tuned['cutlass']),
        default_over_tuned=float(default['cutlass'] / tuned['cutlass']),
        mapal_over_cublas=float(tuned['mapal'] / tuned['cublas']),
        mapal_over_tuned=float(tuned['mapal'] / tuned['cutlass']),
        tuned_over_cublas=float(tuned['cutlass'] / tuned['cublas']),
        unchanged_mapal_control_ratio=float(max(default['mapal'], tuned['mapal']) / min(default['mapal'], tuned['mapal'])),
        unchanged_cublas_control_ratio=float(max(default['cublas'], tuned['cublas']) / min(default['cublas'], tuned['cublas']))))

result = dict(pass_=True, scope='Independent raw ordinary event and selection review; numerical gates reviewed separately',
              statistics='Upper median at n//2; p10/p90 use nearest-rank ceil(p*n)-1, verified against frozen sample_report and all raw arrays',
              summary_sha256=sha(root / 'summary.json'), processes=20, raw_events=sample_count,
              confirmation_events=720, tuning_events=324, selection=selected, rows=rows, table=table)
(out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))
