"""Independent verifier of sealed S58 small-M raw evidence; never executes GPU code."""
import argparse, csv, hashlib, json, math, statistics
from pathlib import Path
import re

SEALED_MANIFEST = '527e4b6a3f4f474edab4fddf97ec837e685d84cd678883dc66a79fa012442002'
LEGS = ('mapal', 'cublas', 'cutlass')
ARMS = ('old', 'candidate', 'candidate_null', 'old_null')

def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def fields(line):
    return dict(re.findall(r'(\w+)=([^\s]+)', line))

def one(lines, prefix):
    rows = [line for line in lines if line.startswith(prefix)]
    require(len(rows) == 1, 'Missing or duplicated row: ' + prefix)
    return rows[0]

def finite(value, positive=False):
    number = float(value)
    require(math.isfinite(number) and (number > 0 if positive else number >= 0), 'Nonfinite/invalid numeric field')
    return number

def verify_bundle(bundle, digest):
    require(sha(bundle / 'manifest.json') == digest, 'Unreviewed packet manifest')
    manifest = json.loads((bundle / 'manifest.json').read_text())
    require(len(manifest['cases']) == 114, 'Wrong packet case inventory')
    for name, expected in manifest['files'].items():
        path = (bundle / name).resolve()
        require(path.is_relative_to(bundle.resolve()), 'Payload path escape')
        require(path.is_file() and sha(path) == expected, 'Payload hash mismatch: ' + name)
    for case in manifest['cases']:
        for variant in case['variants'].values():
            require(sha(bundle / variant['ptx']) == sha(bundle / variant['null_ptx']) == variant['ptx_sha256'], 'Nonidentical PTX null')
    checked = (bundle / 'harness/tile_bench_checked.cu').read_text()
    reference = (bundle / 'reference/tile_bench.cu').read_text()
    require(checked[checked.index('  // --- timing'):] == reference[reference.index('  // --- timing'):], 'Changed timing loop')
    require('cublasGetMathMode' in checked and 'reported_math_mode !=' in checked, 'No checked library math readback')
    return manifest

def expected_records(manifest, profile):
    selected = [case for case in manifest['cases'] if case['profile'] == profile]
    gates = [case for case in selected if case['stage'] == 'gate']
    timing = [case for case in selected if case['stage'] == 'timing']
    require(len(gates) == 27 and len(timing) == 30, 'Wrong selected inventory')
    expected = []
    for phase, cases, arms in [('full_gate', gates, ARMS[:2]), ('sampled_gate', timing, ARMS[:2]), ('timing', timing, ARMS)]:
        for case in cases:
            for arm in arms:
                expected.append((case['id'] + '__' + phase + '__' + arm, case, phase, arm))
    require(len(expected) == 234, 'Verifier inventory error')
    return gates, timing, expected

def verify_process(results, prefix, receipt, observed):
    path = results / (prefix + '.status.json')
    require(path.is_file(), 'Missing process status: ' + prefix)
    actual = json.loads(path.read_text())
    require(actual == receipt, 'Summary/process status disagreement: ' + prefix)
    observed[str(path)] = sha(path)
    for suffix, key in [('.stdout', 'stdout_sha256'), ('.stderr', 'stderr_sha256'), ('.resources.txt', 'resources_sha256')]:
        if key not in actual:
            require(suffix == '.resources.txt', 'Missing raw stream digest')
            continue
        raw = results / (prefix + suffix)
        require(raw.is_file() and sha(raw) == actual[key], 'Raw evidence hash mismatch: ' + prefix + suffix)
        observed[str(raw)] = sha(raw)
    finite(actual['wall_seconds'])
    return actual

def verify_argv(argv, case, phase, arm):
    require(len(argv) >= 11 and Path(argv[0]).name == 'checked', 'Wrong checked binary')
    variant = case['variants'][arm.removesuffix('_null')]
    require(argv[1:4] == [str(case[k]) for k in ('M', 'N', 'K')], 'Wrong command shape')
    relative = variant['null_ptx'] if arm.endswith('_null') else variant['ptx']
    require(Path(argv[4]).as_posix().endswith('/' + relative), 'Wrong command PTX')
    launch = variant['launch']
    require(argv[5:11] == [launch['entry'], *[str(launch[k]) for k in ('gx', 'gy', 'block', 'smem')], '30'], 'Wrong command geometry/repetitions')
    expected_flags = {'exact': [], 'contract': ['--contract'], 'tf32': ['--tf32']}[case['face']]
    if phase == 'full_gate':
        expected_flags += ['--full-values', '--values-only']
    elif phase == 'sampled_gate':
        expected_flags += ['--values-only']
    require(argv[11:] == expected_flags, 'Precision or validation flags differ')
    return variant

def parse_header(stdout, case, variant):
    lines = stdout.splitlines()
    device = one(lines, 'device ')
    library = fields(one(lines, 'libraries '))
    require(int(library['cublas_version']) > 0 and int(library['cuda_runtime_version']) > 0, 'Missing API versions')
    require(library['requested_math_mode'] == ('tf32' if case['face'] == 'tf32' else 'default'), 'Wrong requested math mode')
    require(int(library['reported_math_mode']) >= 0, 'Invalid math readback')
    resource = fields(one(lines, 'mapal kernel='))
    launch = variant['launch']
    require(all(resource.get(key) == value for key, value in {'kernel': launch['entry'], 'grid': f"{launch['gx']}x{launch['gy']}", 'block': str(launch['block']), 'smem': str(launch['smem'])}.items()), 'Runtime launch differs')
    require(all(int(resource[key]) >= 0 for key in ('regs', 'local_bytes')), 'Invalid resource row')
    return {'device': device, 'libraries': library, 'resource': resource}

def host_resources(path):
    text = path.read_text()
    result = {'raw_sha256': sha(path)}
    for label, key in [('Maximum resident set size (kbytes)', 'maximum_rss_kib'), ('User time (seconds)', 'user_seconds'), ('System time (seconds)', 'system_seconds')]:
        match = re.search(re.escape(label) + r':\s*([0-9.]+)', text)
        if match:
            result[key] = finite(match.group(1))
    return result

def parse_stdout(stdout, case, phase, variant):
    """Reparse independently; do not trust summary.parsed or runner functions."""
    lines = stdout.splitlines()
    require(not any('NONFINITE' in line or 'VALUES WRONG' in line for line in lines), 'Failure text on successful process')
    device = one(lines, 'device ')
    library = fields(one(lines, 'libraries '))
    require(int(library['cublas_version']) > 0 and int(library['cuda_runtime_version']) > 0, 'Missing API versions')
    require(library['requested_math_mode'] == ('tf32' if case['face'] == 'tf32' else 'default'), 'Wrong requested math mode')
    require(int(library['reported_math_mode']) >= 0, 'Invalid math readback')
    resource = fields(one(lines, 'mapal kernel='))
    launch = variant['launch']
    expected = {'kernel': launch['entry'], 'grid': f"{launch['gx']}x{launch['gy']}", 'block': str(launch['block']), 'smem': str(launch['smem'])}
    require(all(resource.get(key) == value for key, value in expected.items()), 'Runtime launch differs')
    require(all(int(resource[key]) >= 0 for key in ('regs', 'local_bytes')), 'Invalid resource row')
    full = phase == 'full_gate'
    count = case['M'] * case['N'] if full else 256
    scope = fields(one(lines, 'validation '))
    require(scope == {'scope': 'full' if full else 'sampled', 'outputs': str(count)}, 'Wrong numeric scope')
    require(sum(line.startswith('values ') for line in lines) == 3, 'Wrong value-leg inventory')
    values = {}
    for leg in LEGS:
        value = fields(one(lines, 'values leg=' + leg + ' '))
        require(value['cells'] == str(count), 'Wrong checked cell count')
        finite(value['maxrel_vs_f64'])
        values[leg] = value
    gate = 'within_4x_cublas_tf32' if case['face'] == 'tf32' else 'bit_exact_vs_in_order_fold'
    require(values['mapal'].get(gate) == 'yes', 'Numeric gate not admitted')
    result = {'device': device, 'libraries': library, 'resource': resource, 'values': values, 'scope': scope}
    if phase != 'timing':
        marker = 'FULL_VALUES_PASSED' if full else 'SAMPLED_VALUES_PASSED'
        require(lines.count(marker) == 1, 'Missing admission completion')
        require(not any(line.startswith('samples ') or line.startswith('time ') for line in lines), 'Admission gate unexpectedly timed')
        return result
    require(sum(line.startswith('time ') for line in lines) == 3, 'Wrong time-leg inventory')
    samples = fields(one(lines, 'samples '))
    require(set(samples) == set(LEGS), 'Wrong raw-sample legs')
    result['times'] = {}
    for leg in LEGS:
        row = fields(one(lines, 'time leg=' + leg + ' '))
        require(row['n'] == '30' and all(row[k] == str(case[k]) for k in ('M', 'N', 'K')), 'Wrong timed shape/count')
        raw = [finite(value, True) for value in samples[leg].split(',')]
        require(len(raw) == 30, 'Not thirty raw events')
        ordered = sorted(raw)
        for key, expected_number in [('min_ms', ordered[0]), ('med_ms', ordered[15]), ('max_ms', ordered[-1])]:
            require(float(row[key]) == expected_number, 'Printed/raw statistic differs')
        result['times'][leg] = {'samples_ms': raw, 'median_ms': ordered[15], 'min_ms': ordered[0], 'max_ms': ordered[-1]}
    ratio = fields(one(lines, 'ratio '))
    require(all(finite(ratio['mapal_vs_' + leg], True) > 0 for leg in ('cublas', 'cutlass')), 'Invalid ratio row')
    return result

def symmetric(a, b):
    return max(a / b, b / a)

def compare_cell(case, parsed):
    if not all(arm in parsed for arm in ARMS):
        return {'admitted': False, 'verdict': 'not admitted / incomplete', 'timed_arms_present': list(parsed), 'old_process_medians_ms': [parsed[a]['times']['mapal']['median_ms'] if a in parsed else None for a in ('old','old_null')], 'candidate_process_medians_ms': [parsed[a]['times']['mapal']['median_ms'] if a in parsed else None for a in ('candidate','candidate_null')], 'partial_processes': parsed}
    old = [parsed[arm]['times']['mapal']['median_ms'] for arm in ('old', 'old_null')]
    candidate = [parsed[arm]['times']['mapal']['median_ms'] for arm in ('candidate', 'candidate_null')]
    interval = [min(old) / max(candidate), max(old) / min(candidate)]
    same = case['variants']['old']['ptx_sha256'] == case['variants']['candidate']['ptx_sha256']
    nulls = {variant: {leg: parsed[variant]['times'][leg]['median_ms'] / parsed[variant + '_null']['times'][leg]['median_ms'] for leg in LEGS} for variant in ('old', 'candidate')}
    paired = {leg: [parsed[arm]['times'][leg]['median_ms'] / parsed[arm]['times']['mapal']['median_ms'] for arm in ('candidate', 'candidate_null')] for leg in ('cublas', 'cutlass')}
    return {'admitted': True, 'unchanged_ptx_control': same, 'old_process_medians_ms': old, 'candidate_process_medians_ms': candidate,
            'old_over_candidate_center': statistics.median(old) / statistics.median(candidate), 'old_over_candidate_cross_process_interval': interval,
            'old_over_candidate_abba_pairs': [old[0] / candidate[0], old[1] / candidate[1]],
            'candidate_library_over_mapal_pairs': paired, 'identical_ptx_null_ratios': nulls,
            'mapal_null_factor': max(symmetric(*old), symmetric(*candidate)),
            'processes': {arm: {'times': parsed[arm]['times'], 'resource': parsed[arm]['resource'], 'libraries': parsed[arm]['libraries'], 'process_wall_seconds': parsed[arm].get('process_wall_seconds'), 'host_resources': parsed[arm].get('host_resources')} for arm in ARMS}}

def analyze(bundle, results, expected_digest=SEALED_MANIFEST, outer_summary=None):
    manifest = verify_bundle(bundle, expected_digest)
    summary_path = results / 'summary.json'
    summary = json.loads(summary_path.read_text())
    require(summary['manifest_sha256'] == expected_digest and summary['instrumented'] is False, 'Wrong manifest/instrumented run')
    gates, timing, expected = expected_records(manifest, summary['profile'])
    actual = summary['records']
    require(len({r['id'] for r in actual}) == len(actual), 'Duplicate observed record')
    expected_ids = [item[0] for item in expected]
    require([r['id'] for r in actual] == expected_ids[:len(actual)], 'Unexpected record identity/order')
    observed = {str(summary_path): sha(summary_path)}
    outer_files_verified = 0
    binary_hashes_verified = {}
    if outer_summary:
        outer = json.loads(outer_summary.read_text())
        root = outer_summary.parent
        for name, digest in outer['files'].items():
            raw = (root / name).resolve()
            require(raw.is_relative_to(root.resolve()) and raw.is_file() and sha(raw) == digest, 'Outer retrieval hash differs: ' + name)
        require(outer.get('phase_environment', {}).get('smallm_tf32_override') is None, 'Outer TF32 override differs')
        observed[str(outer_summary)] = sha(outer_summary)
        outer_files_verified = len(outer['files'])
    for name in ('checked', 'reference'):
        if name in summary.get('binaries', {}):
            require((results / name).is_file() and sha(results / name) == summary['binaries'][name], 'Harness binary hash mismatch')
            binary_hashes_verified[name] = sha(results / name)
    # Bind builds/toolchain and negative controls independently, including no NaN suppression in checked copy.
    for name in ('device', 'toolchain', 'build-checked', 'build-reference'):
        path = results / (name + '.status.json')
        if path.exists():
            receipt = verify_process(results, name, json.loads(path.read_text()), observed)
            require(receipt['exit'] == 0, 'Preparatory command failed: ' + name)
            if name.startswith('build-'):
                source = 'harness/tile_bench_checked.cu' if name == 'build-checked' else 'reference/tile_bench.cu'
                argv = receipt['argv']
                require(argv[:3] == ['nvcc', '-O3', '-std=c++17'] and source in argv and '-lcublas' in argv and '-lcuda' in argv, 'Unpinned harness build')
    negatives = []
    for name, code, marker in [('reference_nan', 0, 'within_4x_cublas_tf32=yes'), ('checked_nan', 104, 'NONFINITE'), ('checked_wrong', 1, 'MAPAL VALUES WRONG')]:
        path = results / (name + '.status.json')
        if not path.exists():
            negatives.append({'name': name, 'verified': False, 'reason': 'missing'})
            continue
        receipt = verify_process(results, name, json.loads(path.read_text()), observed)
        text = (results / (name + ('.stdout' if code == 0 else '.stderr'))).read_text()
        require(receipt['exit'] == code and marker in text, 'Negative control changed: ' + name)
        negatives.append({'name': name, 'verified': True})
    for path in results.glob(summary['profile'] + '__*.status.json'):
        require(path.name.removesuffix('.status.json') in {r['id'] for r in actual}, 'Unrecorded process status present')
    checked_records = {}
    versions = set()
    modes = {'default': set(), 'tf32': set()}
    devices = set()
    for record, (rid, case, phase, arm) in zip(actual, expected):
        require(record['case'] == case['id'] and record['stage'] == phase and record['arm'] == arm, 'Record metadata differs')
        result = {'id': rid, 'case': case['id'], 'shape': [case[k] for k in ('M','N','K')], 'face': case['face'], 'phase': phase, 'arm': arm, 'pass': False}
        if 'outcome' not in record and 'process' not in record:
            require(record is actual[-1] and record['pass'] is False and summary.get('fatal_error'), 'Unexplained unstarted record')
            require(not (results / (rid + '.status.json')).exists(), 'Unstarted record has status receipt')
            raw_lengths = {}
            for suffix in ('.stdout', '.stderr', '.resources.txt'):
                path = results / (rid + suffix)
                if path.exists():
                    observed[str(path)] = sha(path)
                    raw_lengths[suffix] = path.stat().st_size
            outcome = 'launch_not_started' if all(length == 0 for length in raw_lengths.values()) else 'unreceipted_incomplete_process'
            result.update(outcome=outcome, reason=summary['fatal_error'], raw_lengths=raw_lengths, status_receipt_present=False)
        elif record['outcome'] == 'quarantined_skipped':
            require(phase != 'full_gate' and 'process' not in record and record['pass'] is False, 'Skipped record ran or passed')
            require(not any((results / (rid + ext)).exists() for ext in ('.status.json', '.stdout', '.stderr')), 'Hidden process for skipped cell')
            result.update(outcome='quarantined_skipped', reason=record.get('reason'))
        else:
            receipt = verify_process(results, rid, record['process'], observed)
            variant = verify_argv(receipt['argv'], case, phase, arm)
            stdout = (results / (rid + '.stdout')).read_text()
            stderr = (results / (rid + '.stderr')).read_text()
            if receipt['exit'] == 0:
                try:
                    parsed = parse_stdout(stdout, case, phase, variant)
                except (ValueError, KeyError, TypeError) as error:
                    require(record['pass'] is False and record['outcome'] == 'protocol_failed', 'Raw malformed success was admitted')
                    result.update(outcome='protocol_failed', error=str(error))
                    checked_records[rid] = result
                    continue
                require(record['pass'] is True and record['outcome'] == 'passed', 'Raw success/summary failure disagreement')
                parsed['process_wall_seconds'] = receipt['wall_seconds']
                resources_path = results / (rid + '.resources.txt')
                if resources_path.exists():
                    parsed['host_resources'] = host_resources(resources_path)
                result.update(outcome='passed', pass_value=True, parsed=parsed, wall_seconds=receipt['wall_seconds'])
                result['pass'] = True
                versions.add((parsed['libraries']['cublas_version'], parsed['libraries']['cuda_runtime_version']))
                modes[parsed['libraries']['requested_math_mode']].add(parsed['libraries']['reported_math_mode'])
                devices.add(parsed['device'])
            else:
                numeric = (receipt['exit'] == 1 and 'MAPAL VALUES WRONG' in stderr) or (receipt['exit'] == 104 and 'NONFINITE' in stderr)
                outcome = 'numeric_not_admitted' if numeric else 'process_failed'
                require(record['pass'] is False and record['outcome'] == outcome, 'Raw failure/summary outcome differs')
                require(not any(line.startswith('time ') or line.startswith('samples ') for line in stdout.splitlines()), 'Failed admission emitted timing')
                if numeric:
                    result['header'] = parse_header(stdout, case, variant)
                result.update(outcome=outcome, exit=receipt['exit'], diagnostics=[line for line in stdout.splitlines() if line.startswith('values ')], stderr=stderr)
        checked_records[rid] = result
    require(len(versions) <= 1 and len(devices) <= 1, 'Mixed library/device versions within one run')
    require(all(len(values) <= 1 for values in modes.values()), 'Inconsistent math readback')
    require(not (modes['default'] & modes['tf32']), 'Default and TF32 readback modes are indistinguishable')
    def record(case, phase, arm):
        return checked_records.get(case['id'] + '__' + phase + '__' + arm, {'pass': False, 'outcome': 'missing'})
    gate_by_id = {case['id']: case for case in gates}
    rows = []
    regressions = []
    for case in timing:
        full = None
        if case.get('gate_id'):
            gate_case = gate_by_id[case['gate_id']]
            full = {'shape': [gate_case[k] for k in ('M', 'N', 'K')], 'variants': {arm: record(gate_case, 'full_gate', arm) for arm in ARMS[:2]}}
        sampled = {arm: record(case, 'sampled_gate', arm) for arm in ARMS[:2]}
        full_ok = full is None or all(item['pass'] for item in full['variants'].values())
        sampled_ok = all(item['pass'] for item in sampled.values())
        if not full_ok:
            require(all(item['outcome'] in ('quarantined_skipped', 'missing') for item in sampled.values()), 'Sampled a cell after failed bounded gate')
        elif any(item['outcome'] == 'quarantined_skipped' for item in sampled.values()):
            raise ValueError('Unexplained sampled-gate quarantine')
        for phase, pair in [('full_gate', full['variants'] if full else {}), ('sampled_gate', sampled)]:
            if pair and pair['old']['pass'] and not pair['candidate']['pass']:
                regressions.append({'case': case['id'], 'phase': phase})
        parsed = {}
        for arm in ARMS:
            item = record(case, 'timing', arm)
            if item.get('pass'):
                require(full_ok and sampled_ok, 'Timed a nonadmitted cell')
                parsed[arm] = item['parsed']
            elif item['outcome'] == 'quarantined_skipped':
                require(not (full_ok and sampled_ok) or any(not record(case, 'timing', prior)['pass'] for prior in ARMS[:ARMS.index(arm)]), 'Unexplained timing quarantine')
        row = {'id': case['id'], 'family': case['family'], 'M': case['M'], 'N': case['N'], 'K': case['K'], 'face': case['face'], 'variant_code': {key: {'ptx_sha256': value['ptx_sha256'], 'launch': value['launch'], 'route': value['route']} for key,value in case['variants'].items()}, 'bounded_full_gate': full, 'actual_sampled_gate': sampled, **compare_cell(case, parsed)}
        if not row['admitted']:
            reason = 'bounded full gate' if not full_ok else 'actual 256-sample gate' if not sampled_ok else 'timing revalidation / incomplete'
            row['verdict'] = 'not admitted: ' + reason
        row['resource_changes'] = {arm: (record(case, 'timing', arm).get('parsed') or sampled.get(arm.removesuffix('_null'), {}).get('parsed') or record(case, 'timing', arm).get('header') or {}).get('resource') for arm in ARMS}
        rows.append(row)
    controls = [row for row in rows if row['admitted'] and row['unchanged_ptx_control']]
    global_control_factor = max([1.0] + [max(row['old_over_candidate_cross_process_interval'][1], 1 / row['old_over_candidate_cross_process_interval'][0]) for row in controls])
    for row in rows:
        if not row['admitted']:
            continue
        threshold = max(row['mapal_null_factor'], global_control_factor)
        row['observed_noise_factor'] = threshold
        lo, hi = row['old_over_candidate_cross_process_interval']
        if row['unchanged_ptx_control']:
            row['verdict'] = 'unchanged PTX control; differences are not a compiler gain'
        elif lo > threshold:
            row['verdict'] = 'observed improvement beyond null/control variation'
        elif hi < 1 / threshold:
            row['verdict'] = 'observed regression beyond null/control variation'
        else:
            row['verdict'] = 'no gain established beyond observed null/control variation'
        row['library_verdicts'] = {}
        for leg, pairs in row['candidate_library_over_mapal_pairs'].items():
            own_library_null = max(row['identical_ptx_null_ratios']['candidate'][leg], 1 / row['identical_ptx_null_ratios']['candidate'][leg])
            noise = max(threshold, own_library_null)
            row['library_verdicts'][leg] = 'candidate faster beyond observed variation' if min(pairs) > noise else ('library faster beyond observed variation' if max(pairs) < 1 / noise else 'unresolved within observed variation')
    complete = len(actual) == 234
    require(summary.get('completed_records', len(actual)) == len(actual), 'Completed-record count differs')
    if complete:
        require(summary.get('inventory_complete') is True and all(item['verified'] for item in negatives), 'Completed run missing inventory/negative-control proof')
        require(set(summary.get('binaries', {})) == {'checked', 'reference'}, 'Completed run missing binaries')
        require(all((results / (name + '.status.json')).is_file() for name in ('device','toolchain','build-checked','build-reference')), 'Completed run missing build/toolchain status')
    if regressions:
        require(summary.get('candidate_acceptance') is not True, 'Candidate regression falsely accepted')
    return {'verification_counts': {'sealed_payload_files': len(manifest['files']), 'outer_files': outer_files_verified, 'harness_binaries': len(binary_hashes_verified), 'hashed_record_and_metadata_files': len(observed)}, 'harness_binary_hashes_verified': binary_hashes_verified, 'evidence_kind': 'synthetic fixture' if summary.get('synthetic') else 'retrieved GPU artifacts', 'manifest_sha256': expected_digest, 'profile': summary['profile'], 'complete_inventory': complete, 'observed_records': len(actual), 'expected_records': 234,
            'combined_source_provenance': manifest.get('integrated_candidate_provenance'), 'emitter_identities': manifest['emitters'], 'negative_controls': negatives, 'all_negative_controls_verified': all(item['verified'] for item in negatives), 'library_versions': sorted(versions), 'reported_math_modes': {key: sorted(value) for key, value in modes.items()},
            'candidate_regressions_at_admission': regressions, 'global_unchanged_ptx_control_factor': global_control_factor if controls else None, 'fatal_execution_error': summary.get('fatal_error'), 'primary_numeric_admissions':sum(item['pass'] for item in checked_records.values() if item['phase'] in ('full_gate','sampled_gate')), 'primary_timing_processes':sum(item['pass'] for item in checked_records.values() if item['phase']=='timing'), 'performance_comparison_available':any(row['admitted'] for row in rows),
            'rows': rows, 'verified_records': checked_records, 'artifact_hashes': observed,
            'limits': ['Every-cell ordered proof applies only to each recorded bounded gate shape; full timing shapes check 256 sampled cells.', 'TF32 uses the unchanged 4x cuBLAS diagnostic gate, not an absolute accuracy guarantee; library errors remain diagnostics.', 'Two ABBA process observations and thirty events each are descriptive; null/control envelopes are not confidence intervals.', 'cuBLAS mode readback does not identify its executed kernel arithmetic. Driver JIT machine-image capture is separate.', 'No default switch or automated production acceptance is performed.'],
            'source_reference': 'https://docs.nvidia.com/cuda/archive/13.0.1/cublas/index.html'}

def write_report(out, result):
    out.mkdir(parents=True, exist_ok=False)
    (out / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    headers = ['family','M','N','K','face','admitted','old_process_medians_ms','candidate_process_medians_ms','old_over_candidate_cross_process_interval','candidate_library_over_mapal_pairs','verdict']
    with (out / 'all-cells.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        for row in result['rows']:
            writer.writerow({key: row.get(key, '') for key in headers})
    process_fields = ['case','phase','arm','M','N','K','face','outcome','mapal_min_ms','mapal_med_ms','mapal_max_ms','cublas_med_ms','cutlass_med_ms','wall_seconds','regs','local_bytes','smem','block','grid','maximum_rss_kib','user_seconds','system_seconds']
    with (out / 'processes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=process_fields)
        writer.writeheader()
        for record in result['verified_records'].values():
            parsed = record.get('parsed', {})
            resource = parsed.get('resource', record.get('header', {}).get('resource', {}))
            host = parsed.get('host_resources', {})
            times = parsed.get('times', {})
            row = {'case':record['case'],'phase':record['phase'],'arm':record['arm'],'M':record['shape'][0],'N':record['shape'][1],'K':record['shape'][2],'face':record['face'],'outcome':record['outcome'],'wall_seconds':parsed.get('process_wall_seconds',record.get('wall_seconds',''))}
            row.update({key:resource.get(key,'') for key in ('regs','local_bytes','smem','block','grid')})
            row.update({key:host.get(key,'') for key in ('maximum_rss_kib','user_seconds','system_seconds')})
            for leg in LEGS:
                if leg in times:
                    row[leg+'_med_ms'] = times[leg]['median_ms']
                    if leg == 'mapal':
                        row['mapal_min_ms'] = times[leg]['min_ms']
                        row['mapal_max_ms'] = times[leg]['max_ms']
            writer.writerow(row)
    lines = [f"Observed {result['observed_records']}/{result['expected_records']} scheduled records; {result['primary_numeric_admissions']} primary numeric admissions and {result['primary_timing_processes']} timing processes; {sum(row['admitted'] for row in result['rows'])}/30 cells admitted.", '', '| Family / M / face | Old ms A/B | Candidate ms A/B | Old/candidate envelope | cuBLAS/candidate A/B | CUTLASS/candidate A/B | Verdict |', '|---|---:|---:|---:|---:|---:|---|']
    def pair(values):
        return '/'.join('—' if value is None else f'{value:.4f}' for value in values)
    for row in result['rows']:
        label = f"{row['family']} / {row['M']} / {row['face']}"
        if row['admitted']:
            libraries = row['candidate_library_over_mapal_pairs']
            lines.append('| ' + ' | '.join([label, pair(row['old_process_medians_ms']), pair(row['candidate_process_medians_ms']), pair(row['old_over_candidate_cross_process_interval']), pair(libraries['cublas']), pair(libraries['cutlass']), row['verdict']]) + ' |')
        else:
            lines.append(f"| {label} | {pair(row['old_process_medians_ms'])} | {pair(row['candidate_process_medians_ms'])} | — | — | — | {row['verdict']} |")
    lines += ['', 'A/B are the two ABBA process observations, not a pooled 60-event median. Event min/median/max, host wall/RSS and GPU resource changes are in processes.csv; every raw event is retained in analysis.json.', '', ('Unchanged-PTX control variation factor: not measured.' if result['global_unchanged_ptx_control_factor'] is None else f"Unchanged-PTX control variation factor: {result['global_unchanged_ptx_control_factor']:.4f}."), '', *result['limits']]
    if result.get('fatal_execution_error'):
        lines.insert(1, 'Execution stopped: ' + result['fatal_execution_error'])
    if not result['performance_comparison_available']:
        lines.insert(1, 'No admitted performance comparison exists; no speedup, library ranking or numeric-quarantine claim is supported.')
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=Path('/private/tmp/mapal-s58-smallm-bench'))
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--expected-manifest', default=SEALED_MANIFEST)
    parser.add_argument('--outer-summary', type=Path)
    args = parser.parse_args()
    result = analyze(args.bundle.resolve(), args.results.resolve(), args.expected_manifest, args.outer_summary)
    write_report(args.out, result)
    print(json.dumps({'complete': result['complete_inventory'], 'records': result['observed_records'], 'admitted_cells': sum(row['admitted'] for row in result['rows']), 'out': str(args.out)}))
if __name__ == '__main__':
    main()
