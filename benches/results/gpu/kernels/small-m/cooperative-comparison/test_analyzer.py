"""Synthetic mutation tests for the independent verifier; no CUDA/network."""
import copy, importlib.util, json, tempfile, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('independent', ROOT / 'analyze.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
BUNDLE = Path('/private/tmp/mapal-s58-smallm-bench')

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((BUNDLE / 'manifest.json').read_text())
        cls.case = next(c for c in cls.manifest['cases'] if c['profile'] == 'rtx-5090' and c['stage'] == 'timing' and c['M'] == 16 and c['face'] == 'contract')

    def output(self, case, phase, arm):
        v = case['variants'][arm.removesuffix('_null')]
        l = v['launch']
        full = phase == 'full_gate'
        mode = 'tf32' if case['face'] == 'tf32' else 'default'
        count = case['M'] * case['N'] if full else 256
        lines = ['device name="NVIDIA RTX 5090" cc=12.0 driver=13030', 'facts sms=170',
                 f'libraries cublas_version=130001 cuda_runtime_version=13000 requested_math_mode={mode} reported_math_mode={3 if mode == "tf32" else 0}',
                 f'mapal kernel={l["entry"]} grid={l["gx"]}x{l["gy"]} block={l["block"]} smem={l["smem"]} regs=32 local_bytes=0',
                 f'validation scope={"full" if full else "sampled"} outputs={count}']
        gate = 'within_4x_cublas_tf32' if case['face'] == 'tf32' else 'bit_exact_vs_in_order_fold'
        for leg in a.LEGS:
            lines.append(f'values leg={leg} cells={count} maxrel_vs_f64=0.001' + (f' {gate}=yes' if leg == 'mapal' else ''))
        if phase != 'timing':
            lines.append('FULL_VALUES_PASSED' if full else 'SAMPLED_VALUES_PASSED')
        else:
            changed = case['variants']['old']['ptx_sha256'] != case['variants']['candidate']['ptx_sha256']
            samples = {}
            for leg in a.LEGS:
                center = 2.0 if changed and arm in ('old', 'old_null') and leg == 'mapal' else 1.0
                raw = [center * .9] * 14 + [center] * 2 + [center * 1.1] * 14
                samples[leg] = ','.join(f'{v:.4f}' for v in raw)
                lines.append(f'time leg={leg} M={case["M"]} N={case["N"]} K={case["K"]} n=30 min_ms={min(raw):.4f} med_ms={center:.4f} max_ms={max(raw):.4f}')
            lines.append('ratio mapal_vs_cublas=1.000 mapal_vs_cutlass=1.000')
            lines.append('samples ' + ' '.join(f'{leg}={samples[leg]}' for leg in a.LEGS))
        return '\n'.join(lines) + '\n'

    def test_raw_parser_and_mutations(self):
        text = self.output(self.case, 'timing', 'candidate')
        parsed = a.parse_stdout(text, self.case, 'timing', self.case['variants']['candidate'])
        self.assertEqual(len(parsed['times']['mapal']['samples_ms']), 30)
        mutations = [text.replace('n=30', 'n=29', 1), text.replace('maxrel_vs_f64=0.001', 'maxrel_vs_f64=nan', 1),
                     text.replace('requested_math_mode=default', 'requested_math_mode=tf32'),
                     text.replace('med_ms=1.0000', 'med_ms=2.0000', 1), text + next(x for x in text.splitlines() if x.startswith('samples '))]
        for mutated in mutations:
            with self.subTest(mutated=mutated[-40:]), self.assertRaises((ValueError, KeyError)):
                a.parse_stdout(mutated, self.case, 'timing', self.case['variants']['candidate'])

    def test_precision_argv_mutation(self):
        c = self.case; v = c['variants']['candidate']; l = v['launch']
        argv = ['/w/checked', *[str(c[k]) for k in ('M','N','K')], '/w/' + v['ptx'], l['entry'], *[str(l[k]) for k in ('gx','gy','block','smem')], '30', '--contract']
        a.verify_argv(argv, c, 'timing', 'candidate')
        with self.assertRaises(ValueError):
            a.verify_argv(argv[:-1] + ['--tf32'], c, 'timing', 'candidate')

    def test_sampled_not_full(self):
        with self.assertRaises(ValueError):
            a.parse_stdout(self.output(self.case, 'sampled_gate', 'candidate'), self.case, 'full_gate', self.case['variants']['candidate'])

    def test_no_win_inside_null_noise(self):
        parsed = {arm: a.parse_stdout(self.output(self.case, 'timing', arm), self.case, 'timing', self.case['variants'][arm.removesuffix('_null')]) for arm in a.ARMS}
        parsed['candidate']['times']['mapal']['median_ms'] = 1.8
        parsed['candidate_null']['times']['mapal']['median_ms'] = 2.2
        c = a.compare_cell(self.case, parsed)
        self.assertLess(c['old_over_candidate_cross_process_interval'][0], c['mapal_null_factor'])

    def fixture(self, root):
        gates, timing, inventory = a.expected_records(self.manifest, 'rtx-5090')
        records = []
        def process(prefix, argv, stdout='', stderr='', code=0, resources=False):
            (root / (prefix + '.stdout')).write_text(stdout)
            (root / (prefix + '.stderr')).write_text(stderr)
            receipt = {'argv': argv, 'exit': code, 'wall_seconds': .25, 'stdout_sha256': a.sha(root / (prefix + '.stdout')), 'stderr_sha256': a.sha(root / (prefix + '.stderr'))}
            if resources:
                path = root / (prefix + '.resources.txt'); path.write_text('Maximum resident set size (kbytes): 50000\nUser time (seconds): 0.12\nSystem time (seconds): 0.02\n')
                receipt['resources_sha256'] = a.sha(path)
            (root / (prefix + '.status.json')).write_text(json.dumps(receipt))
            return receipt
        for binary in ('checked','reference'):
            (root / binary).write_bytes(b'synthetic nonexecutable binary')
        for prefix in ('device','toolchain','build-checked','build-reference'):
            source = 'harness/tile_bench_checked.cu' if prefix == 'build-checked' else 'reference/tile_bench.cu'
            argv = ['nvcc','-O3','-std=c++17','-arch=sm_120',source,'-lcublas','-lcuda'] if prefix.startswith('build-') else ['synthetic']
            process(prefix, argv)
        process('reference_nan',['synthetic'],stdout='values leg=mapal within_4x_cublas_tf32=yes\n')
        process('checked_nan',['synthetic'],stderr='NONFINITE\n',code=104)
        process('checked_wrong',['synthetic'],stderr='MAPAL VALUES WRONG\n',code=1)
        for rid, case, phase, arm in inventory:
            v = case['variants'][arm.removesuffix('_null')]; l = v['launch']
            flags = {'exact':[],'contract':['--contract'],'tf32':['--tf32']}[case['face']]
            if phase == 'full_gate': flags += ['--full-values','--values-only']
            elif phase == 'sampled_gate': flags += ['--values-only']
            path = v['null_ptx'] if arm.endswith('_null') else v['ptx']
            argv = ['/w/checked',*[str(case[k]) for k in ('M','N','K')],'/w/'+path,l['entry'],*[str(l[k]) for k in ('gx','gy','block','smem')],'30',*flags]
            receipt = process(rid, argv, self.output(case, phase, arm), resources=True)
            records.append({'id':rid,'case':case['id'],'stage':phase,'arm':arm,'pass':True,'outcome':'passed','process':receipt})
        summary = {'synthetic':True,'manifest_sha256':a.SEALED_MANIFEST,'instrumented':False,'profile':'rtx-5090','records':records,'binaries':{name:a.sha(root/name)for name in ('checked','reference')},'inventory_complete':True,'completed_records':234,'candidate_acceptance':True}
        (root/'summary.json').write_text(json.dumps(summary))
        return summary

    def test_complete_independent_fixture_and_raw_hash_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); summary=self.fixture(root)
            result=a.analyze(BUNDLE,root)
            self.assertEqual(len(result['rows']),30)
            self.assertTrue(result['complete_inventory'])
            self.assertEqual(sum(row['unchanged_ptx_control'] for row in result['rows']),23)
            first=summary['records'][0]['id']; (root/(first+'.stdout')).write_text('mutated')
            with self.assertRaises(ValueError): a.analyze(BUNDLE,root)

    def test_duplicate_record_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); summary=self.fixture(root);summary['records'].append(copy.deepcopy(summary['records'][0]));(root/'summary.json').write_text(json.dumps(summary))
            with self.assertRaises(ValueError):a.analyze(BUNDLE,root)

    def test_quarantined_gate_cannot_have_timing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); summary=self.fixture(root)
            record=next(r for r in summary['records']if r['stage']=='sampled_gate'and r['case']==self.case['id']and r['arm']=='candidate')
            prefix=record['id'];(root/(prefix+'.stderr')).write_text('MAPAL VALUES WRONG\n');record['pass']=False;record['outcome']='numeric_not_admitted';record['process']['exit']=1;record['process']['stderr_sha256']=a.sha(root/(prefix+'.stderr'));(root/(prefix+'.status.json')).write_text(json.dumps(record['process']));summary['candidate_acceptance']=False;(root/'summary.json').write_text(json.dumps(summary))
            with self.assertRaises(ValueError):a.analyze(BUNDLE,root)

    def test_prelaunch_file_not_found_is_not_numeric_quarantine(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);summary=self.fixture(root)
            record={key:summary['records'][0][key]for key in ('id','case','stage','arm','pass')};record['pass']=False
            summary.update(records=[record],completed_records=1,inventory_complete=False,candidate_acceptance=False,fatal_error="FileNotFoundError: [Errno 2] No such file or directory: '/usr/bin/time'")
            for path in root.glob('rtx-5090__*'):path.unlink()
            for suffix in ('.stdout','.stderr'):(root/(record['id']+suffix)).write_text('')
            (root/'summary.json').write_text(json.dumps(summary))
            result=a.analyze(BUNDLE,root)
            self.assertFalse(result['complete_inventory']);self.assertFalse(result['performance_comparison_available']);self.assertEqual(result['primary_numeric_admissions'],0);self.assertEqual(result['primary_timing_processes'],0)
            self.assertEqual(result['verified_records'][record['id']]['outcome'],'launch_not_started')
            self.assertEqual(len(result['rows']),30);self.assertIsNone(result['global_unchanged_ptx_control_factor'])
            out=root/'analysis';a.write_report(out,result);self.assertIn('No admitted performance comparison', (out/'REPORT.md').read_text())

if __name__=='__main__':unittest.main()
