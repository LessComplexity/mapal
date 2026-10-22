"""CPU-only local integrity and ordering verification after independent cleanup."""
import pathlib,json,tarfile,hashlib,sys,importlib.util,datetime
sys.dont_write_bytecode=True
P=pathlib.Path('/private/tmp/mapal-s58-gpu');R=P/'results';S=P/'staging-v59';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(pathlib.Path(p).read_text())
assert read(R/'cleanup.json')['confirmed_absent'] is True
out=P/'retrieved';out.mkdir(exist_ok=False)
def extract(archive,dest):
 with tarfile.open(archive) as t:
  for m in t.getmembers():assert not m.issym() and not m.islnk() and not m.name.startswith('/') and '..' not in pathlib.Path(m.name).parts and (m.isfile() or m.isdir()),m.name
  t.extractall(dest,filter='data')
extract(R/'trace-rtx-5090.tgz',out)
inputs=P/'immutable-inputs';inputs.mkdir(exist_ok=False)
for n in ['ordering.tgz','ordering-candidate.tgz','matrix-upload.tgz']:extract(S/n,inputs)
summary=read(R/'results-rtx-5090.txt');trace=out/'trace/correctness-pair';assert read(trace/'summary.json')==summary
for n,h in summary['files'].items():assert sha(trace/n)==h,n
assert summary['runner_sha256']==sha(S/'run_checks.py') and summary['timing_admitted'] is False
spec=importlib.util.spec_from_file_location('verify_order',S/'verify_ordering.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
checks={}
for name,bundle,folder,h in [('frozen','ordering-runtime-bundle','ordering-values','141f7bdca06b2f770761b735b02ae12f6891ba80cb9b461ec88fcd59780209e8'),('candidate','ordering-candidate-bundle','candidate-ordering-values','7b492228ccdce858e3ff26ef6592d67b0035cd8848a5b710329f4bd31e0f260e')]:checks[name]=v.verify(inputs/bundle,trace/folder,h)
matrix=trace/'matrix-values/verification.json';matrix_report=read(matrix) if matrix.exists() else None
if matrix_report:
 for n,h in matrix_report['files'].items():assert sha(matrix.parent/n)==h,n
proof={'pass':summary.get('pass') is True,'ordinary_ordering':checks,'matrix_numeric_pass':summary.get('matrix_numeric_pass') is True,'matrix_proof_complete':False,'matrix_error':matrix_report.get('error') if matrix_report else None,'matrix_proof_gaps':matrix_report.get('proof_gaps') if matrix_report else [],'all_adapter_file_hashes_verified':len(summary['files']),'matrix_file_hashes_verified':len(matrix_report['files']) if matrix_report else 0,'trace_archive_sha256':sha(R/'trace-rtx-5090.tgz'),'results_sha256':sha(R/'results-rtx-5090.txt'),'timing_admitted':False,'cleanup':read(R/'cleanup.json'),'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(R/'local-verification.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
