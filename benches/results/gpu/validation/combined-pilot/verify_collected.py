"""CPU-only raw-integrity check; no numerical/timing admission or GPU execution."""
import sys
sys.dont_write_bytecode=True
import hashlib,json,pathlib,tarfile
P=pathlib.Path('/private/tmp/mapal-s58-gpu/combined-v62');R=P/'results';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(pathlib.Path(p).read_text())
assert read(R/'cleanup.json')['confirmed_absent']is True
out=P/'retrieved';out.mkdir(exist_ok=False)
with tarfile.open(R/'trace-rtx-5090.tgz')as t:
 for m in t.getmembers():assert not m.issym()and not m.islnk()and not m.name.startswith('/')and'..'not in pathlib.Path(m.name).parts and(m.isfile()or m.isdir())
 t.extractall(out,filter='data')
trace=out/'trace/combined-v61';summary=read(R/'results-rtx-5090.txt');assert read(trace/'summary.json')==summary
for n,h in summary['files'].items():assert sha(trace/n)==h,n
proof={'raw_integrity_pass':True,'files_verified':len(summary['files']),'adapter_sha256':sha(P/'staging/run_combined.py'),'trace_sha256':sha(R/'trace-rtx-5090.tgz'),'results_sha256':sha(R/'results-rtx-5090.txt'),'smallm_reported_pass':summary['smallm_pass'],'compiled_reported_pass':summary['compiled_pass'],'compiled_state':summary['compiled_state'],'optional_capture_status':summary.get('optional_capture',{}).get('status'),'error':summary.get('error'),'timing_admitted_by_this_verifier':False,'cleanup':read(R/'cleanup.json')};(R/'raw-integrity.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
