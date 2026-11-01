from pathlib import Path
import json,hashlib,re
root=Path('/private/tmp/mapal-gpu-schedule-facts');out=Path('/private/tmp/mapal-gpu-schedule-review');records=json.loads((root/'emission-comparison.json').read_text());proof=[]
for record in records:
 case=record['case'];row={'case':case,'unchanged_required':record['expect_unchanged'],'files':{}}
 for suffix in ('host.ll','device.ll','opt.ll','ptx'):
  paths=[root/'emissions'/era/(case+'.'+suffix)for era in ('before','after')];data=[p.read_bytes()for p in paths];equal=data[0]==data[1]
  if suffix=='opt.ll':
   normalized=[]
   for era,raw in zip(('before','after'),data):
    lines=raw.splitlines(keepends=True);norm=[]
    for line in lines:
     if line.startswith((b'; ModuleID = ',b'source_filename = ')):line=line.replace(str(root/'emissions'/era).encode(),b'EMISSION_DIR')
     norm.append(line)
    normalized.append(b''.join(norm))
   equal=normalized[0]==normalized[1]
  assert equal==record['equal'][suffix],(case,suffix)
  if record['expect_unchanged']:assert equal,(case,suffix)
  row['files'][suffix]={'before_sha256':hashlib.sha256(data[0]).hexdigest(),'after_sha256':hashlib.sha256(data[1]).hexdigest(),'equal':equal,'normalization':'ModuleID/source_filename path only'if suffix=='opt.ll'else None}
 assert record['before']==record['after'],case
 if not record['expect_unchanged']:assert not row['files']['device.ll']['equal'],case
 row['arithmetic_inventory']=record['after'];proof.append(row)
assert len(records)==13 and sum(r['expect_unchanged']for r in records)==8
source=json.loads((root/'source-receipt.json').read_text());assert hashlib.sha256((out/'gpu-before.rs').read_bytes()).hexdigest()==source['baseline_gpu_sha256'];assert hashlib.sha256((out/'gpu-reviewed.rs').read_bytes()).hexdigest()==source['current_gpu_sha256'];assert hashlib.sha256(Path('/Volumes/LessComplex/Personal/mapal/crates/backends/llvm/src/func/gpu.rs').read_bytes()).hexdigest()==source['current_gpu_sha256']
for era,key,path in [('baseline','baseline_binary_sha256',root/'baseline-target/debug/examples/emit'),('current','current_binary_sha256',Path('/Volumes/LessComplex/Personal/mapal/target/debug/examples/emit'))]:assert hashlib.sha256(path.read_bytes()).hexdigest()==source[key]
integ=(root/'integration-tests.log').read_text();checks=re.findall(r'test result: ok\. (\d+) passed; (\d+) failed;',integ);assert all(int(f)==0 for _,f in checks)
report={'pass':True,'reviewed_gpu_sha256':source['current_gpu_sha256'],'source_and_compiler_hashes_verified':True,'files_compared':len(records)*8,'unchanged_successful_controls':8,'intentional_policy_changes_with_unchanged_arithmetic_inventory':5,'mixed_module':next(x for x in proof if x['case']=='mixed-pdl'),'integration_completed_test_groups':len(checks),'integration_completed_passes':sum(int(p)for p,_ in checks),'emissions':proof,'gpu_executed':False,'limits':['Control equality establishes finite compile-time output only','Five changed policy emissions have unchanged arithmetic instruction counts; this alone does not prove runtime values','No new GPU numerical or performance claim']}
(out/'emission-review.json').write_text(json.dumps(report,indent=2)+'\n');print({k:v for k,v in report.items()if k not in ('emissions','mixed_module')})
