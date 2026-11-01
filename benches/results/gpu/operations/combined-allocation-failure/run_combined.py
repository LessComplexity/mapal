"""One reviewed sequential GPU attempt; absolute lease deadline includes startup."""
import hashlib,json,os,pathlib,signal,subprocess,sys,time
ROOT=pathlib.Path('/w');OUT=ROOT/'trace/combined-v61'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def stop(child):
 if child is None or child.poll() is not None:return
 try:os.killpg(child.pid,signal.SIGTERM)
 except ProcessLookupError:return
 try:child.wait(timeout=5)
 except subprocess.TimeoutExpired:
  try:os.killpg(child.pid,signal.SIGKILL)
  except ProcessLookupError:pass
  child.wait(timeout=5)

def main():
 OUT.mkdir(parents=True,exist_ok=False);lease=json.loads((ROOT/'lease.json').read_text());deadline=lease['absolute_deadline_epoch'];reserve=180;records=[]
 summary={'smallm_pass':False,'compiled_pass':False,'compiled_state':'not_run','timing_admitted':False,'commands':records,'absolute_deadline_epoch':deadline,'retrieval_cleanup_reserve_seconds':reserve,'smallm_max_seconds':1100,'compiled_minimum_start_seconds':240,'outer_timeout_aborts_all_later_gpu':True}
 blocked_prefixes=('PIP_','MAPAL_','CUBLAS_','CUBLASLT_','TORCH_','CUPTI_','NSYS_','NCU_','CUDA_INJECTION','NVBIT_','NVPROF_','NVTX_INJECTION')
 blocked_keys={'NVIDIA_TF32_OVERRIDE','LD_PRELOAD','CUDA_PROFILE','COMPUTE_PROFILE','NVPROF_CONFIG_FILE','CUDA_LAUNCH_BLOCKING','CUDA_FORCE_PTX_JIT','CUDA_CACHE_DISABLE','CUDA_MODULE_LOADING','CUDA_DEVICE_MAX_CONNECTIONS','LD_AUDIT','CUDA_PROFILE_CONFIG','CUDA_PROFILE_LOG','CUDA_PROFILE_CSV','COMPUTE_PROFILE_CONFIG','COMPUTE_PROFILE_LOG','COMPUTE_PROFILE_CSV','CUDA_DEVICE_WAITS_ON_EXCEPTION'}
 env={k:v for k,v in os.environ.items() if not k.startswith(blocked_prefixes) and k not in blocked_keys}
 env.update(PYTHONDONTWRITEBYTECODE='1',PIP_CONFIG_FILE='/dev/null')
 compiled_policy={'NVIDIA_TF32_OVERRIDE':'0','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
 summary['phase_environment']={'smallm_removed_keys':sorted(set(os.environ)-set(env)),'smallm_tf32_override':None,'compiled_policy':compiled_policy}
 def remaining():return int(deadline-time.time()-reserve)
 def run(name,args,cap,extra=None,cwd=ROOT):
  folder=OUT/name;folder.mkdir();limit=min(cap,remaining());row={'name':name,'argv':list(map(str,args)),'timeout_seconds':limit,'exit':None};child=None;started=time.monotonic()
  try:
   if limit<=0:raise TimeoutError('Lease reserve reached before '+name)
   with (folder/'stdout').open('wb')as so,(folder/'stderr').open('wb')as se:
    child=subprocess.Popen(row['argv'],cwd=cwd,env={**env,**(extra or {})},stdout=so,stderr=se,start_new_session=True)
    try:row['exit']=child.wait(timeout=limit)
    except subprocess.TimeoutExpired:row['timed_out']=True;stop(child);raise TimeoutError('Outer phase timeout; abort all later GPU work: '+name)
  finally:
   stop(child);row['seconds']=round(time.monotonic()-started,3)
   for n in ['stdout','stderr']:
    if (folder/n).exists():row[n+'_sha256']=sha(folder/n)
   (folder/'status.json').write_text(json.dumps(row,indent=2)+'\n');records.append(row);print(json.dumps({'stage':name,'exit':row['exit'],'timed_out':row.get('timed_out',False)}),flush=True)
  return row['exit']==0
 def idle(name):
  if not run(name,['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],15):raise ValueError('GPU idle query failed')
  if (OUT/name/'stdout').read_text().strip():raise ValueError('GPU process remains; abort later GPU work')
 try:
  if lease.get('host_hourly_rate',1)>.8 or lease['absolute_deadline_epoch']-lease['create_started_epoch']>1800:raise ValueError('Lease cost/deadline contract differs')
  immutable=json.loads((ROOT/'sealed-files.json').read_text())
  for n,h in immutable.items():
   if sha(ROOT/n)!=h:raise ValueError('Sealed input changed: '+n)
  if not run('smallm-payload-check',[sys.executable,ROOT/'smallm-packet/run_packet.py','--bundle',ROOT/'smallm-packet','--profile','rtx-5090','--check-only'],60):raise ValueError('SmallM packet check failed')
  if not run('matrix-payload-check',[sys.executable,ROOT/'matrix-upload/check_payload.py'],60):raise ValueError('Frozen matrix payload check failed')
  idle('gpu-idle-before-smallm');cap=min(1100,remaining()-10)
  if cap<720:raise ValueError('Insufficient absolute lease time for smallM; no GPU phases admitted')
  smallm_ok=run('smallm',[sys.executable,ROOT/'smallm-packet/run_packet.py','--bundle',ROOT/'smallm-packet','--profile','rtx-5090','--results',OUT/'smallm-values','--host-hourly-rate',str(lease['host_hourly_rate']),'--total-timeout',str(cap)],cap+10)
  report=OUT/'smallm-values/summary.json'
  if report.exists():summary['smallm_summary']=json.loads(report.read_text())
  summary['smallm_pass']=smallm_ok and summary.get('smallm_summary',{}).get('pass') is True
  # A completed numerical failure may still leave time for the independent checker.
  idle('gpu-idle-after-smallm')
  if remaining()<240:summary['compiled_not_run_reason']='Less than240seconds before retrieval/cleanup reserve';return 1
  summary['compiled_state']='bootstrap_started'
  python=ROOT/'torch-venv/bin/python'
  bootstrap=[
   ('apt-update',['apt-get','update'],90),
   ('apt-python-development',['apt-get','install','-y','--no-install-recommends','python3-venv','python3-dev'],90),
   ('venv',[sys.executable,'-m','venv',ROOT/'torch-venv'],30),
   ('python-header-gate',[python,'-c',"import json,pathlib,sysconfig; p=pathlib.Path(sysconfig.get_paths()['include'])/'Python.h'; print(json.dumps({'sysconfig_include':str(p.parent),'python_h':str(p),'exists':p.is_file()})); assert p.is_file(), 'Python.h missing from sysconfig include'"],15),
   ('pip-upgrade',[python,'-m','pip','--isolated','install','--no-deps','--upgrade','https://files.pythonhosted.org/packages/44/3c/d717024885424591d5376220b5e836c2d5293ce2011523c9de23ff7bf068/pip-25.3-py3-none-any.whl#sha256=9655943313a94722b7774661c21049070f6bbb0a1516bf02f7c8d5d9201514cd'],60),
   ('pip-version',[python,'-c',"import pip; print(pip.__version__); assert pip.__version__=='25.3'"],15),
   ('torch-install',[python,'-m','pip','--isolated','install','--index-url','https://download.pytorch.org/whl/cu130','--report',OUT/'torch-install-report.json','torch==2.13.0'],240),
   ('torch-version',[python,'-c',"import json,torch; r={'torch':torch.__version__,'cuda':torch.version.cuda,'count':torch.cuda.device_count(),'device':torch.cuda.get_device_name(0)}; print(json.dumps(r)); assert r['torch'].split('+')[0]=='2.13.0' and r['cuda']=='13.0' and r['count']==1 and 'RTX 5090' in r['device']"],30),
   ('pip-freeze',[python,'-m','pip','freeze','--all'],15),
   ('system-packages',['dpkg-query','-W'],15),
  ]
  for n,a,c in bootstrap:
   if not run(n,a,c,extra=compiled_policy):raise ValueError('Explicit bootstrap failed: '+n)
  idle('gpu-idle-before-compiled')
  if remaining()<45:summary['compiled_state']='not_run';summary['compiled_not_run_reason']='Less than45seconds for compiled checker';return 1
  summary['compiled_state']='running';compiled=OUT/'compiled-values'
  ok=run('compiled',[python,ROOT/'matrix-upload/matrix-packet/pytorch/run_reference.py','--case','matrix-m16-d8-h16','--mode','compiled','--scope','matrix','--device','cuda:0','--out',compiled],remaining(),extra={**compiled_policy,'TORCH_LOGS':'graph_breaks,recompiles,output_code','TORCH_COMPILE_DEBUG':'1','TORCHINDUCTOR_CACHE_DIR':str(OUT/'torch-cache-compiled')},cwd=OUT)
  report=compiled/'report.json'
  if report.exists():summary['compiled_report']=json.loads(report.read_text())
  summary['compiled_pass']=ok and summary.get('compiled_report',{}).get('pass') is True and summary.get('compiled_report',{}).get('run_final_gate_pass') is True
  summary['compiled_state']='passed' if summary['compiled_pass'] else 'failed'
  summary['optional_capture']={'status':'skipped-insufficient-time'}
  if remaining()>=120:
   idle('gpu-idle-before-optional-capture')
   summary['optional_capture']={'status':'started-unvalidated'}
   proof_ok=run('optional-capture',[sys.executable,ROOT/'capture-proof/proof.py','--root',ROOT,'--bundle',ROOT/'smallm-packet','--combined-out',OUT,'--out',OUT/'optional-capture-proof','--after-all-ordinary'],90)
   proof_status=OUT/'optional-capture-proof/status.json'
   summary['optional_capture']=json.loads(proof_status.read_text()) if proof_status.exists() else {'status':'missing-status','process_pass':proof_ok}
 except Exception as error:
  summary['error']=type(error).__name__+': '+str(error)
  if records and records[-1]['name']=='optional-capture' and records[-1].get('timed_out'):
   summary['optional_capture']={'status':'outer-timeout-unvalidated'}
 finally:
  summary['pass']=summary['smallm_pass'] and summary['compiled_pass'];summary['files']={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()};data=json.dumps(summary,indent=2)+'\n';(OUT/'summary.json').write_text(data);(ROOT/'results-rtx-5090.txt').write_text(data)
 return 0 if summary['pass'] else 1
if __name__=='__main__':raise SystemExit(main())
