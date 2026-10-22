import sys
sys.dont_write_bytecode=True
import contextlib,importlib.util,io,json,pathlib,subprocess,tempfile,time
from unittest.mock import patch
P=pathlib.Path(__file__).parent;s=importlib.util.spec_from_file_location('combined',P/'run_combined.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);results=[]
for mode in ['success','smallm-failed','smallm-timeout','compiled-timeout','header-missing','torch-version-failed','insufficient-after-smallm','gpu-process-after-smallm','compiled-numeric-failed','sealed-input-changed','apt-update-timeout','insufficient-after-bootstrap','optional-validated','optional-capture-timeout']:
 with tempfile.TemporaryDirectory()as tmp:
  m.ROOT=pathlib.Path(tmp);m.OUT=m.ROOT/'trace/combined-v61';epoch=time.time();clock={'offset':0};(m.ROOT/'lease.json').write_text(json.dumps({'create_started_epoch':epoch,'absolute_deadline_epoch':epoch+1800,'host_hourly_rate':.66}));(m.ROOT/'sealed-files.json').write_text(json.dumps({'bad':'x'}if mode=='sealed-input-changed'else{}))
  class Child:
   def __init__(self,args,**kw):
    self.args=list(map(str,args));self.returncode=None;self.pid=-1;self.stage=pathlib.Path(kw['stdout'].name).parent.name;self.stdout=kw['stdout'];self.env=kw['env']
    if self.stage=='smallm':assert 'NVIDIA_TF32_OVERRIDE' not in self.env and not any(k.startswith(('MAPAL_','CUBLAS_','CUBLASLT_','CUPTI_','NSYS_','NCU_','TORCH_','CUDA_INJECTION','NVBIT_','NVPROF_','NVTX_INJECTION')) for k in self.env) and not any(k in self.env for k in ['LD_PRELOAD','LD_AUDIT','CUDA_LAUNCH_BLOCKING','CUDA_PROFILE_LOG'])
    if self.stage=='compiled':assert self.env['NVIDIA_TF32_OVERRIDE']=='0' and self.env['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
   def poll(self):return self.returncode
   def wait(self,timeout=None):
    st=self.stage
    if st=='system-packages' and mode=='insufficient-after-bootstrap':clock['offset']=1580
    if mode==st+'-timeout':raise subprocess.TimeoutExpired(self.args,timeout)
    if st=='smallm':
     folder=m.OUT/'smallm-values';folder.mkdir();(folder/'summary.json').write_text(json.dumps({'pass':mode!='smallm-failed'}))
     if mode=='insufficient-after-smallm':clock['offset']=1400
    if st=='gpu-idle-after-smallm' and mode=='gpu-process-after-smallm':self.stdout.write(b'1234\n');self.stdout.flush()
    if st=='optional-capture' and mode=='optional-validated':
     folder=m.OUT/'optional-capture-proof';folder.mkdir();(folder/'status.json').write_text(json.dumps({'status':'validated-observation-only'}))
    if st=='compiled':
     folder=m.OUT/'compiled-values';folder.mkdir();(folder/'report.json').write_text(json.dumps({'pass':mode!='compiled-numeric-failed','run_final_gate_pass':mode!='compiled-numeric-failed'}))
    self.returncode=int((st=='smallm' and mode=='smallm-failed')or(st=='python-header-gate' and mode=='header-missing')or(st=='torch-version' and mode=='torch-version-failed'));return self.returncode
  with patch.dict(m.os.environ,{'NVIDIA_TF32_OVERRIDE':'0','CUBLAS_WORKSPACE_CONFIG':':16:8','MAPAL_GPU_TRACE':'1','LD_PRELOAD':'bogus','CUPTI_ACTIVITY':'1','TORCH_LOGS':'all','CUDA_INJECTION64_PATH':'bogus','CUBLASLT_LOG_LEVEL':'5','NVBIT_TOOL':'bogus','NVPROF_TMPDIR':'bogus','NVTX_INJECTION64_PATH':'bogus','LD_AUDIT':'bogus','CUDA_LAUNCH_BLOCKING':'1','CUDA_PROFILE_LOG':'bogus'}),patch.object(m.subprocess,'Popen',Child),patch.object(m,'stop',lambda c:setattr(c,'returncode',-15)if c else None),patch.object(m.time,'time',lambda:epoch+clock['offset']),contextlib.redirect_stdout(io.StringIO()):code=m.main()
  summary=json.loads((m.OUT/'summary.json').read_text());stages=[r['name']for r in summary['commands']]
  assert summary['pass']==(mode in {'success','optional-validated','optional-capture-timeout'}) and code==(0 if mode in {'success','optional-validated','optional-capture-timeout'}else 1),(mode,summary)
  if mode=='smallm-failed':assert 'compiled'in stages and summary['compiled_pass']
  if mode in {'smallm-timeout','insufficient-after-smallm','gpu-process-after-smallm','sealed-input-changed'}:assert not any(x in stages for x in ['apt-update','torch-install','compiled']),(mode,stages)
  if mode in {'apt-update-timeout','insufficient-after-bootstrap'}:assert 'compiled'not in stages
  if mode=='header-missing':assert 'torch-install'not in stages and 'compiled'not in stages
  if mode=='torch-version-failed':assert 'compiled'not in stages
  if mode=='optional-validated':assert summary['optional_capture']['status']=='validated-observation-only'
  if mode=='optional-capture-timeout':assert summary['optional_capture']['status']=='outer-timeout-unvalidated' and summary['compiled_pass'] and summary['commands'][-1]['timed_out'] and summary['commands'][-1]['name']=='optional-capture'
  if mode=='compiled-timeout':assert summary['compiled_pass'] is False and summary['commands'][-1]['timed_out']
  if mode=='success':assert stages.index('smallm')<stages.index('python-header-gate')<stages.index('torch-version')<stages.index('compiled')
  results.append({'mode':mode,'pass':True,'stages':stages})
(P/'adapter-tests.json').write_text(json.dumps({'pass':True,'tests':results,'no_gpu_network_rental':True},indent=2)+'\n');print(json.dumps({'pass':True,'modes':len(results)}))
