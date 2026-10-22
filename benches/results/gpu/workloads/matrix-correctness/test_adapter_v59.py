"""Subprocess-only stubs; no GPU/network/rental/install."""
import sys
sys.dont_write_bytecode = True
import contextlib,importlib.util,io,json,subprocess,tempfile
from pathlib import Path
from unittest.mock import patch
P=Path('/private/tmp/mapal-s58-gpu');spec=importlib.util.spec_from_file_location('pair',P/'staging-v59/run_checks.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);results=[]
modes=['success','ordering-failure','ordering-summary-failure','ordering-timeout','ordering-verify-failure','candidate-ordering-failure','candidate-ordering-summary-failure','candidate-ordering-timeout','candidate-ordering-verify-failure','candidate-ordering-payload-check-failure','matrix-failure','torch-install-failure']
for mode in modes:
 with tempfile.TemporaryDirectory() as tmp:
  m.ROOT=Path(tmp);m.OUT=m.ROOT/'trace/correctness-pair'
  class Child:
   def __init__(self,args,**kwargs):self.args=list(map(str,args));self.returncode=None;self.pid=-1;self.stage=Path(kwargs['stdout'].name).parent.name
   def poll(self):return self.returncode
   def wait(self,timeout=None):
    if self.stage in {'ordering','candidate-ordering'}:
     folder=m.OUT/('ordering-values' if self.stage=='ordering' else 'candidate-ordering-values');folder.mkdir();(folder/'summary.json').write_text(json.dumps({'pass':mode not in {self.stage+'-failure',self.stage+'-summary-failure'}}))
     if mode==self.stage+'-timeout':raise subprocess.TimeoutExpired(self.args,timeout)
    if self.stage=='matrix':
     folder=m.OUT/'matrix-values';folder.mkdir();(folder/'verification.json').write_text(json.dumps({'numeric_pass':mode!='matrix-failure'}))
    self.returncode=int(mode==self.stage+'-failure');return self.returncode
  with patch.object(m.subprocess,'Popen',Child),patch.object(m,'stop',lambda c:setattr(c,'returncode',-15)),contextlib.redirect_stdout(io.StringIO()):code=m.main()
  summary=json.loads((m.OUT/'summary.json').read_text());stages=[r['name'] for r in summary['commands']]
  assert summary['pass']==(mode=='success') and code==(0 if mode=='success' else 1),(mode,summary)
  if mode.startswith('ordering-') or mode.startswith('candidate-ordering-'):
   assert 'apt-update' not in stages and 'matrix' not in stages,(mode,stages)
  if mode.startswith('ordering-') and mode!='ordering-verify-failure':assert 'candidate-ordering' not in stages
  if mode=='success':
   assert stages.index('ordering')<stages.index('ordering-verify')<stages.index('candidate-ordering')<stages.index('candidate-ordering-verify')<stages.index('torch-install')<stages.index('matrix')
   rows={r['name']:r for r in summary['commands']};a=rows['ordering-verify']['argv'];b=rows['candidate-ordering-verify']['argv'];assert a[-1]!=b[-1] and a[2]!=b[2] and a[3]!=b[3]
   assert summary['ordering_pass'] and summary['candidate_ordering_pass']
  if mode.endswith('-timeout'):assert summary['commands'][-1]['timed_out'] is True
  assert all((m.OUT/n/'status.json').is_file() for n in stages)
  results.append({'mode':mode,'pass':True,'stages':stages})
receipt={'pass':True,'tests':results,'no_gpu_network_rental_install':True};(P/'adapter-tests-v59.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'pass':True,'modes':len(results)}))
