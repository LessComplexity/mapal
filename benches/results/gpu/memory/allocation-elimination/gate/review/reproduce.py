import sys
sys.dont_write_bytecode=True
import importlib.util,pathlib,subprocess,tempfile,os,signal,time,json,shutil
from unittest.mock import patch
P=pathlib.Path('/private/tmp/mapal-s58-allocation-gate-review');source=pathlib.Path('/private/tmp/mapal-s58-allocation-gate/packet');spec=importlib.util.spec_from_file_location('gate',source/'run_gate.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
with tempfile.TemporaryDirectory(dir=P) as tmp:
 root=pathlib.Path(tmp);pidfile=root/'grandchild.pid';fixture=root/'phase.py';fixture.write_text("import subprocess,sys,pathlib,time\nc=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)\npathlib.Path(sys.argv[1]).write_text(str(c.pid))\ntime.sleep(60)\n")
 real_popen=subprocess.Popen;parents=[]
 class Proxy:
  def __init__(self,args,**kw):self.child=real_popen([sys.executable,str(fixture),str(pidfile)],**kw);self.pid=self.child.pid;parents.append(self.child)
  def wait(self,timeout=None):return self.child.wait(timeout=min(timeout,.5)if timeout is not None else None)
  def poll(self):return self.child.poll()
 survivor=None
 try:
  with patch.object(g,'check',lambda _:{}),patch.object(g.subprocess,'Popen',Proxy):
   try:g.run(root,root/'out')
   except ValueError:pass
  survivor=int(pidfile.read_text());os.kill(survivor,0);escaped=True
 finally:
  if survivor:
   try:os.kill(survivor,signal.SIGKILL)
   except ProcessLookupError:pass
  for c in parents:
   if c.poll()is None:c.kill()
   c.wait()
 clone=P/'packet-copy';shutil.copytree(source,clone);calls=[]
 class Marker(Exception):pass
 def launch_attempt(*a,**k):calls.append(True);raise Marker('launch attempted with output inside packet')
 with patch.object(g.subprocess,'Popen',launch_attempt):
  try:g.run(clone,clone/'inside-results')
  except Marker:pass
 assert calls and(clone/'inside-results/summary.json').exists()
 receipt={'outer_phase_timeout_leaves_independent_session_descendant_alive':escaped,'all_fixture_processes_killed_by_review_cleanup':True,'results_inside_packet_reach_child_launch':bool(calls),'no_gpu_network_rental':True};(P/'reproductions.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
