"""Local real detached-process and path-boundary review; no GPU/network/rental."""
import sys
sys.dont_write_bytecode=True
import argparse,importlib.util,json,pathlib,subprocess,tempfile,os,signal,shutil
from unittest.mock import patch
P=pathlib.Path('/private/tmp/mapal-s58-allocation-gate-review');a=argparse.ArgumentParser();a.add_argument('packet',type=pathlib.Path);a.add_argument('--expect-fixed',action='store_true');a.add_argument('--out',type=pathlib.Path,required=True);args=a.parse_args();source=args.packet.resolve();spec=importlib.util.spec_from_file_location('gate_reviewed',source/'run_gate.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g);result={}
with tempfile.TemporaryDirectory(dir=P)as tmp:
 root=pathlib.Path(tmp);pidfile=root/'nested.pid';fixture=root/'phase.py';fixture.write_text("import sys,pathlib\nsys.dont_write_bytecode=True\nsys.path.insert(0,sys.argv[1])\nfrom box_ordering import execute\nroot=pathlib.Path(sys.argv[2]);code='import pathlib,time,sys;pathlib.Path(sys.argv[1]).write_text(str(__import__(\\\"os\\\").getpid()));time.sleep(60)'\nexecute([sys.executable,'-c',code,str(root/'nested.pid')],root/'inner.stdout',root/'inner.stderr',60)\n")
 real_popen=subprocess.Popen;parents=[]
 class Proxy:
  def __init__(self,command,**kw):
   self.first=True;self.child=None
   if'--check-only'not in list(map(str,command)):
    self.child=real_popen([sys.executable,str(fixture),str(source),str(root)],**kw);parents.append(self.child)
   self.pid=self.child.pid if self.child else None
  def wait(self,timeout=None):
   if self.child is None:return 0
   first=self.first;self.first=False;return self.child.wait(timeout=min(timeout,.5)if first and timeout is not None else timeout)
  def poll(self):return self.child.poll()if self.child else 0
 survivor=None
 try:
  with patch.object(g,'check',lambda _:{}),patch.object(g.subprocess,'Popen',Proxy):
   try:g.run(source,root/'out')
   except ValueError:pass
  survivor=int(pidfile.read_text())
  try:os.kill(survivor,0);alive=True
  except ProcessLookupError:alive=False
  result['nested_child_dead_after_outer_phase_timeout']=not alive
 finally:
  if survivor:
   try:os.kill(survivor,signal.SIGKILL)
   except ProcessLookupError:pass
  for c in parents:
   if c.poll()is None:c.kill()
   c.wait()
 pathchecks={}
 for kind in ['nested','equal','symlink']:
  clone=root/('packet-'+kind);shutil.copytree(source,clone);calls=[]
  if kind=='nested':out=clone/'inside-results'
  elif kind=='equal':out=clone
  else:
   alias=root/'packet-alias';alias.symlink_to(clone,target_is_directory=True);out=alias/'inside-results'
  class Marker(Exception):pass
  def launch(*unused,**kw):calls.append(True);raise Marker('child launch reached')
  error=None
  with patch.object(g.subprocess,'Popen',launch):
   try:g.run(clone,out)
   except Exception as e:error=type(e).__name__
  pathchecks[kind]={'rejected_before_child_launch':not calls,'error':error,'new_result_directory_created':kind!='equal'and out.exists()}
 result['path_checks']=pathchecks
result['no_gpu_network_rental']=True;result['all_fixture_processes_stopped']=True;args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if args.expect_fixed:assert result['nested_child_dead_after_outer_phase_timeout']and all(x['rejected_before_child_launch']and x['error']=='ValueError'and not x['new_result_directory_created']for x in result['path_checks'].values())
