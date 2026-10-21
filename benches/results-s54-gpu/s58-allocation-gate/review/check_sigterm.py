"""Actual CLI SIGTERM propagation with fake hardware and a real sleeping cc child."""
import argparse,pathlib,tempfile,subprocess,os,sys,time,json,signal
P=pathlib.Path('/private/tmp/mapal-s58-allocation-gate-review');a=argparse.ArgumentParser();a.add_argument('packet',type=pathlib.Path);a.add_argument('--expect-fixed',action='store_true');a.add_argument('--out',type=pathlib.Path,required=True);args=a.parse_args();result={};outer=None;pids={}
with tempfile.TemporaryDirectory(dir=P)as tmp:
 root=pathlib.Path(tmp);tools=root/'tools';tools.mkdir();pidfile=root/'children.json'
 def tool(name,body):
  p=tools/name;p.write_text('#!'+sys.executable+'\n'+body);p.chmod(0o755)
 tool('nvidia-smi',"print('name, uuid, driver_version, power.limit')\nprint('NVIDIA RTX 5090, GPU-FAKE-LOCAL-ONLY, fake, 0 W')\n")
 tool('objdump',"print('fake local disassembler')\n")
 tool('cc',"import pathlib,os,json,time\npathlib.Path(os.environ['REVIEW_CHILD_PIDFILE']).write_text(json.dumps({'cc':os.getpid(),'phase':os.getppid()}))\ntime.sleep(60)\n")
 env={k:v for k,v in os.environ.items()if not k.startswith('MAPAL_')and k not in {'LD_PRELOAD','CUDA_INJECTION64_PATH','CUDA_VISIBLE_DEVICES'}};env.update(PATH=str(tools)+os.pathsep+env['PATH'],PYTHONDONTWRITEBYTECODE='1',REVIEW_CHILD_PIDFILE=str(pidfile))
 try:
  with(root/'outer.stdout').open('wb')as so,(root/'outer.stderr').open('wb')as se:
   outer=subprocess.Popen([sys.executable,'-B',str(args.packet/'run_gate.py'),str(args.packet),str(root/'results')],stdout=so,stderr=se,env=env,start_new_session=True)
   deadline=time.monotonic()+10
   while not pidfile.exists()and time.monotonic()<deadline:
    if outer.poll()is not None:raise AssertionError((root/'outer.stderr').read_text())
    time.sleep(.02)
   assert pidfile.exists(),'fake compiler child did not start';pids=json.loads(pidfile.read_text());os.kill(outer.pid,signal.SIGTERM);outer.wait(timeout=8)
  for name,pid in pids.items():
   try:os.kill(pid,0);alive=True
   except ProcessLookupError:alive=False
   result[name+'_dead_after_outer_sigterm']=not alive
 finally:
  for pid in pids.values():
   try:os.killpg(pid,signal.SIGKILL)
   except ProcessLookupError:pass
  if outer:
   if outer.poll()is None:outer.kill()
   outer.wait()
result.update(no_gpu_network_rental=True,all_fixture_processes_stopped=True);args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if args.expect_fixed:assert result['cc_dead_after_outer_sigterm']and result['phase_dead_after_outer_sigterm']
