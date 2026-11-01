#!/usr/bin/env python3
"""Use only in a separately authorized proof run. No ordinary timing admission."""
import argparse,hashlib,json,os,pathlib,subprocess,sys,resource
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--ptx',required=True);p.add_argument('--observer',required=True);p.add_argument('--kernel',required=True);p.add_argument('--grid',required=True);p.add_argument('--block',required=True);p.add_argument('--shared-bytes',required=True,type=int);p.add_argument('--timeout',type=int,default=60);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
if not 1<=a.timeout<=300 or not a.command: p.error('command and timeout 1..300 required')
def dims(value):
    try:parts=[int(x) for x in value.split(',')]
    except ValueError:p.error('dimensions require three integers')
    if len(parts)!=3 or any(not 0<x<=0xffffffff for x in parts):p.error('dimensions require three positive uint32 values')
    return parts
expected_launch={'grid':dims(a.grid),'block':dims(a.block),'shared_bytes':a.shared_bytes}
if not 0<=a.shared_bytes<=0xffffffff:p.error('shared-bytes requires uint32')
command=a.command[1:] if a.command[0]=='--' else a.command
if not command or not pathlib.Path(command[0]).is_absolute():p.error('absolute executable path required')
out=pathlib.Path(a.out).absolute();out.mkdir(mode=0o700) # exclusive destination
ptx=pathlib.Path(a.ptx).resolve(strict=True);observer=pathlib.Path(a.observer).resolve(strict=True);exe=pathlib.Path(command[0]).resolve(strict=True)
def h(x):
    value=hashlib.sha256()
    with x.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):value.update(chunk)
    return value.hexdigest()
def entry(x):return {'path':str(x),'sha256':h(x)}
# Copy only authorized inputs; no environment, credentials or system-wide inventory.
if not 0<ptx.stat().st_size<=16<<20 or not 0<exe.stat().st_size<=512<<20 or not 0<observer.stat().st_size<=16<<20:p.error('input exceeds cap')
env=os.environ.copy();env['CUDA_INJECTION64_PATH']=str(observer);env['MAPAL_CAPTURE_PTX']=str(ptx);env['MAPAL_CAPTURE_OUT']=str(out/'capture')
receipt={'variant':'production-driver-ptx-observed','command':command,'kernel':a.kernel,'expected_launch':expected_launch,'ptx_sha256':h(ptx),'executable':entry(exe),'observer':entry(observer),'timeout':False,'timing_admissible':False}
# Bound each child output file; observer separately caps total cubin bytes.
def limits(): resource.setrlimit(resource.RLIMIT_FSIZE,(16<<20,16<<20))
with (out/'stdout').open('xb') as stdout,(out/'stderr').open('xb') as stderr:
    process=subprocess.Popen(command,env=env,stdout=stdout,stderr=stderr,start_new_session=True,preexec_fn=limits)
    try: receipt['exit_code']=process.wait(timeout=a.timeout)
    except subprocess.TimeoutExpired:
        import signal
        os.killpg(process.pid,signal.SIGKILL);process.wait();receipt.update(timeout=True,exit_code=process.returncode)
receipt['stdout']=entry(out/'stdout');receipt['stderr']=entry(out/'stderr')
if h(exe)!=receipt['executable']['sha256'] or h(observer)!=receipt['observer']['sha256'] or h(ptx)!=receipt['ptx_sha256']:receipt['exit_code']=-999
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
from validate import validate
try:
    admitted=validate(out/'capture',out/'receipt.json')
    (out/'validation.json').write_text(json.dumps(admitted,indent=2)+'\n')
except (ValueError,KeyError,IndexError,OSError,TypeError) as e:sys.exit('REJECTED: '+str(e))
