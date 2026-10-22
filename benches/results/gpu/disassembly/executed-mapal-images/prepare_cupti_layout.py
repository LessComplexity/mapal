#!/usr/bin/env python3
"""CPU-only package/layout/readiness proof for frozen S60 capture; no install/injection."""
import argparse, hashlib, json, os, pathlib, subprocess

p = argparse.ArgumentParser()
p.add_argument('--cuda-root', type=pathlib.Path, required=True)
p.add_argument('--packet', type=pathlib.Path, required=True)
p.add_argument('--out', type=pathlib.Path, required=True)
a = p.parse_args()
cuda = a.cuda_root.resolve(strict=True)
packet = a.packet.resolve(strict=True)
out = a.out.absolute()
assert not out.exists(), 'Exclusive new output directory required'
assert not out.is_relative_to(packet), 'Output must be outside frozen packet'
assert not any(v and (k.startswith(('CUDA_INJECTION', 'CUPTI_', 'NSYS_', 'NCU_', 'NVBIT_', 'NVPROF_', 'MAPAL_')) or k in ['LD_PRELOAD','LD_AUDIT','CUDA_LAUNCH_BLOCKING','CUDA_FORCE_PTX_JIT','CUDA_MODULE_LOADING','NVIDIA_TF32_OVERRIDE']) for k,v in os.environ.items()), 'Observer/control environment present'
out.mkdir(mode=0o700)
receipt = {'pass': False, 'gpu_executed': False, 'installs_attempted': False, 'packet_mutated': False, 'commands': []}
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()

def run(name, argv, env=None, timeout=30):
    r = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=timeout)
    (out/(name+'.stdout')).write_bytes(r.stdout)
    (out/(name+'.stderr')).write_bytes(r.stderr)
    receipt['commands'].append({'name':name,'argv':list(map(str,argv)),'exit':r.returncode,'stdout_sha256':hashlib.sha256(r.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(r.stderr).hexdigest()})
    assert r.returncode == 0, name+' failed'
    return r.stdout.decode()

try:
    sealed = json.loads((packet/'sealed.json').read_text())
    assert all(sha(packet/n)==h for n,h in sealed.items()), 'Frozen packet seal differs'
    for pkg in ['cuda-cupti-12-8', 'cuda-cupti-dev-12-8']:
        version = run(pkg+'-version', ['dpkg-query','-W','-f','${Version}',pkg])
        assert version == '12.8.90-1', 'Unreviewed CUPTI package version: '+version
        run(pkg+'-files', ['dpkg','-L',pkg])
    inc = cuda/'targets/x86_64-linux/include'
    lib = cuda/'targets/x86_64-linux/lib'
    header = inc/'cupti.h'
    library = lib/'libcupti.so'
    assert header.is_file() and library.is_file(), 'Reviewed package layout unavailable'
    receipt['header'] = {'path':str(header.resolve()), 'sha256':sha(header)}
    receipt['library'] = {'path':str(library.resolve()), 'sha256':sha(library.resolve())}
    layout = out/'toolkit'
    (layout/'extras/CUPTI').mkdir(parents=True)
    (layout/'include').symlink_to(inc, target_is_directory=True)
    (layout/'extras/CUPTI/include').symlink_to(inc, target_is_directory=True)
    (layout/'extras/CUPTI/lib64').symlink_to(lib, target_is_directory=True)
    buildenv = {**os.environ, 'CUDA_HOME':str(layout), 'CAPTURE_BUILD':str(out/'observer-probe')}
    run('observer-build', ['bash',str(packet/'capture_images/build.sh')], buildenv)
    observer = out/'observer-probe/capture.so'
    dependencies = run('observer-ldd', ['ldd',str(observer)])
    assert 'not found' not in dependencies, 'Unresolved observer library'
    dynamic = run('observer-dynamic', ['readelf','-d',str(observer)])
    assert str(layout/'extras/CUPTI/lib64') in dynamic, 'Expected explicit library runpath absent'
    assert all(sha(packet/n)==h for n,h in sealed.items()), 'Packet changed during readiness proof'
    receipt.update({'pass':True, 'cuda_home':str(layout), 'observer_sha256':sha(observer), 'sealed_files_verified':len(sealed)})
except Exception as e:
    receipt['error'] = type(e).__name__+': '+str(e)
finally:
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
raise SystemExit(0 if receipt['pass'] else 1)
