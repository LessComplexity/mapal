import subprocess, pathlib, json, hashlib, re
root=pathlib.Path('/Volumes/LessComplex/Personal/mapal')
out=pathlib.Path('/private/tmp/mapal-gpu-schedule-facts')
old=out/'baseline-target/debug/examples/emit'
new=root/'target/debug/examples/emit'
llvm=pathlib.Path('/opt/homebrew/opt/llvm/bin')
programs=out/'programs'
# Real recognized one-row strided-lane site.
(programs/'strided.mapal').write_text('''fn main() {
    128 -> iota -> ta;
    ta -> map { t -> t -> widen_f32 } -> a;
    32768 -> iota -> tb;
    tb -> map { t -> t -> widen_f32 } -> b;
    128 -> iota -> krange;
    256 -> iota -> tc;
    tc -> map { j -> (2.0, krange) -> fold { acc, k -> a[k] * b[j * 128 + k] + acc } } -> c;
    c[0] -> println;
}
''')
(programs/'mixed.mapal').write_text('fn main() {\n    32768 -> iota -> ta;\n    ta -> map { t -> t % 31 -> widen_f32 } -> a;\n    ta -> map { t -> t % 23 -> widen_f32 } -> b;\n    128 -> iota -> krange;\n    256 -> iota -> tc;\n    tc -> map { j -> (2.0, krange) -> fold { acc, k -> a[k] * b[j * 128 + k] + acc } } -> c;\n    32768 -> iota -> td;\n    td -> map { t -> t / 256 -> i; t % 256 -> j;\n        (0.0, krange) -> fold { acc, k -> acc + a[i * 128 + k] * b[k * 256 + j] }\n    } -> d;\n    c[0] -> println;\n    d[0] -> println;\n}\n')
cases=[
 ('scalar-exact','mm_1024','rtx-4090','sm_89',[],True),
 ('scalar-contract','mm_1024','rtx-4090','sm_89',['--contract'],True),
 ('mma-default','mm_1024','rtx-4090','sm_89',['--contract','--precision=tf32'],True),
 ('wgmma-default','mm_1024','h100-pcie','sm_90a',['--contract','--precision=tf32'],True),
 ('mma-hopper-wgmma-off','mm_1024','h100-pcie','sm_90a',['--contract','--precision=tf32','--gpu-off=wgmma-hopper'],True),
 ('mma-forced','mm_1024','rtx-4090','sm_89',['--contract','--precision=tf32','--gpu-off=arch-rungs'],True),
 ('wgmma-forced','mm_1024','h100-pcie','sm_90a',['--contract','--precision=tf32','--gpu-off=arch-rungs'],True),
 ('scalar-naive','odd_100','rtx-4090','sm_89',[],True),
 ('mixed-pdl','mixed','h100-pcie','sm_90a',['--contract','--precision=tf32','--gpu-on=pdl'],False),
 ('strided-contract','strided','h100-pcie','sm_90a',['--contract'],False),
 ('layout-fallback','at_256','rtx-3090','sm_86',['--contract','--precision=tf32'],False),
 ('type-fallback','rm_128_f64','rtx-3090','sm_86',['--contract','--precision=tf32'],False),
 ('mma-off-fallback','mm_1024','rtx-3090','sm_86',['--contract','--precision=tf32','--gpu-off=mma'],False),
]
records=[]
for name,source,target,sm,flags,unchanged in cases:
  rec={'case':name,'source':source,'target':target,'sm':sm,'flags':flags,'expect_unchanged':unchanged}
  for era,compiler in [('before',old),('after',new)]:
    d=out/'emissions'/era
    d.mkdir(parents=True,exist_ok=True)
    stem=d/name
    with stem.with_suffix('.host.ll').open('w') as host:
      subprocess.run([str(compiler),str(programs/(source+'.mapal')),'-','--rewrite','--target='+target,'--device-out='+str(stem.with_suffix('.device.ll')),*flags],stdout=host,check=True)
    subprocess.run([str(llvm/'opt'),'-O3',str(stem.with_suffix('.device.ll')),'-S','-o',str(stem.with_suffix('.opt.ll'))],check=True)
    subprocess.run([str(llvm/'llc'),'-O3','-mcpu='+sm,'-nvptx-fma-level=0',str(stem.with_suffix('.opt.ll')),'-o',str(stem.with_suffix('.ptx'))],check=True)
    ll=stem.with_suffix('.device.ll').read_text()
    ptx=stem.with_suffix('.ptx').read_text()
    rec[era]={'fmul':len(re.findall(r' = fmul ',ll)),'fadd':len(re.findall(r' = fadd ',ll)),'fma':len(re.findall(r'call (?:float|double) @llvm.fma',ll)), 'mma':ll.count('mma.sync.aligned'),'wgmma':ll.count('wgmma.mma_async'), 'pdl_waits':ll.count('griddepcontrol.wait'), 'ptx_fma':len(re.findall(r'\bfma.rn',ptx))}
  equal={}
  for ext in ['host.ll','device.ll','opt.ll','ptx']:
    b=(out/'emissions/before'/(name+'.'+ext)).read_bytes()
    a=(out/'emissions/after'/(name+'.'+ext)).read_bytes()
    if ext == 'opt.ll':
      b=b.replace(str(out/'emissions/before').encode(),b'EMISSION_DIR')
      a=a.replace(str(out/'emissions/after').encode(),b'EMISSION_DIR')
    equal[ext]=b==a
  rec['equal']=equal
  if unchanged: assert all(equal.values()), rec
  else:
    assert not equal['device.ll'],rec
    assert rec['before']==rec['after'],rec
  records.append(rec)
  print(name, equal, flush=True)
(out/'emission-comparison.json').write_text(json.dumps(records,indent=2)+'\n')
(out/'source-receipt.json').write_text(json.dumps({'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'baseline_gpu_sha256':hashlib.sha256((out/'baseline/crates/backends/llvm/src/func/gpu.rs').read_bytes()).hexdigest(),'current_gpu_sha256':hashlib.sha256((root/'crates/backends/llvm/src/func/gpu.rs').read_bytes()).hexdigest(),'baseline_binary_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),'current_binary_sha256':hashlib.sha256(new.read_bytes()).hexdigest()},indent=2)+'\n')
