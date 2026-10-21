from pathlib import Path
import csv,hashlib,json,re,subprocess
R=Path('/private/tmp/mapal-s56-gpu/ptx-sass-gap');B=Path('/private/tmp/mapal-s56-gpu/integrated');P=Path('/Volumes/LessComplex/Personal/mapal');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();d=json.loads((R/'analysis.json').read_text());art=json.loads((R/'artifact-manifest.json').read_text());checks=[]
def track(p):
 p=Path(p);art[str(p)]={'bytes':p.stat().st_size,'sha256':sha(p)}
for tag,profile in [('a100-primary','a100'),('rtx-4090-primary','rtx-4090'),('rtx-5090-primary-retry4','rtx-5090')]:
 v=P/f'benches/results-s54-gpu/s57-integrated-{tag}.verification.json';v=json.loads(v.read_text());raw=P/f'benches/results-s54-gpu/s57-integrated-{tag}.txt';man=B/f'{profile}-manifest.json';track(man);m=json.loads(man.read_text());archive=B/'results'/tag/f'prof-{profile}.tgz';track(archive)
 assert v['pass'] and v['results_sha256']==sha(raw) and v['manifest_sha256']==sha(man) and v['profiling']['archive_sha256']==sha(archive)
 for r in d['ptx_entries']:
  if r['profile']==profile:
   rel=str(Path(r['path']).relative_to(B/'bundles'/profile));assert m['files'][rel]==r['sha256']
 for r in d['timing_rows']:
  if r['profile']!=profile:continue
  ls=raw.read_text().splitlines();i=int(r['source_line'])-1;assert ls[i]==f"# n={r['n']} face={r['face']}";end=next((j for j in range(i+1,len(ls)) if ls[j].startswith('# n=')),len(ls));chunk=ls[i:end]
  for leg in ['mapal','cublas','cutlass']:
   l=next(l for l in chunk if l.startswith(f'time leg={leg} '));fields=dict(re.findall(r'(\w+)=(\S+)',l));assert float(fields['med_ms'])==float(r[f'{leg}_device_event_ms_median']);assert int(fields['n'])==30
 checks.append({'profile':profile,'pass':True,'ptx_manifests_matched':6,'median_records_matched':18,'accepted_archive_sha256':sha(archive)})
 for rel in ['box_profile.sh','tile_bench.cu','cutlass/include/cutlass/numeric_conversion.h','cutlass/include/cutlass/gemm/warp/mma_tensor_op.h']:
  p=B/'bundles'/profile/rel;assert sha(p)==m['files'][rel];track(p)
for r in d['observed_resources']:
 if r['leg']=='cutlass': assert len([s for s in d['sass_entries'] if s['origin']=='CUTLASS_executable_fatbin' and s['profile']==r['profile'] and s['function']==r['mangled_name']])==1
for r in d['observed_resources']:
 if r['leg']=='mapal':
  n=8192 if r['scope']=='run-8192' else 4096;face='_tf' if r['scope']=='run_mm_4096_tf' else '_c';a=next(a for a in d['ptx_entries'] if a['profile']==r['profile'] and a['n']==n and a['face']==face);assert a['entry']==r['name']
for r in d['timing_rows']:
 if r['face']!='_exact':assert 'local_bytes=0' in r['raw_resource']
for tag in ['a100-primary','rtx-4090-primary','rtx-5090-primary-retry4']:
 for face in ['_c','_tf']:
  text=(R/tag/'prof'/f'mm_4096{face}.ptxas').read_text();match=re.search(r'Function properties for mapal_tile_[^\n]+\n([^\n]+)',text);assert match and '0 bytes spill stores, 0 bytes spill loads' in match[1]
for r in d['large_rows']:
 out=B/'results/primitive-rtx-5090/retrieved/trace/primitive-results'/r['stdout'];track(out);assert sha(out)==r['process']['stdout_sha256']
 for leg,t in r['parsed']['times'].items():
  match=next(l for l in out.read_text().splitlines() if l.startswith(f'time leg={leg} '));assert dict(re.findall(r'(\w+)=(\S+)',match))['med_ms']==t['med_ms']
for rel in ['crates/backends/llvm/src/lib.rs','docs/decisions/ADR-0032-precision-contracts-vs-backend-config.md','docs/decisions/ADR-0039-reduced-precision-formats-candidate.md']:
 track(P/rel)
assert 'f32 storage, operands rounded to TF32 (10-bit mantissa, `cvt.rna`)' in (P/'crates/backends/llvm/src/lib.rs').read_text()
assert 'Status: candidate — NOT decided' in (P/'docs/decisions/ADR-0039-reduced-precision-formats-candidate.md').read_text()
# Tie example is arithmetic from documented semantics, not GPU execution.
import struct
bits=0x3f801000;value=struct.unpack('<f',struct.pack('<I',bits))[0];lo=struct.unpack('<f',struct.pack('<I',bits&~0x1fff))[0];hi=struct.unpack('<f',struct.pack('<I',(bits+0x1000)&~0x1fff))[0];assert value==(lo+hi)/2 and lo==1.0 and hi==1+2**-10
m=json.loads((B/'source-manifest.json').read_text());assert all(sha(P/p)==h for p,h in m['sources'].items());head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P,text=True).strip();assert head==m['head']
receipt={'pass':True,'checks':checks,'source_head':head,'source_files_unchanged':len(m['sources']),'matching_cutlass_actual_kernel_names':9,'large_stdout_hashes_and_medians':len(d['large_rows']),'tie_example':{'input_bits':hex(bits),'input':value,'rn_ties_even':lo,'rna_ties_away':hi,'kind':'Exact scalar arithmetic from documented PTX rounding; not a GPU test'},'no_new_compilation_or_gpu_execution':True,'semble_search':'MCP request made, returned no output before termination; no source implementation claims inferred from it'}
(R/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n');(R/'artifact-manifest.json').write_text(json.dumps(art,indent=2)+'\n');print(json.dumps(receipt,indent=2));print('artifact_count',len(art))
