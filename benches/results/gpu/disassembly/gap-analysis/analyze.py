from pathlib import Path
import collections,csv,hashlib,json,re,sqlite3,subprocess
ROOT=Path('/private/tmp/mapal-s56-gpu/ptx-sass-gap'); BASE=Path('/private/tmp/mapal-s56-gpu/integrated'); REPO=Path('/Volumes/LessComplex/Personal/mapal')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
artifacts={}
def track(p):
 p=Path(p);artifacts[str(p)]={'bytes':p.stat().st_size,'sha256':sha(p)};return p

def ptx_entry(p):
 s=track(p).read_text();matches=list(re.finditer(r'^\.visible \.entry (\w+)\(',s,re.M));m=next(m for m in matches if m.group(1).startswith('mapal_tile_'));end=next((x.start() for x in matches if x.start()>m.start()),len(s));start=s[:m.start()].count('\n')+1
 text=s[m.start():end]; counts=collections.Counter();lines={}
 for n,l in enumerate(text.splitlines(),start):
  l=l.strip();a=re.match(r'(?:@[^\s]+\s+)?([a-z][\w.]*)\s*(.*);',l)
  if not a:continue
  op=a.group(1);counts[op]+=1;lines.setdefault(op,[]).append(n)
 return {'path':str(p),'sha256':sha(p),'entry':m.group(1),'start_line':start,'counts':dict(counts),'opcode_lines':lines,'entry_text':text}

def sass_entries(p):
 s=track(p).read_text();matches=list(re.finditer(r'^\s*Function : (.+)$',s,re.M));out=[]
 for i,m in enumerate(matches):
  end=matches[i+1].start() if i+1<len(matches) else len(s);st=s[:m.start()].count('\n')+1;body=s[m.start():end];counts=collections.Counter();lines={}
  for n,l in enumerate(body.splitlines(),st):
   a=re.match(r'\s*/\*[0-9a-f]+\*/\s+(?:@!?\w+\s+)?([A-Z][A-Z0-9_.]*)\s',l)
   if a:counts[a.group(1)]+=1;lines.setdefault(a.group(1),[]).append(n)
  out.append({'path':str(p),'sha256':sha(p),'function':m.group(1),'start_line':st,'counts':dict(counts),'opcode_lines':lines})
 return out
ptx=[];sass=[];resources=[];timings=[]
for tag,profile in [('a100-primary','a100'),('rtx-4090-primary','rtx-4090'),('rtx-5090-primary-retry4','rtx-5090')]:
 raw=track(REPO/f'benches/results-s54-gpu/s57-integrated-{tag}.txt'); rawlines=raw.read_text().splitlines()
 for ext in ['.json','.verification.json','-kernels.csv'] :track(REPO/f'benches/results-s54-gpu/s57-integrated-{tag}{ext}')
 rows=list(csv.DictReader(open(REPO/f'benches/results-s54-gpu/s57-integrated-{tag}-kernels.csv')))
 for r in rows:
  if r['arm']!='on' or r['n'] not in ['4096','8192']:continue
  suffix={'_c':'_c','_exact':'','_tf':'_tf'}[r['face']];p=BASE/'bundles'/profile/profile/f"mm_{r['n']}{suffix}.ptx";a=ptx_entry(p);a.update(profile=profile,n=int(r['n']),face=r['face']);ptx.append(a)
  start=int(r['source_line']); res=next((n+1,l) for n,l in enumerate(rawlines[start:],start) if l.startswith('mapal kernel='));assert a['entry'] in res[1]
  r.update(raw_resource_line=res[0],raw_resource=res[1],ptx=str(p),ptx_sha256=sha(p));timings.append(r)
 for face,name in [('exact','mm_4096.null'),('contract','mm_4096_c'),('tf32','mm_4096_tf')]:
  p=ROOT/tag/'prof'/f'{name}.sass';a=next(x for x in sass_entries(p) if x['function'].startswith('mapal_tile_'));a.update(profile=profile,face=face,origin='offline_ptxas_not_captured_driver_JIT');sass.append(a);track(ROOT/tag/'prof'/f'{name}.ptxas')
 p=ROOT/tag/'prof/tile_bench.sass'
 for a in sass_entries(p):
  if 'MmaPipelined' in a['function']:face='contract'
  elif 'MmaMultistage' in a['function']:face='tf32'
  else:continue
  a.update(profile=profile,face=face,origin='CUTLASS_executable_fatbin');sass.append(a)
 for dbname in ['run-4096','run-8192','run_mm_4096_tf']:
  p=track(ROOT/tag/'prof'/f'{dbname}.sqlite');c=sqlite3.connect(f'file:{p}?mode=ro',uri=True);c.row_factory=sqlite3.Row
  q='''select s.value as name,m.value as mangled_name,k.registersPerThread,k.blockX,k.blockY,k.blockZ,k.gridX,k.gridY,k.gridZ,k.staticSharedMemory,k.dynamicSharedMemory,k.localMemoryPerThread,count(*) as launches from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName left join StringIds m on m.id=k.mangledName group by 1,2,3,4,5,6,7,8,9,10,11,12'''
  for row in c.execute(q):
   d=dict(row);d.update(profile=profile,scope=dbname,path=str(p),sql=q);d['leg']='mapal' if d['name'].startswith('mapal_tile') else ('cutlass' if d['name'].startswith('void cutlass::Kernel<') else 'cublas');resources.append(d)
  c.close()
largebase=Path('/private/tmp/mapal-s56-gpu/large-composed/primitive/run-bundle-rtx-5090');summary=track(BASE/'results/primitive-rtx-5090/results-rtx-5090.txt');large=json.loads(summary.read_text());manifest=json.loads(track(largebase/'run-manifest.json').read_text());cases={c['id']:c for c in manifest['cases']}
large_rows=[]
for r in large['records']:
 if r['arm']!='base' or not (r['M']==r['N']==r['K']==16384 or r['M']==16):continue
 c=cases[r['case']];a=ptx_entry(largebase/c['ptx']);a.update(profile='rtx-5090-supplement',n=r['N'],M=r['M'],K=r['K'],face=r['face']);ptx.append(a)
 large_rows.append(r)
for a in ptx:a.pop('entry_text')
report={'source_boundary':'Read-only file analysis; no new GPU/CPU compiled execution','ptx_entries':ptx,'sass_entries':sass,'observed_resources':resources,'timing_rows':timings,'large_rows':large_rows}
(ROOT/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
(ROOT/'artifact-manifest.json').write_text(json.dumps(artifacts,indent=2)+'\n')
print('tracked',len(artifacts),'PTX',len(ptx),'SASS',len(sass),'resource groups',len(resources),'timing rows',len(timings),'large rows',len(large_rows))
for s in sass:
 c=s['counts'];interesting={op:n for op,n in c.items() if op.startswith(('FFMA','FMUL','FADD','HMMA','HGMMA','LDG','LDS','STS','STG','LDL','STL','BAR','DEPBAR'))};print(s['profile'],s['face'],s['origin'],sum(c.values()),interesting)
