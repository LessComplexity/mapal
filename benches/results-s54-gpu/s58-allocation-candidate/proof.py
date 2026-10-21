from pathlib import Path
import json,hashlib,subprocess,re
B=Path('/private/tmp/mapal-s58-allocation');E=B/'evidence';emit=[E/'emit-before',B/'target/release/examples/emit'];old=Path('/private/tmp/mapal-ordering-fix/evidence');frozen=Path('/private/tmp/mapal-s56-gpu/integrated/bundles/rtx-5090/tiles')
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main(s):
 s=s[s.index('define internal void @mapal_main('):];return s[:s.index('\n}\n')+3]
def dead(s):
 out=[]
 for result,slot,n in re.findall(r'(%\w+) = call i64 @mapal_gpu_alloc\(ptr (%\w+), i64 (\d+)\)',s):
  if not re.search(r'load i64, ptr '+re.escape(slot)+r'(?!\w)',s) and len(re.findall(re.escape(result)+r'(?!\w)',s))==1:out.append((result,slot,int(n)))
 return out
def norm(s,remove):
 removed={r for r,_,_ in remove};lines=[]
 for l in s.splitlines():
  if any(l.strip().startswith(r+' = call i64 @mapal_gpu_alloc(') for r in removed):continue
  if ' = alloca ' in l or re.search(r'store i64 0, ptr %d\d+',l):continue
  lines.append(l)
 names={}
 def sub(m):
  v=m[0]
  if v not in names:names[v]='%v'+str(len(names))
  return names[v]
 return re.sub(r'%[tdp]\d+',sub,'\n'.join(lines))
fixtures={r['case']:Path(r['source']) for r in json.loads((old/'fixed-emission-commands.json').read_text())}
fixtures['omitted-raw']=Path('/private/tmp/mapal-s56-gpu/ordering-audit/stored-member-ordering.mapal')
for name in ['indep','diamond','mlp2','twin','mlp','fanout']:fixtures['pt512-'+name]=frozen/f'pt_{name}_512.mapal'
rows=[]
for name,src in fixtures.items():
 d=E/'emissions'/name;d.mkdir(parents=True,exist_ok=True);(d/'source.mapal').write_bytes(src.read_bytes())
 for face in ['exact','contract']:
  records=[]
  for label,bin in zip(['before','after'],emit):
   dev=d/f'{face}.{label}.device.ll';host=d/f'{face}.{label}.host.ll';err=d/f'{face}.{label}.stderr';a=[str(bin),str(d/'source.mapal'),'-','--rewrite','--target=rtx-5090','--gpu-on=streams,epilogue','--gpu-off=stream-gate',f'--device-out={dev}']
   if face=='contract':a+=['--contract']
   with host.open('wb') as o,err.open('wb') as e:r=subprocess.run(a,stdout=o,stderr=e,timeout=180)
   assert r.returncode==0
   s=main(host.read_text());records.append({'label':label,'argv':a,'exit':r.returncode,'host_sha256':h(host),'device_sha256':h(dev),'stderr_sha256':h(err),'alloc_calls':s.count('call i64 @mapal_gpu_alloc('),'free_calls':s.count('call void @mapal_gpu_free('),'dead':dead(s),'main':s,'events':re.findall(r'call void @mapal_gpu_event\([^\n]+',s),'readback_streams':re.findall(r'call void @mapal_gpu_d2h(_on)?\((?:i32 (\d+),)?',s)})
  b,a=records;bn=norm(b['main'],b['dead']);an=norm(a['main'],[]);(d/f'{face}.before.normalized.ll').write_text(bn);(d/f'{face}.after.normalized.ll').write_text(an)
  rows.append({'case':name,'face':face,'source_sha256':h(d/'source.mapal'),'device_unchanged':b['device_sha256']==a['device_sha256'],'only_dead_allocs_removed_body_equal':bn==an,'events_unchanged':b['events']==a['events'],'readback_streams_unchanged':b['readback_streams']==a['readback_streams'],'requested_bytes_removed':sum(x[2] for x in b['dead']),'records':[{k:v for k,v in r.items() if k!='main'} for r in records]})
(E/'proof.json').write_text(json.dumps(rows,indent=2)+'\n')
for r in rows:print(r['case'],r['face'],r['records'][0]['alloc_calls'],r['records'][1]['alloc_calls'],r['requested_bytes_removed'],'dev',r['device_unchanged'],'body',r['only_dead_allocs_removed_body_equal'],'events',r['events_unchanged'],'reads',r['readback_streams_unchanged'])
assert all(r['device_unchanged'] and r['only_dead_allocs_removed_body_equal'] and r['events_unchanged'] and r['readback_streams_unchanged'] and not r['records'][1]['dead'] for r in rows)
