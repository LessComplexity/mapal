from pathlib import Path
from fractions import Fraction
from functools import lru_cache
import hashlib,json,math,struct
root=Path('/private/tmp/mapal-s61-gpu/attempt-10/retrieved/trace/composed-native')
packet=Path('/private/tmp/mapal-composed-fused-references/packet-v1')
out=Path('/private/tmp/mapal-native-root-review')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s=json.loads((root/'summary.json').read_text())
refroot=packet/'reference-data';manifest=json.loads((refroot/'manifest.json').read_text())
for name,digest in manifest['files'].items():assert sha(refroot/name)==digest
for name,digest in s['files'].items():assert sha(root/name)==digest
@lru_cache(None)
def num(word):
 x=struct.unpack('<f',bytes.fromhex(word))[0];assert math.isfinite(x);return Fraction.from_float(x)
@lru_cache(None)
def references(case):
 p=refroot/case
 refs={f:json.loads((p/f'shift0.{f}.json').read_text()) for f in ('fp32','fmaf')}
 bounds={f:json.loads((p/f'shift0.{f}.bounds.json').read_text()) for f in refs}
 contract=json.loads((p/'contract.json').read_text()) if (p/'contract.json').exists() else None
 return refs,bounds,contract
processes=[];words=events=0
for r in s['tiny_gates']+s['tuning']+s['confirmations']:
 path=root/r['path'];assert sha(path/'stdout')==r['stdout_sha256']
 rows=[json.loads(l) for l in (path/'stdout').read_text().splitlines()]
 refs,bounds,contract=references(r['case']);shape=manifest['cases'][r['case']]
 meta=[x for x in rows if x['type']=='metadata'];assert len(meta)==1 and meta[0]['backend']==r['backend'] and meta[0]['fusion']==r['fusion']
 nwords=0
 for row in rows:
  if row['type'] not in ('checkpoint','final','sample_final'):continue
  name=row['name'];width=shape['h'] if name in ('hraw','h') else shape['d'];assert row['shape']==[shape['M'],width]
  rv={f:next(x for x in refs[f]['checkpoints'] if x['name']==name)['bits_le'] for f in refs}
  indices=contract['checkpoint_indices'][name] if contract else list(range(len(rv['fmaf'])))
  ri=contract.get('reference_checkpoint_indices',contract['checkpoint_indices'])[name] if contract else indices
  assert row['indices']==indices and len(row['bits_le'])==len(indices)
  for w,i in zip(row['bits_le'],ri):
   actual=num(w)
   for f in refs:
    bound=bounds[f]['-1:'+name][i];assert math.isfinite(bound) and bound>0
    assert abs(actual-num(rv[f][i]))<=Fraction.from_float(bound),(r['path'],name,i,f)
   if r['backend']=='naive':assert w==rv['fmaf'][i]
   nwords+=1
 words+=nwords
 sample=[x for x in rows if x['type']=='sample']
 if r['mode']=='time':
  warm,count=(3,9) if r['tuning'] else (5,30)
  assert len(sample)==warm+count and [x['rep'] for x in sample]==list(range(warm+count))
  assert [x['warmup'] for x in sample]==[i<warm for i in range(warm+count)]
  for clock in ('wall_ms','event_ms'):
   vals=[x[clock] for x in sample[warm:]];assert all(math.isfinite(x) and x>0 for x in vals)
   v=sorted(vals);obs=r['admission']['observations'][clock]
   assert vals==obs['raw'] and obs['upper_median']==v[count//2] and obs['p10']==v[math.ceil(.1*count)-1] and obs['p90']==v[math.ceil(.9*count)-1]
  events+=len(sample)
 else:assert not sample
 processes.append({'path':r['path'],'words':nwords,'samples_including_warmups':len(sample),'stdout_sha256':sha(path/'stdout')})
result={'pass':True,'implementation':'Independent IEEE/Fraction value and raw timing reader; no frozen checker imports','processes':len(processes),'native_files':len(s['files']),'reference_files':len(manifest['files']),'raw_words_checked':words,'timing_pairs_including_warmup':events,'scope':'All native gate, tuning and confirmation exported words; both frozen scalar budgets; ordered naive words; both raw clocks/statistics. Separate reviewer owns protocol/selection and control admission.','records':processes}
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'}))
