"""Read-only allocation proof checker, retaining alloca types and zero ownership."""
from pathlib import Path
import hashlib,json,re
B=Path('/private/tmp/mapal-s58-allocation');E=B/'evidence'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main_body(text):
 text=text[text.index('define internal void @mapal_main('):]
 return text[:text.index('\n}\n')+3]
def dead_allocations(body):
 out=[]
 for line in body.splitlines():
  m=re.search(r'(%\w+) = call i64 @mapal_gpu_alloc\(ptr (%\w+), i64 (\d+)\)',line)
  if not m:continue
  result,slot,count=m.groups();other='\n'.join(x for x in body.splitlines()if x!=line)
  if re.search(re.escape(result)+r'(?!\w)',other):continue
  mentions=[x.strip()for x in other.splitlines()if re.search(re.escape(slot)+r'(?!\w)',x)]
  allowed=lambda x:bool(re.fullmatch(re.escape(slot)+r' = alloca i64',x)or x=='store i64 0, ptr '+slot or x=='call void @mapal_gpu_free(ptr '+slot+')')
  if mentions and all(allowed(x)for x in mentions):out.append({'line':line.strip(),'slot':slot,'bytes':int(count)})
 return out

def canonical(body,remove=()):
 removed={x['line']for x in remove};declarations=[];zeros=[];execution=[]
 for line in body.splitlines():
  stripped=line.strip()
  if stripped in removed:continue
  declaration=re.fullmatch(r'(%\w+) = alloca (.*)',stripped)
  zero=re.fullmatch(r'store i64 0, ptr (%d\d+)',stripped)
  if declaration:declarations.append(declaration.groups())
  elif zero:zeros.append(zero.group(1))
  else:execution.append(line)
 names={}
 def rename(match):
  value=match.group(0)
  if value not in names:names[value]='%canonical'+str(len(names))
  return names[value]
 normalized=re.sub(r'%[tdp]\d+',rename,'\n'.join(execution))
 def name(value):return names.get(value,value)
 return {'body':normalized,'alloca_types':sorted((name(slot),ty)for slot,ty in declarations),'zero_initialized_slots':sorted(name(slot)for slot in zeros)}

def verify():
 receipt=json.loads((E/'receipt.json').read_text());assert sha(E/'candidate.patch')==receipt['patch_sha256']
 for name,digest in receipt['source_hashes'].items():assert sha(B/'source'/name)==digest,name
 for name,digest in receipt['evidence'].items():assert sha(E/name)==digest,name
 assert sha(E/'emit-before')==receipt['emitter_before_sha256'];assert sha(B/'target/release/examples/emit')==receipt['emitter_after_sha256']
 rows=[]
 for row in json.loads((E/'proof.json').read_text()):
  directory=E/'emissions'/row['case'];before=main_body((directory/(row['face']+'.before.host.ll')).read_text());after=main_body((directory/(row['face']+'.after.host.ll')).read_text())
  dead=dead_allocations(before);c1=canonical(before,dead);c2=canonical(after)
  assert c1==c2,(row['case'],row['face'],'body/declaration/zero initialization differs')
  assert not dead_allocations(after)
  assert sha(directory/(row['face']+'.before.device.ll'))==sha(directory/(row['face']+'.after.device.ll'))
  for slot in [d['slot']for d in dead]:assert not re.search(r'load i64, ptr '+re.escape(slot)+r'(?!\w)',before)
  rows.append({'case':row['case'],'face':row['face'],'removed_allocations':len(dead),'omitted_static_bytes':sum(d['bytes']for d in dead),'body_types_zero_ownership_equal':True,'device_LLVM_identical':True})
 # Mutations that the producer's broader declaration normalization might miss.
 sample=next(x for x in json.loads((E/'proof.json').read_text())if x['case']=='omitted-raw'and x['face']=='exact');directory=E/'emissions'/sample['case'];after=main_body((directory/'exact.after.host.ll').read_text());reference=canonical(after)
 mutated=re.sub(r'(%d\d+ = alloca )i64',r'\1i32',after,count=1);assert canonical(mutated)!=reference
 mutated=re.sub(r'store i64 0, ptr (%d\d+)',r'store i64 1, ptr \1',after,count=1);assert canonical(mutated)!=reference
 result={'patch_sha256':receipt['patch_sha256'],'source_head':receipt['source_head'],'variants':len(rows),'independent_equivalence_pass':True,'source_files_hash_verified':len(receipt['source_hashes']),'receipt_artifacts_hash_verified':len(receipt['evidence']),'mutations_rejected':['device-slot alloca i64→i32','device-slot initialization0→1'],'rows':rows}
 print(json.dumps(result,indent=2))
if __name__=='__main__':verify()
