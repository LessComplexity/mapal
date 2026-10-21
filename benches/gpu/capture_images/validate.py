#!/usr/bin/env python3
"""Fail-closed admission for the narrow Mapal driver-PTX observation variant."""
import hashlib,json,pathlib,sys,struct
MAX=16<<20
def digest(p,limit=MAX):
    if p.is_symlink() or not p.is_file() or not 0<p.stat().st_size<=limit: raise ValueError('invalid artifact')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()
def elf_kernel(data,kernel):
    if len(data)<64 or data[:7]!=b'\x7fELF\x02\x01\x01': raise ValueError('invalid ELF64')
    typ,machine,version=struct.unpack_from('<HHI',data,16)
    shoff=struct.unpack_from('<Q',data,40)[0]
    ehsize,_,_,shsize,count,names=struct.unpack_from('<HHHHHH',data,52)
    if (typ,machine,version,ehsize,shsize)!=(2,190,1,64,64) or count<2 or names>=count or shoff<64 or shoff+count*64>len(data): raise ValueError('invalid CUDA ELF headers')
    sections=[struct.unpack_from('<IIQQQQIIQQ',data,shoff+i*64) for i in range(count)]
    for sh in sections:
        if sh[1]!=8 and (sh[4]>len(data) or sh[5]>len(data)-sh[4]): raise ValueError('ELF section out of bounds')
    ns=sections[names]
    if ns[1]!=3: raise ValueError('missing ELF string table')
    table=data[ns[4]:ns[4]+ns[5]]; found=0
    for sh in sections:
        if sh[0]>=len(table): raise ValueError('ELF name offset')
        end=table.find(b'\0',sh[0])
        if end<0: raise ValueError('unterminated ELF name')
        if table[sh[0]:end]==('.text.'+kernel).encode():
            if sh[1]!=1 or sh[5]==0: raise ValueError('empty kernel text')
            found+=1
    if found!=1: raise ValueError('kernel text absent or ambiguous')
def validate(root,receipt):
    root=pathlib.Path(root); receipt=pathlib.Path(receipt)
    if receipt.is_symlink() or not 0<receipt.stat().st_size<=1<<20:raise ValueError('receipt cap')
    r=json.loads(receipt.read_text())
    if not isinstance(r['kernel'],str) or not 0<len(r['kernel'])<=1024:raise ValueError('kernel cap')
    contract=r['expected_launch']
    for key in ('grid','block'):
        if not isinstance(contract[key],list) or len(contract[key])!=3 or any(type(x) is not int or not 0<x<=0xffffffff for x in contract[key]):raise ValueError('invalid expected launch dimensions')
    if type(contract['shared_bytes']) is not int or not 0<=contract['shared_bytes']<=0xffffffff:raise ValueError('invalid expected shared bytes')
    if r['variant']!='production-driver-ptx-observed' or r['exit_code']!=0 or r['timeout']: raise ValueError('incomplete harness')
    for key in ('executable','observer','stdout','stderr'):
        p=pathlib.Path(r[key]['path'])
        # Empty stderr is legitimate, unlike empty code or event artifacts.
        if key=='stderr' and p.is_file() and not p.is_symlink() and p.stat().st_size==0:
            got=hashlib.sha256(b'').hexdigest()
        else: got=digest(p,(512<<20) if key=='executable' else MAX)
        if got!=r[key]['sha256']: raise ValueError('receipt hash mismatch: '+key)
    if digest(root/'input.ptx')!=r['ptx_sha256']: raise ValueError('PTX mismatch')
    ep=root/'events.tsv'
    if ep.is_symlink() or not ep.is_file() or not 0<ep.stat().st_size<=1<<20: raise ValueError('event cap')
    rows=[line.split('\t') for line in ep.read_text().splitlines()]
    if rows[:2]!=[['schema','mapal-cupti-driver-v2'],['ready']]: raise ValueError('missing initialization')
    loads={};images={};finished={};functions={};launches=[];previous=0
    for row in rows[2:]:
        kind=row[0]
        if kind=='error': raise ValueError('observer error: '+str(row))
        e=int(row[1])
        if not previous<e<=4096: raise ValueError('event chronology')
        previous=e
        if kind=='load-enter':
            _,_,pointer,ctx,corr=row
            if e in loads: raise ValueError('duplicate load')
            loads[e]=(pointer,ctx,corr)
        elif kind=='image':
            _,_,load,pointer,module,size,name=row;load=int(load)
            if load not in loads or load in images or loads[load][0]!=pointer or name!=f'module-{load}.cubin': raise ValueError('unlinked image')
            p=root/name;sha=digest(p);b=p.read_bytes()
            if len(b)!=int(size) or not b.startswith(b'\x7fELF') or len(b)<4: raise ValueError('invalid cubin')
            elf_kernel(b,r['kernel'])
            if sum(x['size'] for x in images.values())+len(b)>MAX: raise ValueError('total image cap')
            images[load]={'module':module,'sha256':sha,'size':len(b),'event':e}
        elif kind=='load-exit':
            _,_,load,pointer,ctx,module,corr=row;load=int(load)
            if load not in images or load in finished or loads[load]!=(pointer,ctx,corr) or images[load]['module']!=module: raise ValueError('unlinked module')
            finished[load]=(ctx,module)
        elif kind=='function':
            _,_,load,ctx,module,fn,name=row;load=int(load)
            if finished.get(load)!=(ctx,module): raise ValueError('unlinked function')
            key=(ctx,fn)
            if key in functions and functions[key]!=(load,name): raise ValueError('ambiguous function')
            functions[key]=(load,name)
        elif kind=='launch':
            _,_,load,ctx,module,fn,corr,grid,block,smem=row;load=int(load)
            if finished.get(load)!=(ctx,module) or (ctx,fn) not in functions or functions[(ctx,fn)][0]!=load: raise ValueError('unlinked launch')
            for dims in (grid,block):
                if len(dims.split(','))!=3 or any(int(x)<=0 for x in dims.split(',')): raise ValueError('invalid launch dims')
            if [int(x) for x in grid.split(',')]!=contract['grid'] or [int(x) for x in block.split(',')]!=contract['block'] or int(smem)!=contract['shared_bytes']:raise ValueError('launch contract mismatch')
            if functions[(ctx,fn)][1]!=r['kernel']: raise ValueError('unexpected kernel')
            launches.append({'load':load,'context':ctx,'module':module,'function':fn,'kernel':r['kernel'],'correlation':int(corr),'grid':grid,'block':block,'shared_bytes':int(smem)})
        else: raise ValueError('unknown event')
    if not loads or set(loads)!=set(images) or set(images)!=set(finished) or not launches or {x['load'] for x in launches}!=set(loads): raise ValueError('missing image or launch')
    return {'status':'admitted-observation-only','scope':'Mapal exact driver PTX; library images absent','completion':'harness exit 0 after its existing correctness/synchronization gates; callbacks alone are submissions','timing_admissible':False,'expected_launch':contract,'receipt_sha256':digest(receipt,1<<20),'events_sha256':digest(ep),'ptx_sha256':r['ptx_sha256'],'images':images,'launches':launches}
if __name__=='__main__':
    try: print(json.dumps(validate(sys.argv[1],sys.argv[2]),indent=2,sort_keys=True))
    except (ValueError,KeyError,IndexError,OSError,TypeError) as e: print('REJECTED: '+str(e),file=sys.stderr);sys.exit(1)
