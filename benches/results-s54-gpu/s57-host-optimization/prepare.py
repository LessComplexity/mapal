"""Frozen LLVM host optimization evidence only; no compiler source/GPU execution."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

HERE=Path(__file__).resolve().parent
ROOT=Path('/Volumes/LessComplex/Personal/mapal')
FREEZE=Path('/private/tmp/mapal-s56-gpu/integrated/source-manifest.json')
CERT=HERE.parent/'composition/certification-01'
REL='gpu/recurrent-t16-d8-h16.run/rtx-5090/contract/shipped/host.ll'
FREEZE_SHA='afdb6d16c0fbeb31789d2760a82ec48ffacd206d9f957af8d0cac72608e62133'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')


def snapshot(frozen):
    actual={'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'source_manifest_sha256':sha(FREEZE),
            'sources':{n:sha(ROOT/n)for n in frozen['sources']},
            'inputs':{n:sha(ROOT/n)for n in frozen['inputs']},
            'llvm_tools':{n:sha(row['path'])for n,row in frozen['compiler_files'].items()}}
    assert actual['head']==frozen['head']
    assert actual['source_manifest_sha256']==FREEZE_SHA
    assert len(actual['sources'])==328 and actual['sources']==frozen['sources']
    assert len(actual['inputs'])==4 and actual['inputs']==frozen['inputs']
    assert actual['llvm_tools']=={n:row['sha256']for n,row in frozen['compiler_files'].items()}
    return actual


def main():
    frozen=json.loads(FREEZE.read_text());before=snapshot(frozen)
    write(HERE/'before.json',before)
    cert=json.loads((CERT/'manifest.json').read_text())
    assert sha(CERT/REL)==cert['files'][REL]
    inp=HERE/'host.emitted.ll';assert not inp.exists()
    shutil.copyfile(CERT/REL,inp)
    shutil.copyfile(CERT/REL.replace('host.ll','command.json'),HERE/'original-emission-command.json')
    clang=frozen['compiler_files']['clang']['path']
    common=[clang,'-O2','--target=x86_64-unknown-linux-gnu','-Wno-override-module']
    stages=[('optimized-llvm',[*common,'-S','-emit-llvm',str(inp),'-o',str(HERE/'host.O2.ll')]),
            ('host-object',[*common,'-c',str(inp),'-o',str(HERE/'host.O2.o')]),
            ('host-assembly',[*common,'-S',str(inp),'-o',str(HERE/'host.O2.s')])]
    receipts=[]
    for name,argv in stages:
        start=time.monotonic()
        with(HERE/(name+'.stdout')).open('x')as out,(HERE/(name+'.stderr')).open('x')as err:
            process=subprocess.run(argv,stdout=out,stderr=err,timeout=120)
        receipts.append({'stage':name,'argv':argv,'exit':process.returncode,'seconds':time.monotonic()-start})
        assert process.returncode==0
    after=snapshot(frozen);write(HERE/'after.json',after);assert before==after
    write(HERE/'manifest.json',{'scope':'Frozen composition host LLVM optimized/lowered using frozen clang-O2; no link, runtime or GPU execution',
                              'original_input':str(CERT/REL),'original_input_sha256':sha(CERT/REL),
                              'certification_manifest_sha256':sha(CERT/'manifest.json'),
                              'frozen_manifest_sha256':FREEZE_SHA,'source_head':frozen['head'],
                              'tool':frozen['compiler_files']['clang'],'commands':receipts,
                              'before_after_identical':True,'frozen_sources_checked':328,'frozen_inputs_checked':4,'frozen_llvm_tools_checked':3,
                              'files':{str(p.relative_to(HERE)):sha(p)for p in sorted(HERE.iterdir())if p.is_file()}})
    print(json.dumps({'output':str(HERE),'before_after_identical':True,'commands':len(receipts)}))


if __name__=='__main__':main()
