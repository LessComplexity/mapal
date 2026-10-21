#!/usr/bin/env python3
import hashlib,json,pathlib,subprocess,tempfile,unittest,struct,os
from validate import validate
SOURCE=pathlib.Path(__file__).resolve().parent
if not os.environ.get('CAPTURE_TEST_CALLBACK'):raise SystemExit('set CAPTURE_TEST_CALLBACK to the separately built real-header stub executable')
CALLBACK=pathlib.Path(os.environ['CAPTURE_TEST_CALLBACK']).resolve(strict=True)
class CaptureTests(unittest.TestCase):
 def make(self,root,mode='happy'):
  root=pathlib.Path(root);cap=root/'capture'
  fixture=root/'synthetic.cubin'
  names=b'\0.shstrtab\0.text.kernel\0'; header=bytearray(64)
  header[:7]=b'\x7fELF\x02\x01\x01';struct.pack_into('<HHI',header,16,2,190,1);struct.pack_into('<Q',header,40,64);struct.pack_into('<HHHHHH',header,52,64,0,0,64,3,1)
  empty=bytes(64);string=struct.pack('<IIQQQQIIQQ',1,3,0,0,256,len(names),0,0,1,0);text=struct.pack('<IIQQQQIIQQ',11,1,6,0,256+len(names),8,0,0,8,0)
  fixture.write_bytes(header+empty+string+text+names+bytes(8))
  subprocess.run([str(CALLBACK),str(cap),mode,str(fixture)],check=True)
  for name in ('executable','observer','stdout'): (root/name).write_text('stub '+name+'\n')
  (root/'stderr').write_bytes(b'')
  def item(name):
   p=root/name;return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
  r={'variant':'production-driver-ptx-observed','exit_code':0,'timeout':False,'kernel':'kernel','expected_launch':{'grid':[1,1,1],'block':[1,1,1],'shared_bytes':0},'ptx_sha256':hashlib.sha256((cap/'input.ptx').read_bytes()).hexdigest(),**{n:item(n) for n in ('executable','observer','stdout','stderr')}}
  rec=root/'receipt.json';rec.write_text(json.dumps(r));return cap,rec,r
 def test_real_header_stub_happy_and_callback_lifetime(self):
  with tempfile.TemporaryDirectory() as t:
   c,r,_=self.make(t);v=validate(c,r);self.assertFalse(v['timing_admissible']);self.assertEqual((c/'module-1.cubin').read_bytes(),(c.parent/'synthetic.cubin').read_bytes())
 def test_callback_partiality(self):
  for mode in ('missing','empty','corrupt','duplicate','unlinked','no-launch','launch-failed','oversize','cap','resource-wrong-context','exit-wrong-context','exit-wrong-uid','exit-wrong-correlation','exit-wrong-output','exit-wrong-image'):
   with self.subTest(mode=mode),tempfile.TemporaryDirectory() as t:
    c,r,_=self.make(t,mode)
    with self.assertRaises(ValueError):validate(c,r)
 def test_manifest_corruption(self):
  for key,value in [('variant','offline-cubin'),('exit_code',1),('timeout',True),('kernel','wrong'),('ptx_sha256','0'*64)]:
   with self.subTest(key=key),tempfile.TemporaryDirectory() as t:
    c,r,d=self.make(t);d[key]=value;r.write_text(json.dumps(d))
    with self.assertRaises(ValueError):validate(c,r)
 def test_artifact_corruption(self):
  for target in ('input.ptx','module-1.cubin','events.tsv'):
   with self.subTest(target=target),tempfile.TemporaryDirectory() as t:
    c,r,_=self.make(t);(c/target).write_bytes(b'corrupt')
    with self.assertRaises((ValueError,IndexError)):validate(c,r)
 def test_unlinked_launch_and_bad_chronology(self):
  for old,new in [('launch\t5\t1\t4\t9','launch\t5\t1\t4\t8'),('launch\t5','launch\t2')]:
   with self.subTest(new=new),tempfile.TemporaryDirectory() as t:
    c,r,_=self.make(t);p=c/'events.tsv';s=p.read_text();self.assertIn(old,s);p.write_text(s.replace(old,new))
    with self.assertRaises(ValueError):validate(c,r)
 def test_elf_partiality(self):
  for change in ('machine','section-bound','name'):
   with self.subTest(change=change),tempfile.TemporaryDirectory() as t:
    c,r,_=self.make(t);p=c/'module-1.cubin';b=bytearray(p.read_bytes())
    if change=='machine':struct.pack_into('<H',b,18,0)
    elif change=='section-bound':struct.pack_into('<Q',b,64+2*64+32,1<<32)
    else:b[b.find(b'.text.kernel')+6]=ord('X')
    p.write_bytes(b)
    with self.assertRaises(ValueError):validate(c,r)
 def test_runner_missing_injection_and_no_overwrite(self):
  with tempfile.TemporaryDirectory() as t:
   root=pathlib.Path(t);(root/'input.ptx').write_text('//x');(root/'observer').write_text('stub')
   out=root/'run';command=['python3',str(SOURCE/'run_capture.py'),'--out',str(out),'--ptx',str(root/'input.ptx'),'--observer',str(root/'observer'),'--kernel','kernel','--grid','1,1,1','--block','1,1,1','--shared-bytes','0','--','/bin/echo','stub']
   result=subprocess.run(command,capture_output=True,text=True)
   self.assertNotEqual(result.returncode,0);self.assertTrue((out/'receipt.json').is_file());self.assertFalse((out/'validation.json').exists())
   before=(out/'receipt.json').read_bytes();result=subprocess.run(command,capture_output=True,text=True)
   self.assertNotEqual(result.returncode,0);self.assertEqual(before,(out/'receipt.json').read_bytes())
 def test_exact_launch_contract(self):
  for old,new in [('1,1,1\t1,1,1\t0','999,17,3\t1,1,1\t0'),('1,1,1\t1,1,1\t0','1,1,1\t1025,2,4\t0'),('1,1,1\t1,1,1\t0','1,1,1\t1,1,1\t999999')]:
   with self.subTest(new=new),tempfile.TemporaryDirectory() as t:
    c,r,_=self.make(t);p=c/'events.tsv';s=p.read_text();self.assertIn(old,s);p.write_text(s.replace(old,new))
    with self.assertRaisesRegex(ValueError,'launch contract mismatch'):validate(c,r)
 def test_launch_contract_required_and_typed(self):
  for contract in (None,{'grid':[True,1,1],'block':[1,1,1],'shared_bytes':0},{'grid':[1,1,1],'block':[1,1,1],'shared_bytes':-1}):
   with self.subTest(contract=contract),tempfile.TemporaryDirectory() as t:
    c,r,d=self.make(t)
    if contract is None:del d['expected_launch']
    else:d['expected_launch']=contract
    r.write_text(json.dumps(d))
    with self.assertRaises((ValueError,KeyError)):validate(c,r)
 def test_resource_context_and_exit_correlation_log(self):
  for old,new in [('image\t2\t1\t0x33','image\t2\t1\t0x44'),('load-exit\t3\t1\t0x33\t4\t9\t7','load-exit\t3\t1\t0x44\t4\t9\t7'),('load-exit\t3\t1\t0x33\t4\t9\t7','load-exit\t3\t1\t0x33\t4\t9\t8')]:
   with self.subTest(new=new),tempfile.TemporaryDirectory() as t:
    c,r,_=self.make(t);p=c/'events.tsv';s=p.read_text();self.assertIn(old,s);p.write_text(s.replace(old,new))
    with self.assertRaises(ValueError):validate(c,r)
 def test_symlink_rejected(self):
  with tempfile.TemporaryDirectory() as t:
   c,r,_=self.make(t);p=c/'module-1.cubin';p.rename(c/'other');p.symlink_to(c/'other')
   with self.assertRaises(ValueError):validate(c,r)
if __name__=='__main__':unittest.main(verbosity=2)
