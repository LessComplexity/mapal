import sys
sys.dont_write_bytecode = True
import importlib.util,json,copy,tempfile,datetime,pathlib
from unittest.mock import patch
P=pathlib.Path('/private/tmp/mapal-s58-gpu');spec=importlib.util.spec_from_file_location('launch',P/'run_once_v59.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);o=m.o
read=o.read;sha=o.sha;r=read(P/'renewal-v59.json');snap=read(P/'offers-sanitized.json');snap['queried_utc']=o.now();snap['offers'].sort(key=lambda x:x['id']!=r['offer']['id']);approval={'approved':True,'renewal_sha256':sha(P/'renewal-v59.json'),'launcher_sha256':sha(P/'run_once_v59.py'),'failure_abort_policy_resolved':True}
state={};passed=[]
def fake_read(path):
 name=pathlib.Path(path).name
 if name=='renewal-v59.json':return state['r']
 if name=='root-review-v59.json':return state['a']
 if name=='offers-sanitized.json':return state['s']
 if name=='ledger.json':return state['l']
 return read(path)
for name,change in [
('valid',lambda:None),('denied',lambda:state['a'].update(approved=False)),('wrong-renewal',lambda:state['a'].update(renewal_sha256='bad')),('changed-launcher',lambda:state['a'].update(launcher_sha256='bad')),('policy-unresolved',lambda:state['a'].update(failure_abort_policy_resolved=False)),('stale',lambda:state['s'].update(queried_utc='2000-01-01T00:00:00+00:00')),('changed-offer',lambda:state['s']['offers'][0].update(host_id=-1)),('over-budget',lambda:(state['r']['offer'].update(dph_total=.8),state['s']['offers'][0].update(dph_total=.8))),('over-transfer',lambda:(state['r']['offer'].update(inet_up_cost=1),state['s']['offers'][0].update(inet_up_cost=1))),('active-rate',lambda:state['l'].update(x={'active':True,'listed_dph':8,'machine_id':1})),('active-machine',lambda:state['l'].update(x={'active':True,'listed_dph':.1,'machine_id':r['offer']['machine_id']}))]:
 state.update(r=copy.deepcopy(r),s=copy.deepcopy(snap),a=copy.deepcopy(approval),l={});change()
 with patch.object(o,'read',fake_read),patch.object(o,'inventory',lambda:[]):
  try:m.prepare();assert name=='valid',name
  except ValueError:assert name!='valid',name
 passed.append(name)
row={'instance':999,'machine_id':1,'host_id':2,'prelaunch_instance_ids':[57065574,57100404]};calls=[]
with patch.object(o,'inventory',side_effect=[[{'id':999,'machine_id':1,'host_id':2}],[]]),patch.object(o,'api',lambda *a:calls.append(a)),patch.object(o.time,'sleep',lambda _:None):
 result=o.cleanup_run(row,P,{'rental_started':True});assert result['confirmed_absent'] and calls==[('destroy','instance',999,'-y')]
passed.append('owned-destroy-independent-absence')
with patch.object(o,'inventory',lambda:[{'id':999,'machine_id':10,'host_id':2}]),patch.object(o,'api',lambda *a:(_ for _ in ()).throw(AssertionError('must not destroy'))):
 try:o.cleanup_run(row,P,{'rental_started':True});raise AssertionError('accepted mismatch')
 except ValueError:pass
passed.append('identity-mismatch-no-destroy')
out={'pass':True,'tests':passed,'no_api_or_rental':True,'launcher_sha256':sha(P/'run_once_v59.py'),'renewal_sha256':sha(P/'renewal-v59.json')};(P/'launcher-tests-v59.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
