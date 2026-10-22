import sys
sys.dont_write_bytecode=True
import copy,datetime as dt,importlib.util,json,pathlib,tempfile
from unittest.mock import patch
HOME=pathlib.Path(__file__).parent;spec=importlib.util.spec_from_file_location('launch',HOME/'launch_once.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);prior=pathlib.Path('/private/tmp/mapal-s58-gpu/results/budget-closed.json');offers=json.loads((HOME/'offers-sanitized.json').read_text());selected=next(x for x in offers['offers']if x['id']==34376038);read=m.read;tests=[]
for mode in ['valid','review-denied','wrong-review','consumed','stale','changed-offer','rate-over-cap','transfer-over-cap','payload-changed','prior-budget-changed','aggregate-over-cap','machine-reserved']:
 with tempfile.TemporaryDirectory()as tmp:
  m.P=pathlib.Path(tmp);stage=m.P/'staging';stage.mkdir();(stage/'payload').write_text('frozen');(m.P/'vast_run.sh').write_bytes((HOME/'vast_run.sh').read_bytes());snapshot=copy.deepcopy(offers);snapshot['queried_utc']=m.now();offer=copy.deepcopy(selected);active=[]
  if mode=='stale':snapshot['queried_utc']='2000-01-01T00:00:00+00:00'
  if mode=='changed-offer':snapshot['offers']=[{**offer,'host_id':-1}]
  if mode=='rate-over-cap':offer['dph_total']=.81;snapshot['offers']=[offer]
  if mode=='transfer-over-cap':offer['inet_up_cost']=1;snapshot['offers']=[offer]
  if mode=='aggregate-over-cap':active=[{'active':True,'listed_dph':8,'machine_id':1}]
  if mode=='machine-reserved':active=[{'active':True,'listed_dph':.1,'machine_id':offer['machine_id']}]
  m.write(m.P/'offers-sanitized.json',snapshot);packet={'offer':offer,'prior_budget_path':str(prior),'prior_budget_sha256':m.sha(prior),'offer_snapshot_sha256':m.sha(m.P/'offers-sanitized.json'),'staging':str(stage),'files':{'payload':m.sha(stage/'payload')},'max_seconds':1800,'retrieval_cleanup_reserve_seconds':180,'smallm_max_seconds':1100}
  if mode=='prior-budget-changed':packet['prior_budget_sha256']='bad'
  m.write(m.P/'packet.json',packet);review={'approved':mode!='review-denied','single_attempt':True,'packet_sha256':m.sha(m.P/'packet.json'),'launcher_sha256':m.sha(HOME/'launch_once.py')}
  if mode=='wrong-review':review['packet_sha256']='bad'
  m.write(m.P/'root-review.json',review)
  if mode=='consumed':(m.P/'launch-claim.json').write_text('{}')
  if mode=='payload-changed':(stage/'payload').write_text('changed')
  def fake_read(path):
   if str(path).endswith('/integrated/operations/ledger.json'):return {str(i):x for i,x in enumerate(active)}
   return read(path)
  with patch.object(m,'read',fake_read),patch.object(m,'inventory',lambda:[]):
   try:m.prepare();assert mode=='valid',mode
   except ValueError:assert mode!='valid',mode
  tests.append(mode)
m.P=HOME;row={'instance':999,'machine_id':1,'host_id':2,'prelaunch_ids':[57065574,57100404]};calls=[]
with patch.object(m,'inventory',side_effect=[[{'id':999,'machine_id':1,'host_id':2}],[]]),patch.object(m,'api',lambda *a:calls.append(a)),patch.object(m.time,'sleep',lambda _:None):
 proof=m.cleanup(row,HOME);assert proof['confirmed_absent']and calls==[('destroy','instance',999,'-y')]
tests.append('owned-destroy-independent-absence')
for mode,newrow,live in [('unrelated',{**row,'instance':57065574},[]),('identity-mismatch',row,[{'id':999,'machine_id':3,'host_id':2}])]:
 with patch.object(m,'inventory',lambda:live),patch.object(m,'api',lambda *a:(_ for _ in ()).throw(AssertionError('must not destroy'))):
  try:m.cleanup(newrow,HOME);raise AssertionError('unsafe acceptance')
  except ValueError:pass
 tests.append(mode)
with tempfile.TemporaryDirectory() as tmp:
 m.P=pathlib.Path(tmp)
 with patch.object(m,'prepare',lambda:({},[])),patch.object(m.shutil,'which',lambda _:None):
  try:m.main();raise AssertionError('Missing CLI accepted')
  except ValueError as e:assert str(e)=='Vast CLI absent'
 assert not list(m.P.iterdir())
 tests.append('missing-cli-before-claim-no-write')
m.P=HOME
receipt={'pass':True,'tests':tests,'no_api_rental_upload':True,'launcher_sha256':m.sha(HOME/'launch_once.py')};m.write(HOME/'launcher-tests.json',receipt);print(json.dumps(receipt))
