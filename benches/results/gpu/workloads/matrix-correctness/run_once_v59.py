"""One root-reviewed renewed attempt; never retry this launcher."""
import datetime as dt,hashlib,json,os,pathlib,shlex,subprocess,sys,time
sys.path.insert(0,'/private/tmp/mapal-s56-gpu')
import final_ops as o
from run_integrated import stop_group
P=pathlib.Path('/private/tmp/mapal-s58-gpu');TAG='s58-correctness-pair-rtx-5090'
o.WORK=P;o.OPS=P/'operations';o.LEDGER=o.OPS/'ledger.json';o.OPS.mkdir(exist_ok=True)
sha=o.sha

def prepare():
 r=o.read(P/'renewal-v59.json');approval=o.read(P/'root-review-v59.json')
 o.require(approval.get('approved') is True and approval.get('renewal_sha256')==sha(P/'renewal-v59.json'),'Root review missing or binding changed')
 o.require(approval.get('launcher_sha256')==sha(__file__),'Launcher changed after root review')
 o.require(approval.get('failure_abort_policy_resolved') is True,'Ordering failure policy unresolved')
 snap=o.read(P/'offers-sanitized.json');age=(dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(snap['queried_utc'])).total_seconds()
 o.require(0<=age<=600,'Fresh offer required')
 found=[x for x in snap['offers'] if x['id']==r['offer']['id']];o.require(len(found)==1 and found[0]==r['offer'],'Offer changed')
 f=found[0];o.require(f['num_gpus']==1 and f['gpu_name']=='RTX 5090' and f['cpu_arch']=='amd64' and f['cuda_max_good']>=13,'Wrong target')
 o.require(0<f['dph_total']<=.8 and .22+3420/3600*f['dph_total']+.10<=1,'Cumulative cost exceeds cap')
 o.require(f['inet_down_cost']<=.002604167 and f['inet_up_cost']<=.00390625,'Transfer cost exceeds reserve')
 o.require(sha(P/'vast_run.sh')=='3e2d67a015810bdd98be23d9cac17c8f438c0b4d83d96563e4ae504b0ea6c487','Wrapper changed')
 stage=pathlib.Path(r['staging']);o.require({p.name for p in stage.iterdir()}==set(r['files']),'Staging inventory changed')
 for n,v in r['files'].items():o.require(sha(stage/n)==v['sha256'],'Payload changed: '+n)
 o.require(not (P/'launch-claim.json').exists(),'One attempt already consumed')
 live=o.inventory();o.require(not any(x['id'] in {57110891,57111469} for x in live),'Prior failed instance reappeared')
 ledger=o.read('/private/tmp/mapal-s56-gpu/integrated/operations/ledger.json')
 active=[x for x in ledger.values() if x.get('active')]
 o.require(sum(x['listed_dph'] for x in active)+f['dph_total']<=8,'Aggregate rate exceeds cap')
 o.require(not any(x['machine_id']==f['machine_id'] for x in active),'Machine already task-reserved')
 return r,live

def main():
 os.umask(0o077);r,live=prepare();f=r['offer'];res=P/'results';res.mkdir(mode=0o700)
 row={'tag':TAG,'profile':'rtx-5090','phase':'correctness-pair','offer':f['id'],'machine_id':f['machine_id'],'host_id':f['host_id'],'listed_dph':f['dph_total'],'active':True,'accepted':False,'prelaunch_inventory_utc':o.now(),'prelaunch_instance_ids':[x['id'] for x in live]}
 with (P/'launch-claim.json').open('x') as h:json.dump({'started_utc':o.now(),'row':row},h)
 o.write(o.LEDGER,{TAG:row});o.write(res/'request.json',{'renewal':r,'renewal_sha256':sha(P/'renewal-v59.json'),'root_review_sha256':sha(P/'root-review-v59.json'),'launcher_sha256':sha(__file__),'prelaunch':row});real=shlex.quote(subprocess.check_output(['which','vastai'],text=True).strip())
 capture=res/'capture-create.sh';capture.write_text('vastai() {\n if [ "$1" = create ] && [ "$2" = instance ]; then\n '+real+' "$@" | tee '+shlex.quote(str(res/'create-response.json'))+'\n return "${PIPESTATUS[0]}"\n fi\n '+real+' "$@"\n}\n')
 env={k:v for k,v in os.environ.items() if k not in {'KEEP','WAIT_FILE','IMAGE','BASH_ENV'}};env.update(START_TRIES='36',PROFILE='0',DIFFERENTIAL_ONLY='0',BASH_ENV=str(capture))
 start=time.monotonic();child=None;code=1;execution={'rental_started':False,'timeout_seconds':3300}
 try:
  with (res/'wrapper.log').open('x') as log:
   child=subprocess.Popen(['bash',str(P/'vast_run.sh'),str(f['id']),'rtx-5090','sm_120',r['staging'],str(res),'1'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);execution['rental_started']=True
   try:code=child.wait(timeout=3300)
   except subprocess.TimeoutExpired:code=124;execution['timed_out']=True
 finally:
  stop_group(child);execution.update(exit=code,seconds=round(time.monotonic()-start,3),end_utc=o.now());o.write(res/'execution.json',execution)
  raw=res/'create-response.json';ident=None
  if raw.exists():
   raw.chmod(0o600)
   try:ident=o.read(raw).get('new_contract')
   except (ValueError,AttributeError):pass
  if isinstance(ident,int):
   o.require(ident not in {57065574,57100404} and ident not in row['prelaunch_instance_ids'],'Refusing unrelated instance')
   row['instance']=ident
  cleanup=o.cleanup_run(row,res,execution);o.write(res/'cleanup.json',cleanup)
  row.update(active=not cleanup['confirmed_absent'],cleanup_confirmed=cleanup['confirmed_absent']);o.write(o.LEDGER,{TAG:row})
  print(json.dumps({'exit':code,'instance':ident,'cleanup':cleanup,'results':str(res)}))
 return code
if __name__=='__main__':raise SystemExit(main())
