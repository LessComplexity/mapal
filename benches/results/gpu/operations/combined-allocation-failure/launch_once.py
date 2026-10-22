"""Single reviewed combined attempt; no automatic retries or unrelated destruction."""
import datetime as dt,hashlib,json,math,os,pathlib,shlex,shutil,signal,subprocess,time
P=pathlib.Path('/private/tmp/mapal-s58-gpu/combined-v61');UNRELATED={57065574,57100404};RATE_LIMIT=.8;MAX_SECONDS=1800
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(pathlib.Path(p).read_text())
def write(p,x):pathlib.Path(p).write_text(json.dumps(x,indent=2)+'\n')
def require(c,m):
 if not c:raise ValueError(m)
def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def api(*args):
 private=P/'private-api';private.mkdir(mode=0o700,exist_ok=True);stamp=dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');r=subprocess.run(['vastai',*map(str,args)],capture_output=True,text=True,timeout=8);write(private/(stamp+'.json'),{'args':args,'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr});require(r.returncode==0,'Vast operation failed; private diagnostic retained');return json.loads(r.stdout) if r.stdout.strip().startswith(('[','{'))else r.stdout

def inventory():
 rows=api('show','instances','--raw');require(isinstance(rows,list)and all(isinstance(x.get('id'),int)for x in rows),'Invalid inventory; absence unproved');return rows

def prepare():
 packet=read(P/'packet.json');review=read(P/'root-review.json');require(review.get('approved')is True and review.get('packet_sha256')==sha(P/'packet.json')and review.get('launcher_sha256')==sha(__file__),'Fresh bound root review missing')
 require(review.get('single_attempt')is True,'Exactly one attempt required');require(not(P/'launch-claim.json').exists(),'Single attempt already consumed')
 require(sha(packet['prior_budget_path'])==packet['prior_budget_sha256'],'Prior budget changed');prior=read(packet['prior_budget_path']);require(abs(prior['remaining_original_one_dollar_usd']-.5962744434792181)<1e-12,'Original remaining budget changed')
 snap=read(P/'offers-sanitized.json');require(sha(P/'offers-sanitized.json')==packet['offer_snapshot_sha256'],'Reviewed snapshot changed');age=(dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(snap['queried_utc'])).total_seconds();require(0<=age<=600,'Fresh offer required')
 offer=packet['offer'];matches=[x for x in snap['offers']if x['id']==offer['id']];require(matches==[offer],'Reviewed offer changed');rate=offer['dph_total'];require(math.isfinite(rate)and 0<rate<=RATE_LIMIT,'Hourly rate exceeds cap');require(.5*rate+.10<=prior['remaining_original_one_dollar_usd'],'New attempt exceeds remaining cap')
 require(offer['gpu_name']=='RTX 5090'and offer['num_gpus']==1 and offer['cpu_arch']=='amd64'and offer['cuda_max_good']>=13,'Wrong target');require(offer['inet_down_cost']<=.002604167 and offer['inet_up_cost']<=.00390625,'Transfer rate exceeds reserve')
 stage=pathlib.Path(packet['staging']);require({x.name for x in stage.iterdir()}==set(packet['files']),'Staging inventory changed')
 for n,h in packet['files'].items():require(sha(stage/n)==h,'Staging changed: '+n)
 require(sha(P/'vast_run.sh')=='3e2d67a015810bdd98be23d9cac17c8f438c0b4d83d96563e4ae504b0ea6c487','Frozen wrapper changed')
 require(packet['max_seconds']==1800 and packet['retrieval_cleanup_reserve_seconds']==180 and packet['smallm_max_seconds']==1100,'Reviewed phase caps changed')
 active=[]
 for path in ['/private/tmp/mapal-s56-gpu/integrated/operations/ledger.json','/private/tmp/mapal-s58-gpu/operations/ledger.json']:
  active.extend(x for x in read(path).values()if x.get('active'))
 require(sum(x['listed_dph']for x in active)+rate<=8,'Aggregate rate cap exceeded');require(not any(x['machine_id']==offer['machine_id']for x in active),'Machine already reserved')
 return packet,inventory()

def cleanup(row,results):
 ident=row.get('instance');private=results/'create-response.json'
 if ident is None and private.exists():
  try:ident=read(private).get('new_contract')
  except(ValueError,AttributeError):pass
 if ident is None:
  candidates=[x for x in inventory()if x['id']not in row['prelaunch_ids']and x.get('machine_id')==row['machine_id']and x.get('host_id')==row['host_id']];require(len(candidates)<=1,'Ambiguous allocation; no unrelated destruction');ident=candidates[0]['id']if candidates else None
 if ident is None:
  created=read(private)if private.exists()else{};explicit=created.get('success')is False and not created.get('new_contract');return {'instance':None,'confirmed_absent':explicit,'verified_utc':now(),'method':'Explicit unsuccessful create withoutcontract'if explicit else'Allocation outcome ambiguous; manual owned inspection required'}
 require(isinstance(ident,int)and ident not in UNRELATED and ident not in row['prelaunch_ids'],'Refusing unrelated instance');row['instance']=ident
 for _ in range(3):
  matches=[x for x in inventory()if x['id']==ident]
  if not matches:return {'instance':ident,'confirmed_absent':True,'verified_utc':now(),'method':'Independent exact-ID inventory absence'}
  require(len(matches)==1 and matches[0].get('machine_id')==row['machine_id']and matches[0].get('host_id')==row['host_id'],'Physical ownership mismatch; no destruction')
  api('destroy','instance',ident,'-y');time.sleep(2)
 absent=all(x['id']!=ident for x in inventory());return {'instance':ident,'confirmed_absent':absent,'verified_utc':now(),'method':'Independent inventory after owned destruction'}

def stop(child):
 if child is None or child.poll()is not None:return
 try:os.killpg(child.pid,signal.SIGTERM)
 except ProcessLookupError:pass
 try:child.wait(timeout=5)
 except subprocess.TimeoutExpired:
  try:os.killpg(child.pid,signal.SIGKILL)
  except ProcessLookupError:pass
  child.wait(timeout=5)

def main():
 os.umask(0o077);packet,live=prepare();cli=shutil.which('vastai');require(cli,'Vast CLI absent');offer=packet['offer'];results=P/'results';results.mkdir(mode=0o700);started=time.monotonic();epoch=time.time();row={'offer':offer['id'],'machine_id':offer['machine_id'],'host_id':offer['host_id'],'listed_dph':offer['dph_total'],'active':True,'accepted':False,'prelaunch_ids':[x['id']for x in live],'create_started_utc':now()}
 lease={'create_started_epoch':epoch,'absolute_deadline_epoch':epoch+1800,'host_hourly_rate':offer['dph_total']};stage=pathlib.Path(packet['staging'])
 with(P/'launch-claim.json').open('x')as h:json.dump({'row':row,'packet_sha256':sha(P/'packet.json'),'root_review_sha256':sha(P/'root-review.json'),'lease':lease},h)
 write(stage/'lease.json',lease);write(P/'ledger.json',row);write(results/'request.json',{'packet':packet,'row':row,'lease':lease,'lease_sha256':sha(stage/'lease.json'),'root_review_sha256':sha(P/'root-review.json')})
 real=shlex.quote(cli);capture=results/'capture-create.sh';capture.write_text('vastai() {\n if [ "$1" = create ] && [ "$2" = instance ]; then\n '+real+' "$@" | tee '+shlex.quote(str(results/'create-response.json'))+'\n return "${PIPESTATUS[0]}"\n fi\n '+real+' "$@"\n}\n');env={k:v for k,v in os.environ.items()if k not in {'KEEP','WAIT_FILE','IMAGE','BASH_ENV'}};env.update(START_TRIES='24',PROFILE='0',DIFFERENTIAL_ONLY='0',BASH_ENV=str(capture));child=None;code=1
 try:
  with(results/'wrapper.log').open('x')as log:
   child=subprocess.Popen(['bash',str(P/'vast_run.sh'),str(offer['id']),'rtx-5090','sm_120',str(stage),str(results),'1'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:code=child.wait(timeout=max(1,1650-(time.monotonic()-started)))
   except subprocess.TimeoutExpired:code=124
 finally:
  stop(child);write(results/'execution.json',{'exit':code,'elapsed_seconds_before_independent_cleanup':time.monotonic()-started,'end_utc':now(),'wrapper_cutoff_seconds_from_create':1650,'absolute_limit_seconds':1800})
  try:
   proof=cleanup(row,results);write(results/'cleanup.json',proof);row.update(active=not proof['confirmed_absent'],cleanup_confirmed=proof['confirmed_absent'])
  finally:write(P/'ledger.json',row)
  print(json.dumps({'exit':code,'instance':row.get('instance'),'active':row['active'],'results':str(results)}))
 return code
if __name__=='__main__':raise SystemExit(main())
