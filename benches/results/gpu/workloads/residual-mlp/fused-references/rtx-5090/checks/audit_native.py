"""Independent local replay of the sealed native admission/selection protocol."""
import hashlib,importlib.util,json,math,re
from pathlib import Path
PACKET=Path('/private/tmp/mapal-composed-fused-references/packet-v1')
TRACE=Path('/private/tmp/mapal-s61-gpu/attempt-10/retrieved/trace')
RESULTS=TRACE/'composed-native';OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(x,message):
 if not x:raise ValueError(message)
def load(p):return json.loads(Path(p).read_text())
def hashes(root,files):
 for rel,digest in files.items():
  p=(root/rel).resolve();need(p.is_relative_to(root.resolve())and sha(p)==digest,'Artifact hash/path mismatch '+str(p))
 return len(files)
spec=importlib.util.spec_from_file_location('sealed_native_checker',PACKET/'validate_native.py');checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
CANDIDATES=[{'index':i,'tile':t,'warp':w}for i,t,w in [(0,[128,128,8],[32,64,8]),(1,[64,64,8],[32,32,8]),(2,[32,64,8],[16,32,8]),(3,[16,64,8],[16,32,8]),(4,[64,128,8],[32,64,8]),(5,[32,32,8],[16,16,8]),(6,[16,32,8],[16,32,8])]]
def ids(rows,fusion,indices):
 sites=('p','h-bias-relu','m-bias')if fusion=='fused'else('p','hraw','mraw');out={}
 for site,index in zip(sites,indices):
  found=[r['opaque_algorithm_hex']for r in rows if r['type']=='lt_algorithm'and r['site']==site and r['supported']and r['eligible_index']==index];need(len(found)==1,'Lt identity not unique');out[site]=found[0]
 return out

def main():
 manifest=load(PACKET/'manifest.json');summary=load(RESULTS/'summary.json');outer=load(TRACE/'operator-evidence/summary.json');recovery=load(TRACE.parent.parent/'evidence-recovery-receipt.json')
 need(sha(PACKET/'manifest.json')==summary['manifest_sha256']=='4fb22559cc07319d46ae719425a03e8efc258ebeea1ffeda20e6e40245c8d5b0','Wrong sealed packet');inventory={'native_payload':hashes(PACKET,manifest['files']),'native_results':hashes(RESULTS,summary['files']),'outer':hashes(TRACE/'operator-evidence',outer['files']),'recovery':hashes(TRACE,recovery['files'])};need(sha(Path(recovery['archive']['path']))==recovery['archive']['sha256'],'Recovery archive mismatch');need(recovery['cleanup']['confirmed_absent']and recovery['cleanup']['instance']==57259496,'Cleanup not confirmed')
 refs={case:checker.load_case(case)for case in ('tiny','matrix','matrix-large')};data_sha=sha(PACKET/'reference-data/manifest.json');need(summary['reference_manifest_sha256']==data_sha and summary['finite_candidate_space']==CANDIDATES,'Reference/candidate identity differs')
 builds={}
 for index in (-1,*range(7)):
  b=summary['builds'][str(index)];folder=RESULTS/'build'/str(index);need(b==load(folder/'build-receipt.json')and b['pass']and b['exit_code']==0,'Build receipt failure');need(b['binary']['sha256']==sha(folder/'native')and b['stdout_sha256']==sha(folder/'stdout')and b['stderr_sha256']==sha(folder/'stderr'),'Build/stream hashes mismatch');expected=['nvcc','-O3','-std=c++17','-arch=sm_120','-DCOMPOSED_CUTLASS_CANDIDATE='+str(index),'-I/w/composed-fused-reference/cutlass/include','/w/composed-fused-reference/composed_native.cu','-lcublas','-lcublasLt','-o','/w/trace/composed-native/build/'+str(index)+'/native'];need(b['argv']==expected and b['binary']['path']==expected[-1]and b['manifest_sha256']==summary['manifest_sha256'],'Build argv/source identity differs');builds[index]=b
 all_records=[*summary['tiny_gates'],*summary['tuning'],*summary['confirmations']];all_records.sort(key=lambda r:int(r['path'].split('/')[1].split('-')[0]));need(len(all_records)==148 and[int(r['path'].split('/')[1].split('-')[0])for r in all_records]==list(range(1,149)),'Call inventory/gaps');by_path={};tiny=set();word_count=0
 for position,record in enumerate(all_records,1):
  folder=RESULTS/record['path'];receipt=load(folder/'receipt.json');need(receipt=={k:v for k,v in record.items()if k!='roles'},'Final summary/work receipt differ');need(record['pass']and record['exit_code']==0 and record['manifest_sha256']==summary['manifest_sha256'],'Work status/manifest differs');case=record['case'];shape=refs[case]['shape'];candidate=record['candidate'];backend=record['backend'];fusion=record['fusion'];mode=record['mode'];tuning=record['tuning'];request=load(folder/'request.json');need(request==record['request'],'Request receipt differs');need(candidate in builds and(candidate==-1 if backend!='cutlass'else candidate>=0),'Wrong binary/backend pairing')
  expected_request={'backend':backend,'fusion':fusion,'contract_sha256':data_sha,'mode':mode,'tuning':tuning,'cutlass_candidate':candidate,'lt_indices':record['indices'],'nonce':request['nonce']}
  if backend=='cutlass':expected_request['cutlass_geometry']=CANDIDATES[candidate]
  if 'lt_algorithm_identities'in request:expected_request['lt_algorithm_identities']=request['lt_algorithm_identities']
  need(request==expected_request and re.fullmatch('[0-9a-f]{32}',request['nonce']),'Request field/nonce identity differs');argv=['/w/trace/composed-native/build/'+str(candidate)+'/native','/w/composed-fused-reference/reference-data/'+case+'/inputs.bin',*[str(shape[k])for k in ('M','d','h')],backend,fusion,mode,','.join(map(str,record['indices'])),request['nonce'],data_sha]+(['--tuning']if tuning else[]);need(argv==record['argv'],'Actual work argv differs');need(record['binary_sha256']==builds[candidate]['binary']['sha256'],'Work binary changed')
  identity={'input_sha256':sha(PACKET/'reference-data'/case/'inputs.bin'),'binary_sha256':record['binary_sha256'],'bundle_manifest_sha256':summary['manifest_sha256'],'reference_manifest_sha256':data_sha};need(identity==record['execution_identity'],'Executed input/binary identity differs');text=(folder/'stdout').read_text();rows=checker.parse(text);report=checker.analyze(text,refs[case],expected_request=expected_request);need(report==record['admission']==load(folder/'admission.json')and report['pass'],'Independent numeric/clock admission differs');need(record['stdout_sha256']==sha(folder/'stdout')and record['stderr_sha256']==sha(folder/'stderr'),'Work stream hash differs');word_count+=sum(len(r['bits_le'])for r in rows if r['type']in ('checkpoint','final','sample_final'))
  if mode=='time':
   ready=next(i for i,r in enumerate(rows)if r['type']=='ready_for_validation');prefix=checker.admit_prefix(rows[:ready+1],refs[case],expected_request=expected_request);need(prefix==load(folder/'prefix-admission.json')and record['acknowledgement_sent'],'Pre-clock prefix/ACK missing');need((backend,fusion,candidate)in tiny,'Missing earlier full tiny prerequisite')
  else:need(case=='tiny'and not tuning and not record['acknowledgement_sent'],'Malformed tiny gate');tiny.add((backend,fusion,candidate))
  metadata=report['metadata'];M,d,h=(shape[k]for k in ('M','d','h'));expected_bytes=4*(d*d+2*d*h+h+d+M*d+4*M*d+2*M*h);need(metadata['tensor_buffer_allocations']==12 and metadata['tensor_buffer_bytes']==expected_bytes and metadata['explicit_workspace_bytes']==((32<<20)if backend=='cublaslt'else 0),'Storage accounting differs')
  if backend=='cublas':v=[r for r in rows if r['type']=='cublas_version'];need(len(v)==1 and v[0]['version']==120804,'cuBLAS queried version differs')
  if backend=='cublaslt':
   descriptors=[r for r in rows if r['type']=='lt_descriptor'];sites=['p','hraw','mraw']+(['h-bias-relu','m-bias']if fusion=='fused'else[]);need([r['site']for r in descriptors]==sites,'Lt descriptor inventory differs');site_dims={'p':(d,d),'hraw':(d,h),'mraw':(h,d),'h-bias-relu':(d,h),'m-bias':(h,d)}
   for desc in descriptors:
    K,N=site_dims[desc['site']];bias=desc['site']in ('h-bias-relu','m-bias');need(desc['logical_MNK']==[M,N,K]and desc['api_MNK']==[N,M,K]and desc['ld']==[N,K,N,N]and desc['order']=='column'and desc['compute']=='CUBLAS_COMPUTE_32F_PEDANTIC'and desc['bias_length']==(N if bias else 0)and desc['alpha']==1 and desc['beta']==0 and desc['workspace_cap']==32<<20 and desc['heuristic_status']==0,'Lt descriptor math/layout differs');algorithms=[r for r in rows if r['type']=='lt_algorithm'and r['site']==desc['site']];need(len(algorithms)==desc['returned']<=8,'Heuristic inventory differs');eligible=[r for r in algorithms if r['supported']];need([r['eligible_index']for r in eligible]==list(range(len(eligible))),'Supported filter order differs')
    for algorithm in eligible:need(all(algorithm[k]==0 for k in ('state','check_status','check_state'))and algorithm['workspace']<=32<<20 and algorithm['checked_workspace']<=32<<20,'Unsupported selected candidate');need(all(v['status']==0 for v in algorithm['configuration'].values()),'Unknown queried Lt config field')
   need(ids(rows,fusion,record['indices'])==record['lt_selected_identities'],'Selected Lt identity summary differs')
  by_path[record['path']]={'record':record,'rows':rows,'report':report,'sequence':position}
 # Replay the exact finite selection, including coordinate order and tuple ties.
 for selection in summary['selections']:
  case,backend,fusion=(selection[k]for k in ('case','backend','fusion'));trials=[by_path[r['path']]for r in summary['tuning']if r['case']==case and r['backend']==backend and r['fusion']==fusion];score=lambda x:x['report']['observations']['wall_ms']['upper_median'];need(selection['confirmation_data_used']is False,'Confirmation used to select')
  if backend=='cutlass':need([x['record']['candidate']for x in trials]==list(range(7)),'Finite CUTLASS candidate inventory differs');best=min(trials,key=lambda x:(score(x),x['record']['candidate']));need(selection['candidate']==best['record']['candidate'],'Wrong CUTLASS selected candidate')
  else:
   cursor=1;best=trials[0];need(best['record']['indices']==[0,0,0],'Lt sweep initial tuple differs');indices=[0,0,0];sites=('p','h-bias-relu','m-bias')if fusion=='fused'else('p','hraw','mraw')
   for coordinate,site in enumerate(sites):
    available=sorted(r['eligible_index']for r in best['rows']if r['type']=='lt_algorithm'and r['site']==site and r['supported']);proposals=[best]
    for index in available:
     if index==indices[coordinate]:continue
     expected=list(indices);expected[coordinate]=index;trial=trials[cursor];cursor+=1;need(trial['record']['indices']==expected,'Lt coordinate proposal/order differs');proposals.append(trial)
    best=min(proposals,key=lambda x:(score(x),x['record']['indices']));indices=list(best['record']['indices'])
   need(cursor==len(trials)and selection['indices']==indices and selection['algorithm_identities']==ids(best['rows'],fusion,indices),'Lt coordinate winner/algorithm freeze differs')
  need(selection['tuning_receipt']==best['record']['path'],'Wrong selecting receipt');saved=load(RESULTS/'selections'/f'{case}-{backend}-{fusion}.json');need(saved==selection,'Frozen selection artifact differs')
  confirmations=[r for r in summary['confirmations']if r['case']==case and r['backend']==backend and r['fusion']==fusion];need(len(confirmations)==2,'Selected/control confirmation pair differs')
  for confirmation in confirmations:
   need(by_path[confirmation['path']]['sequence']>max(x['sequence']for x in trials),'Holdout occurred before tuning froze');need(not confirmation['tuning']and confirmation['admission']['timing_admitted']and confirmation['admission']['protocol']['warmups']==5 and confirmation['admission']['protocol']['measured']==30,'Changed heldout protocol')
   if 'selected'in confirmation['roles']:
    need(confirmation['candidate']==selection['candidate']and confirmation['indices']==selection['indices'],'Wrong selected holdout')
    if backend=='cublaslt':need(confirmation['request']['lt_algorithm_identities']==selection['algorithm_identities'],'Selected opaque Lt algorithm changed')
   elif backend=='cublaslt':need(confirmation['indices']==[0,0,0]and confirmation['request']['lt_algorithm_identities']==ids(trials[0]['rows'],fusion,[0,0,0]),'Heuristic-zero control changed')
   else:need(confirmation['candidate']==0,'Matched-default geometry changed')
 need(len(summary['selections'])==8 and len(summary['confirmations'])==20 and len(tiny)==18 and not summary['failures']and not summary['rejections'],'Incomplete native study')
 for phase in ('native_comparison','torch_bootstrap','composed_controls'):
  identity=(TRACE/'operator-evidence'/('gpu-identity-before-'+phase)/'stdout').read_text().strip();need(identity==outer['device_identity'],'Cross-phase GPU/driver identity changed');need(not(TRACE/'operator-evidence'/('gpu-idle-before-'+phase)/'stdout').read_text().strip(),'GPU not idle before phase')
 need(outer['pass']and outer['native_comparison_pass']and outer['composed_controls_pass']and outer['torch_bootstrap_pass'],'Outer phase failed')
 report={'schema':'residual-mlp-native-independent-admission-v1','pass':True,'native_gate_processes':18,'native_tuning_processes':110,'native_confirmation_processes':20,'admitted_native_confirmation_rows':20,'native_checked_raw_value_words':word_count,'payload_artifact_hashes_verified':inventory,'manifest_sha256':sha(PACKET/'manifest.json'),'native_summary_sha256':sha(RESULTS/'summary.json'),'outer_summary_sha256':sha(TRACE/'operator-evidence/summary.json'),'recovery_receipt_sha256':sha(TRACE.parent.parent/'evidence-recovery-receipt.json'),'device_identity':outer['device_identity'],'cleanup':recovery['cleanup'],'selection':summary['selections'],'confirmation_rows':[{'path':r['path'],'case':r['case'],'backend':r['backend'],'fusion':r['fusion'],'candidate':r['candidate'],'indices':r['indices'],'roles':r['roles'],'observations':r['admission']['observations'],'metadata':r['admission']['metadata']}for r in summary['confirmations']],'all_selections_replayed':True,'all_selected_lt_algorithms_frozen':True,'pre_ack_prefixes_rechecked':130,'limits':['Full tiny; representative256 cells per stage/final, no full representative tensor equality claim.','In native all timed selected y outputs are checked after clocks; original controls retain their original checked scope.','Libraries pass unchanged bounded numerical criteria; Mapal Contract remains its separate ordered contract.','Native, bootstrap and original controls are successive blocks on one GPU; no causal fusion/cache/allocation claim or randomized block comparison.','Mapal generation-before-clock and retained-reference input cache boundary remains unequal.','Actual library API counts do not identify hardware kernel launch count; executed SASS/counters remain uncaptured.']}
 (OUT/'admission.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k not in ('selection','confirmation_rows','limits')}))
if __name__=='__main__':main()
