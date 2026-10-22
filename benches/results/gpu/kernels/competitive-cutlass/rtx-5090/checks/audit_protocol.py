"""Read-only independent admission of retrieved ordinary GPU artifacts."""
import hashlib,importlib.util,json,math,re,struct
from pathlib import Path
PACKET=Path('/private/tmp/mapal-s61-competitive-reference/packet-v2')
RESULTS=Path('/private/tmp/mapal-s61-gpu/attempt-09/retrieved/trace/competitive-ordinary')
OUTER=RESULTS.parent/'operator-evidence'
OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(x,message):
 if not x:raise ValueError(message)
def verify_files(root,files):
 for name,digest in files.items():
  target=(root/name).resolve();require(target.is_relative_to(root.resolve())and sha(target)==digest,'Hash/path mismatch '+str(target))
 return len(files)
def fields(line):return dict(re.findall(r'(\w+)=([^\s]+)',line))
def unique(text,prefix):
 found=[x for x in text.splitlines()if x.startswith(prefix)];require(len(found)==1,'Missing/duplicate '+prefix);return fields(found[0])
def raw_samples(text,n):
 found=re.findall(r'^samples mapal=(\S+) cublas=(\S+) cutlass=(\S+)$',text,re.M);require(len(found)==1,'Wrong samples record inventory');out={}
 for arm,raw in zip(('mapal','cublas','cutlass'),found[0]):
  values=list(map(float,raw.split(',')));require(len(values)==n and all(math.isfinite(v)and v>0 for v in values),'Invalid raw samples');v=sorted(values);out[arm]={'raw_event_ms':values,'upper_median_ms':v[n//2],'p10_ms':v[math.ceil(.1*n)-1],'p90_ms':v[math.ceil(.9*n)-1],'n':n}
 return out
spec=importlib.util.spec_from_file_location('frozen_independent_gate',PACKET/'numerical_gate.py');gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
def main():
 manifest=json.loads((PACKET/'manifest.json').read_text());require(sha(PACKET/'manifest.json')=='ab7495e45fb491a696290a444abf230b4704aeb0f3e68af68aa95c0e40c1d17f','Wrong admitted packet');summary=json.loads((RESULTS/'summary.json').read_text());outer=json.loads((OUTER/'summary.json').read_text());cases={r['id']:r for r in json.loads((PACKET/'cases.json').read_text())};candidates={r['id']:r for r in json.loads((PACKET/'candidates.json').read_text())}
 inventory={'payload':verify_files(PACKET,manifest['files']),'ordinary':verify_files(RESULTS,summary['files']),'outer':verify_files(OUTER,outer['files'])};require(summary['manifest_sha256']==sha(PACKET/'manifest.json')and summary['mapal_variant']=='old'and not summary['instrumented'],'Wrong run identity')
 builds={};rejected={}
 for name,candidate in candidates.items():
  build=summary['candidate_builds'][name];folder=RESULTS/'build'/name;receipt=json.loads((folder/'build-receipt.json').read_text());status=json.loads((folder/'status.json').read_text());require(receipt['candidate']==name and receipt['build_index']==candidate['build_index']and receipt['manifest_sha256']==summary['manifest_sha256']and receipt['build_status_sha256']==sha(folder/'status.json'),'Build receipt mismatch');require(build['process']==status and receipt['pass']==build['pass']==status['pass'],'Build status disagreement');require('-DCOMPETITIVE_CANDIDATE_INDEX='+str(candidate['build_index'])in status['argv']and status['argv'][0]=='nvcc'and'-arch=sm_120'in status['argv'],'Build candidate/compiler identity mismatch')
  if build['pass']:
   require(build['binary_sha256']==receipt['binary']['sha256']==sha(folder/'checked'),'Executed build hash mismatch');builds[name]=receipt
  else:
   stderr=(folder/'stderr').read_text();require('ThreadMap::Iterations::kColumn must be > 0'in stderr,'Unexpected compile failure');rejected[name]={'reason':'PredicatedTileIterator static assertion: ThreadMap::Iterations::kColumn must be > 0','stderr_sha256':sha(folder/'stderr')}
 records=[];gate_pass=set();numerical_cells=0
 for stage,key in (('gate','gates'),('tuning','tuning'),('confirmation','confirmations')):
  for record in summary[key]:
   case=cases[record['case']];candidate=candidates[record['candidate']];require(record['candidate']in builds,'Unbuilt candidate executed');folder=RESULTS/record['path'];require(folder==RESULTS/stage/(case['id']+'__'+candidate['id']),'Record path changed');status=json.loads((folder/'status.json').read_text());require(status==record['process']and status['pass']and status['exit']==0 and status['terminated_exit']==0,'Process failure');require(status['stdout_sha256']==sha(folder/'stdout')and status['stderr_sha256']==sha(folder/'stderr'),'Process stream hash mismatch');require(record['binary_sha256']==sha(RESULTS/'build'/candidate['id']/'checked'),'Record binary identity mismatch')
   variant=case['variants']['old'];oracle=manifest['oracles'][case['oracle_id']];reference=json.loads((PACKET/oracle['json']).read_text());require(record['oracle_text_sha256']==sha(PACKET/oracle['text'])and record['oracle_json_sha256']==sha(PACKET/oracle['json'])and record['mapal_ptx_sha256']==sha(PACKET/variant['ptx'])==variant['ptx_sha256'],'Input/oracle/PTX identity mismatch');launch=variant['launch'];n=9 if stage=='tuning'else 30
   expected=['/w/trace/competitive-ordinary/build/'+candidate['id']+'/checked',*[str(case[k])for k in ('M','N','K')],'/w/competitive-reference/'+variant['ptx'],launch['entry'],*[str(launch[k])for k in ('gx','gy','block','smem')],str(n),'--cutlass-candidate',candidate['id'],'--oracle-file','/w/competitive-reference/'+oracle['text'],'--oracle-sha256',sha(PACKET/oracle['text']),'--mapal-ptx-sha256',variant['ptx_sha256'],'--manifest-sha256',summary['manifest_sha256'],'--mapal-variant','old']
   if case['face']=='contract':expected+=['--contract']
   if case['stage']=='gate':expected+=['--full-values']
   if stage=='gate':expected+=['--values-only']
   if stage=='tuning':expected+=['--tuning']
   require(status['argv']==expected,'Executed argv changed');text=(folder/'stdout').read_text();identity=unique(text,'identity ');require(identity=={'candidate':candidate['id'],'mapal_variant':'old','mapal_ptx_sha256':variant['ptx_sha256'],'oracle_sha256':sha(PACKET/oracle['text']),'manifest_sha256':summary['manifest_sha256'],'tf32_override':'0','tuning':str(int(stage=='tuning')),'profile_only':'0'},'Harness runtime identity changed');require(not any(x.startswith('PROFILE_PROTOCOL')for x in text.splitlines()),'Instrumented stdout in ordinary run')
   numeric=gate.check(text,reference,case['face']);require(numeric['pass']and numeric==record['numerical']==json.loads((folder/'independent-numerical.json').read_text()),'Independent numerical disagreement');numerical_cells+=sum(a['cells']for a in numeric['arms'].values())
   plan=unique(text,'competitive ');require(plan['candidate']==candidate['id']and plan['tb']==','.join(map(str,candidate['threadblock']))and plan['warp']==','.join(map(str,candidate['warp']))and plan['instruction']=='1,1,1'and int(plan['stages'])==candidate['stages']and int(plan['block'])==candidate['threads'],'Executed geometry differs');require(all(plan[k]=='f32'for k in ('dtype','accumulator'))and all(plan[k]=='row'for k in ('layout_A','layout_B','layout_C'))and plan['alpha']=='1'and plan['beta']=='0'and plan['operator']=='SIMT'and plan['split_k_slices']=='1'and plan['workspace_bytes']=='0','Precision/storage/math contract differs')
   require([int(plan[k])for k in ('stride_A','stride_B','stride_C')]==[case['K'],case['N'],case['N']],'Packed input/output strides differ');grid=list(map(int,plan['grid'].split(',')));require(grid==[math.ceil(case['M']/candidate['threadblock'][0]),math.ceil(case['N']/candidate['threadblock'][1]),1],'CUTLASS grid differs')
   libraries=unique(text,'libraries ');require(libraries['requested_math_mode']=='default'and libraries['reported_math_mode']=='0'and libraries['cublas_version']=='120804','Actual cuBLAS mode/version differs')
   if case['stage']=='timing':require((candidate['id'],case['gate_id'])in gate_pass,'Missing preceding full gate')
   checked_hashes={}
   for arm in ('mapal','cublas','cutlass'):
    words=re.findall(r'^checked_value leg='+arm+r' ordinal=\d+ row=\d+ col=\d+ bits=([0-9a-f]{8})$',text,re.M);checked_hashes[arm]=hashlib.sha256(b''.join(struct.pack('<I',int(word,16))for word in words)).hexdigest()
   row={'stage':stage,'case':case['id'],'candidate':candidate['id'],'face':case['face'],'family':case['family'],'path':record['path'],'numerical':numeric,'checked_output_words_sha256':checked_hashes,'input_A_sha256_frozen_generator':reference['input_A_sha256'],'input_B_sha256_frozen_generator':reference['input_B_sha256'],'resources':plan,'libraries':libraries,'stdout_sha256':sha(folder/'stdout'),'binary_sha256':record['binary_sha256'],'pass':True}
   if stage=='gate':require(not re.search(r'^(time |samples |ratio )',text,re.M),'Gate emitted timings');gate_pass.add((candidate['id'],case['id']))
   else:
    row['samples']=raw_samples(text,n)
    for arm,values in row['samples'].items():require(all(record['events'][arm][k]==v for k,v in values.items()),'Summary sample statistics differ from raw')
   records.append(row)
 selections={}
 for family,selection in summary['selection'].items():
  eligible=[r for r in records if r['stage']=='tuning'and r['family']==family];require(len(eligible)==len(builds)==selection['candidate_count'],'Tuning candidate space incomplete');winner=min(eligible,key=lambda r:(r['samples']['cutlass']['upper_median_ms'],r['candidate']));require(winner['candidate']==selection['candidate']and winner['samples']['cutlass']['upper_median_ms']==selection['tuning_upper_median_event_ms']and selection['selection_uses_confirmation_data']is False,'Selection changed or used wrong metric');require(selection['binary_sha256']==builds[winner['candidate']]['binary']['sha256'],'Selected binary changed');selections[family]=selection
  for face in ('exact','contract'):
   confirmations=[r for r in records if r['stage']=='confirmation'and r['family']==family and r['face']==face];require({r['candidate']for r in confirmations}=={'default-control',winner['candidate']},'Missing/extra selected/default holdout');require(all(r['samples']['mapal']['n']==30 for r in confirmations),'Wrong heldout sample count')
 require(len(records)==68 and len(gate_pass)==48 and len(selections)==2,'Wrong full study inventory')
 report={'schema':'competitive-primitive-independent-admission-v1','pass':True,'admitted_heldout_processes':8,'admitted_heldout_arm_rows':24,'independently_checked_raw_value_words':numerical_cells,'hash_inventory':inventory,'manifest_sha256':summary['manifest_sha256'],'summary_sha256':sha(RESULTS/'summary.json'),'outer_summary_sha256':sha(OUTER/'summary.json'),'ordinary_pass':summary['pass'],'outer_batch_pass':outer['pass'],'outer_failure_is_separate_ncu':outer.get('competitive_ordinary_pass')is True and not outer.get('ncu_launch_pass',True),'builds_admitted':sorted(builds),'builds_rejected':rejected,'selection':selections,'records':records,'limitations':['Actual GPU A/B bytes were not exported/hash-receipted; input hashes name the frozen deterministic generator inputs.','Ordinary raw checked values are pre-timing exports; timed outputs are not exported per observation.','Libraries pass bounded componentwise numerical gates to both scalar faces; no ordered-bit guarantee is inferred.','Mapal old variant is frozen baseline; finite2families/2faces/12attempted6compiledgeometry space, no generic selector claim.','Timing is original 3-leg round-robin event interval with256MiB memset before start; stdout samples are rounded to4decimal places in ms.','Separate NCU failure does not admit traced timing or kernel SASS conclusions.']}
 (OUT/'admission.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k not in ('records','limitations','selection','builds_rejected')}));return report
if __name__=='__main__':main()
