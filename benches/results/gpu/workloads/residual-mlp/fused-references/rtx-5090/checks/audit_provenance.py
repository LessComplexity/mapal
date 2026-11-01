import pathlib,json,hashlib,tarfile,re,datetime
P=pathlib.Path(__file__).parent;R=P/'retrieved/trace';sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest();load=lambda q:json.loads(q.read_text())
outer=load(R/'operator-evidence/summary.json');native=load(R/'composed-native/summary.json');recovery=load(P/'evidence-recovery-receipt.json');verified={}
for label,folder in [('operator','operator-evidence'),('native','composed-native'),('bootstrap','torch-bootstrap'),('controls_matrix','composed-controls/matrix'),('controls_matrix_large','composed-controls/matrix-large')]:
 d=load(R/folder/'summary.json');assert all((R/folder/n).is_file()and sha(R/folder/n)==h for n,h in d['files'].items()),label;verified[label]={'sealed_files':len(d['files']),'summary_sha256':sha(R/folder/'summary.json'),'pass':d['pass']}
assert all(sha(R/n)==h for n,h in recovery['files'].items());archive_rows={}
with tarfile.open(P/'results/trace-rtx-5090.tgz')as t:
 for m in t.getmembers():
  if m.isfile():
   assert m.name.startswith('trace/');n=m.name[6:];assert n not in archive_rows;archive_rows[n]=hashlib.sha256(t.extractfile(m).read()).hexdigest()
assert archive_rows==recovery['files'];names=['native_comparison','torch_bootstrap','composed_controls'];ids=[(R/'operator-evidence'/('gpu-identity-before-'+n)/'stdout').read_text().strip()for n in names];assert len(set(ids))==1
assert all(not(R/'operator-evidence'/('gpu-idle-'+side+'-'+n)/'stdout').read_text().strip()for n in names for side in ['before','after'])
math={k:set()for k in ['cublas_versions','cublas_math_modes','lt_versions','lt_compute_types','lt_epilogues','cutlass_operators','cutlass_instruction_shapes','native_driver_api_versions','native_runtime_versions','tf32_overrides']}
for q in(R/'composed-native/calls').glob('*/stdout'):
 for line in q.read_text().splitlines():
  try:d=json.loads(line)
  except ValueError:continue
  kind=d.get('type')
  if kind=='metadata':
   for a,b in [('native_driver_api_versions','driver'),('native_runtime_versions','runtime'),('tf32_overrides','tf32_override')]:math[a].add(d[b])
  elif kind=='cublas_version':math['cublas_versions'].add(d['version']);math['cublas_math_modes'].add(d['math'])
  elif kind=='lt_version':math['lt_versions'].add(d['version'])
  elif kind=='lt_descriptor':math['lt_compute_types'].add(d['compute']);math['lt_epilogues'].add(d['epilogue'])
  elif kind=='cutlass_plan':math['cutlass_operators'].add(d['operator']);math['cutlass_instruction_shapes'].add(tuple(d['instruction']))
math={k:sorted(v)for k,v in math.items()};torch=[]
for case in ['matrix','matrix-large']:
 for mode in ['eager','compiled']:
  q=R/'composed-controls'/case/('matrix-torch-'+mode+'.json');d=load(q);torch.append({'case':case,'mode':mode,**{k:d[k]for k in ['torch','cuda','precision','pass']},'report_sha256':sha(q)})
components=[]
for name in ['composed-fused-native-reference-v1.tar.gz','composed-matrix-v4.tgz']:
 q=P/'staging'/name
 with tarfile.open(q)as t:
  m=next(x for x in t.getmembers()if x.name.endswith('/manifest.json')and x.name.count('/')==1);raw=t.extractfile(m).read();d=json.loads(raw)
 components.append({'archive':name,'archive_sha256':sha(q),'archive_bytes':q.stat().st_size,'manifest_sha256':hashlib.sha256(raw).hexdigest(),'sealed_payload_files':len(d['files']),'cutlass_commit':d.get('cutlass_commit'),'emitter_sha256':d.get('emitter_sha256'),'source_files':{k:v for k,v in d['files'].items()if not k.startswith(('cutlass/','reference-data/','ptx/','oracles/','cpu-evidence/'))and k.endswith(('.cu','.py','.cpp','.hpp','.c'))}})
claim=load(P/'launch-claim.json');cleanup=load(P/'results/cleanup.json');assert cleanup['confirmed_absent']and cleanup['instance']==57259496 and cleanup['instance']not in claim['row']['prelaunch_ids'];assert datetime.datetime.fromisoformat(cleanup['verified_utc']).timestamp()<claim['lease']['absolute_deadline_epoch']
hits=[];suspect_paths=[];scanned=0
patterns=[rb'-----BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY-----',rb'AKIA[0-9A-Z]{16}',rb'\bsk-[A-Za-z0-9]{40,}',rb'Authorization:\s*Bearer\s+[A-Za-z0-9_-]{32,}',rb'https?://[^\s"\x27<>]+[?&](?:api_key|access_token|token)=[A-Za-z0-9_-]{20,}']
for folder in ['composed-native','composed-controls']:
 for q in(R/folder).rglob('*'):
  if not q.is_file():continue
  scanned+=1;b=q.read_bytes();n=str(q.relative_to(R))
  if any(re.search(z,b)for z in patterns):hits.append(n)
  if any(z in q.relative_to(R).parts for z in ['.ssh','.aws','.env','credentials','id_rsa','id_ed25519']):suspect_paths.append(n)
assert not hits and not suspect_paths
r={'title':'Native and same-host controls public provenance audit','pass':True,'scope':'Recovery, software/math metadata, identity, source/command order and cleanup only; no numerical rerun or performance ranking.','device_identity':ids[0],'identity_same_across_three_phases':True,'all_six_idle_receipts_empty':True,'nvcc':re.search(r'release ([0-9.]+), V([0-9.]+)',(R/'operator-evidence/nvcc-version/stdout').read_text()).groups(),'math_metadata':math,'torch_controls':torch,'phase_order':names,'phase_commands_in_exact_recorded_order':outer['commands'],'phase_wall_seconds':{x['name']:x['seconds']for x in outer['commands']if x['name']in names},'per_phase_timestamp_limit':'Absolute per-phase UTC starts were not recorded. Exact receipt order and measured durations are recorded; no inferred UTC boundaries.','native_build_commands':[load(q)for q in sorted((R/'composed-native/build').glob('*/build-receipt.json'))],'components':components,'verified_seals':verified,'recovery':{'all_recovered_files':len(archive_rows),'archive_sha256':sha(P/'results/trace-rtx-5090.tgz'),'all_archive_members_match_recovered_manifest':True,'recovery_receipt_sha256':sha(P/'evidence-recovery-receipt.json')},'ownership':{'create_utc':claim['row']['create_started_utc'],'owned_id':57259496,'machine_id':58084,'host_id':98309,'preexisting_ids_protected':claim['row']['prelaunch_ids'],'cleanup':cleanup,'cleanup_sha256':sha(P/'results/cleanup.json'),'absence_before_absolute_deadline':True},'public_raw_tree_scan':{'files_scanned':scanned,'high_confidence_credential_pattern_hits':hits,'suspicious_credential_paths':suspect_paths},'limitations':['OS driver580.142 and CUDA driver API13000 use different version schemes. Native runtime12080 is CUDA12.8; Torch reports bundledCUDA13.0.','Math settings are reported/source-bound configuration, not captured library instruction evidence. No NCU/counters were run.','Private library allocations remain unmeasured.','Pattern scans cannot prove absence of every possible secret; no unexpected secrets found in reviewed ordinary trees.'],'redaction':'No raw provider/API/account data, endpoints, credentials, payment data. No shared edits or new rental.'}
q=P/'public-native-control-provenance.json';q.write_text(json.dumps(r,indent=2)+'\n');q.chmod(0o600)
print(json.dumps({'report':str(q),'sha256':sha(q),'verified_seals':verified,'math':math,'ordinary_files_scanned':scanned,'unexpected_secrets':0,'cleanup_confirmed':True}))
