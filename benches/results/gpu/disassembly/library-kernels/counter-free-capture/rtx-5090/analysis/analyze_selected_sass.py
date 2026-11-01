"""Read admitted immutable images/decoder output; no GPU/remote calls or timing use."""
import collections,hashlib,json,re
from pathlib import Path
R=Path('/private/tmp/mapal-s61-gpu/attempt-13/retrieved/trace/library-images');OUT=Path(__file__).resolve().parent
REVIEW=Path('/private/tmp/mapal-library-image-review/repair-review/live-admission-v2/attempt13/admission.json')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
independent=json.loads(REVIEW.read_text());assert independent['pass']and all(v['pass']for v in independent['legs'].values())
summary=json.loads((R/'summary.json').read_text());assert summary['pass']and not summary['timing_admissible']and not summary['performance_counters_requested']
labels={('mapal',0):'mapal-independent-output',('cublas',0):'cublas-gemm',('cublas',1):'cublas-reduction',('cutlass',0):'cutlass-16x64x16'}
allrows=[];copy=[]
for leg in('mapal','cublas','cutlass'):
 a=json.loads((R/(leg+'-admission.json')).read_text());assert len(a['selected_launches'])==len(a['sass'])
 for index,(launch,decoded)in enumerate(zip(a['selected_launches'],a['sass'])):
  label=labels[(leg,index)];file=R/(leg+'-sass-'+str(index))/'stdout';text=file.read_text();assert sha(file)==decoded['sass_sha256'];assert launch['symbol']==decoded['symbol']and launch['module_sha256']==decoded['module_sha256']and launch['elf_symbol_index']==decoded['elf_symbol_index'];image=next(i for i in a['images']if i['event']==launch['image_event']);f=R/(leg+'-profile')/'capture'/image['path'];assert sha(f)==launch['module_sha256'];entry=image['texts'][launch['symbol']]
  ins=[]
  for line,raw in enumerate(text.splitlines(),1):
   match=re.match(r'\s*/\*([0-9a-fA-F]+)\*/\s+(?:(@!?U?P(?:\d+|T))\s+)?([A-Z][A-Z0-9_.]*)\b(.*)',raw)
   if match:ins.append({'line':line,'address':'0x'+match[1],'predicate':match[2],'opcode':match[3],'operands_and_annotations':match[4].strip(),'raw':raw.strip(),'constant_false_predicate':match[2]in('@!PT','@!UPT')})
   elif re.match(r'\s*/\*[0-9a-fA-F]+\*/',raw):raise ValueError('Unparsed instruction '+raw)
  assert len(ins)*16==entry['size'];assert [int(i['address'],16)for i in ins]==list(range(0,entry['size'],16))
  opcounts=collections.Counter(i['opcode']for i in ins);false=collections.Counter(i['opcode']for i in ins if i['constant_false_predicate']);unknown=collections.Counter(re.findall(r'\?[A-Za-z0-9_]+',text))
  families={name:[i for i in ins if i['opcode'].split('.')[0]in bases]for name,bases in {'fma':{'FFMA'},'add':{'FADD'},'mul':{'FMUL'},'global_load':{'LDG'},'global_store':{'STG'},'shared_load':{'LDS'},'shared_store':{'STS'},'global_to_shared':{'LDGSTS'},'dependency_barrier':{'LDGDEPBAR','DEPBAR'},'block_barrier':{'BAR'},'warp_sync':{'WARPSYNC'},'shuffle':{'SHFL'},'call':{'CALL','JCAL','BRX','JMX'},'tensor_mnemonics':{'HMMA','IMMA','MMA','WGMMA','UMMA','TCGEN05'}}.items()}
  anchors={name:[v[0],v[-1]]if len(v)>1 else v for name,v in families.items()if v}
  resources={'registers_per_thread':40 if leg=='mapal'else 56 if leg=='cutlass'else None,'local_bytes':0 if leg in('mapal','cutlass')else None,'static_shared_bytes':0 if leg=='cutlass'else None,'dynamic_shared_bytes':launch['smem'],'source':'profile stdout lines5 and7 for explicit Mapal/CUTLASS function attribute queries; selected cuBLAS internal attributes not recorded','no_max_register_name_inference':True}
  allrows.append({'label':label,'leg':leg,'launch':launch,'decoded':decoded,'text_bytes':entry['size'],'instruction_rows':len(ins),'opcode_counts':dict(sorted(opcounts.items())),'constant_false_opcode_counts':dict(false),'family_counts':{k:len(v)for k,v in families.items()},'anchors':anchors,'decoded_unknown_annotations':dict(unknown),'resource_queries':resources,'sass_file':str(file),'sass_public_name':label+'.sass','sass_sha256':sha(file),'status_file':str(file.parent/'status.json'),'status_sha256':sha(file.parent/'status.json'),'text_section_range_complete_in_listing':True})
  copy.extend([{'source':str(file),'target':label+'.sass','sha256':sha(file)},{'source':str(file.parent/'status.json'),'target':label+'.status.json','sha256':sha(file.parent/'status.json')}])
for leg in('mapal','cublas','cutlass'):
 for name in(leg+'-admission.json',):copy.append({'source':str(R/name),'target':name,'sha256':sha(R/name)})
copy.append({'source':str(R/'summary.json'),'target':'capture-summary.json','sha256':sha(R/'summary.json')})
copy.append({'source':str(R/'mapal-profile/capture/input.ptx'),'target':'mapal-loaded.ptx','sha256':sha(R/'mapal-profile/capture/input.ptx')})
for name in('gpu-version','decoder-version','nvcc-version'):copy.append({'source':str(R/name/'stdout'),'target':name+'.txt','sha256':sha(R/name/'stdout')})
copy.append({'source':str(REVIEW),'target':'independent-admission.json','sha256':sha(REVIEW)})
report={'schema':'admitted-selected-sass-comparison-v1','pass':True,'timing_admissible':False,'dynamic_metrics_measured':False,'case':summary['case'],'harness_sha256':summary['harness_sha256'],'independent_admission':{'path':str(REVIEW),'sha256':sha(REVIEW),'pass':True},'host':(R/'gpu-version/stdout').read_text().strip(),'driver_api_version':13030,'tool_paths':summary['tool_paths'],'tool_versions':{n:(R/n/'stdout').read_text().strip()for n in('decoder-version','nvcc-version')},'selected_kernel_inventory':{leg:len([r for r in allrows if r['leg']==leg])for leg in('mapal','cublas','cutlass')},'rows':allrows,'limits':['Instruction counts are static decoded listing counts including padding/constant-false instructions; not runtime cost, executed instruction/traffic counts or occupancy.','All four listings have unknown decoder annotations. Decoded mnemonic presence remains a narrower observation; full scheduling annotation compatibility is not claimed.','cuBLAS internal function resource attributes and exact K-partition extents/order are unrecorded. GridZ32 and exact reduction symbol are observed independently.','Frozen old Mapal executable/PTX predates source fix50a57e5; these images do not validate or time that fix.']}
(OUT/'comparison.json').write_text(json.dumps(report,indent=2)+'\n');(OUT/'files-to-copy.json').write_text(json.dumps({'public_target':'benches/results/gpu/disassembly/library-kernels/counter-free-capture/rtx-5090/','files':copy,'full_captured_images':'private immutable archive; not duplicated here'},indent=2)+'\n')
print(json.dumps({'pass':True,'rows':len(allrows),'resources_and_counts':[{k:r[k]for k in('label','instruction_rows','family_counts','resource_queries')}for r in allrows]},indent=2))
