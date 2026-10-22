from pathlib import Path
import hashlib,json
R=Path('/private/tmp/mapal-s56-gpu/ptx-sass-gap');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=R/'completion-before-correction.json'
if not old.exists():old.write_bytes((R/'completion.json').read_bytes())
prior=json.loads(old.read_text());review=R/'independent-review.md';report=R/'REPORT.md';text=report.read_text();d=json.loads((R/'analysis.json').read_text());manifest=json.loads((R/'artifact-manifest.json').read_text())
assert 'three-leg sampled value gate' not in text and 'external-library numeric gates' not in text
assert '| Requested face | Actual Mapal route |' in text
assert '| 16×18432×7168 | contract | fp32_fma | 2.0644 | 0.3400 | 0.5530 |' in text
assert '| 16×18432×7168 | tf32 | fp32_fma | 2.0716 | 0.3338 | 0.3359 |' in text
assert 'max_sample(abs(got - f64_ref) / max(abs(f64_ref), 0.001))' in text
assert 'no independent fixed absolute or relative accuracy bound for TF32' in text
assert 'Reduced::Tf32' in text and 'candidate / NOT decided' in text
values={}
for case,expected in [('deepseek_expert_m16_n2048_k7168__tf32',[3.302,3.302,3.248]),('square16k_m16384_n16384_k16384__tf32',[.1043,.1059,.1059])]:
 row=next(x for x in d['large_rows'] if x['case']==case);actual=[float(row['parsed']['values'][leg]['maxrel_vs_f64']) for leg in ['mapal','cublas','cutlass']];assert actual==expected;values[case]=dict(zip(['mapal','cublas','cutlass'],actual))
assert sha(R/'comparison.csv')==prior['outputs']['comparison.csv']['sha256']
assert sha(R/'analysis.json')==prior['outputs']['analysis.json']['sha256']
manifest[str(review)]={'bytes':review.stat().st_size,'sha256':sha(review)}
(R/'artifact-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
assert all(sha(p)==v['sha256'] for p,v in manifest.items())
correction={'pass':True,'original_report_sha256':prior['outputs']['REPORT.md']['sha256'],'corrected_report_sha256':sha(report),'independent_review_original':{'path':str(review),'sha256':sha(review),'changed':False},'changes':['Requested precision face exposed separately from actual emitted arithmetic route','Only Mapal outputs have numerical acceptance gates; library errors are diagnostics','Exact clamped diagnostic and large admitted M16/16K TF32 diagnostics exposed; no fixed absolute bound claimed','Current Reduced::Tf32 source explicitly specifies cvt.rna; future nearest-even ADR remains undecided'],'diagnostics_order':['mapal','cublas','cutlass'],'bound_diagnostics':values,'timing_and_instruction_analysis_unchanged':True,'source_head':'b4152c5f4d9ad802d0388c1c64b628091d494e6a','no_code_build_or_gpu_actions':True}
(R/'correction-receipt.json').write_text(json.dumps(correction,indent=2)+'\n')
names=['REPORT.md','analysis.json','comparison.csv','artifact-manifest.json','verification.json','analyze.py','verify.py','write_report.py','refresh_receipts.py','initial-receipt.json','correction-receipt.json']
outputs={name:{'bytes':(R/name).stat().st_size,'sha256':sha(R/name)} for name in names}
(R/'completion.json').write_text(json.dumps({'inspected_artifacts':len(manifest),'all_hashes_rechecked':True,'correction_receipt':'correction-receipt.json','outputs':outputs},indent=2)+'\n')
print('pass;',len(manifest),'input hashes;',len(outputs),'output hashes; report',sha(report),'review',sha(review))
