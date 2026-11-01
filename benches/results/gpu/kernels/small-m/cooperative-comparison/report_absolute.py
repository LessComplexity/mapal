"""Add review tables to the unchanged sealed-packet verifier output."""
import argparse, csv, json
from pathlib import Path

def write(out):
    a=json.loads((out/'analysis.json').read_text())
    columns=['id','family','M','N','K','face','ptx','admitted','old_mapal_ms_A_B','candidate_mapal_ms_A_B','candidate_cublas_ms_A_B','candidate_cutlass_ms_A_B','old_candidate_envelope','cublas_candidate_ratio_A_B','cutlass_candidate_ratio_A_B','noise_factor','verdict']
    records=[]
    for row in a['rows']:
        same=row['variant_code']['old']['ptx_sha256']==row['variant_code']['candidate']['ptx_sha256']
        record={k:row[k]for k in ['id','family','M','N','K','face','admitted','verdict']}
        record.update(ptx='unchanged control'if same else'changed',old_mapal_ms_A_B=row['old_process_medians_ms'],candidate_mapal_ms_A_B=row['candidate_process_medians_ms'])
        if row['admitted']:
            p=row['processes'];record.update(candidate_cublas_ms_A_B=[p[arm]['times']['cublas']['median_ms']for arm in ['candidate','candidate_null']],candidate_cutlass_ms_A_B=[p[arm]['times']['cutlass']['median_ms']for arm in ['candidate','candidate_null']],old_candidate_envelope=row['old_over_candidate_cross_process_interval'],cublas_candidate_ratio_A_B=row['candidate_library_over_mapal_pairs']['cublas'],cutlass_candidate_ratio_A_B=row['candidate_library_over_mapal_pairs']['cutlass'],noise_factor=row['observed_noise_factor'])
        records.append(record)
    with(out/'absolute-cells.csv').open('w',newline='')as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(records)
    def fmt(x):
        if isinstance(x,list):return '/'.join('—'if v is None else f'{v:.6g}'for v in x)
        return str(x)
    labels=['Family / M×N×K / face','PTX','Old ms A/B','Candidate ms A/B','cuBLAS ms A/B','CUTLASS ms A/B','cuBLAS/candidate A/B','CUTLASS/candidate A/B','Verdict']
    keys=['old_mapal_ms_A_B','candidate_mapal_ms_A_B','candidate_cublas_ms_A_B','candidate_cutlass_ms_A_B','cublas_candidate_ratio_A_B','cutlass_candidate_ratio_A_B','verdict']
    lines=['Same-process library comparisons retain the requested numerical face; they do not prove identical library arithmetic. Exact is the ordered separate multiply/add gate; Contract is the ordered fmaf gate and permits contraction. Requested-TF32 uses the frozen 4× cuBLAS diagnostic gate, not an absolute accuracy guarantee. A/B are separate ABBA processes: old A, candidate A, candidate B (identical-PTX null), old B (identical-PTX null). No cross-face comparison is made. Nonadmitted cells retain quarantine; missing times are never zero.', '', 'Ratios are library milliseconds divided by candidate milliseconds: greater than one favors the candidate. Old/candidate ratios greater than one favor the candidate. A changed-PTX row is not a gain unless the unchanged verifier verdict exceeds the observed null/control envelope; unchanged-PTX differences are controls.','','| '+' | '.join(labels)+' |','|'+'|'.join(['---']*len(labels))+'|']
    for r in records:
        label=f"{r['family']} / {r['M']}×{r['N']}×{r['K']} / {r['face']}"
        lines.append('| '+' | '.join([label,r['ptx'],*[fmt(r.get(k,'—'))for k in keys]])+' |')
    lines+=['','Numerical admission, raw event medians, null/control noise, resources, hashes and rejected records remain in analysis.json and REPORT.md. This table adds no acceptance rule.']
    (out/'ABSOLUTE.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);write(p.parse_args().out)
