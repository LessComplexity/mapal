#!/usr/bin/env python3
"""All corrected cells: independent frozen-LCG integer S and rational minimal ceiling."""
from fractions import Fraction
from pathlib import Path
import hashlib,json,math
from audit_budget import A,C,MASK,SEED,transform,jump,inventory
ROOT=Path('/private/tmp/mapal-s61-competitive-reference/prepared-oracles-v2')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def qvalues(state,count,m=A,b=C):
    out=[]
    for _ in range(count):
        out.append(((state>>40)&0xffffff)-(1<<23));state=(state*m+b)&MASK
    return out
reports=[]
for path in sorted(ROOT.glob('m16-*.json')):
    ref=json.loads(path.read_text());M,N,K=[ref['shape'][k]for k in ['M','N','K']]
    expected=[(i,j)for i in range(M)for j in range(N)]if path.stem.endswith('-full')else inventory(M,N)
    assert len(ref['cells'])==len(expected)
    assert [(c['row'],c['col'])for c in ref['cells']]==expected
    assert ref['budget_rule']=='exact-rational-binary64-ceiling-v1'
    aa={i:qvalues(jump(SEED,i*K+1),K)for i in range(M)}
    bm,bc=transform(N);bb={j:qvalues(jump(SEED,M*K+j+1),K,bm,bc)for j in sorted({j for i,j in expected})}
    txt=path.with_suffix('.txt');lines=txt.read_text().splitlines()
    assert lines[0]==f'COMPETITIVE_ORACLE_V1 {M} {N} {K} {len(expected)}' and len(lines)==len(expected)+1
    cached={};nonexact=0
    for ordinal,(cell,(i,j),line)in enumerate(zip(ref['cells'],expected,lines[1:])):
        assert cell['ordinal']==ordinal
        if(i,j)not in cached:cached[i,j]=sum(abs(x*y)for x,y in zip(aa[i],bb[j]))
        s=cached[i,j];assert int(cell['sum_abs_products_numerator'])==s and int(cell['sum_abs_products_denominator'])==1<<46
        exact=Fraction(2*K*s,((1<<23)-K)*(1<<46))+Fraction(4*K,1<<149)
        assert Fraction(int(cell['budget_exact_numerator']),int(cell['budget_exact_denominator']))==exact
        budget=Fraction.from_float(cell['budget']);assert budget>=exact
        assert Fraction.from_float(math.nextafter(cell['budget'],-math.inf))<exact,'Not minimal upward binary64 ceiling'
        words=line.split();assert [int(x)for x in words[:3]]==[ordinal,i,j]
        assert words[3]==f"{int.from_bytes(bytes.fromhex(cell['operationwise_bits_le']),'little'):08x}" and words[4]==f"{int.from_bytes(bytes.fromhex(cell['fmaf_bits_le']),'little'):08x}"
        assert float.fromhex(words[5])==cell['budget']
        nonexact+=Fraction.from_float(cell['sum_abs_products'])!=Fraction(s,1<<46)
    reports.append({'json':str(path),'json_sha256':sha(path),'text':str(txt),'text_sha256':sha(txt),'shape':ref['shape'],'cells_verified':len(expected),'all_exact_integer_S_match':True,'all_exact_fraction_bounds_match':True,'all_minimal_outward_binary64_budgets_match':True,'all_text_bitwords_and_budgets_match_json':True,'S_nonexact_in_binary64':nonexact})
assert len(reports)==4 and sum(x['cells_verified']for x in reports)==6656
report={'pass':True,'scope':'All6656 frozen cells independently verified for exact S, rational bound and minimal upper binary64/text serialization; FP32/fmaf reference folds not rerun','algorithm':'Independent modular-LCG jumps, exact q integers and Python integer dot absolute sums; Fraction rational comparison','script_sha256':sha(Path(__file__)),'oracles':reports}
Path(__file__).with_name('corrected-budget-verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
