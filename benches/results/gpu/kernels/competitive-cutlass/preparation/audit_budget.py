#!/usr/bin/env python3
"""Independent exact-integer/rational audit of frozen LCG inputs and FP32 budgets."""
from fractions import Fraction
import hashlib, json, math, pathlib
A=6364136223846793005; C=1442695040888963407; MASK=(1<<64)-1; SEED=0x5eed

def transform(n):
    m,b,cm,cb=1,0,A,C
    while n:
        if n&1:m,b=(m*cm)&MASK,(b*cm+cb)&MASK
        cb=((cm+1)*cb)&MASK;cm=(cm*cm)&MASK;n>>=1
    return m,b

def jump(state,n):
    m,b=transform(n);return (m*state+b)&MASK

def inventory(M,N):
    out=[(0,0),(M-1,N-1),(0,N-1),(M-1,0)];state=0xce11
    for _ in range(252):
        state=(state*A+1)&MASK;out.append(((state>>33)%M,(state>>13)%N))
    return out

def audit(M,N,K):
    bm,bc=transform(N); gamma=Fraction(K,(1<<23)-K);gf=(2*K*(2.0**-24))/(1-2*K*(2.0**-24))
    rows=[]
    for ordinal,(i,j)in enumerate(inventory(M,N)):
        sa=jump(SEED,i*K+1);sb=jump(SEED,M*K+j+1);total=0;products=[]
        for k in range(K):
            aq=((sa>>40)&0xffffff)-(1<<23);bq=((sb>>40)&0xffffff)-(1<<23)
            term=abs(aq*bq);total+=term;products.append(math.ldexp(term,-46))
            sa=(sa*A+C)&MASK;sb=(sb*bm+bc)&MASK
        sf=math.fsum(products);s=Fraction(total,1<<46)
        exact=2*gamma*s+Fraction(4*K,1<<149)
        current=math.nextafter(2*gf*sf+4*K*(2.0**-149),math.inf)
        delta=Fraction.from_float(current)-exact
        converted=float(exact)
        if Fraction.from_float(converted)<exact:converted=math.nextafter(converted,math.inf)
        rows.append({'ordinal':ordinal,'row':i,'col':j,'sum_abs_integer_numerator':total,'sum_abs_denominator':1<<46,'sum_abs_exact_in_binary64':Fraction.from_float(sf)==s,'current_budget_hex':current.hex(),'exact_budget_numerator':exact.numerator,'exact_budget_denominator':exact.denominator,'current_budget_is_outward':delta>=0,'current_minus_exact_over_ulp':float(delta/Fraction.from_float(math.ulp(current))),'exact_outward_budget_hex':converted.hex()})
    failures=[x for x in rows if not x['current_budget_is_outward']]
    return {'shape':{'M':M,'N':N,'K':K},'cells_audited':len(rows),'S_not_exact_in_binary64':sum(not x['sum_abs_exact_in_binary64']for x in rows),'gamma_exact_numerator':gamma.numerator,'gamma_exact_denominator':gamma.denominator,'gamma_binary64_hex':gf.hex(),'gamma_binary64_rounding':('exact'if Fraction.from_float(gf)==gamma else'up'if Fraction.from_float(gf)>gamma else'down'),'budgets_below_exact_bound':len(failures),'first_nonexact_S':next((x for x in rows if not x['sum_abs_exact_in_binary64']),None),'first_underbound':failures[0]if failures else None,'all_cells':rows}

if __name__=='__main__':
    report={'scope':'CPU exact-rational audit of frozen seeded input law; no GPU or oracle reference rerun','input_law':'(LCGtop24-2^23)/2^23, uint64 modular LCG, A then B row-major','cases':[audit(16,2048,7168),audit(16,1024,4096)]}
    p=pathlib.Path(__file__).with_name('budget-audit.json');p.write_text(json.dumps(report,indent=2)+'\n')
    for r in report['cases']:print(json.dumps({k:v for k,v in r.items()if k!='all_cells'},indent=2))
