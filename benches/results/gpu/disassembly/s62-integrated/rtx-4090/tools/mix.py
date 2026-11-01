#!/usr/bin/env python3
"""s62 disasm: static instruction mix of one kernel's SASS (cuobjdump -sass section) or
PTX entry. Whole-function counts plus the hot loop (the backward branch whose body holds
the most FMAs): FMA, global loads by width, async copies, shared loads/stores, barriers,
shuffles, branches/predicates, integer/address ops, local (spill) traffic; FMAs per global
load and per loaded byte; accumulators (distinct d of `FFMA d, a, b, d`) and k-steps per
loop trip (loop FMAs / accumulators). Static counts only: no dynamic claim.
usage: mix.py <file.sass|file.ptx> [function-substring]  -> one JSON object per function"""
import json, re, sys
from collections import Counter

SASS_LINE = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)\s*([^;]*);")

def classify_sass(op, args):
    b = op.split(".")[0]
    if b in ("FFMA", "FFMA2"): return "fma"
    if b == "FMUL": return "fmul"
    if b == "FADD": return "fadd"
    if b in ("LDG", "LD"):
        w = 128 if ".128" in op else 64 if ".64" in op else 32
        return f"ldg{w}"
    if b in ("STG", "ST"): return "stg"
    if b == "LDGSTS":
        w = 128 if ".128" in op else 64 if ".64" in op else 32
        return f"ldgsts{w}"
    if b in ("LDS", "LDSM"):
        w = 128 if ".128" in op else 64 if ".64" in op else 32
        return f"lds{w}"
    if b == "STS": return "sts"
    if b in ("BAR", "DEPBAR", "LDGDEPBAR", "ARRIVES", "SYNCS"): return "bar"
    if b == "SHFL": return "shfl"
    if b in ("BRA", "BRX", "JMP", "EXIT", "RET", "BSSY", "BSYNC", "WARPSYNC", "CALL"): return "branch"
    if b in ("ISETP", "FSETP", "PLOP3", "P2R", "R2P", "PSETP"): return "pred"
    if b in ("IMAD", "IADD3", "LEA", "SHF", "LOP3", "IABS", "IMNMX", "SEL", "SGXT", "BMSK", "IMUL", "LEA.HI", "ISCADD", "FLO", "POPC", "PRMT", "IDP"): return "int"
    if b in ("LDL", "STL"): return "local"
    if b in ("MOV", "S2R", "S2UR", "CS2R", "ULDC", "UMOV", "UIADD3", "ULEA", "UIMAD", "USHF", "ULOP3", "R2UR", "UISETP", "USEL", "UPLOP3", "LDC", "MOV32I"): return "mov"
    if b in ("NOP",): return "nop"
    if b in ("HMMA", "IMMA", "DMMA", "HGMMA", "UTCMMA", "QMMA", "OMMA"): return "mma"
    if b in ("FMNMX", "FSEL", "FCHK", "MUFU", "F2F", "I2F", "F2I", "FRND"): return "fother"
    return "other:" + b

def sass_functions(text):
    parts = re.split(r"\n(?=\s*Function : )", text)
    for p in parts:
        m = re.match(r"\s*Function : (\S+)", p)
        if m: yield m.group(1), p

def analyse_sass(body):
    ins = []
    for line in body.splitlines():
        m = SASS_LINE.search(line)
        if m:
            ins.append((int(m.group(1), 16), m.group(2) or "", m.group(3), m.group(4).strip()))
    tot = Counter(classify_sass(o, a) for _, _, o, a in ins)
    loops = []
    for i, (addr, pred, op, args) in enumerate(ins):
        if op.startswith("BRA"):
            t = re.search(r"0x([0-9a-f]+)", args)
            if t and int(t.group(1), 16) <= addr:
                tgt = int(t.group(1), 16)
                body_ins = [x for x in ins if tgt <= x[0] <= addr]
                loops.append((tgt, addr, body_ins))
    hot = None
    if loops:
        fp = lambda L: sum(1 for x in L[2] if classify_sass(x[2], x[3]) in ("fma", "fmul", "fadd"))
        tgt, end, bi = max(loops, key=lambda L: (fp(L), -len(L[2])))
        c = Counter(classify_sass(o, a) for _, _, o, a in bi)
        accs = set()
        for _, _, o, a in bi:
            r = [x.strip() for x in a.split(",")]
            if o.split(".")[0] == "FFMA" and len(r) == 4 and r[0] == r[3]:
                accs.add(r[0])
            if o.split(".")[0] == "FADD" and len(r) == 3 and r[0] in (r[1], r[2]):
                accs.add(r[0])
        hot = summarize(c)
        hot.update(span=f"0x{tgt:x}-0x{end:x}", instructions=len(bi), accumulators=len(accs),
                   ksteps_per_trip=round((c["fma"] or c["fadd"]) / len(accs), 2) if accs else None)
    out = summarize(tot); out["instructions"] = len(ins)
    return {"whole": out, "hot_loop": hot}

def summarize(c):
    ldg = {w: c.get(f"ldg{w}", 0) for w in (32, 64, 128)}
    cpa = {w: c.get(f"ldgsts{w}", 0) for w in (32, 64, 128)}
    lds = {w: c.get(f"lds{w}", 0) for w in (32, 64, 128)}
    gbytes = sum(n * w // 8 for w, n in ldg.items()) + sum(n * w // 8 for w, n in cpa.items())
    gloads = sum(ldg.values()) + sum(cpa.values())
    fma = c.get("fma", 0) + c.get("fmul", 0) + c.get("fadd", 0)
    s = {"fma": c.get("fma", 0), "fmul": c.get("fmul", 0), "fadd": c.get("fadd", 0),
         "ldg": sum(ldg.values()), "ldg_by_width": ldg, "ldg_bytes": gbytes, "ldgsts": sum(cpa.values()), "ldgsts_by_width": cpa,
         "lds": sum(lds.values()), "lds_by_width": lds, "sts": c.get("sts", 0), "stg": c.get("stg", 0),
         "bar": c.get("bar", 0), "shfl": c.get("shfl", 0), "branch": c.get("branch", 0), "pred": c.get("pred", 0),
         "int": c.get("int", 0), "mov": c.get("mov", 0), "local": c.get("local", 0), "mma": c.get("mma", 0)}
    s["fp_per_ldg"] = round(fma / gloads, 2) if gloads else None
    s["fp_per_ldg_byte"] = round(fma / gbytes, 3) if gbytes else None
    s["other"] = {k: v for k, v in c.items() if k.startswith("other:")}
    return s

def analyse_ptx(text, want):
    res = {}
    for m in re.finditer(r"\.entry (\w+)\((.*?)\n\}", text, re.S):
        name, body = m.group(1), m.group(2)
        if want and want not in name: continue
        c = Counter()
        for line in body.splitlines():
            t = line.strip().split()
            if not t or t[0].startswith(("//", ".", "$", "{", "}")): continue
            op = t[0] if not t[0].startswith("@") else t[1]
            if op.startswith("fma.rn.f32"): c["fma"] += 1
            elif op.startswith(("mul.rn.f32", "mul.f32")): c["fmul"] += 1
            elif op.startswith(("add.rn.f32", "add.f32")): c["fadd"] += 1
            elif op.startswith("ld.global"): c["ldg" + ("128" if ".v4" in op else "64" if ".v2" in op else "32")] += 1
            elif op.startswith("st.global"): c["stg"] += 1
            elif op.startswith("cp.async.ca") or op.startswith("cp.async.cg"):
                sz = line.split(",")[-1].strip().rstrip(";")
                c["ldgsts" + ("128" if sz == "16" else "64" if sz == "8" else "32")] += 1
            elif op.startswith("cp.async"): c["bar"] += 1
            elif op.startswith("ld.shared"): c["lds" + ("128" if ".v4" in op else "64" if ".v2" in op else "32")] += 1
            elif op.startswith("st.shared"): c["sts"] += 1
            elif op.startswith(("bar.", "barrier")): c["bar"] += 1
            elif op.startswith("shfl"): c["shfl"] += 1
            elif op.startswith("bra"): c["branch"] += 1
            elif op.startswith("setp"): c["pred"] += 1
            else: c["int"] += 1
        hdr = re.search(r"\.maxntid ([\d, ]+)", text[m.start():m.end() + 200])
        s = summarize(c); s["maxntid"] = hdr.group(1) if hdr else None
        res[name] = s
    return res

if __name__ == "__main__":
    path = sys.argv[1]; want = sys.argv[2] if len(sys.argv) > 2 else ""
    text = open(path, errors="replace").read()
    if "Function :" in text:
        out = {n: analyse_sass(b) for n, b in sass_functions(text) if want in n}
    else:
        out = analyse_ptx(text, want)
    print(json.dumps(out, indent=1))
