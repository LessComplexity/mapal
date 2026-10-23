#!/usr/bin/env python3
"""s62 disasm (rtx-4090): tables from one box session.
usage: analyze.py <results-rtx-4090.txt> <trace dir> > tables.md
Times: per leg the median of each process (tile_bench / mlp_bench print them), then the mean
of the two process medians. Null: |null/shipped - 1| for the byte-identical copy, per process.
Hardware usage: bytes = A+B+C (+ 2*S*M*N*4 split workspace when a reduce kernel runs), the
lower bound of DRAM traffic after the harness's 256 MiB L2 flush; FLOPs = 2MNK."""
import json, re, statistics, subprocess, sys
from collections import defaultdict
from pathlib import Path

RES, TR = Path(sys.argv[1]), Path(sys.argv[2])
HERE = Path(__file__).resolve().parent
txt = RES.read_text(errors="replace").splitlines()
# RTX 4090 (AD102, sm_89): 128 SMs x 128 FP32 lanes x 2 FLOP x 2.52 GHz boost; 384-bit GDDR6X 21 Gb/s.
PEAK_FLOPS, PEAK_BW = 128 * 128 * 2 * 2.52e9, 1008e9
FACTS = {}
for l in txt:
    m = re.search(r"facts (sms=.*)", l)
    if m:
        FACTS = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", m.group(1))}
        break

def kv(s):
    return dict(re.findall(r"(\w+)=(\S+)", s))

# ---- kernel cells
cell_t = defaultdict(lambda: defaultdict(dict))   # cell -> leg -> pass -> med
cell_arm = defaultdict(dict)                      # cell -> stem -> arm line kv
cell_val = defaultdict(dict)
for l in txt:
    m = re.match(r"pass=(\d) cell=(\S+) (.*)", l)
    if not m: continue
    p, c, rest = int(m.group(1)), m.group(2).removeprefix("mm_"), m.group(3)
    if rest.startswith("time leg="):
        d = kv(rest); cell_t[c][d["leg"]][p] = float(d["med_ms"])
    elif rest.startswith("arm stem="):
        d = kv(rest); cell_arm[c][d["stem"]] = d
    elif rest.startswith("values leg="):
        cell_val[c][kv(rest)["leg"]] = rest

def t2(c, leg):
    v = cell_t[c].get(leg, {})
    return statistics.mean(v.values()) if v else None

def dims(c):
    return tuple(int(x) for x in c.split("x"))

nulls = {}
for c in cell_t:
    r = []
    for s in (f"mm_{c}", f"mm_{c}_c"):
        for p in (1, 2):
            a, b = cell_t[c].get(s, {}).get(p), cell_t[c].get(s + ".null", {}).get(p)
            if a and b: r.append(abs(b / a - 1))
    nulls[c] = max(r) if r else None
NULL = max(v for v in nulls.values() if v is not None) if nulls else 0.0
libvar = {}
for c in cell_t:
    for leg in ("cublas", "cutlass", "cutlass_tuned"):
        v = cell_t[c].get(leg, {})
        if len(v) == 2: libvar[(c, leg)] = abs(v[1] / v[2] - 1)

def verdict(r, null):
    if r is None: return "—"
    if abs(r - 1) <= null: return f"on-par ({r:.2f})"
    return f"{'better' if r < 1 else 'worse'} ({r:.2f})"

FACES = [("", "Exact"), ("_c", "Contract"), ("_r", "reassoc"), ("_rc", "reassoc+C")]
out = []
P = out.append
P(f"Box null variation (byte-identical PTX copy, per process, max over cells and E/C): **{NULL*100:.1f}%**; "
  f"per cell: " + ", ".join(f"{c} {v*100:.1f}%" for c, v in nulls.items() if v is not None) + ".")
P(f"Library process-to-process variation: max {max(libvar.values())*100:.1f}%." if libvar else "")
P("")
P("### Per-cell comparison (kernel ms, mean of two process medians; ratio = Mapal / other, < 1 = Mapal faster; on-par = within max(box null, that library leg's process-to-process variation))")
P("")
P("| cell | face | Mapal | cuBLAS | stock CUTLASS | tuned CUTLASS | vs cuBLAS | vs stock | vs tuned | Mapal base (4e1102c) |")
P("|---|---|---:|---:|---:|---:|---|---|---|---:|")
for c in cell_t:
    cb, cs, ct = t2(c, "cublas"), t2(c, "cutlass"), t2(c, "cutlass_tuned")
    for suf, name in FACES:
        m = t2(c, f"mm_{c}{suf}")
        if m is None: continue
        b = t2(c, f"mm_{c}{suf}.base")
        f4 = lambda x: f"{x:.4f}" if x else "—"
        rv = lambda x, leg: verdict(m / x, max(NULL, libvar.get((c, leg), 0))) if x else "—"
        P(f"| {c} | {name} | {m:.4f} | {f4(cb)} | {f4(cs)} | {f4(ct)} | {rv(cb, 'cublas')} | {rv(cs, 'cutlass')} | {rv(ct, 'cutlass_tuned')} | {f4(b)} |")
P("")

# ---- observer kernels
obs = defaultdict(list)
for l in txt:
    if l.startswith("obs_kernel name="):
        f = l.split("\t")
        name = f[0].split("name=")[1].split()[0]
        sym = f[0].split()[-1]
        obs[name].append({"symbol": sym, "grid": f[1], "block": f[2], "dyn_smem": int(f[3]), "image": f[5], "res": f[6] if len(f) > 6 else ""})

def res_fields(s):
    return {k: int(v) for k, v in re.findall(r"(REG|STACK|SHARED|LOCAL):(\d+)", s)}

def occupancy(threads, regs, smem):
    F = FACTS or {"regs_per_sm": 65536, "max_threads_per_sm": 1536, "max_blocks_per_sm": 24,
                  "smem_per_sm": 102400, "smem_reserved_per_block": 1024}
    warps = -(-threads // 32)
    rw = -(-max(regs, 1) * 32 // 256) * 256
    by_regs = (F["regs_per_sm"] // rw) // warps if rw else 99
    by_thr = F["max_threads_per_sm"] // (warps * 32)
    by_smem = F["smem_per_sm"] // (-(-(smem + F["smem_reserved_per_block"]) // 128) * 128)
    bps = max(0, min(F["max_blocks_per_sm"], by_regs, by_thr, by_smem))
    lim = min(("regs", by_regs), ("threads", by_thr), ("smem", by_smem), ("blocks", F["max_blocks_per_sm"]), key=lambda x: x[1])[0]
    return bps, bps * warps, lim

def nblocks(g):
    a = [int(x) for x in g.split(",")]
    return a[0] * a[1] * a[2]

def nthreads(b):
    a = [int(x) for x in b.split(",")]
    return a[0] * a[1] * a[2]

def mixes(capdir):
    rows = {}
    for s in sorted((TR / "obs" / capdir / "sass").glob("*.sass")):
        j = json.loads(subprocess.run([sys.executable, str(HERE / "mix.py"), str(s)], capture_output=True, text=True).stdout)
        rows[int(s.stem)] = next(iter(j.values())) if j else None
    return rows

LEGS = [("cublas", "cuBLAS"), ("cutlass", "stock CUTLASS"), ("cutlass_tuned", "tuned CUTLASS"),
        ("mapal", "Mapal E"), ("mapal_c", "Mapal C"), ("mapal_r", "Mapal reassoc"), ("mapal_rc", "Mapal reassoc+C")]
P("### Executed kernels and resources (observer: launched in the profiler region, joined to the saved image; res-usage of that image)")
P("")
P("| cell | implementation | kernel | grid (blocks) | block | regs | local | static smem | dyn smem | blocks/SM (limit) | warps/SM of 48 | waves (128 SMs) |")
P("|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|")
mixtab = []
for c in cell_t:
    for key, label in LEGS:
        cap = f"{c}.{key}" if key.startswith("cu") else f"{c}.mapal{key[5:]}"
        for i, k in enumerate(obs.get(cap, [])):
            rf = res_fields(k["res"])
            thr, nb = nthreads(k["block"]), nblocks(k["grid"])
            smem = rf.get("SHARED", 0) + k["dyn_smem"]
            bps, wps, lim = occupancy(thr, rf.get("REG", 0), smem)
            waves = nb / (bps * 128) if bps else float("inf")
            short = k["symbol"] if len(k["symbol"]) < 60 else k["symbol"][:57] + "..."
            P(f"| {c} | {label} | `{short}` | {k['grid']} ({nb}) | {thr} | {rf.get('REG','?')} | {rf.get('LOCAL','?')} | {rf.get('SHARED','?')} | {k['dyn_smem']} | {bps} ({lim}) | {wps} | {waves:.2f} |")
        mx = mixes(cap) if (TR / "obs" / cap / "sass").exists() else {}
        for i, j in mx.items():
            if j and i < len(obs.get(cap, [])):
                k = obs[cap][i]
                M, N, K = dims(c)
                S = int(k["grid"].split(",")[2]) if key == "cublas" else int(cell_arm[c].get(f"mm_{c}{key[5:]}", {}).get("S", 1)) if key.startswith("mapal") else 1
                useful = M * N * S / (nblocks(k["grid"]) * nthreads(k["block"]))
                mixtab.append((c, label, k["symbol"], j, useful, key.startswith("mapal")))
P("")
P("### SASS instruction mix (static; executed images; hot loop = backward branch with the most FFMA)")
P("")
P("| cell | implementation | kernel | instr | FFMA (loop) | FMUL+FADD (loop) | LDG 32/64/128 (loop) | LDGSTS 32/64/128 (loop) | LDS 32/64/128 (loop) | STS (loop) | BAR (loop) | branch+pred (loop) | int+mov (loop) | FP ops/global load (loop) | FP ops/global byte (loop) | outputs/thread (useful) | accum regs | k-steps/trip |")
P("|---|---|---|---:|---|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|")
for c, label, sym, j, useful, is_mapal in mixtab:
    w, h = j["whole"], j["hot_loop"] or {}
    if "reduce" in sym.lower() or (w["fma"] + w["fmul"] + w["fadd"]) < 64:   # split-K reduce / epilogue kernels: no GEMM loop
        h = {}
    g = lambda d, k: d.get(k, "—") if d else "—"
    lw = lambda d, f: "/".join(str(d[f][x]) for x in ("32", "64", "128")) if d else "—"
    short = sym if len(sym) < 40 else sym[:37] + "..."
    macs = (h.get("fma", 0) + (h.get("fmul", 0) + h.get("fadd", 0)) / 2) if h else 0
    acc = useful if is_mapal else (h.get("accumulators") or 0)
    ks = f"{macs / acc:.0f}" if h and acc else "—"
    accs = f"{h.get('accumulators')}" if h and not is_mapal else ("= outputs" if h else "—")
    P(f"| {c} | {label} | `{short}` | {w['instructions']} | {w['fma']} ({g(h,'fma')}) | {w['fmul']+w['fadd']} ({(h.get('fmul',0)+h.get('fadd',0)) if h else '—'}) | "
      f"{lw(w,'ldg_by_width')} ({lw(h,'ldg_by_width')}) | {lw(w,'ldgsts_by_width')} ({lw(h,'ldgsts_by_width')}) | {lw(w,'lds_by_width')} ({lw(h,'lds_by_width')}) | "
      f"{w['sts']} ({g(h,'sts')}) | {w['bar']} ({g(h,'bar')}) | {w['branch']+w['pred']} ({(h['branch']+h['pred']) if h else '—'}) | {w['int']+w['mov']} ({(h['int']+h['mov']) if h else '—'}) | "
      f"{w['fp_per_ldg']} ({g(h,'fp_per_ldg')}) | {w['fp_per_ldg_byte']} ({g(h,'fp_per_ldg_byte')}) | {useful:.1f} | {accs} | {ks} |")
P("")

# ---- hardware usage
P(f"### Hardware usage (peak: {PEAK_BW/1e9:.0f} GB/s DRAM, {PEAK_FLOPS/1e12:.1f} TFLOP/s FP32 at 2.52 GHz boost; ridge {PEAK_FLOPS/PEAK_BW:.0f} FLOP/B)")
P("")
P("| cell | implementation | ms | bytes moved (MB) | GB/s | % DRAM peak | GFLOP/s | % FP32 peak | FLOP/B | roof ms | ms / roof | bound |")
P("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
for c in cell_t:
    M, N, K = dims(c)
    base_bytes, flops = 4 * (M * K + K * N + M * N), 2 * M * N * K
    for leg, label, cap in [("cublas", "cuBLAS", f"{c}.cublas"), ("cutlass", "stock CUTLASS", f"{c}.cutlass"),
                            ("cutlass_tuned", "tuned CUTLASS", f"{c}.cutlass_tuned"),
                            (f"mm_{c}", "Mapal E", f"{c}.mapal"), (f"mm_{c}_c", "Mapal C", f"{c}.mapal_c"),
                            (f"mm_{c}_r", "Mapal reassoc", f"{c}.mapal_r"), (f"mm_{c}_rc", "Mapal reassoc+C", f"{c}.mapal_rc")]:
        t = t2(c, leg)
        if t is None: continue
        ks = obs.get(cap, [])
        S = 1
        if len(ks) > 1:   # a separate reduce kernel: S partial slabs written then read
            S = max(int(x) for x in ks[0]["grid"].split(",")) if leg == "cublas" else int(cell_arm[c].get(leg, {}).get("S", 1))
            if leg == "cublas":
                S = int(ks[0]["grid"].split(",")[2])
        a = cell_arm[c].get(leg)
        if a and int(a.get("S", 1)) > 1: S = int(a["S"])
        by = base_bytes + (2 * S * M * N * 4 if S > 1 else 0)
        roof = max(by / PEAK_BW, flops / PEAK_FLOPS) * 1e3
        bound = "memory" if by / PEAK_BW >= flops / PEAK_FLOPS else "compute"
        eff = roof / t
        cls = bound if eff >= 0.5 else f"{bound} roof, latency/issue-limited"
        if t < 0.02: cls += " (launch-scale)"
        P(f"| {c} | {label} | {t:.4f} | {by/1e6:.1f}{f' (S={S})' if S>1 else ''} | {by/t/1e6:.0f} | {by/t/1e6/PEAK_BW*1e9*100:.0f}% | {flops/t/1e6:.0f} | {flops/t/1e9/PEAK_FLOPS*1e12*100:.1f}% | {flops/by:.1f} | {roof:.4f} | {t/roof:.2f} | {cls} |")
P("")
# ---- residual MLP whole programs
mlp = defaultdict(list)          # (prog, face, arm, ld) -> iter ms
for l in txt:
    if l.startswith("mlp prog="):
        d = kv(l); it = re.search(r"iter ms=([0-9.]+)", l)
        if it: mlp[(d["prog"], d["face"], d["arm"], d["ld"])].append(float(it.group(1)))
mtr = defaultdict(list)          # (prog, face, arm, ld) -> traced kernel ms list
for l in txt:
    if l.startswith("mlp_trace prog="):
        d = kv(l); k = re.search(r"kernels=\[(.*?)\]", l)
        if k: mtr[(d["prog"], d["face"], d["arm"], d["ld"])].append(sum(float(x) for x in re.findall(r"ms=([0-9.]+)", k.group(1))))
lib = defaultdict(lambda: defaultdict(dict)); first = defaultdict(lambda: defaultdict(list))
for l in txt:
    m = re.match(r"pass=(\d) prog=(\S+) (time|first_call) (.*)", l)
    if not m: continue
    d = kv(m.group(4))
    if m.group(3) == "time": lib[m.group(2)][d["leg"]][int(m.group(1))] = float(d["med_ms"])
    else: first[m.group(2)][d["leg"]].append(float(d["wall_ms"]))
med = lambda v: statistics.median(v) if v else None
P("### Residual MLP whole programs (M128; Mapal `iter ms` = host clock around the computation in a fresh process, median of the reps; library steady = cudaEvent GPU ms, mean of two process medians; library first = host clock around the first call + sync in a fresh process)")
P("")
P("| program | face | Mapal on LAZY | Mapal on EAGER | base LAZY | base EAGER | null LAZY | Mapal traced kernel sum | eager cuBLAS steady / first | cuBLASLt fused steady / first | stock CUTLASS steady / first | tuned CUTLASS steady / first |")
P("|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|")
for prog in sorted({k[0] for k in mlp}):
    for face in ("_exact", "_c"):
        g = lambda arm, ld: med(mlp.get((prog, face, arm, ld)))
        f4 = lambda x: f"{x:.4f}" if x else "—"
        ls = lambda leg: (f"{statistics.mean(lib[prog][leg].values()):.4f} / {med(first[prog][leg]):.4f}" if lib[prog].get(leg) else "—")
        P(f"| {prog} | {face} | {f4(g('on','LAZY'))} | {f4(g('on','EAGER'))} | {f4(g('base','LAZY'))} | {f4(g('base','EAGER'))} | {f4(g('null','LAZY'))} | {f4(med(mtr.get((prog, face, 'on', 'LAZY'))))} | {ls('eager')} | {ls('lt')} | {ls('cutlass')} | {ls('cutlass_tuned')} |")
P("")
print("\n".join(out))
