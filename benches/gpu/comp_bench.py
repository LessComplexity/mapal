#!/usr/bin/env python3
"""competitors: the library and hand-kernel legs beside torch_bench.py's PyTorch modes, on the same workloads (the same
operand laws and layer, `torch_bench.workload` / `llm_workload`), gated and timed the same way (`torch_bench.gate`,
`timed`: 5 warm-ups, min / median / max over `reps` CUDA-event-bracketed calls, each after the 256 MB L2 flush).

legs (family, precision):
  triton / triton-tf32 / triton-bf16   hand-written Triton: the v3.6.0 tutorials' matmul (autotuned; this suite's
                                       epilogues in its fused-activation slot), softmax and fused attention (forward,
                                       GQA) — triton_kernels.py. GEMM / MLP chains / softmax rows / attention rows /
                                       the LLM prefill's attention.
  liger / liger-tf32 / liger-bf16      Liger kernels: RMSNorm, RoPE, SwiGLU in the LLM layer; cuBLAS projections, SDPA
                                       (f32: the memory-efficient kernel; bf16: flash).
  fa2-bf16                             FlashAttention-2 (flash-attn): flash_attn_func (prefill, causal, GQA),
                                       flash_attn_with_kvcache (decode: appends the step's K / V and attends, split-KV).
  fa3-bf16                             FlashAttention-3 (flash_attn_interface, the hopper/ build): the same two calls.
  flashinfer-bf16                      FlashInfer: BatchDecodeWithPagedKVCacheWrapper (decode; each sequence's cache one
                                       page, planned per step) and single_prefill_with_kv_cache (prefill, causal).
  trtllm-bf16                          TensorRT-LLM: reported unavailable when `tensorrt_llm` does not import (see
                                       benches/results/evidence/local-rtx-5080/competitors.md for why it is not here).
The LLM legs swap kernels into torch_bench.llm_workload (`ops=`): everything else in the layer is the torch leg's.
FlashAttention / FlashInfer have no f32 kernels: they are bf16 legs only (the `_bf16w` rows' fair legs, the f32 rows'
bf16 reference). Values: Mapal's printed values (vals/<stem>.txt) as torch_bench gates them, with its tolerance per
precision (f32 1e-3; model-scale tf32 1e-2; bf16 5e-2 against Mapal's f32 values); a GEMM cell (`mm_MxNxK`, no
torch stem) against the f64 product of its operands.

usage: comp_bench.py <reps> <vals dir> <stem>...   (COMP_LEGS=<leg,...>: only those legs)
prints `comp versions ...`, `comp leg=<leg> unavailable reason=...` per missing package, then per (stem, leg)
`comp prog=<stem> leg=<leg> family=<family> prec=<f32|tf32|bf16> n= min_ms= med_ms= max_ms= within= maxdev= tol=
setup_s= peak_mb= values=` (or `... FAILED <error>`).
"""
import importlib
import math
import os
import re
import sys
import time

import torch

import torch_bench as tb

dev = tb.dev
LEGS = {
    'triton': ('triton', 'f32'), 'triton-tf32': ('triton', 'tf32'), 'triton-bf16': ('triton', 'bf16'),
    'liger': ('liger', 'f32'), 'liger-tf32': ('liger', 'tf32'), 'liger-bf16': ('liger', 'bf16'),
    'fa2-bf16': ('fa2', 'bf16'), 'fa3-bf16': ('fa3', 'bf16'), 'flashinfer-bf16': ('flashinfer', 'bf16'),
    'trtllm-bf16': ('trtllm', 'bf16'),
}
MODULES = {'triton': 'triton_kernels', 'liger': 'liger_kernel', 'fa2': 'flash_attn', 'fa3': 'flash_attn_interface',
           'flashinfer': 'flashinfer', 'trtllm': 'tensorrt_llm'}
MOD, MISSING = {}, {}


def load(family):
    if family in MOD or family in MISSING:
        return MOD.get(family)
    try:
        MOD[family] = importlib.import_module(MODULES[family])
    except Exception as ex:  # noqa: BLE001 — a missing package is a reported leg, not the end of the run
        MISSING[family] = f'{type(ex).__name__}: {str(ex)[:160]}'
    return MOD.get(family)


def legs_for(stem):
    """The legs that apply to a stem at its precision (+ the bf16 reference legs of an f32 model-scale row)."""
    tf = stem.endswith('_tf32')
    base = stem.removesuffix('_tf32').removesuffix('_bf16w')
    w16 = stem.endswith('_bf16w')
    lib16 = ['liger-bf16', 'fa2-bf16', 'fa3-bf16', 'flashinfer-bf16', 'trtllm-bf16']
    if re.fullmatch(r'p[tx]_llm_(dec|blkdec)_b\d+_l\d+', base):
        return lib16 if w16 else ['liger-tf32'] if tf else ['liger'] + lib16
    if re.fullmatch(r'p[tx]_llm_(pre|blkpre)_t\d+', base):
        pre16 = ['triton-bf16'] + lib16
        return pre16 if w16 else ['liger-tf32', 'triton-tf32'] if tf else ['liger', 'triton'] + pre16
    if base.startswith('pt_pj_'):
        return ['triton-tf32'] if tf else ['triton', 'triton-bf16']
    if re.fullmatch(r'mm_\d+x\d+x\d+', base):
        return ['triton']
    if re.fullmatch(r'pt_(epi|mlp|mlp2|smlp|softmax_rows|lsm|attn)_[\d_]+', base):
        return ['triton-tf32'] if tf else ['triton']
    return []


def tables_hf(st):
    """HF's duplicated RoPE tables (cos / sin of width dh), built once from the layer's own half-width tables."""
    def tables(w):
        st['cos'] = torch.cat((w['cs'], w['cs']), -1)
        st['sin'] = torch.cat((w['sn'], w['sn']), -1)
    return tables


def liger_ops():
    from liger_kernel.ops.rms_norm import LigerRMSNormFunction
    from liger_kernel.ops.rope import LigerRopeFunction
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction
    st = {}

    def rope_qk(q, k, pv):  # Liger RoPE takes [batch, heads, seq, dh] and cos / sin [1, seq, dh]
        if pv is None:  # prefill: q [T, H, dh]
            T = q.shape[0]
            qq, kk = LigerRopeFunction.apply(q.unsqueeze(0).transpose(1, 2), k.unsqueeze(0).transpose(1, 2),
                                             st['cos'][:T].unsqueeze(0), st['sin'][:T].unsqueeze(0))
            return qq.transpose(1, 2)[0], kk.transpose(1, 2)[0]
        c, s = st['cos'].index_select(0, pv).unsqueeze(0), st['sin'].index_select(0, pv).unsqueeze(0)
        qq, kk = LigerRopeFunction.apply(q.unsqueeze(2), k.unsqueeze(2), c, s)  # decode: q [B, H, dh], one position
        return qq[:, :, 0], kk[:, :, 0]
    return dict(tables=tables_hf(st), rms=lambda x, g: LigerRMSNormFunction.apply(x, g, 1e-5), rope_qk=rope_qk,
                silu_mul=lambda a, b: LigerSiLUMulFunction.apply(a, b))


def flash_ops(mod):
    """FlashAttention-2 (flash_attn) / -3 (flash_attn_interface): the same two entry points."""
    def first(r):
        return r[0] if isinstance(r, tuple) else r

    def attn_dec(q, k, v, kc, vc, p, pv, ti):  # appends this step's K / V at p (cache_seqlens), attends rows <= p
        B = q.shape[0]
        seq = p.to(torch.int32).expand(B).contiguous()
        return first(mod.flash_attn_with_kvcache(q.unsqueeze(1), kc, vc, k=k.unsqueeze(1), v=v.unsqueeze(1),
                                                 cache_seqlens=seq, causal=True))[:, 0]

    def attn_pre(q, k, v):
        return first(mod.flash_attn_func(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), causal=True))[0]
    return dict(attn_dec=attn_dec, attn_pre=attn_pre)


def flashinfer_ops(fi, B, L, S, H, Hk, dh):
    ws = torch.empty(128 << 20, dtype=torch.uint8, device=dev)
    # FlashInfer's own guidance: the tensor-core decode kernel from a GQA group of 4 query heads per KV head.
    wrapper = fi.BatchDecodeWithPagedKVCacheWrapper(ws, 'NHD', use_tensor_cores=H // Hk >= 4)
    indptr = torch.arange(B + 1, dtype=torch.int32)  # host tensors (as a serving loop plans): one page per sequence
    indices = torch.arange(B, dtype=torch.int32)

    def attn_dec(q, k, v, kc, vc, p, pv, ti):  # kc [B, L, Hk, dh] = [pages, page size, Hk, dh] (NHD)
        kc.index_copy_(1, pv, k.unsqueeze(1))
        vc.index_copy_(1, pv, v.unsqueeze(1))
        wrapper.plan(indptr, indices, torch.full((B,), ti + L - S + 1, dtype=torch.int32), H, Hk, dh, L,
                     q_data_type=q.dtype, kv_data_type=kc.dtype)
        return wrapper.run(q, (kc, vc))

    def attn_pre(q, k, v):
        return fi.single_prefill_with_kv_cache(q, k, v.contiguous(), causal=True)
    return dict(attn_dec=attn_dec, attn_pre=attn_pre)


def triton_pre(prec):
    tk = MOD['triton']

    def attn_pre(q, k, v):  # the tutorial's [Z, H, N, D] contiguous layout
        T, H, dh = q.shape
        z = [t.transpose(0, 1).contiguous().unsqueeze(0) for t in (q, k, v)]
        return tk.attention(*z, True, dh ** -0.5, prec)[0].transpose(0, 1)
    return dict(attn_pre=attn_pre)


def llm_leg(stem, leg, fam, prec):
    base = stem.removesuffix('_tf32')
    P = tb.LLM_PX if stem.startswith('px_') else tb.LLM
    H, Hk, dh = P['H'], P['Hk'], P['dh']
    # Liger's layer attends through SDPA: at f32 the memory-efficient kernel (K / V repeated per query head; the math
    # fallback that GQA alone gets at f32 needs H·T² scores — 8 GiB at T = 8192), at bf16 torch's pick (flash).
    mode = {'f32': 'sdpa-efficient' if fam == 'liger' else 'eager',
            'tf32': 'sdpa-efficient-tf32' if fam == 'liger' else 'eager-tf32', 'bf16': 'bf16'}[prec]
    if fam == 'liger':
        ops = liger_ops()
    elif fam in ('fa2', 'fa3'):
        ops = flash_ops(MOD[fam])
    elif fam == 'flashinfer':
        m = re.search(r'_b(\d+)_l(\d+)', base)
        B, L = (int(m[1]), int(m[2])) if m else (1, 1)
        ops = flashinfer_ops(MOD[fam], B, L, 3 if stem.startswith('px_') else tb.LLM_STEPS, H, Hk, dh)
    elif fam == 'triton':
        ops = triton_pre({'f32': 'ieee', 'tf32': 'tf32', 'bf16': 'ieee'}[prec])
    else:
        raise LookupError(f'no llm ops for {leg}')
    fn, args = tb.llm_workload(base.replace('_pv', ''), mode, ops)
    if fn.__name__ == 'run':
        return fn, (args[1],)
    return fn, args


def gemm_leg(stem, prec):
    """The GEMM / MLP / softmax / attention rows on the Triton tutorial kernels (operands from torch_bench.workload)."""
    tk = MOD['triton']
    tp = 'tf32' if prec == 'tf32' else 'ieee'
    base = stem.removesuffix('_tf32')
    m = re.fullmatch(r'mm_(\d+)x(\d+)x(\d+)', base)
    if m:  # a tile_bench cell: the suite's operand laws, gated against the f64 product
        M, N, K = map(int, m.groups())
        a, b = tb.op('a', M * K).view(M, K), tb.op('b', K * N).view(K, N)
        return (lambda a, b: tb.corners(tk.matmul(a, b, tp))), (a, b)
    _, args = tb.workload(base, 'bf16' if prec == 'bf16' else 'eager')
    if base.startswith('pt_pj_') or base.startswith('pt_mm_'):
        return (lambda a, b: tb.corners(tk.matmul(a, b, tp))), args
    kind = re.match(r'pt_(epi|mlp2|mlp|smlp|softmax_rows|lsm|attn)_', base)[1]
    if kind == 'epi':
        return (lambda a, b: tb.corners(tk.matmul(a, b, tp, 'affine', 0.5, 1.0))), args
    if kind == 'mlp':
        return (lambda a, b, e: tb.corners(tk.matmul(tk.matmul(a, b, tp, 'affine', 0.5, 0.0), e, tp))), args
    if kind == 'mlp2':
        def mlp2(a, b, e, f, g, p, b1, b2):
            return tb.corners(tk.matmul(tk.matmul(a, b, tp, 'bias_relu', bias=b1), e, tp),
                              tk.matmul(tk.matmul(f, g, tp, 'bias_relu', bias=b2), p, tp))
        return mlp2, args
    if kind == 'smlp':
        return (lambda x, w1, w2, b1: tb.corners(tk.matmul(tk.matmul(x, w1, tp, 'bias_relu', bias=b1), w2, tp))), args
    if kind == 'softmax_rows':  # softmax((a·b) / n): the matmul tutorial, then the softmax tutorial
        n = int(base.rsplit('_', 1)[1])
        return (lambda a, b: tb.corners(tk.softmax(tk.matmul(a, b, tp), 1.0 / float(n)))), args
    if kind == 'lsm':
        return (lambda q, w: tb.corners(tk.softmax(tk.matmul(q, w, tp), 0.25))), args
    d = int(base.rsplit('_', 1)[1])  # attn: softmax(q·kᵀ / sd)·v, one head, the fused-attention tutorial
    sd = float(f'{d ** 0.5:.1f}')
    return (lambda q, k, v: tb.corners(tk.attention(q[None, None], k[None, None], v[None, None], False, 1.0 / sd,
                                                    tp)[0, 0])), args


def gate_f64(stem, outs, tol):
    """A GEMM cell: c[0], c[last] against the f64 product's, relative to max(|ref|, rms(c))."""
    M, N, K = map(int, re.fullmatch(r'mm_(\d+)x(\d+)x(\d+)', stem).groups())
    a, b = tb.op('a', M * K).view(M, K).double(), tb.op('b', K * N).view(K, N).double()
    ref = (a @ b).reshape(-1)
    c = outs[0]
    r = math.sqrt(float((c.double() ** 2).mean()))
    dv = max(abs(c[i].item() - ref[i].item()) / max(abs(ref[i].item()), r, 1e-30) for i in (0, -1))
    return ('yes' if dv <= tol else 'no'), dv


def main():
    reps, vals_dir, stems = int(sys.argv[1]), sys.argv[2], sys.argv[3:]
    keep = [m for m in os.environ.get('COMP_LEGS', '').split(',') if m]
    vers = {f: getattr(load(f), '__version__', '?') if load(f) else 'missing' for f in MODULES}
    import triton
    vers['triton'] = triton.__version__ if 'triton' not in MISSING else 'missing'
    cc = '.'.join(map(str, torch.cuda.get_device_capability())) if dev == 'cuda' else '-'
    print(f'comp versions torch={torch.__version__} ' + ' '.join(f'{k}={v}' for k, v in vers.items()) +
          f' device="{torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"}" cc={cc}', flush=True)
    for f, why in MISSING.items():
        print(f'comp family={f} unavailable reason="{why}"', flush=True)
    flush = torch.empty(64 << 20, device=dev) if dev == 'cuda' else None  # 256 MB
    for stem in stems:
        llm = stem.startswith(('pt_llm_', 'px_llm_'))
        scale = llm or stem.startswith('pt_pj_')
        for leg in legs_for(stem):
            fam, prec = LEGS[leg]
            if (keep and leg not in keep) or not load(fam):
                continue
            tf = prec == 'tf32'
            torch.backends.cuda.matmul.allow_tf32 = tf
            torch.backends.cudnn.allow_tf32 = tf
            tol = 5e-2 if prec == 'bf16' else 1e-2 if scale and tf else 1e-3
            fn = args = outs = None
            try:
                if dev == 'cuda':
                    torch.cuda.empty_cache()
                    base_b = torch.cuda.memory_allocated()
                t0 = time.perf_counter()
                fn, args = llm_leg(stem, leg, fam, prec) if llm else gemm_leg(stem, prec)
                if dev == 'cuda':
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                outs = fn(*args)  # the first call: autotuning, JIT, plans
                setup = time.perf_counter() - t0
                if stem.startswith('mm_'):
                    ok, dv = gate_f64(stem, outs, tol)
                else:
                    ok, dv = tb.gate(stem, outs, vals_dir, tol)
                vals = tb.llm_vals(stem.removesuffix('_tf32'), outs) if llm else tb.ends(outs)
                outs = None
                ts = tb.timed(fn, args, reps, flush)
                peak = (torch.cuda.max_memory_allocated() - base_b) / 2 ** 20 if dev == 'cuda' else float('nan')
                print(f'comp prog={stem} leg={leg} family={fam} prec={prec} n={len(ts)} min_ms={ts[0]:.4f} '
                      f'med_ms={ts[len(ts) // 2]:.4f} max_ms={ts[-1]:.4f} within={ok} maxdev={dv:.3e} tol={tol:g} '
                      f'setup_s={setup:.1f} peak_mb={peak:.1f} values={",".join(f"{v:.6g}" for v in vals)}',
                      flush=True)
            except Exception as ex:  # noqa: BLE001 — one leg's failure is a row, not the session
                print(f'comp prog={stem} leg={leg} family={fam} prec={prec} FAILED {type(ex).__name__}: '
                      f'{str(ex)[:200]}', flush=True)
            fn = args = outs = None


if __name__ == '__main__':
    main()
