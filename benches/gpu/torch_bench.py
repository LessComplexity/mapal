#!/usr/bin/env python3
"""S64 evidence suite, the PyTorch legs: each workload of gen_tiles.py's timed programs written the way a
PyTorch user would, eager and `torch.compile(mode="max-autotune-no-cudagraphs")`, with the same operand
laws (gen_tiles.py OPS / BIAS / law / decode — change both or neither), pure FP32 (TF32 off).

Timing (chain_bench.cu's convention): 5 warm-ups, then `reps` repetitions, each after a 256 MB L2 flush,
bracketed by CUDA events; min / median / max ms. Values: every output's [0] and [last], in the program's
print order, gated against Mapal's printed values (box_run.sh `vals/<stem>.txt`) as chain_bench does:
|torch − mapal| ≤ 1e-3 · max(|mapal|, rms(torch output)).

usage: torch_bench.py <reps> <vals dir> <stem>...   (TORCH_MODES=<mode,...>: only those legs)
model-scale (`pt_llm_*`, `pt_pj_*`): also `sdpa` and the bf16 reference legs, and per-mode tolerances (`llm_workload`).
prints `torch prog=<stem> mode=eager|compile n= min_ms= med_ms= max_ms= within=yes|no|no-mapal maxdev=
compile_s=` per leg, after one `torch version= cuda= cudnn= device=` line.
"""
import math
import os
import re
import sys
import time

import torch

OPS = {'a': (7, 13, 101, 50, 7.0), 'b': (11, 57, 103, 51, 9.0), 'e': (5, 3, 97, 48, 11.0),
       'f': (3, 1, 89, 44, 13.0), 'g': (13, 7, 107, 53, 15.0), 'p': (17, 29, 109, 54, 17.0)}
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = True  # cuDNN autotunes each conv shape (untimed first call + warm-ups)


def law(length, mul, add, mod, off, div, shift=0):
    """`((t + shift)·mul + add) % mod − off`, widened, ÷ div (gen_tiles.py operand / law)."""
    t = torch.arange(length, device=dev, dtype=torch.int64) + shift
    return ((t * mul + add) % mod - off).to(torch.float32) / div


def op(key, length, shift=0):
    return law(length, *OPS[key], shift=shift)


def bias(n, mod, off, div):
    return ((torch.arange(n, device=dev) % mod) - off).to(torch.float32) / div


def expo(x):
    """exp(x) (gen_tiles.py `expo`: the `exp` builtin, ADR-0042)."""
    return torch.exp(x)


def corners(*outs):
    return [t.reshape(-1) for t in outs]


# model-scale (gen_tiles.py `llm_*`): the Llama-3-8B-like layer written the way a PyTorch user writes it — rsqrt,
# rotate_half RoPE, softmax / SDPA with GQA, F.silu — on the same tensors (gen_tiles.py `llm_ops`: OPS's values,
# `t % mod` first). Modes: `eager` / `compile` (matmul -> masked softmax -> matmul, the program's math; at 32·T²
# scores past 4 GiB the prefill attention loops over the KV-head groups as the program does), `sdpa` (eager,
# F.scaled_dot_product_attention: the efficient backend at f32; a decode step slices the cache to the attended
# positions), `-tf32` twins, and `bf16` / `bf16-compile`: the layer in bfloat16 with SDPA (flash), the real-world
# reference — gated loosely against Mapal's f32 values, reported apart. A decode `run` is S steps (one timed rep).
LLM = dict(D=4096, H=32, Hk=8, dh=128, F=14336)
LLM_PX = dict(D=64, H=4, Hk=2, dh=16, F=96)
LLM_STEPS = 16


def llm_law(length, key, body):
    mul, add, mod, off, _ = OPS[key]
    t = torch.arange(length, device=dev, dtype=torch.int64)
    return body((((t % mod) * mul + add) % mod - off).to(torch.float32))


def llm_div(k):
    return float(f"{29 * k ** 0.5:.1f}")


def llm_tensors(P, rows, positions, cache, ffn, dt):
    D, H, Hk, dh, F = P['D'], P['H'], P['Hk'], P['dh'], P['F']
    NQ, half = (H + 2 * Hk) * dh, dh // 2
    w = dict(x0=llm_law(rows * D, 'a', lambda v: v / 29.0).view(rows, D),
             g1=llm_law(D, 'e', lambda v: 1.0 + v / 400.0),
             wqkv=llm_law(D * NQ, 'b', lambda v: v / llm_div(D)).view(D, NQ),
             wo=llm_law(D * D, 'e', lambda v: v / llm_div(D)).view(D, D),
             cs=llm_law(positions * half, 'f', lambda v: v / 50.0).view(positions, half),
             sn=llm_law(positions * half, 'g', lambda v: v / 50.0).view(positions, half))
    if cache:
        w['kc0'] = llm_law(cache, 'p', lambda v: v / 29.0)
        w['vc0'] = llm_law(cache, 'a', lambda v: v / 29.0)
    if ffn:
        w.update(g2=llm_law(D, 'b', lambda v: 1.0 + v / 400.0),
                 wg=llm_law(D * F, 'f', lambda v: v / llm_div(D)).view(D, F),
                 wu=llm_law(D * F, 'g', lambda v: v / llm_div(D)).view(D, F),
                 wd=llm_law(F * D, 'p', lambda v: v / llm_div(F)).view(F, D))
    return {k: v.to(dt) for k, v in w.items()}


def llm_rms(x, g):
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5).to(x.dtype) * g


def llm_rope(v, c, s):  # v [..., dh]; c, s broadcast to [..., dh / 2]
    h = v.shape[-1] // 2
    v1, v2 = v[..., :h], v[..., h:]
    return torch.cat((v1 * c - v2 * s, v2 * c + v1 * s), -1)


def llm_ffn(w, h1):
    hn = llm_rms(h1, w['g2'])
    return h1 + (torch.nn.functional.silu(hn @ w['wg']) * (hn @ w['wu'])) @ w['wd']


LLM_CELLS = lambda rows, D: [0, (rows - 1) * D + 1, rows * D // 2 + 3, rows * D - 1]  # noqa: E731 — the printed cells


LLM_WEIGHTS = ('wqkv', 'wo', 'wg', 'wu', 'wd')  # gen_tiles.py: `_bf16w` stores these as bf16 (ADR-0043)


def llm_workload(stem, mode):
    """-> (fn, args); fn(*args) returns the layer's output [rows, D]. A decode workload is ('run', step).
    `_bf16w` (half-precision, ADR-0043): the weights rounded once to bf16 — the f32 legs compute on those values
    (the program's own semantics), the bf16 legs are bf16 throughout."""
    w16 = stem.endswith('_bf16w')
    stem = stem.removesuffix('_bf16w')
    P = LLM_PX if stem.startswith('px_') else LLM
    D, H, Hk, dh = P['D'], P['H'], P['Hk'], P['dh']
    G, KD, NQ = H // Hk, Hk * dh, (H + 2 * Hk) * dh
    bf = mode.startswith('bf16')
    dt = torch.bfloat16 if bf else torch.float32
    sdpa = bf or mode.startswith('sdpa')
    F_ = torch.nn.functional
    m = re.fullmatch(r'p[tx]_llm_(dec|blkdec)_b(\d+)_l(\d+)', stem)
    if m:
        ffn, B, L = m.group(1) == 'blkdec', int(m.group(2)), int(m.group(3))
        S = 3 if stem.startswith('px_') else LLM_STEPS
        w = llm_tensors(P, B, L, B * L * KD, ffn, dt)
        if w16:
            w.update({k: v.to(torch.bfloat16).to(dt) for k, v in w.items() if k in LLM_WEIGHTS})
        kc, vc = w['kc0'].view(B, L, Hk, dh), w['vc0'].view(B, L, Hk, dh)
        pos = torch.arange(L, device=dev)

        def step(t, x, ti=None):  # t: a 0-d tensor (one compiled graph for every step); the cache is updated in place
            p = t + (L - S)
            pv = p.view(1)
            qkv = llm_rms(x, w['g1']) @ w['wqkv']
            c, s = w['cs'].index_select(0, pv), w['sn'].index_select(0, pv)
            q = llm_rope(qkv[:, :D].view(B, H, dh), c, s)
            kc.index_copy_(1, pv, llm_rope(qkv[:, D:D + KD].view(B, Hk, dh), c, s).view(B, 1, Hk, dh))
            vc.index_copy_(1, pv, qkv[:, D + KD:].view(B, 1, Hk, dh))
            if sdpa and ti is not None:  # eager SDPA: the attended prefix of the cache (ti: the step, a host int)
                n = ti + L - S + 1
                o = F_.scaled_dot_product_attention(q.view(B, H, 1, dh), kc[:, :n].transpose(1, 2),
                                                    vc[:, :n].transpose(1, 2), enable_gqa=True)
            elif sdpa:  # compiled: the whole cache, masked (static shapes)
                o = F_.scaled_dot_product_attention(q.view(B, H, 1, dh), kc.transpose(1, 2), vc.transpose(1, 2),
                                                    attn_mask=(pos <= p).view(1, 1, 1, L), enable_gqa=True)
            else:
                sc = torch.einsum('bkgd,blkd->bkgl', q.view(B, Hk, G, dh), kc) / dh ** 0.5
                pr = torch.softmax(sc.masked_fill(pos > p, float('-inf')), -1)
                o = torch.einsum('bkgl,blkd->bkgd', pr, vc)
            h1 = x + o.reshape(B, D) @ w['wo']
            return llm_ffn(w, h1) if ffn else h1

        def run(step):
            x = w['x0']
            for t in range(S):
                x = step(torch.tensor(t, device=dev), x, t if sdpa and 'compile' not in mode else None)
            return x
        return run, ('step', step)
    m = re.fullmatch(r'p[tx]_llm_(pre|preg|blkpre)_t(\d+)', stem)
    if m:
        ffn, T = m.group(1) == 'blkpre', int(m.group(2))
        w = llm_tensors(P, T, T, 0, ffn, dt)
        grouped = H * T * T * 4 > (4 << 30) or m.group(1) == 'preg'  # the program's own split (gen_tiles.py)

        def layer(x0, w):
            qkv = llm_rms(x0, w['g1']) @ w['wqkv']
            c, s = w['cs'].view(T, 1, -1), w['sn'].view(T, 1, -1)
            q = llm_rope(qkv[:, :D].view(T, H, dh), c, s).transpose(0, 1)          # [H, T, dh]
            k = llm_rope(qkv[:, D:D + KD].view(T, Hk, dh), c, s).transpose(0, 1)  # [Hk, T, dh]
            v = qkv[:, D + KD:].view(T, Hk, dh).transpose(0, 1)
            if sdpa:
                o = F_.scaled_dot_product_attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), is_causal=True,
                                                    enable_gqa=True)[0]
            else:
                mask = torch.ones(T, T, dtype=torch.bool, device=dev).triu(1)
                parts = []
                for g in (range(Hk) if grouped else [None]):
                    qg = q.view(Hk, G, T, dh) if g is None else q.view(Hk, G, T, dh)[g:g + 1]
                    kg, vg = (k, v) if g is None else (k[g:g + 1], v[g:g + 1])
                    sc = (qg @ kg.unsqueeze(1).transpose(-1, -2)) / dh ** 0.5
                    parts.append(torch.softmax(sc.masked_fill(mask, float('-inf')), -1) @ vg.unsqueeze(1))
                o = torch.cat(parts).view(H, T, dh)
            h1 = x0 + o.transpose(0, 1).reshape(T, D) @ w['wo']
            return llm_ffn(w, h1) if ffn else h1
        return layer, (w['x0'], w)
    raise LookupError(f'no llm workload for {stem}')


def llm_vals(stem, out):
    D = (LLM_PX if stem.startswith('px_') else LLM)['D']
    f = out.reshape(-1)
    v = [f[i].item() for i in LLM_CELLS(out.shape[0], D)]
    return v + ([float((f.double() ** 2).sum())] if stem.startswith('px_') else [])


def workload(stem, mode='eager'):
    """-> (fn, args): fn(*args) returns the program's outputs in print order. `mode` picks the dtype / attention
    form of the model-scale workloads (`llm_workload`) and the projections' bf16 reference."""
    if stem.startswith(('pt_llm_', 'px_llm_')):
        return llm_workload(stem.replace('_pv', ''), mode)
    pj = re.fullmatch(r'pt_pj_m(\d+)_n(\d+)_k(\d+)', stem)
    if pj:  # gen_tiles.py projection: M x K activations times a K x N weight
        M, N, K = map(int, pj.groups())
        dt = torch.bfloat16 if mode.startswith('bf16') else torch.float32
        a, b = op('a', M * K).view(M, K).to(dt), op('b', K * N).view(K, N).to(dt)
        return (lambda a, b: corners(a @ b)), (a, b)
    m = re.fullmatch(r'pt_(\w+?)_(\d+)', stem)
    if stem.startswith('pt_rm_m'):
        M, d, h = map(int, re.findall(r'\d+', stem))
        x, w1, w2 = op('a', M * d).view(M, d), op('b', d * h).view(d, h), op('e', h * d).view(h, d)
        b1 = bias(h, 13, 6, 5.0)
        return (lambda x, w1, w2, b1: corners(torch.relu(x @ w1 + b1) @ w2 + x)), (x, w1, w2, b1)
    if stem.startswith('pt_fir_'):
        n, taps = map(int, stem.split('_')[2:])
        # The signal holds n + taps samples; outputs 0..n-1 read up to n + taps - 2.
        c, s = op('e', taps).view(1, 1, taps), op('a', n + taps)[:n + taps - 1].view(1, 1, n + taps - 1)
        return (lambda s, c: corners(torch.nn.functional.conv1d(s, c))), (s, c)
    if stem.startswith('pt_conv_'):
        n, kk = map(int, stem.split('_')[2:])
        w = n + kk - 1
        wt, img = op('e', kk * kk).view(1, 1, kk, kk), op('a', w * w).view(1, 1, w, w)
        return (lambda img, wt: corners(torch.nn.functional.conv2d(img, wt))), (img, wt)
    if stem.startswith(('pt_convmc_', 'px_convmc_')):  # gen_tiles.py `conv_mc`: NCHW, C -> C, valid
        N, C, n, kk = map(int, stem.split('_')[2:])
        wt = op('e', C * C * kk * kk).view(C, C, kk, kk)
        img = op('a', N * C * (n + kk - 1) ** 2).view(N, C, n + kk - 1, n + kk - 1)
        return (lambda img, wt: corners(torch.nn.functional.conv2d(img, wt))), (img, wt)
    # generic-fusion chains (gen_tiles.py `attention`, `smlp`, `firew_rowsum`, `ewconvew`).
    if stem.startswith('pt_attn_'):
        T, d = map(int, stem.split('_')[2:])
        q, kk, vv = (op(k, T * d).view(T, d) for k in 'abe')
        sd = float(f'{d ** 0.5:.1f}')  # the program's literal

        def attn(q, kk, vv):
            s = q @ kk.T
            ex = expo((s - s.max(dim=1, keepdim=True).values) / sd)
            return corners((ex / ex.sum(dim=1, keepdim=True)) @ vv)
        if mode.startswith('sdpa'):  # long-rows: the library's fused attention (exact exp: a time reference,
            # its values differ from the program's `expo` by the approximation, so it never gates)
            def sdpa(q, kk, vv):
                o = torch.nn.functional.scaled_dot_product_attention(q[None], kk[None], vv[None], scale=1.0 / sd)
                return corners(o[0])
            return sdpa, (q, kk, vv)
        return attn, (q, kk, vv)
    if stem.startswith('pt_lsm_'):  # gen_tiles.py `long_softmax`: few long rows of q·W scores
        R, L = map(int, stem.split('_')[2:])
        q, w = op('a', R * 16).view(R, 16), op('b', 16 * L).view(16, L)

        def lsm(q, w):
            s = q @ w
            ex = expo((s - s.max(dim=1, keepdim=True).values) / 4.0)
            return corners(ex / ex.sum(dim=1, keepdim=True))
        if mode.startswith('sdpa'):  # the library softmax (exact exp: a time reference, never gates)
            return (lambda q, w: corners(torch.softmax((q @ w) / 4.0, dim=1))), (q, w)
        return lsm, (q, w)
    if stem.startswith('pt_smlp_'):
        M, d, h = map(int, stem.split('_')[2:])
        x, w1, w2 = op('a', M * d).view(M, d), op('b', d * h).view(d, h), op('e', h * d).view(h, d)
        b1 = bias(h, 13, 6, 5.0)
        return (lambda x, w1, w2, b1: corners(torch.relu(x @ w1 + b1) @ w2)), (x, w1, w2, b1)
    if stem.startswith('pt_firew_rowsum_'):
        R, C, taps = map(int, re.findall(r'\d+', stem))
        c, s = op('e', taps).view(1, 1, taps), op('a', R * C + taps - 1).view(1, 1, R * C + taps - 1)
        return (lambda s, c: corners(torch.relu(torch.nn.functional.conv1d(s, c) * 0.5).view(R, C).sum(dim=1))), (s, c)
    if stem.startswith('pt_ewconvew_'):
        n, kk = map(int, stem.split('_')[2:])
        w = n + kk - 1
        wt, img = op('e', kk * kk).view(1, 1, kk, kk), op('a', w * w).view(1, 1, w, w)
        return (lambda img, wt: corners(torch.relu(torch.nn.functional.conv2d(img * 0.5 + 0.25, wt)))), (img, wt)
    if stem.startswith('pt_decode_'):
        T, d = map(int, stem.split('_')[2:])
        w = 16 * d
        wk = law(d * d, 7, 13, 101, 50, float(w)).view(d, d)
        wv = law(d * d, 11, 57, 103, 51, float(w)).view(d, d)
        F = law(T * d, 7, 13, 101, 50, 64.0).view(T, d)      # row t's written values, before + o
        Q = law(T * d, 5, 3, 97, 48, float(w)).view(T, d)     # q at step t

        def step(t, o, K, V):  # K, V: the cache (row t = x[t]·W), zero rows past t
            tv = t.view(1)  # a tensor index: one compiled graph for every step
            xr = F.index_select(0, tv).view(d) + o
            K.index_copy_(0, tv, (xr @ wk).view(1, d))
            V.index_copy_(0, tv, (xr @ wv).view(1, d))
            return (K @ Q.index_select(0, tv).view(d)) @ V

        def run(step):
            K, V = torch.zeros(T, d, device=dev), torch.zeros(T, d, device=dev)
            o = torch.zeros(d, device=dev)
            for t in range(T):
                o = step(torch.tensor(t, device=dev), o, K, V)
            return [o, o, o.sum().view(1)]
        return run, ('step', step)
    if stem.startswith(('pt_causal_', 'pt_prefill_')):
        # gen_tiles.py `decode(causal=True)` / `decode(prefill=P)`: softmax over rows <= t (expo, max-subtracted).
        n = [int(v) for v in stem.split('_')[2:]]
        T, P, d = n if len(n) == 3 else (n[0], 0, n[1])
        w, scale = 16 * d, d ** 0.5
        wk = law(d * d, 7, 13, 101, 50, float(w)).view(d, d)
        wv = law(d * d, 11, 57, 103, 51, float(w)).view(d, d)
        F = law(T * d, 7, 13, 101, 50, 64.0).view(T, d)
        Q = law(T * d, 5, 3, 97, 48, float(w)).view(T, d)
        rows = torch.arange(T, device=dev)

        def softmax_v(s, mask, V):  # s: (..., T); masked rows contribute 0
            s = torch.where(mask, s, -1e9)
            p = torch.where(mask, expo((s - s.max(dim=-1, keepdim=True).values) / scale), 0.0)
            return (p @ V) / p.sum(dim=-1, keepdim=True)

        def step(t, o, K, V):  # K, V: the cache, zero rows past t
            tv = t.view(1)
            xr = F.index_select(0, tv).view(d) + o
            K.index_copy_(0, tv, (xr @ wk).view(1, d))
            V.index_copy_(0, tv, (xr @ wv).view(1, d))
            return softmax_v(K @ Q.index_select(0, tv).view(d), rows <= t, V)

        def run(step):
            K, V = torch.zeros(T, d, device=dev), torch.zeros(T, d, device=dev)
            o, po = torch.zeros(d, device=dev), None
            if P:  # prefill: the prompt's K/V and causal attention, every prompt row's output
                K[:P], V[:P] = F[:P] @ wk, F[:P] @ wv
                po = softmax_v(Q[:P] @ K.T, rows.view(1, T) <= torch.arange(P, device=dev).view(P, 1), V)
                o = po[P - 1]
            for t in range(P, T):
                o = step(torch.tensor(t, device=dev), o, K, V)
            return [o, o, o.sum().view(1)] + ([po.sum().view(1)] if P else [])
        return run, ('step', step)
    if stem.startswith('pt_loopmm_'):
        n, iters = map(int, stem.split('_')[2:])
        x0, wm = law(n * n, 7, 13, 101, 50, 7.0).view(n, n), law(n * n, 11, 57, 103, 51, float(8 * n)).view(n, n)

        def loopmm(x, wm):
            for _ in range(iters):
                x = x @ wm
            return [x.reshape(-1)[:1]]
        return loopmm, (x0, wm)
    if stem.startswith('pt_rt_mlp2_'):
        n = int(stem.split('_')[-1])
        b, e, g, p = (op(k, n * n).view(n, n) for k in 'begp')
        b1, b2 = bias(n, 13, 6, 5.0), bias(n, 11, 5, 4.0)

        def rt(b, e, g, p, b1, b2):
            firsts = []  # every request's h[0]: keeps torch.compile from dropping requests 0..6
            for it in range(8):
                a, f = op('a', n * n, it).view(n, n), op('f', n * n, it).view(n, n)
                h = torch.relu(a @ b + b1) @ e + torch.relu(f @ g + b2) @ p
                firsts.append(h.reshape(-1)[0])
            return corners(h) + [torch.stack(firsts)]
        return rt, (b, e, g, p, b1, b2)
    probe, n = m.group(1), int(m.group(2))
    a, b, e, f, g = (op(k, n * n).view(n, n) for k in 'abefg')
    if probe == 'softmax_rows':
        def sm(a, b):
            c = a @ b
            ex = expo((c - c.max(dim=1, keepdim=True).values) / float(n))
            return corners(ex / ex.sum(dim=1, keepdim=True))
        return sm, (a, b)
    if probe == 'rowsum':
        return (lambda a, b: corners((a @ b).sum(dim=1))), (a, b)
    if probe == 'ew':
        return (lambda a, b: corners((a * 3.0 + 2.0) * b)), (a, b)
    if probe == 'fanout':
        def fo(a, b, e):
            c = a @ b
            return corners(c, c * 0.5 + 1.0, c @ e)
        return fo, (a, b, e)
    if probe == 'diamond':
        def dm(a, b, e, g):
            c = a @ b
            return corners(c @ e + c @ g)
        return dm, (a, b, e, g)
    if probe == 'mlp':  # d = ((a·b)·0.5)·e
        return (lambda a, b, e: corners(((a @ b) * 0.5) @ e)), (a, b, e)
    if probe == 'mlp2':  # d1 = relu(a·b + bias)·e ; d2 = relu(f·g + bias2)·p
        p = op('p', n * n).view(n, n)
        b1, b2 = bias(n, 13, 6, 5.0), bias(n, 11, 5, 4.0)
        return (lambda a, b, e, f, g, p, b1, b2: corners(torch.relu(a @ b + b1) @ e, torch.relu(f @ g + b2) @ p)), (a, b, e, f, g, p, b1, b2)
    if probe == 'twin':
        return (lambda a, b, e, f: corners(a @ b, e @ f)), (a, b, e, f)
    if probe == 'epi':
        return (lambda a, b: corners((a @ b) * 0.5 + 1.0)), (a, b)
    if probe == 'indep':
        return (lambda a, b, e, f: corners(a @ b + e @ f)), (a, b, e, f)
    if probe == 'loop':
        wl = law(n * n, 11, 57, 103, 51, float(n)).view(n, n)

        def lp(x, wl):
            for _ in range(8):
                x = torch.relu(x @ wl)
            return corners(x)
        return lp, (a, wl)
    raise LookupError(f'no workload for {stem}')


def ends(outs):
    v = []
    for o in outs:
        v += [o[0].item(), o[-1].item()] if o.numel() > 1 else [o[0].item()]
    return v


def gate(stem, outs, vals_dir, tol=1e-3):
    p = os.path.join(vals_dir, f'{stem}.txt')
    if not os.path.exists(p):
        return 'no-mapal', float('nan')
    want = [float(x) for x in open(p).read().split() if re.fullmatch(r'-?[\d.]+(e-?\d+)?', x)]
    if stem.startswith(('pt_llm_', 'px_llm_')):  # the printed cells (+ Σ out² for px), against the output's rms
        got, r = llm_vals(stem, outs), math.sqrt(float((outs.double() ** 2).mean()))
        dev_ = max((abs(g - w) / max(abs(w), r if i < 4 else abs(w), 1e-30) for i, (g, w) in enumerate(zip(got, want))),
                   default=float('nan'))
        return ('yes' if len(got) == len(want) > 0 and dev_ <= tol else 'no'), dev_
    got = ends(outs)
    loop = stem.startswith(('pt_decode_', 'pt_causal_', 'pt_prefill_', 'pt_gen_'))
    if loop:  # out[0], out[d-1], Σout (+ Σpo after a prefill)
        got = [outs[0][0].item(), outs[0][-1].item()] + [o[0].item() for o in outs[2:]]
    rms = [math.sqrt(float((o.double() ** 2).mean())) for o in outs]
    rms_of = [r for r, o in zip(rms, outs) for _ in range(2 if o.numel() > 1 else 1)]
    if loop:
        rms_of = [rms[0], rms[0]] + [abs(g) for g in got[2:]]
    dev_ = max((abs(g - w) / max(abs(w), r, 1e-30) for g, w, r in zip(got, want, rms_of)), default=float('nan'))
    ok = len(got) >= len(want) > 0 and dev_ <= tol
    return ('yes' if ok else 'no'), dev_


def timed(fn, args, reps, flush):
    sync = torch.cuda.synchronize if dev == 'cuda' else (lambda: None)
    for _ in range(5):
        fn(*args)
    sync()
    ts = []
    for _ in range(reps):
        if flush is not None:
            flush.zero_()
        if dev == 'cuda':
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            fn(*args)
            e1.record()
            e1.synchronize()
            ts.append(e0.elapsed_time(e1))
        else:
            t0 = time.perf_counter()
            fn(*args)
            ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts


def main():
    reps, vals_dir, stems = int(sys.argv[1]), sys.argv[2], sys.argv[3:]
    print(f'torch version={torch.__version__} cuda={torch.version.cuda} cudnn={torch.backends.cudnn.version()} '
          f'device="{torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"}"', flush=True)
    flush = torch.empty(64 << 20, device=dev) if dev == 'cuda' else None  # 256 MB
    for stem in stems:
        # pt_gen is pt_prefill's math in one loop; `_tf32` programs are the same math under the program's `: tf32` face.
        # (a model-scale `_pv` diagnostic computes its base stem's layer: `workload` maps it.)
        base = re.sub(r'^pt_gen_', 'pt_prefill_', stem.removesuffix('_tf32'))
        llm = base.startswith(('pt_llm_', 'px_llm_'))
        try:
            fn, args = workload(base)
        except Exception as ex:  # noqa: BLE001 — a stem without a torch leg is a row, not the end of the run
            print(f'torch prog={stem} NOWORKLOAD {type(ex).__name__}: {str(ex)[:200]}', flush=True)
            continue
        # ADR-0041: the `: tf32 | f32` programs (and the conv family, whose default cuDNN leg is TF32) also get
        # the TF32-on legs (matmul and cuDNN allow_tf32), beside the pure-FP32 ones above.
        modes = ['eager', 'compile']
        if 'tf32' in stem or 'convmc' in stem or '_mlp' in stem:
            modes += ['eager-tf32', 'compile-tf32']
        # model-scale: SDPA legs; the bf16 reference (reported apart, gated loosely against Mapal's f32 values); a
        # `_tf32` twin times only its TF32 legs (the f32 and bf16 ones are its base stem's).
        scale = llm or base.startswith('pt_pj_')
        if llm:
            modes += ['sdpa'] + (['sdpa-tf32'] if 'tf32' in stem else [])
        # long-rows: the attention rows against SDPA, the long softmax rows against torch.softmax (both exact exp).
        sdpa_ref = base.startswith(('pt_attn_', 'pt_lsm_'))
        if sdpa_ref:
            modes += ['sdpa', 'sdpa-compile']
        if scale:
            modes = [m for m in modes if m.endswith('-tf32')] if 'tf32' in stem else modes + ['bf16', 'bf16-compile']
        # TORCH_MODES=<mode,...> keeps only those legs (a CUPTI trace of one leg).
        keep = [m for m in os.environ.get('TORCH_MODES', '').split(',') if m]
        modes = [m for m in modes if not keep or m in keep]
        for mode in modes:
            tf = mode.endswith('-tf32')
            torch.backends.cuda.matmul.allow_tf32 = tf
            torch.backends.cudnn.allow_tf32 = tf
            tol = 5e-2 if mode.startswith('bf16') else 1e-2 if scale and tf else 1e-3
            try:
                fn, args = workload(base, mode) if scale or sdpa_ref else (fn, args)
                t0 = time.perf_counter()
                comp_mode = 'compile' in mode
                if fn.__name__ == 'run':  # decode: the step is what compiles, the loop stays Python
                    st = torch.compile(args[1], mode='max-autotune-no-cudagraphs') if comp_mode else args[1]
                    f, a = fn, (st,)
                else:
                    f = torch.compile(fn, mode='max-autotune-no-cudagraphs') if comp_mode else fn
                    a = args
                outs = f(*a)
                comp = time.perf_counter() - t0
                ok, dv = gate(stem, outs, vals_dir, tol)
                ts = timed(f, a, reps, flush)
                vals = llm_vals(base, outs) if llm else ends(outs)
                print(f'torch prog={stem} mode={mode} n={len(ts)} min_ms={ts[0]:.4f} med_ms={ts[len(ts) // 2]:.4f} '
                      f'max_ms={ts[-1]:.4f} within={ok} maxdev={dv:.3e} compile_s={comp:.1f} '
                      f'values={",".join(f"{v:.6g}" for v in vals)}', flush=True)
            except Exception as ex:  # noqa: BLE001 — one leg's failure is a row, not the session
                print(f'torch prog={stem} mode={mode} FAILED {type(ex).__name__}: {str(ex)[:200]}', flush=True)
            if scale:  # the next mode's tensors (GBs at this scale): drop this one's graphs and cached blocks
                fn = args = f = a = outs = None
                torch._dynamo.reset()
                torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
