#!/usr/bin/env python3
"""S64 evidence suite, the PyTorch legs: each workload of gen_tiles.py's timed programs written the way a
PyTorch user would, eager and `torch.compile(mode="max-autotune-no-cudagraphs")`, with the same operand
laws (gen_tiles.py OPS / BIAS / law / decode — change both or neither), pure FP32 (TF32 off).

Timing (chain_bench.cu's convention): 5 warm-ups, then `reps` repetitions, each after a 256 MB L2 flush,
bracketed by CUDA events; min / median / max ms. Values: every output's [0] and [last], in the program's
print order, gated against Mapal's printed values (box_run.sh `vals/<stem>.txt`) as chain_bench does:
|torch − mapal| ≤ 1e-3 · max(|mapal|, rms(torch output)).

usage: torch_bench.py <reps> <vals dir> <stem>...
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
    """exp(x) for x <= 0 as (1 + x/256)^256 (gen_tiles.py `expo`)."""
    v = torch.clamp(x / 256.0 + 1.0, min=0.0)
    for _ in range(8):
        v = v * v
    return v


def corners(*outs):
    return [t.reshape(-1) for t in outs]


def workload(stem):
    """-> (fn, args): fn(*args) returns the program's outputs in print order."""
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
        return attn, (q, kk, vv)
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
    if probe == 'indep':
        return (lambda a, b, e, f: corners(a @ b + e @ f)), (a, b, e, f)
    if probe == 'loop':
        wl = law(n * n, 11, 57, 103, 51, float(n)).view(n, n)

        def lp(x, wl):
            for _ in range(8):
                x = torch.relu(x @ wl)
            return corners(x)
        return lp, (a, wl)
    raise SystemExit(f'torch_bench: no workload for {stem}')


def ends(outs):
    v = []
    for o in outs:
        v += [o[0].item(), o[-1].item()] if o.numel() > 1 else [o[0].item()]
    return v


def gate(stem, outs, vals_dir):
    p = os.path.join(vals_dir, f'{stem}.txt')
    if not os.path.exists(p):
        return 'no-mapal', float('nan')
    want = [float(x) for x in open(p).read().split() if re.fullmatch(r'-?[\d.]+(e-?\d+)?', x)]
    got = ends(outs)
    loop = stem.startswith(('pt_decode_', 'pt_causal_', 'pt_prefill_'))
    if loop:  # out[0], out[d-1], Σout (+ Σpo after a prefill)
        got = [outs[0][0].item(), outs[0][-1].item()] + [o[0].item() for o in outs[2:]]
    rms = [math.sqrt(float((o.double() ** 2).mean())) for o in outs]
    rms_of = [r for r, o in zip(rms, outs) for _ in range(2 if o.numel() > 1 else 1)]
    if loop:
        rms_of = [rms[0], rms[0]] + [abs(g) for g in got[2:]]
    dev_ = max((abs(g - w) / max(abs(w), r, 1e-30) for g, w, r in zip(got, want, rms_of)), default=float('nan'))
    ok = len(got) >= len(want) > 0 and dev_ <= 1e-3
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
        fn, args = workload(stem)
        for mode in ('eager', 'compile'):
            try:
                t0 = time.perf_counter()
                if fn.__name__ == 'run':  # decode: the step is what compiles, the loop stays Python
                    st = args[1] if mode == 'eager' else torch.compile(args[1], mode='max-autotune-no-cudagraphs')
                    f, a = fn, (st,)
                else:
                    f = fn if mode == 'eager' else torch.compile(fn, mode='max-autotune-no-cudagraphs')
                    a = args
                outs = f(*a)
                comp = time.perf_counter() - t0
                ok, dv = gate(stem, outs, vals_dir)
                ts = timed(f, a, reps, flush)
                print(f'torch prog={stem} mode={mode} n={len(ts)} min_ms={ts[0]:.4f} med_ms={ts[len(ts) // 2]:.4f} '
                      f'max_ms={ts[-1]:.4f} within={ok} maxdev={dv:.3e} compile_s={comp:.1f} '
                      f'values={",".join(f"{v:.6g}" for v in ends(outs))}', flush=True)
            except Exception as ex:  # noqa: BLE001 — one leg's failure is a row, not the session
                print(f'torch prog={stem} mode={mode} FAILED {type(ex).__name__}: {str(ex)[:200]}', flush=True)


if __name__ == '__main__':
    main()
