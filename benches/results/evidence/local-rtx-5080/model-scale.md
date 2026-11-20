# local-rtx-5080 — model-scale suite: one Llama-3-8B-like layer (2026-11-21)

Machine: Threadripper 9970X + RTX 5080 (sm_120, 84 SMs, 64 MB L2, 16 GB, ~960 GB/s DRAM), CUDA 13, driver 620.70,
clang/LLVM 23, branch `local-rtx-5080` + this suite. Competitors on the same box: cuBLAS / cuBLASLt (`gemm_lib.cu`),
PyTorch 2.11.0+cu130 eager, `torch.compile(mode="max-autotune-no-cudagraphs")`, `F.scaled_dot_product_attention`
(GQA), TF32 twins, and a bf16 run of the same layer as the real-world reference (reported apart, never a like-for-like
row).

**Protocol.** Every timed process held the shared bench lock (`flock`) for its run, ran only with no
`MapleStory.exe`, `sensors` Tctl < 85 C (37–66 C seen) and CPUs 0–7 under 10 % busy (/proc/stat over 2 s; 0 % at every
start and end), and was pinned `taskset -c 0-7`: another agent's builds load CPUs 8–31 / 40–63, so the usual
"load < 3" wait never ends. Conditions per run: `conditions.txt` beside each raw result. Mapal: `box_run.sh`, value
gates before timing, the program's own `iter ms` (median of 5 processes; 3 for decode, 1 for the host-placed natural
forms — `model_scale_once.sh`), arms interleaved ABBA, `null` = a copy of the shipped build. Libraries: median of 30
events-timed repetitions after a 256 MB L2 flush; `gemm_lib` also `-fresh` legs (see L2 boundary below).

## Programs (`benches/gpu/gen_tiles.py` `llm_*`, `benches/algorithms/`)

d_model 4096, 32 query heads × 128, 8 KV heads (GQA 4:1), SwiGLU FFN 14336, RMSNorm (eps 1e-5), RoPE (rotate_half),
f32, idiomatic Mapal (flat maps over index spaces, folds, guards, a `loop` for decode steps).

| family | programs | what one timed run is |
|---|---|---|
| (a) decode, attention half | `pt_llm_dec_b{1,8,32}_l{2048,8192}` | 16 steps: RMSNorm → fused QKV (4096×6144) → RoPE → KV-cache append at position L−16+t (cache `[B][L][8][128]` K and V, loop-carried) → GQA attention over positions ≤ p (masked, max-subtracted softmax) → O (4096×4096) + residual; the output is the next step's input |
| (c) block decode | `pt_llm_blkdec_b1_l4096` | the same + RMSNorm → gate, up (4096×14336) → SiLU(gate)·up → down + residual, per step |
| (b) prefill, attention half | `pt_llm_pre_t2048`, `pt_llm_pre_t8192` | causal, T tokens; at 8192 the 32-head score arrays (2³¹ cells) do not fit i32 or 16 GB: a loop over the 8 KV-head groups, each adding its rows of the O projection to the residual |
| (c) block prefill | `pt_llm_blkpre_t2048` | attention half + FFN half |
| (d) projections | `pt_pj_m{1,16,128,2048}_n{6144,4096,28672}_k4096`, `pt_pj_m*_n4096_k14336` | QKV (fused), O, gate/up (fused), down, `composed`'s projection |
| TF32 | `*_tf32` (prefill, block prefill, m ≥ 128 projections) | ADR-0041 `: tf32 \| f32` on every fold |
| small twins | `px_llm_{dec_b2_l16,blkdec_b2_l16,pre_t12,preg_t12,blkpre_t12}` (+ `_pv`) | the same generators at D 64, 4 heads, 2 KV heads, F 96 |
| diagnostics | `*_pv` | the same layer with the RoPE indices written provable (findings 2–3); **not benchmarks** |

Not in the Core op set and written out: exp (`e^(z/256)`'s degree-5 Taylor polynomial, squared 8 times, 0 below −80:
~5e-5 relative), sqrt (Heron, 16 steps from (1+a)/2), sin/cos (the rotary tables are inputs, as engines precompute
them). Operand laws are OPS's with `t % mod` first (no i32 overflow at 2²⁸ cells); weights have std 1/√K. Printed:
four cells of the output (the first, the last row's second, one mid, the last) + `iter ms`.

## Value gates

| gate | scope | result |
|---|---|---|
| small scale vs the interpreter | `px_llm_*` (9 programs): exact face bit for bit, default face within ADR-0040 D2's bound (`.kmax`), each shipped and GPU-only (`--gpu-off=place`: the arm that runs the device kernels at this size), every site candidate | **58 / 58 pass** |
| full scale, Mapal vs its exact face | every timed program: the exact face built for the device (`XREF`: the interpreter's order bit for bit) is the oracle; default face within D2 (D3 for `_tf32`) of it, every arm and site candidate | **all pass** (run 2; run 1 failed three TF32 projection corners on the relative fallback under cancellation, \|d\| = 0.1 of M ≈ 4e4 — the projections now carry `_xabs` magnitude oracles and gate absolutely) |
| full scale, torch / cuBLAS vs Mapal | every torch and gemm_lib f32 leg: \|leg − mapal\| ≤ 1e-3 · max(\|mapal\|, rms(out)); TF32 legs against Mapal's TF32 program ≤ 1e-2 | **all within**: 123 f32 legs (maxdev 3e-8 – 4.7e-6), 54 TF32 legs (≤ 1.2e-3) |
| natural forms | `model_scale_once.sh`: each value within D2 of its `_pv` twin's exact oracle (the same function) | all pass |
| bf16 reference | deviation from Mapal f32 (reported, not gated as like-for-like) | 1e-3 – 1.7e-2 |

The small and full-scale torch layer (written with `torch.rsqrt`, `softmax`, SDPA, `F.silu`) agrees with the Mapal
programs (written with the Taylor exp and Heron sqrt) to 1e-7 – 1e-6: the programs compute the intended layer.

## Results — the layer (decode: ms per step = the 16-step run / 16)

Mapal columns: the natural program as shipped; the `_pv` diagnostic as shipped / its GPU-only arm (`--gpu-off=place`).
Best competitor: the fastest f32 leg within the gate (TF32 legs for `_tf32`); ratio = Mapal / it.

| workload | Mapal natural | Mapal `_pv` shipped / GPU-only | best competitor (leg) | ratio natural / GPU-only | torch bf16 ref | explanation (CUPTI, plan) |
|---|---:|---:|---|---:|---:|---|
| (a) decode B1 L2048 | 217.8 | 165.6 / 0.669 | 0.258 (torch-compile) | 846 / 2.60 | 0.149 | natural: RoPE unproven → attention + O on the host (findings 2–4); GPU-only: O+residual cell kernel 296 us (16 CTAs, 213 GB/s), QKV 118 us, P·V 99 us, RMSNorm 78 us (1 CTA) vs torch's fused rms+QKV 122 us, O 82 us |
| (a) decode B1 L8192 | 345.3 | 159.2 / 1.366 | 0.344 (torch-compile) | 1002 / 3.96 | 0.189 | as above; QKᵀ and P·V cell kernels grow with L |
| (a) decode B8 L2048 | 1131 | 771.8 / 1.215 | 0.678 (torch-compile) | 1667 / 1.79 | 0.218 | as above |
| (a) decode B8 L8192 | 2196 | 804.6 / 3.585 | 2.466 (torch-compile) | 891 / 1.45 | 0.485 | as below |
| (a) decode B32 L2048 | 4676 | 3125 / 3.676 | 2.277 (torch-compile) | 2053 / 1.61 | 0.462 | as below |
| (a) decode B32 L8192 | 8624 | 3009 / 12.55 | 9.523 (torch-compile) | 906 / 1.32 | 1.473 | GPU-only: QKᵀ 7.42 ms (K read once per query head: 4 GiB, finding 7), cache-update map 2.55 ms (whole cache, finding 8), P·V 1.66, O 0.62; torch f32 re-lays K / V out every step (`clone_permute` 4.5 + 2.5 ms) for its bmm (1.24 + 1.22 ms); bf16 reads a 1 GiB cache once (0.64 + 0.68 ms) |
| (c) block decode B1 L4096 | 718.2 | 825.3 / 2.300 | 1.074 (torch-compile) | 669 / 2.14 | 0.559 | GPU-only: down+SiLU+residual cell kernel 821 us (224 MB at 272 GB/s, 16 CTAs), O 298, gate / up 271 / 262 (naive, ~840 GB/s), P·V 197, RMSNorm 126 + 83 |
| (b) prefill T2048 | 280 017 | 86.84 (all device) | 8.322 (torch-compile) | 33 648 / 10.4 | 1.925 | QKᵀ cell kernel 38.7 ms (0.9 TFLOPS), O+residual cell 29.6 ms (2.3 TFLOPS), P·V cell 14.1 ms, QKV tile site 2.58 ms (stream-k, 40 TFLOPS: cuBLAS's simt 256x128 takes 2.16 ms in torch), softmax flow 1.30 ms (= torch's fused triton softmax, 1.30 ms), RMSNorm 0.38 ms (one thread per row) |
| (b) prefill T2048 TF32 | — | 86.23 | 6.068 (torch-compile-tf32) | — / 14.2 | 1.925 | the QKV site becomes mma (2.02 ms); the cell kernels above (82 of 86 ms) have no TF32 path |
| (b) prefill T8192 | not run | — | 83.71 (torch-compile) | — | 11.32 | no device form (finding 10); host estimate > 1 h. torch eager 164.7 ms, SDPA f32 OOM (math backend, 8 GiB), bf16 eager 13.16 |
| (c) block prefill T2048 | 934 279 | 217.0 (all device) | 27.93 (torch-compile) | 33 448 / 7.77 | 8.028 | + down+SiLU+residual cell kernel 117 ms (2.05 TFLOPS; 260–273 ms in 3 of 4 traced runs, untraced `iter ms` stable at 217: open), gate / up tile sites 5.83 / 6.42 ms (41 / 37 TFLOPS) |
| (c) block prefill T2048 TF32 | — | 213.5 | 18.53 (torch-compile-tf32) | — / 11.5 | 8.028 | as above |

Null variation (shipped vs its copy, interleaved): 0.0–0.1 % on the prefills; decode arms 3 processes each.

## Results — (d) the projections (ms; QKV = n6144, O = n4096 k4096, gate/up = n28672, down = n4096 k14336)

Best competitor over cuBLAS / cuBLASLt (flushed and `-fresh`), torch eager / compile, within the gate.

| projection | m | Mapal | best competitor (leg) | ratio | Mapal TF32 | best TF32 (leg) | ratio | torch bf16 | null |
|---|---:|---:|---|---:|---:|---|---:|---:|---:|
| QKV | 1 | 0.0986 | 0.0532 (cuBLAS fresh) | 1.85 | | | | 0.0635 | 1.0 % |
| O | 1 | 0.0503 | 0.0225 (Lt fresh) | 2.24 | | | | 0.0471 | 0.2 % |
| gate/up | 1 | 0.5614 | 0.5018 (Lt fresh) | 1.12 | | | | 0.2744 | 1.5 % |
| down | 1 | 0.2518 | 0.2314 (cuBLAS fresh) | 1.09 | | | | 0.1413 | 0.8 % |
| QKV | 16 | 0.0954 | 0.0881 (Lt fresh) | 1.08 | | | | 0.0635 | 0.0 % |
| O | 16 | 0.0572 | 0.0553 (cuBLAS fresh) | 1.03 | | | | 0.0430 | 10.4 % |
| gate/up | 16 | 0.5766 | 0.5222 (Lt fresh) | 1.10 | | | | 0.2744 | 1.3 % |
| down | 16 | 0.2576 | 0.2765 (torch-compile) | 0.93 | | | | 0.1393 | 2.3 % |
| QKV | 128 | 0.2449 | 0.1802 (cuBLAS fresh) | 1.36 | 0.2389 | 0.1249 (cuBLAS fresh) | 1.91 | 0.0860 | 2.6 % |
| O | 128 | 0.1909 | 0.1352 (cuBLAS fresh) | 1.41 | 0.1359 | 0.0819 (Lt fresh) | 1.66 | 0.0532 | 3.3 % |
| gate/up | 128 | 0.8053 | 0.7496 (Lt fresh) | 1.07 | 0.6938 | 0.5898 (cuBLAS fresh) | 1.18 | 0.3195 | 0.5 % |
| down | 128 | 0.4824 | 0.4014 (cuBLAS fresh) | 1.20 | 0.4573 | 0.3092 (cuBLAS fresh) | 1.48 | 0.1720 | 1.4 % |
| QKV | 2048 | 2.648 | 2.578 (torch-compile) | 1.03 | 2.025 | 1.756 (cuBLAS fresh) | 1.15 | 0.8765 | 0.0 % |
| O | 2048 | 1.840 | 1.747 (torch-eager) | 1.05 | 1.422 | 1.165 (cuBLAS fresh) | 1.22 | 0.5960 | 0.1 % |
| gate/up | 2048 | 11.79 | 12.25 (Lt fresh) | 0.96 | 9.081 | 8.080 (cuBLAS fresh) | 1.12 | 3.947 | 0.1 % |
| down | 2048 | 6.433 | 6.425 (torch-compile) | 1.00 | 4.940 | 4.253 (torch-compile-tf32) | 1.16 | 2.085 | 0.3 % |

Run 1 (the same bundle less the magnitude oracles and `-fresh` legs) agrees within 1–2 % on every common row.

- **m = 2048, f32: parity.** Mapal's stream-k FFMA tile (168 CTAs, 236 regs, 16.6 KB smem) 11.58 ms (CUPTI) =
  41.5 TFLOPS against cuBLAS's `cutlass_80_simt_sgemm_256x128_8x4` 11.96 ms.
- **TF32: 1.12–1.22× at m 2048, 1.18–1.91× at m 128.** Mapal's `Tf32Mma` tile (mt 4 × nt 4, bk 16, 2 stages, 37 KB
  smem, 122 regs) 9.07 ms = 53 TFLOPS; cuBLAS's `cutlass_80_tensorop_s1688gemm_128x128_32x3` (bk 32, 3 stages,
  98 KB smem) 7.98 ms = 60 TFLOPS. At m 128 Mapal's mma grid is 128 CTAs (1.5 waves, no split) where cuBLAS splits K
  8 ways (`64x64_16x6`, 1024 CTAs): 122 vs 101 us at O.
- **m = 128, f32: 1.07–1.41×.** Mapal's FFMA tile 169 us (512 CTAs, 128 regs) vs cuBLAS's `simt_sgemm_128x64_8x5`
  split-K 5 at 135–149 us.
- **m = 1 / 16, weights ≥ 235 MB: 0.93–1.12×**, DRAM-bound (Mapal's gate/up GEMV 534 us = 880 GB/s; cuBLAS's gemvx
  531 us).
- **m = 1, weights ≤ 96 MB: 1.85–2.24×, but not the kernel**: Mapal's O kernel is 22.8 us under CUPTI (the fresh
  cuBLASLt 22.5 us); `iter ms` is 50 us — the runtime's site decision, launch and synchronize inside the timed region
  (the cost `gpu-analysis.md` §1 measured, 50–90 us per program with a site table).

## L2 boundary (why `-fresh` legs)

A Mapal program generates its inputs on the device right before `t0`, so the last-written ≤ 64 MB of them sit in
L2 when the clock starts; torch and the flushed library legs start cold. `gemm_lib`'s `-fresh` legs regenerate the
operands before each repetition (Mapal's boundary): QKV / O at m = 1 (96 / 64 MB of weights) take 0.053 / 0.0225 ms
fresh against 0.088 / 0.084 ms flushed — cuBLASLt reads the fresh weights at ~2.9 TB/s from L2. Against the flushed
legs Mapal looked 0.60–0.84× (faster); against its own boundary it is 1.85–2.24× slower. Weights ≥ 235 MB move by
< 10 %. The layer programs' inputs are 160 MB – 2 GB: only their first step sees the L2 effect.

## Traces

`model_scale_trace.sh`: one CUPTI-traced process per leg (`benches/local/ktrace.cpp`) for 16 rows — every
projection shape class (m 1 L2-resident and DRAM-bound, m 128 f32 / TF32, m 2048 f32 / TF32) with its gemm_lib and
torch legs; the prefill, block prefill and TF32 prefill with torch-compile, SDPA and bf16-compile; decode B1 L2048,
B32 L2048 / L8192 and the block decode (GPU-only arm) with torch-compile, eager, SDPA and bf16-compile.
`ktrace_sum.py` sums per kernel (Mapal over its timed kernels; torch / gemm_lib whole-process). The kernel times above
are these traces' (Mapal's also `MAPAL_GPU_TRACE`'s per-launch ms, which agree within 1 %). Summaries and TSVs are in
the raw dir (`trace/`).

Where the time goes, ranked by the gap it explains:

| cause | rows | Mapal kernel time | what the competitor runs |
|---|---|---|---|
| host placement from unproven RoPE indices, serial host code (findings 2–4) | every natural layer program | 160 ms – 15.6 min of host time | — |
| GEMM + residual in one map → per-cell kernel (finding 6) | prefill O / down, decode O / down | 29.6 + 117 ms (prefill block), 0.30 + 0.82 ms per step | cutlass simt / triton GEMM with the add fused or separate |
| attention as per-cell kernels, K re-read per query head (finding 7) | prefill, decode at B ≥ 8 or L 8192 | 38.7 + 14.1 ms (prefill), 7.42 + 1.66 ms per step (B32 L8192) | cuBLAS batched GEMMs / triton bmm templates (f32), flash (bf16) |
| whole-cache update map each step (finding 8) | decode B > 1 | 2.55 ms per step (B32 L8192) | `index_copy_` of one row: 2 us |
| one-thread-per-row RMSNorm (finding 9) | every layer row | 78–378 us | triton fused reduction, 25–122 us (fused into the QKV GEMV at decode) |
| runtime site decision inside the timed region | m = 1 projections | ~27 us of 50 | — |
| TF32 mma tile shallower than cutlass's | TF32 projections | 1.12–1.91× | `s1688gemm` 128x128_32x3 / 64x64_16x6 split-K |

## Open findings (compiler; not fixed here) — minimal repros

Each repro is run with `target/release/examples/emit <f> - --rewrite --target=rtx-5080 --host=x86-64-v3
--device-out=<f>.dev.ll` (placement: the `; mapal.move` / `; mapal.host` lines; kernels: `; mapal.plan`) or the
interpreter (`examples/run`).

1. **A loop's exit value is invisible to a later map or fold body** (`L1101 unresolved name`), also through an
   alias (`out -> res;`); a top-level index (`out[3]`) or a fold over the array itself works. Every model-scale
   program prints indexed cells for this reason, and the 8192 prefill adds its O projection inside the loop.
   ```
   fn main() {
       (1.0, 4) -> fill -> x0;
       2 -> iota -> r2;
       mut x: [f32; 4] <- x0;
       mut t: i32 <- 0;
       loop { (t < 3) -> { -true-> { x -> map { v -> v * 2.0 } -> x; t + 1 -> t; -> loop; } -false-> x -> out; } }
       r2 -> map { j -> out[2 + j] } -> last;   // error[L1101] unresolved name `out`
       last[1] -> println;
   }
   ```
2. **`bounds_proof` does not use a guard's condition** (`algo.rs` `bounds_proof`: interval analysis, no path facts).
   Llama's rotate_half reads the partner cell `c ± 64` under `c % 128 < 64`; the index keeps its trap:
   ```
   fn main() {
       256 -> iota -> r;
       r -> map { t -> t % 101 -> widen_f32 } -> a;
       r -> map { c -> c % 128 -> i; (i < 64) -> { -true-> a[c + 64]; -false-> a[c - 64]; } } -> b;
       b[0] -> println;
   }   // CPU build: 2 trap calls; written `a[c - i + (i + 64) % 128]`: 0
   ```
3. **`bounds_proof` never ranges a loop-carried scalar**: a decode step's position (`cs[p * 64 + i]`, `p = L − 16 + t`,
   `t < 16`), and the KV-group loop's `g` (`q[… + (g * 4 + hh) * 128 …]`). No rewrite of the index proves it
   (`(p % L + L) % L` keeps the trap: the merge view is unranged).
   ```
   fn main() {
       128 -> iota -> r;  r -> map { t -> t % 101 -> widen_f32 } -> tab;  4 -> iota -> r4;
       (0.0, 4) -> fill -> x0;  mut x: [f32; 4] <- x0;  mut t: i32 <- 0;
       loop { (t < 4) -> { -true-> { 28 + t -> p; r4 -> map { i -> x[i] + tab[p * 4 + i] } -> x; t + 1 -> t; -> loop; }
                           -false-> x -> out; } }
       out[0] -> println;
   }   // 1 trap call
   ```
   **Effect of 2 and 3:** a site with an unproven index is not device-capable, so RoPE — and everything that reads its
   output: the cache update, the attention, the O projection — runs on the host. The natural programs measure
   218 ms – 8.6 s per decode step (B 1 – 32) against torch's 0.26 – 9.5 ms, 280 s for the 2048 prefill and 934 s for
   the 2048 block (torch 8.3 / 27.9 ms). The `_pv` twins (rotate_half's partner as `(ri + 64) % 128` of its head,
   the decode position's rotary values as the tables' closed form) are trap-free and bit-identical in the interpreter.
4. **A GPU build's host part is serial.** The natural prefill's host code calls no `mapal_par_*` (the CPU-only build of
   the same program: 52 `mapal_par_task`, 165 `mapal_par_dep`); the 280 s ran at 99.7 % of one core.
5. **`place` puts the decode step's RMSNorm and its O projection + residual on the host** even when every index is
   proven (`_pv` shipped: `; mapal.move … carried=…x:two h2d=16384 d2h=16384` per step at B 1). The host O
   projection is the serial `wo[k * 4096 + j]` column walk: 160 ms – 3.1 s per step. With `--gpu-off=place` the
   same program keeps the caches and `x` resident (`h2d=0 d2h=0`) at 0.67 – 12.6 ms per step. A placement fact is
   missing: a per-step host round trip of a loop-carried state against a device loop body.
6. **A GEMM whose map body adds a residual after its fold is not a tile site** — `fold { … } -> y; x[e] + y` becomes a
   per-cell kernel (one thread per output, serial K fold), while the same fold with the residual as its own map is
   a tile site with 6 candidates:
   ```
   o -> map { e -> e / 512 -> i; e % 512 -> j; (0.0, rk) -> fold { acc, k -> acc + a[i * 512 + k] * b[k * 512 + j] } -> y; a[e] + y } -> c;
   // device_sites=0, one `kind=cell` kernel;  `… } -> y;` then `o -> map { e -> a[e] + y[e] } -> c;`: device_sites=1, 6 `kind=tile`
   ```
   The layer's O and down projections are written this way (`y = x + attn · Wo`): prefill O 29.7 ms at 2.3 TFLOPS
   (the QKV tile site next to it: 39 TFLOPS), down 117 ms at 2.05 TFLOPS; decode O 0.30 ms per step (64 MB at
   213 GB/s: 16 CTAs on 84 SMs), down 0.82 ms (224 MB at 272 GB/s).
7. **Attention runs as per-cell kernels.** QKᵀ with the causal guard and the GQA head index (`e / T² → h`, `h / 4 →
   KV head`) and P·V are not tile sites: prefill 2048 scores 38.7 ms (0.9 TFLOPS), P·V 14.1 ms (2.4 TFLOPS). At decode
   the 4 query heads of a KV head each read that head's K rows: 4 GiB of K per step at B 32 / L 8192, 7.4 ms
   (580 GB/s effective) where one read of K is 1.1 ms; P·V 1.66 ms.
8. **`write_delta` misses a per-sequence cache append.** `[B][L][8][128]` written at position p writes B strided rows
   per step (contiguous only at B = 1), so the carried cache is `Delta::Top` and its update map runs over the whole
   cache every step: 2.56 ms per step at B 32 / L 8192 (K and V, 2 GiB).
9. **RMSNorm's row map is one thread per row**: `kind=cell work=4113` — 4096-long fold and 16 Heron steps serial in
   one thread, 0.08 – 0.15 ms per norm per step even at B 1.
10. **The 8192-token prefill has no device form.** The 32-head score arrays are 2³¹ cells (past i32 `iota`, and
    8 GiB each); the KV-group loop that splits them indexes by its counter (finding 3). Its host estimate is > 1 h
    per run (the 2048 one is 280 s; the attention grows 16×): Mapal is not timed there, the torch legs are.

Method notes (fixed here): a TF32 corner whose |exact| ≪ M fails the relative fallback inside D3's absolute bound
(run 1, three projections) — the projections carry `_xabs` magnitude oracles now; flushed library legs overstate
their time at Mapal's L2 boundary (above). Competitor note: torch's f32 SDPA with `enable_gqa` falls back to the
math backend (it materializes the repeated K/V: OOM at B 32 / L 8192, 8.8 ms per step at B 32 / L 2048), so the
`sdpa` legs are never the best f32 leg; flash runs at bf16 only.

## Intended vs realized (end check, this suite)

No compiler code changed here, so the end check states what the facts should yield on these programs and what the
emitted plan, the PTX/SASS checks and the traces show. `evidence.py` (run 2): layer bundle 7937 checks, 577
mismatches (257 in the exact-face oracle builds, 114 in the `px_` twins, 206 in shipped builds and their site
candidates); projection bundle 9792 checks, 871 mismatches. On shipped builds the top mismatches are
`streamk: one wave of blocks` (44 / 35), `arch-rungs` (35 / 24), `face-registers: no spill` (27 / 22),
`async-h2d` (25), `kmajor-a` (20), `splitk-atomic-zero` (16 / 13), `min-blocks` (15 / 14) — none of them is one of
this suite's gap causes (those are placement and site-recognition decisions: the findings above), so they stay open as the
part's existing evidence items.

| family | intended (facts → kernels, movement) | realized | time |
|---|---|---|---|
| decode | weights read once per step; KV cache and `x` device-resident (`h2d = d2h = 0` per trip); only the new row written (`write_delta`); QKV / O as tile sites; QKᵀ reading each K row once per KV head; RMSNorm a tree row fold | natural: traps on RoPE → cache, attention, O on the host, serial. `_pv` shipped: caches resident, `x` `two` (16–512 KB each way per step), RMSNorm + O on the host. `_pv` GPU-only: all resident, but O + residual a cell kernel, K read 4×, the cache map whole each step (`Delta::Top`), RMSNorm one thread per row | 669–2053× (natural), 1.32–3.96× (GPU-only) of torch-compile f32 |
| prefill | QKV / O / gate / up / down as tile sites; per-head QKᵀ and P·V as tile sites; softmax one flow kernel | QKV / gate / up tile sites at 37–41 TFLOPS (cuBLAS's level); softmax one flow kernel (1.30 ms = torch's); O and down (+ residual), QKᵀ, P·V per-cell kernels at 0.9–2.4 TFLOPS | 7.8–10.4× (f32), 11.5–14.2× (TF32) |
| projections | tile sites; TF32 `mma.sync` | m 2048 f32 at parity; TF32 mma at 88 % of cutlass's; m 128 1.07–1.41×; DRAM-bound GEMVs at 0.93–1.12×; the m = 1 kernel at cuBLASLt's speed with 27 us of runtime around it | 0.93–2.24× |

## Commands

```sh
# bundles (the stems / XREF / PX / ARMS lists are in the raw dir's `commands.txt`)
CELLS="" XREF="<stems>" PX="<px stems>" ARMS="rtx-5080:<stem>::place:off ..." CUTLASS_SRC=<cutlass> \
    benches/local/gpu_bundle.sh <bundle> <stems>
# timed runs (bench lock, CPUs 0-7 idle + pinned, Tctl, no game)
BENCH_LOCK=<lock> TORCH_PY=<venv>/bin/python PROBE_ARMS="on null" PROBE_REPS=5 \
    benches/local/model_scale_run.sh <bundle> <results> 30
BENCH_LOCK=<lock> benches/local/model_scale_once.sh <bundle> <results> <natural stems>   # host-placed forms
BENCH_LOCK=<lock> TORCH_PY=... benches/local/model_scale_trace.sh <bundle> <out> ktrace.so <stem>:<torch modes>...
python3 benches/local/model_scale_report.py <results>/results-rtx-5080.txt ...
```

Raw output (results, box.out, CUPTI traces, conditions, the bundles' `bundle.log`):
`.claude/raw-evidence/local-rtx-5080/model-scale/` (local, not in git; not yet uploaded with
`benches/gpu/archive_raw.sh`).
