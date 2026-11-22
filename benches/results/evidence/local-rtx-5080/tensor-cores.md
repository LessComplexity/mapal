# local-rtx-5080 — tensor cores: 16-bit products on `mma.sync`, a bf16 KV cache, narrow `cp.async` panels (2026-11-23)

Branch `local-rtx-5080` @ `fedc07ca`, RTX 5080 (sm_120, 84 SMs, 64 MB L2, ~960 GB/s), CUDA 13, clang/LLVM 23, PyTorch
2.11.0+cu130. Decision: [ADR-0043](../../../../docs/decisions/ADR-0043-narrow-float-storage.md) slice 2 ("Implementation:
slice 2"). The cross-arch round 60f0f8ce's half-precision findings this answers: bf16-weight decode at B ≥ 8 1.5–2.7×
torch bf16 (attention over an f32 KV cache), narrow panels never staged by `cp.async`, the block decode's narrow
gate / up site spilling at 255 registers.

## What changed

| # | fact / property / rule | OFF arm |
| --- | --- | --- |
| 1 | **D4 members** `bf16` / `f16` in the compute set (`-> fold: bf16 \| f32 { … }`): `Precision::half` (weaker first), `unit` (`u_t`), `TileSite::half`; `precision_plan.bound` = `(1 + u_t)²(1 + γ_{K+1}(2⁻²³)) − 1` (`bound_coef_unit`); the interpreter's `--weakest` rounds an f32 multiplicand to the member. An `f16` member rounds no f32 operand (D4 needs a value-range fact; none yet): it is realized only over `f16` storage. | — (semantics) |
| 2 | **Capabilities as geometry**: `Caps::matrix` = `MatrixOp`s — shape per issuing group, operand format (`MmaFmt`: tf32, bf16, f16, e4m3, e5m2), homes of `a` / `b` / accumulator (`Home::{Regs, Smem, Tmem}`), measured accumulator `acc_u`. sm_80/86: warp tf32 + 16-bit; sm_89/120: + fp8 m16n8k32; sm_90a: + `wgmma` (smem descriptors, five formats); sm_100a: `tcgen05` (TMEM accumulator, one issuing thread — modeled, not emitted). Rules read `Caps::matrix_op(fmt, home)`; `caps_of_cc` stays the device query's one table. | — |
| 3 | **The accumulator, measured** (`benches/local/mma_calib.cu`): bf16 / f16 m16n8k16 and e4m3 / e5m2 m16n8k32 all truncate — `1 + 1.5·2⁻²⁴ → 1`, `1 − 2⁻²⁵ → 1 − 2⁻²⁴`, sub-ulp addends dropped; worst 4.6 / 5.3 / 4.0 / 4.4 units of `2⁻²⁴·M` an instruction over 2²⁰ random ones. `acc_u = −23`: `{f32}` never takes a tensor core (its bound needs `2⁻²⁴`), not even two narrow operands (exact products); the program's member does. | — |
| 4 | **`half-tile`**: `mma.sync.m16n8k16.row.col.f32.bf16` (f16) in the TF32 kernel's m16n8 geometry: 16-byte `cp.async` panels in each operand's storage, `ldmatrix.x4` (`a`), `.x4.trans` / `.x2.trans` (`b` k × n) or `.x4` (`b` n × k), `cvt.rn.bf16x2.f32` of an f32 operand's fragment pairs, f32 accumulators in registers; warp rows capped at the site's, residency bounded by shared memory. Order-free face only. | `--gpu-off=half-tile` |
| 5 | **`half-fill`**: the 16-bit tile where the site's flops per operand byte reach the part's FFMA balance (`ffma_per_dram_byte_pm`) or its own grid fills the schedulers (`fill_warps`); else the FMA tile, which splits `k`. | `--gpu-off=half-fill` |
| 6 | **`narrow-radix`** (a fact arm): `tile_plan_radix` reads the exact widening of a narrow element — attention over a bf16 KV cache (`kc[…] -> widen_f32`) is a mixed-radix tile site (`narrow=b:bf16 radix=b`), not a per-cell kernel. | `--gpu-off=narrow-radix` |
| 7 | **`narrow-async`**: the FMA tile streams a narrow `b` panel in its format (8-byte `cp.async`, LDGSTS.E.64), widening each fragment as read (LDS.64 + PRMT), on the `cp.async` path rules or a table candidate pick. | `--gpu-off=narrow-async` |
| 8 | **`staged-regs`**: a register-staged FMA tile whose panel alone (`(bm + bn)·bk / threads`) exceeds the thread's registers is refused: the block decode's one-row narrow sites (516 registers, 216–244 B spilled on every rented part), the 5090's 1 × 18432 × 7168 GEMV (2512 B; its three c0 golden rows move). A first form counting panel + tile refused the B8 projection's 8 × 8 tile (132 + 128, no spill under ptxas): the panel alone separates them. | `--gpu-off=staged-regs` |
| 9 | **Fix**: a host frame holding a `bf16` / `f16` array was allocated short of it (`llt_bytes` sized `bfloat` / `half` 0): a 16 × 6144 × 4096 projection's 50 MB weight wrote past a 1 MB heap frame (segfault; base `19a8a44e` too). | — |

Programs (`gen_tiles.py`): `_bf16kv` (weights and KV cache `bf16`, each cache row rounded once as written, f32 compute),
`_bf16` (the same storage, every fold `: bf16 | f32` — torch bf16's face), prefill `_bf16w` / `_bf16`, `pt_pj_*_bf16`
(weight `bf16`, `: bf16 | f32`). `torch_bench.py`: f32 legs on the stored values, bf16 legs bf16 throughout.

## Intended vs realized (rtx-5080; `ptxas -v`, `cuobjdump -sass`, CUPTI `ktrace.cpp`)

| site | intended | realized |
| --- | --- | --- |
| 2048 × 6144 × 4096, `: bf16 \| f32` (`a` f32, `b` bf16) | HMMA bf16 m16n8k16, LDGSTS.128 panels, LDSM of `b`, F2FP rounding of `a`, no spill | `mt 4 nt 8`, 2 × 2 warps, `bk` 16 × 3 stages; 190 regs, 0 spill; 32 HMMA.16816.F32.BF16, 18 LDGSTS.E.BYPASS.128, 4 LDSM.16.MT88.4, 16 LDS.64 + 16 F2FP.BF16.F32.PACK_AB; CUPTI 1060.5 us (48.6 TFLOP/s) vs the FMA tile 2555.0 us |
| 128 × 6144 × 4096 | the same | 76 regs, 0 spill; 32 HMMA, 8 LDGSTS.128, 8 LDSM, 32 LDS.64 + 32 F2FP |
| decode attention, bf16 cache (B8 L2048) | scores / P·V tile sites reading 2-byte cache elements at the head digits | `narrow=b:bf16 radix=b` tiles; scores 125.5 → 118.4 us, P·V 65.7 → 40.2 us a step (CUPTI); OFF: per-cell kernels (B1 L8192: 588 us a call) |
| `half-fill` (16 × 6144 × 4096) | the FMA tile below the balance and the fill | 15.9 flop / B against 54.9, 96 one-warp blocks: the FMA split (36.9 us) over the 16-bit tile (50.5 us) |
| block decode gate / up / down (bf16w) | no register-staged tile past the file | 67 kernels, 0 spill (was 6 at 112–240 B) |
| intents (`evidence.py` over run 3's bundle) | `half-tile` on / off, `narrow-async`, `staged-regs`, `narrow-tile` | half-tile on 95 ok / off 114 ok; narrow-async 704 ok, 2 `no spill` (12–16 B); narrow-tile 6552 ok, 128 `no spill`; staged-regs 1498 ok, 138 misses — every miss a 4–36 B spill on an 8 × 8 register-staged tile (table candidates, as on the f32 twins in half-precision.md): the panel-only budget does not see them |
| real device, generated | `testgen::HalfProg` on the RTX 5080 (`mapal run`) against the weakest member's interpreter | 48 / 48 within D3; where the 16-bit tile ran, ≤ 1 unit of `2⁻²⁴·M` (the products the interpreter's: the accumulator's truncation alone) |

## Time (ms per program run; decode: 16 steps; median of 5 processes per arm, interleaved; value gates: 550 within, 0 outside)

Final run (run 3, bundle of `fedc07ca`): ON = shipped, OFF = the arm named. Best competitor: the fastest of torch eager /
compile / bf16 / bf16-compile and the competitor legs (Liger, FlashAttention-2, FlashInfer, Triton) at bf16, run 1
(the same programs; projections re-gated in run 2).

| workload | OFF (arm) | ON | best competitor (bf16) | ON / best | explanation (CUPTI, per call / step) |
| --- | ---: | ---: | --- | ---: | --- |
| pj 2048 × 6144 × 4096 bf16 | 2.611 (half-tile) | **1.081** | 0.887 torch bf16-compile | 1.22 | FMA tile 2555 → HMMA 1060 us; gap: the f32 `a` panel at 4 bytes leaves `bk` 16 (open) |
| pj 2048 × 4096 × 4096 / × 28672 / k 14336 | 1.711 / 12.34 / 6.011 | **0.764 / 4.649 / 2.455** | 0.594 / 3.925 / 2.095 | 1.29 / 1.18 / 1.17 | 2.2–2.7× the FMA tile |
| pj 128 × 6144 / 4096 / 28672 / k 14336 | 0.188 / 0.136 / 0.771 / 0.419 | **0.118 / 0.075 / 0.386 / 0.250** | 0.088 / 0.053 / 0.317 / 0.170 | 1.34 / 1.41 / 1.22 / 1.47 | 1.6–2.0× the FMA tile |
| pj 16 × 6144 / 28672 (half-fill) | 0.058 / 0.306 (half tile) | **0.052 / 0.292** (FMA) | 0.064 / 0.274 | 0.82 / 1.06 | below the balance, under-filled: the split FMA tile |
| prefill T2048 bf16 | 9.467 (half-tile) | **5.886** | 1.934 bf16-compile | 3.04 | QKV 2545 → 1067, O 2158 → 804 us; attention (radix FMA tiles 2067 + 1159 us, flow 1303 us) unchanged — the 16-bit tile over radix sites is open; bf16w (f32 compute) 8.467 |
| block prefill T2048 bf16 | 29.31 (half-tile) | **13.53** | 8.034 | 1.68 | gate / up / down on HMMA as well; bf16w 27.55 |
| decode B1 L2048 bf16kv | 5.076 (narrow-radix) | **2.405** | 2.279 bf16-compile | 1.06 | bf16w 2.358 (was 1.04): at B1 the cache is 4 MB a step, the projections dominate |
| decode B1 L8192 bf16kv | 14.07 | **3.651** | 3.011 | 1.21 | bf16w 3.889 (1.30) |
| decode B8 L2048 bf16kv | 9.354 | **4.700** | 3.451 | 1.36 | bf16w 5.427 (1.60): P·V 65.7 → 40.2, scores 125.5 → 118.4, O 41.1 → 16.4 us (its 33.5 MB L2-resident beside the half-size cache) |
| decode B8 L8192 bf16kv | 30.75 | **11.94** | 7.601 | 1.57 | bf16w 15.64 (2.10) |
| decode B32 L2048 bf16kv | 27.00 | **12.06** | 7.306 | 1.65 | bf16w 15.29 (2.10) |
| decode B32 L8192 bf16kv | 100.67 | **40.36** | 22.30 FlashInfer | 1.81 | bf16w 54.86 (2.46) |
| block decode B1 L4096 bf16kv | 15.33 | **10.29** | 8.834 | 1.17 | bf16w 10.23 |
| decode `_bf16` (bf16 face) | — | = `_bf16kv` (±0.5 %) | | | decode sites stay below the balance at B ≤ 32: the FMA tile; B32 half-fill OFF 13.71 vs 12.07 |
| decode B8 / B32 bf16w (narrow-async) | 5.489 / 16.51 / 16.09 / 58.36 | **5.427 / 15.64 / 15.29 / 54.86** | | | B8 L2048 / L8192, B32 L2048 / L8192: −1.1 / −5.3 / −5.0 / −6.0 %; B32 O 51.7 → 46.1 us (the streamed pick); the rest of the B32 difference (~44 us a step) is not in the top five sites: open |
| staged-regs (B1 decode f32 / bf16w, block decode) | 3.911 / 2.500 / 17.26 | **3.842 / 2.358 / 17.13** | | | −1.8 / −5.7 / −0.7 %: the spilling candidates out of the tables; kvn B1 L8192 −18 %, L2048 −8.5 %; causal / prefill −0.5 / −0.4 %; decode_1024, gen null; pj m1 rows: the same pick in the timed region (14.2 us, CUPTI) — the 0.029 / 0.032 ms difference is host time (null 0.0326) |

**What moved, and why.** The bf16 KV cache alone takes 13–26 % off B ≥ 8 decode (the attention sites read half the
bytes, and the projections' weights stay L2-resident beside the smaller cache): torch bf16's lead at B ≥ 8 drops from
1.6–2.5× to 1.36–1.81×. The 16-bit tensor-core tile is 1.6–2.7× Mapal's FMA tile on every projection at m ≥ 128 and
the prefills' projections, 1.17–1.47× cuBLASLt bf16. The remaining decode gap is the scores site (118 us at B8 L2048:
the radix tile copies its 2-byte elements one at a time — no vector copy along a contiguous `k` digit), and torch's
decode runs FlashInfer / fused SDPA with bf16 activations; the prefill gap is attention on the FMA tile.

## Tests

- `half_tile.rs` (4): the member's facts and bound; the realization per arm on the 5080 (PTX `mma.sync…bf16`,
  `ldmatrix…trans`, `cvt.rn.bf16x2.f32`, `cp.async…16`; OFF / exact face: none; `half-fill`: the skinny GEMV refused,
  a 256 × 512 × 512 site taken; every part with the warp bf16 product takes it); `{f32}` over two bf16 operands refused
  (truncating accumulator) and taken under `: bf16 | f32`; **48 generated programs** (`testgen::HalfProg`: GEMMs
  16–128 × 8–128 × 16–256 and attention heads 16–64 × 16–64, one or two heads; f32 / bf16 / f16 storage; `{f32}`,
  `: bf16 | f32`, `: f16 | f32`; `b` k × n or n × k; epilogues) on the emulated device in every mode
  (`MAPAL_EMU_SERIAL=0/1/2`, `MAPAL_EMU_REVERSE`) and under OFF, within D3 composed through the softmax (ADR-0043 D3).
  Coverage: half kernels 24, `a` rounded 10 / narrow 13, `b` rounded 6 / narrow 14, n × k 9, epilogue 9, attention on
  HMMA 6, f16 kernels 2, narrow radix 6, FMA fallback 17. The same 48 on the real 5080: 48 / 48 within the bound.
- `narrow.rs`: the forced `cp.async` arm on the exact face — interpreter bytes in every mode (8 streamed narrow tiles
  over the 64 programs); the bf16-cache decode twin's narrow radix sites (exact bytes, OFF none).
- `profile.rs` `matrix_instructions_are_geometry`, `func/mod.rs` `narrow_elements_are_two_bytes`,
  `compute_type.rs` `narrow_members`; `horizontal.rs` / `zero_fold.rs` / `c0-rules.tsv`: the `staged-regs` moves.
- Emission identity: all 170 generated `pt_*` programs + 2 cells × 5 faces emit byte-identical modules against
  `19a8a44e` except the 28 the two rules move (24 `staged-regs`, 4 `narrow-async`), all timed above.

## Open

1. **fp8 storage types** (`f8e4m3`, `f8e5m2`, saturating `to_f8*`, the program's scale) and their `m16n8k32` tile: the
   instructions are modeled and their accumulators measured (truncating, `acc_u = −23`); the types are not implemented.
2. The 16-bit tile over **mixed-radix sites** (attention scores / P·V with head digits) and **split-k** (the skinny
   sites below the balance); **swap-AB** for sites under 16 rows.
3. An f32 `a` panel **rounded to bf16 as it is staged** (half the shared memory: `bk` 32 at 2048 rows, `ldmatrix`).
4. The radix tile's **vector copies** of a contiguous `k` digit (the bf16-cache scores site's 2-byte scalar copies).
5. `wgmma` 16-bit emission (sm_90a) and `tcgen05` (sm_100a): modeled, not emitted; the H100's fp8 `wgmma`
   accumulator is unmeasured here (published ~14 bits) — `mma_calib.cu` on the part.
6. 8–24 B spills on 8 × 8 register-staged tiles the panel-only budget admits (table candidates); the unexplained
   ~44 us a step of B32 L2048 `narrow-async`.
7. A CLAUDE.md facts-table row for `Caps::matrix` / the members (left to the maintainer: agents do not edit it).

Raw (local, not in git): `.claude/raw-evidence/local-rtx-5080/tensor-cores/` (`mma_calib.txt`, `run1`–`run3` results,
evidence tables, conditions, commands; `kt/` CUPTI traces; `gen_device2.txt` the real-device generated check;
`tools/` the summaries).
