# local-rtx-5080 — half precision: `bf16` / `f16` storage, the decode rows with bf16 weights (2026-11-22)

Branch `local-rtx-5080` @ `3b404428`, RTX 5080 (sm_120, 84 SMs, 64 MB L2, ~960 GB/s), CUDA 13, clang/LLVM 23, PyTorch 2.11.0+cu130.
Decision: [ADR-0043](../../../../docs/decisions/ADR-0043-narrow-float-storage.md) (bf16 / f16 / fp8 storage, conversions,
semantics and bounds, CPU / GPU lowering, cost model; slice 1 implemented, slices 2–3 open).

## What changed (slice 1: storage with f32 compute)

| # | fact / rule | OFF arm |
| --- | --- | --- |
| 1 | **Types** `bf16`, `f16` (`Ty::Narrow`, `Value::Narrow`): storage only — no arithmetic, comparison or print; arrays, loop state, guards, updates move them. **Builtins** `x -> to_bf16` / `to_f16` (`Operation::Narrow`: one rounding of an `f32`, ties to even, subnormals kept, `±inf` past the threshold; L1616 otherwise) and `widen_f32` / `widen_f64` from a narrow value (exact). Oracle `mapal_ir::narrow` (`round`, `widen`, `NarrowForm`); the rewriter folds constants with it. | — (semantics) |
| 2 | **Realizations, bit for bit on every face**: host `to_bf16` = the integer sequence `(b + 0x7fff + lsb) >> 16` (AVX512-BF16's `vcvtneps2bf16` flushes subnormal inputs; `__truncsfbf2` is a call), host `to_f16` = `fptrunc` (F16C `vcvtps2ph`), device both `fptrunc` → `cvt.rn.bf16.f32` / `cvt.rn.f16.f32`; widening `fpext` (`SHF.L.U32` / `cvt.f32.f16`). CUDA backend `__nv_bfloat16` / `__half`, `__float2bfloat16_rn` / `__float2half_rn`. | — |
| 3 | **`tile_plan_narrow`** (mapal-ir, stored fact): sum-of-products sites whose operand is the exact widening of a narrow element, strided lanes and epilogues included, each read carrying its format (`TileRead::narrow`, `elem_bytes`); disjoint from the other tile plans. `precision_plan.narrow` (each rounding's `u`, `abs`, `max`); `bound` covers the narrow sites (the f32 site's `γ_{K+1}`). | `--gpu-off=narrow-tile`: such a map is a per-cell kernel |
| 4 | **Device realization**: the one-thread-per-cell kernel (`naive`, split / atomic / epilogue forms) loads the operand in its format (`LDG.E.U16`) and widens it; the **register-staged FMA tile** widens each panel as it is copied (`<4 x bfloat>` = `LDG.E.64`, `fpext`, `STS.128` of f32), so every fragment read and FFMA is the f32 tile's. `cp.async` (moves bytes unconverted) and the TF32 `mma` / `wgmma` tiles refuse a narrow operand. | the same arm |
| 5 | **Cost model**: operand bytes per storage format — `PlanShape.{ea, eb}` (L2 panel term, DRAM term), `profile::site_bytes` (placement roofline), `row_split_pays` on `b`'s bytes, `residence_plan` / `unit_plan` / `elem_bytes` (2). The plan model's picks for the B1 QKV / O sites: 113.9 / 76.1 us (f32) → 57.6 / 38.6 us (bf16) — measured 114.2 / 75.2 → 60.9 / 38.8 us. | — |
| 6 | Kernel ids are unchanged for every program without a narrow operand: `TileRead`'s `Debug` omits `narrow: None`, `launch_key` drops the host rung. **All 251 `benches/algorithms` programs emit byte-identical modules** (743 module texts: rtx-5080 host + device, generic host) against HEAD `7a2b67bf` — no suite row can move. | — |

Programs: `gen_tiles.py` `w16` (`LLM_WEIGHTS`: wqkv, wo, wg, wu, wd stored `-> to_bf16`, read `w[i] -> widen_f32 -> wv`),
stems `pt_llm_dec_b{1,8,32}_l{2048,8192}_bf16w`, `pt_llm_blkdec_b1_l4096_bf16w`, twins `px_llm_{dec,blkdec}_b2_l16_bf16w`;
`torch_bench.py` builds the same weights (`.to(bfloat16)`): its f32 legs compute on the bf16 values (like for like), its
`bf16` / `bf16-compile` legs are bf16 throughout (weights, activations, KV cache) — the real-world competitor.

## Intended vs realized (rtx-5080; `ptxas -v`, `cuobjdump -sass`, CUPTI `ktrace.cpp`)

| | intended | realized |
| --- | --- | --- |
| narrow GEMV (B1 QKV, `naive` splitk 8) | the weight read in 2 bytes, widened in a register, the f32 FFMA chain; half the DRAM bytes | 38 regs, 0 spill (f32: 40); hot loop `LDG.E.U16.CONSTANT` 36 + `LDG.E.CONSTANT` 36 (x) + `SHF.L.U32` + 36 `FFMA` (f32: 128 `LDG.E` + 64 `FFMA`); CUPTI 114.2 → 60.9 us (100.7 → 50.3 MB: 881 → 827 GB/s) |
| narrow O + residual (`naive` post, splitk 8) | the same, epilogue unchanged | 75.2 → 38.8 us (67.1 → 33.6 MB: 892 → 865 GB/s) |
| narrow staged tile (B8 / B32 QKV, O) | panel copies of `<4 x bfloat>`, widened before `STS`; no `cp.async`; fragments and FFMA the f32 tile's | `LDG.E.64` + `PRMT`/`SHF` + `STS.128`, 0 `LDGSTS` (f32 picks `LDG.E.128` / `LDGSTS`); B32 QKV tm 8 × tn 8, 128 regs, 0 spill; CUPTI B8 QKV 120.8 → 63.2, O 79.4 → 42.1 us; B32 QKV 125.1 → 76.2, O 79.3 → 51.8 us |
| intent `narrow-tile` (`intent.py`, `evidence.py` `ldg16` / `ldg64`) | every kernel of a narrow site: format loads in the hot loop, register-staged, f32 products, no spill | bundle 1: 368 / 368 ok; bundle 2 (staged tile): 767 ok, 25 `no spill` mismatches — table candidates (whole-file / deep-panel tiles, 8–240 B), none picked; the f32 twins' candidates spill alike (32 rows) |
| conversions | `cvt.rn.bf16.f32`, `cvt.rn.f16.f32`, `cvt.f32.bf16`, `ld.global.nc.b16` in PTX; host: no libcall | `realization_per_side` asserts each; the host's `to_bf16` is the integer sequence (vectorizes), `to_f16` `vcvtps2ph` |
| movement | weights device-resident, uploaded never (made by `gen` kernels on the device) | `h2d=2 d2h=4` per run as f32 (the stores are 2-byte `gen` cells) |

## Time (ms per decode step = 16-step `iter ms` / 16; median of 5 processes per arm, interleaved; `null` ≤ 0.3 %)

Mapal f32: the f32 program (run 1). bf16w OFF: `--gpu-off=narrow-tile`. bf16w ON: the shipped build (run 2, after the
register-staged narrow tile). torch bf16: the faster of `bf16` / `bf16-compile` (run 1). Ratio: ON / torch bf16.

| workload | Mapal f32 | bf16w OFF | bf16w ON | ON / f32 | torch bf16 (leg) | ratio | explanation (CUPTI, per step) |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| decode B1 L2048 | 0.2391 | 0.5359 | **0.1494** | 0.62 | 0.1428 (compile) | 1.05 | QKV 114.2 → 60.9, O 75.2 → 38.8 us; the rest unchanged (RMSNorm one block 17.2 us, attention 30 us); torch: fused RMSNorm+QKV GEMV 68.8, O 46.7 us — Mapal's projections are faster, its RMSNorm (`grid=1`) and attention on an f32 cache are the gap |
| decode B1 L8192 | 0.3314 | 0.6653 | **0.2423** | 0.73 | 0.1851 (compile) | 1.31 | the same −89 us of projections; attention reads a 67 MB f32 cache (torch: 34 MB bf16) |
| decode B8 L2048 | 0.4337 | 0.7429 | **0.3413** | 0.79 | 0.2169 (compile) | 1.57 | QKV 120.8 → 63.2, O 79.4 → 42.1 us (staged tile); QKᵀ cell 126.8 + P·V 80 us unchanged |
| decode B8 L8192 | 1.0644 | 1.4206 | **0.9745** | 0.92 | 0.4821 (eager) | 2.02 | attention over the f32 cache dominates |
| decode B32 L2048 | 1.0331 | 2.1668 | **0.9569** | 0.93 | 0.4631 (compile) | 2.07 | QKV 125.1 → 76.2, O 79.3 → 51.8 us; QKᵀ 477 + P·V 312 us of 959 (K read 4× per KV head, f32 cache) unchanged |
| decode B32 L8192 | 3.5195 | 4.8233 | **3.4370** | 0.98 | 1.486 (compile) | 2.31 | attention |
| block decode B1 L4096 | 1.0708 | 1.9100 | **0.6369** | 0.59 | 0.5566 (compile) | 1.14 | + gate / up / down at half the bytes |

**The mixed result and its missing realization.** Run 1 shipped the narrow operand on the one-thread-per-cell kernel
only: B1 rows won as above, B8 and B32 lost (B8 L2048 0.4496, B32 L2048 1.735, B32 L8192 4.219 ms against the f32
program's 0.4337 / 1.033 / 3.52): CUPTI B32 QKV 445 us, O 459 us against the f32 tile's 125 / 79 — one thread a cell
reads each weight 32 times. What separated winners from losers was the realization, not the format: the f32 rows take
the staged tile from 8 rows up. The register-staged tile widening its panels (row 4 above) is that realization; run 2
has every row at or below the f32 program. OFF (per-cell maps) is slower than both everywhere.

Value gates: XREF (the exact face on the device: the interpreter's order bit for bit) — every arm of every row within
ADR-0040 D2 (224 `value-bound … within`, 0 outside); small twins against the interpreter: run 1 17 / 17, run 2 14 / 14;
torch f32 legs on the bf16 weights within 3.3·10⁻⁸ – 3.6·10⁻⁷ of Mapal (the same function), bf16 legs 2.4·10⁻³ –
1.7·10⁻² (bf16 activations and cache). Conditions: bench lock, no game, CPUs 0–7 idle and pinned, Tctl 34–66 C (run 1
ended at load 6.5 while bundle 2 was built on CPUs 8–63; its nulls 0.0–0.3 %). Raw (local, not in git):
`.claude/raw-evidence/local-rtx-5080/half-precision/` (`commands.txt`, run1, run2, trace, static).

## Tests

- `mapal-ir` `narrow.rs` (4): every one of the 2 × 65536 narrow values round-trips; the host's `bf16` integer sequence
  is the oracle on every 97th `f32` (4.4·10⁷ values); F16C `vcvtps2ph` / `vcvtph2ps` are the oracle on every 13th `f32`
  (3.3·10⁸); ties, overflow thresholds, subnormals, the single rounding from `f64`.
- `mapal-lower` `narrow_types.rs` (5): L1616 / L1203 / L1614, no arithmetic / comparison / print on narrow, reserved
  names, the formatter, `Narrow` / `Widen` edges. `mapal-interp` `narrow.rs` (3): the oracle's bytes through arrays,
  updates, guards, loops; rounding edges.
- `mapal-backend-llvm` `narrow.rs` (3): the facts (site, format, bytes, disjointness, `precision_plan.narrow`, the
  f32 bound), the realization per side and the OFF arm (PTX `cvt.rn.bf16.f32`, `cvt.rn.f16.f32`, `cvt.f32.bf16`,
  `ld.global.nc.b16`), and **64 generated programs** (`testgen::NarrowProg`: bf16 / f16, rows 1–32, model / large /
  overflowing / subnormal weights, both operands narrow, strided lanes, epilogues, guards, a loop carrying a narrow
  array with an element update, a whole fold over widened values): exact face = interpreter bytes on the host and the
  emulated device under `MAPAL_EMU_SERIAL=0|1|2` and `MAPAL_EMU_REVERSE`; default face within D2 (sums) / D3 (5 ulps of
  the format past the roundings) in every mode and under the OFF arm. Coverage (each required > 0): f16 35, bf16 29,
  narrow sites 64, both-narrow 25, strided 20, epilogue 33, device naive narrow kernels 42, **staged narrow tiles 31**,
  narrow cells 64, inf 5, NaN 13, subnormal 15, guards 30, loops 29.
- `mapal-backend-cuda` differential: 6 generated narrow programs raw and rewritten through `nvcc`, byte for byte.

## Open

1. **Tensor-core 16-bit products (ADR-0043 slice 2)**: `mma.sync.m16n8k16.f32.bf16` (16-byte `cp.async` panels of 8
   elements, `ldmatrix(.trans)` `.b16`), admitted under `{f32}` only where `Caps::mma_acc` (the accumulator's measured
   rounding) meets `2⁻²⁴`, else under `: bf16 | f32`; `Caps::mma16` as a capability property. Not started.
2. **fp8 (slice 3)**: `f8e4m3` / `f8e5m2`, saturating conversions, `mma.m16n8k32` — not started.
3. **bf16 KV cache / attention**: the remaining gap at L8192 and B ≥ 8 is attention over an f32 cache (torch reads
   bf16): a `_bf16kv` program (cache stored `-> to_bf16`) reaches the narrow strided sites (QKᵀ) and cells already;
   not measured.
4. `cp.async` for narrow panels (a 16-bit shared-memory panel and a widening fragment read) — the register-staged
   tile is what ships; 25 table candidates spill (8–240 B), never picked.
5. An `f64` source for `to_bf16` / `to_f16` (single rounding: round-to-odd to f32 first); host AVX512-BF16 / AMX
   compute (a host capability property) under `: bf16 | f32`.
6. RMSNorm's one-block kernel (17 us a step, `grid=1`) is now 11 % of the B1 step — pre-existing (`model-scale.md`).
