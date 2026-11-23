# sweep-gemm — open problems in GEMM cells / projections / MLP chains / GEMVs / tensor tiles (RTX 5080, 2026-11-23)

Branch `sprint/sweep-gemm` from `local-rtx-5080` @ `a31cd014` (the integrated head): `62bb645b` (round-store), `3ac6a386` (FHFMA in the evidence tooling), this file. RTX 5080 (sm_120, 84 SMs, 64 MB L2),
CUDA 13, LLVM 23. Raw (not in git): `.claude/raw-evidence/local-rtx-5080/sweep-gemm/` (`rs/` the projections' probe run
and evidence, `rs2/` the first LLM run, `rs3/` the final LLM run and evidence, `suite-diff-188.tsv`, `tools/`).

**Run in parallel (the user asked for speed).** Several agents shared the box: builds and tests on CPUs 16–23 / 48–55
(`CARGO_BUILD_JOBS=12 RUST_TEST_THREADS=8 MAPAL_TESTGEN_JOBS=4 MAPAL_PAR=2`, `nice 5`, tests stopped above 85 C Tctl),
the shared bundle cache, every timed batch one hold of the bench lock (4–6 min). The machine was loaded (load 18–63
from the other agents) during every timed run: each A/B is same-session ABBA with a `null` arm (the ON binary again),
and the decisions read ratios, not the absolute times.

## Ranking (cost = Σ over parts of ln(Mapal / best) for the row class, `matrix.md`)

| class | Σ ln ratio (5 parts) | rows | 5080 cells | what limits it | status here |
| --- | ---: | ---: | --- | --- | --- |
| m128 bf16 projections | 9.99 | 4 | 1.21–1.46 | the f32 `a` panel: 4-byte smem rows, LDS + F2FP a fragment pair, `bk` 16 | **closed generic (round-store)** |
| m1 f32 GEMVs | 9.18 | 6 | 1.04–1.89 | per-call host time in the timed region + the split's grid | attempted, open (§3) |
| m2048 bf16 projections | 9.03 | 4 | 1.09–1.22 | the same f32 `a` panel | **closed generic (round-store)** |
| m128 TF32 projections | 8.27 | 4 | 1.18–1.65 | the TF32 tile (= the 5080's FFMA rate) depth / split | not reached (§4) |
| m16 bf16 projections | 7.49 | 4 | 0.93–1.23 | the GEMV (FMA) path over an f32 `a` | **closed generic (round-store)**, smaller |
| m16 f32, m2048 TF32, m1 bf16, m128 f32, mlp_512, MLP TF32 | ≤ 4.4 each | | | | not reached |

## 1. `round-store` — an f32 array read only by `bf16` products is stored rounded by its producer (closed, generic)

The `_bf16` face (`fold: bf16 | f32`, torch bf16's face) rounds each multiplicand to bf16. An f32 activation `a`
that only such products read was rounded **at every fragment read** of the 16-bit tile (`ea=4`: an f32 smem panel,
LDS.64 + F2FP per fragment pair, `bk` 16 at 2048 rows) — the open item of tensor-cores.md (3), hopper.md (4: "R4
rounding at the producer, `round_at` has no consumer"). ADR-0041 D5's `round_at` covered scalars only.

| # | fact / rule | OFF arm |
| --- | --- | --- |
| F | **`precision_plan.store`** (`NarrowStore`): an f32 array a `Map` produces (its body's one f32 result) whose every element read — through the captures of nested maps and folds (`Pair` → `Index`, `Map`/`Fold` capture → body `Proj`) — is a multiplicand of a `Mul` admitting `bf16`; any other use (print, call, iteration, an f32 product, an epilogue's own read, an `f16` member: D4 has no value-range fact) keeps it f32. Carries the array's carriers (tuples, body inputs, projections) and reads. | — (fact) |
| R | **`CategoryIr::narrow_stored`**: the graph with those arrays stored `bf16` — the producer's body ends in `Narrow`, every carrier's type names the narrow array, every read widens (`Widen`) before its products: the same `bf16(x)` each product's member took at the read, computed once. **Guard (a capability as data):** each array is taken in turn and kept only where every map keeps every site plan it was on (the affine tile sites — strided, epilogue and narrow plans as one class —, the mixed-radix sites, row / whole folds, cells, row-block unit groups, and a streamed group's members with f32 operands — the flash stream kernel takes no narrow element yet). Facts re-derived on the new graph; idempotent. | — |
| U | **`round-store`** rung: `emit_detailed` takes `narrow_stored` on the face that realizes members (order free); exact / fixed / sequential faces keep the oracle's f32 products. | `--gpu-off=round-store` |

The guard is what the first runs found: without it, rounding the decode's softmax `p` at its producer demoted the
`P·V` radix tile to a per-cell kernel (dec B1 L2048 4.31 vs 2.25 ms), and rounding the prefill's `q` / `k` took the
scores out of the flash stream (5.65 vs 3.95 ms). Both arrays now stay f32; the rest of those programs' arrays are
stored.

Changed programs (`suite_diff.py --off round-store` over all 188 committed `pt_*` programs): exactly the 25 `_bf16`
face programs (16 projections, 9 LLM rows); 163 identical. The 16 projections' device text is the same with and without
the guard (checked after it landed).

### Intent → realized (`ptxas -v`, `cuobjdump -sass`, sm_120; `evidence.py` over both bundles)

| site | intent | realized |
| --- | --- | --- |
| 128 × 4096 × 4096 `: bf16 \| f32` | both panels 2-byte (`narrow=a:bf16+b:bf16 ea=2`), LDGSTS.128, LDSM fragments, no conversion a fragment, no spill | `mt 2 nt 2`, `bk` 64 × 2 (was `bk` 32 × 3, `ea=4`); hot loop 48 HMMA / 36 LDSM / 12 LDGSTS / 9 LDS in 151 instructions (was 8 HMMA / 2 LDSM / 19 LDS in 90: 11 → 3.1 instructions an HMMA); 48 registers (was 62), 0 spill |
| 128 × 28672 × 4096, 2048 × 6144 × 4096 | the same | `mt 4 nt 8`, `bk` 32 × 2 (was 16 × 3): 64 HMMA / 16 LDSM / 8 LDGSTS / 3 LDS (was 32 / 4 / 6 / 19); 218 registers (190), 0 spill |
| GEMV / m16 FMA sites (`naive`, `tile`) | f32 products of the two widened operands | FHFMA.BF16 (sm_120's f32 FMA over 16-bit multiplicands: the exact widening fused) — 64 a hot loop, 8 LDG.128 |
| intents | `round-store:on` (new, `intent.py`) | 42 ok (projections) + 12 ok (LLM rows), 0 mismatch; every other mismatch class is one the OFF arm has too |
| evidence tooling | FHFMA counted as the f32 FMA it is (`sass_stats.split`) | the 92 `face-fma` / `narrow-tile` / `vec-cells` "no FFMA" mismatches it raised are gone (projections bundle 586 → 494 mismatches, all in the OFF arm's classes) |

### Time (ms per program run, same-session ABBA, tuned pick; value gates first: 25 / 25 programs pass on every arm)

Best competitor from `matrix.md`'s rtx-5080 column (bf16 legs). The session's absolute times run ≤ 5 % above the
matrix's (a loaded box); the OFF/ON ratio is the decision.

| workload | OFF | ON | OFF/ON | fastest competitor (leg) | ON / best | explanation |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| pj_m128_n4096_k4096_bf16 | 0.0754 | **0.0640** | 1.18 | 0.0532 (torch-bf16-compile-ro) | 1.20 (was 1.45) | 2-byte `a` panel: `bk` 64, LDSM, no F2FP |
| pj_m128_n4096_k14336_bf16 | 0.2515 | **0.2221** | 1.13 | 0.170 (torch-bf16-compile-cg) | 1.31 (1.46) | as above |
| pj_m128_n6144_k4096_bf16 | 0.1051 | **0.0947** | 1.11 | 0.0881 (torch-bf16) | 1.07 (1.22) | as above, cut S = 8 |
| pj_m128_n28672_k4096_bf16 | **0.3914** | 0.4037 | 0.97 | 0.3174 (triton-bf16) | 1.27 (1.21) | **loss**: the panel rule's `bk` 32 × 2 over the 2-byte panels against `bk` 16 × 3 (§2) |
| pj_m2048_n6144_k4096_bf16 | 1.079 | **0.938** | 1.15 | 0.8865 (torch-bf16-compile-ro) | 1.06 (1.22) | `bk` 32, LDSM |
| pj_m2048_n4096_k4096_bf16 | 0.718 | **0.696** | 1.03 | 0.598 (torch-bf16-compile-cg) | 1.16 (1.19) | |
| pj_m2048_n4096_k14336_bf16 | 2.282 | **2.253** | 1.01 | 2.097 (torch-bf16-compile-cg) | 1.07 (1.09) | |
| pj_m2048_n28672_k4096_bf16 | 4.700 | **4.574** | 1.03 | 3.963 (torch-bf16-compile-cg) | 1.15 (1.18) | |
| pj_m16_n6144_k4096_bf16 | 0.0602 | **0.0564** | 1.07 | 0.0635 (torch-bf16-compile) | 0.89 | FMA split reads a 2-byte `a` (FHFMA) |
| pj_m16_n28672_k4096_bf16 | 0.2929 | **0.2826** | 1.04 | 0.2744 (torch-bf16-compile) | 1.03 | |
| pj_m16_n4096_k4096_bf16 | 0.0609 | 0.0588 | 1.04 | 0.043 (torch-bf16-compile) | — | kernel 0.039 / 0.038: null; 12 reps ABBA |
| pj_m16_n4096_k14336_bf16 | 0.1269 | 0.1334 | 0.95 | 0.1393 (torch-bf16-compile) | — | kernel 0.111 / 0.112 ms: null (host side) |
| pj_m1_n6144_k4096_bf16 | 0.0380 | **0.0320** | 1.19 | 0.0635 (torch-bf16-compile) | 0.50 | `a` 8 KB → 4 KB a cell row |
| pj_m1_n4096_k4096_bf16 | 0.0316 | **0.0273** | 1.16 | 0.0471 (torch-bf16-compile-cg) | 0.58 | |
| pj_m1_n4096_k14336_bf16 | 0.1057 | 0.1025 | 1.03 | 0.1413 (torch-bf16) | 0.73 | |
| pj_m1_n28672_k4096_bf16 | 0.2636 | 0.2567 | 1.03 | 0.2744 (torch-bf16-compile) | 0.94 | |
| llm_pre_t2048_bf16 | 3.953 | **3.805** | 1.04 | 1.932 (torch-bf16-compile-cg) | 1.97 (2.04) | QKV / O read the stored norm and attention output; flash stream kept (guard) |
| llm_blkpre_t2048_bf16 | 11.63 | **10.75** | 1.08 | 8.043 (torch-bf16-compile-cg) | 1.34 (1.44) | the FFN's `h` and the norms stored |
| llm_blkdec_b1_l4096_bf16 | 9.860 | **9.280** | 1.06 | 0.5556 /step (torch-bf16-compile) | 1.04 /step (1.11) | GEMVs over stored activations |
| llm_dec_b1_l8192_bf16 (16 steps) | 3.617 | **3.172** | 1.14 | 0.1814 /step | 1.09 /step (1.10) | |
| llm_dec_b1_l2048_bf16 | 2.253 | 2.224 | 1.01 | 0.1455 /step | 0.96 /step | |
| llm_dec_b8_l2048_bf16 | 3.951 | **3.857** | 1.02 | 0.2221 /step | 1.09 /step | |
| llm_dec_b8_l8192 / b32 l2048 / b32 l8192 _bf16 | 11.20 / 10.51 / 38.32 | 11.20 / 10.47 / 38.25 | 1.00 | | | attention-bound: null |

`null/on` (the ON binary run as its own arm) 0.96–1.02 on every row but the m16 GEMVs' 0.87–0.96 (process noise on
a loaded box: their kernels are equal, the rows are marked null).

### Tests

- `mapal-lower` `compute_type::precision_plan_store_narrow`: the fact on a GEMV (one store; the rewritten graph
  `validate`-clean, idempotent, one `Narrow` and one `Widen` more, a narrow array type) and its refusals (a print, an
  f32 product, `: f16 | f32`, an elementwise reader, a `{f32}` fold's product).
- `half_tile.rs` **`round_store_on_generated_programs`**: 40 generated GEMMs (`testgen::half_src_with(p, true)`: the
  magnitudes read their own copies of the operands; dims, laws, `b` k×n / n×k, epilogues drawn; formats / sets leaned
  to f32 operands under `bf16 | f32`) — the fact per operand (the epilogue's own read of `a` keeps it f32), the
  rewritten graph valid and idempotent, values within D3 from both sides against the weakest member's interpreter in
  all four emulator modes and under the OFF arm (equal text where nothing is stored). Coverage: 40 programs, 32 arrays
  stored, 6 `a` / 10 `b` panels 2-byte on the 16-bit tile, 12 stored operands read by the FMA tile, 2 refused, 6 with
  epilogues — all non-zero. `realization_per_arm` now pins both arms (stored `ea=2`; OFF the per-fragment rounding).
- Workspace subset (`mapal-ir`, `mapal-lower`, `mapal-rewrite`, `mapal-backend-llvm`, 89 test binaries): **998
  passed, 1 failed** — `round_store_on_generated_programs`, where the first guard counted a site moving from the
  strided plan to the narrow plan (its operand now stored) as a lost class; the guard now takes the affine tile plans as
  one class, the rerun passes (5 / 5, the counters above; `generated_programs_within_the_member_bound`'s 15 counters
  unchanged: `half_src` emits the same text). The 25 changed programs' device text is identical under both guards.

**Other parts.** No hardware property enters `round-store` (a fact rewrite on the order-free face), so it has no
`rung_replay.tsv` line; every part with the warp bf16 product reads the same 2-byte panels (`realization_per_arm`
checks the 3090 / A100 / 4090 / 5090 / H100 profiles' 16-bit tile). The rented rows (A100 1.75–2.94×, H100 2.0–3.8×
on bf16 projections) are measured at the next round; the H100's gap is mostly the missing `wgmma` 16-bit emission
(hopper.md), which a stored `a` also makes reachable (both operands narrow).

## 2. The 2-byte panels' depth (attempted, open — the one row round-store moved slower)

`pj_m128_n28672_k4096_bf16` ON 0.4037 vs OFF 0.3914 (12-rep ABBA, kernel 0.396 / 0.382). The tile is the same
(128 × 128, 224 blocks, two an SM); the panel rule (`gpu_mma_panels`: the deepest `bk` two stages fit, S52) now
fits `bk` 32 × 2 where the 4-byte `a` left `bk` 16 × 3. Forced arms (`--geom=bk=16,stages=3|4`, ABBA 10 reps): `bk`
16 × 3 0.3628, **16 × 4 0.3431** (0.88× OFF, 1.08× triton-bf16). `mma-depth` (the most bytes in flight) picks 16 × 4
there but loses on the other 16-bit sites (ABBA 8 reps, ON vs ON + `--gpu-on=mma-depth`): 128 × 4096 × 4096 0.0648 /
0.0682, × 14336 0.2228 / 0.2241 (null), 2048 × 4096 × 4096 0.696 / 0.736, 2048 × 6144 0.938 / 1.069 — the same split
TF32 showed (f32-tensor-cores.md). What the five points separate: the winner streams 235 MB from DRAM over a 1.33-wave
grid (its last third runs one block an SM: no second block covers a barrier); the losers are L2-resident (32 MB), a
3-wave cut (0.4 % tail) or compute-bound (m2048). The fact is the tail wave's residency × the operand's residence
(`residence_plan`) against the stage depth; five points do not pin a threshold, so no rule ships.

## 3. m = 1 f32 GEMVs (attempted, open)

`pj_m1_n4096_k4096` 1.89× (0.0388 vs cuBLASLt-fresh's 0.0205), `n6144` 1.74×, `n2048_k7168` 1.36×. The timed region of
the one-call program (`MAPAL_GPU_TRACE`): the output's zero launch (`op=zero`, 16 KB), the split kernel (rules: 4
cells a thread, `S` = 32, 128 blocks, atomic combine), the 4-byte copy back — the library leg is CUDA-event time of
its kernel alone. Checked here:

- **gemv-tma open 3 (the L2 knee)** does not bind at today's rule: `calib.cu bw` (`B-bw-rtx-5080.tsv`) puts the L2's
  90 % point at 32 KiB in flight an SM (DRAM 8 KiB); the rules' plan already keeps 49.9 KB an SM in flight
  (`fly_sm=49932`). A property for it (the L2 knee) would not move this row.
- Forced splits (`--splitk=S`, ABBA 10 reps, traced kernel ms — the box loaded by the other agents, host times 15–40 %
  above the matrix's): S 8 0.037, 16 (1 cell, 256 blocks) 0.030, 32 (rules) 0.033, 64 (4 cells, 256 blocks) 0.029,
  128 0.041. 256 blocks look ~10 % faster than 128, inside this session's noise; no rule follows from it.
- What is left is the call's shape, not the kernel: the zero launch and the synchronize a one-call program pays and
  an event-timed library kernel does not (regressions.md § 4's per-call host time, the open property
  `dep_launch_ns` measures device-side only). Open: an in-block split (threads over `k` slices, a shared-memory tree,
  one launch, no zero) — cuBLAS's gemv form — is not a Mapal tile form today.

## 4. Not reached in this sweep

- **m128 TF32** (1.34–1.65 on the 5080, the 5080's TF32 rate is its FFMA rate): cuBLAS's `s1688gemm` 128 × 128 × 32
  × 3 at one block an SM needs XOR-swizzled 96 KiB panels (f32-tensor-cores.md); a TF32 `round-store` (operands
  rounded to TF32 once at the producer, the per-fragment `cvt.rna` gone) needs a TF32-exact array fact (no storage
  format carries it).
- m16 f32 / m2048 TF32 / m128 f32 / `mlp_512` / `smlp`: unchanged, see their earlier open items.

## For integration

README line (`local-rtx-5080/README.md`):

`| sweep-gemm | [sweep-gemm.md](sweep-gemm.md) | round-store: an f32 array read only by bf16 products stored rounded by its producer (precision_plan.store, narrow_stored + site-plan guard); 25 _bf16 programs: m128 bf16 1.11-1.18x, m2048 n6144 1.15x, m1 GEMVs 1.16-1.19x, LLM bf16 rows 1.01-1.14x, m128 n28672 0.97x (panel depth, open); FHFMA counted as FFMA |`

CLAUDE.md facts-table row:

`| precision_plan.store (NarrowStore), narrow_stored | arrays a map produces whose every element read (through nested captures) is a multiplicand of a bf16-admitting product: stored rounded once by the producer (bf16 storage, the reads widened) where no map leaves a site plan, unit group or f32 stream member it was on |`

Tailored-fix registry: none (no tailored fix in this sweep).
