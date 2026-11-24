# bugs-properties — the PTX ISA as a capability, measured properties, the split / cut / walk prices

Branch `fix/bugs-properties` on `8ff66d34` (cross-arch 5fcb401b's confirmation round). A replay round: no rental.
Every rule change below is replayed against the round's same-box OFF / ON arms
(`.claude/raw-evidence/cross-arch/5fcb401b/<part>/`, `analysis/arms.md`) and earlier rounds' (`rung_replay.tsv`),
the RTX 5080 locally (changed programs, ABBA). Raw output: `.claude/raw-evidence/local-rtx-5080/bugs-properties/`.

## 1. Bug — `_f8w` modules rejected on sm_89 (finding 1): the encoding is a capability

**Repro** (`f8w-ptxas-before.txt`): every `_f8w` decode row and px twin emitted for `rtx-4090` / `cuda-ada` went out at
`llc -mcpu=sm_89`'s default `.version 7.8`; `ptxas -arch=sm_89` rejects `cvt.rn.f16x2.e4m3x2` ("requires PTX ISA
.version 8.1"): 157 errors on `pt_llm_dec_b1_l2048_f8w` (the round's count), 3 845 on the B8 rows — 16 of 55 PTX over
seven profiles. After (`f8w-ptxas-after.txt`): `.version 8.1` on sm_89, **55 / 55 through `ptxas`**, every other
profile at its target's own ISA.

**Capability** (`profile.rs`): `Ptx { isa, sm }` — the PTX ISA and SM an encoding needs. `Caps::cvt_f8` is now
`Option<Ptx>` (sm_89: 8.1 — the widening's encoding there; from sm_90: 7.8), `Caps::ptx` the target's own floor
(`ptx_of_target`: sm_80 7.0, sm_86 7.1, sm_89 / sm_90 7.8, sm_90a 8.0, sm_120 8.7 — `llc` refuses a lower one: "PTX
version 7.8 does not support target 'sm_90a'"). The PTX encoding table sits beside `caps_of_cc`'s availability table;
no rule reads it. **The emitter declares the largest need** of the module: its target's and those of the capabilities
whose instructions it emitted (`narrow::defs` reports whether it emitted the conversions), as the module's
`; mapal.ptx isa=81 sm=89 mattr=+ptx81` line. A function attribute does not move `.version` (llc 23 reads the
TargetMachine's features only: tested), so **every compile step passes it**: `mapal build` (`Ptx::declared`),
`bundle.sh` and `integrated_bundle.sh` (`ptx_mattr`).

**Preflight**: `evidence.py static` now fails the bundle loudly when `ptxas` reports an error (or leaves no SASS) for
any distinct PTX of any part (`ptxas preflight FAILED <sm> <stem>.ptx: …`, exit 1 under `set -e`); `bundle.sh` runs
`ptxas -arch` on each PTX as it is made. The round logged all 25 errors at build time and nothing read them.

**Tests**: `f8.rs::generated_f8_programs_declare_their_ptx_isa_per_profile` — 16 generated fp8 programs
(`testgen::F8Prog`) × 8 GPU profiles: **128 device modules, 128 through `llc` at the declaration and `ptxas
-arch=<the profile's target>`**; fp8 instructions per profile (3090, A100, 4090, cuda-ada, H100, H100 PCIe, 5090, 5080)
[0, 0, 16, 16, 16, 16, 16, 16] (a part with the conversions emits them; the 3090 / A100 never); the declaration ≥
`Caps::ptx` and ≥ `Caps::cvt_f8` wherever the module carries the instruction. `profile::ptx_isa_per_part`,
`narrow::tests` (the emitted-instruction report). The existing value test (`generated_f8_programs_exact_bytes_…`,
48 programs, the four emulator modes, every OFF arm) passes unchanged.

**The evidence checker's hot loop** (found on the way, finding 2's diagnosis): `evidence.py` / `sass_stats.py` keyed
the hot loop on FFMA + HMMA counted alike, so the tf32x3 kernel's guard path (the cold FFMA recompute: 128 FFMA, 32
LDG) out-counted its k loop (96 HMMA) — the round's "40 LDG, 0 LDGSTS" and 113 `split-tile` intent misses on the A100
were that loop. A matrix instruction now weighs its multiply-accumulates a thread (`TENSOR_MACS` 32): the A100
kernel's hot loop is the HMMA loop — **8 LDGSTS, 0 LDG**, 1 639 instructions per 4 k-steps (re-assembled locally from
the emitted module, `ptxas -arch=sm_80`, 145 registers, no spill).

## 2. Measured against derived properties (the round's `calib-<part>.txt`)

| property | part | before (how) | now (how measured) | derivation for an unmeasured part |
|---|---|---|---|---|
| `mma.sync` TF32 / 16-bit, flops an SM a clock | H100, H100 PCIe | 532 / 1 064 (Mapal's realized tile, a lower bound) | **1 351 / 2 702** (`calib.cu matrix`: 32 warps of independent chains, clock64) | `mma_rates_of_cc(90)` = the H100's |
| same | A100 | 1 024 / 2 048 (datasheet) | **1 012 / 2 024** (`calib.cu matrix`) | `mma_rates_of_cc(80)` = the A100's |
| barrier, 256 threads | all | 41 cycles (the 5080's) at the part's clock | **`Gpu::bar_cycles`**, the SM design's: 3090 57, A100 54, 4090 / cuda-ada 57, H100 / PCIe 42, 5080 / 5090 45 (`calib.cu bar`, clock64) | `bar_cycles_of_cc` (native); `bar_ns` = cycles at the clock: 3090 36.5 (meas. 33.5 at the query clock), A100 38.3 (38.1), 4090 22.6 (22.5), H100 23.2 (21.1) |
| loaded panel latency `lat_ns` | 3090 / A100 / H100 | 899 / 995 / 776 (the 5080's 573 ns × its clock / the part's) | **452 / 728 / 629** — plan-cost's own quantity from the round's `calib.cu bw` sweep (bytes in flight over the DRAM rate below 90 % of peak; median of 101 / 112 / 116 points, IQR 439–486 / 686–802 / 605–660; the 4090 374 against its 380, the 5080 502 against 573) | **a DRAM round trip is ns, not SM cycles**: the 5080's 573 unscaled (`plan_cost`, `slice_cost`) — 0.8–1.5× of every measured part, where the clock-scaled value was 1.2–2.0× |
| `l2_keep_bytes` | 3090 / A100 / H100 | the nominal L2 | **½ L2** (3 / 20 / 25 MiB): `l2keep.cu` re-reads a 1.0 × L2 cyclic stream below the L2 / DRAM midpoint (1 205 / 1 563 / 2 579 GB/s; the A100 / H100 at their DRAM rate: two L2 partitions hold a line every SM reads twice) — a bound: the sweep began at 1.0 × L2 | open: no profile property separates a partitioned L2 (nominal stays; `l2keep.cu` now sweeps from 0.25 × L2) |
| `coop_body_cycles` | 3090 / A100 / 4090 / H100 | 4 390 (the 5080's) | **2 102 / 4 048 / 7 292 / 348** (`calib.cu body`, three runs) | the 5080's (unchanged) |

Not taken: the boxes' sustained clocks (a box's power limit, the calibration line's — as before) and `pageable_pm`
(a host property: `rung_replay.tsv` carries it as a condition).

**The changed-programs replay** (the round's 137 suite programs per part, the base emitter against this branch, text
less evidence comments; `replay/diffsuite-final.txt`, attribution by plan line):

| part | changed | `split-issue` | `streamk-epi` | `cut-price` | measured properties: kernels | tune cold flag only |
|---|---:|---:|---:|---:|---|---:|
| rtx-3090 | 33 | — | 12 | 5 | 10: attention streams at `lat_ns` 452 (899 derived) — `attn_2048_128` 2 → 16 slices, `attn_2048_64` 4 → 16, `attn_1024_64` `kb` 16 → 32; the seven B1 L8192 decode rows' row slices 256 → 128 blocks | 6 |
| a100 | 56 | 47 (13 with the walk on the FFMA tile) | — | 8 | `attn_2048_64` (coupled, §3) | — |
| rtx-4090 | 2 | — | — | 1 (`pj_m128_n28672_k4096_tf32` whole: a memory-bound site, as the 5080's) | (1 comment only) | — |
| h100 | 12 | — | — | 1 | (2 comments only) | 9 |
| rtx-5080 | 4 | — | — | 4 (local A/B, §5) | — | — |

**The H100's `mma.sync` rate moves no f32 GEMM onto tf32x3**: the rate rule alone flips (`split_rate_pays`: 1 351 ≥
3.9 × 256), the split realization's issue does not (§3). The 4090 / 5080 measured properties move nothing (their
kernels' prices read measured values already). Moves onto measured arms: all to the measured winner (the A100's 12
split-tile OFF wins, the 3090's 2 `streamk-epi` OFF wins, the 7 `cut-price` whole / S moves); the rest are
predictions for the next round's arms (`## For integration`).

## 3. The rules: split-tile, mma-combine, streamk-epi

### `split-tile` — the realization's issue (finding 2, replayed on the A100's 12 / 12 losses)

The SASS (above) refutes the round's diagnosis: the k loop streams its panels by `cp.async`. What it pays is
**instruction issue**: per k-step of its 2 × 4 warp tile, 24 HMMA, 32 flushes (FADD) and ~352 instructions of the
split itself (`hi = rna(x)` is an integer sequence on every SM — no F2F — plus the guard): 1 639 per 4 k-steps, 40
useful flops per warp-instruction, a ceiling of 160 useful flops an SM a clock (4 schedulers) against the FFMA tile's
128 on the A100's 64-lane SM, 256 on the others. It realized 63.6 (0.40 of the ceiling) against the FFMA tile's 86.6
(0.68 of its peak). **Rule** `rules::split_issue_pays` (beside `split_rate_pays`, read by `split_pays` at the tile):
`schedulers · useful / insns ≥ 1.7 · 2 · fp32_lanes`, `insns = (P + 4)·mt·nt + 22 · (16·mt + 8·nt)·k/32` for a form that
splits its operands (`SPLIT_ELEM_INSNS` 22 counted in the A100 kernel, the same PTX on sm_86 / 90 / 120 within 4 %),
none for `narrow1`. The margin 1.7 = the A100's measured break-even (ceiling 1.25 × FFMA realized 0.73 ×). Replay:
A100 tf32x3 off (1.25 < 1.7, 12 / 12 measured OFF wins), H100 off at its measured 1 351 (0.63), 5080 off (0.63; its
forced arm 4.10 vs 1.83 ms), `narrow1` on everywhere (ceiling ≥ 10 × FFMA). `rung_replay.tsv`: `split-tile a100 off
split=tf32x3,tile=2x4`.

Moved: **the A100's 47 f32 programs** leave tf32x3 for the FFMA tile (12 measured: OFF / ON 0.56–0.79 → the ON side
of this branch is the round's OFF arm; the rest unmeasured, the same realization). One coupled move: the A100's
`attn_2048_64` materializes — its split side, priced with FFMA tiles, now undercuts the stream by 0.6 % (152.6 against
153.5 µs; the stream won 2.08× against the tf32x3-materialized side, which the price had at 328 µs for 301 measured,
the stream 154 for 145 — accurate within 6–9 % on both): a prediction of even, a next-round arm. A margin on the
stream price (materialize only ≥ 5 % cheaper) held it, but streamed the 5080's `attn_2048_64` / `attn_1024_64`, which
the local A/B measured **0.90 / 0.98** (materialized faster): withdrawn — the price's error is the part's, not one
margin's.

### `mma-combine` — the cut's own work priced (finding 5)

The wave count `⌈blocks·S / slots⌉·(P/S + stages)` took every cut whose waves shrank; 9 of the 20 measured cuts lost.
**`MmaCut::ns`**: the waves at one block's share of the SM's matrix rate, plus — where it cuts — the zero pass and its
bytes at the fill rate, and `S · cells` `red` adds at the part's `atomic_ops_per_ns` (`plan_cost`, as `streamk_priced`
prices stream-K) **where the cut's working set (`a`, `b`, the output) spills the L2 the part keeps**; inside it the
adds resolve in the L2 beside the other pieces' products, overlapped (the 5080's 2048³ MLP GEMM, 48 MiB in its 66 MiB
keep, wins its cut 1.06× locally: 1.568 against 1.667 ms — the serial price of its 33.5 M adds, 84 µs, would refuse
it). Replay (`gpu::tests::mma_cut_replay`, every arm with its plan line's tile, grid and `S`):

| part | program | S | priced OFF / ON | measured OFF / ON | decision |
|---|---|---:|---:|---:|---|
| rtx-3090 | mlp_2048_tf32 / mlp2_2048_tf32 | 8 | 0.88 | 0.943 / 0.946 | whole ✓ |
| rtx-3090 | pj_m128_n4096_k14336 bf16 / tf32 | 8 | 1.00 / 1.07 | 0.928 / 1.074 | whole ✓ / cut ✓ |
| rtx-3090 | pj_m128_n6144_k4096 bf16 / tf32 | 8 | 2.05 / 2.32 | 1.273 / 1.494 | cut ✓ |
| a100 | mlp_2048_tf32 / mlp2_2048_tf32 | 4 | 0.94 | 0.914 / 0.836 | whole ✓ |
| a100 | pj_m128_n4096_k14336 bf16 / tf32 | 8 | 1.08 / 1.28 | 1.106 / 1.342 | cut ✓ |
| a100 | pj_m2048_n4096_k14336 bf16 / tf32 | 2 / 8 | 1.08 / 1.04 | 1.090 / 0.956 | cut ✓ / whole ✓ |
| rtx-4090 | pj_m128_n4096_k14336 bf16 / tf32 | 2 | 1.90 / 1.95 | 1.087 / 1.020 | cut ✓ / (null) |
| rtx-4090 | pj_m128_n6144_k4096 bf16 / tf32 | 4 | 1.18 / 1.14 | 0.864 / 0.922 | **open**: cut, as before |
| h100 | pj_m128_n6144_k4096 bf16 | 4 | 0.60 | 0.899 | whole ✓ |
| rtx-5080 | mlp_2048_tf32 | 8 | 1.09 (adds in the L2) | 1.058 (f32-tensor-cores; local 1.06) | cut ✓ |
| rtx-5080 | pj_m128_n6144 / m128_n4096_k14336 / m2048_n4096_k14336 tf32 | 8 | 1.32 / 1.07 / 1.21 | 1.316 / 1.091 / 1.106 | cut ✓ |

**18 of 20** non-null arms keep their measured side (the wave count: 11); every decision the price changes (cut →
whole: 3090 ×3, A100 ×3, H100 ×1; `S` 8 → 4 on the 5080's 2048 × 4096 × 4096, its measured best) moves to the measured
winner. Open (attempted): the 4090's 128 × 6144 × 4096 — its `b` stream at the DRAM rate (55 / 105 µs) bounds both
arms; a DRAM floor in the price catches it but misprices the A100's under-filled 128 × 4096 × 14336 (its uncut
kernel runs at 2.8× its peak estimate: the tensor tile's efficiency at its occupancy is the missing fact). Both stay
cut, as shipped.

### `streamk-epi` — the walk's panel reuse in the L2 (finding 6)

Measured on the 3090: the stream-K and ordered kernels of `loop_4096` share their k loop (SASS: 2 048 FFMA, 96 LDS, 8
LDG a trip; 230 / 224 registers, no spill), yet run 6.00–6.14 against 5.29–5.58 ms; the workspace's partials are ≤ 47
µs. The walk's CTAs start `iters` units apart, so the blocks reading one panel read it at skewed `k`: its reuse is the
L2's across the walk, where the ordered wave shares each panel in lockstep. On a 6 MiB L2 (75 KiB an SM) 128 MiB of
operands lose it; 2 MiB keep it (`indep_512` 1.05, `mlp_512` 1.13). **Rule** `rules::streamk_epi_pays`: the walk
where `l2_bytes_per_sm ≥ L2_PER_SM` (`grid-share`'s property) or the site's operands fit `l2_keep_bytes`; else the
group's ordered kernel (`epi_walk` → `epi_order`; `--gpu-on=streamk-epi` forces the walk). Replay: 3090 `loop_4096` /
`epi_4096` off ✓, the 512s on ✓, the 2048s off (nulls); H100 (388 KiB) / 5080 / A100 / 4090 unchanged.
`rung_replay.tsv`: 5 lines (`operands=`).

## 4. Tests

`cargo test --release --no-fail-fast -p mapal-backend-llvm -p mapal-cli` (RUST_TEST_THREADS=8, MAPAL_TESTGEN_JOBS=4,
the emulator modes): **498 tests, all pass** after four updates to the measured state — `split_tile.rs` (tf32x3 by
rule on no part), `calls.rs` (the four parts' measured bodies), the devmap device snapshot (the `; mapal.ptx` line)
and `golden/c0-rules.tsv` (2 entries, both the 3090's `streamk-epi`: `smlp_4096_256_1024`, `epi_2048`). In it:
`rung_rules_replay` (144 picked, 74 null, 14 mixed, 21 open — the same 21 as before), `mma_cut_replay` (18 + 2
open), `mma_pieces_on_the_measured_cells` (the 5080's 2048 × 4096 × 4096 at its measured S = 4), `ptx_isa_per_part`.
Generated programs: the fp8 encoding test above (16 programs × 8 profiles, counters non-zero, 128 / 128 through
`ptxas`); the `split-tile`, `mma-combine` and `streamk-epi` generated-program value tests (`split_tile.rs`,
`streamk_epi.rs`, `half_tile.rs`, `f8.rs`, four emulator modes, every OFF arm) pass unchanged — the realizations'
values are untouched, only which one a site takes.

## 5. Local RTX 5080 (changed programs, ABBA)

`benches/local/gpu_changed.sh --base-emit` (the base emitter's build as `.on-base`; suite programs) and the same flow
over the changed projections outside `suite.txt` (`gpu_bundle.sh` with `ARMS=…:base:base`), `gpu_run.sh` at
`TORCH=0`, 5 ABBA probe repetitions, value gates first (all pass), under the bench lock; raw
`local/{results,abba}-gc{1,2}.txt`. Median `iter ms` [min–max]; base / this > 1 = this branch faster.

| program (5080) | change | base | this | base / this | null |
|---|---|---:|---:|---:|---:|
| pj_m2048_n4096_k4096_tf32 | `cut-price` S 8 → 4 | 1.3359 | 1.3304 | 1.004 | 0.0 % |
| pj_m128_n4096_k14336_bf16 | cut 8 → whole | 0.2220 | 0.2190 | 1.014 | 0.3 % |
| pj_m2048_n4096_k4096_bf16 | cut 4 → whole | 0.6875 | 0.6763 | **1.017** | 0.1 % |
| pj_m2048_n4096_k14336_bf16 | S 8 → 4 | 2.2535 | 2.2720 | 0.992 | 9.6 % |
| mlp_2048_tf32 / mlp2_2048_tf32 | (the serial-adds price refused the cut; withdrawn: adds in the L2) | 0.7861 / 1.5683 | 0.8309 / 1.6670 | 0.946 / 0.941 | 0.3 / 0.1 % |
| attn_2048_64 / attn_1024_64 | (a stream margin streamed them; withdrawn) | 0.0895 / 0.0377 | 0.0994 / 0.0385 | 0.901 / 0.977 | 1.6 / 0.9 % |

The shipped branch keeps the first four (null or win) and emits the base's text for the last two rows (the
measurements that withdrew those two drafts). Nothing else changes on the 5080 (its properties are measured; its
emission differs from the base in those four programs only, `replay/diffsuite-final.txt`). The CPU was at 84–86 °C
during the second run (the host's other load); the GPU-side probe medians are 0.0–1.6 % null.

## 6. Workloads

| workload | part | OFF | ON | fastest competitor (leg) | explanation |
|---|---|---:|---:|---|---|
| pj_m2048_n4096_k4096 (f32) | a100 | 3.832 | 5.213 → **3.832** (= OFF) | 3.959 (torch-compile-cg) | `split-issue`: tf32x3 issue-bound, 160 against the FFMA tile's 128 useful flops / SM / clk |
| mlp_2048 (f32) | a100 | 2.014 | 2.718 → **2.014** (= OFF) | 2.022 (torch-compile-cg) | the same; 12 / 12 split-tile programs (0.56–0.79) |
| loop_4096 | rtx-3090 | 48.77 | 66.08 → **48.77** (= OFF) | 54.85 (chain-lt) | `streamk-epi`: 75 KiB L2 / SM, 128 MiB of operands — the walk loses its panels' L2 reuse |
| epi_4096 | rtx-3090 | 5.559 | 5.939 → **5.559** (= OFF) | 7.084 (torch-compile-ro) | the same |
| mlp_2048_tf32 | rtx-3090 | 1.402 | 1.487 → **1.402** (whole) | 1.197 (torch-compile-ro-tf32) | `cut-price`: 155 µs of `red` (output spills the 6 MiB L2) against 68 µs of waves |
| mlp_2048_tf32 | a100 | 0.5148 | 0.5632 → **0.5148** (whole) | 0.3676 (torch-compile-cg-tf32) | `cut-price`: 68 µs of adds against 62 µs of waves |
| pj_m128_n6144_k4096_bf16 | h100 | 0.0695 | 0.0773 → **0.0695** (whole) | 0.0323 (torch-bf16-compile) | `cut-price`: its waves 0.60 of the partials' cost |
| pj_m128_n6144_k4096 bf16 / tf32 | rtx-4090 | 0.0931 / 0.2019 | 0.1078 / 0.2189 (still cut) | 0.0901 / 0.1249 (torch-bf16-compile-cg / cublas-tf32-fresh) | open: the DRAM stream bounds both arms |
| pj_m2048_n4096_k4096_bf16 | rtx-5080 | 0.6875 (cut 4) | **0.6763** (whole) | 0.598 (bf16-compile-cg) | local ABBA 1.017 |
| `_f8w` decode (7 rows) | rtx-4090 | — | failed to load → **loads** (PTX 8.1, ptxas-checked) | (no fp8 leg) | `ptx-isa` |

OFF = the round's measured OFF arm, ON = its shipped arm; "→" = what this branch emits — the OFF arm's text where
marked (`= OFF`, `whole`: the same kernel), so its time is that arm's measured time.


## 7. Open problems attacked

- Finding 1 (fp8 on sm_89): **closed** — a capability's encoding, the declaration, the preflight, the generated test.
- Finding 2 (split-tile on the A100): **closed** — the diagnosis corrected (the checker's hot loop), the realization's
  issue priced; the H100's measured rate now in its profile without moving it onto the loser.
- Properties: **closed** for the matrix rates, the barrier (an SM-design property), the latency (ns) and the bodies;
  **attempted-open** for the L2 keep below 1.0 × L2 (a bound; `l2keep.cu` sweeps from 0.25 next round).
- Finding 5 (mma-combine): **18 / 20** by the cut's price (adds overlapped where the working set fits the L2 keep);
  attempted-open: the 4090's 128 × 6144 × 4096 (bf16 / TF32: the DRAM stream bounds both arms — needs the tensor
  tile's efficiency at its occupancy before a DRAM floor can enter the price), still cut as shipped.
- Finding 6 (streamk-epi on the 3090): **closed** by a property rule over the L2 share / keep.
- Not attacked here: finding 3 (`wgmma-ws` launch cost), finding 4 (the stream on HBM parts at long rows).

## For integration

- `local-rtx-5080/README.md` row: `bugs-properties` — the PTX ISA as a capability (`Ptx`, `Caps::ptx`,
  `Caps::cvt_f8`), the `ptxas` preflight, the measured matrix / barrier / latency / body properties, `split-issue`,
  `cut-price`, `streamk-epi`'s L2 rule; link this file.
- CLAUDE.md facts table, backend row (`Gpu` properties): add `bar_cycles` (the SM design's barrier cycles) and
  `Caps::ptx` / `Ptx` (the encoding a capability needs; the module's `; mapal.ptx` declaration); `rules::rungs` row:
  `split_issue_pays`, `streamk_epi_pays`.
- `docs/tailored-fixes.md`: nothing (no tailored fix).
- Next round ARMS (replay predictions to time): the 3090's `attn_2048_128` / `attn_2048_64` / `attn_1024_64` stream
  geometry at the measured latency, its B1 L8192 decode row slices (256 → 128), the 3090 / A100 epilogue groups'
  2048s (`streamk-epi` off, measured null), the A100's 35 unmeasured f32 programs off tf32x3 and its `attn_2048_64`
  (materialized with FFMA tiles, priced even with the stream), the 3090 / A100 `cut-price` moves on the k 4096
  projections (`S` 8 → 4, whole), `l2keep.cu` below 1.0 × L2.
