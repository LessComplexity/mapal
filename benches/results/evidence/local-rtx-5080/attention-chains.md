# attention-chains — the streamed attention on HBM parts; the A100's MLP / fan / loop regression

Branch `fix/attention-chains` on `8ff66d34` (the 5fcb401b confirmation round, `cross-arch/README.md` §5fcb401b). The
round's open items attacked: (1) the streamed (flash) attention wins on the GDDR parts and loses at long rows on the
A100 / H100 (A100 `attn_2048_128` priced 0.256 ms, measured 0.732, OFF 0.317; `llm_pre_t8192` 0.754); (2) the A100's
MLP class 1.248 → 1.537 and fan / loops 1.205 → 1.494 between 3a8e654f and 5fcb401b; (3) the other attention / chain /
loop rows the round shows losing. Validated on three rented A100 SXM4 40 GB boxes (same-box ABBA, 5 repetitions, every
arm value-gated against the others: `value stem=` lines) and the RTX 5080.

## What was wrong, and the fix

| # | finding (evidence) | missing fact / property | fix | status |
|---|---|---|---|---|
| A | the stream's consumer (`P·V`) loads 4-way bank-conflicted at `oc = d / g ≥ 16`: lane `x` read columns `x·oc …` — lanes `oc·4` bytes apart, every second lane on the same banks (16-byte loads, `oc` 16: 4 wavefronts where the price counts one). The HBM parts' picks (`g` 4 / 8 at `d` = 128) took the worst of it | a layout the price assumed (`wf(g)` wavefronts) and the kernel did not realize | `chunk_col`: 16-byte chunks interleaved across the row group — chunk `q÷4` of lane `x` at `4x + 4g·(q÷4)` (consumer loads, output columns, the split-KV partials) | closed |
| B | after A, the A100 still priced one-row-a-thread geometries 1.5–1.9× fast and under-filled grids up to 1.7× slow (§B) | the products' issue and their shared loads overlapped in the price (`max`) — the A100's 64 FP32 lanes double the issue term, which hid the loads; the loads' rate (`DeviceCost::lds_bcast_pm`, new: `calib.cu lds bcast`, 0.246 / 0.241 warp loads a clock on the 5080 / A100) was the 5080's fitted LSU constant; a grid's fill counted slots, not SMs | `stream_ns`: `max(issue, loads) + ½·min(issue, loads)` (`STREAM_LOAD_SERIAL`, fitted on both parts' sweeps); the busiest SM's `⌈blocks/sms⌉` blocks at the SM's rate, slowed only where its resident warps leave a scheduler fewer than two | closed on the A100 (7 / 8 rows win, §B); H100 unmeasured |
| C | `split-tile` (tf32x3) on the A100: 0 W / 12 L in the round, every f32 GEMM / MLP / fan / loop row — its k loop issues 1 543 SASS beside 96 `HMMA` (sm_80 = sm_90 = sm_120: `cvt.rna.tf32.f32` is an integer sequence) | `split_rate_pays` read the matrix rate only; the operand split's ALU issue on the FP32 lanes is the property that separates the A100 (1 024 TF32 flops a clock on 64 lanes) | `split_rate_pays(g, op, kind, (mt, nt))`: the warp tile's `k`-step at the slower of the matrix pipe and the FP32 lanes (`split_elem_instr`: 21 an element for tf32x3, 0 for narrow1); replay line `split-tile a100 … off` | closed (A100 1.27–1.85× faster on 14 rows) |
| D | `mma-combine` on the A100's TF32 MLPs loses (§D) | the cut's `red` traffic and the tail wave's rate (an SM alone runs a block at the SM's rate) are not in the wave model | — | attempted-open |

No tailored fix: every change reads a fact of the emitted kernel (its load / issue counts, its warp tile) and a measured
or derived physical property (FP32 lanes, the broadcast-load rate, SMs, schedulers).

## A — the consumer's chunks (RTX 5080, `attn_2048_128`, forced geometries, kernel µs, 3 ABBA processes)

| geometry (`b·kb : rt g threads : s`) | consumer `oc` | lane-contiguous (5fcb401b) | interleaved | × |
|---|---:|---:|---:|---:|
| b8xkb32:rt1g8t64:s8 (the A100's 5fcb401b pick) | 16 | 509 | 293 | 1.74 |
| b16xkb32:rt1g8t128:s8 | 16 | 441 | 226 | 1.95 |
| b32xkb32:rt1g8t256:s4 | 16 | 546 | 276 | 1.98 |
| b32xkb16:rt2g8t128:s4 | 16 | 323 | 187 | 1.73 |
| b64xkb32:rt2g8t256:s2 | 16 | 305 | 169 | 1.80 |
| b128xkb16:rt4g8t256 | 16 | 801 | 571 | 1.40 |
| b32xkb16:rt2g4t64 | 32 | 462 | 356 | 1.30 |
| b32xkb32:rt2g16t256:s4 | 8 | 255 | 218 | 1.17 |
| b64xkb32:rt4g16t256:s2 (the GDDR parts' pick) | 8 | 177 | 163 | 1.09 |
| b32xkb64:rt4g32t256:s2 | 4 | 185 | 186 | 1.00 |
| b8xkb32:rt1g32t256:s8 | 4 | 288 | 288 | 1.00 |

A100 (box 1), the round's own geometry `b8xkb32:rt1g8t64:s8`: 0.938 → 0.479 ms. The gain follows the conflict degree
`⌈g·gcd(8, oc/4)/8⌉ / ⌈g/8⌉` (4 at `oc` 16, 2 at 8, 1 at 4). The 5fcb401b kernels' conflicts alone do not explain
their price errors: counted as wavefronts they move the max-of-pipes price by < 15 % (`tools/conflicts.md`) — the
replays cost issue, which §B prices.

## B — the stream's price on the A100 (box 1, `attn_2048_128`, interleaved layout, kernel µs)

| geometry | measured | price 5fcb401b model | price now | |
|---|---:|---:|---:|---|
| b32xkb32:rt4g16t128:s4 | **316** | 314 | **289** | now the cheapest |
| b32xkb16:rt2g8t128:s4 | 343 | 379 | 337 | |
| b32xkb32:rt2g16t256:s4 | 345 | 318 | 346 | |
| b64xkb32:rt4g16t256:s4 | 397 | 415 | 382 | |
| b16xkb64:rt1g16t256:s4 | 434 | 291 | 382 | |
| b16xkb32:rt1g8t128:s8 | 440 | 295 | 379 | |
| b64xkb32:rt4g16t256:s2 (64 blocks) | 445 | 740 | 379 | fill over SMs |
| b64xkb32:rt2g8t256:s2 (64 blocks) | 453 | 740 | 388 | |
| b32xkb64:rt4g32t256:s2 | 465 | 418 | 384 | |
| b8xkb16:rt1g8t64:s16 | 477 | 273 | 399 | |
| b8xkb32:rt1g8t64:s8 | 480 | **256** | 363 | the 5fcb401b pick (the cheapest then) |
| b8xkb32:rt1g32t256:s8 | 524 | 446 | 484 | |
| b64xkb16:rt4g8t128:s2 | 561 | 1101 | 740 | |
| b32xkb16:rt2g4t64 (64 blocks of 2 warps) | 945 | 1457 | 1476 | starved: 2 warps an SM |
| b128xkb16:rt4g8t256 (16 blocks) | 1616 | 1590 | 1457 | |
| the split realization (OFF) | 326 | 373 (tf32x3 tile) | 257 (FFMA tiles, §C) | |

The model before: `max(issue, lds, lsu / 0.5) / eff`. The sweeps' rms of measured / price about their geomean
(`tools/fit3.py`; A100 box 1 + box 2: `attn_2048_128`, `attn_2048_64`, `attn_1024_64`, `llm_pre_t8192`, 33 geometries;
RTX 5080: `attn_2048_128`, 12): max at the 5080's 0.5 — 1.31 / 1.15; max at the measured 0.246 — 1.17 / 1.11; in
series — 1.14 / 1.05; **half in series — 1.13 / 1.07** (shipped). The broadcast-load rate is a measured property of the
part (`calib.cu lds bcast`, new kind; equal on both parts: `col` 0.246 / 0.241, `row` 0.397 / 0.393 — the LSU is not
what separates them, the 64 lanes are).

**The rule's picks on the A100, box 3 (EPYC 7532, 400 W), same-box ABBA × 5, iteration ms (`iter ms`, untraced):**

| program | 5fcb401b emitter (base) | now (rule) | split (OFF) | now / OFF | now pick | round OFF / ON |
|---|---:|---:|---:|---:|---|---:|
| attn_2048_128 | 0.941 | **0.298** | 0.324 | 0.92 | b32xkb32:rt4g16t128:s8 | 0.433 |
| attn_2048_64 | 0.188 | **0.160** | 0.205 | 0.78 | b16xkb16:rt4g8t32:s16 | 2.075 |
| attn_1024_64 | 0.083 | **0.063** | 0.086 | 0.74 | b16xkb32:rt2g8t64:s8 | 1.546 |
| attn_1024_128 | 0.275 | 0.134 | **0.124** | 1.07 | b16xkb64:rt4g32t128:s8 | — |
| llm_pre_t2048 | 24.70 | **17.01** | 18.96 | 0.90 | b32xkb16:rt4g8t64 | 0.955 |
| llm_pre_t8192 | 231.2 | **100.7** | 129.5 | 0.78 | b32xkb16:rt4g8t64 | 0.754 |
| llm_blkpre_t2048 | 72.46 | **53.91** | 55.63 | 0.97 | b32xkb16:rt4g8t64 | 0.996 |
| llm_pre_t2048_tf32 | 8.53 | **7.20** | 9.18 | 0.78 | b32xkb16:rt4g8t64 | 0.897 |
| llm_pre_t2048_bf16w | 20.73 | **16.91** | — | — | b32xkb16:rt4g8t64 | 0.944 |

Every A100 stream row the round lost now beats the split (the layer programs include §C's projections on both arms);
`attn_1024_128` (no round arm) takes a stream 7 % slower than its split.
The one row the rule leaves on the table: `llm_pre_t8192`'s `b32xkb16:rt4g8t64` 100.7 against the forced
`b32xkb32:rt4g16t128` 80.2 ms (priced 6.69 vs 6.72 ms a launch: a tie the model cannot split — open, below).

## C — `split-tile` on the A100 (box 1, same-box ABBA × 5, kernel ms; base = the 5fcb401b emitter, tf32x3)

| row (round class) | base | now (FFMA tile) | × | | row | base | now | × |
|---|---:|---:|---:|---|---|---:|---:|---:|
| mlp_2048 (MLP) | 3.492 | 2.567 | 1.36 | | rowsum_2048 | 1.753 | 1.306 | 1.34 |
| mlp2_2048 (MLP) | 6.991 | 5.233 | 1.34 | | softmax_rows_2048 | 1.774 | 1.327 | 1.34 |
| mlp_512 (MLP) | 0.178 | 0.096 | 1.85 | | pj_m2048_n4096_k4096 | 5.222 | 4.898 | 1.07 |
| indep_512 (fan) | 0.178 | 0.096 | 1.85 | | pj_m128_n4096_k4096 | 0.587 | 0.374 | 1.57 |
| diamond_2048 (fan) | 5.235 | 3.862 | 1.36 | | llm_pre_t2048 (+ §A/B) | 24.66 | 17.58 | 1.40 |
| fanout_2048 (fan) | 3.498 | 2.610 | 1.34 | | llm_pre_t8192 (+ §A/B) | 206.7 | 115.2 | 1.79 |
| twin_2048 (fan) | 3.459 | 2.555 | 1.35 | | | | | |
| loop_4096 (loops) | 80.48 | 63.50 | 1.27 | | | | | |

Which rule moved the classes: the round's `off-split-tile` arm (0 W / 12 L) and the emissions (`evidence-a100.tsv`:
`plan.split=tf32x3` on rowsum / softmax / twin / indep / mlp / fan / loop kernels, 3a8e654f's FFMA + stream-K tile
before) — every MLP and fan / loop row of the A100 took tf32x3 at 5fcb401b, on no other part. The A100's split k loop
(`pj_m2048_n4096_k4096`, 2 × 4 m16n8 tiles a warp, `bk` 32): 1 543 SASS a trip beside 96 `HMMA` (367 LOP3, 265
ISETP, 196 SEL, 192 FADD, 152 IADD3, 128 FSETP) — ALU-bound at 21 instructions an operand element on 64 FP32 lanes:
the rule's realized rate 89 useful flops an SM a clock against 1.3 · 128 (old: 341 ≥ 166). The derivation against the
kernels: A100 realized 72 % of its ALU bound (0.735× the FFMA tile, measured 0.74), the 5080's forced arm 84 % of its
matrix bound (sm_120's 128 lanes: MMA-bound) — one rule for both. The H100's measured `mma.sync` (1 351, finding 2 of
the round) no longer flips tf32x3 on (its 128 lanes: 178 < 333). Replay: `split-tile a100 F C off
split=tf32x3,tile=2x4` (and `tile=2x4` on the 5080's two lines); `rung_rules_replay` 141 picked.

## D — `mma-combine` on the A100 (box 1, kernel ms, ON / OFF) — attempted-open

| site | S | ON | OFF | ON / OFF |
|---|---:|---:|---:|---:|
| mlp_2048_tf32 | 4 | 0.704 | 0.655 | 1.075 (loses) |
| mlp2_2048_tf32 | 4 | 1.429 | 1.199 | 1.19 (loses) |
| pj_m2048_n4096_k14336_tf32 | 8 | 3.592 | 3.450 | 1.04 (loses) |
| pj_m128_n4096_k14336_tf32 | 8 | 0.416 | 0.591 | 0.70 (wins) |

Same verdicts as the round (0.836 / 0.914 / 0.956 / 1.342). The wave model `⌈blocks·S/slots⌉·(P/S + stages)` predicts
0.68 for mlp_2048 (256 blocks on 216 slots). Two terms it lacks, both the stream fix's: the uncut grid's tail wave runs
its 40 blocks one an SM, at the SM's rate (not a slot's) — the uncut time is ~25 % lower than modeled — and the cut's
`red.add` traffic (S × 2048² adds at `atomic_ops_per_ns`, derived 248 on the A100: 68 µs at S = 4, 10 % of the site).
With both the model predicts S = 4 at 1.01 and S = 2 at 0.91 (unmeasured): not shipped — it needs its own sweep on the
A100 and the 5080's winners (`f32-tensor-cores.md`) replayed.

## Tests

`flash_attention` (6, incl. `stream_on_generated_attention`: 128 generated programs, **value widths now up to 32**,
new counters `narrow groups streamed` 73, `contig consumer` 22, `chunks>1` 10 — the interleaved chunks through the
emulator in every mode, 1 023 runs; `stream_priced_on_every_part` asserts the A100's streams take four rows a thread),
`streams` 19, `row_block` 9, `split_tile` 4 (tf32x3 nowhere by rule; every part under the ON arm), `site_choice`,
`mapal-rewrite stream_plan` 3, the backend lib 125 (`rung_rules_replay` 141 / 72 / 10 / 21; `lds_bcast_is_measured_or_derived`;
the native-gpu calibration line carries `lds_bcast_pm=246`).

## RTX 5080: changed programs (`gpu_changed.sh --base-emit` the 5fcb401b emitter, `gpu_run.sh` ABBA × 5, iter ms)

`suite_diff`: the 35 README programs, both cells and the five stream layer programs — 6 differ (the 5080 never took
tf32x3, and its attention rows stay split by price: `attn_1024_64` differs in an indented `; mapal.flow.host`
comment only — a null by construction, timed as one). Value gates pass on every arm (10–13 arms a program); the intent
check adds no mismatch (69 / 6 133, the same groups on both arms: kmajor-a, full-tile, flow-host).

| program | base ms | now ms | now / base | null |
|---|---:|---:|---:|---:|
| llm_pre_t8192 | 50.74 | 44.86 | **0.884** | 0.1 % |
| llm_blkpre_t2048 | 27.13 | 25.10 | **0.925** | 0.4 % |
| llm_pre_t2048_bf16w | 6.802 | 6.378 | **0.938** | 0.1 % |
| llm_pre_t2048 | 6.780 | 6.475 | **0.955** | 0.2 % |
| llm_pre_t2048_tf32 | 5.487 | 5.348 | **0.975** | 0.0 % |
| attn_1024_64 (equal code) | 0.0399 | 0.0398 | 0.998 | 0.4 % |

The 5080's picks are unchanged (`b64xkb32:rt4g16t256`); the gains are §A's 2-way conflicts at `oc` = 8.

**Safety net** (`gpu_bundle.sh` + `gpu_run.sh`, the whole README suite and both cells, 5 repetitions): 35 / 35 value
gates pass, evidence 12 969 checks / 202 mismatches (the standing groups); every program's text equals 5fcb401b's on
the 5080 and its Mapal ms the round's within null (attn_1024_64 0.0399 / 0.0393, mlp_2048 0.887 / 0.886, loop_4096
26.93 / 26.93, rowsum_2048 0.460 / 0.459) except the host-bound `indep_512` (43.4 / 38.2 µs: equal code, the host at
80–90 °C under the other agents' builds).

## Rentals

| box | offer | $/h | seconds | cost | instance (destroyed) |
|---|---|---:|---:|---:|---|
| A100 SXM4 40GB, EPYC 7K62, 400 W | 57328776 | 0.473 | 837 | $0.110 | 62336601 |
| A100 SXM4 40GB, Ryzen 7 7700, 300 W | 56853932 | 0.549 | 233 | $0.036 | 62338775 |
| A100 SXM4 40GB, EPYC 7532, 400 W | 51341862 | 0.843 | 436 | $0.102 | 62341304 |
| **total** | | | | **$0.25** (cap $3) | every `mapal-ac-*` instance destroyed, checked by label |

The three boxes differ in pace (box 1 ran the sub-ms kernels ~1.3× slower than box 2: its probes' GPU clock); every
comparison above is within one box. The boxes ran a lean harness (`tools/box_run.sh`: link on the box, one value run
an arm, then ABBA rounds of an untraced and a traced process), no `evidence.py` pass; the A100 kernels' static facts
(registers, no spill) are `ptxas -arch=sm_80` locally, and the intent check runs in the 5080 flow above. Review: by
CLAUDE.md a GPU run without its `evidence.tsv` is not a result — the three A100 boxes' times (the streamed attention
1.17–2.8×, the MLP / fan / loop rows) are timing only until a round with the evidence pass re-times them; the
integration and README cite them as such.

## Open

1. `llm_pre_t8192` on the A100: the price ties `b32xkb16:rt4g8t64` and `b32xkb32:rt4g16t128` (6.69 / 6.72 ms a launch)
   that measure 61.4 / 40.9 ms (stream kernel, all launches): 2-warp blocks over 512 windows of 16 keys a row — the
   per-window wait the price charges once a wave (`fixed`) is exposed when a window's products are shorter than a
   window load. `attn_1024_128`: 0.134 against the split's 0.124 ms (priced 78.8 / 80.9 µs). The 3090 / 4090 / H100
   picks B moved are unmeasured (the 5080 measured the 4090's new `llm_pre_t2048` pick, `b128xkb16:rt4g8t256`, 6.73 ms
   against the old `b64xkb32:rt4g16t256` 7.40).
2. The H100: B moves its picks (`llm_pre_t8192` `b128xkb64:rt8g16t256`, `attn_2048_64` stream); its `lds_bcast_pm` is
   derived (0.25 — the A100's and 5080's measured 0.24 agree with the derivation). No H100 was rented.
3. The split realization's price on small sites (RTX 5080 `attn_1024_64`: split 37.8 µs measured, 18.6 priced; the
   stream `b64xkb32:rt4g8t128:s8` 30.3 µs, priced 22.2 — the rule keeps the split): the tile model's fixed costs at
   1024 × 1024 × 64.
4. §D `mma-combine`.

Raw (not in git): `.claude/raw-evidence/local-rtx-5080/attention-chains/` — `a100-r{1,2,3}/` (results, summaries,
vast logs), `tools/` (one.sh, sweep.py, mkbundle.sh, box_run.sh, res.py, fit3.py, conflicts.py, the bundle lists, the
fit outputs, the 5080 sweeps).

## For integration

- `benches/results/evidence/local-rtx-5080/README.md`, a row: `attention-chains` — the stream's consumer chunks
  interleaved (A100 attn 2048 × 128 0.94 → 0.28 ms, 1.17× the split); the stream priced half in series at the
  measured broadcast-load rate (`lds_bcast_pm`) with the grid's fill over SMs; `split_rate_pays` with the operand
  split's ALU issue (A100 MLP / fan / loop rows 1.27–1.85× faster) → [attention-chains.md](attention-chains.md).
- CLAUDE.md facts table, the `backend: Gpu properties` row: add `DeviceCost::lds_bcast_pm` (the stream's broadcast
  16-byte shared loads a clock, `calib.cu lds bcast`) and `rules::split_elem_instr` (the split form's ALU issue the
  split rate rule reads at the FP32 lanes).
- `docs/tailored-fixes.md`: nothing (no tailored fix).
- `cross-arch/README.md` §5fcb401b open items 2 and 4: closed here for the A100 (B, C); the H100 parts of item 4 open.
