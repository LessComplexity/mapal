# review-fixes — the independent review of the sprint after f3a9e792 (RTX 5080, 2026-11-22)

Four findings, each fixed with a test that fails without the fix. Machine and tools as in `README.md`; raw output in
`.claude/raw-evidence/local-rtx-5080/review-fixes/`.

| finding | cause | fix | test (fails without the fix) |
| --- | --- | --- | --- |
| `row-slices` + rescale law (ADR-0042 D7) prints NaN when a slice's scores are all `−∞` (a mask) | a slice's running max starts at the identity `−∞`; `exp(κ(−∞ − (−∞)))` = NaN terms, and the combine's `NaN · exp(a(m_r, m)) = NaN · 0` = NaN | `flow.rs` `rescale_shift`: a slice whose max partial is the step's identity takes the stand-in shift 0 (its terms `exp(−∞) = 0`); the combine still rescales by the true shift, `0 · exp(a(−∞, m))` = 0 beside a finite row max and NaN where the row max is `−∞` too — the sequential fold either way | `exp.rs` `rescale_over_masked_slices` (seeds −1e9 and `−∞`, 2 / 4 / 8 slices, 4 emulator modes: NaN in every row without); generated `long_row` programs: a third open with a masked softmax (below) |
| a loop's carried array whose next value reads neither the state nor the counter returns the initial value on the device | the rewrite hoists such a map out of the loop (made once); `resident` still admitted it as a device carry (F3), and the back edge swaps the carry's and the next value's buffers each trip — the second swap hands the state its first buffer back | `resident.rs` `carries`: a carry needs its next value made in the trip (`loop_plan` decide / advance order); otherwise the state forwards the one object's pointer (`carried=…:one`, nothing crosses a trip) | `onto_device.rs` `hoisted_next_state_is_no_device_carry` (the review's repro and a device-read twin, default and exact faces, 4 modes); generated `onto` programs: `OntoProg::hoisted` (below) |
| `calls` chains ship on rtx-4090 / rtx-3090 / a100 with no timing; `chain_ns` prices only the boundaries | the rule had no term for what the bodies become inside one kernel; on the only timed part every forced chain lost by more or less than its boundaries say | `DeviceCost::coop_body_ns` (measured; rtx-5080 2500 ns) and `coop_first_ns` (a chain outside every loop) in `chain_ns`; a part without `coop_body_ns` is `unpriced` (the rule's convention for its other measured inputs: `gsync_ns`, `dep_launch_ns`); `--calls=force`, the arm that measures it, unchanged | `calls.rs` `unmeasured_bodies_form_no_chain` (4090 / 3090 / a100 shipped: no chain; forced: chains; 5080: refused by cost); unit `the_bodies_price_decides_where_the_boundaries_alone_did_not` |
| no machine-checkable intent for `row-slices`, `exp-hw`, `exp-rescale` | — | `intent.py` entries; `evidence.py` facts `ptx.ex2`, `sass.mufu_ex2`, `flow.slices / phases / n_rescale`, `mod.exp32` (the kernel's `@mapal.exp.f32` calls through its internal callees), `mod.hw_exp`, `host.calls.launch_coop`, the order-free faces | the bundle check below |

## 1. Masked slices (rescale law)

Repro (`rs3`: one row of 131072 softmax scores, `j ≥ 20000` overflowing to `−∞`; `slices=64 phases=2 laws=rescale`):
interpreter `9.604226e-30 5.638e-41 0 0 −inf 261.2165`; HEAD on the RTX 5080: NaN for every `y` and `den`; now
`9.6042335e-30 5.638e-41 0 0 −inf 261.2163` (within the D2 bound).

Intended vs realized: the slice fold gains one `fcmp oeq` + `select` per item (PTX/SASS: one `FSETP` + `FSEL`); no other
instruction moves. Emission of all 260 `benches/algorithms` programs on the rtx-5080, HEAD vs this step: 259
byte-identical, `pt_lsm_1_131072` differs only by that select. Time, CUPTI kernel duration of its sliced unit, ABBA 8
rounds (16 processes an arm): **5.888 vs 5.888 us (new / HEAD 1.000; null 4.3 %)**. (The process-time ABBA of this
50 us program is noise: spreads 79 % / 42 %.)

Generated programs (`testgen::long_row_strategy`): a third (`LongRow::masked`, a hash of the drawn nodes — the other
programs stay the stream's) open with a softmax over `a0` masked past `cut − i · step` (slices and rows wholly masked).
160 programs × s = 2, 8 × 3 modes against the interpreter and the OFF arm: 242 sliced units, 528 runs; coverage (each
> 0): 114 rescale laws (112 on masked rows), 4 factor, 2 shift, 242 multi-phase, 10 three-phase, 188 stored folds,
46 stored row cells, 114 per-row cells, 70 one-row units, 72 looped. Without the fix: 23 workers fail (`sm` NaN).

## 2. A next state made once

Repro (`fail1b`: `mut x <- b0; loop { x = b0 / 2 + a0 }`, two trips): interpreter `0.022460938 −0.4189453 1.2890625
0.52246094`; HEAD on the RTX 5080 `0.09375 0.09375 0.5 1.4375` (= `b0`, `carried=0:4096:device`); now the interpreter's
values, `carried=0:4096:one h2d=0 d2h=0`. No `benches/algorithms` program changes emission.

Generated programs (`testgen::onto_prog_strategy`): `OntoProg::hoisted` (derived from the drawn fields) adds a second
loop whose next value is `out / 2 + a` — the first loop's device-carried exit, made once on the device. 96 programs,
57 with it (`hoisted-next`); the other counters unchanged (77 device-carried, 342 proven by conditions, …). Without the
fix the emulator's values differ in every mode.

Open: such a state could stay on the device without a carry (after a trip it is the next value's buffer); today a
device reader of it inside the loop reads the forwarded pointer's object.

## 3. `calls`: the bodies' price

Shipped builds at HEAD (`; mapal.calls.host`): rtx-4090 decode (3664 vs 4320 ns), causal / gen / prefill (5170 vs
8640); rtx-3090 decode, causal, gen, prefill, rt_mlp2 (3894 vs 5888); a100 causal / gen / prefill (10279 vs 12300);
rtx-5080 and h100 none. None of these parts was timed with a chain.

The rtx-5080's forced chains against the shipped build (this step, PDL shipped; ABBA 4 rounds, 8 processes an arm):

| workload | shipped (ms) | forced (ms) | forced / shipped | a trip, measured | a trip, priced (per − sep) | beyond the price |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| decode_1024_256 | 8.771 | 12.689 | 1.447 | +3.83 us | +1.86 | **+1.97** |
| causal_1024_256 | 15.305 | 16.933 | 1.106 | +1.59 | +2.31 | −0.72 |
| gen_1024_512_256 | 7.810 | 8.577 | 1.098 | +1.50 | +2.31 | −0.81 |
| prefill_1024_512_256 | 7.833 | 8.679 | 1.108 | +1.65 | +2.31 | −0.66 |
| rt_mlp2_1024 (2 chains a trip, 8 trips) | 2.670 | 2.775 | 1.040 | +13.2 | +3.42 + 38 / 8 (first cooperative launch) | **+2.51 a chain** |

Nulls 0.3–4.0 %. CUPTI: decode's persistent kernel 9.59 us against 4.25 + 2.28 + 1.69 = 8.22 us of its kernels (its
rows level 5.0 us against 4.2 alone, kernel-merging Calls); rt_mlp2's two persistent GEMM pairs 166.5 / 165.3 us at
128 / 127 registers against 161.0 / 163.7 us of their four kernels at 95–96 (+3.55 us a chain, its one barrier
inside). Experiments on decode's persistent kernel (PTX variants, CUPTI): the barrier's poll backed off
(`nanosleep.u32 100`) 9.57–9.58 us; a relaxed poll with one acquire fence after it 10.00–10.12 us; every load
non-coherent (`ld.global.nc`, a timing probe) 9.56–9.71 us — neither the waiters nor the load path explain the rows
level. The residual is mixed (decode and rt_mlp2 slower, the causal family faster): the bodies' change inside one
kernel is a missing fact (open), so the rule prices it by the largest measured residual, `coop_body_ns` = 2500 ns a
chain, and the process's first cooperative launch (`coop_first_ns`, 38 us) on a chain outside every loop.

Decisions now: rtx-5080 unchanged (every suite chain refused by cost; emission byte-identical); rtx-4090 / rtx-3090 /
a100 / h100 `refused=unpriced` — their emission is the OFF arm's (the candidate-0 golden `c0-rules.tsv`: 21 rtx-4090 /
3090 rows regolded, each equal to HEAD's emission with `--gpu-off=calls`). With the rtx-5080's measured body price applied to their
boundaries (unit test), decode's chain loses there (the boundaries' 656 / 522 ns saving is inside the 1970 ns decode
residual) and causal's six-launch chain still pays — so measuring `coop_body_ns` on a part (its `--calls=force` suite
chains against the shipped build) is what turns its chains on; the rented parts' next round carries that measurement.
(Superseded by §5: the price is calibrated by a microbenchmark and derived on every part.)

## 4. Intents (`benches/gpu/intent.py`)

- `row-slices` (kernel; `flow.slices > 1`): one release arrival per phase boundary (`ptx.gsync_arrive == flow.phases −
  1`), a cooperative launch (`host.calls.launch_coop`), the partials stored (`st.global`), no call, no spill; OFF: one
  block a row, no barrier.
- `exp-rescale` (a sliced unit with a rescale law): an `exp` in the kernel (`mod.exp32`); OFF: no rescale law (phases:
  the arm's diff).
- `exp-hw` (a kernel whose body reaches `@mapal.exp.f32`, order-free face): the module's hardware form, `ex2.approx` in
  PTX, `MUFU.EX2` in SASS; OFF and the exact / contract / fixed faces: neither.

Bundle (`gpu_bundle.sh`: lsm_1_131072 with its `.off-row-slices` / `.off-exp-hw` / `.off-exp-rescale` arms,
lsm_4_32768, softmax_rows_2048, decode_1024_256, attn_1024_64, causal_1024_256): **44 checks of the three new intents,
44 ok** (`row-slices` on 15 / off 1, `exp-hw` on 24 / off 1, `exp-rescale` on 2 / off 1). The bundle's 78 other
mismatches are HEAD's (the emission is HEAD's but for lsm_1's slice select): `pdl:on` "griddepcontrol at kernel entry"
against the exit trigger, `full-tile`, `face-registers`, `kmajor-a`, … — the untriaged set of `gpu/evidence-rtx-5080.tsv`.

## 5. `calls`: the bodies' price calibrated and derived — every part priced

Decision: calibrate + derive; no part refuses chains as `unpriced`. The property is `DeviceCost::coop_body_cycles`
(SM cycles at the part's `clock_khz`); `Gpu::coop_body_ns` reads it in ns. A part without its own measurement derives it
from the rtx-5080's measured cycles (`COOP_BODY_CYCLES_RTX_5080`) at its own clock. `chain_ns` adds it on every part.

**Calibration** (`benches/gpu/calib.cu body`; builds on sm_80 / 86 / 120, nothing beyond cooperative launch and
`%globaltimer`). The chain has two levels of members. Level 0 is two one-block row folds (decode's rows: 256 threads,
k = 256, 8 loads in flight a thread). Each further level is one wide fold of `nb` blocks that reads the level before.
The chain runs two ways: as separate launches (`__restrict__`: Mapal's launch kernels) and as one cooperative
persistent kernel built as `persistent_kernel` builds it (bodies inlined, the first block straight-line, the rest in a
grid-stride loop, a grid barrier per level, no `__restrict__`). Each arm runs 129 chains back to back, device-bound,
with work and with the bodies skipped by a runtime flag (the same code). The four arms rotate ABBA over 15 reps.
body = (pers − pers_empty) − (sep − sep_empty). The empty arms hold exactly the boundaries the rule prices (barriers
against launches). The table uses each chain's span, from its first block's start to its last block's end. The span
leaves out the chain's own launch. Back-to-back cooperative launches here start on a **2.048 µs grid**: every
persistent start-to-start is a multiple of 2048 ns (empty 2048 / 4096, with work 8192–18432), and `%globaltimer` ticks
every 32 ns. Six runs on the RTX 5080 (2026-11-22, load < 3, bench lock; raw in `coop-body/` of the raw directory); the table gives
the median [range] of the six run medians:

| chain | grid | sep span (us) | body, span (ns) | body, start to start (ns) |
| --- | --- | ---: | ---: | ---: |
| 2 levels, 1 round (126-block wide level) | 126 | 6.76 | 136 [60–179] | 405 [337–460] |
| 3 levels, 1 round | 126 | 7.89 | 376 [292–420] | 232 [−758–317] |
| 4 levels, 1 round | 126 | 9.02 | 638 [493–665] | 104 [−43–200] |
| 3 levels, 2 rounds (840-block wide levels) | 420 | 10.80 | −64 [−1789–1544] | 701 [2–2796] |
| **3 levels, 4 rounds (1680-block wide levels: the grid-stride form)** | 420 | 15.06 | **1792 [1083–2596]** | 2414 [972–3312] |

The two forms' outputs match bit for bit in every shape.

SASS and usage (`cuobjdump`, sm_120). Separate: `row_k` 25 registers and `wide_k` 36, with 16 `LDG.E.CONSTANT` a body
trip. Persistent: `pers_k<3>` 48 registers with the same 16 loads a trip as `LDG.E` (no read-only path), 2
`LDG.E.STRONG.GPU` (the barrier's poll) and no stack. So the persistent kernel holds 5 blocks a SM against the wide
member's own 6, the 1536-thread limit. At one round that does not bind (126 blocks on 84 SMs): ~250 ns a wide level.
At 4 rounds the 1680 blocks take 4 rounds of 420 against 3.33 waves of 504 alone. That is 5/6 of the loads in flight
per SM in the grid-stride form: +1.8 us. The 2-round null (the separate arm is bimodal across runs, 11.5 / 15.2 us)
is open. Clocks: 2.86–2.91 GHz during the chains (`run_khz`), 2.49–2.64 GHz sustained under FFMA, 2.45 GHz in the
profile (E1). **`coop_body_cycles` = 1792 ns × 2.45 GHz = 4390.** The profile clock keeps the part's price at what it
measured; at the runs' own sustained clocks it would be 2864–6672, median 4680. The property uses the 4-round chain:
it is the grid-stride form, and the largest change among the shapes.

**Against the suite's 2500 ns** (the largest forced-chain residual, §3), the calibration measures 0.72 of it. Its
decode-like one-round chain measures 0.38 us against decode's 1.97 us residual. Two terms explain the difference:
(1) Mapal's LLVM persistent kernels change the bodies more than nvcc's do here. rt_mlp2's GEMM pairs run at 127–128
registers against 95–96 (+3.55 us of kernel time a chain), and decode's rows level takes 5.0 us against 4.2 alone
(CUPTI). The synthetic bodies keep their loads (the same LDG count a trip). (2) The 2.048 us cooperative dispatch grid.
The suite's residual is trip to trip, so it includes the trip's rounding up to that grid. `coop_launch_ns` (2044)
prices one quantum, an empty kernel's (decode's x → persistent-kernel gap 1559 ns against 272, kernel-merging Calls).
The span leaves it out; start to start keeps it (4-round: 2414 vs 1792). The suite's residuals also have mixed sign
(the causal family −0.66..−0.81), so the per-chain property is a bound, as before.

**Measured vs derived, per part** (`profile.rs`; native-gpu on the 5080 reads `coop_body_cycles=4390` from
`calib/rtx-5080.calib`, the hand profile byte for byte):

| part | clock (GHz) | source | ns a chain |
| --- | ---: | --- | ---: |
| rtx-5080 | 2.45 | measured (`calib.cu body`) | 1792 |
| rtx-4090 | 2.52 | derived | 1742 |
| rtx-5090 | 2.41 | derived (no barrier / launch measured: no chain priced) | 1822 |
| h100 | 1.807 | derived | 2429 |
| h100-pcie | 1.755 | derived (no barrier / launch measured) | 2501 |
| rtx-3090 | 1.561 | derived | 2812 |
| a100 | 1.41 | derived | 3113 |

**Decisions per part**: all 260 `benches/algorithms` programs, default emission, `; mapal.calls.host`, no `unpriced`
anywhere:

| part | chains formed | refused by cost (selected) |
| --- | --- | --- |
| rtx-5080 | none (unchanged: its boundaries alone refuse every chain, 2044 ns cooperative against 486 / 650 launches) | decode, causal / gen / prefill, rt_mlp2, firew |
| rtx-4090 | causal / gen / prefill (6 units, 5 barriers: 6912 vs 8640 ns); llm_blkdec_b1_l4096 (+ `_bf16w`, `_pv`; 8 units: 7916 vs 11520) | decode (656 saved < 1742), rt_mlp2, firew |
| rtx-3090 | causal / gen / prefill (8248 vs 8832) | decode, rt_mlp2, firew |
| a100 | none | causal (13392 vs 12300), decode, rt_mlp2, firew |
| h100 | px_llm_pre_t12 (+ `_pv`; 7 units: 9728 vs 12054), px_llm_blkpre_t12 (+ `_pv`; 2 chains, 13 units: 18547 vs 22386), value programs only, untimed | **smlp, firew_rowsum** (2749 + 2429 = 5178 vs 3444), decode, causal, rt_mlp2 |

Replay checks. The H100's smlp / firew_rowsum formed on the boundaries alone in cross-arch 7a2b67bf and ran 1.55 / 2.3x
slower; they are refused now. The one-round synthetic value would have re-formed them: 0.38 us is 520 ns at 1.807 GHz,
against 695 ns saved. That is why the property takes the grid-stride chain. The rtx-5080's forced chains (§3) stay
refused. What separates those H100 losers from the chains that now form is phase length. The H100 chains' phases are
long (70 + 55 us, 31 + 12 us) and the change inside one kernel grew with them (36–74 us). Against that, they save 695
ns. The chains that now form have short phases (the causal family's launches ~2.5 us). A per-chain bound refuses the
H100 rows only because their saving is small: the phase-time term stays open (cross-arch open item 2). The derivation
follows the clock: the a100's causal chain forms at `--cost=clock_khz=2300000`, and the 3090's is refused at 1 GHz.

The golden `c0-rules.tsv` (horizontal off) has 21 rtx-4090 / 3090 rows regolded, each equal to the build that forms
exactly its chains:
- decode_1024_256 contract / exact on both parts: its 5-unit chain forms (5406 vs 7200, 6706 vs 7360 ns). The code
  less `; mapal.` lines equals `--calls=force`'s.
- The other 17: every chain is refused by cost, and the code equals `--gpu-off=calls`'s. Fifteen of them equal their
  pre-af0da8a5 hash, because the line reads `refused=cost` again. rq_mlp2_512 on the 3090 had formed on the
  boundaries alone.

Tests:
- Unit: `the_rule_prices_barriers_against_launches` and `the_bodies_price_decides_where_the_boundaries_alone_did_not`
  (derived = the 5080's cycles at the part's clock; the a100 flips at 2.3 GHz; the H100 replay).
- Integration: `every_part_prices_chains` replaces `unmeasured_bodies_form_no_chain`. It checks every GPU profile's
  source and ns, the five programs × five parts' decisions, no `unpriced`, and both clock flips through `--cost=`.
- `calls_on_generated_programs` (the generated-program property test) and `decode_values_through_the_persistent_kernel`
  pass unchanged.
- native-gpu: `native_gpu_is_the_query_and_the_calibration` holds with the calibration line.
- Workspace, release: 124 binaries, 1450 passed, 2 failed. The failures are `row_block`
  `attention_pricing_is_one_rule_across_parts` and `row_block_ships_its_rule_and_refuses_unpriced`, both on
  px_attn_64_16 with `calls` off. They fail identically at HEAD (stashed and rebuilt) and with a freshly measured
  native-dispatch cache: this host's native placement, outside this step.

Open:
1. The phases' time under one kernel, the change that grew to 36–74 us on the H100. A per-chain bound does not model it.
2. `calib.cu body` on the rented parts: `box_run.sh` runs it in all three calibration passes, and their lines then
   carry measured cycles.
3. The 2.048 us cooperative dispatch grid as a property: a trip's rounding is not priced.
4. The 2-round null.

## Tests

`exp.rs` 4, `long_rows.rs` 4, `onto_device.rs` 11, `calls.rs` 4 + 5 unit (`func::calls`). Each new test checked
against the code before its fix: masked slices (NaN), the generated long rows (23 failing workers), the hoisted state
and the generated onto programs (values), `unmeasured_bodies_form_no_chain` (the 4090 forms decode). Workspace,
release: 124 binaries, 1448 passed, 4 failed — the two `native-gpu` tests (the calibration line lacked
`coop_body_ns`: added, 110 / 110 lib tests pass) and the candidate-0 golden (regolded, above) fixed and re-run;
`streams::pdl_is_hopper_and_later_only` fails at HEAD too (it expects `pdl` on a straight-line program; the cross-arch
rule ships it on repeating launches only) — pre-existing, outside this step.

## Open

1. The bodies' change inside a persistent kernel (decode's rows level, rt_mlp2's GEMM pairs at 128 registers; the
   causal family faster): the mechanism. Until it is a fact, `coop_body_ns` is a measured bound per part.
2. A state whose next value is made once could stay device-resident without a carry.
3. `coop_body_ns` on the rtx-4090 / rtx-3090 / a100 / h100: their forced suite chains. (§5: calibrated by `calib.cu
   body` and derived on every part; its own open items there.)

# Review after 60f0f8ce — a narrow operand's counter step, a callee's device loop (RTX 5080, 2026-11-23)

Two findings of the independent review of the commits after 60f0f8ce, each fixed with a generated test that fails
without the fix. Raw output in `.claude/raw-evidence/local-rtx-5080/review-narrow-callee/`.

| finding | cause | fix | test (fails without the fix) |
| --- | --- | --- | --- |
| `uniform-base` × `narrow-radix`: a `bf16` KV cache read under a head-group loop gives wrong values | the launch moved a narrow operand's pointer by `coef · 4` bytes (the product's `elem`), not `coef · 2` (its storage): `kc` moved `gg · 32` bytes for a step of 8 elements = 16 bytes; the tile launch path (`resident` off) also copied a narrow operand as `n · 4` bytes, `n · 2` past its host array | `resident.rs` `emit_site_resident` and `gpu.rs`'s tile launch: each operand's term and bytes from its own storage (`TileRead::elem_bytes`) | `device_geometry.rs` `uniform_base_on_generated_programs`: `SliceProg::narrow` (K/V stored `to_bf16`, read `-> widen_f32`) and the tile launch path run on every narrow-moved program and a third of the rest. Without the resident fix: wrong values in the 4 modes; without the launch fix: SIGSEGV in the emulator (the over-read) |
| a function that runs a device loop and returns its array (`-> ret`) returns zeros or uninitialised bytes; pre-existing (60f0f8ce too) | `resident`'s `uses` counted only in-function readers; a returned value has none, so the loop exit (and any device output returned) was `keep` — never copied out, and a `devgen` array could be built on the device only | `resident.rs` `uses`: a value made in the function that escapes it (`last_use_plan`'s `escapes`: it reaches `Return` / `Output`) is a `Host` use — the caller reads a host value. The callee's loop stays device-carried; one `mapal_gpu_d2h` of the result at the exit | `caching_memory.rs` `caching_on_generated_programs`: `HistoryProg::callee` (the loop in a callee called once, or from `main`'s loop, each result the next call's input) and `params` (which inputs are parameters). Without the fix: `0 … NaN` against the interpreter |

## 1. The narrow operand's step

Repro (the review's `narrow_uniform.mapal`: QKᵀ over a `bf16` K cache, 2 KV groups, `narrow=b:bf16 radix=a ubase=a,b`)
on the RTX 5080 itself (`--gpu-off=place` so the small sites are device sites): interpreter `−158 −53 204 −211`;
HEAD `−63 52 89 −11` (resident) and `−13 46 90 33` (`resident` off); now the interpreter's values on every arm
(shipped, `place`, `place,resident`, `place,uniform-base`, `place,narrow-radix`). Host IR: `mul i64 %gg, 16` (was 32).

The review's alignment claim is not real: `uniform_aligned` is right in element units. A kernel's widest load of an
operand is `lanes = 16 / e` elements of its own storage — 16 bytes of `f32`, `<4 x bfloat>` (8 bytes, `align 8`) of a
narrow operand — and the kernels that copy 16 narrow bytes at once take no mixed-radix site (`narrow_streams`,
`half_operands`, `vec_fit` all require `site.radix.is_none()`). So `coef · 4 % 16 == 0` ⇔ `coef % lanes == 0` keeps both
aligned. The PTX of the repro: the narrow `kc` parameter `.align 2`, read by `ld.global.nc.b16`; the 16-byte
`ld.global.nc.v4.b32` are the f32 `q` panel's. The check now states it per operand (`(coef · e_o) % (lanes · e_o)`, the
same decisions). A `d = 4` head (an 8-byte `bf16` step) forms no narrow tile site in the generated programs (its
attention is a per-cell kernel), so that step is reasoned, not run on the hardware.

## 2. A callee's device loop

Repro (the review's `ONCE`: `inner` runs 3 trips of `wt` map, projection fold, `hy` carried; `(w0, h0) -> inner -> out`)
on the RTX 5080: interpreter `−33.25 39 −43.5`; HEAD `−0 −0 −0` (`place,calls` off), `−2 −76432.12 −76432.12` (`place`
off); now the interpreter's on every arm. `NESTED` (inner called from a 3-trip loop): `15072.133 252118.98 −183187.8` on
every arm (was uninitialised). Host IR of `inner`: the same `carried=0:1024:device h2d=0 d2h=0` trip, plus one
`call void @mapal_gpu_d2h(ptr %o1, …, i64 1024)` at the exit — the one copy the dataflow requires (the result leaves
the function).

Open: in a callee whose `g` (read by the per-row scale and the exit's sign cell) is a parameter, the history loop's
carries stay on the host (`carried=…:two / one`, the history crossing every trip) where the same loop in `main`, or
with `g` built in the callee, is device-carried. A parameter operand of a device cell is the missing piece; values are
right either way.

## Emission and time

The shipped emission of every `benches/**/*.mapal` program (448) is byte-identical to HEAD (less `; mapal.` comments,
host and device) on the rtx-5080, the h100 and the rtx-5080 with `place` off; `suite_diff.py --base HEAD`: 35 suite
programs + 2 cells × 5 faces the same — nothing to time (null by construction). With `--gpu-off=place,resident` (a
measurement arm) 48 `bf16` programs differ only in `mapal_gpu_tile`'s narrow operand bytes (`n · 2`, was `n · 4`).

## Tests

`uniform_base_on_generated_programs` (64 programs, 4 emulator modes, candidate picks, the CPU build, two OFF arms):
narrow-uniform-tiles 7, launch-path-uniform 1, launch-path-narrow 1, uniform-tiles 38, uniform-post-tiles 38,
uniform-splits 9, uniform-refused 26, carry-takes 24, candidate-runs 38, off-arm-runs 22.
`caching_on_generated_programs` (96 programs): callee-device-loops 38, called-in-a-loop 14, retained 223,
retained-cell-kernels 174, gathered-sites 33, same-cells 47, cone-shared 30, fit-smaller 1, off-arm-runs 32. Each
checked against the code without its fix (a toggle, removed): the narrow step — wrong values in every mode; the launch
path's sizes — SIGSEGV; the escape — `0 … NaN`. `mapal-backend-llvm` (44 binaries, 447 tests) and `mapal-rewrite`
(22 binaries, 99 tests) pass. (The first run hit 94 °C at `site_choice` and was stopped; the rest ran at 8 test
threads on 8 cores.)
