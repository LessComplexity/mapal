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
3. `coop_body_ns` on the rtx-4090 / rtx-3090 / a100 / h100: their forced suite chains.
