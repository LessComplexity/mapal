# regressions — the losers of the cross-arch 3a8e654f round, attacked

Branch `local-rtx-5080` on `55ba3e42`. The four regressions the rented round on the final code found
([cross-arch 3a8e654f](../cross-arch/README.md#3a8e654f-final-code-all-competitors), findings 1, 2, 4 and the worst
rows), each traced to the SASS / plan difference in that round's captures (`probe_kernels` per-kernel sums,
`facts.json`, the OFF / ON arms measured on the live boxes), fixed as a fact, property or realization the rule reads,
and **replayed**: the new emitter on each part's profile against the arm the box measured faster. The RTX 5080 times
only the programs the change moves (changed-programs ABBA, `gpu_changed.sh --base-emit <55ba3e42 emitter>`).
Raw: `.claude/raw-evidence/local-rtx-5080/regressions/` (`ab2/` the bf16 rows, `ab3/` the lsm row-block arm,
`replay/` the per-part emissions and PTX).

## 1. B1 decode, 1.2–4.3× slower on the 3090 / A100 / H100 — closed (a realization bug)

**Cause (captures).** The O projection with its residual epilogue (`h1 = x + o·wo`, a `tile-post` site) ran an
ordered 16-block kernel (`combine none, S 1, gx 16`): H100 235 µs a call against 25–29 µs in either OFF arm. The rule
had planned the split — `stream-fly` sizes a post site's workspace split (`split_stream`, partials inside the L2:
H100 S 64, A100 / 3090 S 32, 256 / 128 blocks, a block on every SM) — but the launch dropped it: a site with **one**
candidate launched outside its table, and that path runs an epilogue group's ordered kernel. On the H100 / A100
`staged-regs` leaves the site one candidate (its register-staged tiles refused), on the 3090 the candidates collapse
too; the 4090 / 5080 keep several and launched the split through the table (unchanged there).

**Fix.** `make_table`: a `tile-post` group whose candidate 0 is a split launches through its site table even alone
(the workspace reduce evaluates the epilogue once every sum is whole, `group_order`). **Intent**: `stream-fly` now
checks post sites on a free face too ("a block on every SM", the knee in flight); `evidence.py` takes a part's SMs /
threads / blocks / shared memory from its `TargetProfile` where `PARTS` has no row (the A100 / H100 had none, so
every `occ.sms` check held vacuously there — the shipped kernel passed 441 / 441).

**Replay** (`replay/r1.*`, the 55ba3e42 emitter → now; the post site's plan, the shipped launch):

| part | `dec_b1_l2048` 55ba3e42 | now | `_bf16w` 55ba3e42 | now | box: shipped → best OFF arm (ms / 16 steps) |
|---|---|---|---|---|---|
| h100 | ordered, gx 16, S 1 (16 blocks / 132 SMs) | split S 64 × 4 vector blocks (256) | ordered 16 | split S 64 × 16 (1024) | 5.208 → 2.025 / 1.908 |
| a100 | ordered 16 | split S 32 / 16 × 16 (512 / 256; one cell a thread by `vec-spread`, § 3) | ordered 16 | split S 32 × 16 | not armed (1.88× torch) |
| rtx-3090 | ordered 16 | split S 32 × 4 (128 / 82 SMs) | ordered 16 | split S 64 × 2 (128) | 5.195 → 4.287 |
| rtx-4090 / 5080 | split S 32 × 4 (table) | unchanged | split (table) | unchanged | 3.887 → 3.934 (null) |

Every planned split now launches (H100 / A100 / 3090: the OFF arms' fast kernel shape or better — 256 blocks, the
knee in flight). RTX 5080 `dec_b1_l2048` text unchanged (A/B null by construction; `_bf16w` moves only by § 2).
Tests: `gemv_stream.rs` `post_split_launches_through_its_table` (four parts × one / eight candidates: the split
launched, no ordered post kernel, the reduce carries the epilogue, the table called); the generated GEMV programs
count `post_split` (> 0). Open, not this step: `staged-regs`' panel-alone budget refusing tiles ptxas fits without a
spill (open item 2 of the round; the H100 `decode_1024_256`).

## 2. bf16 KV cache 1.33–1.75× the f32 cache at B ≥ 8 on the A100 / H100 — closed at its cause

**Separating property (captures).** H100 `dec_b8_l8192` per-kernel sums (16 steps): the bf16 cache's scores kernel
(`Q·Kᵀ`, a mixed-radix narrow site, `c 8192, k 128`) **10.64 ms** against the f32 cache's **4.10**; its `P·V` 1.20
against 1.95 (the halved bytes pay there). The scores kernel's PTX: **256 `ld.global.nc.b16`** a thread against the
f32 cache's 64 16-byte loads — the narrow operand's pointer was declared aligned to its element (`align 2`), so LLVM
formed no vector from the digit copies adjacent along `k`. 4× the f32 cache's load instructions at half its bytes:
hidden under GDDR's DRAM time (3090 / 4090 / 5080 0.60–0.88×), the bound where HBM feeds each SM 1.3–2.2× the
GDDR parts' share (A100 14.4, H100 25.4 against 11.4 KB/µs an SM). No choice between the caches is needed once the narrow read issues fewer
instructions than the f32 read.

**Fix (fact).** A narrow operand's pointer is an allocation's like any operand's (the arena's `ARENA_ALIGN`, a
device allocation's), so it carries the tile's 16-byte alignment; one a launch moves by a captured number
(`uniform-base`) carries the alignment of the whole register-staged vectors it moves by (`uniform_aligned`). The
scores kernel: **32 16-byte loads** (8 bf16 each) a thread — half the f32 cache's. **Validated**: every emulated
launch now checks every pointer alignment a kernel declares at its entry (`tests/emu`, `emu_align`: a claim that
does not hold aborts, as the device faults) — the whole backend suite passes under it; every bf16 row's value gate
passes on the 5080 (no misaligned-address fault).

**RTX 5080, changed programs** (8 of the bf16 rows; every bf16-storage candidate's text moves, the f32 candidates' does not —
`suite-candidates.tsv`): ABBA × 5, iter ms median (min), 55ba3e42 → now:

| program | 55ba3e42 | now | ×  | traced kernel × | null |
|---|---:|---:|---:|---:|---:|
| llm_dec_b8_l2048_bf16kv | 5.283 (4.715) | **4.112** (4.106) | **1.285** | 1.126 | 0.999 |
| llm_dec_b1_l2048_bf16kv | 2.585 (2.579) | **2.235** (2.233) | **1.157** | 1.136 | 1.002 |
| llm_dec_b32_l8192_bf16kv | 41.08 (40.47) | **38.37** (38.35) | **1.071** | 0.98–1.06 | 1.000 |
| llm_dec_b8_l8192_bf16kv | 11.95 (11.95) | 12.77 (**11.19**) | bimodal: min 1.068 | 1.082 | 0.999 |
| llm_blkdec_b1_l4096_bf16kv | 10.337 | **9.877** | **1.047** | 1.050 | 1.000 |
| llm_dec_b1_l2048_bf16w | 2.467 | 2.400 | 1.028 | 1.041 | 0.998 |
| pj_m1_n4096_k4096_bf16 | 0.0359 | 0.0365 | null (±20 %) | null | 1.026 |
| pj_m128_n6144_k4096_bf16 | 0.1209 | 0.1212 | null | 0.996 | 0.986 |

(The machine ran other agents' builds — load 57–69 —; `dec_b8_l8192` is bimodal on both arms, its traced kernel time
1.082.) The H100 / A100 gain is expected larger (their scores kernel was 2.6× the f32 cache's); no rented box this
step — the next round measures it.

## 3. `vec-cells` loses 3–20 % on the A100 / 4090 / H100 GEMVs — the missing property, mostly closed

**Separating property.** On the HBM parts the vector grid (a quarter of the scalar grid's blocks at the split
`stream-fly` gives it) spreads unevenly over the SMs: A100 126 / 112 / 112 / 112 vector blocks on 108 SMs — the most
loaded SM 1.71–1.93× the mean against the scalar grid's 1.07–1.21 — lost 0.93–0.97; 192 blocks (1.125 both) won
1.035. H100 144 / 192 / 224 blocks on 132 (1.18–1.83 against 1.03–1.15) lost 0.83–0.96; 256 (1.03 both) null. On
the 3090 / 5080 (11.4 KB/µs a SM) and the 4090 (7.9) the spread does not separate the arms (3090 null at every spread,
5080 won 1.03–1.42 at spreads to 1.75). Property: **the part's DRAM share an SM** (`dram_bytes_per_us_per_sm`) at or
above `DRAM_PER_SM_SPREAD` = 12 900 (between 11 414 and 14 398) asks the vector grid to spread no less evenly than
one cell a thread at the same split (`rules::grid_spread_pm`, `rules::vec_cells_pay`). OFF arm: `vec-spread`.

**Replay** — `rung_replay.tsv` 11 lines (`blocks=<vector>,scalar=<scalar>`): A100 3 picked off / 1 on / 1 null, H100
3 off / 2 null, 3090 null, 4090 mixed; `rung_rules_replay` 122 picked / 71 null / 10 mixed / 19 open. The emitter on
each part's profile (`replay/vec-plans.txt`): the six m = 1 projections now take the arm the box measured faster on
10 of 12 HBM cells (A100: scalar on the four losers, vector on `n6144`; H100: scalar on the three losers, vector at
256 blocks); the RTX 5080 unchanged (its share below the threshold: every GEMV's text equal, A/B null).
**Open**: H100 `n4096_k14336` won the vector grid by 3.5 % at the losers' spread (224 blocks; `OPEN` in the replay);
the 4090's three losses (0.80–0.97) are at equal spread, on the shapes whose 59–100 MB operand the 72 MB L2 holds
across the timed calls (1.1–1.5 TB/s, above DRAM) — a property of repeated calls, not of the program (replay line
`mixed`). Tests: `gemv_stream.rs` `vec_cells_spread_per_part` (A100 / H100 / 5080 on 1 × 6144 × 4096 and
1 × 18432 × 7168), the 64 generated GEMV programs under `vec-spread` OFF at 2 and 8 SMs (values in every emulator mode;
counters `spread_refused` / `spread_kept` > 0).

## 4. Small GEMVs (m ≤ 16) 2.1–2.2× cuBLASLt, 4090 `lsm_32_8192` 5.09× — explained, open

**The 4090's GEMVs.** From the round's probes, per call: iteration ms − the traced kernel ms = **15–29 µs on the 4090
box** (EPYC 7702P host) against 3–12 on the 5080, 5–11 on the A100, ~5 on the 3090, 12–20 on the H100 box. The
library legs are CUDA-event device time (`gemm_lib.cu`), which a launch-and-synchronize program's wall time is not.
Kernels alone: `pj_m16_n4096_k4096` 44 µs vs `lt-fresh` 32.8 (1.34×), `pj_m1_n4096_k4096` 26 vs 19.5 (1.33×) — on
the shapes whose 59–67 MB operand sits in the 4090's 72 MB L2 (`-fresh` regenerates it just before the clock:
2.0–3.4 TB/s), the same L2-resident class as § 3's 4090 losses. So 2.1–2.2× = 1.33× kernel × the box's per-call host
time; the DRAM-streamed shapes are 1.06–1.48× there.

**`lsm_32_8192` on the 4090.** 3 dependent launches (the scores tile; the softmax unit sliced 16 ways in 2 phases)
plus the box's per-call time: 41.3 µs against torch compile-cg's 8.2 (the 5080 runs the same 3 launches in 14.4–16.5).
**Attempted**: one kernel — the row-block unit the cost model refuses (`row_block=s:cost:b1:9394/2756` on the 5080),
forced (`--row-block=1`, ABBA on the 5080): **33.7 µs against 23.0** for the rules' split (`ab3/`): the refusal is
right — 32 rows are 32 blocks on 84 SMs. **Open (missing property)**: the host time a timed call pays per dependent
launch and synchronize on the box (no calibration measures it; the profile's `dep_launch_ns` is the device-side
gap), which the launch-count decisions (`row-block`, `calls`, `row-slices`' phases) would need to price.

## Tests

`cargo test -p mapal-backend-llvm --lib --tests` (44 test binaries + 115 unit tests): every one passes, the emulator
checking every declared pointer alignment at each launch's entry. New: `post_split_launches_through_its_table` (four
parts × one / eight candidates), `vec_cells_spread_per_part`; `narrow.rs` `realization_per_side` now asks the narrow
pointer for its kernel's operand alignment (was `align 2`). Generated GEMV programs (64, every emulator mode, the
`vec-cells` / `stream-fly` / `splitk-atomic` / `vec-spread` OFF arms at 1, 2, 6, 8 and 84 SMs): `post_split` 35,
`spread_refused` 1, `spread_kept` 12, the earlier counters 4–63. `rung_rules_replay` 122 picked / 71 null / 10
mixed / 19 open with the 11 new lines. Also fixed: `flow_units` counted an extended group's tile placed as a cell
kernel as an absorbed device launch (`attempt to subtract with overflow` in debug on the generated attention
programs, `flash_attention.rs`).
