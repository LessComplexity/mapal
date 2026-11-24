# sweep-runtime-2 — the second open-problems sweep: memory / launches / runtime / rules / harness (RTX 5080, 2026-11-24)

Branch `sprint/sweep-runtime-2` from `local-rtx-5080` @ `4c29387e` (integration 2). Machine shared with the other
sweep agents the whole time (load 40–55, Tctl 80–92 C from their work): every build, test and timed run of this step
pinned to CPUs 24-31,56-63 (`nice -n 5`, `CARGO_BUILD_JOBS=12 RUST_TEST_THREADS=8 MAPAL_PAR=2
MAPAL_TESTGEN_JOBS=4`), each GPU run under the bench lock, started below 85 C with no game running. These CPUs sit
on CCD 3 (CPPC ceiling 166): the device-bound rows below compare within this set, ABBA; host-bound rows are not
compared across sets. Raw output (not in git): `.claude/raw-evidence/local-rtx-5080/sweep-runtime-2/`
(`sk-epi/` the first cut, `sk-epi2/` the stream-K A/B, `confirm-s1/` and `confirm/` the two tune
sessions, `f8w/` the clock trace, `tp.out` the test run, `scripts/`).

## Items and status

Every item the first sweep (`sweep-runtime.md`) and the integration (`open-problems.md`) left in this area, ranked
by cost (time lost × rows × parts).

| # | item (source) | rows × parts | status |
| --- | --- | --- | --- |
| 1 | an epilogue group forfeits stream-K (sweep-runtime §1: fan / MLP 2048 rows 4–9 %, `absorb_pays` 9 / 13) | 8 rows × 4 parts | **closed, generic** (capability: the stream-K ticket; OFF `--gpu-off=streamk-epi`; §1) |
| 1b | found on the way: an ordered launch behind a stream-K plan launched the stream-K tile's geometry (k-fill sites, `--cands=1` / streams) | — | **closed** (`c113cc18`, §1) |
| 2 | the tune's pick varies between sessions (fp8's M3 in-context confirmation unbuilt; three rows) | every tuned row | **closed, generic** (harness step `confirm_pick.py`; OFF `CONFIRM=0`; §2) |
| 3 | `pdl`'s sign on short programs (hardware-properties open 1) | 3 rows | **closed earlier, verified**: `pdl_pays` reads `launches_repeat` — mlp_512 / fir / convmc take no PDL (`mapal_gpu_init(i32 9)`, 0 `griddepcontrol`), decode does (13, 24 waits); the open-problems row was stale |
| 4 | `calls` pricing (cross-arch open 6: rounds, host issue, the 4090 block decode formed but not realized) | 6 rows × 3 parts | **closed earlier, verified**: the price carries `rounds` and `issue_ns` (`; mapal.calls.host … rounds=1 issue_ns=2850 priced=…`); the 4090 block decode's formed chain launches cooperatively (1 `mapal_gpu_launch_coop`); 3a8e654f measured every causal / decode decision's sign right on four parts |
| 5 | `llm_dec_b1_l2048_f8w` 0.892 vs 1.15 ms on equal code (integration 2) | 1 row | attempted-open: re-measured 0.894 on integration 2's binary; the GPU's P-state / clocks now recorded through every timed run (§3) |
| 6 | the ~1 µs runtime residual between equal-set binaries (1–5 %) | ~20 small rows | open: not reached (needs a host-issue trace per launch on idle CPUs; the shared machine never idled) |
| 7 | convmc 1.68–2.25× / loopmm 1024³ 1.11–1.37 on the rented parts | 2 rows × 4 parts | open: not reached (tile / window geometry on parts this step cannot measure; no rented round) |

No tailored fix; nothing for `docs/tailored-fixes.md`.

## 1. An epilogue group keeps its site's stream-K walk (closed)

**Cause (sweep-runtime §1, SASS / plan evidence).** An S54 epilogue group ran `Order::Sequential`: its site took
the ordered tile kernel wherever the rules streamed the same site alone. `indep_2048`'s grouped site
(`g = c + d` in its store) ran `combine=none` at 0.531 ms against the plain site's `streamk 168 × 196` at 0.456 ms:
256 tiles on 168 slots, the ordered kernel idles half its second wave. The rule that would have split the group
(`absorb_pays`) replayed 9 / 13 — the split plus a member launch loses at 4096 and 512.

**Capability (`gpu.rs`, `Kernel::ticket_split` / `Kernel::ticket`, runtime `site_run`).** The group takes the walk
its site's stream-K rule gives (`epi_order`: its atomic and workspace splits stay off — one launch either way):

- a piece that is its tile's whole fold (`⌊t·P/iters⌋ = ⌊((t+1)·P − 1)/iters⌋`: one block) stores through the
  members as the ordered kernel does — no atomics, no ticket;
- a piece of a shared tile adds its partials into the tile's workspace slot — slot `c₀`, the tile's first block (a
  block starts at most one tile it does not finish, so the slots are distinct and the workspace is `ctas` tiles:
  11 MB at 4096², not the 64 MB output) — then draws the tile's ticket: release fence, barrier, thread 0's
  `atom.acq_rel.gpu.add.u32`, the old count through shared memory; the block drawing the last ticket reads the slot
  (volatile, after an acquire fence) and evaluates the members on the whole sums;
- the runtime zeroes the workspace of a recipe with `ws` and no reduce kernel and appends it after the table's
  pointers (`mapal_gpu.c`, the emulator alike).

First cut (every piece through the workspace, the output-sized workspace zeroed per call): 2048 rows won 2–6 %,
the two 4096 rows lost 1.4–1.5 % — a mixed result. What separated them: at 6.1 waves (4096²) the walk's saved
iterations are small against a 64 MB zero pass and 64 MB of atomics, at 1.52 waves (2048²) large. Whole tiles
stored directly and the slot-sized workspace removed the cost the walk added; every row then won.

**Intent vs realized (bundle `sk-epi2`, ptxas / PTX / SASS).** Ticket kernel at 2048 / 4096: 233 / 236 registers,
0 spill (the plain stream-K kernel 233), one `atom.acq_rel.gpu.global.add.u32` a piece, `fence.release.gpu` /
`fence.acquire.gpu`, the partials `atom.relaxed.gpu.global.add.f32` (RED in SASS); host recipe
`{ id, rid 0, zero 0, ws 11 010 720 (4096²) / 16 778 240 B before the slots }`. intent.py `streamk-epi` (new):
pieces add, one ticket a piece, one wave, no local memory in the hot loop. Evidence run: 9 609 checks, 175
mismatch (the pre-existing classes; 173 on the first cut), 0 in `streamk-epi`.

**Time (changed programs, ABBA 5 reps; 12 programs differ, 3 of them equal up to kernel names; null ≤ 1.2 %).**
(Corrected by review-fixes-2: the run passed `REPS` 8 to `gpu_run.sh`, but the probe section repeats
`PROBE_REPS` times, default 5 — the raw `sk-epi` / `sk-epi2` results hold rep 1–5 of every program and arm; the
medians below are those 5.)

| workload | OFF (`--gpu-off=streamk-epi`) | ON | fastest competitor (leg) | explanation |
| --- | ---: | ---: | ---: | --- |
| rt_mlp2_2048 | 16.24 | **14.71** (0.906) | 15.55 (torch-compile-cg) | two 2048 groups a trip walk stream-K; now 0.95× torch |
| fanout_2048 | 0.9886 | **0.9265** (0.937) | 0.9154 (chain-streams) | both grouped sites stream (256 tiles / 168 slots) |
| mlp2_2048 | 1.9501 | **1.8332** (0.940) | 1.769 (chain-lt2) | the bias + relu groups stream |
| mlp_2048 | 0.9464 | **0.8921** (0.943) | 0.9134 (chain-lt) | now 0.98× cuBLASLt's chain |
| loop_4096 | 28.41 | **27.03** (0.951) | 28.55 (chain-lt) | 6.1 waves: whole tiles stored, 1 shared tile a block |
| indep_2048 | 1.0102 | **0.9634** (0.954) | 0.9196 (torch-compile) | the join's group streams |
| diamond_2048 | 1.4505 | **1.3979** (0.964) | 1.368 (torch-compile-cg) | — |
| epi_4096 | 3.6829 | **3.5529** (0.965) | 3.545 (torch-compile-ro) | now even with torch |
| smlp_4096_256_1024 | 0.1668 | 0.1667 | — | null (its 4096 × 256 group site streams; the walk's saving is within noise) |
| indep_512, mlp_512, rt_mlp2_1024 | = | = | — | equal up to kernel names (their sites do not stream) |

Matrix 5080 cells these rows held (integration 2): 1.08 / 1.09 / 1.05 / 1.04 / 1.10 / 1.05 / 1.04 / 0.99 → 1.01 /
1.05 / 1.02 / **0.98** / 1.04 / **0.95** / 1.00 / **0.95** (same order as the table: fanout, indep, diamond, mlp,
mlp2, rt_mlp2, epi, loop).

**Other parts (replay).** The ticket applies where the site's stream-K rule streams it: the emitter gives it to 8–9
suite programs each on the RTX 3090 / 5090 / H100 (the same rows), none on the RTX 4090 and A100 (their filled
class does not stream). The rented rounds' OFF arm of §1 is the epilogue-off arm (split + member launch), which this
kernel dominates (no member launch, no output-sized zero); measured on the 5080 only — the next rented round's
`--gpu-off=streamk-epi` arm is the replay line.

**Tests.** `tests/streamk_epi.rs` (new): generated GEMMs (`testgen::gemm_epi_src_at`, four chain kinds: affine,
bias + relu, a join, two members with the site's own output stored) — plans at the generator's extents: 35 groups,
**23 stream-K tickets exactly where the site alone streams, 12 ordered**, none under OFF; values through the
emulator (free, `MAPAL_EMU_SERIAL=1`, `=2`, `MAPAL_EMU_REVERSE=1`) on the 5080 (k-fill: every tile shared) and on
its properties with 2 SMs (whole tiles beside shared ones): 4 ticket programs, 10 whole tiles, 13 shared, 12
emulated runs, all equal to the interpreter and to the OFF arm. The OFF arm's emulated reference exposed 1b: at
`--cands=1` an ordered group kernel at a k-fill site launched the full tile's grid and block over its shrunk kernel
(SIGSEGV; on a device, wrong cells) — the fallthrough launch now re-derives the unsplit geometry for any combine.
`site_choice` golden: the 6 rows whose group now streams (smlp / epi_2048 on the 5090, 3090, H100 PCIe).

## 2. The tune's pick confirmed in its program (closed)

**Cause.** The trial (`mapal_choice.h`) times a site's candidates alone — back to back, the L2 evicted where the
site's operand lives in DRAM — and keeps a challenger beaten by > max(3 %, 1 µs) after three head-to-head pairs. It
never sees the program around the site, and two sessions' trials picked differently on equal code (integration 2:
`llm_dec_b32_l2048` 1.047, one site's candidate 3 against 0; two earlier rows).

**Step (harness, generic: `benches/gpu/confirm_pick.py`, `box_run.sh` after each stem's tune run).** Each pick the
trial made is run in its program: the program with the pick against the same wisdom with that site's candidate 0
appended (the last line of a key wins), three alternating pairs (0, w), (w, 0), (0, w) of whole processes. The pick
stays where the median pair ratio w / 0 is under 1 − 1 % and most pairs agree; otherwise candidate 0 — the rules'
plan, the same in every session — is appended to the wisdom. OFF: `CONFIRM=0` (the trial's pick stands). The
median, not the mean: one disturbed process (a 40 µs program ran 4.19× in one pair) must not decide.

**Measured (RTX 5080, two sessions, `RETUNE=1`, five programs with tuned sites; `confirm-s1/`, `confirm/`).**

| program | site | trial pick | session 1 pairs (w / 0) | session 2 pairs | decision (both sessions) |
| --- | --- | ---: | --- | --- | --- |
| indep_512 | 6dbed9ba / e3cd54db | 5 / 4 | 0.761 0.767 0.756 / 0.943 0.942 0.940 | 0.73 / 0.94 (med) | keep / keep |
| mlp_512 | fd914c30 / b2329fab | 4 / 5 | 0.940 0.953 0.955 / 0.906 0.901 0.909 | 0.935 0.944 **4.188** / 0.93 | keep / keep |
| rt_mlp2_1024 | 14c19377 / 420f3471 / e5dfd648 | 4 / 4 / 4 | 0.984–0.986 / 0.988 0.988 1.054 / 0.985 | 0.986 / 0.985 0.895 0.987 / 0.947 | keep (3) |
| smlp_4096_256_1024 | 23fa5d12 | 3 | 0.876 0.869 0.873 | 0.888 | keep |
| llm_dec_b32_l2048 | c08176c1 | 3 | 0.956 0.955 0.956 | 0.950 | keep (in context 4.5 % faster) |
| llm_dec_b32_l2048 | ee74b398 | 1 (session 1 only) | 1.027 0.976 1.000 | — (trial kept 0) | **revert** → 0 |

Final picks are equal across the two sessions (9 kept, 1 reverted where only one session's trial took it). The first
session ran the geometric-mean form (it would have reverted mlp_512's fd914c30 on the 4.188 outlier and rt_mlp2's
420f3471 on one 1.054 pair, keeping each in the other session — the two sites the median makes agree); the table's
decisions are the median rule replayed on both sessions' recorded pairs. The variance row's site (c08176c1) keeps
candidate 3 because the program measures it 4.5 % faster in both sessions (which candidate integration 2's slower
session ran there is not recorded; from now the program's own measurement decides, and the log says so).

**Tests.** `test_harness.py` `test_confirm_pick_keeps_a_program_win_and_reverts_a_loss`: a fake program whose `iter
ms` follows the wisdom keeps the win, reverts the loss (candidate 0 appended, the other key untouched), skips a key
the wisdom decided (`src=wisdom`).

## 3. `llm_dec_b1_l2048_f8w`'s session gap (attempted, open)

**Measured.** Integration 2's own binary (`bM/pt_pt_llm_dec_b1_l2048_f8w`, its wisdom and PTX), 5 processes under
the lock with the GPU sampled every 100 ms: **0.892–0.896 ms** — the 0.892 state, not the 1.15 ms both of
integration 2's bundles ran. Between processes the GPU sat at **P8, 210 MHz SM / 405 MHz memory**; under the
program P1 / P0 at 2 677–2 685 MHz SM and 14 801–15 001 MHz memory. So the gap follows a machine state the run
conditions did not record — what clocks the GPU holds while a short program's iterations run (the fp8 working set
of 42 MB in a 64 MB L2 that the display shares is the other candidate).

**Changed (generic, harness).** `gpu_run.sh` now samples the P-state, SM / memory clocks, utilization and power
5 a second through every timed run (`results/gpu-clocks.csv`, summary `gpu-clocks.txt`: samples per state with
the GPU busy). Status: attempted-open — the 1.15 ms state did not recur here, so the next run that shows it will
say whether its iterations ran below P0 / P1 or at full clocks (then the L2 share).

## Tests (workspace)

Full workspace through `test_par.py` (2 binaries at once on this step's 16 CPUs, Tctl-guarded: the workers
waited 1 304 s of 2 012 s for the other agents' heat): **126 binaries, 1 525 passed, 1 failed, 7 ignored** — the
failure `calls::every_part_prices_chains` expected a `calls` line for smlp on the RTX 5090, whose grouped site now
streams (a stream-K site is no chain member, as a plain one): the test now accepts an empty line exactly where a
grouped stream-K kernel exists, and passes (1 526 with it). New: `streamk_epi.rs` (2 tests: 35 generated groups —
23 tickets, 12 ordered; 4 ticket programs valued, 10 whole + 13 shared tiles, 12 emulated runs in three modes),
`test_confirm_pick_keeps_a_program_win_and_reverts_a_loss`. Every GPU value gate passed (12 + 5 programs, every
arm within its bound; the order-free face's ticket sums differ from the ordered kernel's in the last bits, as any
stream-K walk's).

## For integration

- README (local-rtx-5080, after **sweep-runtime**): **sweep-runtime-2** (`sweep-runtime-2.md`): an epilogue group
  keeps its site's stream-K walk (`streamk-epi`: whole tiles stored, shared tiles through a ticket) — rt_mlp2_2048
  0.906, fanout 0.937, mlp2 0.940, mlp 0.943, loop_4096 0.951, indep 0.954, diamond 0.964, epi_4096 0.965 (mlp_2048,
  rt_mlp2_2048 and loop_4096 now ahead of their fastest competitor); the tune's trial picks confirmed in the program
  (`confirm_pick.py`); an ordered launch behind a stream-K plan fixed; the GPU's P-state and clocks recorded
  through every timed run (`gpu_run.sh` → `gpu-clocks.txt`).
- CLAUDE.md facts table: no new fact (the ticket is a capability of the stream-K kernel; the walk's whole / shared
  tiles are arithmetic on `ctas`, `iters`, `P` of the launch the rules already give).
- `docs/tailored-fixes.md`: no row.
- `open-problems.md`: rows "epilogue groups forfeit stream-K" → closed (`streamk-epi`); "tune's pick varies" →
  closed (`confirm_pick.py`); "`pdl`'s sign", "`calls` pricing" → closed earlier (verified, §Items 3–4);
  "`llm_dec_b1_l2048_f8w`" → attempted-open (0.894 re-measured; clocks now recorded); "~1 µs residual", "convmc /
  loopmm on the rented parts" → open (not reached).
- The full GPU suite (integration): a bundle built from this branch carries `confirm_pick.py`; tuned programs log
  `confirm …` lines after their `tune` line.
