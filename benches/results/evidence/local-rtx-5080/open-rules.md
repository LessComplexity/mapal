# open-rules — the cross-arch open rules against their replay (RTX 5080 box, 2026-11-23)

Branch `local-rtx-5080` after `5dccfc07` (the cross-arch 60f0f8ce round's section is in
`../cross-arch/README.md`). No rentals: every rule change is validated by **replay** against the measured arms of the
cross-arch rounds (7a2b67bf, 60f0f8ce; raw `.claude/raw-evidence/cross-arch/<round>/`) and the RTX 5080's, by property
tests over generated programs, and by the RTX 5080's emission (no timed suite program changes code on this part, below).
Scratch and replay scripts: `.claude/raw-evidence/local-rtx-5080/open-rules/`.

| item | what the round found | fact / property now | OFF arm | replay before → after |
| --- | --- | --- | --- | --- |
| (b) bug | 4090 block decode records `formed=1` but launches no cooperative kernel (ON = OFF) | planning drew a `tile-post` site's candidates under the wrong arms and order: the emission's table holds only workspace splits (an epilogue group splits into a workspace only, `group_order`), the plan priced its atomic candidates; the merged tuples came out empty and every member launched alone. Fix: one function, `gpu::epi_order`, for both (`FnEmit::site_choice`). Second cause found by the sweep: a merged unit whose member before its last copies out (`px_mlp2_*`): the planner tested the target only (`hz_copied`: target, unit and epilogue stores, the emission's own keep test); such a member now ends its `horizontal` group and a `calls` unit with it is refused (`copied`) | `--gpu-off=calls`, `--gpu-off=horizontal` | 2990 emissions (299 programs × 5 parts × shipped / `--calls=force`): formed = cooperative launches in every one (was: 4090 block decode ×3, 3090 block decode, h100 px block prefill, `px_mlp2_*` forced on 4 parts) |
| (a)/(5) `calls` | the body price (5080's 4-round body at the part's clock) refuses decode's one-round chain on the 3090 / 4090 (it won 1.48 / 1.20 formed); measured body / derived 0.48 … 1.89, every one-round chain ≈ 0 | **rounds**: the chain's widest level in rounds of its persistent grid (`chain_rounds`) — one round prices no body term, the grid-stride form the part's body; **phase length**: 22 ‰ (`CALLS_PHASE_PM`, the 5080's rt_mlp2 GEMM pairs +2.2 %, the smallest measured) of the members' work at the part's peaks (flops / FFMA rate or bytes / DRAM rate); **every part priced**: unmeasured boundaries derived from the 5080's cycles at the part's clock (`Gpu::chain_cost`, `CHAIN_CYCLES_RTX_5080`: 5090 562 / 660 / 494 / 2078 ns) | `--gpu-off=calls`; `--calls=force` | 21 measured chains: 17 → **18** signs right + 1 null (below) |
| (c) intents | `exp-hw` / `fma-pairs` / `calls-kernel` checked on exact-face builds; `pdl` expects the entry trigger (876 H100 mismatches) | `evidence.py build_face`: the face from the host line (`fma=`, `order=`), not the stem (`.xref` and the px differentials are exact builds); `pdl`: griddepcontrol iff the module is programmatic (`host.init_flags & 4`: its launches repeat); `calls-kernel`: barriers counted in the emitted IR (`mod.gsync == levels − 1`), PTX ≥ (opt specializes one arrival on a disjoint path: the exact chains' 8th, `i32 1`) | — | H100 `pdl`: 89 / 89 stems consistent (was 876 kernel mismatches); exact-face `exp-hw` 264–345 and `fma-pairs` 3 042 rows leave scope; `calls-kernel` 64 → 0 by the IR count (`opt -O3` of the a100 `.xref` chain: 7 calls → 8 PTX arrivals, one `i32 1` copy) |
| (d) `row-slices` | no lsm row split on any rented part | the rule needs `row_iter_ns`, `lat_ns`, `coop_first_ns`, measured on the 5080 only: unmeasured → derived (`Gpu::slice_cost`, `SLICE_RTX_5080`: the walk and latency in the 5080's cycles at the part's clock, the first cooperative launch as measured) | `--gpu-off=row-slices` | lsm 1×131072 / 4×32768 / 32×8192: rented parts 1 / 1 / 1 block a row → 64 / 64 / 8–32 slices (5080 unchanged: 64 / 64 / 8) |
| (1) `wide-fma` | FFMA per DRAM byte does not separate the parts (3090 wins 12 / 0 shipped off; H100 mixed; A100 loses) | part: **FP32 lanes per SM** (128; the A100's 64-lane SM) and a scarce DRAM share or a contracting face; site (`rules::wide_fma_pays`): where an SM's DRAM share ≥ 14.6 KB/µs (`DRAM_PER_SM`: the H100s) the wide tile's grid must exceed one wave of it (`blocks > sms × blocks_per_sm`) | `--gpu-off=wide-fma`, `--gpu-on=wide-fma` (forces it past the site price) | `rung_replay.tsv` wide-fma 14 picked / 1 null / 1 mixed / 2 open (was 5 / 0 / 1 / 0 of 6); the new H100 lines 5 / 5 by grid |
| (2) `min-blocks` | H100 18 L / 7 W, A100 6 L / 2 W | open (below) | `--gpu-on=min-blocks` | 3 lines open, unchanged |
| (3) `warp-fill` | mixed on 3090 / A100 / H100 | open (below): at one warp a scheduler the shrink trades the atomic split for an unsplit tile | `--gpu-on/off=warp-fill` | 3 mixed, unchanged |
| (4) `grid-share` / `panel-regs` | 3090 L2-per-SM threshold | open (below) | arms | unchanged |
| (6) tune | 4090 block decode 1.126 × rules, A100 prefill T8192 1.338 | open (below) | `MAPAL_GPU_PICK` | unchanged |

`rung_replay.tsv` + `rules::tests::rung_rules_replay`: **211 lines — 116 picked, 68 null, 9 mixed, 18 open** (was 200:
108 / 67 / 9 / 16). New: the 7a2b67bf round's wide-fma arms (8 lines); the H100 PCIe line split by face and size.

## (b) Formed but not realized

`calls_close` priced a chain whose members include the block decode's gate / up and down projections — `tile-post`
sites (the residual and the activation after the fold). The plan drew their candidates as `self.order` / the free
arms found them (atomic split-K: `zero_ok`), the emission as the epilogue group draws them (`group_order`: atomic and
stream-K off, so a free-order split is a workspace split, `ws > 0`). `hz_tuples` keeps only `ws == 0` rows: none, so
`hz_launch` fell back to the members' own launches at the last member — `formed=1`, the launch order deferred, no
`launch_coop`, PTX = OFF's. Now `hz_shape`, `calls_place` and `calls_c0_atomic` draw through `site_choice` (the same
`epi_order` as the emission): a post site whose rule candidate is a workspace split is no member. The 4090 block decode
now refuses both chains by cost (8 332 vs 11 520 ns priced → `refused=cost:2` once the post sites leave). The sweep
found the second cause (`px_mlp2_64_f64`, `px_mlp2_128`, forced): a `horizontal` unit `{d1, c2}` whose first member's
output is copied to the host right after it; the copy is a runtime call between the members and aborts the group.

Test: `calls.rs calls_on_generated_programs` asserts **cooperative launches = formed chains** on every generated
program; `testgen::HzProg::post` adds `tile-post` sites (`u · 0.5 + q[i · k]`) to `calls_prog_strategy`. Reverting
`site_choice` to the old planning view fails it (program #3: formed 1, 0 cooperative launches).

## (a) `calls`: rounds, phases, every part priced

Replay (`calls_replay.sh`: each measured chain emitted by HEAD and by this tree on the part's profile; W the chain won,
L lost):

| part | program | measured | HEAD | now (per / sep ns, rounds) | |
| --- | --- | --- | --- | --- | --- |
| rtx-3090 | causal / gen | W 1.356 / 1.359 (60f0f8ce OFF) | formed | formed (5 667 / 8 832, 1) | ✓ |
| rtx-3090 | decode_1024_256 | W (7a2b67bf formed 17.66 vs refused 26.10 ms) | refused | **formed** (4 124 / 4 416, 1) | ✓ (was ✗) |
| rtx-4090 | causal | W 1.164 | formed | formed (5 366 / 8 640, 1) | ✓ |
| rtx-4090 | decode_1024_256 | W (12.23 vs 14.73 ms) | refused | **formed** (3 859 / 4 320, 1) | ✓ (was ✗) |
| a100 | causal / gen | L 1.013 / 1.015 (forced) | refused | formed (10 612 / 12 300, 1) | ✗ open |
| h100 | causal / gen | L 1.105 / 1.110 (forced) | refused | refused (PDL launches) | ✓ |
| a100 | llm_dec_b1 l2048 / l8192 | L (refusal 0.89 / 0.91) | refused | refused | ✓ |
| rtx-4090 | diamond_2048 | L (refusal 0.915) | refused | refused (3 phases of 2048³: 22 ‰ of 624 µs) | ✓ |
| h100 | smlp / firew_rowsum | L (refusal 0.68 / 0.52) | refused | refused (phases ≥ 70 µs; firew 16 rounds) | ✓ |
| h100 | mlp / mlp2 TF32 | L (0.89 / 0.93) | refused | refused (phases) | ✓ |
| h100 | lsm_32_8192 | L (0.58; the 7a2b67bf arm's own spread 0.93) | refused | formed (2 787 / 3 444, 1) | ✗ open |
| rtx-5080 | decode / causal / rt_mlp2 | L 1.447 / 1.106 / 1.040 (forced) | refused | refused | ✓ |
| rtx-3090 | block decode | null 1.004 (forced) | refused | refused | — |

18 / 20 signs right (HEAD 18 / 20 with the other two wrong: the two decode losses of 1.20–1.48× against a marginal
1.3–1.7 % A100 loss and an outlier on a noisy box). What separates the shapes is what the round named: the one-round
chains (`calib.cu body`: −0.37 … +0.07 µs on four parts) pay no body term; the grid-stride form pays the part's
4-round measurement; long phases pay in proportion (the H100 smlp chain: two 256-block GEMM levels at one round, 1.47×
slower formed — the boundaries save 695 ns, the phases' 22 ‰ is ≥ 1 540). **Open:** (i) the A100's one-round causal
chain runs its phases +10 % inside the persistent kernel (41.3 vs 37.5 µs a trip; the calib chain does not show it);
(ii) the host's issue time per launch (`DeviceCost::issue_ns`; the EPYC hosts' decode trips were host-bound,
19.9–25.5 µs against 16.9 µs of kernels) is not a price input: the decisions above replay without it, and modelling
it needs the members' device time, which the phase term bounds from below only; (iii) the 5080's 2.048 µs
cooperative-dispatch quantum (`review-fixes.md` §5) is measured on the 5080 only — `coop_launch_ns` prices one quantum.

Derived boundaries (rtx-5090, h100-pcie; the 5080's cycles at the part's clock): both now price chains (5090: 562 /
660 / 494 / 2 078 ns barrier / dependent / programmatic / cooperative) and refuse the suite's (decode, causal, rt_mlp2)
as the 5080 does. `; mapal.calls.host` gains `rounds=`.

## (1) `wide-fma`

The wide tile (`8 × 16`, 128 threads) and the base one (`8 × 8`, 256 threads) cover the same `128 × 128` block: the
grid is the same, the wide one has half the warps a SM. Geometry of the measured sites (`widegeo.py`, this tree's
plan lines; ON/OFF from 7a2b67bf):

| part | ON wins | ON loses | separating |
| --- | --- | --- | --- |
| rtx-3090 (11.4 KB/µs/SM) | 12 / 25: mlp_2048 0.934 (1.56 waves), pj_m128_n28672 0.937 (stream-K, 1 wave), the m2048 projections | none | DRAM share < 14.6: any grid |
| rtx-4090 (7.9, scarce) | 19 / 27 incl. pj_m128_n28672 0.944 at 224 blocks of a 256 wave | none | scarce |
| h100 (25.4) | stream-K m2048 projections (512–3 584 blocks / 264) 0.934–0.964; 4096³ (1 024) 0.970 / 0.973 | mlp_2048 / mlp2 (256 / 264) 1.037 / 1.052; pj_m128_n28672 (224) 1.167 | blocks > one wave |
| a100 (64 lanes) | pj_m2048_n28672 0.752 (18 % null), pre T2048 0.958 | 9: 1.025–1.306 | FP32 lanes (FFMA issue = half its shared-load issue) |

The H100 smlp (0.727 ON) is confounded: its ON arm also dissolved the 7a2b67bf calls chain (see (a)). The H100 PCIe
lines (the S5x register-staged tile: 2048² / 4096² lost above one wave, 8192² +1.5 %) stay open: the SXM part's
current tile won 4096³ at 3.88 waves; re-measure the PCIe part on the current tile.

## (d) `row-slices`

The lsm programs stayed one block a row on every rented part because `row_slices` returned at `row_iter_ns == 0`.
Derived from the 5080's cycles at the part's clock (87 ns × 2.45 GHz = 213 cycles a lane-iteration; `lat_ns` 1 404
cycles), the first cooperative launch as measured (38 µs, host): 3090 / a100 / 4090 / h100 now slice 1×131072 and
4×32768 into 64 and 32×8192 into 8 / 16 / 16 / 32. The 5080's emission is unchanged. Timing on the parts: next round.

## (c) Evidence tooling

`build_face(host, stem)` (`evidence.py`): `contract` / `free` from `; mapal.host fma= order=`; the stem suffix only
where no host line. `mod.gsync`: `@mapal_gsync` calls per persistent kernel in the emitted device IR. Intents: `pdl`
(programmatic module ⇔ griddepcontrol), `calls-kernel` (IR count exact, PTX at least). Replay on the 60f0f8ce H100
rows: 89 / 89 stems whose kernels carry griddepcontrol exactly where the module's `PDL` init flag is set (17 stems are
generated cells / px programs, not in `benches/algorithms`).

## Open, with what the data separates

- **`min-blocks`** — per site: the derived residency's register cap (`regs_per_sm / (blocks · threads)`) against the
  tile's own registers. The rented rounds have no per-site ON/OFF ptxas data on the current tiles (the 8bba2a50 arms
  predate k-fill / stream-K fill / gemv sites); measure `--gpu-on=min-blocks` with the evidence pass next round.
- **`warp-fill`** — on the current geometry (`wfill.py`) the shrink at one warp a scheduler trades an atomic split
  (256 blocks of 128) for an unsplit tile (128 of 256): attn (k = 64) ON wins 0.675 / 0.812 on the 3090 / A100;
  loopmm (k = 1024) OFF wins 1.054 / 1.167 there and ON wins 0.80 on the 5080 (its `fill_balance`). The separating
  fact candidate: the split's fold length per partition. The m16 projections' sites changed since (stream sites).
- **`grid-share` / `panel-regs` (3090)** — 7a2b67bf: ON wins decode B8 0.493 / 0.710 (device-geometry kernels) where
  8bba2a50's OFF won the m16 projections 1.22–2.20: per-site, not the L2-per-SM threshold; re-measure on this tree.
- **tune (6)** — the trial's steady state on the 4090 block decode (1.126) and the A100 prefill T8192 (1.338):
  `tune-<part>.tgz` in the raw rounds; not replayed here.
- `calls`: (i)–(iii) above.

## RTX 5080: this step's emission against HEAD

All 299 `benches/algorithms` programs, default and exact faces (`diff5080.sh`, host less `; mapal.calls.host`):
**no device text changes**; host text differs in 6 builds — `pt_prefill_1024_512_256` (only its `; mapal.hz.host`
intent line: one launch fewer counted as a `horizontal` shape), and 5 untimed differentials (`px_mlp2_64_f64` ×2,
`px_mlp2_128`, `px_twin_128`, `share_256`, exact face: a `horizontal` group now ends at the member whose output is
copied out). No timed suite program's code changes on this part: the A/B is null by construction (equal text, the
same objects); nothing to time. The rule changes act on the rented parts' emissions (3090 130, H100 108, A100 16,
4090 15 of the 131 suite / model-scale programs change text, most of them kernel names: a kernel's name hashes its
rung set) — the next cross-arch round times them.

## Static evidence on the rented parts' emissions

`integrated_bundle.sh` (no arms) for rtx-3090 / a100 / rtx-4090 / h100 on decode_1024, causal, mlp_2048, lsm_1_131072,
block decode, pj_m128_n28672, smlp; `evidence.py static / facts / check` (ptxas -v, SASS for each part's SM): the
intents this step touches all hold — `calls` 6 / 6, 4 / 4, 6 / 6 (3090 / a100 / 4090: the decode and causal chains
launch cooperatively), `calls-kernel` 30 / 30, 10 / 10, 30 / 30, `row-slices` 15 / 15 / 5 / 15, `pdl` 88 /
88 (h100), `exp-hw` and `fma-pairs` 100 % (no exact-face misreads). `arch-rungs` now excepts the site-priced rungs
(`row-split`, `wide-fma`: off at a site where the part's rung is on) and the rungs list carries gemv-tma's
`vec-cells` / `stream-fly` / `bulk-stage` (391 rows in 60f0f8ce; here 2–4 a part left, the `tile-post` groups' split
without atomics — pre-existing). The `wide-fma:on` misses (1–6) are site-table alternatives where ptxas gave fewer
registers than the plan holds — pre-existing class. Tables: `static/evidence-<part>.tsv` in the raw directory.

## Tests

| test | result |
| --- | --- |
| `rules::tests::rung_rules_replay` | 211 lines: 116 picked / 68 null / 9 mixed / 18 open, the open set asserted |
| `calls` unit tests | `the_rule_prices_barriers_against_launches` (rounds, phases, derived 5090 boundaries), `the_rounds_and_phases_decide_where_the_boundaries_alone_did_not` (the replay's chains) |
| `tests/calls.rs` | 4 / 4; `calls_on_generated_programs` 48 programs: 43 chains formed = 43 cooperative launches, 25 three-level, 37 zero passes, 11 with `tile-post` sites, 400 emulator runs (picks 0 / 3, grids 1 / 3, `MAPAL_EMU_SERIAL=1`, `=2`, `MAPAL_EMU_REVERSE=1`) |
| `tests/hardware_properties.rs` | 2 / 2; the rung rules on 36 generated programs on drawn parts: sides (lo / hi) DRAM-scarce 14 / 22, regs 15 / 21, L2 24 / 12, DRAM ≥ 14.6 22 / 14, **FP32 lanes < 128 14 / 22** (asserted: a 64-lane part's rules pick no wide tile); wide-fma sites on 9 / **priced off by their grid 1** (asserted > 0); 36 emissions moved; 144 emulator runs (free, `MAPAL_EMU_SERIAL=1`, `=2`, `MAPAL_EMU_REVERSE=1`) equal the every-rung arm |
| `tests/long_rows.rs`, `resident.rs`, `site_choice.rs` | the 4090's one row of 131072 slices (derived walk), OFF none; two host goldens gain the `; mapal.calls.host` line (a generic GPU now prices its chain: refused); `c0-rules.tsv` regolded 77 / 204 rows (the line's `rounds=`, the derived boundaries on rtx-5090 / h100-pcie, wide-fma on rtx-3090 / h100-pcie, calls on rtx-3090 / 4090) |
| backend suite | 447 tests in 44 binaries (lib + 43 integration): 443 passed and the 4 above updated (then passed) |
