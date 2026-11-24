# gapfix-rows-operands — row units and operand paths (gaps-disasm C2, C7, C8, C12)

Branch `gapfix/rows-operands` from `fac7691b`. Four causes of the RTX 5080 loss catalog
([gaps-disasm.md](gaps-disasm.md)), each attacked generically: a fact, a capability or a rule, its OFF arm,
generated-program tests, and the evidence pass (intent → realized SASS → time). Local A/Bs: same-session ABBA
on the shared RTX 5080 (busy machine: coarse), value-gated, rules' pick on both arms (`MAPAL_GPU_PICK=rules`);
kernel times from CUPTI (`ktrace.so`). Rented: one RTX 3090 (6 MiB L2) for the quiet A/B of the L2 rule.
Raw output (never in git): `.claude/raw-evidence/local-rtx-5080/gapfix-rows-operands/` (`abba_local.txt`,
`ktrace/`, `sass/`, `rtx-3090/`, test logs).

## What changed

| cause | fact / property / rule | OFF arm | tests |
|---|---|---|---|
| C2 read-once operand on the default L2 policy | capability `Caps::l2_evict` (`ld.global.cs`, every SM from `sm_20`); rule `rules::stream_ld_pays(g, read_once_bytes)` = `l2_evict && keep < bytes ≤ STREAM_KEEP_SHARE·keep` (`keep` = `Gpu::l2_keep_bytes`, share 4: see the 3090 section), applied per tile operand the grid reads once (one row block for `b`, one column block for `a`; a thread a cell: one row) — register-staged FMA tile, GEMV, vector cells; `gload` emits `ld.global.cs.nc` on whole 4 / 8 / 16-byte loads (SASS `LDG.E.EF`); plan line `evict=a\|b\|ab` | `--gpu-off=stream-ld` | `stream_ld_on_generated_programs` (40 products, 1–70 rows, the keep set from each program's `b`: streaming exactly on read-once operands within the share past the keep — sound and complete —, none under the keep, past the share or OFF, plan field = text, interpreter bytes in 3 emulator modes; cover counters in the test log) |
| C7 row units: a barrier a tree level | `tree_total` (flow units): an f32 tree by butterfly shuffles in each warp — each pair in its lower lane's operand order, so both lanes hold the same bits — and one shared exchange of the warp partials (one barrier), combined as a balanced tree in registers; depth unchanged (the default face's bound is the shared tree's; trees exist only where the order is free) | `--gpu-off=shfl-tree` | the existing generated suites over unit trees, all pass: `tree_gate_on_generated_chains` / generic_fusion (12), row_block (9), long_rows `row_slices` (4), flash_attention (6), fold_algebra (3) |
| C8 closed-form values loaded | **fact** `elem_plan` through captures (`CategoryIr::capture_law`): a body fn's capture `Proj` takes the law of what every site passes, where the law names nothing of the enclosing fn (`Index`, pairs of it) — the fold / map element read already consumes `elem_plan`, so a fold over a captured `iota` computes its index | `--gpu-off=iota-law` | `iota_law_on_generated_loops` (24 norm loops: iota outside / inside the loop, a mapped control; law read host + device where an iota, the control byte-identical to OFF; values in 3 emulator modes, exact and default faces; cover fired host 17, device 17, control 7) |
| C12 8-way STS of the k-major `b` copy | `kvec_swizzle`: panel word `k·bn + n` at `k·bn + (n ⊕ σ(k))`, `σ(k) = ((k ÷ lanes) mod G)·S`, `S` the key rows a warp's chunk spans (≥ a fragment vector), `G = banks ÷ S` from the bank geometry (`GPU_WAVEFRONT_BYTES`: 32 banks × 4 B); fragment reads apply the same σ; the bank model (`panel_excess`) prices it; plan line `swz=GxS` | `--gpu-off=kvec-swizzle` | `kvec_swizzle_spreads_the_scattered_stores` (bank model: conflict-free at `bk ≤ 32`, halved past it, reads unchanged); `kmajor_b_on_generated_decode` + the swizzle's OFF arm (9 of 40 programs swizzled, values on 3 arms) |

Also: `Gpu::with_cost` committed only 30 of the 31 `DeviceCost` names (`--cost=l2_keep_bytes=` alone was dropped)
— the count is now computed. Intents: `stream-ld` (EF loads where `evict=`, none OFF), `kvec-swizzle` (copy shape),
notes for `shfl-tree` / `iota-law` (arm diffs); `evidence.py` counts `LDG.E.EF`. Replay: 14 `rung_replay.tsv`
lines (`read_once=` cond; the 5080's and the rented 3090's), totals in the test.

## Evidence: intent → realized → time (RTX 5080, local)

SASS from the bundle PTX through `ptxas -arch=sm_120` + `cuobjdump`; columns LDG / LDG.E.EF / BAR / SHFL / STS /
LDS of the named kernel, ON → OFF.

| workload | rung | kernel | SASS ON (OFF) | program OFF → ON ms (ratio) | kernel (CUPTI) | leader (leg), matrix ratio before |
|---|---|---|---|---|---|---|
| pj_m1_n6144_k4096 | stream-ld | GEMV `4eb3` | LDG 64, EF 32 (0): the `b` vectors | 0.0843 → 0.0732 (0.869) | | cuBLAS gemvx (`LDG.E.EF`), 1.621 → ~1.41 |
| pj_m16_n6144_k4096 | stream-ld | tile `1e61` | LDG 18, EF 16 (0) | 0.0995 → 0.0787 (0.791) | | cuBLASLt 128x32 sK3, 1.096 → ~0.87 |
| pj_m1_n4096_k14336 | stream-ld | GEMV | `b` EF | 0.2469 → 0.2293 (0.929) | | cuBLAS gemvx sK4, 1.067 → ~0.99 |
| pj_m128_n6144_k4096 | stream-ld | tile | `b` EF | 0.2073 → 0.1996 (0.963) | | CUTLASS simt sK7, 1.127 → ~1.09 |
| pj_m1_n4096_k4096 | stream-ld | — | 67 MB under the 64.6 MiB keep: no EF (rule off; gaps arm null) | not built (= shipped) | | |
| attn_1024_64 | shfl-tree | softmax unit `af15` | BAR 3 (19), SHFL 10 (0), STS 10 (26), LDS 13 (43) | 0.0391 → 0.0367 (0.936) | | Triton softmax 5 BAR / 26 SHFL, 1.284 → ~1.20 |
| attn_2048_64 | shfl-tree | softmax unit | as above | 0.0866 → 0.0831 (0.959) | | 1.154 → ~1.11 |
| lsm_32_8192 | shfl-tree | unit `57d9` | BAR 6 (38), SHFL 20 (0), STS 4 (36), LDS 8 (68) | 0.0204 → 0.0211 (host-bound, noise) | 3.04 → 2.30 µs; all kernels 10.0 → 8.6 µs | torch-compile, 1.882 |
| attn_2048_128 / softmax_rows_4096 / rowsum_4096 | shfl-tree | | trees cut the same way | 0.991 / 1.003 / 0.997 (null: GEMM-bound programs) | | |
| llm_dec_b32_l2048_bf16kv | iota-law | norm cell `6631` | LDG 8 (16): no `rD[i]` load | 7.97 → 7.80 (0.979) | norm 17.6 → 6.8 µs a step | Triton norm 1–1.5 µs, 1.088 → ~1.065 |
| llm_dec_b32_l8192_bf16kv | iota-law | norm | as above | 26.54 → 26.36 (0.993) | | flashinfer, 1.092 |
| llm_dec_b32_l8192_bf16kv | kvec-swizzle | scores `3bede` | same mix: LDG 66, STS 132, LDS 64; addresses swizzled (bank model 8-way → 1) | 28.53 → 28.60 (1.002) | 776 → 777 µs (null) | flashinfer HMMA, 1.092 |
| llm_dec_b32_l2048_bf16kv | kvec-swizzle | scores | as above | 8.85 → 8.38 (0.946, wide spread) | | |

Leader column: the leading leg from gaps-disasm's per-row table; "→ ~x" is the matrix ratio scaled by this
step's ON/OFF (an estimate, different sessions).

**Explained.** C2: the only SASS change is the `b` operand's loads becoming `LDG.E.EF` (cuBLAS gemvx's own
policy); the operand exceeds the L2's keep so it comes from DRAM either way, and evict-first stops it from
displacing the lines the L2 keeps (`a`, the output, the split's partial sums) — the gain grows with the share of
non-`b` traffic (m16 0.79 > m128 0.96). C7: the 256-lane tree's 8 barrier levels become 5 register shuffle levels
+ one exchange; the shared stores / loads of the tree drop by the same count. C8: the norm's `(0.0, rD)` fold no
longer loads `rD[i]` before `x[b·4096 + i]` (a dependent index-then-data pair per element): 16 → 8 LDG in the
cell body, 17.6 → 6.8 µs. C12: the swizzle removes the scattered stores' 8-way conflict (bank model) without
moving the kernel's time — the stores are off its critical path; the scores kernel is bound by the per-element
narrow copy (64 `LDG.E.64` + widening a trip) with one-warp CTAs (5 warps an SM, no copy / compute overlap), the
mechanism gaps-disasm could not split from the conflict: this arm splits it on the 5080 (conflict: 0 %). On
the rented 3090 the same arm wins 0.960 on the whole decode program — its shared-memory store rate a scheduler
is the scores kernel's limit there — so the rung ships ON: the rule (`bn` whole bank rows) holds on both parts,
null on one, a win on the other.

## RTX 3090 (rented, 6 MiB L2)

One RTX 3090 (vast.ai, $0.166/h, instance destroyed after the run; none of this step's remain), the bundle's
probe section: every arm value-gated against the oracle, then ABBA, 5 repetitions (`box_run.sh`, `TORCH=0
TIMING=0 E2E=0`). Raw: `rtx-3090/results-rtx-3090.txt`.

| workload | rung | OFF → ON ms | ratio | reading |
|---|---|---|---|---|
| pj_m1_n6144_k4096 | stream-ld | 0.1419 → 0.1412 | 0.995 | null: the GEMV 0.129 ms on both arms |
| pj_m1_n2048_k7168 | stream-ld | 0.0978 → 0.0972 | 0.995 | null |
| pj_m1_n4096_k4096 | stream-ld | 0.1032 → 0.1057 | 1.025 | null (spreads overlap) |
| pj_m1_n4096_k14336 | stream-ld | 0.2899 → 0.3036 | 1.047 | the GEMV 0.284 → 0.289 ms: the hint's cost, nothing kept |
| llm_dec_b32_l2048_bf16kv | stream-ld | 10.190 → 10.164 | 0.997 | null |
| lsm_32_8192 | shfl-tree | 0.0481 → 0.0357 | 0.743 | win (spreads apart) |
| llm_dec_b32_l2048_bf16kv | iota-law | 10.325 → 10.164 | 0.984 | win (spreads apart) |
| llm_dec_b32_l2048_bf16kv | kvec-swizzle | 10.593 → 10.164 | 0.960 | win (10.59–10.60 against 10.15–10.21): on the 3090 the scattered stores are on the scores kernel's path |

**C2 was mixed — the missing fact is the share of the operand the L2 keeps.** The 5080 wins are not "no reuse":
`pj_m1_n6144_k4096` streams its 100 MiB `b` at 1.19 TB/s OFF and 1.37 TB/s ON against a 0.96 TB/s DRAM, and
`pj_m1_n4096_k14336` (235 MB) 0.95 → 1.03 TB/s — L2 hits across calls, more of them with evict-first (its lines
do not displace what the L2 holds, so a share `keep ÷ bytes` of the operand survives to the next call). On the
3090 that share is 0.03–0.11 (6.5 MB keep): nothing survives, and the hint only costs (the 14336 GEMV +1.8 %).
The rule (`rules::stream_ld_pays`) now streams an operand past the keep by at most `STREAM_KEEP_SHARE` = 4 times
(the L2 keeps at least a quarter of it): the 5080 rows (shares 0.27–0.65) stream, the 3090 rows (≤ 0.11) do not.
The 5080's measured winners are unchanged by the bound; the 3090's become the OFF arm's text. A mixed-radix
operand (a decode step's keys: its bytes span batch digits `k·c` does not count) claims no share; the 3090's
decode keeps one streamed operand, P·V's `a` (the 8.4 MB probabilities, read once and dead after).

## Open (attempted, not closed)

- **C7 IEEE quotient a row element** (`e / s`, `div.rn.f32` + slow-path `CALL`): `x · (1/s)` is two roundings
  where the default face states one for an element-wise quotient (ADR-0040 "full precision"); admitting it is a
  precision-plan decision (a quotient by a fold-constant divisor with its own bound), not a backend rule — left for
  an ADR. Not measured: the SASS shows 1 `CALL` (the slow path, not taken on normal operands).
- **C7 16-byte row loads / row buffer in registers / rows per CTA**: the unit walks stay coalesced 4-byte loads;
  not attempted in this step (the tree's barriers were the measured mechanism).
- **C8 the norm on 4 of 84 SMs** (32 rows × one warp): 6.8 µs against the leader's 1–1.5. The bound is the bytes in
  flight (32 warps × ~1 KiB), so a spread over more CTAs at one warp a row moves nothing; it needs several warps
  a cell — the same shuffle + exchange tree as `shfl-tree`, inside the cell-lanes body (`bulk.rs` fold lanes, a
  block-wide lane count from `warp_lanes`). Not built in this step.
- **C8 smlp's closed-form bias kernel** (`b1[t % 1024]` materialized, then read by the epilogue): the epilogue's
  read could take the bias's law (an `Apply` over `Index`); not attempted.
- **C12 the narrow copy path** (8-byte bf16 vectors widened per element, one-warp CTAs): the swizzle arm shows the
  conflict is not the cost; the fix is the copy's width and overlap (16-byte narrow lane loads / a narrow
  `cp.async` stage, 4–8 warps) — a change of the register-staged copy's geometry, not attempted.
- **C2 on `cp.async` panels**: the async copies take no eviction hint (`cp.async … .L2::cache_hint` with a
  `createpolicy` policy); the rule names only realized loads (`evict=` checked against SASS), so those operands stay
  on the default policy.
- **Suite diff**: `suite_diff.py` / text equality count 28–34 suite programs "moved" for every rung — the site
  tables' candidate ids hash the arms, so a table's alternatives are renamed under any OFF arm; the moved kernels
  proper are the rows above.

## For integration

- `CLAUDE.md` facts table, `elem_plan` row: append "— through captures: a body fn's capture takes the closed law
  (`Index`, pairs) of what every site passes (`capture_law`; gapfix `iota-law`)". Backend properties row: add
  `Caps::l2_evict` (L2 eviction priority on a load) and `rules::stream_ld_pays` (a read-once operand against
  `Gpu::l2_keep_bytes`).
- `benches/results/evidence/local-rtx-5080/README.md`: link this file under the gap-fix sprint; rows C2 / C7 / C8 /
  C12 with the table above.
- `matrix.*`: the 5080 column moves on pj_m1_n6144 (~0.87×), pj_m16_n6144 (~0.79×), pj_m1_n4096_k14336 (~0.93×),
  pj_m128_n6144 (~0.96×), attn_1024_64 (~0.94×), attn_2048_64 (~0.96×), llm_dec_b32_l2048_bf16kv (~0.98×) —
  to be re-timed by the integrated run, not edited by hand.
- `docs/tailored-fixes.md`: nothing (no tailored fix).
- Merge notes: `GpuRungs` gains `iota_law`, `stream_ld`, `shfl_tree`, `kvec_swizzle` (one commit, field-list
  union); `Caps` gains `l2_evict`; `KERNEL_RUNGS` gains `stream-ld`, `kvec-swizzle`; `rung_rules_replay`'s totals
  are this branch's 158/74/16/23 + 4/3/0/0 — sum with the other branches' additions.
