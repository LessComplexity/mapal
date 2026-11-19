# fold-algebra — FMA and fold regrouping from one dataflow fact (2026-11-19, rtx-5080 + Threadripper 9970X)

Commit `76199924` (code + tests), on `local-rtx-5080` after the site-init fix `0fa2643f`. Design:
`.claude/sprints/fold-algebra/design.md`. Raw logs: `.claude/raw-evidence/local-rtx-5080/fold-algebra/`.

## The fact

`mapal_ir::CategoryIr::step_algebra(f) -> StepAlgebra` (`crates/mapal-ir/src/algo/algebra.rs`, stored in
`FuncFacts::step_algebra`):

| field | what it proves |
| --- | --- |
| `fma[s] = FmaPair { mul, slot, sub }` | `s` (float `Add`/`Sub`) reads the product of float `Mul` `mul`, read by nothing else (compute once), and is the only reader of its operand pair: one rounding is admitted where the face contracts |
| `fused_mul[mul] = s` | the same pairs keyed by the `Mul` |
| `regroup = Regroup { fold, step, op, exact, k, reads }` | `f` is the body of `fold` and `fold_tree_step(fold)` admits partials: `op` (+, ×, max/min), `exact` (integer: bits kept), trip count, per term read its stride along the fold axis (`Some(1)` the folded element or a component of it, `Some(ck)` a whole fold's affine read, `None` unknown) — layout, never admission |
| `StepAlgebra::flags(m, contract, free, pairs, regroup)` | the one realization rule both backends read |

Consumers (no workload names, no per-backend re-derivation):

- `FnEmit::emit_arith` (host functions **and** device cell / unit bodies — the same emitter): `contract` on a
  pair's two morphisms, `reassoc` on a float regroup step where the order is free; where the face contracts with a
  sequential or fixed order (bit-for-bit oracles) a pair is one explicit `llvm.fma` (stashed factors, same block),
  the device tile's `OrderedFma` — not a flag a target without FMA would not fuse (the emulator's baseline x86-64
  showed exactly that: `radix_register_tile_values` failed with flags alone).
- GPU `Arithmetic::scalar`: a tile site's step is a pair by construction (property-tested), and the host fuses every
  pair, so the device fuses on `contract` whatever B's lane stride or radix form — the `clane == 1` admission is gone.
- OFF arms: `--no-fma-pairs` (`GpuArms::pairs`), `--no-fold-regroup` (`GpuArms::regroup`). With both, the emission
  equals the pre-sprint golden (`site_choice::candidate_zero_is_the_pre_sprint_emission`, every row).

## Tests

| test | result |
| --- | --- |
| `fold_algebra.rs::the_fact_on_known_steps` | dot (pair + float regroup, reads `[1, 1]`), sub pair (no regroup), float sum (`[1]`), max (select), int sum (exact) |
| `fold_algebra.rs::flags_and_realized_intents` | flags per face (default / sequential / exact / both OFF arms); x86-64-v3 asm: `vaddps` + vector `vfmadd` ON, no `vaddps` OFF |
| `fold_algebra.rs::generated_programs_keep_values_and_invariants` | 64 generated programs (chain / recur / radix / geom): 130 pairs, 227 regroups, all 62 tile sites are pairs; 60 valued programs bit-exact on the exact face, within the bound on the default face and both OFF arms (2 trap as the interpreter does, 2 past its budget) |
| rewritten from the old policy | `gpu.rs::strided_contract_fuses_from_the_step_algebra`, `epilogue::the_exact_face_device_unit_has_no_contraction`, `retain_delta::strided_site_fuses_the_host_and_device_step_alike` — new rule + the OFF arm's old behaviour |
| workspace (`--release`, 40 jobs) | every mapal-ir / llvm binary ok (105 binaries, 1338 tests); `mapal-backend-cuda` `differential_testgen_closed_sweep` fails as before this change (sm_89 JIT timeouts on sm_120; it depends only on mapal-ir, untouched facts); its 5 extra 15 s timeouts under load pass idle |
| `benches/gpu/test_harness.py` | 27 / 27 (new intent `fma-pairs`, `contract` scope in `evidence.py`) |

## Intended vs realized

| optimization | intent (machine-checked) | realized |
| --- | --- | --- |
| CPU regroup (order-free float fold) | LLVM vectorizes the step into lane partials + tree | 1M sum: 32 × `vaddss` → 36 × `vaddps zmm` (OFF: 32 × `vaddss`); dot fold: `vfmadd…ps` partials |
| CPU pairs (free face) | `contract` pair → one FMA | dot / saxpy-like map: `vfmadd` (asm); sequential face: `call @llvm.fma.f32` per pair, `fneg` for sub pairs |
| GPU strided / radix tile | `fma-pairs`: contracting face ⇒ `OrderedFma`; `face-fma`: PTX `fma.rn`, SASS FFMA, no FMUL | rtx-5080 static evidence: `fma-pairs` 394 ok / 0 mismatch (OFF arm: the 12 attn / decode tile kernels it fuses); `face-fma` 786 ok, 2 mismatch pre-existing (firew body FMUL, also OFF). attn QKᵀ candidates: 512 FMUL + 512 FADD → 512–1024 FFMA, 0 FMUL / FADD |
| no data movement change | — | host / device transfers identical (only arithmetic text differs) |

## End check — time

**CPU** (Threadripper 9970X, interleaved cycles; ms median, 1t legs min). Only the generic fold path changed:
in FIR, conv2d, saxpy, matmul and NN the functions whose IR differs are uncalled map / fold bodies (the tiled sites
replace them; `fn4` / `fn7` have no callers), so their timed code is byte-identical ON vs OFF — their ON/OFF
differences (matmul 1024 0.954 vs 0.860, 4096 37.1 vs 39.8; FIR fma-par 0.320 vs 0.237) are run-to-run variation
by construction, not a result.

| workload | OFF | ON | NumPy 1t | |
| --- | ---: | ---: | ---: | --- |
| sum reduction 1M, 1 thread | 0.459 | **0.0328** | 0.144 | **14×**; Mapal now 4.4× faster than NumPy (was 3.2× slower) |
| sum reduction 1M, 64 threads | 0.468 | 0.078 | — | the fold is one task (never split); slower than 1t: data handed off from the generating cores (cpu-analysis open #4) |

**GPU** (RTX 5080, `LADDER=1 benches/local/gpu_run.sh`, ABA: ON, OFF, ON; values gated first: 0 failures in every
leg). Full table: `fold-algebra/gpu-on-off.md`. Every row whose verdict is not null was checked:

| row | ON/OFF | timed code | cause |
| --- | ---: | --- | --- |
| indep_512 | 0.76 | device identical, host timed slice identical | **tuned pick**: ON's wisdom picked candidate 5, OFF's candidate 7; the OFF binary with ON's wisdom runs 0.0385 ms = ON (ABBA) — per-site-choice tune robustness, not this change |
| attn_1024_64 | 0.91 | QKᵀ fused (FFMA) | same-box ABBA, 11 processes × 4 legs: ON 0.0434 / OFF 0.0431 ms — **null**: the fused hot loop does not move a 43 µs launch / latency-bound program |
| rowsum_4096, pj_m16_n18432, 4096³ cell, conv_2048_5, diamond, twin (faster) and pj_m16_n2048, convmc_tf32, firew, loopmm (slower) | 0.91 – 1.10 | **byte-identical** timed code (device text equal, host diffs only in untimed / uncalled functions) | cross-run variation; these rows measure the real null between separate runs of this harness: ±10 % (the 2-run A/A underestimates it). pj_m16_n2048 ABBA: ON 0.0482 / OFF 0.0475 ms |

So on the GPU the change is **neutral in time** on this suite (its arithmetic is realized; no row moves beyond the
cross-run null), and on the CPU it is a **14× win where a generic order-free fold exists** (the sum) and byte-identical
elsewhere.

**Integrated build vs the first local run** (site-init fix `0fa2643f` + this; Mapal / best competitor, run 2 → ON):
indep_512 3.40 → 1.07, mlp_512 2.93 → 0.93, attn 3.32 → 1.42, convmc_tf32 2.56 → 1.39, convmc 1.51 → 0.66,
smlp 1.65 → 1.12, pj_m16_n2048 1.61 → 0.61, pj_m1_n2048 1.48 → 0.46, rowsum_2048 1.17 → 1.00,
pj_m16_n18432 1.10 → 1.00; the 2048 MLP / chain rows 1.02 – 1.11. These come from the site-init fix (the OFF
column already has them); fold-algebra adds nothing measurable on top.

## Open

1. GPU flow-unit row trees as warp shuffles (softmax: `smem` + 19 BAR, no SHFL) — the `Regroup` fact admits it;
   not implemented.
2. CPU: an order-free whole fold split across threads (per-thread partials + tree) — reads `Regroup` too; the 64-thread
   sum (0.078 ms) is slower than one lane (0.033) until then, plus the data-handoff cost.
3. CPU max / min regroup: LLVM vectorizes `select` reductions only with `nnan`, which the float face never grants —
   needs explicit lane partials from `Regroup`.
4. `Regroup.reads` is read by no consumer yet (LLVM picks the vector width); the GPU lanes rule could read it.
5. per-site-choice tuning picked a 25 % slower candidate for indep_512 in one bundle (trial noise on a ~40 µs program).
