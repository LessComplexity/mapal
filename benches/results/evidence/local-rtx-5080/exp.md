# exp — `x -> exp`: the oracle's algorithm, a bounded hardware form, the rescale law (2026-11-21, rtx-5080 + Threadripper 9970X)

Step goal: `exp` as a standard pipeline builtin (like `widen_f32`; never `exp(x)`), specified first (ADR-0042), then
front to back; the combine law that lets a softmax's sum fold beside its max; the softmax / attention / model-scale
programs written with it. Raw output (suite runs, ABBA, CUPTI traces, SASS, identity sweep, test logs) is in
`.claude/raw-evidence/local-rtx-5080/exp/` (local, not in git).

## What changed

| | fact / rule | OFF arm |
| --- | --- | --- |
| 1 | **Language** (ADR-0042 D1–D3): `exp` resolved by name in pipeline position (`mapal-lower` `is_pure_builtin`; typing keeps the width, a literal's variable flows through; **L1615** for a non-float; reserved, L1009); `Operation::Exp` (`FnBuilder::exp`, validate: width-keeping float edge); the interpreter and the rewriter's constant folder compute **`E_b`** (`mapal-ir/src/exp.rs`: clamp, round-to-integer by the shifter, Cody–Waite, Taylor of degree 7 / 13, two exact power-of-two scales, the special values). `elem_plan` / `map_law`: a body with `exp` is classifiable (pure, trap-free). | — (semantics) |
| 2 | **`precision_plan.exp`** (D6): each `Exp`'s `ExpForm` — the oracle's constants and the default face's bound `rel·eˣ + abs` (`2⁻²¹`, `2⁻¹⁴⁹` f32; `2⁻⁵⁰`, `2⁻¹⁰⁷⁴` f64). | — (fact) |
| 3 | **Realizations** (D4/D5, `crates/backends/llvm/src/exp.rs`): every body calls `@mapal.exp.f32/f64`; the module defines it per side and face — the oracle's steps (exact / contract / fixed faces: no fast-math flag), the host's contracted polynomial (default face: FMA, straight-line selects, vectorizes), the device's hardware form (default face, f32: two-FMA reduction, `ex2.approx.ftz.f32`, the oracle's scaling and specials) where the part's measured `ex2` error meets the bound. **`Caps::ex2`** (a capability property, units of 2⁻²⁴): measured exhaustively on the RTX 5080, 2^−22.709 → 3 (`benches/local/exp_calib.cu`); the rule is `(u + 1)·2⁻²⁴ ≤ 2⁻²¹`, never a name. CUDA backend: `mapal_exp_f32/f64`, the oracle with `__fmul_rn`/`__fadd_rn` on the device. | `--gpu-off=exp-hw` / `--no-exp-hw`: the oracle's steps on every face |
| 4 | **`fold_combine`'s rescale law** (D7, `algo/algebra.rs` `rescale_law`): `⊕ = +` and the term `exp(κ·(x − v))·y` through products / quotients free of `v` (`κ` positive constants, `x` free of `v`) → `DepLaw::Rescale` with its `RescaleArg` (the `Exp`, the side of `x`, the steps, `up`, the array of `x`); composes through a product path at depth (attention's `P·V` keeps `mx` a rescale through `p = ex ÷ den`). **`row-slices` consumer**: where the value is an earlier fold of the unit over the same `x`, row and slices whose step keeps the matching extreme, the sum's partial over a slice takes that slice's max as its shift (same item, same phase) and the combine multiplies partial `r` by `exp(a(x := m_r, v := m))`: a softmax's three phases become two. Exact / fixed faces: no slicing, the phase. | `--no-exp-rescale` / `--gpu-off=exp-rescale` (a fact arm, `FactArms::exp_rescale`) |
| 5 | **Programs**: `gen_tiles.py` `expo` / `llm_exp` / decode's `ex` are `-> exp`, SiLU `v / (1 + exp(−v))`; 50 `benches/algorithms` programs regenerated and formatted (`mapal fmt`); torch legs (`torch_bench.py expo`) use `torch.exp`, `chain_bench.cu` `expf`. No program kept the polynomial (the long-rows test's own polynomial softmax still covers the phase path). Editors list `exp`. | — |
| 6 | CLI: a GPU `mapal run` writes `mapal_choice.h` beside the runtime source it compiles (it failed to compile before). | — |

## Accuracy (exhaustive where it says so; raw `exp_calib.txt`, `scratch proto`)

| form | measured | bound (D4) |
| --- | --- | --- |
| `E_32`, separate steps (oracle) — every f32 in `[lo, hi]` (2 239 849 421) | ≤ 1.221 ulp, relative 2^−23.21 (normal results) | 2^−21 |
| `E_32` contracted (host default) — exhaustive | ≤ 0.939 ulp | 2^−21 |
| hardware form on sm_120 — exhaustive, vs the device's f64 `exp` | ≤ 2.068 ulp | 2^−21 (≈ 4 ulp at a binade's top) |
| `ex2.approx.ftz.f32` on `[−½, ½]` — exhaustive | relative 2^−22.709 = 2.447 units of 2⁻²⁴ → `Caps::ex2 = 3` | admits `u ≤ 7` |
| CUDA `expf` / `__expf` (reference) — exhaustive | 1.931 / 63.78 ulp | `__expf` outside any fixed bound |
| `E_64` (oracle) vs glibc `exp`, 2·10⁸ samples | ≤ 1.0 ulp | 2^−50 |

## Byte identity

- The 201 `benches/algorithms` programs without `exp` × {default, exact} × {rtx-5080, rtx-4090, h100-pcie,
  x86-64-v3}: **1608 builds, 2760 host / device modules equal `5b6c0a29`** (line multisets, kernel ids and metadata
  numbers blanked: the two new rung fields change every kernel-id hash; `ident/`).
- The exact face of the `exp` programs is the interpreter's bytes: `mapal test --exact` PASS on the 13 small ones
  (`px_attn_64_16`, `px_softmax_rows_128`, `px_lsm_2_512`, the 7 `px_llm_*`, `causal_16_8`, `gen_16_8_8`,
  `prefill_16_8_8`) on the native CPU **and the RTX 5080** (real device); 48 generated `ExpProg` programs on the host
  build and the emulated device; the CUDA backend (nvcc, the real device) on 6 generated programs raw and rewritten.

## Intended vs realized (rtx-5080; `ptxas -v`, `cuobjdump -sass`, CUPTI `ktrace.cpp`)

| | intended (fact + rule) | realized |
| --- | --- | --- |
| device f32, default face | `MUFU.EX2` after a two-FFMA reduction, no libcall; f64 the contracted polynomial | one `exp`: 22 SASS (3 FFMA, 3 FMUL, 1 MUFU.EX2, F2I, FMNMX, 3 FSEL, …) vs the oracle's 38 (12 FMUL, 11 FADD); every exp kernel of the suite's modules has `MUFU.EX2` ON and none OFF (`sass/`) |
| exact face | the oracle's steps on both sides, no `contract`, no `@llvm.fma` | checked by `exp.rs` `realization_per_side_and_face`; device = interpreter bytes (above) |
| host default | contracted polynomial, vectorizable | a plain `x -> map { v -> v -> exp }` loop: vectorized at width 16 (`vfmadd213ps %zmm` ×7); a map reading captures stays scalar ("cannot identify array bounds", pre-existing) |
| pt_lsm_1_131072 | `mx`, `den` partials in phase 0 (rescale: `den` against its slice's max), `y` in phase 1: 2 phases, 1 grid barrier | `slices=64 phases=2 laws=rescale ns_sliced=43831` (OFF: `phases=3 laws=`, 44957); 36 regs (OFF 40), 0 spill; SASS 896 instr, 10 MUFU.EX2, 43 BAR (OFF 58), 1 grid-barrier atomic (OFF 2), 21 LDG (OFF 15: the combine reads `mx`'s partials); CUPTI flow kernel **7.94 → 5.89 us** (polynomial program at HEAD: 7.42) |
| pt_lsm_4_32768 | one block a row (outside a loop the rule adds the first cooperative launch to slicing; it prices the walk at the polynomial's `row_iter_ns`) | unsliced both arms; flow kernel 54.27 (OFF) → **49.92 us** (ON); the polynomial program's 34.30 — see open 2 |
| softmax_rows_2048, attn_* | `ex` cells / units take `MUFU.EX2` | the softmax flow unit 374.8 → 374.5 us (polynomial 372.0): memory-bound, the form does not show; attention kernel totals equal within 0.6 % |

## Time (ms; suite `LADDER=1 gpu_run.sh` ON / OFF / ON; OFF = `--no-exp-hw --no-exp-rescale`; ABBA 8 rounds = 16 single-iteration processes an arm, ON/OFF)

| workload | OFF | ON run 1 / run 2 | best competitor | ABBA ON/OFF median (null A / B) | explanation |
| --- | ---: | ---: | --- | --- | --- |
| lsm_1_131072 | 0.0552 | 0.0501 / 0.0462 | 0.0236 torch-compile | 1.031 (68 % / 58 %): null | flow kernel 7.94 → 5.89 us (one barrier fewer, MUFU); the process's first cooperative launch (39–53 us host setup, long-rows) dominates a one-iteration region |
| lsm_4_32768 | 0.0703 | 0.0660 / 0.0659 | 0.0184 torch-sdpa | **0.950** (9.5 % / 11.9 %) | one block a row (4 blocks): latency-bound walk, the hardware form's shorter chain 54.3 → 49.9 us; vs the polynomial program 1.46× (open 2) |
| lsm_32_8192 | 0.0283 | 0.0266 / 0.0267 | 0.0103 torch-sdpa-compile | 0.980 (33 % / 25 %): null | one block a row; 32 blocks |
| softmax_rows_2048 | 0.4788 | 0.4770 / 0.4773 | 0.469 torch-compile | — | memory-bound unit (374 us, unchanged by the form) |
| softmax_rows_4096 | 3.797 | 3.794 / 3.807 | 3.551 torch-compile | — | as above |
| attn_1024_64 | 0.0421 | 0.0405 / 0.0416 | 0.0297 torch-compile; SDPA 0.0451 / 0.0307 | 1.004 (8.8 % / 7.2 %): null | tile kernels and launches dominate; no streamed unit (open 1) |
| attn_2048_64 | 0.0820 | 0.0808 / 0.0811 | 0.0748 torch-compile | 0.995 (23 % / 3.5 %): null | as above |
| attn_2048_128 | 0.1381 | 0.1363 / 0.1369 | 0.1044 torch-sdpa-compile | — | as above |
| causal_1024_256 | 15.19 | 15.12 / 15.17 | 51.72 torch-compile | — | ±0.5 % |
| decode_1024_256 | 8.907 | 8.835 / 8.764 | 15.78 hand CUDA (cache) | 1.002 (2.5 % / 2.9 %): null | exp is a small share of a decode step |
| prefill_1024_512_256 | 7.870 | 7.826 / 7.986 | 25.66 torch-compile | — | null |
| gen_1024_512_256 | 7.703 | 7.665 / 7.711 | 25.64 torch-compile | — | null |
| llm_dec_b1_l8192_pv (16 steps) | 5.334 | 5.267 / 5.273 | 2.997 torch-bf16-compile (f32 compile: model-scale.md) | — | −1.2 %, consistent across both ON runs (null 0.2 %); unexplained at kernel level (not traced) |
| llm_pre_t2048_pv | 8.837 | 8.843 / 8.824 | 1.92 torch-bf16-compile | — | null |

Value gates: the probe gate 0 failures in all three runs; every default-face arm `value-bound … within` the exact
face; every torch leg of these programs `within=yes` against Mapal (attention including SDPA, now gated: torch's
`exp` and Mapal's agree to 3·10⁻⁸ – 8·10⁻⁷; the softmax / lsm rows ≤ 1.3·10⁻⁷; the model-scale f32 legs 7–16·10⁻⁸,
bf16 4·10⁻³). Evidence checker: 7724 checks, ON 326 / OFF 328 mismatches (the pre-existing tile-kernel set, untriaged).

## Tests

- `mapal-ir` `exp.rs` (3): `E_32` over every 61st f32 of `[lo, hi]` (3.7·10⁷) ≤ 1.25 ulp and within the bound, the
  special values and range edges; `E_64` 4·10⁶ samples ≤ 1.5 ulp of libm; the hardware form's admission from the
  measured error (3 and 7 admitted, 8 not, f64 never).
- `mapal-lower` `exp_builtin.rs` (4): width kept (f32 / f64, bodies, bindings, a literal's default and pin), L1615,
  L1009, the formatter keeps the stage. `mapal-interp` `widen.rs` `exp_value_contract`: `E_b` at both widths, specials.
- `mapal-rewrite` `fold_combine.rs` (5): attention's laws (`den`: `mx` Rescale, `x − mx` scaled by 1/8, against `s`;
  `o`: `mx` Rescale through `p`, `den` Factor ÷; the OFF fact arm gives the phases); `exp_rescale_laws` — `exp(x − m)`,
  `exp((x − m)·c)`, `w·exp(x − m)` rescale, `exp(m − x)` over a min rescales down, `exp(x·m)`, `exp((x − m)·x)`,
  `exp(x − m) + 1` and a max over `exp(x − m)` are phases; the generated sweep now requires every law > 0 (rescale
  included).
- `mapal-backend-llvm` `exp.rs` (3): the realization per side, face and arm (+ `ex2.approx.ftz.f32` in llc's PTX);
  48 generated `testgen::exp_prog_strategy` programs (overflow, subnormal / zero results, `±inf` and `NaN`
  arguments, `f64`, a fold body, a softmax) — exact face byte-identical on the host build and the emulated device,
  the default face within 10⁻⁴ (coverage counters inf / NaN / zero / f64 / device-ex2 each > 0); the rescale in
  row-slices — 3 shapes × s = 2, 4, 8, 2 phases, 1 barrier, `laws=rescale`, values within the bound under free
  threads, `MAPAL_EMU_SERIAL=1`, `2` and `MAPAL_EMU_REVERSE=1` with the cooperative grid capped at 1–3 blocks; the
  OFF arm 3 phases; the exact face unsliced and byte-identical. `long_rows.rs`: the generator's new `exp` forms give
  `law:rescale` > 0 among 160 programs (130 sliced units, 318 runs).
- `mapal-backend-cuda` `differential_exp_programs_raw_and_rewritten`: 6 generated programs × raw / rewritten, nvcc,
  the real device, byte for byte. `editors/test.sh`: both builtin lists identical (13 names).
- Suites (workspace, release, `MAPAL_CUDA=/opt/cuda`): **120 binaries, 1428 passed, 1 failed** — `site_choice`'s c0
  emission hash for `pt_attn_1024_64`, whose program changed (12 golden rows regenerated, the rest equal) — then 1429
  passed. The pre-existing CUDA `differential_testgen_closed_sweep` timeout did not fire this run.

## Open

- Fixed after review (`review-fixes.md`): a slice whose scores are all `−∞` (a mask) made the rescale law's terms
  `exp(−∞ − (−∞))` = NaN — the slice now takes the stand-in shift 0 (generated long rows: a third masked); `exp-hw`,
  `exp-rescale` and `row-slices` have `intent.py` entries (44 checks ok).

1. **Flash-style attention.** The fact gives attention exactly flash's laws (`P·V`: `mx` Rescale, `den` Factor ÷),
   but no consumer streams a row-block unit over key blocks (long-rows open 1): at 1024–2048 keys the scores are
   L2-resident and the row-block unit stays refused by its model; at model scale the attention tile sites are not one
   unit. Needs a streamed tile phase in `func/flow.rs` (per key block: q·Kᵀ, block max, rescale of the running
   `(den, O)` by `exp(m_old − m_new)`).
2. **Missing fact: the walk's per-element cost.** `row_iter_ns` (87 ns a lane-iteration) was measured on the
   polynomial body; the rule prices every unit's walk at it. With the real `exp` (22 SASS on a dependent chain against
   the polynomial's ~12) the one-block-a-row walk of `lsm_4_32768` costs 49.9 us where the rule assumes 33.9, so it
   keeps one block a row (4 of 84 SMs) where slicing measured 6.4 us (long-rows' first rule). The walk cost should
   read the unit's per-element dependent work (the cells' op chain, an IR quantity) times a measured issue / latency
   property — a mixed result until then (lsm_1_131072 0.79× the polynomial program's kernel, lsm_4_32768 1.46×).
3. `Caps::ex2` is measured on the RTX 5080 only; the other profiles carry the same constant (the same SFU
   instruction), unmeasured there. `exp_calib.cu` measures a part.
4. The rescale consumer needs the value to be the max (min) of the same `x` in the same unit; any other rescale is a
   phase in `row-slices` (the fact still says Rescale).
5. Host: a map reading captured arrays does not vectorize ("cannot identify array bounds", pre-existing); `exp` itself
   is straight-line and does where the loop does.
6. `llm_dec_b1_l8192_pv` −1.2 % ON vs OFF in both suite runs: not traced (no ABBA).
