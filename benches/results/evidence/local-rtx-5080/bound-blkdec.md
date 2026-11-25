# bound-blkdec — `px_llm_blkdec_b2_l16_bf16` outside the probe gate: the gate's bound, not the realization (2026-11-26, rtx-5080)

The five-GPU round on 9be9886d (`cross-arch/README.md`, "b8429388 / 9be9886d", finding 1) printed element 2 of
`px_llm_blkdec_b2_l16_bf16_d` as 0.13053171 against the exact face's 0.12759829 on the 3090, A100, 4090 and H100:
relative 2.30·10⁻², outside `value_gate.py --bf16`'s 1.62·10⁻². Raw output stays local (scratchpad
`bound-blkdec/`), not in git.

## Reproduced and bisected (RTX 5080, c808622c, `mapal run --target=native-gpu`)

| build | element 2 | relative to exact |
| --- | ---: | ---: |
| exact face (interpreter) | 0.12759829 | — |
| default face, shipped; the five parts' value | **0.13053171** | 2.30·10⁻² |
| `--gpu-off=round-store` | 0.12759823 | 5·10⁻⁷ |
| `--gpu-off=` narrow-window, finite-check, mma-walk, stream, stream-mma, half-tile, narrow-radix, narrow-tile, device-trap, cand-made, place | 0.13053171 | 2.30·10⁻² |
| interpreter over `narrow_stored()`'s graph (`xn`, `q`, `att`, `hn`, `hh` stored bf16) | 0.13053171 | bit for bit |
| interpreter, `xn`, `q`, `hn`, `hh` rounded in the source | 0.12938361 | 1.40·10⁻² (d945c641's 5080 print: 0.1293837) |
| … and `att` (`round-post`, 4a2e4805: `P·V ÷ total` stored rounded) | 0.13053171 | 2.30·10⁻² |
| `xn` alone | 0.12946804 | 1.47·10⁻² |
| weakest-member oracle (every admitted multiplicand rounded) | 0.12636243 | 9.7·10⁻³ |

The device computes exactly the program with five arrays stored bf16 (`round-store`); each one's every element read
is a multiplicand of a `fold: bf16 | f32` product, so each rounding is admitted (ADR-0041 D3 / ADR-0043 D4, one rounding
per multiplicand). `round-post` added the fifth (`att`); d945c641 stored four and printed 1.40·10⁻², inside by luck.
No realization rounds what the facts do not admit, and the cache's `to_bf16` is the program's own (both faces).

## Root cause: the gate's number was not a bound of this program

`value_gate.py --bf16` gave the whole program one site's coefficient, relative: `2·((1 + u_t)²·(1 + γ_{K+1}) − 1)` —
two multiplicand roundings. This program admits a chain of them (per decode step: the norms, QKV, scores, `P·V`, O,
gate/up, down; three steps), writes its cache with `to_bf16` of values that carry the default face's deviation, and
element 2 sits under cancellation (0.128 in a row of magnitude ~1.2). ADR-0041 names the case as its falsifier — "a
chain where every per-site gate passes but the program output exceeds the composed bound" — but no composed bound
existed; the gate's constant stood in for it.

## Fix: the composed bound, `bound(o)` (ADR-0041 D5, amended)

`mapal_interp::run_bound` (`crates/mapal-interp/src/bound.rs`; `mapal run --interp --bound`): the oracle runs the exact
face, records every float operation on a tape with its own default-face deviation read from the IR facts — a member's
known rounding error per `(product, slot)` (`precision`), a regrouped fold's order (`step_algebra.regroup`,
`2·γ_{K+1}(u_a)·M`), an fma pair (`step_algebra.fma`), a rounding of a deviating operand that may reach a tie (ADR-0043
D3, decided per value against its own bound), a math contract (ADR-0042 / ADR-0046) — and composes them to first order
through the program's derivatives (one reverse sweep per printed value). The gate (`value_gate.py --bound`,
`tiles/<stem>_xbound.expect`, written by `bundle.sh` / `integrated_bundle.sh` for programs admitting a member;
`box_run.sh` uses it where present) checks `|got − exact| ≤ bound` per printed value.

| printed value | exact | device (5080 = the four parts) | \|d\| | before: one-site, relative | after: composed |
| --- | ---: | ---: | ---: | ---: | ---: |
| `out[0]` | −0.25194228 | −0.2524522 | 5.1·10⁻⁴ | 4.1·10⁻³ | 0.213 |
| `out[65]` | 0.12759829 | 0.13053171 | 2.9·10⁻³ | **2.1·10⁻³ (outside)** | 0.142 |
| `out[67]` | 1.1519748 | 1.1526203 | 6.5·10⁻⁴ | 1.9·10⁻² | 0.126 |
| `out[127]` | 1.7824088 | 1.7796792 | 2.7·10⁻³ | 2.9·10⁻² | 0.117 |
| `Σ out²` | 178.37428 | 178.42642 | 5.2·10⁻² | 2.9 | 17.3 |

The composed bound is mostly possible grid steps (0.142 of 0.142 at `out[65]`; member roundings 6.6·10⁻³, order
2.0·10⁻³): once a rounding's deviation reaches ~`u_t`, every later bf16 rounding — the cache's `to_bf16`, each stored
multiplicand — may land a grid step away with a sign the a-priori analysis cannot know, and they compound over three
steps. With `: bf16 | f32` on the projections only (the full model's bf16 class form) it is 0.126; without the
permission (`px_llm_blkdec_b2_l16_bf16kv`: the same storage, f32 products) 1.7·10⁻³. Every other member-admitting px program on the 5080 is inside its composed bound
(`px_llm_dec_b2_l16_bf16` worst 0.017 of it, `px_llm_pre_t12_bf16` 0.065, `px_convmc_2_4_6_3_tf32` inside).

## Test (`crates/backends/llvm/tests/composed_bound.rs`, generator `testgen::BlockProg`)

32 generated decode blocks (1–2 rows, width 8–32, cache 4–16, 1–3 steps, optional Newton RMS norm and ReLU FFN, bf16
weights and cache, every fold `: bf16 | f32`): the default face emulated on the rtx-5080 in all four emulator modes and
under `--gpu-off=round-store` (160 runs), and the weakest-member oracle — all within each printed value's composed bound;
every program stores arrays bf16 (`round-store`) and its `P·V ÷ sum` with the epilogue's rounding (`round-post`).
On the one-site coefficient the test fails: device values outside it in 5 of 32 programs. Without `: bf16 | f32` (the
f32 twin, same exact face) the same device values are outside the twin's composed bound in 32 of 32: the roundings are
caught where the program does not admit them. `crates/mapal-interp/tests/bound.rs`: a dot product's bound equals
`|Σ δ_k·w_k| + 2·γ_17(2⁻²³)·M` exactly; an exact value rounded deviates by 0; a narrow rounding short of its tie absorbs
the deviation; a Newton square root's bound is its last step's (≤ 2·u_t·|x|), not sixteen. Crates: `mapal-interp`,
`mapal-cli`, `mapal-rewrite` all pass; `mapal-backend-llvm` 540 passed, 0 failed, 6 ignored (GPU-only).

## Full model (RTX 5080, this branch, `MAPAL_GPU_CALIB=…/rtx-5080.calib mapal build --target=native-gpu`)

No emitter source changed (the interpreter, the CLI and the bench tooling): f32 class 128 / 128 tokens equal HF f32 at
128 and 2048; bf16 class 128 / 128 equal to `integ-bf16` (bf16-integrate pass D) at 128 and 2048.

## Open

- The f32 probe gate (K-chain, relative, no member) is not a composed bound either: on `px_llm_blkdec_b2_l16` the
  composed bound at `out[0]` is 6·10⁻³ relative against the gate's 2.6·10⁻⁴ (attention's derivatives reach 20). The
  realized f32 deviation is ~10⁻⁷, so nothing fails; giving every DEFAULT program its `_xbound` (the tooling takes any
  program) would loosen the f32 gates ~20× on sound grounds — left for a decision, no f32 false failure seen.
- (Closed here) `mapal test` judged a member program against the weakest oracle within `weakest_rel` (one site, the
  TF32 unit even for a bf16 member): it failed a correct build of this program (`--target=native`: 0.12759829 vs
  0.12636243 beyond 2·10⁻³). It now judges against the exact face within the composed bound: PASS.
- First order: a float comparison whose operands lie within their bound of each other may flip its arm (assumed not).
