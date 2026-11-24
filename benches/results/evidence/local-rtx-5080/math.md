# math — `sqrt`, `sin`, `cos` and one builtin-realization framework (2026-11-25, rtx-5080 + Threadripper 9970X)

Step goal (ADR-0046): `x -> sqrt`, `x -> sin`, `x -> cos` as pipeline builtins like `exp`; one framework for every
math builtin — a contract, a reference definition from plain operations (portable to a target with no function unit),
target realizations as capability data admitted by the contract and chosen by cost — with `exp` moved onto it
unchanged. Raw output (calibration text, probe outputs) stays local (`scratchpad perf/math/{calib,scripts}`), not in git.

## Contracts and references (ADR-0046 D3 / D4)

| builtin | exact / contract / fixed faces | default face | reference (plain operations) | measured |
| --- | --- | --- | --- | --- |
| `exp` | `E_b` (ADR-0042) | `2⁻²¹·eˣ + 2⁻¹⁴⁹` / `2⁻⁵⁰·eˣ + 2⁻¹⁰⁷⁴` on `[lo, hi]` | unchanged; module text byte-identical (`tests/golden/exp-adr-0042.ll`) | ADR-0042's |
| `sqrt` | correctly rounded `√x` | `2⁻²²·√x` / `2⁻⁵¹·√x` | digit recurrence, 25 / 54 steps, one rounding | **every f32: 0 differ** from IEEE `sqrt`; f64 2·10⁵ samples: 0 |
| `sin` | `S_b` | `2⁻²¹·\|sin x\| + 2⁻¹⁴⁹` / `2⁻⁵⁰·…`, every finite `x` | integer Payne–Hanek (`\|x\| ≥ π/4`), Cephes / fdlibm | **every f32: ≤ 1.586 ulp, 0 outside**; f64 4·10⁵ samples ≤ 2.5 ulp of libm |
| `cos` | `C_b` | same | same | **every f32: ≤ 1.580 ulp, 0 outside** |

The f32 sweeps are `crates/mapal-interp/tests/math_exhaustive.rs` (`--ignored`, 3 + 2.3 min on 8 cores): every one of
the 2³² f32 inputs, the huge arguments up to `f32::MAX` included.

## Device units (`benches/gpu/calib.cu math`, exhaustive over each domain against the f64 function)

| unit | domain | worst relative | worst absolute | specials | cycles / warp op (FFMA 1.06) | admitted |
| --- | --- | --- | --- | --- | --- | --- |
| `sqrt.approx.f32` | every positive f32 | **2^−23.253** | — | ok | 8.13 | `sqrt` default face (≤ 2⁻²²) |
| `sqrt.approx.ftz.f32` | same | 1 (flushes subnormals) | — | — | — | not declared |
| `x · rsqrt.approx.f32` | same | 2^−22.935 | — | — | 8.13 | not declared (no gain over `sqrt.approx`) |
| `sqrt.rn.f32` | same | 0 mismatches vs correctly rounded | — | ok | 19.69 | every face (`Exact`) |
| `sin.approx.f32` | `[−0.7854, 0.7854]` | 1 (not relative near 0) | 2^−21.457 | ok | 8.24 | **refused** (the bound is relative) |
| `cos.approx.f32` | `[−0.7854, 0.7854]` | **2^−21.580** | 2^−22.003 | ok | 8.24 | the `cos r` part: `2^−21.58 + 2⁻²³ = 2^−21.12 ≤ 2⁻²¹` |
| `ex2.approx.ftz.f32` | `[−½, ½]` | 2^−22.709 (2.447 units) | — | — | 8.00 | `exp` (ADR-0042, unchanged) |
| CUDA `sqrtf` / `sinf` / `cosf` | every finite f32 | 0.500 / 1.495 / 1.511 ulp | | | | context |

## The RTX 5080's realizations (the selection over its declared units, `math::choose`)

| builtin | exact faces (device / host) | default face (device / host) |
| --- | --- | --- |
| `exp` f32 | `E_32` / `E_32` | `ex2` form / contracted `E_32` |
| `sqrt` f32 / f64 | `sqrt.rn.f32` / `.f64`; host `llvm.sqrt` | **`sqrt.approx.f32`** / `sqrt.rn.f64`; host `llvm.sqrt` |
| `sin`, `cos` f32 | `S_32`, `C_32` | **Payne–Hanek + polynomial `sin r` + `cos.approx` `cos r`** / contracted references |
| `sin`, `cos` f64 | `S_64`, `C_64` | contracted references |
| `plain-fp` host (no unit) | every reference, `sqrt` the digit recurrence | contracted references |

## On the device (65 536 cells: 4 range classes up to `3·10²⁵·k`, `sin`, `cos`, `sqrt`; `mapal run --target=rtx-5080 --gpu-off=place`)

| face | `sin` worst | `cos` worst | `sqrt` worst | outside the contract | cells ≠ interpreter |
| --- | --- | --- | --- | --- | --- |
| interpreter (`S_32`, `C_32`, IEEE) | 2^−23.08 | 2^−23.09 | 2^−24.00 | 0 | — |
| device, `--exact` | 2^−23.08 | 2^−23.09 | 2^−24.00 | 0 | **0 (byte-identical)** |
| device, default face | 2^−21.69 | 2^−21.53 | 2^−23.29 | **0** | 45 173 (within the bound) |

## Tests

`mapal-ir` math (17: references vs ADR-0042's `E_32` bit for bit, traces = native evaluation, `sqrt_def` correctly
rounded, sweeps, specials, contract table, realization rules for `exp` / `sqrt` / `sin` / `cos`, bad target overrides
refused, costs); `mapal-backend-llvm` `math` (4: realizations per side and face with the PTX's `sqrt.approx.f32`,
`cos.approx.f32`, `mul.hi.u64`, `clz.b64`; 24 generated programs — exact bytes on generic, the RTX 5080 (emulated)
and `plain-fp`, default within the bound; every emulator mode; L1615 / L1009) and the exp golden; CUDA differential
(6 generated math programs raw + rewritten on the RTX 5080, byte for byte); `mapal-lower` `math_builtin` (4);
`mapal-interp` `math_value_contract` and the exhaustive sweeps. Generated-program coverage: programs 24, sqrt 11,
sin 10, cos 17, f64 9, fold 10, huge / inf 24, NaN 24, subnormal 24, device units 24.

## Open

- The exact faces' `sin` / `cos` take the Payne–Hanek reduction for every `|x| ≥ π/4` (≈ 126 operations f32, ≈ 253
  f64, branch-free): a Cody–Waite form for moderate arguments with a branch to the integer reduction is a cheaper
  realization the framework can take once its error is stated (the step language is straight-line; a branch is the
  missing piece).
- `MathForm::compose` states the condition-number composition; no value gate reads it yet (as `exp`'s).
- The native profile reads no `math` calibration line: a part other than the RTX 5080 has the SM's `ex2` and
  `sqrt.rn` only (its approximate units unmeasured → the references), until its calibration is parsed into units.
