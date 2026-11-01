# GPU scheduling follows the realized arithmetic

The backend already uses IR shapes, affine addressing, independent output cells and ordered reductions together with GPU registers, threads, shared memory and architecture capabilities. The audit found a specific broken connection: the policy key could describe requested arithmetic rather than the arithmetic actually emitted. That connection is now corrected in `crates/backends/llvm/src/func/gpu.rs`.

| Case | Before | After |
| --- | --- | --- |
| Strided-lane site with Contract requested | Emitted separate multiply/add but could select Contract policy | Selects Exact policy and emits the same separate arithmetic |
| TF32 requested, tensor realization rejected | Scalar fallback could retain TF32 policy | Resolves the actual scalar arithmetic, then selects its scalar policy |
| Supported scalar or tensor realization | Existing source/layout/hardware selection | Preserved in the eight before/after controls |

`Launch::arithmetic` records one of four realizations: separate multiply/add, ordered FMA, TF32 MMA or TF32 WGMMA. `Arithmetic::scalar` derives scalar eligibility from existing layout facts. `launch` attempts eligible tensor routes, then reselects policy from the actual scalar mode if they decline. `kernel_text` and `Kernel::step` consume that same fact. Shapes or benchmark names introduce no new special cases.

The correction preserves the preliminary geometry used for Fill, the measured policy table, forced-arm precedence, WGMMA-before-MMA selection and geometry formulas. Module-wide dependent-launch flags and kernel waits still share their existing derivation. Source reduction order, seeds and operand semantics remain intact. The conservative register estimate during warp-fill adjustment is intentional policy, established by source history and its test; it was not changed.

The [architecture](../../../components/backend-nvptx/ARCHITECTURE.md#9-realized-arithmetic-drives-per-site-policy) and [implementation map](../../../components/backend-nvptx/IMPLEMENTATION.md) name the facts and their consumers. This is an internal consistency correction, not a new arithmetic permission or a general schedule search.

## Verification

| Evidence | Result |
| --- | --- |
| Two regressions using the original API | Both compile and fail on the old policy after confirming emitted arithmetic; both pass after correction |
| Backend library tests | 65 pass, including six new focused tests and all 13 GPU module tests |
| Backend integration tests | 176 pass across 14 test binaries; one optional performance baseline remains ignored |
| Eight successful scalar/default/forced tensor controls | Raw host LLVM, device LLVM and PTX are byte-identical before/after; optimized LLVM is identical after normalizing only source-path metadata |
| Five affected cases, including a mixed scalar/tensor module | Device policy output changes; raw device-LLVM arithmetic counts, PTX FMA counts and dependent-launch wait counts remain unchanged; mixed module retains two waits and matching host flags |
| Independent review | Source and 104 emitted files rechecked; no blocking finding |

The first sandboxed library run had three native hardware-detection failures. All 65 library tests pass with host-query access. The integration run had clang available and executed the compile/run targets; `MAPAL_REQUIRE_CLANG` was unset, as recorded rather than represented as a forced-guard run. LLVM 22.1.8 lowers all 26 before/after device modules through `opt -O3` and `llc -O3 -nvptx-fma-level=0`. Formatting and diff checks pass.

The complete [verification receipt](../../../../benches/results/gpu/scheduling/fact-driven-selection/verification-receipt.json), [independent review](../../../../benches/results/gpu/scheduling/fact-driven-selection/independent-review/FINAL-REVIEW.md), logs, source audits and [emission archive](../../../../benches/results/gpu/scheduling/fact-driven-selection/emission-evidence.tar.gz) are retained. Baseline source is `d80d8f5`; corrected `gpu.rs` SHA256 is `e932fcb811d5a245d4dfcc03134928af85af19443210014089c5e66f97092205`.

No new NVIDIA execution, ptxas/SASS resource validation or performance gain is established for this compiler correction. Static operation counts supplement source review and existing tests; they do not alone prove runtime numerical equality. The separate frozen library-capture run does not validate this new compiler source.

The changed policy does affect optimized PTX. In the strided Contract example, the loop
body has eight separate multiply/add steps instead of four, and its virtual 32-bit
register declaration changes from22 to35. This is static loop unrolling and virtual
register text, not a measurement of dynamic work or physical registers. The layout
fallback adds `.minnctapersm 16`; that is a compiler occupancy hint, not achieved
occupancy. These [exact PTX observations](../../../../benches/results/gpu/scheduling/fact-driven-selection/ptx-policy-effects.json)
explain why unchanged raw LLVM arithmetic counts do not imply unchanged final instruction
counts or performance. Their benefit remains unmeasured.

## What remains to optimize

Output tiling and staging can preserve an ordered reduction while exploiting independent cells. Parallel partial sums or tree reductions change floating-point rounding and require an explicit permission or equivalence proof; existing Contract grants FMA contraction with retained order. Hardware facts constrain legal choices, while measured confirmation remains necessary to select profitable ones. The [experiment sequence](optimization-experiments.md) separates geometry, dispatch, retention, allocation and fusion so each proposed improvement can be tested on its own terms.
