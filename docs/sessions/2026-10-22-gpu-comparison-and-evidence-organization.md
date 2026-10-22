# GPU comparison and evidence organization

Checkpoint recorded 2026-10-22 after the seventh task rental closed. Performance analysis continues locally; this is a recoverable checkpoint, not a claim that the cuBLAS gap is closed.

## Continuation brief

The allocation correction is integrated and independently validated. Ten composed timing rows and four actual Mapal Contract images are admitted. cuBLAS remains ahead in both composed shapes. Competitive CUTLASS preparation is reviewed but has no GPU result: two subsequent providers failed before SSH/upload/computation. The operator is diagnosing those failures before another rental. Two coding/analysis agents continue independently.

Start with [current results](../performance/gpu/current-results.md) and [live continuity](../next-session.md). The latter owns changing execution state. Inspect:

```sh
cd /Volumes/LessComplex/Personal/mapal
git status --short
openspec list --json
cat docs/next-session.md
cat /private/tmp/mapal-s61-root/live-state.json
cat /private/tmp/mapal-s61-gpu/attempt-07/results/cleanup.json
```

## Completed work and decisions

| Item | Verified outcome |
| --- | --- |
| Unused raw-output storage | `resident.rs:FnEmit::chain_prologue` consumes existing `Epi.store_out`; skips only an unmaterialized grouped raw buffer. Arena guard and live-member allocation/dependency positions remain. No new IR fact. |
| GPU allocation gate | 640 ordinary fresh processes and28 separate traced processes pass; requested counts/bytes decrease exactly as predicted and controls remain stable. Counter timings cannot establish speedup. |
| Local integration | Required-clang workspace1187pass/3native-detectionfail/1ignored; complete host-access backend rerun59pass including those3.1190distinct passing tests, not a full unrestricted workspace rerun. |
| Small-M panel alternative |48records/2160events pass. FMA improves1.253×/1.159× over original; Exact regresses1.227×/1.013×. cuBLAS beats all four cells. Do not promote the blanket alternative. |
| Residual MLP | All10 rows pass independent admission at M128/d256/h512 and M128/d1024/h2048. Mapal0.082370/0.138100ms versus cuBLAS0.056230/0.102531ms; Mapal beats the restricted eager/graph-disabled Inductor arms. Input reset/cache state differs; no causal retention claim. |
| Executed images | Four old/cooperative Contract captures join exact PTX/context/module/function/launch to callback ELF and exact-entry SASS;924inputs/69outputs rehash. No B/Exact/TF32/library image. Driver-JIT ptxas13.3 versus standalone decoder12.8 limits detailed ISA interpretation. |
| CUTLASS control audit | Original128×128×8 SIMT tile creates only8/16CTAs at the measured M16 shapes. This is a configuration mismatch, not a causal proof or CUTLASS limit. |
| Stronger comparison | Twelve explicit SIMT tiles, both arithmetic-face gates, frozen input/oracle/budget inventory, nine tuning samples then thirty held-out samples. Default control retained.983sealed files; no GPU compilation or measurement yet. |
| Numerical proof repair | Pre-GPU review proved one final `nextafter` did not enclose all rounded intermediate calculations. Exact integer/Fraction ceilings independently verify6656cells. No fitted tolerance or timing feedback. |
| Profiler interface repair | NCUv2 appended `--values-only`, forbidden by the actual harness with profiling. Narrow v3 removes it; extracted real C++ guards pass9 valid/reject6 invalid calls;14NCU CPU tests pass. Neither version reached GPU in the failed rentals. |
| Organization | User requested descriptive names and topic folders.2849files moved;2802raw evidence SHA256s unchanged;351report links checked. Historical session logs and captured paths remain immutable; migration map resolves them. |

Keep ordered-bit guarantees separate from library budget admission. Keep static instructions, actual selected images, diagnostic counters and ordinary timings separate. Do not infer missing IR facts or change compiler policy from a library name, tile count or static opcode count.

## Live state at this checkpoint

| Handle | State and next action |
| --- | --- |
| Branch `s54-gpu-fusion`, prior committed head29ad618 | Root is preparing a local checkpoint with integration, results and requested organization; no push. Preserve unrelated `oainotes.md`. |
| `/root/comparison_run` | Read-only provider failure diagnosis in `/private/tmp/mapal-s61-gpu/attempt-07/`; compare private records with successful05. No further create until cause evidence reviewed. Never print raw provider errors. |
| `/root/composed_analysis_s61` | Native fused/unfused cuBLASLt/CUTLASS reference implementation in `/private/tmp/mapal-composed-fused-references/`; no CUDA compile/GPU result. |
| `/root/gap_analysis_s61` | Independent IR fact map and NCU/C++ argument audit under `/private/tmp/mapal-s61-gap-analysis/`; reviewed NCUv3 prepared. |
| GPU rentals | None live. Owned IDs57242062,57245029,57247103,57248819,57249550,57253902,57254920 independently absent. Last absence18:33:16.146Z. |
| Protected user instances |57065574,57100404,57126023,57133432 plus every fresh prelaunch ID. Never access/stop/destroy them. Only exact own create-response ID plus physical match establishes ownership. |
| Private recovery | `.claude/worktrees/_archive-s54-20261021/s61-performance-analysis/`, mode0700; raw traces and reviewed archives mode0600. Public recovery receipts under `benches/results/gpu/operations/performance-comparison/` and relevant experiment preparation directories. |
| Authorization | Existing explicit benchmark uploads/runs and equivalent-host approvals persist; user removed monetary cap. Operational timeouts remain. No new budget question. An earlier credential rotation remains outstanding. |

Reviewed competitive archive: `/private/tmp/mapal-s61-competitive-reference/competitive-fp32-reference-v2.tar.gz`, SHA713433bf62bc7dc744bc1f2e3a917f73a83e3a1553726202094e569698475064; manifestab7495e45fb491a696290a444abf230b4704aeb0f3e68af68aa95c0e40c1d17f.

Corrected profiler archive: `/private/tmp/mapal-s61-gap-analysis/ncu-extension-v3.tar.gz`, SHA243dd5297c9a12e814b7a4e244372258cf756bbb34e239577e3287b11601c29b. Ordinary timing must precede its selected-binary diagnostic; use original admitted confirmation argv plus only `--profile-leg`. Preserve all internal library kernels, first live Mapal smoke, actual tool/version/unit checks, raw reports and independent pre/post numerical gates.

## OpenSpec and remaining work

| Change | Progress | Next duty |
| --- | ---: | --- |
| benchmark-large-composed-gpu-workloads |5/15| Competitive primitive confirmation, supported fused-library controls, broader shapes and causal retention/graph-mode comparisons. |
| optimize-ordered-smallm-gpu |4/7| Keep alternatives isolated; prove selector profitability and preserve wide routes before integration. |
| eliminate-unmaterialized-gpu-allocations |11/15| Ordinary latency/overlap evidence, final reconciliation and archive only completed obligations. |
| capture-executed-cuda-images |8/11| Actual cuBLAS/CUTLASS dispatch/SASS/counters and evidence-backed fact/realization gap mapping. |

The immediate next step is the operator's sanitized startup diagnosis. Once resolved, execute the already-reviewed competitive packet and corrected profiler on a newly owned instance. The fused-reference worker can continue independently. cuBLASLt column-major reinterpretation preserves row-major bytes and bias semantics; no precision fallback is permitted. Runtime support and fixed-budget gates remain mandatory.

The IR source audit shows shape, affine layout, order, independent outputs and substantial storage/fusion facts already reach lowering. Concrete limits include GPU ksplit rejection, separability/vector admission and conservative same-cell fusion boundaries. Which realization closes the cuBLAS gap remains unproved; see [fact map](../performance/gpu/disassembly/ir-facts-and-realization-gaps.md).

## Documentation and checks

Implementation → architecture → component/root status were reconciled for allocation elimination and actual Mapal image admission. [Allocation review](../components/backend-nvptx/reviews/allocation-and-performance.md) and [organization review](../components/backend-nvptx/reviews/evidence-organization.md) record FRAMEWORK§4.5 boundaries. No new production placement or runtime transmission is introduced by evidence relocation.

`cargo fmt --all -- --check` passes; strict OpenSpec7pass/0fail; `supercharge-drift`0dead/575refs. GPU baseline generator output is identical after relocation except its descriptive title; NN and packing report generators are byte-identical. Changed shell/Python benchmark entry points pass syntax checks. The staged whitespace check reports only original bytes in the captured allocation patch and test log; those raw artifacts are preserved intentionally. No historical session log was edited.

New results use `docs/performance/<topic>/` and `benches/results/<topic>/<experiment>/`, with descriptive filenames. [Performance index](../performance/README.md), [evidence index](../../benches/results/README.md), and [path map](../../benches/results/path-map.json) are the entry points. Do not reintroduce session-number result titles or catch-all directories.
