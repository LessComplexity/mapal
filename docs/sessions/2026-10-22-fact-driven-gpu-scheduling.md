# 2026-10-22 — Fact-driven GPU scheduling and observer findings

## Continuation brief

The scoped compiler correction is implemented, independently reviewed and ready for its
local checkpoint. Existing GPU facts now select policy using the actual arithmetic on
`Launch`, including strided Contract and rejected TF32 fallbacks. The broader performance
task continues: the first new library-observer pilot rejects attribution, and its owner
and reviewer are repairing the three proved instrumentation boundaries locally.

Resume with `cat docs/next-session.md`, `openspec list --json`, and
`cat /private/tmp/mapal-s61-root/live-state.json`. No live task rental exists at this
checkpoint; no new library image/SASS or compiler speedup is admitted.

## Completed work and decisions

| Decision | Result and evidence |
| --- | --- |
| Propagate actual arithmetic through existing Launch | Built in `gpu.rs`; shared IR already has the necessary shape/layout/order facts |
| TF32 permission versus selected implementation | Tensor selection retains existing gates/precedence; rejection resolves and rekeys scalar policy |
| Scalar semantics and module PDL | Retained source order/seed/operand behavior and shared host-flag/kernel-wait derivation |
| Warp-fill register budget | Retained: source history, documentation and an existing geometry test establish intentional conservative policy |
| Blanket small-M fit | Not promoted; previous measured regressions remain disqualifying |
| Timing attribution | Existing warmed resident clocks include completion. Matched input/sync and inclusive request controls remain needed |
| Failed capture evidence | Immutable, no deduplication or reconstructed attribution. Repair standalone observer, then repeat a fresh reviewed smoke |

The source correction is at `crates/backends/llvm/src/func/gpu.rs`, SHA256
`e932fcb811d5a245d4dfcc03134928af85af19443210014089c5e66f97092205`.
Baseline is `d80d8f5d66ddee712bf552337b3ea1684cad5eea`. Only that production file changes.

## Checks and evidence

- Two original-API regressions compile and fail on the old code after confirming emitted
  arithmetic. All65 library and176 integration tests pass; one optional baseline ignored.
  The three initial sandbox native-detection failures are retained; all65 pass with host
  query access. Integration clang is present/executed, but its forced guard was unset.
- Eight successful controls preserve raw host LLVM, device LLVM and PTX; optimized LLVM
  differs only in normalized source paths. Five affected cases preserve arithmetic
  inventories and PDL waits while policy output changes.26 device modules lower through
  LLVM22.1.8. Independent review checks104 emitted files. No GPU execution or speed gain
  is claimed for this source correction.
- Exact commands, tools and source/binary hashes: `benches/results/gpu/scheduling/fact-driven-selection/verification-receipt.json`.
  Its archive retains194 program/emission files. Independent review and the two pre-fix
  failures are beside it. Formatting and diff checks pass.
- Latest recorded checks before this handoff: strict OpenSpec8/0; drift0dead/581refs;
 456report links resolve; all2,802 relocated historical raw hashes remain unchanged.
  Code graph refreshed:10,177nodes/20,038edges/711communities. Semantic document extraction
  lacks an LLM key; prior gbrain indexing hit an existing PGLite lock, left untouched.

## Observer pilot: precise rejection

Approved bundle was9,130,009bytes, packet
`864e0a146e07f4296450d7fa205f1249f6d7779269f22563b70ec35cdf744651`.
The user explicitly approved this bundle and equivalent RTX5090 hosts. The pilot used
owned57265999, machine135433/host346497; creation20:03:13.056Z, independently absent
20:05:14.535Z. All pre-existing machines stayed protected. All11 task rentals are closed.

Build/injected Mapal process/pre-post checks pass. Independent verification checks1,536
raw words, including Mapal ordered FMA256/256 in each phase, and all453 recovered file
hashes. The library selected-call capture legs are skipped after Mapal attribution fails;
the numeric gates still contain all three implementations. No decoder command ran.

Three actual raw predicates fail: CUPTI correlation IDs are reused by distinct callback
invocations (including repeated CBIDs); a device helper STT_FUNC lives in another
function's executable section; provisional UID0 context-pointer callbacks precede the
authoritative creation event. These are instrumentation schema findings, not performance
causes. The observer owner is repairing them with source-supported identities and
positive/negative tests; no changed packet is approved by this checkpoint review yet.

Public evidence: `benches/results/gpu/disassembly/library-kernels/counter-free-capture/rejected-correlation/`.
Private trace archive SHA256 `26453f6d0cfbdba31c111f4eee9824204f123e0d919d0e2b8304f7f723b82a80`;
persistent copy under `.benchmark-artifacts/gpu/library-images/counter-free/rtx-5090/2026-10-22/rejected-correlation/`.

## Live handoff state

| Handle | State | Inspect / next action |
| --- | --- | --- |
| Branch `s54-gpu-fusion` | Compiler/report checkpoint pending; last commit d80d8f5 | `git status --short`; preserve unrelated `oainotes.md`, no push |
| `/root/gap_analysis_s61` | Local observer repair active | `/private/tmp/mapal-library-image-capture/`; no host actions |
| `/root/composed_analysis_s61` | Independent repair review active | `/private/tmp/mapal-library-image-review/repair-review/` |
| `/root/gpu_schedule_facts` | Compiler implementation finished | `/private/tmp/mapal-gpu-schedule-facts/verification-receipt.json` |
| `/root/comparison_run` | Idle after exact-owned cleanup | `/private/tmp/mapal-s61-gpu/attempt-11/operator-outcome.json` |
| Failed trace | Immutable, retained | `/private/tmp/mapal-s61-gpu/attempt-11/retrieved/trace/` |

Known protected unrelated IDs57065574,57100404,57126023,57133432 plus every fresh prelaunch
inventory ID. Access/cleanup requires the exact new create-response ID, matching physical
host and absence from prelaunch inventory. Never print provider/account/API/create/wrapper
logs. Prior credential rotation remains outstanding. Monetary caps were removed; explicit
bundle/upload approvals persist within scope, with operational deadlines retained.

## OpenSpec and next actions

At the pre-close snapshot: `derive-gpu-schedules-from-facts`7/8, with only this final
handoff/check/archive task remaining; implementation checks and reconciliation complete.
`capture-executed-cuda-images`8/11; `benchmark-large-composed-gpu-workloads`6/15;
`optimize-ordered-smallm-gpu`4/7; `eliminate-unmaterialized-gpu-allocations`11/15.
Archive only the scoped arithmetic correction after this handoff. Broad changes remain open.

Next: review the observer repair, reproduce its boundary tests, seal the exact source,
then have the operator prepare a guarded replacement run. A successful observer must
admit all actual dispatch/image joins before any library-SASS conclusion. Preserve the
old failed trace. The [experiment sequence](../performance/gpu/disassembly/optimization-experiments.md)
records controlled follow-ups for geometry, clocks, CPU submission, retention, allocation,
fusion and separately permitted reduction alternatives.

## Architecture and docs reconciled

Existing Dat Launch gains actual Arithmetic. Existing Trn launch resolves it; policy
selection and scalar emission consume it. No shared IR/device fact, runtime port or
reassociation permission is added. Backend IMPLEMENTATION, ARCHITECTURE §9, component/root
STATUS and FRAMEWORK review are reconciled in that order. Observer repair is explicitly
planned under the existing observation model and OpenSpec design, with no library admission.

Reports: fact-driven scheduling, optimization experiments and measurement boundaries;
raw evidence: scheduling, residual-MLP measurement audit and library-observer preparation/
rejection folders. Current-results, topic indexes and living handoff are refreshed.
