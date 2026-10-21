## Context

At planning time all five frozen GPU primaries and two confirmations had completed, while the separately approved ordering/matrix pilot still required baseline b4152c5 and its exact binaries. The isolated diagnosis uses a git-archive source snapshot and Cargo target under `/private/tmp/mapal-replay-diagnostic/`; it is not a git worktree and cannot replace shared binaries. The pilot attempts later closed before upload, permitting reviewed source integration while retaining the old binaries.

Existing evidence is `/private/tmp/mapal-s56-gpu/large-composed/recurrent-checkpoint-root-cause.md` (SHA e05abcadb16ed26acfb73750573f5eb4aaa3062b2d9e2f192acf1f6b69aa67be) and the immutable composition/minimization receipts. The initial static candidate was SCC-only loop reconstruction omitting an acyclic exit payload. The isolated diagnostic below confirms exact numeric feeder provenance before repair selection.

## Goals / Non-Goals

Goals: identify the exact failed replay precondition, preserve failing evidence, implement only the demonstrated generic repair in the temporary snapshot, and provide a meaningful negative control plus semantic regressions.

Non-goals: changing shared production source/HEAD/binaries, GPU runs, allocation/stream changes, numerical tolerance changes, broad optimizer redesign or new fallback semantics.

## Model and consolidation

Existing Dat are `CategoryIr`, `RewritePlan`, `LoopPlan` and the old-to-new object map. Existing Trn is `replay : (CategoryIr, consistent RewritePlan) -> CategoryIr | NonCanonicalLoop`, implemented by `replay.rs:replay/reconstruct_loop/reconstruct_a/feed`. Canonical loop attribution and complete decide/advance cones are deduced by `mapal-ir/src/algo.rs:CategoryIr::loop_plan`.

No new Dat, Loc, Trm, public signature or model field is planned. If the candidate is confirmed, reconstruct the required region from existing plan facts rather than maintaining a second loop-region representation. Replay is expected to rebuild a canonical accepted graph without reading an unmapped resolved feeder, preserve builder well-formedness, and commute with observable evaluation. FRAMEWORK §3 consolidation and §4.5 law5 composition soundness apply; law1's input-presence obligation is checked at the existing compiler-memory execution site. Physical location/transmission laws have no new boundary in this pure rewrite change.

## Decisions and proof obligations

1. Before edits, bind the shared HEAD,328 source entries,4 frozen inputs and3 LLVM tools to the original frozen manifest. Archive b4152c5 read-only into a new temporary directory. Use `CARGO_TARGET_DIR` there and `--offline --locked`.
2. Instrument only the temporary driver and loop replayer: record pass/round/channel counts and each back/exit route feeder's identity, resolved alias, object/type/definition, SCC/driver membership, mapped status and planned removal/constant state. Run the existing tiny counter-print source, then the original checkpoint source. Keep full commands/status/stdout/stderr and hashes.
3. Select a repair only after binding674 and5907 to a concrete violated invariant. Revise the candidate if diagnostics refute it. A missing feeder is not repaired by inventing a value, dropping an observation or silently disabling optimization.
4. If confirmed, derive complete required loop work from existing `LoopPlan` decide/advance facts, retain deterministic object order and done/remap bookkeeping, and honor existing internal-pack/alias/drop/constify handling. Preserve invariant producers before the header and independent loop ownership.
5. First force empty-plan identity replay on small legal loop shapes, asserting the acyclic required payload shape and equal complete interpreter outcomes. Then test affected passes/full rewrite, zero-trip and multi-iteration effects, computed exit values, sequential loops and guarded trap controls. Negative-control the regressions against the unmodified snapshot.
6. Remove temporary tracing from the proposed fix patch. Confirm shared source/tool/binary hashes remain unchanged and save a narrow reviewable patch, proof note and exact test commands. Do not apply it to shared production source in this task.

## Risks / Trade-offs

- A route or plan-removal bug can resemble a scheduling omission: diagnostic object/pass provenance is required before selection.
- Broadening loop ownership can accidentally consume invariants or another loop: use the canonical existing region and focused boundary tests.
- Newly included non-SCC objects may have plan entries: preserve ordinary replay classification rather than bypassing those entries.
- A test can pass because the fixpoint driver bypassed replay: explicitly invoke empty-plan replay in the regression.
- A temporary proof can be mistaken for shipped behavior: keep the shared application/reconciliation tasks open and label every artifact as isolated.

## Migration Plan

This task produces temporary evidence and a proposed patch only. Shared application, appropriate host checks, implementation/architecture/status reconciliation, §4.5 review and drift validation precede a later archive. No current baseline artifact is overwritten.

## Confirmed isolated diagnostic

Both fixtures fail in round1 ConstFold, alias14 and every other plan channel0. Tiny674 is exit-route675 slot0; original5907 is exit-route5908 slot0. Each is an unmapped Temporary `(IoToken, Array<f32;2 or8>)` outside the SCC but inside the canonical driver cone, Pair-built from the merge's token/y projections, with no drop/constify entry. Other route feeders are mapped. This confirms SCC-only omission, rather than alias redirection or producer removal. Full command/status/definition traces are preserved under `/private/tmp/mapal-replay-diagnostic/evidence/`; the instrumented binary and diagnostic-only patch are retained there. The temporary repair derives membership from existing SCC+decide/advance target facts and delegates object classification to existing `reconstruct_a`, including non-SCC plan entries. At diagnostic capture, shared production source was unchanged.

## Isolated validation and integration handoff

`/private/tmp/mapal-replay-diagnostic/PROOF.md` and `evidence/ready-for-review.json` bind the final formatted patch32384fc81b26faff5006019b37d330daab9475bc2e1aa799afde2091282d8203. The exact final regressions fail5/5 on unchanged replay and pass with the repair; the whole rewrite crate passes86/0. Raw, identity-replayed and fully rewritten original interpretation all match the complete8408-value scalar fp32 checkpoint reference and8838 output lines. All8 original RTX5090/A100 Exact/Contract shipped/full-recompute GPU-facing LLVM emissions pass. Tiny native Exact/Contract values pass. Original default native Exact/Contract and one `--no-tile` Contract emission each reach a40-second cap; those native paths and their latency remain unresolved, with no GPU/performance certification claimed.

The pre-integration receipt confirms all328 shared source entries,4 frozen inputs,3 tools and HEAD unchanged. Root subsequently applied the exact two reviewed source/test files and owns the full shared required-clang check and documentation reconciliation; the historical frozen receipt does not describe shared source after that application. Old baseline binaries/payloads remain separate. Integration tasks remain open until those checks finish.
