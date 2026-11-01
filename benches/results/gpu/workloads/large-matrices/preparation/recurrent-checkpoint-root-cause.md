# Recurrent checkpoint replay failure — bounded root-cause analysis

Reviewer: `/root/audit_legacy_worktrees`, 2026-10-21. Only this report was written. No source/HEAD/frozen-binary change, Cargo invocation, patch, GPU, rental or upload occurred. Existing minimization was read before source investigation; no minimization or long compiler test was repeated.

**Finding:** the failed invariant is replay's feeder-remapping precondition. There is a precise source-level defect candidate: `reconstruct_loop` rebuilds only SCC members before reading an exit payload, although valid lowered loops can have a required acyclic payload pack outside that SCC. The canonical `LoopPlan` already identifies the larger driver cone. **The numeric identity of the observed ObjectId5907/674 feeder has not been traced to that pack, so this remains a strongly supported candidate for this particular panic, not a fully confirmed root cause or an approved fix.**

## Preserved failure and distinguishing evidence

Original source: `composition/generated/recurrent-t16-d8-h16.check.mapal`,27,897 bytes, SHA256 `0e4c05b403550da3a3afaeebcea8323e2af1dca887aa6a94d95c1e0dd7c57142`. Immutable certification manifest SHA `e65b89fba065ea650b19298d1e820a7845ad767478715af192107e8a8318cf0a`.

`composition/CERTIFICATION.md` and certification-01 receipts establish interpreter acceptance/exact output, then rewritten native Contract plus all eight GPU profile/face/retention emissions failing at replay.rs989 with unmapped `ObjectId(5907v1)`. `examples/emit.rs` calls `mapal_rewrite::rewrite` before backend emission. The failure therefore precedes backend/GPU arithmetic, allocation, streams or runtime; it is not evidence about those mechanisms. No-rewrite native timeout and A100 CPU-placement emission do not establish optimized coverage.

I independently rehashed every source/stdout/stderr row in `rewrite-investigation/summary.json` (SHA `d26dd0ce7f6deec21668f258705e6e7843539df7ab0a96dfaef68f3f23b6a018`) against the existing records; all match. Its original.json SHA is `c0abd0b851798b7a15e846c49e0a6560ea1da782eb16a053cb2392f7f13e3fa6`. Before/after snapshots are equal.

| Existing variant | Interpreter | Rewritten emitter | Feeder shown in saved stderr |
|---|---:|---:|---|
| tiny-check |0|101|1119v1 |
| tiny-check-no-input-observation |0|101|936v1 |
| tiny-y-check-only |0|101|690v1 |
| tiny-counter-print-only |0|101|674v1 |
| tiny-run |0|0|— |
| counter-loop-print |0|0|— |
| array-map-loop-print |0|0|— |
| array-update-fold-loop-print |0|0|— |

The counter-only failing variant removes checkpoint arrays/formatting and still fails: printing all intermediates is unnecessary. Pure run-only recurrent arithmetic succeeds; printing in an arbitrary simple loop does not universally fail. The differing numeric IDs are graph keys, not stable source identities. Existing stacks identify `reconstruct_a` but do not identify its exact inlined feeder call or rewrite pass/round.

## Exact failed invariant

`crates/mapal-rewrite/src/replay.rs:984` (`feed`) resolves aliases, then requires `remap.get(resolved)` to exist. At989 it panics when absent. Its documented precondition is that resolved feeders precede consumers and have already been reconstructed. The failure is:

```text
a feeder used now ∉ domain(old→new remap)
```

The saved error prints the same old and resolved ID, so this lookup did not redirect to a different terminal alias. It does not prove which prior pass ran, whether that object was wrongly skipped/dropped, or which original source construct owns it. Those require the diagnostic below. The accepted source is lowered/sealed, and replay's canonical-loop gate ran before reconstruction; no input CategoryIr well-formedness violation was established. This is an internal replay-order/remapping failure, rather than a user syntax error.

## Precise static candidate and provenance chain

1. Lowering `emit.rs:2622` (`emit_loop`) appends IoToken to carried state when the loop body is effectful. `emit_routing_guard` restores the pre-jump token/state for the exit arm. For `-false-> y -> final`, `emit_exit_arm`, `ExitTerminal::Bind` at3104–3117 constructs a fresh `(token, value)` payload pack whenever the loop carries a token, then calls `loop_exit` on that payload.
2. `mapal-ir/src/builder.rs:2435` (`loop_exit`) constructs a separate `(payload, cond)` route and the LoopExit target. The inner token/value pack feeds only the exit route. It cannot reach the merge/back edge, so it is acyclic and outside the loop SCC. This is legal Core; a loop driver can require values outside its SCC.
3. `replay.rs:927` (`is_internal_pack`) does not classify this inner pack as an atomic packed-source primitive: its sole consumer is the exit route's `Pair` edge, and `reads_packed_source(Pair)` is false. It therefore needs its own materialization/remap entry. The outer route itself is rebuilt atomically by `loop_exit`.
4. `replay.rs:256` (`reconstruct_loop`) obtains the canonical `loop_plan`, but builds membership from **only** `lp.scc_objects` at295. Its inner global-topo walk explicitly skips every non-SCC object at317. Before the outer flat walk can reach an acyclic exit payload, the routine asks `feed(..., ef[0])` at365 while wiring `fb.loop_exit`.
5. Thus an unaliased, not-previously-built token/value exit pack is required before replay maps it. No fallback in `feed` can repair that ordering. The same structural gap covers other acyclic computed route payloads; token-bearing packs are a concrete lower-generated case, not a special architecture rule.
6. `mapal-ir/src/algo.rs:3005` (`loop_plan`) explicitly models this situation: the comments and body cone include loop-variant route payloads that leave the SCC. `body_order` at3123 includes their producers, and `decide_order`/`advance_order` expose that full region. The replayer reads the canonical plan but restricts reconstruction to less than the plan's driver region.

This chain establishes an implementation assumption that is too narrow and a source-generated shape exposing it. It does **not** establish that the observed5907 or674 is exactly that inner pack, or that no earlier missing feeder precedes it in those graphs. The current panic does not include object metadata, route/merge identity, plan channel or phase. No numeric-key guess is treated as provenance proof.

The passing simple-loop controls do not refute this candidate. `driver.rs:106` skips replay entirely when a pass plan is empty; pass-through rewrite is not the identity-replay test. Whether each saved successful control took that bypass was not recorded and is not asserted here. An explicit empty-plan `replay` regression is needed to eliminate this ambiguity.

## Smallest confirmation after the freeze

No diagnostic patch was made. Add temporary tracing immediately before driver.rs109 to print pass/round/plan counts. In `reconstruct_loop`, before wiring back and exit routes, print both slot-feeder vectors and, for each resolved feeder: old/resolved ID, object kind/type/name/loc, defining in-edges, membership in `lp.scc_objects` and decide/advance target sets, `remap.get`, and plan alias/drop/constify state. This is sufficient to distinguish an unscheduled valid payload from a plan that removed its producer. Do not fix the panic by manufacturing an unknown value or skipping the effect.

Run only the existing millisecond counter-print minimization first, with an isolated post-freeze diagnostic build directory:

```sh
CARGO_TARGET_DIR=/private/tmp/mapal-recurrent-replay-diagnostic-target \
  cargo run --offline --locked --release -p mapal-backend-llvm --example emit -- \
  /private/tmp/mapal-s56-gpu/large-composed/rewrite-investigation/variants/tiny-counter-print-only/source.mapal \
  - --rewrite --contract --target=apple-m
```

That command is a **future** diagnostic requiring the freeze to end; it was not executed. If674 is the exit payload pack and the lookup precedes its reconstruction, repeat the same trace on the original source to bind5907. If the trace identifies another feeder/pass mechanism, revise this candidate before choosing a fix.

The frozen CLI has neither IR-dump nor pass-isolation flags; mapal-cli dump-ir is a stub. Semble MCP/CLI indexing stalled and was terminated. Existing `/usr/bin/lldb` resolves `FnBuilder::loop_exit` in the frozen emitter, but a single attempt to launch the tiny existing fixture under it failed with `process exited with status -1 (no such process)`. No register or IR provenance was obtained. No signing/debug permissions or frozen tool was changed.

## Conditional generic fix and regression scope

If the trace confirms this candidate, the smallest generic repair is to reconstruct the canonical loop's complete required driver region from existing `LoopPlan` decide/advance target membership plus SCC objects before wiring its routes. Preserve global deterministic object order, existing done/remap bookkeeping, internal-pack handling, and per-object alias/drop/constify semantics for newly included non-SCC objects. Keep loop-invariant producers before the header and neighboring-loop ownership intact. A token-only special case would leave the computed-payload form uncovered; no such patch is proposed or selected now.

Meaningful regression additions to `tests/identity.rs` should force **actual** `replay(ir, empty_plan)` on a tiny effectful loop with a bound exit value, validate the rebuilt graph, and compare full interpreter output/outcome. Assert in the test that the required token/value payload is outside the SCC and belongs to the canonical decide cone. Include a pure computed exit-payload control, zero-trip and multi-iteration cases, and the existing simple print loop so an empty-plan driver bypass cannot hide the failure. Then run each pass/full rewrite on the minimized recurrence, preserving all checkpoint/output gates; include two sequential loops and guarded-loop trap/effect controls to catch accidental cross-loop closure or exit-step extra work. Existing property/golden suites remain relevant after a demonstrated repair. None was rerun here.

## Freeze receipt

Current HEAD remains `b4152c5f4d9ad802d0388c1c64b628091d494e6a`. Independent final read-only hashing matches all328 source entries,4 frozen emitter/interpreter/runtime inputs and3 LLVM tools against source-manifest SHA `afdb6d16c0fbeb31789d2760a82ec48ffacd206d9f957af8d0cac72608e62133`.

Reviewed key source hashes: replay.rs `cbbbc4346f7b8adb31993774cd1216bdef78b7e2bd5226be32debce15f69e1b5`; driver.rs `a23979a31807baac40483763acd9dbc7944b72afdafe8502d011bbccf68ea5c8`; lower/emit.rs `9582eb50d0f900ea7a91b8913aa7e9f37bb360e3a28078a666c7a928dfb29af7`; ir/algo.rs `5b4d8320a6a7162d5e1943afcc29fd2ded03aaba9f71cf492dcc72c515e637f6`; ir/builder.rs `825543829c016852bcb9a1dd0e1fb5443314d2ee2bf18b49154a3d7d3f9a0040`.
