## Context

The dated [implementation touch map](implementation-touch-map.md) identifies the existing
writer relation, all three consumers and focused regression scope. It is a source-grounded
plan, not an implemented correction or runtime result; its input hashes describe the
documents as read before this link was added.

See proposal.md. The frozen baseline is `b4152c5`; all source, HEAD, compiler/runtime
binaries and final benchmark inputs stay unchanged until its five-profile and independent
confirmation runs finish. Temporary investigation uses the already-built tools only.

[The evidence report](../../../docs/performance/gpu/memory/ordering-investigation.md) records three
distinct findings: unused raw output allocations in saved stream-enabled LLVM; a current
stored-member host readback with no emitted producer wait; and a stored member crossing
into a second tile stream without an emitted event. Same-stream and member-row-fold
concerns were refuted for their inspected shapes. A legacy-cell case has a real allocation
between producer and consumer, so its runtime ordering remains unresolved.

No GPU wrong output or speedup is established by these compiler-output investigations.
The first omitted-raw fixture is currently ordered by nonzero frees; elimination must not
silently remove those context waits. The stronger materialized-raw fixture has no such
intervening call. Both have interpreter oracles and unchanged-source receipts.

## Goals / Non-Goals

**Goals:** consume existing physical-writer and storage facts consistently; preserve every
required dependency while omitting proved unnecessary raw storage; validate across
independent, chained, forked and host-observed dataflow shapes.

**Non-Goals:** source hints, benchmark-name rules, new numerical policy, allocation
hoisting, pooled ownership, arena reuse under streams, new CUDA runtime ABI, graph replay
or a new parallel dependency representation. The existing arena guard stays intact.

## Decisions

1. **Evidence before edits.** Preserve and inspect the generated LLVM fixtures and runtime
   paths. After baseline validation, run ordinary untraced GPU reproductions before editing
   source, including the materialized-raw and different-chain fixtures and their controls.
   Record outputs, statuses, source/binary hashes and actual producer/consumer ordering.
   Tracing and clock insertion are not equivalent controls because they synchronize. A
   passing GPU execution alone cannot prove an absent dependency safe; an explicit emitted
   edge or documented runtime ordering must establish the guarantee. Report observed
   wrong values separately from the compiler-output evidence.
2. **Use the existing group as the writer authority.** `Epi.store_out`, `Member.store` and
   member outputs already describe what the grouped kernel writes. `FnEmit::arena` already
   maps a member's birth to its base launch. If the proof duty requires a writer lookup,
   factor that existing relation into a small private query, canonicalizing through the
   current alias relation. Do not retain another independently maintained map or infer a
   writer from allocation timing. `ResidentPlan::stream_of` and its existing consumers are
   the investigated mismatch points. Check all three consumers: launch selection, feeder
   events and host element readback.
3. **Preserve the eligibility boundary for unordered consumers.** The existing `chains`
   veto is expressed over raw targets. Inspect emitted legacy-consumer cases before
   changing it. A stored member requiring a legacy consumer must have an established
   completion edge or cause the conservative chain fallback. Do not assume a real
   allocation's incidental wait is the intended generic dependency mechanism. The observed
   row-fold case already has an ordered whole readback and must remain a control.
4. **Only then omit dead raw allocations.** `chain_prologue` must allocate a base target
   precisely when the selected launch stores it: an ungrouped site or a group with
   `store_out`. Keep stored member allocation positions unchanged. Existing zero-initialized
   slots and `mapal_gpu_free`'s zero-slot return retain ownership; however, the resulting
   removed synchronization must be covered by the dependency proof above. Do not remove
   live allocation/free calls or change timer placement to obtain a better timed result.
5. **Separate correctness and profitability.** First show emitted pointer, event, copy and
   lifetime behavior and value/trap equivalence. Then compare the same forced-stream
   realization before/after on supported hardware, with ordinary iteration, process wall,
   allocation counts and requested-live-byte accounting. The shipped 5090 streams-OFF
   baseline is not an isolated control for this change. Report unchanged cases and losses.

### Model and consolidation

No new runtime `Dat`, `Loc` or `Trm` is introduced. Existing `Dat` are the IR array identity,
canonical alias and grouped placement record. The planned private query has signature
`(array identity, existing groups/placement) → actual producing launch or no device writer`.
It is partial for unplaced/host-produced objects; unknown ownership retains the existing
conservative placement rather than inventing a dependency. Its semantics are the producer
of the physical bytes after grouping, not necessarily the original logical Map producer.

Existing `Trn` are storage admission and host/device emission in `resident.rs`; existing
`Trm` are Host/Gmem copies and launch control. Cross-stream event waits remain ordering
within Gmem, not array transmissions. The implementation uses the existing runtime ports
`mapal_gpu_event` and `mapal_gpu_d2h_on` where justified. FRAMEWORK §3 consolidation reuses
the existing group; §4.5 laws1/4 require readiness/mediation and law5 requires grouping and
stream placement to compose. No new compiler pass or machine fact in CategoryIR is needed.

## Risks / Trade-offs

- Removed allocation/free calls can remove waits → demonstrate completion edges for every
  surviving consumer before elimination; preserve the no-clock adversarial fixtures.
- Wrong alias/writer mapping can wait the wrong chain or omit required storage → test
  stored members, raw fanout, host readers, different-chain readers and unchanged controls.
- A conservative legacy-consumer fallback can change performance → record its exact
  emitted scope, value evidence and measured cost; do not claim a general speedup.
- Requested-byte predictions are mistaken for measured VRAM → label the static accounting
  and report actual allocation counters separately. No pool or enlarged allocation exists.
- A timing benefit appears only after moving work outside a clock → no live allocation
  relocation in this change; measure process wall as well as the existing iteration clock.

## Migration Plan

Finish the frozen baseline, preserve its artifacts and cleanup receipts, and run the
targeted reproductions. Implement only corrections supported by those artifacts and the
documented ordering contract, with a regression that fails the old emission. Apply the
dead-storage correction after its safety obligation is satisfied. Validate targeted and
workspace tests, then hardware correctness and performance controls. Reconcile the NVPTX
implementation map, architecture/status and §4.5 review; run drift and archive.

The proof artifacts remain immutable. Source corrections can be reverted independently
of the baseline/report commits; do not discard measured failures or refuted hypotheses.
