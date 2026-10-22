# Isolated grouped-writer ordering correction

Scope: temporary source snapshot `/private/tmp/mapal-ordering-fix/source`, historical HEAD `b4152c5`, target `source/target`. Only `resident.rs` and `tests/streams.rs` differ from that snapshot's frozen source set. Shared source/HEAD/tool changes were not made by this agent. The parent deliberately integrated replay/docs separately; `after.json` records that exception rather than claiming all328 shared sources remained frozen. Relevant ordering sources, GPU runtime, four frozen binaries/runtime archives and three LLVM tools still match the before receipt.

This closes the code/runtime/API obligation for the selected grouped-output dependency correction. It does **not** establish GPU wrong output, race frequency, allocation-elimination safety for every possible consumer, or a performance gain. OpenSpec1.3 ordinary untraced GPU baseline is still pending; allocation DCE remains deferred. The parent owns shared integration and final review.

## Proven emission and the required completion relation

The previously preserved materialized-raw fixture has grouped writers on nonzero streams, then member corner copies via legacy `mapal_gpu_d2h`, with no runtime call between the last launch and first member copy. The fork has one same-stream child and a second-stream child reading the same stored member with no event. The new capture fixture feeds a stored member only through another group's `Epi.caps`; both arithmetic faces similarly omit its event in old emission. The calibrated old-source regression run exits101: these three dependency assertions fail, and the member-cell conservative-veto assertion fails. The member-rowfold completion control passes. Exact failure output and the initial assertion-calibration/constructor failures remain preserved.

The distinction is physical ownership: each stored `Member.out` is written inside its base group's kernel, not at the member's logical Map node. `Epi.store_out=false` can suppress the raw result without suppressing that launch or its stored members. `FnEmit::arena` already used the member→base relation for buffer birth; the correction factors that scan once as private `grouped_writer`. `ResidentPlan::stream_of` canonicalizes the existing carry alias, then queries that owner and the existing chain assignment. Base launch lookup and no-chain/host fallback remain. No new maintained map, pass, runtime port, policy row, hint or annotation is introduced.

Runtime source `crates/mapal-rt/gpu/mapal_gpu.c` fixes the relevant semantics:

- `init_all:159–161` creates chain streams with `CU_STREAM_NON_BLOCKING`; legacy stream0 therefore supplies no implicit dependency on those producers. See [CUDA12.8 driver stream management](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-driver-api/group__CUDA__STREAM.html) and [default-stream synchronization](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-driver-api/stream-sync-behavior.html).
- `mapal_gpu_d2h_on:388–393` submits the copy to its supplied stream, then synchronizes that stream before returning host values. Correctly selecting the producing stream orders its earlier launch, the copy and the subsequent host load. The [driver stream API](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-driver-api/group__CUDA__STREAM.html) specifies that stream synchronization waits for its operations to complete.
- `event:290–295` records on the producer stream and enqueues a wait on the consumer stream; `mapal_gpu_event:306–308` maps logical stream IDs through the unchanged slot mapping. A recorded event captures prior producer work, and a wait delays later consumer work until that captured work completes ([event API](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-driver-api/group__CUDA__EVENT.html), [stream API](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-driver-api/group__CUDA__STREAM.html)). Re-recording a reusable event does not alter the state already captured by an enqueued wait. Ordinary same-stream work retains FIFO ordering; no extra event is emitted for that edge.
- `launch_on:321–324` automatically joins prior legacy generation into a chain. It does not join arbitrary other chain writers. `launch_on:352–353` synchronizes the context only under tracing, which is why tracing cannot replace ordinary baseline evidence.
- `dev_alloc:230–236` returns a nonzero cached slot without a new allocation; it provides no such dependency for the saved fork. `mapal_gpu_free:553–564` returns immediately on a zero slot; for a live slot and busy chains it explicitly waits for the context before freeing. Neither function changed. No allocator-side wait is assumed in the new dependency proof.

## Correction and conservative boundary

All three existing stream consumers use the same grouped-writer lookup: launch selection, feeder events (ordinary tile operands plus `Epi.caps`) and element readback. Existing event sorting/deduplication and same-stream filtering remain. In the capture case, the event precedes the consuming launch, with the existing live stored-member allocation between them; the test explicitly permits this allocation and does not hoist it. The host readback selects streams2/1 for the respective stored members; four legacy generator-corner reads remain legacy, and raw readbacks retain their existing stream IDs.

The existing unordered-consumer veto now resolves its actual read operands through this same writer fact, including canonical identity, before testing candidate-chain membership. A separate legacy cell reading a stored member therefore uses the same conservative all-legacy placement as the raw-output control. The old member-cell fixture contains a real allocation that may mediate or mask execution; no old-runtime wrong value is claimed. The correction restores the existing veto's intended eligibility relation rather than relying on that allocation as an ordering contract or adding a new cell-event protocol. Member-rowfold remains the already ordered whole copy on its producer's stream, while base-rowfold retains its legacy veto.

The arena's `plan.chain.is_empty()` guard, straight-line loop/guard eligibility, `chain_prologue`, runtime ABI, allocation/free implementation and live allocation positions are unchanged. Existing arena birth now calls the factored identical scan. The lookup unit covers canonical view/exit→merge identity, separate next-slot identity, unassigned/host fallback and a base launch whose raw result is unstored. A real carried-loop emission control stays legacy in Exact and Contract. These checks do not admit asynchronous loop/guard work.

## Checks and preserved artifacts

- `old-final-tests.log`: exact final regression source against old production yields four intended failures and two unchanged rowfold/loop controls passing, exit101. Fixed production was restored in a finally block; `fixed-final-streams.log` then passes all17 stream tests. The earlier calibrated log is also retained. No compile error substitutes for an emission failure.
- `focused-unit-2.log`: one alias/writer lookup unit passes.
- `focused-integration-2.log`: epilogue13, resident13, rowfold6, streams17 pass (49 integration checks;50 including the unit). All cargo builds/tests used offline+locked mode and the isolated target.
- `fixed-emission-commands.json`:14 successful emissions, seven sources×Exact/Contract. All14 device LLVM units are byte-identical to their old counterparts. Alloc/free call counts remain identical for materialized-raw, fork and capture; the legacy-cell fallback intentionally selects the existing legacy placement, rather than promising unchanged allocation policy there.
- `ordering-only.patch`: only the two owned files. `ordering-proof.json` binds source/fixture/LLVM/command/log hashes, call sequences, controls and the before/after boundary. Failed development attempts are retained with explanations; no numeric gate was relaxed.

The smallest next step is independent patch/API/emission review, then the authorized baseline and before/after ordinary GPU correctness gates on exactly matched placements. Do not remove allocations, broaden stream eligibility, claim a speedup or merge this temp patch solely because the focused checks pass.

## Exact local resume (candidate remains unapplied)

```sh
cd /private/tmp/mapal-ordering-fix/source
CARGO_TARGET_DIR=/private/tmp/mapal-ordering-fix/source/target cargo test --offline --locked --release -p mapal-backend-llvm --lib grouped_stream_lookup
CARGO_TARGET_DIR=/private/tmp/mapal-ordering-fix/source/target cargo test --offline --locked --release -p mapal-backend-llvm --test streams --test epilogue --test resident --test rowfold
```

The isolated emitter is `/private/tmp/mapal-ordering-fix/source/target/release/examples/emit`; exact fourteen fixture argv/output paths are in `fixed-emission-commands.json`. No local build/test or GPU job from this task remains active. Shared integration, ordinary GPU baseline1.3, new-code GPU equivalence, allocation DCE and performance acceptance remain pending. Keep the raw baseline and candidate artifacts separate.
