# Planned implementation touch map: grouped writers and dead raw storage

2026-10-21, author `/root/audit_s54_worktrees`. **Planned change only:** no source fix, runtime equivalence or performance improvement is established here. This note maps the already planned OpenSpec change; it does not repeat the LLVM/GPU ordering proof. Semble did not return within15 seconds; bounded fallback used the known source files and preserved report. No builds, new LLVM, tests, HEAD changes or GPU actions occurred. Only this note was written.

## Authority and smallest production scope

The implementation can stay in **`crates/backends/llvm/src/func/resident.rs`** plus focused existing integration-test files. No new CategoryIR fact, pass, persistent map, runtime ABI, allocation owner or machine policy is needed. Existing types and facts:

| Existing authority | Exact scope |
|---|---|
| `FnEmit.groups` (`func/mod.rs:383`) | `SecondaryMap<MorphismId,Epi>`, keyed by the base tile launch; populated before `resident_plan` by `core.rs:set_gpu_arms` (91–120). |
| `Epi` / `Member` (`resident.rs:150–163`) | `store_out` says whether the raw tile result is materialized; `Member.out` and `Member.store` identify each materialized member. `Epi.caps()` (167) lists distinct external Gmem operands in stable first-use order. |
| `FnEmit::epilogues` (694–850) | Local `made` initially maps tile outputs→site and adds `made.insert(mb.out,s)` at823. Final use counts decide storage. This temporary construction map is not retained and should not become another maintained writer table. |
| `FnEmit::arena` (1446; local `site_of`1469–1474) | Already scans `self.groups` for member output→base launch and includes that launch in the member's birth. This is the existing relation to factor. |
| `ResidentPlan::alias` (304) | Canonicalizes current carry views/exit to the merge's existing slot. `next` is deliberately separate: loop-back swaps its slot. `dslot`1606 and operand `dvalid` checks already use this relation. |
| `ResidentPlan.chain` (237) | Sole assignment of eligible base site→nonzero stream. `chains` builds it from the existing PathPlan; no member needs its own assignment. Empty means legacy placement. |

**No reusable general writer query exists today.** `member_src` (595) answers a member body's operand source, not its writer. `produced_on_device` (1588) is only a boolean. `epilogues::made` is construction-local; `arena::site_of` is the narrow reusable relation.

The smallest organization is to factor that member-output→base-site scan into one private query beside these types/functions. Existing `ResidentPlan::stream_of` can receive the existing groups reference, canonicalize `o` through `alias`, consult the grouped owner, then preserve the current logical-producer/chain lookup and zero fallback. Arena can use the same fact instead of its local duplicate. Equivalently, a private FnEmit wrapper can own the lookup; avoid implementing two scans with different rules. Exact private naming is an implementation choice, not a new API requirement.

The query answers **where a grouped value is produced**, while storage admission separately answers whether its bytes exist outside that kernel. Do **not** require `store_out` to find a base launch's stream: an unstored raw result still identifies the launch which computes its stored members. Likewise, group ownership is independent of whether an internal member remains register-only. An externally consumed member must be stored by the existing use-count rule; do not manufacture a pointer for an unstored value. A host/unplaced producer with no chain remains stream0; this fallback does not itself prove readiness for an otherwise unknown asynchronous writer. Unproved ownership must keep the conservative eligibility boundary.

## All three current stream lookup consumers

An exhaustive literal scan finds one definition and exactly three calls, all in resident.rs:

| Consumer | Current site and planned use of the shared fact | Invariant |
|---|---|---|
| Launch selection | `emit_site_resident:1985`, `plan.stream_of(self.ir,target)` | Base tile launch keeps its existing assigned stream, including groups with `store_out=false`; first-chained/prologue behavior must not disappear. Rowfold/off-chain launch remains0. |
| Feeder events | `emit_site_resident:2004–2025`, lookup at2016 | Resolve **both** ordinary site operands and `group.caps()` through canonical grouped ownership. Existing sorting/dedup and `p!=0 && p!=s` filters remain. Emit event(record=producer,wait=consumer) before its launch, not after it. Same-stream FIFO needs no event. |
| Host element readback | `resident_elem:2147–2165`, lookup at2160 | Choose the stored member's actual producer stream, then existing `d2h` routes to `mapal_gpu_d2h_on`. Keep proven-Index bounds and element offsets unchanged. Raw output and aliased carry reads retain their established paths. |

Whole stored-output copy-out **already** uses the current launch's `s` in `emit_site_resident:2070–2088`; retain that correct path. Do not independently rediscover the member writer there. Runtime `mapal_gpu_d2h_on` queues behind the named stream and waits before returning host values; event record/wait orders device streams, not host completion. The runtime ports need no change.

## Eligibility veto and conservative legacy consumers

`chains:1304` admits only straight-line tile sites; `straight_line:410` excludes loop cones and guard-owned work. `DevSite::RowFold` is explicitly excluded from candidate chain order. Existing PathPlan ancestors, wave gate, minimum-two-chain rule and free deferral remain unchanged.

The current veto at1338–1353 compares actual off-chain site reads and `Cell::operands()` against a vector containing only candidate raw targets. Its planned consistency update should ask whether the **canonical operand's grouped writer** is among those same candidate sites. Reuse the query and existing `order`; do not create another member→stream map or expand all IR consumers indiscriminately. Keep `DevSite::reads()` and `Cell::operands()` as the actual-read selectors (including cell-law exclusions).

A stored member fed to a placed legacy cell/rowfold/off-chain site must either have an established explicit completion mechanism or cause this existing conservative chain fallback. The minimum scoped route is the veto; adding events to the separate `emit_cell` path would expand this correction. Whether the current legacy-cell runtime behavior is masked by its real allocation remains unresolved in the preserved evidence; do not call it a reproduced GPU failure. This is a planned generic eligibility obligation under tasks1.4/2.3, not an already validated fix.

Preserve the **refuted member-rowfold control**: `rowfold.rs:rowfold_sites` (36–84) requires an operand whose logical producer is a recognized tile site. A grouped member's logical Map producer does not satisfy that rule, so the preserved member-rowfold case remains a host fold after its already ordered producer-stream whole copy. Do not extend rowfold eligibility to grouped members or veto every source-language Fold. Preserve base-output cell/device-rowfold controls that already select legacy placement through the existing veto. Unrecognized, loop, guard and carry cases do not gain nonzero streams.

## Dead allocation removal comes after ordering

`chain_prologue:2133–2141` currently preallocates every candidate raw target. The actual emission at `emit_site_resident:1992–2002` instead allocates/stores raw output only for an ungrouped launch or `Epi.store_out=true`; otherwise it passes literal0 in that pointer position. Member allocations are separately made at2034–2040 only for `Member.store`.

After tasks1.2–1.4/2.1–2.3 are satisfied, align the prologue's raw allocation condition with that same selected group's `store_out` fact. `gpu.rs:emit_tiled_map_gpu` (2735–2778) and `epi_kernel` (2827–2875) already pass the same group/store facts to host and device emission; no kernel change is required. Keep the ordered list of candidate sites, live raw allocations, stored-member allocation positions, pointer arity/order and zero output arguments unchanged. This is omission, not allocation hoisting.

Do not rewrite `free_after`, slot ownership or teardown merely to obtain lower counts. `dslot` initializes0; `mapal_gpu_free` returns immediately on0 before synchronization. Thus an omitted raw allocation can remove the old nonzero free's context wait, which is why the stored-member dependency proof is a prerequisite. Do not infer safety from raw-pointer deadness alone.

**Keep `if arena && plan.chain.is_empty()` at1278 intact.** Shared writer extraction may replace arena's duplicate member-birth lookup, but must preserve the inclusive interval/coloring and effective min(original producer,group writer) birth. No stream-aware coloring, pooling, lifetime extension, timer move or runtime effect-attribute addition belongs here.

## Smallest coherent regression scope

Use `tests/streams.rs` for new combined behavior, with a separate direct `emit_detailed` helper taking Contract/Exact. Its existing `emit` helper (16–31) deliberately disables epilogue, devmap and arena and **cannot** exercise this bug class; preserve that helper and its existing mechanism tests. Force streams ON and stream-gate OFF explicitly on RTX5090, whose production table now disables streams; enable epilogue and the relevant devmap mechanism. Leave arena enabled and assert its guard when chains actually open. No table/policy change is part of this test setup.

| Regression / control | Meaningful expected evidence after the planned correction |
|---|---|
| Preserved materialized-raw member corner reads | Both faces: actual grouped launch stream selects the member's `d2h_on`; retain late raw reads and no inserted clock/wait-masking fixture changes. Old emitted lookup must fail the new ordering assertion. |
| Preserved forked member→two tile readers | Same-stream child has no redundant event; different-chain child has exactly the required producer→consumer event immediately before its launch. Include repeated operands to retain wait deduplication. Old emission must fail the missing-event assertion. |
| Stored member used only as another group's Gmem capture | Verify the event is found through `Epi.caps`, not only the second tile's ordinary operands. This covers the second feeder branch of the same consumer. |
| Existing member-rowfold and base-rowfold/cell controls | Member remains host fold after ordered whole copy; base controls stay legacy. No new rowfold eligibility or spurious kernel/stream appears. |
| Stored-member legacy-cell/off-chain consumer | Subject to the established ordering premise, the conservative veto sees the grouped writer and leaves chains empty. Preserve the original real-allocation evidence; do not claim it already caused wrong values. |
| Alias/loop/guard/unknown controls | Extend existing emitted assertions as needed: carry views/exit keep canonical slot/swap semantics and legacy placement; guard/loop regions remain unchained; a parameter/host producer gets no fabricated device writer/event. |
| Prologue/storage correction after prerequisites | On grouped independent/diamond/MLP2 cases, unstored raw output remains literal0 and its dead prologue allocation disappears; live raw fanout and stored-member allocation positions remain. Twin and single-chain/MLP controls stay unchanged. Allocation/live-byte accounting must match actual stores; arena still absent whenever chains open. |

Reuse existing control coverage rather than writing tests that merely restate a table: `streams::{independent_device_only_sites_take_two_streams,diamond_waits_on_its_join_edge_only,a_single_chain_keeps_one_stream,a_whole_copy_out_keeps_one_stream,every_off_arm_is_the_one_stream_pageable_text}`; `epilogue::{a_launch_never_passes_two_objects_on_the_same_arena_bytes,mlp2_two_groups_store_only_what_the_next_site_reads,fanout_stores_the_shared_output_once,loop_carries_the_activation_on_the_device}`; resident arena tests and rowfold placement/layout tests. Only extend those files when the new combined tests do not cover a necessary assertion.

Future focused commands (not run here):

```sh
cargo test --release -p mapal-backend-llvm --test streams
cargo test --release -p mapal-backend-llvm --test epilogue --test resident --test rowfold
```

Compile-time regression success is only the first gate. Preserve ordinary untraced before/after GPU value/trap runs, actual binaries/PTX and completion evidence as already required by the plan. Compare the same forced-stream realization for allocation counts, requested storage, iteration and process wall; streams-OFF production policy is not an isolated DCE control. No gain is promised.

## Inputs and boundary

Plan reviewed: `openspec/changes/eliminate-unmaterialized-gpu-allocations/{design,tasks}.md`; evidence authority: `docs/performance/s57-ordering-evidence.md` and its preserved proof links. Their compiler-output findings were reused, not regenerated. All nine code/test files listed below match the frozen source receipt; source and immutable payloads remain untouched.

| File | SHA256 |
|---|---|
| `crates/backends/llvm/src/func/resident.rs` | `139b65d2eece8a3963290fa6166042d0d6e5031ceb0932c009ea18c33e34995d` |
| `crates/backends/llvm/src/func/mod.rs` | `5838c11670aa347e09afcfdeab71a62da3c95e56372aaf29b607e8993dd0645c` |
| `crates/backends/llvm/src/func/core.rs` | `55998dddf7560ae2c24000e6bc85da922fd3f47940779cccb810751b93700f42` |
| `crates/backends/llvm/src/func/gpu.rs` | `77a32c03a6ad02780b534e5e07842b32fbb9ff3ca79ea5480ce7da4f7a53a833` |
| `crates/backends/llvm/src/func/rowfold.rs` | `29804dd073c378a13c249e0b34f8456825986675c44036dd803514da4740da7d` |
| `crates/backends/llvm/tests/streams.rs` | `59c1efbbcc36079a3f9a29cccb5fc990c15001eec801d6fd5732cb1379993cc0` |
| `crates/backends/llvm/tests/epilogue.rs` | `73e89baa79b761c64db72373b085e859b4ab3a11ce9cb907e2419a5628ff1d42` |
| `crates/backends/llvm/tests/resident.rs` | `420ec5509a1fc0e7a8f6eb546c5d3c7b07938bdef1a99b953204044c8a0a3f90` |
| `crates/backends/llvm/tests/rowfold.rs` | `ca937c6b4b0f9d57770e0618c44f88cc9cc699b8dbe4ca8cf6f87e6ba759cb50` |
| `openspec/changes/eliminate-unmaterialized-gpu-allocations/design.md` | `564097d30372f2f1798b70a098f040a7617eaaccaf2c868421896d958ca61f3f` |
| `openspec/changes/eliminate-unmaterialized-gpu-allocations/tasks.md` | `64026e97cfa7ff1e3ff4c8ded18954b050fd8007435a64dfccaf995dc02b7b45` |
| `docs/performance/s57-ordering-evidence.md` | `1ec8ce5bab4187ce25e0ed383b032049cdc50517dbfc10ea73fc68712d1bcc40` |
