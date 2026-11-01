# Host/GPU call-effects audit — saved evidence only

Author: `/root/audit_s54_worktrees`, 2026-10-21. Read-only source/artifact inspection; only this note was written. No new LLVM, builds, test code, runtime changes or GPU actions.

The common graph already carries useful dependence, shape and lifetime facts, and LLVM demonstrably optimizes the surrounding CPU code. The host LLVM boundary exposes GPU work as external runtime calls with almost no effect/alias contract. This is a proven representation limit in the audited path, **not a demonstrated missed optimization or explanation of a timing gap**. CPU worker-task execution on GPU profiles is also explicitly disabled; asynchronous GPU submission and eager initialization remain possible.

## Exact saved module and absence/presence findings

Input is rewritten RTX5090 Contract `recurrent-t16-d8-h16.run`, as bound by `host-optimization-evidence/original-emission-command.json`. Saved host LLVM is optimized separately at clang-O2 before the GPU runtime C and Linux archive are linked; that host compilation cannot inspect their C bodies through these declarations.

| Audited item | Emitted host LLVM | Saved O2 host LLVM |
|---|---|---|
| GPU declarations | Eight bare declarations, lines12–19: init, alloc, h2d, d2h, launch, gen, free, launchv | Same eight, lines15–29; only `local_unnamed_addr` added |
| GPU memory/alias attributes | No function/argument `readonly`, `readnone`, `memory(...)`, `noalias`, `nocapture`/`captures(none)`, `nofree`, `nosync`, `nounwind` or `willreturn` contracts | Still absent on those GPU declarations/calls; some actual pointer arguments gain `nonnull` |
| Scoped alias metadata | No `!alias.scope`, `!noalias` or `!tbaa` anywhere in this host module | Still none |
| Lifetime intrinsics | Zero calls |22 starts at56–77 and22 ends at501–522, for host allocas; the intrinsic declarations/attributes are at526–532 |
| Other function attributes |26 internal pure helpers have `readonly nounwind willreturn`; GPU calls are in unannotated mapal_main (50 calls) and main (init) | Helpers/entry wrapper are inlined into main; the remaining memory/capture attributes belong only to lifetime intrinsics |

A device address crosses this host ABI as `i64`, not a typed address-space pointer. LLVM sees the host pointer to an allocation slot, host copy source/destination, or launch argument array, plus opaque integer handles; it receives no kernel read/write footprint, device-buffer alias relation, launch-to-completion token or LastUsePlan annotation. `local_unnamed_addr` and `nonnull` are not such summaries. With no memory attribute, LLVM uses the conservative read/write effects contract; host stack lifetime markers describe those allocated objects, not completion of device users. [LLVM language reference: memory effects](https://llvm.org/docs/LangRef.html#function-attributes), [memory-use markers](https://llvm.org/docs/LangRef.html#memory-use-markers).

The observed optimization is positive evidence: emitted lines1820–1843 compute q with an eight-element helper loop; O2 lines403–422 calculate it with two four-lane vector sequences and upload32 bytes. Two preceding K/V launch calls remain at398/402. All51 static GPU call sites, including seven tile-launch sites and their ID order, survive. The module alloca count697→22 and definition count28→1 include unused fallback scaffolding; they are not dynamic operation counts, overlap measurements or speedup evidence. Preserved observations: `host-optimization-evidence/observations.json`.

## Hints the backend actually supplies elsewhere

* `func/mod.rs:FnAttrs::analyze` (45) derives trap/effect/termination capability transitively; `func/drive.rs:FnEmit::emit` (110–133) emits clean helper `readonly nounwind` and conditionally `willreturn`. A clean bare by-reference pointer additionally gets `noalias nocapture readonly`. This is a real host optimization contract; the inspected GPU-calling functions do not carry it. This audit does not certify every possible source-function/placement combination.
* `func/gpu.rs:RESIDENT_DECLS` (134), `GPU_*_DECL`, `ARENA_DECL`, `LAUNCHV_DECL` and `S54_DECLS` (64–176) deliberately emit bare external declarations. `lib.rs:emit_detailed` (735–774) includes them according to emitted calls. Runtime C `const` parameters do not retroactively annotate the already compiled host module.
* Device IR is separate. For example `func/gpu.rs` rowfold kernel headers (688–690) and tile headers (1554–1556) use `ptr addrspace(1) noalias readonly` inputs and a noalias output; `epi_params` (1334–1347) marks extra read captures readonly and stored outputs noalias. These aid device optimization, not host-side reasoning about the runtime call.
* A targeted exception exists at `func/resident.rs:emit_site_resident` (2075–2088): when a copied-out result is a device rowfold, its host destination is marked `nocapture` at the d2h call site. This is **not present in the saved recurrent module** and is not a general runtime declaration. Its source comment reports earlier vectorization motivation; no new performance claim is established here.

## Common facts versus their consumers

| Fact/decision | Existing symbol and actual consumer |
|---|---|
| Common target-independent graph rewrites | `mapal-rewrite/src/driver.rs:rewrite` (67): Inline→LiftLoops→ConstFold→Cse→Dce→MapFusion through the shared replayer, with canonical-loop gate and fixpoint. The inspected emission used --rewrite. These are graph transformations before backend placement, not LLVM GPU-call attributes. |
| Semantic last use, escape and loop-carried status | `mapal-ir/src/algo.rs:LastUsePlan` (619), `CategoryIr::last_use_plan` (3199), `dead_after` (696); `func/core.rs:FnEmit::new` (41) stores the plan. `func/ops.rs:update_in_place_source` (573–593) uses it to omit eligible host array copies. |
| Device lifetime and layout | `func/resident.rs:resident_plan` (1060; death/free mapping1250–1270), `resident_after` (1811) generate explicit allocation/copy/free placement. `arena` (1446) colors eligible intervals with true grouped-writer birth positions; its guard at1278 keeps linear coloring disabled when chain streams open. These facts are consumed by the emitter, not encoded as LLVM lifetime intrinsics for Gmem. |
| Generic task dependence, guards/loops and observations | `algo.rs:PathPlan/Task/WaitEntry` (104–167), `CategoryIr::path_plan` (2055): data dependencies2398, clock ordering2434 and observation/completion checkpoints2501–2595. These are execution-plan facts, not proof that an enqueued GPU consumer has completed. |
| GPU eligibility/placement | `func/gpu.rs:gpu_site` (181) gates a float affine tile with nonzero dimensions and no ksplit; `tile_plan` adds eligible strided-lane sites. `func/core.rs:set_gpu_arms` (91–120) combines profile/rungs/retention; `resident.rs:device_sites`, `slot_use`, `uses` (312–405) classify device, host-whole and element readers. Eligibility and profitability are separate decisions. |
| CPU workers versus GPU chains | `lib.rs`621–628 requires `profile.gpu().is_none()` before `emit_parallel`; GPU profiles get a sequential host walk. `lib.rs`690 passes PathPlan to GPU entry emission; `resident.rs:chains` (1304) uses ancestors to construct device chains/events. This is not a joint heterogeneous CPU worker/GPU scheduler. |

## Runtime effects that must remain explicit

`crates/mapal-rt/gpu/mapal_gpu.c` makes blanket purity/argument-only assumptions unjustified without a separate proof:

| Runtime entry | Observable implementation facts relevant to a future effect contract |
|---|---|
| init / alloc / arena (183,249,256) | init writes globals and may launch an initialization pthread; load_module joins or initializes. dev_alloc reads/writes the slot, may return its cached handle, invokes cuMemAlloc, and updates counters/trace. An alloc call is not necessarily a fresh allocation. |
| h2d / d2h (260,276) | Call CUDA copies using host pointers/device handles; update runtime globals/counters and optionally print. d2h is a host-observation boundary, not a readonly host destination. |
| launch / launchv / gen (362,370,529) | Enqueue kernels and update stream/global state. launchv copies its host argument array into local arrays during the call; that host-array use and the later lifetime of the referenced device allocations are different obligations. Trace mode synchronizes at353/542. |
| event / d2h_on / pin (306,388,410) | Event record/wait establishes stream ordering; d2h_on explicitly waits its stream before returning host values. pin stores host addresses until unpin (436), so a universal noncapture claim would be false. |
| free / sync (553,578) | free returns immediately for a zero slot; otherwise may context-synchronize for busy chains, frees and clears the slot. sync joins eager init and waits the context. Removing/moving such calls can change ordering. |

The nonblocking streams are created at161. `launch_on` (315–359) inserts legacy→chain event joins as needed; cross-chain waits are emitted separately. Those runtime edges and the already preserved member-writer ordering investigation remain separate from LLVM alias metadata. Host logical last use, API return and completion of every asynchronous consumer are three different facts. No metadata change can substitute for the last one.

## Smallest warranted evidence experiment, deferred

The present q example already optimizes; it does not justify adding attributes. A bounded first experiment could test the **existing** rowfold d2h noncapture hint: select one saved rowfold/softmax host emission containing that exact call-site hint, preserve it, and remove only that existing hint in a new temporary copy. Compare frozen-clang-O2 optimized LLVM/assembly for the adjacent CPU loop and preserve identical GPU call/ID/order. This tests a concrete cross-boundary host optimization hypothesis without inventing a broader effect contract. First confirm the actual call and loop exist; no fallback to an unrelated example.

Only a demonstrated generated-code difference would justify ordinary runtime value checks and matched iteration/process timing later. Refutation is equal optimized code or a difference unrelated to the target loop. Any later strengthening must first prove exact host byte ranges, capture behavior (including pinning), aliases, global/trace/init effects, errors/observations and stream-completion lifetime; no launch/copy/free/event/sync reordering is presumed legal. No experiment or attribute addition was executed or selected here.

## Binding

| Saved artifact | SHA256 |
|---|---|
| `host-optimization-evidence/host.emitted.ll` | `49d79c826244cb63ac6c1daa088b307b814f79e115bbc7676446d7f5a3a22c7d` |
| `host-optimization-evidence/host.O2.ll` | `25bbaf95aa420672e821ad25d28874552fcd71fb2f627b81d7a4a2f4e091a924` |
| `host-optimization-evidence/observations.json` | `1a687b2284cc60957e669e31279c00b8a0fc616cfbbab3f4cd84b69d2432c5c5` |
| `host-optimization-evidence/manifest.json` | `ab2f9822e07a7dfdf718ced573991d70e02d342213ac5799fa8c13ead38ae3ef` |

All12 repo files read for this audit match the frozen328-source receipt. HEAD/tool/binary mutation was neither requested nor performed. The completed handwritten/upload/ordering payloads were not changed.
