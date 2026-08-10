# Full project review — 2026-08-01

**Status: informational, non-binding, and read-only with respect to implementation.**
This review records defects, contract gaps, release risks, and strengths. It does not
change language semantics or claim that a finding is fixed. Decisions still move
through the project's normal ADR/design process.

Snapshot reviewed: `main` at `3134a5cfdc2efc6aeda188115f50d4fac47a6667`.
The workspace contains 11 Rust packages, 144 Rust files under `crates/`, and roughly
79,000 lines of Rust including tests. The review covered source → syntax → lowering →
IR → checking → interpretation → rewriting → LLVM/CUDA/runtime realization, plus CI,
editors, documentation, packaging, and the planned NVPTX/Verilog/CLI edges.

Evidence labels used below:

- **Reproduced** — a focused probe or the project's own hosted run demonstrated it.
- **Code-confirmed** — both sides of the violated invariant are present in code/docs;
  no target hardware was available for the last step.
- **Admitted** — the repository already documents the limitation.
- **Planned risk** — no shipped implementation exists yet; the plan is incomplete.

---

## 1. Executive verdict

Mapal is a serious research compiler with an unusually strong semantic spine. The
crate boundaries are mostly clean, the IR builder/validator pairing is thoughtful,
the dependency footprint is tiny, and the differential/property/golden-test culture
is substantially better than most projects at this stage.

It is **not release-ready**. The decisive reasons are not missing polish:

1. hosted `main` is red and has been red for nine consecutive pushes;
2. the parser can accept a program while silently deleting part of it;
3. lower accepts invalid struct literals and rejects a valid scoped loop through an
   internal-error path;
4. the reference interpreter's advertised totality is false for lower-reachable and
   resource-exhausting inputs;
5. the runtime has confirmed scheduler and integer-overflow defects;
6. CUDA ownership is incomplete, and target/hardware capability is partly implicit;
7. the user-facing CLI is still a stub.

The right readiness statement is:

| Use | Verdict |
| --- | --- |
| Language/compiler research on controlled inputs | **Usable, with known sharp edges** |
| Performance and geometry experiments | **Strongest part of the project** |
| External preview for users who can tolerate breakage | **Not yet** |
| Production or service processing untrusted source | **No** |
| Packaged multi-backend compiler | **No — CLI/Verilog unfinished; CUDA evidence stale** |

Local green and project green must not be conflated. On the review machine,
`cargo test --workspace --release` passed every runnable test. The current hosted
commit fails on both Linux and macOS.

---

## 2. What is strong

### 2.1 Architecture and semantic discipline

- The pipeline is visibly layered: syntax does not know IR; IR is the low-dependency
  hub; lower owns the seam; the interpreter is the semantic oracle; backends consume
  sealed IR.
- `IrBuilder::seal` plus independent `validate()` is a sound construction/oracle
  pairing. Many graph invariants are enforced structurally instead of by convention.
- Placement-dependent geometry is largely deduced into named records (`PathPlan`,
  `TilePlan`, `LastUsePlan`, `BoundsProof`) rather than rediscovered independently in
  every backend.
- Unsafe code is concentrated in the runtime/FFI seam instead of spread across the
  compiler.

### 2.2 Verification culture

- The local release suite discovered 1,048 tests: 1,047 passed, one performance
  baseline was intentionally ignored, and none failed on this machine.
- LLVM differential tests actually compile and execute generated code; CI sets
  `MAPAL_REQUIRE_CLANG=1` so that leg cannot silently disappear.
- Rewrite has property tests for validation, determinism, idempotence, and oracle
  equivalence. Syntax/lower use golden and rejection tests. CUDA has a large textual
  and structural suite even without hardware.
- The repository contains useful negative controls and records measurement context,
  especially in the performance work.

### 2.3 Operational restraint

- Normal workspace runtime dependencies reduce essentially to `slotmap 1.1.1`; most
  other dependencies are test/benchmark support.
- Formatting is clean. The repository has a license, contribution guide, security
  policy, and code of conduct.
- Known backend limitations are often stated honestly rather than hidden.

These strengths make the defects below tractable: the project already has the
places where the fixes and regression tests belong.

---

## 3. Release blockers

### B1. `main` is host-dependent and hosted CI is red

**Evidence: reproduced in hosted CI.**

`mapal_backend_llvm::emit(&CategoryIr)` delegates to `EmitOpts::default()`
(`crates/backends/llvm/src/lib.rs:164-168`). That default now selects `target:
"native"` (`lib.rs:145-160`), which probes sysctl/sysfs/CPU facts. This contradicts:

- the IR-only public signature;
- the crate-level claim that emission is deterministic (`lib.rs:10`); and
- the field documentation saying `generic` is the default and the host is not probed
  unless requested (`lib.rs:132-137`).

The hidden input is already breaking the gate. The
[current hosted run](https://github.com/LessComplexity/mapal/actions/runs/30640922649)
fails:

- Linux: nine LLVM differential/structural tests whose expected tiling shape is no
  longer stable under the runner's detected profile;
- macOS: `native_reproduces_the_hand_written_move_geometry` assumes the runner is the
  author's M4 Pro and expects five cores per L2, while the hosted machine reports
  three (`crates/backends/llvm/src/profile.rs:1430-1448`).

There are nine consecutive failed push runs from `d52c1368` through the reviewed
tip. The last success was `1877b73c`. `README.md:160-161` and the living status
roll-up still say the suite is green.

**Impact:** the public transform is not a function of its stated input, structural
tests are machine-specific, and caches/artifacts cannot be reasoned about from the
API. CI is functioning—it caught this—but its release criterion is unmet and its
result now depends on runner hardware.

**Invariant-level root fix:** make target/profile Dat explicit and pin structural
tests to named profiles. Restoring `generic` for the existing `emit(&IR)` is the
smallest API-preserving policy, but it reverses the intentional S46 native-default
decision and therefore needs re-ratification. If S46 stands, expose native emission
through an explicitly named/profile-bearing API and include the resolved profile in
the semantic signature and artifact/cache key. Replace the macOS exact-M4 test with
capability/invariant tests, keeping exact M4 geometry only behind an explicit M4
profile test. Update green claims only after a hosted push is actually green.

### B2. Mixed guard blocks can silently lose source code

**Evidence: reproduced.**

`crates/mapal-syntax/src/parser.rs:1915-1942` computes `has_arm`, retains only guard
arms, and returns `StageKind::Guard`; a parsed tail is never carried into that return.
This source parses cleanly while `999` disappears from the AST:

```mapal
true -> { -true-> 1; -false-> 2; 999 } -> ret
```

That contradicts the documented block split in
`docs/components/syntax/DESIGN.md:620-625`, where a tail makes this a statement block
or a diagnostic case.

**Impact:** accepted source can execute differently from what the user wrote. This
is more severe than a crash because all downstream validators see only the already
truncated tree.

**Smallest root fix:** never recover a mixed/tail block by filtering nodes. Emit the
existing mixing diagnostic and retain an error/statement-block node that contains
the complete parsed source. Add the exact probe above as a negative-control test.

---

## 4. High-priority implementation findings

### H1. Struct constructors accept duplicate and unknown fields

**Evidence: reproduced.**

`crates/mapal-lower/src/emit.rs:1535-1554` inserts initializers into a `BTreeMap`, so
a duplicate silently overwrites its predecessor. Lines 1556-1568 iterate only the
declared field list, so unknown initializers are silently ignored. Both of these
lower successfully:

```mapal
P { x: 1, x: 2 }
P { x: 1, y: 2 }
```

This contradicts `docs/components/lower/DESIGN.md:603` and weakens the language's
static contract.

**Root fix:** validate the initializer name set once, before emitting values: reject
the second occurrence and every name absent from the declaration, then report
missing declared fields. Keep source order for diagnostics; use the existing type
declaration as the only authority.

### H2. Loop effect detection ignores lexical scope

**Evidence: reproduced.**

The canonical analysis in `crates/mapal-lower/src/effects.rs:173-185` checks whether
a name is locally bound before treating it as a function call. The separate scan in
`emit.rs:3863-3871` looks only at the text and `fn_sigs`. A local mutable `g` that
shadows an effectful function `g` is therefore classified as an effectful call. The
loop is marked token-carrying at `emit.rs:2650`, then a pure function fails through
the internal `effectful loop without token` path at `emit.rs:2685-2688`.

**Impact:** valid scoped source is rejected as a compiler-internal inconsistency.
Duplicating semantic analysis already produced divergent answers.

**Root fix:** use the existing scope-aware effect result; do not maintain a second
text-only effect scanner.

### H3. A lower-reachable loop shape panics the reference interpreter

**Evidence: code-confirmed and already admitted by component status.**

`crates/mapal-interp/src/loops.rs:31-34` unwraps `loop_plan` with
`unreachable!`. Lower explicitly admits a nested form whose inner exit terminates in
`ret` (`docs/components/lower/DESIGN.md:733-742`). Interpreter status says the
multi-merge shape is lower-reachable and panics (`docs/components/interp/STATUS.md:25-28`).

This directly contradicts the public interpreter contract at
`crates/mapal-interp/src/lib.rs:5-8`: every validate-clean run is documented to
return `Done`, `Diverged`, or `Trapped`, never panic or hang. LLVM/CUDA at least
return a structured `Unsupported` for noncanonical shapes.

**Root fix:** choose one honest boundary. Either lower rejects every shape outside
interpreter M1, or the interpreter gains a non-panicking `Unsupported` outcome. Do
not leave a validate-clean, lower-reachable program in an `unreachable!` class.

### H4. Interpreter fuel does not bound bulk work or allocation

**Evidence: code-confirmed.**

`Iota` and `Fill` allocate their complete result in
`crates/mapal-interp/src/eval.rs:402-426` after paying the single morphism step. A
valid static count can approach `i32::MAX`; budget `1` can therefore request tens of
gigabytes before the interpreter has an opportunity to return `Diverged`.

**Impact:** the “fueled, total, never hangs” oracle can abort the process through OOM
on valid IR. Any future service accepting source would turn this into a denial-of-
service boundary.

**Root fix:** charge/check fuel before bulk construction and make allocation failure
an explicit outcome. At minimum, bound element work by remaining fuel and use a
fallible reserve path.

### H5. Deep acyclic type graphs stack-overflow lower

**Evidence: reproduced.**

`crates/mapal-lower/src/tys.rs:344-360` recursively resolves referenced structs
without a graph-depth budget or memoized result. A parse-clean chain of 10,000
shallow type declarations aborts with stack overflow. A 2,000-node chain also
produced 1,937 duplicate L1209 diagnostics at one span. The post-resolution type
depth check happens too late to protect the traversal.

This contradicts the total lowering promise in `crates/mapal-lower/src/lib.rs:33-34`
and `docs/components/lower/DESIGN.md:938-941`.

**Root fix:** resolve declarations by an explicit graph walk with visiting/resolved
states, memoize each result/diagnostic, and enforce the existing depth limit while
walking rather than after recursion.

### H6. Zero-length runtime splits can schedule pinned work onto a worker

**Evidence: reproduced against the public runtime ABI.**

For a zero-length `Split`, `slice_ranges` returns no ranges. The empty completion
branch at `crates/mapal-rt/src/lib.rs:656-668` unlocks every dependent when its count
reaches zero. The ordinary completion branch correctly adds `!pinned` at lines
730-739. A zero split followed by a pinned dependent is therefore queued to the
pool, where `execute` aborts at line 786 with `pinned task reached a pool worker`.
The focused caller exited 134 and then observed a poisoned mutex.

Current surface array types reject length zero, so ordinary current Mapal source may
not reach this cross-product. The runtime ABI explicitly supports zero splits and
tests them, making this a real library invariant failure and a blocker for dynamic
or zero-length shapes.

**Root fix:** mirror the existing `!pinned` condition in the empty-range completion
path and add the zero-split → pinned-dependent regression.

### H7. `MAPAL_SLICE` alignment overflows and can re-open an OOB class

**Evidence: reproduced in debug and release.**

`MAPAL_SLICE` accepts any positive `i64` (`crates/mapal-rt/src/lib.rs:198-206`). Its
round-up formula performs unchecked `forced + quantum - 1` at line 226. With
`MAPAL_SLICE=9223372036854775807`, debug panics. Release wraps negative, later
collapses the effective quantum to one, and produces ranges not aligned to the
kernel's declared quantum.

Lines 213-222 explicitly say this alignment is a correctness condition: a
mid-panel SME slice previously wrote out of bounds.

**Root fix:** perform a checked unsigned ceiling division/multiplication and reject
an override that cannot be represented. The declared quantum must never be reduced.

### H8. Rewrite equivalence and compiled trap observability disagree

**Evidence: code/contract inconsistency; one related blind spot was previously
reproduced in S38.**

Rewrite property tests deliberately treat any two traps as equivalent
(`crates/mapal-rewrite/tests/property.rs:46-63` and `tests/inline.rs:56-63`), matching
RW2 in `docs/components/rewrite/DESIGN.md:83-84,286`. The runtime, however, exposes
the kind as user-visible stderr (`mapal trap: div_zero` vs `index_oob`) at
`crates/mapal-rt/src/lib.rs:68-80`. LLVM differential expectations discard both
trap kind and stdout for trapping runs (`crates/backends/llvm/tests/differential.rs:208-216`).

S38 already demonstrated the second blind spot: compiled output before a trap was
lost/reordered while the 1,280-run sweep stayed green. The remaining open item says
inlined morphisms still need call-site source positions
(`docs/sessions/2026-07-27-s38-trap-order-is-source-order.md:218-221`). Map fusion
allows loop-free bodies that may trap, so per-element fusion can also change which
independent trap is first while satisfying the current quotient.

**Root decision/fix:** decide whether trap kind/order and pre-trap output are
observables. If they are, make R1 and differential checks exact and restrict/reorder
rewrites accordingly. If all traps truly are one bottom value, remove kind-specific
public diagnostics. The current quotient can prove RW2, but it cannot prove parity
for the user-visible abnormal-run behavior the compiled runtime currently exposes.

### H9. CUDA returned arrays transfer a free duty that callers never adopt

**Evidence: admitted and code-confirmed; no CUDA hardware available here.**

The callee removes or guards escaping device buffers so it does not free returned
storage (`crates/backends/cuda/src/func.rs:521-555`). `emit_call` only assigns the
returned handle (`func.rs:1048-1070`); it never registers the allocation/arena with
the caller. The CUDA DESIGN and STATUS already admit that caller-frees is not
mechanized. If an arena member escapes, the whole arena can remain pinned.

**Impact:** repeated array-returning calls, especially in a loop or long-running
process, grow VRAM until failure. Fresh-process differential tests hide the leak.

**Root fix:** use a caller-owned output-buffer ABI, or return an explicit owner-base
and deleter descriptor and register it at the call site. A bare pointer cannot carry
the ownership transmission the design claims.

### H10. macOS SME detection can enable an SME2 emitter on an SME1 host

**Evidence: code-confirmed, hardware-conditional.**

Native detection enables an `Sme` profile after checking only
`hw.optional.arm.FEAT_SME` (`crates/backends/llvm/src/profile.rs:948-970`). The
emitted leaf requests `+sme,+sme2` (`src/module.rs:248-250`), while the documented M4
profile separately records SME, SME2, and SME_F32F32 capability
(`profile.rs:792-800`).

**Impact:** on a future/other SME1-only Mac, native contracted emission may contain
instructions the machine does not support and fail with `SIGILL`.

**Root fix:** require every feature the emitter names, including SME2 and the needed
floating-point capability, or emit a capability-specific feature set.

---

## 5. Medium findings and quality debt

| Area | Finding | Evidence / minimal correction |
| --- | --- | --- |
| Syntax | Missing commas between type fields are accepted. | `parser.rs:553-565` treats comma as optional; `type P { a: i32 b: i32 }` parses and lowers. Require a comma or `}` after a field. |
| Syntax API | `LineIndex::line_col` promises mid-codepoint rounding but can panic. | `loc.rs:89-107` slices UTF-8 at an arbitrary byte. Round to a boundary before slicing; test an interior byte. |
| Lower | `type i32 { ... }` is accepted despite reserved scalar names. | The reserved-name pass in `lower/src/lib.rs:52-61` visits functions only; type indexing at `tys.rs:43-55` accepts it. Apply one shared reserved-name check to both declaration kinds. |
| Lower | The D1 sink detector is an unconditional `false` stub. | `typing.rs:1152-1169`; multiple bad non-Unit functions ending in `print` produce only the first fail-fast emission error. Implement the already-designed predicate or delete the pre-pass claim. |
| Check | T0201 suggests wrapping effects in `seq`, but fanout context remains sticky through `seq`. | `check/src/effects.rs:177-184,205-212`; tests confirm the advice still fails. Change the diagnostic, not the semantic rule. |
| Interp API | `run()` seeds every non-token entry with `Unit`, although `seal` allows any Named entry type. | `interp/src/lib.rs:35-53`; `ir/src/builder.rs:247-280`. Constrain entry types or return `Unsupported` and direct typed callers to `eval_call`. |
| IR API | Bare slotmap IDs collide across builders. | Admitted at `docs/components/ir/DESIGN.md:472`. One lower builder makes it safe today; add a builder nonce only when a second construction client becomes real. |
| CUDA | Nested-array/arena byte arithmetic is unchecked. | `cuda/src/kernel.rs:655-710,845-847`; `arena.rs:179-194`. Large dimensions can wrap to under-allocation. Use checked arithmetic and return `Unsupported`. |
| Editors | The VS Code grammar is not associated with the registered Mapal language. | `editors/vscode/package.json:17` registers `mapal`; line 34 contributes the grammar for `flow`. Grammar-unit tests bypass the manifest. The [VS Code syntax guide](https://code.visualstudio.com/api/language-extensions/syntax-highlight-guide) defines this field as the registered language id. Change it to `mapal` and add a manifest assertion. |
| Examples | LLVM and CUDA both build an example named `emit`. | Cargo warns both write `target/release/examples/emit`; the last writer determines whether LLVM flags work. This breaks the exact benchmark command in `docs/next-session.md:76-80`. Give the examples unique names. |
| CLI | The CLI is an honest stub, but even its binary name is stale. | `crates/mapal-cli/Cargo.toml:6-8` builds `flow`; `main.rs` says `mapal` and exits 1. Rename now; build commands only when the CLI milestone starts. |
| Rustdoc | Strict workspace docs fail. | Private/unresolved intra-doc links in `mapal-rt` and `mapal-ir`. Fix the links and put `RUSTDOCFLAGS=-D warnings` in CI once green. |
| Clippy | `-D warnings` fails before reaching a clean baseline. | Four immediate `mapal-ir/src/algo.rs` warnings; ordinary clippy reports roughly 17 workspace warnings. Clear existing warnings, then gate new ones. |
| CI coverage | CI omits editor integration, strict clippy/rustdoc, dependency audit, and CUDA execution. | `.github/workflows/ci.yml`; CUDA is compile-only on hosted Linux. Add gates only after the current release suite is deterministic. |
| CI supply chain | Actions use movable tags. | `actions/checkout@v4`, `dtolnay/rust-toolchain@stable`, `Swatinem/rust-cache@v2`. Pin third-party actions by commit when release hardening starts. |
| Maintainability | `EmitOpts::kc_nest` documents that it lost on every measured machine and says to delete it, but retains a large implementation/test surface. | `llvm/src/lib.rs:71-110`. Delete the refuted NEON lever; keep a separately justified SME experiment only if it is still active. |

No automated dependency advisory verdict was possible because `cargo-audit` is not
installed. A quick check of the official [RustSec database](https://rustsec.org/advisories/)
did not surface a `slotmap 1.1.1` advisory, but that is not a substitute for an
automated locked-dependency scan.

---

## 6. Component-by-component assessment

| Component | Assessment |
| --- | --- |
| `mapal-syntax` | Good recovery-oriented parser and broad test base, but silent guard-tail deletion is a release blocker. Delimiter and UTF-8 boundary tests need adversarial cases. |
| `mapal-ir` | Strong central model, builder invariants, and independent algorithms. Public cross-builder IDs are intentionally unsafe; `algo.rs` is large and currently blocks strict clippy. Consumer coverage is useful but lexical, not semantic. |
| `mapal-lower` | Most correctness risk is concentrated here. It contains duplicate semantic scans and a very large emitter; scope drift, struct validation, and recursive type resolution show the cost. Reuse existing analyses rather than add a third pass. |
| `mapal-check` | Small, clear, and 30 tests pass. One diagnostic recommends an invalid repair; effect legality itself appears consistent. |
| `mapal-interp` | A valuable executable oracle with clear trap/fuel types, but its totality claim is stronger than the implementation. Unsupported loops and allocation exhaustion need values, not panics/abort. |
| `mapal-rewrite` | Strong property harness and rebuild-through-builder discipline. The chosen trap quotient is now inconsistent with runtime diagnostics and differential observation, so “oracle equal” is weaker than users will infer. |
| LLVM backend | Deepest and best-tested realization. Native-by-default made target facts implicit and broke CI. Exact-machine tests and SME capability checks need separation from portable invariants. |
| `mapal-rt` | Compact std-only runtime with localized unsafe FFI, but scheduler state transitions have already diverged between branches. The zero-range and slice-overflow repros warrant focused invariant tests. |
| CUDA backend | Large structural suite and candid design docs. Hardware was unavailable; returned-array ownership and large-size arithmetic remain incomplete. The last recorded RTX 4090 execution is July 2026 and predates later work. |
| Verilog backend | Honest stub/planned component; no implementation claim should be made. Local Verilator/iverilog availability makes the status note stale, not the backend built. |
| CLI | Not a product surface yet. Stub exits 1, and Cargo installs `flow`, not `mapal`. |
| Editors | Grammar rule tests pass; VS Code packaging is wired to the old language id. |
| Documentation | Exceptional breadth and design history, but living roll-ups now contain stale paths, counts, syntax, architecture, and green claims. History is starting to obscure current truth. |

### 6.1 Security and reliability boundary

Mapal is currently a local compiler/research tool, not a network service, so there is
no authentication, tenant isolation, or persistent-data surface to assess. Its real
trust boundaries are source text, environment overrides, generated native/device
code, FFI pointers, and target capability discovery.

- Untrusted source is not safe today: acyclic type graphs can exhaust the process
  stack, and valid bulk IR can exhaust memory outside the interpreter's fuel model.
- `MAPAL_SLICE` is an environment input that can trigger a debug panic or violate a
  runtime alignment invariant in release. Treat performance overrides as validated
  inputs, not trusted developer constants.
- Unsafe/FFI code is localized, which is good, but the zero-split abort and CUDA size
  arithmetic show that safe scheduling/allocation invariants need tests at the ABI
  boundary.
- The normal dependency surface is small. The remaining supply-chain work is an
  automated advisory scan and immutable CI action pins, not a dependency rewrite.

Do not expose compilation as a service until source-size/depth/allocation limits and
non-panicking unsupported/error paths are explicit and tested.

---

## 7. FRAMEWORK grounding: Dat / Trn / Loc / Trm

### 7.1 Current whole-system objects

| Atom | Real objects in this repository |
| --- | --- |
| **Dat** | source bytes, tokens/AST, diagnostics, resolved types, `CategoryIr`, rewrite plans, target profiles, emitted LLVM/CUDA text, runtime frames/tasks/ranges, device buffers, interpreter `Outcome` and output |
| **Trn** | lex, parse, lower, seal, validate, check, rewrite, interpret, emit, compile, schedule, execute, allocate/free, H2D/D2H, render diagnostics |
| **Loc** | compiler process, external clang/native process, runtime host spine, pool workers, CUDA host, CUDA device; FPGA/NVPTX placements remain planned |
| **Trm** | in-process pass handoffs, host profile probing, spawned-process stdin/stdout/exit, task dependency queues and frame sharing, CUDA allocation handles/copies/trap readback |

The current `docs/architecture-map.md:53-67` still describes the compiler as Loc-
collapsed until the backend seam and says CUDA was the first genuine two-site
execution. S24 added a real worker pool, host-pinned spine, work frames, dependency
queues, waits, and trap transmissions. Those placements are absent from the living
system map. The same file says all coherence laws passed “as of S15” at lines 96-98,
well before the implementation now under review.

### 7.2 Coherence-law audit (§4.5)

| Law | Review result | Grounded reason |
| --- | --- | --- |
| 1. Placement honesty | **Fail** | `emit(&IR)` reads undeclared host-profile Dat; zero-split completion places pinned work on a pool worker instead of its declared host spine. |
| 2. Transmission well-typing | **Fail** | CUDA transmits an array handle without the ownership/free-duty Dat required at both ends; the planned NVPTX copies likewise lack materialized buffer endpoints. |
| 3. Placement totality | **Not auditable** | The living map has no `TrnLoc`/`DataLoc`/`TrmCmp` rows for host spine, pool workers, profile probing, or the planned NVPTX lifecycle, so their projections cannot be checked. |
| 4. Dependency mediation | **Not justified** | Worker ↔ host dependency/wait boundaries are absent from the map; the zero-range bug demonstrates that their dispatch mediation is load-bearing. |
| 5. Composition soundness | **Fail** | The global roll-ups claim to be deduced from component state but retain stale paths/counts and say green while hosted gates are red. Parser data loss and the trap quotient are additional behavioral-composition defects. |
| 6. `runsAt` is a relation | **Not demonstrated by the living map** | Code has genuine multi-placement execution, but the architecture map has not been reconciled with host-spine/pool-worker relations or native-profile probing. |

The smallest architectural repair is not a new abstraction. It is to make existing
hidden Dat explicit (`TargetProfile`, ownership), route sibling state transitions
through the same invariant, and stop claiming totality where the implementation has
an `unreachable!`/abort class.

---

## 8. Planned NVPTX review — risks, not shipped defects

No NVPTX crate exists, so these findings must not be counted as current runtime
bugs. They are blockers to resolve before implementation:

1. `crates/mapal-ir/tests/consumer_coverage.rs:44,56-139` registers whole backend
   directories and accepts a lexical `query(` occurrence. Existing LLVM CPU calls
   can permanently keep the gate green even if a future LLVM-NVPTX realization
   consumes none of the required geometry records. Unused helper code can also
   satisfy it, and the query list omits the newer move geometry query.
2. `docs/components/backend-nvptx/plans/plan-s41-the-nvptx-leg.md` declares Host ↔
   Gmem copies and launch as total morphisms, but its build sequence names module
   load and kernel launch without a complete device/context, allocation, H2D/D2H,
   synchronization, ownership, error, free, and teardown lifecycle. Host frame
   pointers cannot stand in for device array objects.

Before coding, make the consumer gate realization-scoped and exercised rather than
source-grep based. Ground device-buffer Dat and every lifecycle Trn/Trm in the plan.
Do not scaffold a backend around an incomplete ownership model.

---

## 9. Verification performed

| Check | Result |
| --- | --- |
| `cargo test --workspace --release` | **Pass locally:** 1,047 passed, 1 intentionally ignored, 0 failed. LLVM differential tests ran with clang. |
| Hosted CI at reviewed tip | **Fail:** Linux and macOS release jobs; CUDA compile job passed. Nine consecutive push failures. |
| `cargo fmt --all --check` | Pass. |
| `cargo clippy --workspace --all-targets -- -D warnings` | Fail: four immediate `mapal-ir` warnings. |
| Ordinary workspace clippy | Exit 0 with roughly 17 warnings. |
| `RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps` | Fail: private/unresolved intra-doc links in `mapal-rt` and `mapal-ir`. |
| `sh editors/test.sh` | Pass: Neovim and grammar rule suites. Does not validate VS Code manifest association. |
| `mapal-lower` + `mapal-check` targeted suites | Pass: 161 + 30 tests. Focused probes still reproduced parser/lower defects. |
| `mapal-rt --release` | Pass: 21 existing tests. External zero-split/pinned probe aborted as described. |
| LLVM golden suite | Pass: 32. |
| CUDA suite | Pass locally: 163 structural/text tests; `nvcc` absent, so no hardware execution. |
| IR consumer coverage | Pass: 4; review found the gate structurally too weak for a second LLVM realization. |
| Dependency tooling | `cargo-audit`, `cargo-deny`, `cargo-machete`, and `cargo-geiger` unavailable. Normal dependency tree is very small. |

Passing tests alongside these repros is not paradoxical; it identifies the exact
blind spots: recovery-path data retention, declaration-set validation, adversarial
depth/allocation, cross-product scheduler states, manifest wiring, ambient target
facts, and abnormal-run observables.

---

## 10. Documentation and handoff integrity

The documentation is a strength, but “living” documents need a smaller truth
surface:

- `README.md:160-161` says tests/CI are green; hosted `main` is red.
- `docs/IMPLEMENTATION.md:18` cites the old LLVM `src/func.rs` layout even though it
  is now a directory; line 58 says 673 tests; lines 60-63 say no known divergences.
- `docs/architecture-map.md:53-67,96-117` predates the parallel runtime and still
  certifies the S15 coherence audit.
- lower STATUS/IMPLEMENTATION describe obsolete call-form `iota`/`fill`; syntax and
  checker example counts disagree; an AST comment retains pre-ADR loop syntax.
- the global `docs/STATUS.md` has accumulated historical narrative and duplicated
  counts instead of remaining a short, deduced roll-up.
- `docs/next-session.md:76-80` instructs the next operator to build the whole
  workspace and then run the ambiguous `target/release/examples/emit`, which may be
  the CUDA binary and reject the LLVM flags.

Do not perform a mass rewrite before the defects are fixed. First establish one
machine-readable source for test/CI state and explicit target selection; then trim
living docs to links and current facts. Historical session logs should remain
immutable.

---

## 11. Recommended repair order

### Phase 0 — restore a truthful gate

1. Re-ratify S46: either restore deterministic `generic` for `emit(&IR)`, or make the
   native profile an explicit API input/artifact key rather than ambient state.
2. Pin structural tests to profiles and make platform tests capability-based.
3. Push until Linux and macOS are green; only then update README/STATUS claims.

### Phase 1 — eliminate silent source/semantic corruption

1. Preserve/reject mixed guard tails instead of filtering them.
2. Reject duplicate/unknown struct fields.
3. Require type-field commas.
4. Replace the scope-blind loop effect scan with the existing scoped analysis.

### Phase 2 — make advertised totality real

1. Reject unsupported loop shapes at lower or return `Unsupported` from interp.
2. Bound/fallibly allocate `Iota`/`Fill` by fuel.
3. Make type resolution memoized and non-recursive/depth-bounded.
4. Fix `LineIndex` byte-boundary handling and constrain `run()` entry types.

### Phase 3 — repair runtime/backend invariants

1. Add the pinned check to zero-range completion.
2. Make slice rounding checked and quantum-preserving.
3. Require complete SME capabilities.
4. Mechanize CUDA returned-buffer ownership and use checked size arithmetic.

### Phase 4 — settle abnormal-run semantics

1. Decide whether trap kind/order and pre-trap output are observable.
2. Align interpreter, runtime, rewrite R1, and differential tests with that decision.
3. Stamp inlined morphisms with call-site position if source order remains semantic.

### Phase 5 — product and maintenance surface

1. Fix the VS Code language id and unique example names.
2. Rename the stub binary to `mapal`; do not build a CLI framework until M5 starts.
3. Delete the measured-losing `kc_nest` lever and stale inline benchmark prose.
4. Clear rustdoc/clippy debt, then add the gates.
5. Reconcile living architecture/status docs from code and hosted state.
6. Repair the NVPTX plan before creating its crate.

This sequence deliberately puts correctness and truthful evidence before feature
work. It does not require a new framework or broad refactor: most root fixes are in
the shared function that already owns the invariant, followed by one focused
regression.

---

## 12. Review limits

- No source implementation was changed, so every defect remains open.
- No NVIDIA compiler/GPU was available; CUDA execution, leak growth, and current
  RTX 4090 behavior were not revalidated.
- SME1-only hardware was unavailable; that finding is capability-logic inspection.
- The full local test suite is machine-local evidence, not a substitute for the red
  hosted matrix.
- This was a broad static/dynamic review, not a formal proof, exhaustive fuzzer run,
  side-channel audit, or benchmark campaign.
- The user's pre-existing `docs/STATUS.md` modification and untracked `oainotes.md`
  were preserved and not used as review scratch space.
