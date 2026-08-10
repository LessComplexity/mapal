# Session — full project review — 2026-08-01

## 0. Continuation brief

User invoked `$category-architect start` and requested a full code-to-implementation
review. The session restored the latest immutable handoff, treated the review as a
new bounded workstream, and did not resume or modify the prior matmul optimization
work.

Repository state at start and handoff:

- branch: `main`
- reviewed commit: `3134a5cfdc2efc6aeda188115f50d4fac47a6667`
- user-owned pre-existing state: modified `docs/STATUS.md`, untracked `oainotes.md`
- no review worktree, server, benchmark, or background process was started

Primary artifact:

- [`../notes/2026-08-01-full-project-review.md`](../notes/2026-08-01-full-project-review.md)

## 1. Work completed

- Restored `docs/next-session.md` and the latest immutable S44–S47 session log.
- Inventoried the Cargo workspace, package graph, source size, dependencies, CI,
  editors, examples, component maps, and living status documents.
- Audited syntax/check/lower, IR/interpreter/rewrite, LLVM/runtime/CUDA, CLI,
  Verilog, and the planned NVPTX realization.
- Ran local release tests, formatting, clippy, strict rustdoc, editor integration,
  targeted package suites, dependency-tree checks, and focused reproducers.
- Inspected hosted Actions state for the reviewed commit and recent main history.
- Mapped findings onto FRAMEWORK Dat/Trn/Loc/Trm and §4.5 coherence laws.
- Wrote the full ranked review. No implementation fix was attempted.

## 2. Decisions

1. **Review truth is separate from fix truth.** A passing local suite does not close
   a finding reproduced by a focused probe, and documentation was not changed to
   imply repairs.
2. **Hosted `main` is the current release gate.** Its nine consecutive failures are
   P0 even though this machine passes the release suite.
3. **Restore explicit target Dat before adding machinery.** `generic` default plus
   explicit native emission is the smallest API-preserving repair, but it reverses
   the intentional S46 policy and needs re-ratification. Keeping S46 requires an
   explicitly profile-bearing API/artifact key, not ambient state.
4. **Silent source deletion outranks crashes.** The mixed guard-tail parser recovery
   is a release blocker because every later pass receives a clean but incomplete AST.
5. **Known/admitted/planned issues stay distinct.** CUDA ownership is shipped and
   admitted; SME2 is conditional; NVPTX lifecycle gaps are plan blockers, not current
   runtime defects.
6. **Do not overwrite the standing optimization handoff.** `docs/next-session.md`
   remains unchanged. The repair order lives in the review until the user chooses a
   remediation workstream.

No ADR was created; the review proposes decisions but does not ratify semantics.

## 3. Checks and evidence

| Check | Result |
| --- | --- |
| `cargo test --workspace --release` | 1,047 pass, 1 ignored, 0 fail locally |
| current hosted Actions run | Linux + macOS fail; CUDA compile succeeds |
| recent main history | nine consecutive failed pushes; last success `1877b73c` |
| `cargo fmt --all --check` | pass |
| strict workspace clippy | fail on four immediate IR warnings |
| ordinary workspace clippy | exit 0 with warnings |
| strict workspace rustdoc | fail on unresolved/private links |
| editor test script | pass; does not cover VS Code manifest association |
| lower/check targeted tests | 161 + 30 pass |
| runtime release tests | 21 pass; focused zero-split/pinned caller aborts |
| CUDA local suite | 163 pass; no `nvcc`, no hardware execution |
| LLVM golden suite | 32 pass |
| dependency scanners | not installed; normal dependency graph is tiny |

Focused reproducers confirmed:

- parser silently drops a tail from a mixed guard block;
- parser accepts missing type-field commas;
- lower accepts duplicate/unknown struct fields;
- scope-blind loop effect classification reaches internal L1901;
- 10,000 acyclic type declarations stack-overflow lower;
- zero-length split plus pinned dependent aborts the runtime;
- maximal `MAPAL_SLICE` panics in debug and loses quantum in release;
- workspace `emit` example collision selects the wrong backend binary.

## 4. Live handoff state

- No code changes are in flight.
- No generated test artifact remains in the repository.
- Existing user changes remain untouched.
- Hosted run for the reviewed tip:
  <https://github.com/LessComplexity/mapal/actions/runs/30640922649>
- Review-created files are this immutable log and the linked note only.

## 5. Open items, prioritized

### P0

1. Make LLVM target/profile Dat explicit, pin structural tests, re-ratify S46's
   default policy, and make hosted main green on Linux and macOS.
2. Fix mixed guard-tail parsing so accepted ASTs retain all source data.

### P1

1. Reject duplicate/unknown struct fields.
2. Reuse scope-aware effect analysis in loop emission.
3. Remove interpreter panic for lower-reachable noncanonical loops.
4. Bound bulk interpreter allocation by fuel/fallible allocation.
5. Make type resolution memoized and depth-safe.
6. Fix zero-split pinned scheduling and slice-rounding overflow.
7. Decide and align trap observability across interp/rewrite/runtime/differential.
8. Mechanize CUDA returned-array ownership; gate SME on complete capabilities.

### P2

1. Fix commas, UTF-8 line indexing, reserved type names, sink diagnostics, and T0201
   repair text.
2. Fix VS Code language association and unique example names.
3. Rename the CLI binary, clear strict clippy/rustdoc debt, and reconcile living
   architecture/status docs after the gate is green.
4. Repair NVPTX consumer/lifecycle design before implementation.

Done conditions and minimal root fixes are specified in the review's §11.

## 6. Architecture/model changes

None ratified. The review found that the living architecture map no longer models:

- runtime host-spine vs pool-worker Locs;
- task/dependency/frame transmissions;
- host profile probing as input to LLVM emission;
- CUDA returned-buffer ownership transmission.

Coherence audit result: Laws 1, 2, and 5 fail on confirmed examples; Laws 3 and 4
cannot currently be justified from the living placement/transmission map; Law 6 is
the intended model but is not demonstrated by that map.

## 7. Documentation reconciled

- Added the full review note.
- Added this immutable session record.
- Did **not** edit README, STATUS, architecture maps, component docs, or
  `docs/next-session.md`, because the review did not fix the underlying code and the
  user's existing STATUS edit must be preserved.

## 8. Files changed by this session

- `docs/notes/2026-08-01-full-project-review.md` — new
- `docs/sessions/2026-08-01-full-project-review.md` — new
