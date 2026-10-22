# ADR-0038: Native default stands; the resolved profile is explicit emission input

Date: 2026-08-02 · Status: **accepted** (S46 was Sapir's decision; this ADR records the
review-mandated explicitness repair — Sapir may demote to candidate if he wants the
generic default restored instead). Related: ADR-0032 (D4 performance tailors),
plan-s31-target-profiles, review B1 (`docs/notes/2026-08-01-full-project-review.md`).

## Context

S46 made `EmitOpts::default().target = "native"`: the default build detects the
machine, and every published benchmark number since is native-emitted. The
2026-08-01 full-project review (B1) showed the cost of doing that implicitly:

- `emit(&CategoryIr)` reads host Dat (sysctl/sysfs/CPU facts) that its signature
  does not declare — placement-honesty fail (FRAMEWORK §4.5 law 1).
- The crate doc still claimed "emission is deterministic" and the `target` field
  doc still said `generic` is the default — both stale.
- Hosted CI went red for nine consecutive pushes: nine Linux differential tests
  assert tile-shape constants derived from `generic`'s NEON geometry while
  running under the runner's *detected* profile, and one macOS test asserts the
  hosted runner is the author's M4 Pro (`cpusperl2` 5 vs 3).

The review offered two repairs: restore deterministic `generic` for `emit(&IR)`
(reversing S46 — needs re-ratification), or keep S46 and make the profile an
explicit, named, artifact-bearing input.

## Decision

**D1 — S46 stands.** The default target stays `native`; detection still falls
back to `generic` when a fact cannot be read. The shipped performance path is
the default build, and the benchmark record (`benches/results/cpu/matmul/timing-boundary…`) is
native-emitted — restoring `generic` would silently de-perf every default build
and invalidate the record's comparability. Keeping S46 ratifies the status quo;
it is the *reversal* that would need a new decision.

**D2 — The artifact carries its profile.** Emission gains

```rust
pub struct Emission { pub text: String, pub profile: &'static str }
pub fn emit_detailed(ir: &CategoryIr, opts: &EmitOpts) -> Result<Emission, EmitError>
```

`emit` / `emit_with_opts` keep their signatures and delegate (`.text`).
`profile` is the *resolved* profile's name — `"native"` when detection
succeeded, `"generic"` on fallback or by name, or the named profile. Artifact
identity is therefore `(source, opts, resolved profile name)`; no in-repo cache
exists to key, and the emit sweep records hashes per explicit `--target=` face.

**D3 — The determinism claim is restated honestly.** Emission is a function of
`(IR, EmitOpts, resolved TargetProfile)` — deterministic per profile, host-
dependent only through the explicitly named `native`. Crate and field docs say
this; the stale "`generic` is the default" text is corrected.

**D4 — Test policy: structure pins a name, values may float.** Any assertion on
emitted *shape* (tile constants, alloca sizes, lane bounds) emits with an
explicit named profile (`generic` unless testing another profile by name) —
never the ambient default. Value-parity legs may run on the default (native)
emission, and one dedicated differential test asserts native emission is
value-correct on whatever host runs it. Exact-machine geometry is tested as
machine-independent invariants everywhere, plus part-gated exact checks that
fire only on the documented part.

**D5 — Green claims follow the hosted gate.** README/STATUS may claim green
only after a hosted push passes Linux and macOS with these repairs.

## Alternatives rejected

- **Restore `generic` as the default.** De-perfs every default build on SME /
  cache-aware rungs, invalidates the published record's comparability, and — as
  the review itself notes — reverses an intentional Sapir decision. That path
  requires its own ADR, not silence.
- **Leave structural tests on the ambient default.** Nine consecutive red
  pushes: the assertion constants are `generic`-derived, so any non-NEON runner
  fails them. Machine-flaky gates are not gates.
- **Branch tests on detected facts** (skip when detection differs). The banned
  silent-pass class (S43): a gate that cannot fail is not a gate. Invariants
  run everywhere; exactness is gated on the *part*, loudly.

## What would falsify it

- A host where native emission is value-*divergent* from the interp oracle —
  that is a detection or emission bug, and D1 falls with it.
- A second class of machine-dependent test appearing after D4 — evidence the
  policy is unenforceable and the default must be pinned instead.
- Hosted CI red again on profile grounds after this repair.
