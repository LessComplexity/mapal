# Plan — S48: restore a truthful gate (review Phase 0)

Date: 2026-08-02 · Status: active · Decision record: ADR-0038 · Finding: review B1
(`docs/notes/2026-08-01-full-project-review.md` §3, §11 Phase 0)

## The model (FRAMEWORK §2/§4)

| Atom | Here | Realization |
| --- | --- | --- |
| `Dat` | `TargetProfile` — an *input* to emission, today read ambiently from the host | `crates/backends/llvm/src/profile.rs::TargetProfile`, `resolve(name)` |
| `Dat` | `Emission` — the artifact, today bare text with its provenance dropped | new `Emission { text, profile }` in `crates/backends/llvm/src/lib.rs` |
| `Trn` | `emit : CategoryIr × EmitOpts × Profile → Emission` — the profile was a hidden fourth argument | `emit_detailed`; `emit`/`emit_with_opts` delegate |
| `Trm` | host profile probing (sysctl/sysfs) — a transmission from the OS site into the compiler | unchanged mechanism (`profile::native`); now surfaced in the artifact |

Coherence repair (§4.5): law 1 (placement honesty) — the host Dat emission reads
becomes part of the declared input/output; law 5 (composition soundness) — the
gate's claims become deducible from pinned inputs instead of the runner's silicon.

## Baseline (this session, before any change)

- `cargo test --workspace --release`: green (matches review's 1047).
- Emit sweep: `benches/results-s48/emit-baseline.hashes` — 174 cells, 171
  hashed + 3 known `examples/vector.mapal` EMIT-FAILED markers.
- numpy A/B N=4096 (perflock): **mapal 39.506 ms median (3478.9 GF/s) vs numpy
  44.230 ms (3107.4 GF/s), ratio 1.120×, DISJOINT.**

## Steps

1. **ADR-0038** — record that S46 stands with the explicitness repair. (done)
2. **`Emission` + `emit_detailed`** in `lib.rs`; `emit`/`emit_with_opts`
   delegate. Restate the crate-doc determinism claim and the `target` field doc
   (both stale). No emission-logic change: sweep hashes must not move.
3. **Pin the nine hosted-failing differential tests** (`tests/differential.rs`):
   every shape assertion (`alloca [64 x float]`, `udiv .., 4`, `icmp uge .., 16`,
   `[16 x float]`, `[32 x double]`, pack-frame structure) emits with
   `target: "generic"` by name. In `differential_zen3_profile_is_value_invariant`,
   compare explicit-`generic` vs `zen3` (machine-independent), keeping the
   value-parity legs. Add `differential_native_profile_is_value_correct` —
   default (native) emission vs interp oracle, values only.
4. **Audit the other LLVM test files** (`move_panel.rs`, `sme_rung.rs`,
   `tile_sites_pin.rs`, `golden_ll.rs`) for ambient-default shape assertions;
   pin any found the same way.
5. **Split `native_reproduces_the_hand_written_move_geometry`** (`profile.rs`)
   into invariants that hold on any aarch64-mac (detection succeeds, `l1d`
   present, `l2_cores ≥ 1`, `move_block` agrees with a same-facts profile) and
   exact M4-Pro equality gated on the part actually being the documented one.
6. **Gates, in order:** `cargo test -p mapal-backend-llvm --release`;
   `cargo test --workspace --release`; fmt/clippy on touched files; emit-sweep
   diff vs baseline (must be empty modulo intended rows); numpy A/B N=4096
   (must stay disjoint-faster, ≈39–40 ms band).

## Done when

- Local workspace release suite green; the nine tests pass with `generic` pinned
  on any machine (they already cannot see the host).
- Emit-sweep diff vs baseline is empty (no emission-logic change happened).
- numpy A/B remains in the baseline band.
- Pushed and **hosted Linux + macOS green** (push is a separate confirmation);
  only then do README/STATUS green claims get repaired (review §10).

## Explicitly out of scope

- `examples/vector.mapal` parse failure (P1, syntax queue).
- `--target=native` fallback semantics, SME capability gating (review H10 — Phase 3).
- README/STATUS edits before the hosted gate is actually green.
