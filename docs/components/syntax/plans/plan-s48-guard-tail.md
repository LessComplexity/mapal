# Plan — S48: mixed guard tails must not silently delete source (review P0-2)

Date: 2026-08-02 · Status: active · Finding: review B2
(`docs/notes/2026-08-01-full-project-review.md` §3) — **release blocker**

## The model (FRAMEWORK §2/§4)

| Atom | Here | Realization |
| --- | --- | --- |
| `Dat` | parsed block content — `items: Vec<BlockItem>` plus the block `tail` | `crates/mapal-syntax/src/parser.rs::parse_block_body` |
| `Trn` | `parse_classified_block : tokens → StageKind` — must be **total over its input and data-preserving on accept** | `crates/mapal-syntax/src/parser.rs:1899` |
| `Trn` | downstream consumers (lower) — receive the complete AST or a loud error, never a filtered one | `mapal-lower` unchanged |

Coherence repair (§4.5 law 5, composition soundness): parse ∘ (consume source)
must preserve information — accepted source may not execute differently from
what was written. Today's `has_arm` branch (parser.rs:1931-1942) filters to arms
and drops statements **and the tail**: `true -> { -true-> 1; -false-> 2; 999 }
-> ret` parses cleanly with `999` gone, and because the tail is not an item,
`check_arm_mixing` never even emits P0006.

## The defect, precisely

`parse_classified_block` computes `all_arms` (requires `tail.is_none()`) and
`has_arm`. A block of arms **with a tail** falls into the mixed branch, which
returns `StageKind::Guard(arms)` — the tail is never inspected, and no
diagnostic names it (P0006 fires only for arm/statement *item* mixing).

## Fix (smallest root repair, per review)

Never recover a mixed/tail block by filtering. In the `has_arm` branch:

1. Ensure the mixing diagnostic names the loss: emit P0006 when items mix arms
   and statements (already done by `check_arm_mixing`), and emit it for the
   arms-plus-tail case too, pointing at the tail span.
2. Return a node that **retains the complete parsed source**: the full
   `StageKind::StmtBlock` (items + tail) rather than `StageKind::Guard(filtered)`
   — lower then sees every statement and either handles or loudly rejects arms
   in a statement block. Verify at implementation time which `StageKind` payload
   retains items+tail without inventing a new variant; an `Error` node alone is
   *not* sufficient unless the complete source is retained in it.
3. Add the review's probe as a negative-control test:
   `true -> { -true-> 1; -false-> 2; 999 } -> ret` must (a) carry a diagnostic
   and (b) retain `999` in the AST — no path may produce a clean two-arm guard.

## Gates

1. `cargo test -p mapal-syntax --release` (parser suites incl. the new control).
2. `cargo test --workspace --release` — lower/check/golden consumers must stay
   green (they see complete ASTs now; any behavior change must be a loud
   rejection, and the suite will say if one exists).
3. Emit-sweep diff vs `benches/results-s48/emit-baseline.hashes` — expected
   empty (no example uses a mixed guard block; a diff here means the fix
   changed accepted programs, which must then be justified row by row).
4. numpy A/B N=4096 stays in the baseline band (39–40 ms median).

## Done when

The probe cannot lose `999`: either a diagnostic plus a complete AST, or a loud
rejection — and every pre-existing suite stays green. Doc sweep in the same
change: `docs/components/syntax/DESIGN.md:620-625` (the documented block split)
and syntax IMPLEMENTATION/STATUS rows for `parse_classified_block`.
