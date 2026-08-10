# Runtime — status

Reconciled 2026-08-10 (S50). Code: `crates/mapal-rt/src/lib.rs`. Gate: 21 tests in-file,
plus every workspace suite that runs a compiled program.

## Built

- [x] Task DAG execution: register → seal → launch → wait → finish
- [x] Deps, pinning, host-spine pinned execution, trap flag, watermarks
- [x] Compile-time slice sizing honoured (`slice_elems`, `oversub`, `width`)
- [x] Work stealing, and the heterogeneity tolerance it buys (S33)
- [x] **Per-lane wake** — a dispatch wakes only the lanes it filled (S50 P0b)
- [x] **Lock-free completion** — `execute` takes no lock; every slice but the last is
      one atomic (S50 P2)
- [x] **Host spin before parking**, budget zero when oversubscribed (S50 P1)
- [x] **One executor per lane**, host included (S50 P1)
- [x] **One-slice tasks run inline**, never queued (S50 P3)

## Not built / refuted

- [ ] **Spinning workers — REFUTED, do not retry as specified.** 0.80–0.93× on real
      shapes; `handoff_floor` over-predicted because it gave every thread work every
      wave. See `performance/s50c-p1-spinning-workers-refuted.md`.
- [ ] Persistent kernel / static per-lane schedule with a counter barrier — the
      structural end state; not designed.
- [ ] `Work` still carries an `Arc<Run>` (one atomic pair per slice); `take_any`'s miss
      path still locks every lane.
- [ ] `path_plan_is_acyclic` in the backend is still load-bearing — `path_plan` does not
      guarantee a DAG for every shape, only for the sibling-merge class S49 fixed.

## Needs work

| Item | Where | Note |
| --- | --- | --- |
| `raptorlake` dispatch economics | `backend-llvm/profile.rs` | owed; box unreachable at S50 close, rule inert there |
| `dispatch_ns` tracks the RUNTIME, not the machine | same | re-derive whenever the handoff changes; recipe in `performance/s50e-*.md` §2 |
| ARCHITECTURE §4 rules have no automated pin | here | rules 1–6 are prose; rule 1 was violated and only found by a spinning host |

## Where to dig

| Question | Doc |
| --- | --- |
| what a dispatch costs, and why | `performance/s49b-the-dispatch-price.md` |
| the nested-run deletion | `performance/s50-p0-the-nested-runs.md` |
| the shared lock | `performance/s50b-p2-the-shared-lock.md` |
| spinning, refuted | `performance/s50c-p1-spinning-workers-refuted.md` |
| the dispatch threshold | `performance/s50d-*.md`, `s50e-*.md` |
| the design that produced all of it | `plans/plan-s50-scheduler-handoff.md` |
