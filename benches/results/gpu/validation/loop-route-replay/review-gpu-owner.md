# Independent loop-route replay review

Reviewer: `/root/gpu_validation_prep`. This is a read-only review of the isolated candidate; no source edits, compilation, rental or GPU execution were performed by this reviewer.

What this change does: the replayer reconstructs required acyclic route work in addition to a loop's strongly connected component (SCC), before wiring the back and exit routes. It uses the existing canonical `LoopPlan` decide/advance targets and the existing object dispatcher.

**Verdict: no concrete blocker found in this narrow repair.** Shared integration checks and native/GPU execution evidence remain separate requirements.

## Exact reviewed inputs

| Input | SHA256 |
|---|---|
| Original reviewed patch, before test formatting | `3e0132cb046639b1d4d37cdc9c437df39d432b70d746bcc4723538186e338fc0` |
| Final formatted `evidence/fix.patch` | `32384fc81b26faff5006019b37d330daab9475bc2e1aa799afde2091282d8203` |
| Candidate `crates/mapal-rewrite/src/replay.rs` | `13060ed30adc9737fedce93c4b015e8faf0d1eba2169d0de90523d21c784347c` |
| Final `crates/mapal-rewrite/tests/loop_routes.rs` | `e6423f871f1aa512da714a20680969fa2038f763df6686445f378a796c63ce9f` |
| `evidence/ready-for-review.json` | `79742e1afcca37b3f41f6c06562bb13b609dcfeb7f24f09be45d90a1d9936be0` |

The evidence root is `/private/tmp/mapal-replay-diagnostic/`. The reviewer rehashed the final candidate and logs, and reread the final formatted regression file. The replay implementation is byte-identical to the initially reviewed candidate; the final test layout retains the same five tested behaviors.

## Checked behavior

- `replay.rs:reconstruct_loop` derives membership from the existing SCC plus the targets of `LoopPlan::decide_order` and `advance_order`. `object_topo_order` still completes all gating incoming edges before reconstructing a consumer, while `done` prevents duplicate reconstruction. The initial value and loop header wiring are unchanged.
- `mapal-ir/src/algo.rs:CategoryIr::loop_plan` remains the attribution authority. Its route cone is bounded by merge reachability and excludes loop-boundary morphisms. This keeps invariant producers outside the loop and prevents a sequential loop's exit from becoming owned by a preceding loop.
- Newly included objects pass through `reconstruct_a`, retaining normal drop, alias, constant replacement and internal-product handling. In particular, the repair does not bypass `constify` for acyclic route work. The dedicated regression confirms a real planned multiplication is removed and complete interpreter behavior is preserved.
- The existing `is_canonical` gate is unchanged. It rejects multi-merge nested SCCs before reconstruction; existing identity tests verify both direct rejection and the whole-graph identity fallback. This repair does not claim new support for noncanonical nested loops.
- The five new regressions force empty-plan replay, so a fixpoint shortcut cannot hide the missing reconstruction. They cover zero and multiple iterations, pure computed exits, observed print effects, sequential-loop dependencies, planned constant replacement, and an advance-side divide-by-zero that must be skipped on an exit step and trap on an advance step. Complete interpreter results, including output and trap outcome, are compared.

## Independently checked saved test evidence

No tests were rerun by this reviewer. SHA256 values and counts were independently checked against the existing raw stdout/stderr files and receipts.

| Run | Observed count | Stdout SHA256 | Stderr SHA256 |
|---|---|---|---|
| `evidence/negative-control-final` | 0 passed, 5 failed | `b5816fd6679662362a55a5c0ddb03574512145f9e1b97961916904493a75438e` | `ebd22a5bea3e4ee549504aa6eac7f0e5f98c8c58c2f435ed6704b9e27be1579c` |
| `evidence/rewrite-package-final` | 86 passed, 0 failed | `e4d665146ab91a2d832211fb7c44ce875c9b70e5b464a67c175ab9a6b36375ab` | `e37a140db02b70c1b22cb4c0117ca826f679ce58155ac26867823f91cf0cf739` |
| Earlier focused candidate tests | 26 passed, 0 failed | `a0a6a448ad19d3fd2142a0b7fbe92b4e05b0b8599d60a62569cb8578adc61f07` | `f2e50cdf8113549984db6b03653bdd485596fbe53a08a7e9ead8ab11b4bec116` |

The exact final commands in the receipt are `cargo test --offline --locked --release -p mapal-rewrite --test loop_routes` against the old implementation as the negative control, and `cargo test --offline --locked --release -p mapal-rewrite` against the corrected implementation, both using the isolated source and target directories. The negative control's five failures are expected evidence of test sensitivity, not acceptance failures on the fixed source.

Not checked by this review: fresh native execution, GPU values, runtime stream ordering, or performance. Successful LLVM emission alone does not establish those properties. The separate native timeouts remain limitations and are not converted into passes. Shared full-workspace validation and documentation reconciliation remain the integrating owner's responsibility.
