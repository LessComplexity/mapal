# Loop-route replay repair: isolated proof

The replay failure is confirmed and a narrow repair is ready. The final patch is `evidence/fix.patch`, SHA256 **32384fc81b26faff5006019b37d330daab9475bc2e1aa799afde2091282d8203**. It changes `crates/mapal-rewrite/src/replay.rs` and adds `crates/mapal-rewrite/tests/loop_routes.rs`. Temporary tracing and the diagnostic helper are excluded. Shared integration and the required-clang workspace check are owned by the root agent; their results are separate from this isolated proof.

## Bound baseline and diagnosis

The source snapshot was created with `git archive` from **b4152c5f4d9ad802d0388c1c64b628091d494e6a**, without a worktree. `evidence/shared-before.json`, `frozen-source-manifest.json`, and `source-b4152c5.tar` bind the baseline. All 328 source entries, four existing compiler/runtime inputs, three LLVM tools and HEAD matched again in the **pre-integration** `evidence/shared-after.json` (SHA256 **02f0b301a44e47e86872528bf9fac267dc7c090daee976c16cbff5bbd1252345**). This is a historical check before root applied the repair, not a claim that the current shared source still equals b4152c5. Frozen baseline binaries were not replaced by this agent.

The diagnostic-only driver/replayer records and binary are retained in `evidence/diagnostic-source.patch`, `instrumented-{driver,replay}.rs` and `diagnostic-emit`. Both actual failures occur in **round 1, ConstFold**, with alias=14 and every other plan channel zero:

| Fixture | Merge / exit route | Required exit slot0 | Definition and membership |
| --- | --- | --- | --- |
| Existing tiny counter-print source | 525 / 675 | ObjectId674v1, `(IoToken, Array<f32;2>)` | Pair m748 from529, m749 from527; unmapped, outside SCC, inside canonical driver, no drop/constify |
| Original recurrent check | 2948 / 5908 | ObjectId5907v1, `(IoToken, Array<f32;8>)` | Pair m6979 from2952, m6980 from2950; unmapped, outside SCC, inside canonical driver, no drop/constify |

Exact fixture SHA256s are respectively **01b77ce1ff970fd26e28e22f3d779d06ba5c5403e809a9d3e0f72450ccf45ce2** and **0e4c05b403550da3a3afaeebcea8323e2af1dca887aa6a94d95c1e0dd7c57142**. Commands, exit101, object definitions, source locations, resolved identities and mapped route controls are in `evidence/{tiny674,original5907}/{command.json,stderr}`. The instrumentation shifts the panic to replay.rs:1003; unchanged code panics at line989.

The failed invariant is in reconstruction: `feed` requires a mapped resolved object before `loop_exit` consumes it. Lowering's `emit_exit_arm` legitimately builds the `(token,value)` payload from the merge projections. That payload is not on a cyclic path, so SCC-only reconstruction skips it. It is required canonical loop work, already described by `CategoryIr::loop_plan` in `mapal-ir/src/algo.rs:3117–3137`. The identity mapping of each failed feeder, absent drop/constify, and mapped back/condition controls distinguish this omission from alias removal, a numerical error, GPU behavior or an invalid input graph.

## Repair and semantic evidence

`reconstruct_loop` now derives membership from the existing SCC plus targets of the existing decide/advance orders, walks the same global object order, and delegates recipe handling to existing `reconstruct_a`. This retains internal-pack, alias, drop and constify handling. No value is invented; optimization is not bypassed; observations and exit payloads remain present. No new public type, model field, runtime interface or loop analysis is introduced.

The five new directed tests force empty-plan replay, assert the required payload is outside the SCC but inside the canonical decide cone, validate every reconstructed graph, and compare complete interpreter outcomes. They cover observed and pure computed exits, zero/multiple iterations, actual non-SCC constification, sequential loops/invariants, and selected/unselected traps and effects. Each of the six rewrite passes and the full pipeline is exercised.

| Check | Result | Receipt |
| --- | --- | --- |
| Exact final formatted regressions on unchanged replay | 0 pass / **5 fail**, all missing feeders | `negative-control-final.{json,stdout,stderr}` |
| Focused new routes + identity/guard/ConstFold suites | **26 pass / 0 fail** | `focused-tests.{stdout,stderr}` |
| Full rewrite crate, including lift/inline/fusion/properties/goldens | **86 pass / 0 fail** | `rewrite-package-final.{json,stdout,stderr}` |
| Two candidate files rustfmt; reverse-apply patch check | pass | `ready-for-review.json` |
| Tiny raw / identity / full rewrite interpreter | exact complete RunResult equality | `fixed-checks/tiny674/semantics/command.json` |
| Original raw / identity / full rewrite interpreter | Done; identical complete RunResults; **8408 numeric bits, 8838 lines** each match pinned scalar fp32 labels/order/values | `test-receipts.json`, `fixed-checks/original5907/semantic-outputs/` |
| Tiny native Exact and Contract emit/link/run | both pass; exact ordinary stdout `0\n1\n` | `fixed-checks/tiny674/native/` |
| Original GPU-facing LLVM emission | **8/8 pass**, 3.2–4.9 seconds: RTX5090/A100 × Exact/Contract × shipped/full-recompute | `fixed-emission-receipts.json`, corresponding host/device LLVM |

The original scalar comparisons check all immutable inputs and every intermediate checkpoint, not just final output. The diagnostic helper `source/crates/mapal-rewrite/examples/audit_semantics.rs` is retained solely for reproduction and excluded from the repair patch. `check_repair.py` stops on the first failed command; `check_remaining_emission.py` preserves each subsequent bounded emission independently. The saved command records are authoritative, including timeouts.

The original default native Exact and Contract emissions each reached the **40-second** cap with empty stdout/stderr. Exactly one separate Contract `--no-tile` attempt also reached40 seconds. The tiny native program succeeds, and all original GPU-facing emit arms succeed, but complete original native execution and original Contract/fmaf output are **not certified** here. The cause of native emission latency is unresolved; no performance conclusion or claim that the repair fixes it is made. No GPU execution was performed.

## Reproduce and integrate

All Cargo commands used an isolated target and `--offline --locked`. The focused negative control restores `evidence/baseline-replay.rs`; the fixed source is retained in `evidence/candidate-replay.rs`. Run the tests from the temporary snapshot:

```sh
CARGO_TARGET_DIR=/private/tmp/mapal-replay-diagnostic/target cargo test --offline --locked --release -p mapal-rewrite
CARGO_TARGET_DIR=/private/tmp/mapal-replay-diagnostic/target cargo build --offline --locked --release -p mapal-backend-llvm --example emit
```

Each compiler/native invocation is saved with argv, cwd, cap, status and output hashes. LLVM clang is `/opt/homebrew/Cellar/llvm/22.1.8/bin/clang-22`, SHA256305697d3868243dca6e4aabf4f628df7de135d701eb4fd78f6a5d7528fef4395. Native links use the frozen shared `target/release/libmapal_rt.a`, not a rebuilt runtime.

The existing required-clang baseline command is in `docs/sessions/2026-10-21-s56-validation-baseline.md:39`. Root's shared integration uses a **new** target so the old GPU payload and compiler/runtime binaries remain intact:

```sh
MAPAL_REQUIRE_CLANG=1 CARGO_TARGET_DIR=/private/tmp/mapal-replay-shared-host-target cargo test --offline --locked --workspace --release --no-fail-fast -- --test-threads=1
```

Normal host access is required for the existing sysctl/native-launch checks, as documented by the baseline. This agent has not run the full shared workspace gate. The OpenSpec `repair-loop-route-replay` remains active for root's integration, checks, documentation reconciliation and final drift/archive work.
