# 2026-10-21 — S55 worktree consolidation and S54 recovery

## 0. Continuation brief

The main checkout is now the single registered worktree, on **`s54-gpu-fusion`**. Original integrated S54 HEAD was `2c38f21`; consolidation added `a613f53` and `4db1f48`, recovering two later l2-raster result/citation commits. The feature branches were already merged; no new runtime selection or placement behavior was introduced here. All 51 previous local branch refs remain. The S54 integrated table is still provisional: its final verifier/integrator never ran.

Next action: use OpenSpec to scope the remaining validation/build cleanup and disputed table choices, then complete the integrated five-part measurement. Read `sessions/2026-10-21-s54-fusion-placement.md` for the original findings; the corrections and commands below supersede its worktree cleanup and H100 target instructions.

First inspect:

```sh
cd /Volumes/LessComplex/Personal/mapal
git status -sb
git worktree list
cat docs/next-session.md
cat .claude/worktrees/_archive-s54-20261021/validation/host-summary.json
```

## 1. Work completed

- Audited nine S54 and nine older `agent-*` worktrees, plus three stale records for missing detached worktrees.
- S54 trees were clean, with disposable Cargo/Python caches. Every S54 tip was ancestral to `s54-gpu-fusion` except two l2-raster commits: `1e2cf2a` and `86d6310`.
- Cherry-picked those commits as `a613f53` and `4db1f48`. The two recovered result files and l2-raster table match the source branch. An independent comparison of `profile.rs` with citation values removed was byte-identical: no rung decisions changed.
- Older worktree HEADs were all ancestral, but their uncommitted experiments were not all represented in current source. Preserved binary diffs, changed/untracked file snapshots, and approximately 338 MB of `target/tmp/` raw experiments before removal. Did not apply those historical experiments to S54.
- Created and verified a complete git bundle, including the three stale detached HEADs. Verified all 18 archive hashes and 2,224 archived file payloads. All 51 previous local branches remain; only the integration branch advanced.
- Removed the 18 redundant worktrees and pruned the three missing-tree records. There is one working checkout.
- Preserved all six originally loose root files. Five were identical to files committed on S54 and moved aside before checkout; `oainotes.md` remains untracked and byte-identical (6,653 bytes).

## 2. Decisions

| Decision | Verdict | Why |
| --- | --- | --- |
| Consolidate on `s54-gpu-fusion` | kept | It already held the merged S54 implementation and latest handoff |
| Recover later l2-raster data | kept | Two result/citation commits had not reached the integrated branch |
| Keep historical branch refs and archived dirty experiments | kept | Removing worktrees must not erase unique uncommitted research |
| Apply old dirty patches to current code | discarded | They are historical variants, not an identified continuation change |
| Change disputed rung selections during cleanup | deferred | Consolidation provides no new integrated performance verdict |
| Treat an example-name collision as the cause of S54's earlier failures | discarded as a conclusion | The collision exists, but differential tests invoke the emitter library directly |
| Use `sm_90a` for H100 | kept | `bundle.sh` documents it; the H100 emitter uses wgmma, and profiling passes the target to ptxas |

## 3. Tests and checks

Source under test: **`4db1f48`**; subsequent tracked changes are continuity/status documentation.

| Check | Result | Meaning |
| --- | --- | --- |
| Independent archive / refs audit | pass | One worktree; 51 previous branch refs retained; all archives readable; detached histories in verified complete bundle |
| Two sandbox `cargo test --workspace --release --no-fail-fast` runs | 1147 passed / 25 failed / 1 ignored; 1158 passed / 14 failed / 1 ignored | Not a clean baseline: three hardware-detection failures plus native execution timeouts |
| Host `cargo test --release -p mapal-backend-llvm --lib native_ -- --nocapture` | 5 passed / 0 failed | Hardware detection works with host access |
| Host required-clang differential example check | 1 passed / 0 failed, 40.77 s | The example suite that timed out also passes with host access |
| Host full suite, serial, run 1 | 1172 passed / 0 failed / 1 ignored, 200.3 s | Complete CPU/emission baseline under the command below |
| Host full suite, serial, run 2 | 1172 passed / 0 failed / 1 ignored, 178.7 s | Independent repeat under the same host/serial conditions |
| `git diff --check` | pass | Consolidation documentation has no whitespace errors |
| supercharge drift | 6 dead / 452 refs | Existing documentation debt, listed below; no archive/signoff over this failure |

The successful full-suite command is:

```sh
MAPAL_REQUIRE_CLANG=1 cargo test --workspace --release --no-fail-fast -- --test-threads=1
```

Use normal host access. The sandbox denies `/usr/sbin/sysctl` reads for L1/L2/SME facts; `profile.rs:sysctl_u64` converts those failures to `None`. The other failures were native-run timeouts, including trivial programs, not observed value mismatches. Host access plus serial execution passed; these checks do not isolate which condition caused every timeout or explain the original S54 run's 20 failures. The test source documents serialized macOS first-execution validation as a possible contention source. No code fix was made.

Both `mapal-backend-llvm` and `mapal-backend-cuda` still define an `emit` example, and `profile.rs` still has the duplicated test attribute. Preserve these distinctions when addressing the build/test cleanup.

Logs and JSON summaries are under `.claude/worktrees/_archive-s54-20261021/validation/`.

## 4. Live handoff state

| Type | Handle / location | State | Inspect / resume | Cleanup |
| --- | --- | --- | --- | --- |
| checkout | `/Volumes/LessComplex/Personal/mapal`, `s54-gpu-fusion` | Single registered worktree; local, not pushed; consolidation doc commit follows `4db1f48` | `git status -sb`; `git log -4 --oneline` | Keep |
| user file | `oainotes.md` | Untracked, unchanged | `git status --short` | Keep |
| archive | `.claude/worktrees/_archive-s54-20261021/` | Local, git-ignored; approximately 124 MB at initial verification | Read `README.md`, `manifest.json`, `branches-before.txt`, `worktrees-before.txt` | Keep until historical experiments are no longer needed; not a worktree |
| histories | archive `all-refs.bundle` | Complete history, including detached-only HEADs `d3ca82c`, `6168863`, `1daddaa` | `git bundle verify .claude/worktrees/_archive-s54-20261021/all-refs.bundle` | Keep with archive |
| local validation | archive `validation/` | All validation commands completed; no cargo/rustc/GPU job command found in final process check | Read `summary.json`, `host-summary.json`, `host-*.log` | Keep logs |
| remote instance | Vast `57065574`, label `stocklab-orderflow-20261021` | Running at read-only inspection; GTX 1660 S, unrelated label/part to S54 | `vastai show instances` | Left unchanged; not an S54 cleanup target |
| S54 workflow records | `~/.claude/projects/-Volumes-LessComplex-Personal-mapal/5796dddd-f57b-41ae-ba33-9afb66b7242d/` | Workflow script and journal paths from S54 handoff still exist | See original S54 log's exact paths | Keep |

No GPU rentals were started and no remote state was changed. Do not repeat the original handoff's broad claim that the Vast account is empty. Do not remove the archive during future worktree cleanup; it is an ordinary ignored directory, not a registered tree.

To restore old experiments, follow the archive README: create a new worktree on the recorded branch, apply its saved binary patch, and extract its tar archive. The manifest records original HEADs, statuses, filenames and checksums.

## 5. In-flight work state

`openspec list --json` after consolidation:

```json
{
  "changes": [],
  "root": null,
  "status": [
    {
      "severity": "error",
      "code": "no_openspec_root",
      "message": "No OpenSpec root found from the current directory.",
      "target": "openspec.root",
      "fix": "Run openspec init to create a root here."
    }
  ]
}
```

This is unavailable work-state tracking, not evidence that S54 is complete. No per-change status exists to query. Before software changes, run `openspec init`, apply the supercharge context/rules from its OpenSpec reference, and use the propose/apply flow. Existing S54 plan/model remain source context; no parallel plan document was created here.

## 6. Open items, in continuation order

| Priority | Item | Next action | Done when |
| --- | --- | --- | --- |
| P0 | Build/test cleanup | Through OpenSpec, remove the example-name collision and duplicate test attribute; retain timeout evidence rather than guessing its cause | Build outputs unambiguous and required-clang baseline repeatable |
| P0 | RTX 5090 streams conflict | Review builder/verifier evidence; shipped ON, verifier recommended OFF | Explicit measured decision in the final table |
| P0 | H100 retain-resident conflict | Review evidence; shipped ON, verifier recommended OFF | Explicit measured decision in the final table |
| P0 | l2-raster conflict | Decide whether its small kernel-only gains justify retention; verifier refuted an end-to-end gain | Explicit accept/drop verdict, including newly recovered data |
| P0 | Integrated S54 verification | Complete five-part run and final 4090/5090 table confirmations | Complete value-gated results and independent final verification; partial prefixes do not count |
| P1 | Performance reports and reconciliation | Write `docs/performance/s54-gpu-fusion.md` and `s54-per-arch-selection.md`; reconcile component implementation → architecture → status, then roll up root status | Every result and loss recorded, mappings reconciled, drift clean before archive/signoff |
| P1 | Six existing drift findings | Resolve the rows in section 9 while reconciling their components | Drift reports zero dead refs |
| P1 | Largest placement losses | After final table verification, prioritize 4090 `pt_mlp2_4096` (55.1 ms vs cuBLASLt 11.4 ms), then decode (0.05–0.12× library speed) | New controlled measurements and scoped changes |
| P1 | Remaining S53 queue | Exact face; Ampere/Hopper TF32; helper clock-fence bug; eager-init race | See S54/S53 logs; ADR-0039 remains Sapir's decision |

Corrected build/bundle commands after validation and selection work:

```sh
cd /Volumes/LessComplex/Personal/mapal
cargo build --release -p mapal-rt
cargo build --release -p mapal-rt --target x86_64-unknown-linux-gnu
cargo build --release -p mapal-interp --example run
cargo build --release -p mapal-backend-llvm --example emit  # LAST after workspace tests
S=/private/tmp/mapal-s55
mkdir -p "$S/b"
PREV="" bash benches/gpu/bundle.sh "$S/b" \
  rtx-3090:sm_86 a100:sm_80 rtx-4090:sm_89 h100-pcie:sm_90a rtx-5090:sm_120
```

Build LLVM `emit` last while the collision still exists; update commands and all consumers if the examples are renamed. Reuse the previous comparison bundle's CUTLASS checkout when available; otherwise `bundle.sh` clones it and records its commit. Local tools/LLVM 22.1.8 and the x86 Rust target were present at inspection. These bundle/rental commands were not executed during consolidation.

For each appropriate fresh offer, substitute its ID and profile/SM pair:

```sh
PROFILE=1 bash benches/gpu/vast_run.sh <offer> <profile> <sm> "$S/b" "$S/res" 30
python3 benches/gpu/table.py "$S/res/results-<profile>.txt"
vastai show instances
```

Use `sm_90a` for H100 here too. Inspect complete differential gates, benchmark rows and profiling archives; a script's exit status or `integrated-*-prefix.txt` alone does not certify the build. `vast_run.sh` rents and destroys its own instance on exit; inspect leftovers after interruption and clean up only instances belonging to this work.

## 7. Architecture / model changes

No model delta: no Dat/Trn/Loc/Trm, placement, inter-component port, shared object, or code root changed. The accepted S54 placement model remains in `docs/components/backend-nvptx/ARCHITECTURE.md`; source changes here are only `GPU_RUNGS_OFF` citation values in `crates/backends/llvm/src/profile.rs`. The independent source comparison establishes that this consolidation introduces no new §4.5 placement/transmission/composition behavior. It does not certify the unfinished S54 integrated implementation.

## 8. Docs reconciled

| Doc | Change |
| --- | --- |
| `docs/next-session.md` | Single-checkout state, backup retention, corrected continuation/build/H100 instructions; old CPU-side handoff retained |
| `docs/STATUS.md` | Short current continuation prefix; labels the older detailed body historical and S54 final verification pending |
| This session log | Consolidation evidence, live state, validation, drift debt, exact continuation |
| Ignored archive `README.md` | Recovery instructions and preservation policy |

Prior immutable session logs were not edited. Component implementation/model reconciliation is an open S54 integration task, not completed by this checkout cleanup.

## 9. Drift and optional tooling

`supercharge-drift` (skill `scripts/drift-check.sh . --json`) returned **6 dead / 452 refs**. The affected implementation maps did not change from original S54 HEAD `2c38f21`; these findings are retained as open debt, not repaired speculatively during worktree consolidation:

| Document | Dead reference |
| --- | --- |
| `docs/IMPLEMENTATION.md` | `llvm/func/ops.rs` |
| `docs/IMPLEMENTATION.md` | `cuda/func.rs` |
| `docs/components/backend-llvm/IMPLEMENTATION.md` | `func/mod.rs:WINDOW_SUBROWS` |
| `docs/components/ir/IMPLEMENTATION.md` | `mapal-backend-llvm/tests/differential.rs` |
| `docs/components/ir/IMPLEMENTATION.md` | `crates/backends/llvm/src/func.rs` |
| `docs/components/lower/IMPLEMENTATION.md` | `crates/mapal-lower/src/lib.rs:is_collection_builtin` |

Preflight found graphify 0.9.5, OpenSpec 1.10.0, semble 0.5.5 and gbrain 0.46.28.0. There is no graph or OpenSpec root. Graph refresh skipped because no graph exists and no runtime behavior changed. `gbrain doctor` did not complete within 30 seconds, so memory indexing is unavailable in this run; this committed session file remains the continuity source. Recheck `gbrain doctor` before attempting capture.

## 10. Tracked files changed

Recovered commits: two `benches/results-s54-gpu/l2-raster-*.txt` result files, `l2-raster-tables.md`, and `profile.rs` citations. Continuity commit: `docs/STATUS.md`, `docs/next-session.md`, this session log. Backups and validation logs are intentionally local and ignored; `oainotes.md` is retained untracked. Nothing was pushed or merged to `main`.
