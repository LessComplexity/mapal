# S56 — validation baseline restored

Date: 2026-10-21. Branch: `s54-gpu-fusion`, starting at `44972af` (S55 consolidation). This is the immutable checkpoint after item 1 of the user's authorized continuation; GPU work follows under a separate OpenSpec change.

## Request and decisions

The user asked to proceed one item at a time, use Vast where needed, and report completion/results for each item. Worktree consolidation was already complete: one checkout, 51 historical branches retained, 18 physical worktrees preserved before removal. Keep `.claude/worktrees/_archive-s54-20261021/` and untracked `oainotes.md`.

Item 1 originally covered build collisions, test registration and six drift references. Its full test run found a real, reproducible rewrite defect, so validation was extended to repair that defect rather than dismissing the failure.

## Changes

- CUDA's developer example is `emit_cuda`; LLVM retains `emit`. Updated active shell/doc callers and usage text. Both examples coexist after a workspace build. Historical commands in immutable sessions/results remain historical.
- Restored `sme_kc_reproduces_the_measured_optimum`'s missing test attribute, removed the duplicate registration from its neighbor, and corrected its moved arithmetic comment. Production profile decisions did not change.
- Fixed ConstFold dependency preservation. Folding an untaken `Index(...) % 1` to zero detached the Index from its guard and made it unconditional. `analyze_const_fold` now excludes guard-owned target objects using existing `guard_plan` facts. Guard-owned loops are compressed to LoopEnter handles, so existing `loop_plan` decide/advance/exit targets are protected too; SCC membership already covers cycle members.
- Preserved random seed `c23d83c935cd8f7c04867477376224d317bab38db7a38f905e12af25351198f4` and added four directed regressions. Negative controls reproduced both the initial outcome change and the incomplete fix's loop replay panic.
- Reconciled rewrite DESIGN/IMPLEMENTATION/STATUS and the section 4.5 review. Repaired the six stale root/LLVM/IR/lower references and CUDA example mappings. CUDA arena coloring remains explicitly planned, not claimed built.
- Initialized OpenSpec with supercharge rules and archived `stabilize-s54-validation`; its emitter and rewrite-equivalence contracts are merged into specs. Prepared `verify-s54-gpu-selection` for the remaining work.
- Rebuilt the local derived graph: 8,373 nodes / 18,515 edges; ignored `graphify-out/` in git. `graphify affected analyze_const_fold` points to rewrite driver/lib consumers.

## Validation

| Check | Result |
| --- | --- |
| Initial full required-clang host run | 1171 passed / 1 failed / 1 ignored; new persisted ConstFold counterexample |
| First repair full run | 1174 passed / 1 failed / 1 ignored; calc IR golden changed; independent review additionally found omitted loop payload protection |
| Final rewrite crate | 81 passed / 0 failed; includes focused guard/loop regressions, properties and goldens |
| Final host workspace run 1 | **1176 passed / 0 failed / 1 ignored**, 78.0 s |
| Final host workspace run 2 | **1176 passed / 0 failed / 1 ignored**, 65.7 s |
| Workspace examples | build succeeds; all nine target names unique |
| Emitter commands | LLVM emits LLVM IR; CUDA emits CUDA C++; missing args exit2; old CUDA `--example emit` rejects |
| Formatting/shell/whitespace | pass |
| Drift | **0 dead / 460 references** |
| OpenSpec validation/archive | strict validation passed; archived with two added requirements |

Full workspace command, with normal host access for sysctl and native launching:

```sh
MAPAL_REQUIRE_CLANG=1 cargo test --workspace --release --no-fail-fast -- --test-threads=1
cargo build --workspace --release --examples
```

The guard cutoff is deliberately conservative while guards survive. Calc's output remains the same five lines; its optimized graph shrank from 107 to 81 objects because untaken branches no longer pre-fold into retained dead constants. Deferred guard resolution uses 11 fixpoint rounds instead of 4 (cap32). Only calc's graph/report goldens changed. Independent review re-ran the loop failure under ConstFold-only and full rewrite, plus 40 nested unknown-guard comparisons, with matching outcomes.

Logs and runners: `/private/tmp/mapal-s55-validation/`, copied into the ignored persistent backup `.claude/worktrees/_archive-s54-20261021/validation-s56/`. Final summaries are `final-summary.json`; complete logs are `final-host-{1,2}.log` and `final-examples.log`. Failed-run evidence is retained separately.

## Model delta and reconciliation

No new runtime Dat/Trn/Loc/Trm. The existing `Trn analyze_const_fold : CategoryIr → RewritePlan` narrows its partial alias/constify candidates to preserve rewrite R1/R4. Guard and loop facts remain deduced from the IR's existing analyses. The CUDA command rename changes only the developer artifact binding. See `docs/components/rewrite/reviews/review-stabilize-s54-validation.md` for the built section 4.5 check.

Reconciled component maps first, rewrite/CUDA design intent where applicable, component status, root implementation/status, then current continuation. Existing architecture-map atoms/signatures are unchanged. This checkpoint does not certify GPU correctness or performance.

## Live execution state

- No test/build process remains running at this checkpoint.
- No Vast instance has been created by this continuation. Do not touch unrelated rentals; S55 observed stocklab instance57065574.
- GPU harness preparation is present but not included in this item-1 commit: `benches/gpu/box_run.sh`, `vast_run.sh`, `test_harness.py`. Ten local stub tests passed. It adds differential-only mode, failure propagation/cleanup, and forced-ON retain-resident timing. Actual GPU execution is pending.
- `/private/tmp/mapal-s56-gpu/build_smoke.py` is prepared but has not run. It builds native/Linux runtime, interpreter and LLVM emitter, then fresh smoke bundles for all five profiles. It refuses to reuse an existing smoke directory.
- Current source/oracle pipeline changed with the rewrite repair: regenerate benchmark oracles; do not copy historical expectations solely because input sources match.
- gbrain doctor again timed out (20 s); indexing unavailable. The committed session file remains the continuity record.

OpenSpec snapshot:

```json
{"changes":[{"name":"verify-s54-gpu-selection","completedTasks":0,"totalTasks":13,"status":"in-progress"}]}
```

## GPU continuation facts

The saved `integrated-*-prefix.txt` files are complete **pre-arena-fix** runs with failures. The fix `b995cf9` is ancestral to current HEAD. Historical post-fix `bJ-*` bundles exist, but their `r4-*` result directories contain only rental/SSH logs. Fresh correctness checks are required.

Smoke first: `px_bias_100`, `px_mlp2_64_f64`, `px_residual_128`, `px_fan_128`, `e2e_trap`, both faces/applicable TF32 and arena/epilogue/resident/devgen/eager-init/async/arch-rungs/streams arms. A `== done` marker alone never certifies full timing results; check exact bundle inventory, failures, library/value gates, rows/repetitions and profile artifacts.

CUTLASS cache: `/private/tmp/claude-501/-Volumes-LessComplex-Personal-mapal/5796dddd-f57b-41ae-ba33-9afb66b7242d/scratchpad/arch-select/b1/cutlass`, commit `91843b71d90737e0a9b86f7d9df213cff5a3526f`. Current build helper copies it into a fresh bundle; it does not reuse compiled programs.

Vast availability was checked read-only. Offers must be refreshed before rental. 5090 machines12031 and139986 are independent of earlier completed streams machines36493/131570/140080/151541. The only offered H100 PCIe was machine81035/host415974, **already used** by retain-resident confirmation; the first earlier host was machine150296/host670790. Reusing81035 is a fresh session, not a third independent host. Existing evidence favors conservative OFF if a robust ON gain cannot be demonstrated.

Correct profiles: `rtx-3090:sm_86`, `a100:sm_80`, `rtx-4090:sm_89`, **`h100-pcie:sm_90a`**, `rtx-5090:sm_120`. Do not substitute A100 PCIe/H100 SXM silently. Explicit probe arms must include forced-ON streams when the default is OFF; the prepared decode patch likewise includes forced-ON retain-resident. Separate traced kernel improvements from untraced iteration/wall results.

## Exact next commands

```sh
cd /Volumes/LessComplex/Personal/mapal
openspec instructions apply --change verify-s54-gpu-selection --json
python3 benches/gpu/test_harness.py -v
bash -n benches/gpu/box_run.sh benches/gpu/vast_run.sh
python3 /private/tmp/mapal-s56-gpu/build_smoke.py
```

After the smoke bundle succeeds, select a refreshed offer and run the existing wrapper with normal host network access:

```sh
DIFFERENTIAL_ONLY=1 PROFILE=0 START_TRIES=36 \
  bash benches/gpu/vast_run.sh <offer-id> rtx-5090 sm_120 \
  /private/tmp/mapal-s56-gpu/smoke /private/tmp/mapal-s56-gpu/smoke-5090-results 1
```

Capture raw CLI failures to local logs rather than echoing authentication-bearing tracebacks. Verify the task-created instance is destroyed. Then resolve the three selection disputes, finish five-profile validation plus 4090/5090 confirmations, and publish both S54 reports with every loss. The largest MLP2/decode gaps follow verification; ADR-0039 remains the user's decision.
