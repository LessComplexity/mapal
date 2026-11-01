# S61 kernel evidence and next proof obligations

Prepared independently from frozen S60 inputs. No source change, preparation-test rerun, rental or remote access. GPU results pending. This file is an analysis attachment, not an OpenSpec implementation plan or promotion decision.

## Grounded model and scope

FRAMEWORK §4.5 placement honesty and transmission well-typing require preserving identities across the following boundaries; §6.6 prevents an observed result from silently changing the contract.

| Kind | Datum/transformation/location/transmission | Exact implementation/evidence mapping |
|---|---|---|
| Dat | Numerical face, tile geometry and rung selection | `crates/backends/llvm/src/func/gpu.rs:launch`; `crates/backends/llvm/src/profile.rs:TargetProfile::gpu_tile` (frozen B extends it with `gpu_tile_experiment`) |
| Trn | Base geometry → fill-context → selected arithmetic/launch | `func/gpu.rs:launch`, especially base call before `rungs_at` and later final `gpu_tile`/`gpu_mma_tile` calls; frozen `/private/tmp/mapal-s60-kernel/source/crates/backends/llvm/src/func/gpu.rs:265` |
| Dat | PTX, exact entry, launch geometry and null identity | `bcomparison/cases.json`, `bcomparison/sealed.json`, `diagnostics/cases.json`, under `/private/tmp/mapal-s60-kernel/` |
| Trn | Output inventory/hash/record/gate/event admission | `/private/tmp/mapal-s60-kernel/analysis/read_b.py:analyze` and `numeric` |
| Trn | Exact input PTX/context/module/function/launch → ELF text/SASS association | `/private/tmp/mapal-s60-kernel/analysis/read_images.py:analyze`, `elf_text`, `sass_counts` |
| Loc | Host benchmark process and CUDA device context | Runtime evidence boundary in those readers; actual IDs remain pending until artifacts arrive |
| Trm | Driver PTX submission and kernel dispatch | `read_images.py:analyze` joins `load-enter → image → load-exit → function → launch`; this is a diagnostic association, not a dynamic-traffic measurement |

## Gap/action matrix

| Priority | Evidence-backed finding | Status/limit | Next discriminating action |
|---|---|---|---|
| P0 | S59 blanket S58 selection wins wide M16 by 4.82–5.18× but regresses smaller Exact/Contract M16 by 37–47%; no cuBLAS win established. | Proven historical measurement, `docs/performance/s59-smallm-comparison.md`; different host/run results cannot be pooled as same-run comparisons. | Admit new four-cell B rows only after packet/result hashes, all bounded full gates, actual 256-sample gates, six timing arms and identical-PTX null envelope pass. Compare old/S58/B within new run. |
| P0 | B retains legacy successful helper geometries but does not retain S58's wide routes. Wide B Exact/Contract/requested-TF32 return scalar; S58 cooperative/MMA improvements therefore do not compose automatically with B. | Proven launch-context records and frozen `gpu_tile`/`launch` source; no B-wide timing result in packet. | Keep alternatives isolated. If a future combined selection is proposed, measure complete launch alternatives with face/rung context explicit; do not splice a helper threshold and assume S58 wide wins survive. |
| P0 | Four planned executed-image captures cover only old/S58 Contract dense win and Qwen loss, not B, Exact or TF32. | Proven `diagnostics/cases.json`; B images, tensor images and libraries' images remain absent. | Accept only exact captured entries after all observer joins and disassembly checks. Never label captured S58 SASS as B or extrapolate to Exact/TF32/cuBLAS/CUTLASS. |
| P1 | B changes cooperative aspect/panel depth, but total blocks and block threads are identical to S58 on both actual families: 64/32 blocks and 32 threads. | Proven `bcomparison/cases.json` and frozen PTX. Deepseek S58 64×1 → B32×2; Qwen32×1 →16×2. More SM coverage cannot explain any B gain. | Interpret B comparison as a coupled aspect/panel/register experiment. Obtain dynamic counters in a separate diagnostic run before attributing a timing difference to traffic, stalls or occupancy. |
| P1 | For the Deepseek Contract kernel, S58 PTX uses row/column increments16/32 and K-step8; B uses8/64 and K-step16, with shared1664→4864 B. Both use scalar `fma.rn.f32` and `bar.sync`, no `mma.sync`/`cp.async`. | Proven frozen PTX for exact entry `mapal_tile_37c5043169cbcc55`. PTX register declarations are virtual counts; no physical register count follows. | When B results arrive read driver resources independently. Future ablations can vary aspect and depth separately with unchanged face/rungs and ordered-fold gates. These remain experiments, not selected fixes. |
| P1 | S59 cooperative grids64/32 against170SMs bound simultaneous participating SMs to64/32. | Proven geometry bound; lower occupancy is a hypothesis, not full timing causation. B preserves this bound. | Require achieved-occupancy/active-warps, memory-transaction and issue/stall counters to test mechanisms; measure ordinary uninstrumented distributions separately. |
| P1 | A and B each passed918,000 finite geometry checks; B admits13,208 new legal cases and changes8/57 emitted modules, leaving49 identical to old. A's fill-context coupling discards wide Contract/TF32 wins. | Preserved preparation evidence; legality and unchanged-code inventories do not establish general profitability. No tests rerun here. | Use current runtime outcome to narrow further algorithms. Any generic admission needs neighboring sizes, device/resource conditions and large controls, beyond four measured cells. |
| P2 | Actual-shape output gates sample256 cells; exhaustive checks are bounded N256/N128 shapes. | Proven reader/case contract. Passing cannot establish exhaustive correctness of all actual outputs. | Describe evidence at exactly that scope; extend exhaustive or independent coverage only if the proposed production domain requires it. |
| P2 | Two process medians per arm with30 events/leg and null envelope provide a descriptive difference test. | No confidence intervals, generalization theorem, cross-host equality or promotion approval. | Report all raw medians, global null factor, complete excluded/rejected cells, resources and library ratios; keep runner acceptance distinct from performance decision. |
| P1 | Historical 16K square baseline has cuBLAS faster on all three faces. Mapal base medians285.6049/191.6580/115.7243ms versus cuBLAS157.0908/154.1048/80.9953ms (Exact/Contract/TF32); repeat Mapal287.3385/193.2769/115.8799ms. | Proven S57 host214845 baseline, `docs/performance/s57-large-composed-evidence.md` under “16K square”; Mapal's small base FMA lead over pinned CUTLASS194.0900ms is not a repeatable optimization gain. No new 16K run in B packet. | Keep large-shape algorithms and controls in the optimization backlog. First identify emitted/actual routes and then test multiple legal panel/reuse/pipelining alternatives with face-specific numerical admission and matched same-run library/null controls. Do not pool historic timings with S61 or declare current B regressions/gains at16K. |

## Algorithm alternatives already represented

1. Old independent scalar outputs: default baseline and an available measured fallback.
2. S58 shared cooperative ordered fold: measured wide improvements plus measured smaller-shape regressions.
3. B shared cooperative ordered fold with a different aspect/panel depth: finite legality established; profitability pending in four smaller cells; wide path intentionally remains old behavior.
4. A post-shrink one-wave filter: finite legality established; its launch/rung context fails preservation of wide Contract/TF32 improvements. No timing-based fix inferred.

Output-parallel aspect and K-panel-depth alternatives preserve the possibility of the ordered numerical contract and are suitable hypotheses to isolate. A conventional parallel split-K reduction would require a separate numerical-contract proof because re-association is not permitted by the current Exact/Contract gates; it is not an accepted substitute.

## Reproduction and pending artifact inspection

`python3 /private/tmp/mapal-s61-gap-analysis/read_static.py` audits only existing sealed input/PTX/null identities and emits `static-facts.json`.

On retrieved results, run the existing independent readers with new output directories owned here:

```sh
python3 /private/tmp/mapal-s60-kernel/analysis/read_b.py --packet /private/tmp/mapal-s60-kernel/bcomparison --results <retrieved-B-root> --out /private/tmp/mapal-s61-gap-analysis/B-independent
python3 /private/tmp/mapal-s60-kernel/analysis/read_images.py --packet /private/tmp/mapal-s60-kernel/diagnostics --results <retrieved-capture-root> --out /private/tmp/mapal-s61-gap-analysis/images-independent
```

Failed/skipped phases remain explicit non-admission. Do not rerun frozen preparation tests or infer a GPU success from the CPU-stub evidence.

## Exact PTX anchors

Prefix `/private/tmp/mapal-s60-kernel/bcomparison/ptx/rtx-5090__timing__deepseek_expert_m16_n2048_k7168__contract/`:

- `s58/kernel.ptx:33` and `:34`: `%ctaid.y *16`, `%ctaid.x *32`; `:90` and `:91`: loop limit7160 and step8; `:102`: first ordered `fma.rn.f32`; `:89` and `:244`: loop barriers.
- `B/kernel.ptx:33` and `:34`: `%ctaid.y *8`, `%ctaid.x *64`; `:105` and `:106`: loop limit7152 and step16; `:123`: first ordered `fma.rn.f32`; `:104` and `:409`: loop barriers.
- Entry `mapal_tile_37c5043169cbcc55` at line12 in both. `.maxntid32` and `.minnctapersm16` at17–18 are compiler directives, not achieved occupancy. `.reg` at21–22 names virtual PTX registers, not the driver/JIT allocated count.

Input audit completed:963 sealed B files and924 sealed capture files match; all24 B kernel/null PTX pairs are byte-identical per arm/case. Capture metadata retains unused `null_ptx` paths that are not payload files; capture is values-only and no null timing is inferred.

Wide-route evidence is preserved in `static-facts.json:wide_route_contexts`. The original `evidence/launch-context-summary.json` hash is `ae6aa09bec822146dbe651cedc8b059fb94ab865ae27c9b23a8155d6d7ec13c4`; raw B probe log hash `eb3fe7fe8f475475d28a4fd07a726fe5f41425d3aa7d6d588607ddee9331dfb1`; frozen `profile-B.rs` hash `6a01e1082ffd5d7665d96b2a898df1e7c65b0764cf1868b600b69d6fef88925d`. All three match the frozen `evidence-manifest.json`.

| Wide M16 face | Old route/grid/block/shared | S58 route/grid/block/shared | B route/grid/block/shared |
|---|---|---|---|
| Exact | scalar1152×1/256/0 | cooperative576×1/32/1536 | scalar1152×1/256/0 |
| Contract | scalar1152×1/256/0 | cooperative576×1/32/1664 | scalar1152×1/256/0 |
| Requested TF32 | scalar1152×1/256/0 | MMA1152×1/64/10752 | scalar1152×1/256/0 |

These are compiler launch selections, not new measurements of B-wide performance. The four-cell B runtime comparison cannot establish a generally improved whole-selector policy.

## S61 attempt02 boundary

Independent inspection of `/private/tmp/mapal-s61-gpu/attempt-02/retrieved/trace/batch-s60/summary.json` and its hash inventory confirms the required composed phase exited1 before optional B/capture execution. Input `kernel-B-check` and `diagnostic-check` exited0, but neither result directory exists and `diagnostics_status` is `not_run`. No B timing, GPU numerical gate or executed-image result is admitted from this attempt. The exact summary hash and checked-file count are in `attempt-02-boundary.json`. Main composed failure diagnosis and remote scheduling/cleanup are owned by other workers; no cause is inferred here.
