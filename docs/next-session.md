# Continue the GPU performance comparison

Start with [current results](performance/gpu/current-results.md), the [performance index](performance/README.md), and [raw evidence](../benches/results/README.md). Use descriptive topic/experiment names for new reports and folders. Session numbers belong only in historical provenance and immutable handoffs. The [path map](../benches/results/path-map.json) resolves relocated artifacts without changing their contents.

## Current state

- Branch `s54-gpu-fusion`; checkpoint **dece72f** commits the integrated allocation correction, admitted earlier results and requested evidence organization. New competitive and fused-library results plus the profiler failure are reconciled in the working tree. Preserve unrelated `oainotes.md`. Local commits are authorized; no push.
- The unused raw-output allocation correction passes640 ordinary GPU processes and28 separate counter processes;1,190 distinct local tests pass across workspace and host-access backend rerun, with one ignored. No allocation latency gain is claimed.
- Latest residual MLP: all20 native confirmations and10 original controls admitted on one RTX5090. Tuned fused cuBLASLt is2.24×/2.16× faster than Mapal; tuned fused CUTLASS has lower observed medians too. Mapal beats restricted eager/graph-disabled Inductor. Input reset/cache state differs; no causal retention claim. See [fused libraries](performance/gpu/workloads/fused-library-comparison.md).
- Small-M cooperative/deeper alternatives remain isolated. The deeper alternative improves two FMA cells and regresses Exact. Four actual old/cooperative Mapal Contract cubin/SASS captures are admitted.
- **Competitive CUTLASS now measured:** twelve configurations attempted, six compile/pass gates. Selected16×64×16 improves default2.32–2.34× and beats the frozen Mapal baseline in all four cells. cuBLAS remains faster. See [the full comparison](performance/gpu/kernels/competitive-cutlass.md).
- Independent admission verifies983payload/336ordinary/63outer artifacts. A second numerical implementation imports no harness checker and verifies all254,976raw words with Fraction/IEEE arithmetic. Root independently checks1,044event samples and held-out selection. No full GPU A/B byte dump or per-timed-output export was captured; retain those limits.
- Organization moved2,849files; all2,802raw hashes remain unchanged. The relocation check passed372links; the newer comparison reports bring the current check to400links, zero broken. Strict OpenSpec7/0; drift0dead/575refs. Code graph refreshed; document semantic extraction lacks an LLM key. gbrain indexing encountered an existing PGLite lock, which was not removed.

## Next work and owners

| Owner | Work | Private working directory |
| --- | --- | --- |
| Root | Integration, artifact review, independent admission and shared docs | `/private/tmp/mapal-s61-root/` |
| `/root/composed_analysis_s61` | Native admission complete; independently review callback capture | `/private/tmp/mapal-composed-fused-references/` |
| `/root/gap_analysis_s61` | Callback capture of actual library images without hardware counters | `/private/tmp/mapal-library-image-capture/` |
| `/root/comparison_run` | Completed exact-owned cleanup and independent provenance audit; idle | `/private/tmp/mapal-s61-gpu/attempt-10/` |

The new fused-reference source has actual-C++ protocol proof:54positive/12negative cases,
plus real subprocess descendant cleanup/cancellation tests. Request identity, candidate,
nonce, geometry and cross-process opaque cuBLASLt algorithm configuration are bound.
Native archive is sealed: SHA256 `12fd10e54a2f9f56efda2d4cdc7f9939c2e2ea03f7ad54ba452beb31b52a6c4c`,
manifest `4fb22559cc07319d46ae719425a03e8efc258ebeea1ffeda20e6e40245c8d5b0`.
Root verifies all962payload/963archive files and check-only passes. The GPU run completed with all phases exiting0. Full independent admission passes:148 native executions,770,816raw words,8selections and all30 confirmation/control rows.
Use the original composed-v4 controls on
that same newly owned host. Matched-geometry fused/unfused library deltas can price their
epilogues; whole-program Mapal comparisons alone cannot isolate fusion or retention.

The actual Nsight Compute2025.1.1 run failed with **ERR_NVGPUCTRPERM**, exit1, no timeout
and no `.ncu-rep`. An error split one pre-gate value line; reject it rather than reconstruct
it. All three post gates pass256sampled values, but cannot override diagnostic failure.
Library legs/dynamic phase were skipped. All15diagnostic seals and the selected binary
match. There is no library SASS or counter admission. Do not change host driver/security
controls. The callback extension must prove complete module/function/dispatch joins,
capture all internal kernels and reject missing/ambiguous metadata. No counters requested.

Broader shapes, graph-enabled PyTorch, recurrence, causal retention/allocation and generic
selector integration remain open. The [IR fact map](performance/gpu/disassembly/ir-facts-and-realization-gaps.md)
identifies available facts and concrete lowering limits without asserting a timing cause.
Public new evidence belongs under `benches/results/gpu/kernels/competitive-cutlass/`,
`workloads/residual-mlp/`, or `disassembly/library-kernels/` as appropriate.

## Machines and authorization

**No live task rental.** Attempt10 instance57259496, machine58084/host98309,
completed native fused/unfused references, pinned Torch bootstrap and original composed-v4
controls on the same RTX5090 UUID. All phases exit0, no timeout. All1,492 recovered files
are bound by the private recovery receipt;75 outer hashes verified. Independent numerical,
selection and timing admission passed. Exact-owned cleanup independently confirmed
absence at2026-10-22T19:17:06.789Z. Frozen operator packet SHA256:
`0d6c4db7758ea5e34510c9b8c7c5522d1e7cd653ec543dfda93775b9159b35f8`.
Trace: `/private/tmp/mapal-s61-gpu/attempt-10/retrieved/trace/`.

All ten task rentals are closed:57242062,57245029,57247103,57248819,57249550,
57253902,57254920,57256063,57256816,57259496. Attempt09 completed ordinary comparison
and a separate failed NCU diagnostic; independently absent18:51:00.514Z.

The user explicitly removed monetary caps and authorized equivalent RTX5090 uploads/runs
of reviewed benchmark content. Do not ask for an already-granted budget approval. Use
operational deadlines and retrieve evidence before exact-owned-instance cleanup.

**Protect every pre-existing instance.** Known unrelated IDs57065574,57100404,57126023,
57133432 plus every fresh prelaunch ID. Only an exact new ID from this task's own create
response, matching physical host and absent from the prelaunch inventory, permits access
and cleanup. Novel inventory membership alone is insufficient.

Earlier startup failures are preserved. The installed CLI's `--cancel-unavail` option was
proved offline to change only that request field; it requests explicit scheduling rejection
instead of a stopped contract. The successful latest request includes it. No causal provider
or account repair is claimed. Raw provider/account/API/create/wrapper logs stay private;
never print them. An earlier credential rotation remains outstanding.

## Inspect and resume

```sh
cd /Volumes/LessComplex/Personal/mapal
git status --short
openspec list --json
cat docs/performance/gpu/current-results.md
cat /private/tmp/mapal-s61-root/live-state.json
cat /private/tmp/mapal-s61-gpu/attempt-10/packet.json
```

Completed competitive trace: `/private/tmp/mapal-s61-gpu/attempt-09/retrieved/trace/`.
Persistent private recovery: `.benchmark-artifacts/gpu/competitive-cutlass/rtx-5090/2026-10-22/`.
The [ordinary recovery receipt](../benches/results/gpu/kernels/competitive-cutlass/rtx-5090/recovery.json)
binds its source/full-trace archives; all337ordinary files are also preserved in the public
raw archive. Original older traces remain under attempts02/03/05 and their historical
archive; [older recovery](../benches/results/gpu/operations/performance-comparison/recovery.json).
Historical session logs retain original paths; the migration map gives current locations.

Latest raw native/control evidence: `benches/results/gpu/workloads/residual-mlp/fused-references/rtx-5090/`. Persistent private full trace: `.benchmark-artifacts/gpu/residual-mlp/fused-references/rtx-5090/2026-10-22/`. No GPU work remains for this admission. Callback prototype is unsealed under `/private/tmp/mapal-library-image-capture/`;32 focused CPU cases pass, but no live library image has been admitted.
