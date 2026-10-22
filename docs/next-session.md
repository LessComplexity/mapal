# Continue the GPU performance comparison

Start with [current results](performance/gpu/current-results.md), the [performance index](performance/README.md), and [raw evidence](../benches/results/README.md). Use descriptive topic/experiment names for new reports and folders. Session numbers belong only in historical provenance and immutable handoffs. The [path map](../benches/results/path-map.json) resolves relocated artifacts without changing their contents.

## Current state

- Branch `s54-gpu-fusion`, last committed head `29ad618`; integration, verified results and evidence reorganization are uncommitted. Preserve unrelated `oainotes.md`. Local commits are authorized; no push.
- The unused raw-output allocation correction is integrated. 640 ordinary GPU processes and 28 separate allocation-counter processes pass; 1,190 distinct local tests pass across the workspace and host-access backend rerun, with one ignored. No allocation latency gain is claimed.
- The residual MLP comparison admits all ten rows. cuBLAS is 1.35–1.47× faster than Mapal at the two M128 shapes. Mapal beats the restricted eager/graph-disabled Inductor arms. Input reset/cache state differs; no causal retention claim.
- The panel-depth alternative improves two small-M FMA cells and regresses Exact. It is not promoted. Four actual old/cooperative Mapal Contract cubin/SASS captures are admitted; library kernels remain to be captured.
- Topic reorganization moved 2,849 files. All 2,802 raw evidence hashes are unchanged, 372 report links resolve, and the baseline generator is equivalent apart from its descriptive title. Strict OpenSpec passes 7/0; drift is 0 dead / 575 references. The code graph was refreshed; document graph extraction needs an unavailable LLM key. Cross-project indexing encountered a PGLite lock, so the immutable repository log remains the record.

## Work in progress

The next experiment puts CUTLASS on competitive terms: twelve explicit FP32 SIMT tiles, full bounded gates and representative sampled gates against both scalar FP32 references, nine tuning samples, then thirty held-out confirmation samples with the original default control. Every arm shares inputs, precision budget, preallocated buffers and clock/cache protocol. Ordered-bit guarantees remain separate from numerical-budget admission.

The source review found that floating intermediate rounding made the proposed bound one ULP too small in some cells. This preparation was never run on GPU. The corrected exact-integer/rational computation independently verifies all 6,656 oracle cells and their minimal outward binary64 bounds. Do not use the rejected oracle preparation.

After ordinary admission, the same selected binary is reused for Nsight Compute 2025.1.1. First profile the admitted 16×2048×7168 Contract case: all three legs with LaunchStats and actual SASS exports, then supported occupancy/memory/scheduler/instruction sections if the smoke passes. Preserve all internal library kernels. Instrumented data cannot enter ordinary timing. Runtime report attribution is not a physical cubin-byte identity claim.

Then test supported cuBLASLt/CUTLASS fused epilogues on the same residual MLP, including unfused controls. Broader shapes, graph-enabled PyTorch, recurrence and causal retention/allocation comparisons remain open. Do not infer a fusion advantage or change compiler selection before evidence.

| Owner | Work | Private working directory |
| --- | --- | --- |
| Root | Integration, artifact review, result admission and shared docs | `/private/tmp/mapal-s61-root/` |
| `/root/composed_analysis_s61` | Competitive CUTLASS harness and composed reference extension | `/private/tmp/mapal-s61-competitive-reference/` |
| `/root/gap_analysis_s61` | Independent numerical audit and NCU runner/parser | `/private/tmp/mapal-s61-gap-analysis/ncu-extension/` |
| `/root/comparison_run` | Newly owned competitive comparison rental | `/private/tmp/mapal-s61-gpu/attempt-09/` |

The current task rental is **57256816**, created at **2026-10-22T18:49:03.684Z** on
machine **56076 / host 391071**, with operational deadline **19:44:03.684Z**. The ownership
guard checked the exact create-response ID, physical identity and fresh prelaunch exclusions
at 18:49:04.710Z before SSH/upload. Launcher session: **43594**. It is waiting for provider
startup; no new timing has been admitted. Root-reviewed packet SHA256:
`9786978505e5a580f7c11f5cd650ced54ea9e308fcf1807e870cd6856d885e1e`.

The previous three providers failed during startup before SSH, upload or GPU work.
Owned instances 57253902, 57254920 and 57256063 were independently
absent at 2026-10-22T18:28:15.478Z, 18:33:16.146Z and 18:43:05.966Z respectively. The first
reported an image-build lease error; the latter two returned unsuccessful creates with
empty status. Their normalized launch requests match the successful earlier run. Root
cause is unknown. Read-only diagnosis and an offline test of the installed CLI established
that `--cancel-unavail` changes only the scheduling cancellation field and requests an
explicit scheduling error instead of a stopped contract. That is the sole new create
option; payload and ownership guards are unchanged. No account cause or causal fix is
claimed. Never print raw API/create/wrapper logs.

The corrected NCUv3 archive is
`243dd5297c9a12e814b7a4e244372258cf756bbb34e239577e3287b11601c29b`.
It removes `--values-only`, which the actual harness forbids with profiling. The failed
hosts executed neither profiler version. Extracted real C++ guards now pass all9 valid
variant/leg combinations and reject6 negative modes/environment cases. Fourteen independent
NCU CPU tests also pass; live profiler behavior remains unproved.

The composed-reference worker is implementing supported cuBLASLt/CUTLASS fused and unfused
controls under `/private/tmp/mapal-composed-fused-references/`. No CUDA compile or GPU
admission exists yet. The [IR fact map](performance/gpu/disassembly/ir-facts-and-realization-gaps.md)
shows program facts already available and concrete backend limitations without inferring
a timing cause. Public destinations for new measurements remain
`benches/results/gpu/kernels/competitive-cutlass/` and
`benches/results/gpu/disassembly/library-kernels/`.

## Machines and authorization

The user explicitly removed the monetary cap and authorized equivalent RTX 5090 uploads/runs of the reviewed benchmark content. Do not ask for an already-granted budget approval. Use operational deadlines and retrieve evidence before exact-owned-instance cleanup.

**Protect every pre-existing instance.** Known unrelated IDs are 57065574, 57100404, 57126023 and 57133432; every ID in a fresh prelaunch inventory is also protected. Access/cleanup is permitted only for the exact new ID returned by this task's own create response, with the expected physical host. Inventory novelty alone is insufficient.

All eight earlier task rentals are independently absent: 57242062, 57245029, 57247103, 57248819, 57249550, 57253902, 57254920 and 57256063. The last was confirmed absent at 2026-10-22T18:43:05.966Z. The current owned instance is listed above. Provider API/create/wrapper logs remain private; never print raw provider errors or credentials. An earlier credential rotation remains outstanding.

## Inspect and resume

```sh
cd /Volumes/LessComplex/Personal/mapal
git status --short
openspec list --json
cat docs/performance/gpu/current-results.md
cat /private/tmp/mapal-s61-root/live-state.json
cat /private/tmp/mapal-s61-gpu/attempt-09/packet.json
```

Original completed traces remain under `/private/tmp/mapal-s61-gpu/attempt-{02,03,05}/retrieved/trace/`. Private recovery archives are in `.claude/worktrees/_archive-s54-20261021/s61-performance-analysis/`; use the relocated [recovery receipt](../benches/results/gpu/operations/performance-comparison/recovery.json). Historical session logs retain their original paths and names; the migration map provides their current equivalents.
