## Context

See proposal.md. The frozen five-profile matrix is running from `b4152c5`; this extension
must preserve its source, tools and inputs. Source-grounded readiness inspections are
`/private/tmp/mapal-s56-gpu/large-shapes-readiness.md` and `composed-llm-readiness.md`.
Their findings below select the approach; proposed placements still require emission.

`tile_bench.cu:main` already accepts M/N/K and checks 256 sampled outputs per leg in
O(256K). Its explicit GPU allocation sum is `4*(MK+KN+3MN)+256MiB`, or 5.25GiB for
16K square, before driver/library overhead. The generator, square runner and square
report keys do not expose that rectangular support. `gen_tiles.py:decode` is an
unnormalized recurrence; it does not implement transformer softmax/masking. Core supports
f32/f64; the reduced face is TF32, not BF16/FP8 storage.

## Goals / Non-Goals

**Goals:** measure large and rectangular primitive kernels; test several operations in one
program; distinguish inferred retained computation from hand-maintained incremental state;
compare matched naive, handwritten-library and PyTorch realizations with reproducible
correctness, latency and memory evidence.

**Non-Goals:** full model execution, model loading, new numeric types, new compiler
optimizations, a new rental system, guaranteed speedups, or pretending a synthetic
recurrence is normalized attention. Compiler defects require their own evidence/fix.

## Decisions

1. **Separate immutable scope.** Keep supplemental manifests/results apart from the frozen
   five-profile matrix. Preparation may use existing frozen tools on new temporary files;
   repository source/tool changes wait for baseline hardware gates. Hash before/after
   source/tool inputs. Preserve failures and unsupported realizations.
2. **Reuse the existing primitive instrument.** Generate corner-only affine matmul sources
   following `gen_tiles.py:program`/`bench`, then reuse the emit/opt/llc flags from
   `bundle.sh:emit_one` and the unchanged `tile_bench M N K` CLI. A full 16K-cubed CPU
   interpreter oracle is unnecessary. Existing sampled ordered-f32, ordered-FMA and TF32
   gates remain mandatory and explicitly sampled. Inspect actual entry/grid/block/shared
   memory/PTX for every shape; small M may take a scalar fallback. Preserve null controls.
3. **Pin shape and memory meaning.** Start with 16384-square and `(K,N)` pairs
   `(7168,18432)`, `(7168,2048)`, `(4096,1024)`, with M in `1,16,128,2048,8192`.
   Published config revisions are in `s57-model-configs/receipt.json`. These are individual
   projection shapes, not complete models or routed MoE execution. A 32K square is an
   optional later capacity case (20.25GiB explicit harness allocation); never admit it
   from the size of one matrix alone. Key reports by M/N/K/layout/profile/face/cache policy.
4. **Use two complementary compositions.** A matrix program computes
   `p=XWo; h=ReLU(pWup+b1); m=(hWdown)+b2; y=.25*(p+m)` to test projections, fusion and live
   storage. A recurrent program updates one X row, computes `K=XWk`, `V=XWv`,
   `s=qKᵀ/T`, `a=sV`, then the same projected residual MLP; y feeds the next row.
   Preserve these explicit stages and addition order in every reference; do not reassociate
   the residual and bias. This second program exercises possible delta retention. Begin with full small-state
   checks at T16/d8/h16, then T1024/d256/h512. Larger compositions require their own
   capacity/oracle checks; primitive validation alone does not validate a recurrence.
5. **Bound the data without hiding branches.** Use fixed modulo-17 codes in [-8,8],
   weight divisor `32*input_width`, q divisor `32*d`, bias divisors 128/256, and the
   existing modulo-101 token law divided by 64. Pin all seeds in one fixture contract.
   The proposed real-arithmetic bound is `|y|<.013<1/16`; verify finite FP32 values and
   intermediate signals as well. Check K/V/a/p/h and final y so bias dominance cannot
   conceal a failed branch. Every implementation receives identical input f32 values.
6. **Make the controls explicit.** Mapal full recomputation disables BOTH `retain-delta`
   and `retain-resident`; `retained_sites` enables either path. Keep transmission-delta,
   fusion and allocation ablations separately named. Handwritten and PyTorch references
   each include full recomputation and explicit incremental K/V variants. Those explicit
   variants do not prove compiler discovery. Prove actual restricted launches, grouped
   stores, allocation positions and copies before saying multiple mechanisms composed.
7. **Compare actual PyTorch modes.** Record installed version/build, eager versus compiled
   backend/mode, fullgraph behavior, compile scope (step or sequence), graph breaks,
   recompiles, CUDA graph use and precision flags. Test compiled execution rather than
   counting a silent eager fallback. Use FP32 settings matching the declared comparison;
   record TF32 separately and do not substitute autocast/BF16/FP8. Compilation and graph
   setup are measured separately from steady execution. Pin flags to the installed
   version using its official documentation.
8. **Match clocks and state.** Report cold complete request/process, warm complete
   sequence/iteration wall and GPU-event region separately. End each measured region
   only after required GPU work completes. Match input residency, state reset, first
   token, output observation and hardware-cache protocol. Retained Mapal K/V has a full
   first pass; a handwritten zero-initialized cache can start with one row—retain this
   algorithmic difference in counts/results. Actual allocations, transferred bytes,
   requested live bytes and measured allocated/reserved peaks are distinct metrics.
   Reuse the primitive's cold-L2 protocol initially; composed resident operation has
   its own labeled protocol. Runtime profiling is separate from ordinary timing.
9. **Preserve numerical contracts.** Small Mapal Exact/Contract cases use their respective
   interpreter/CPU oracles; external parallel GEMMs use declared error checks and cannot
   promise Mapal's fold order. Define large composition checkpoint/full-output policy
   and tolerance before GPU timing, validate intermediate branches, and reject missing
   or failed values/statuses. Never relax a gate merely to publish a faster timing.

### Model and consolidation

No compiler/runtime Dat/Trn/Loc/Trm changes. Benchmark `Dat` are one shape/data contract,
variant configuration and measurement/evidence record. Planned generator/reference/runner
`Trn` map `(contract, implementation, environment)` to either validated observations or
an explicit failure/unsupported record. Host and GPU remain existing `Loc`; copies and
launches use existing ports. Extend/reuse `gen_tiles.py`, `tile_bench.cu` and
`chain_bench.cu` concepts rather than duplicate a CUDA arithmetic harness or rental layer.
New PyTorch code is benchmark tooling only.

FRAMEWORK §3 consolidation keeps dimensions/seeds/state reset in one shared contract;
§4.5 laws1/4 require actual placement/readiness evidence, law2 requires consistent
dimensions/types across references, and law5 requires full composed outputs to agree.
New runner/report touch sites are planned until implemented and mapped in docs.

## Risks / Trade-offs

- Tiny-M fallback or unsupported library shape → preserve emitted proof and unsupported
  outcome; do not claim a tensor path from the requested precision alone.
- Large intermediates/workspaces exceed capacity → calculate every explicit buffer,
  inspect free memory and record allocation failures; no implicit smaller-shape substitute.
- Recurrence or a final bias hides wrong work → bounded inputs plus intermediate/full
  small-state oracles; accept large results only under a declared checkpoint policy.
- Unequal cache/reset/clock boundaries manufacture a win → same state contract and
  separate cold/warm/device results, with emitted and runtime work counts.
- Compiled PyTorch silently runs eager or changes precision → explicit mode/error reporting
  and generated/executed-kernel evidence where available; no unsupported parity claim.
- Scope swallows compiler tuning → benchmark first, report losses, and route only proved
  compiler defects through the separate generic change.

## Migration Plan

Prepare independent temporary fixtures and bind their emitted artifacts; finish frozen
hardware validation before repository source changes. Implement the smallest generator,
reference and PyTorch extensions, then small correctness before large GPU measurements.
Run the 16K primitive increment first and report it, followed by matrix and recurrent
compositions. Reconcile benchmark/implementation docs and evidence, run drift and archive
only completed work. Existing frozen artifacts remain recoverable and unchanged.
