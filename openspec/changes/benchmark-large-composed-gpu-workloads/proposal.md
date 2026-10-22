## Why

The accepted selection measurements stop at 8192-square GEMMs and do not compare a
single composed Mapal program against PyTorch. Sapir requested 16K and LLM-relevant
shapes, plus evidence that dataflow-derived retention, fusion and allocation decisions
work together across operations, including comparisons with naive and handwritten code.

## What Changes

- Add a separate, provenance-bound benchmark extension with 16384-square GEMM and
  rectangular projections taken from verified published model configurations. Include
  small token counts and prefill-sized token batches; admit larger cases by actual
  capacity checks rather than assuming that one matrix's size is the total footprint.
- Test multiple operations in one Mapal program, with a supported attention/projection,
  MLP and residual composition selected after inspecting current lowering. Exercise
  invariant and changed-row flows so retained computation is demonstrated by emitted
  code and runtime work counts, not merely called a cache.
- Compare matched work against explicit full recomputation, handwritten CUDA/library
  implementations, and PyTorch eager/compiled modes. Preserve unchanged inputs, outputs,
  arithmetic policies and measurement boundaries; report unsupported modes and losses.
- Keep cold setup/compilation, ordinary complete-iteration time and device timing
  separate. Record allocation activity, peak memory and transfers alongside latency;
  reject timings whose value checks fail.
- Distinguish compiler-derived reuse from an explicitly programmed library KV cache,
  allocator caching and hardware caching. Large shape coverage does not establish full
  LLM or BF16/FP8 deployment parity.

## Capabilities

### New Capabilities

None. This adds benchmark tooling and evidence, not language/runtime behavior.

### Modified Capabilities

None. `.openspec.yaml` opts out of spec deltas. Compiler corrections, if evidence
requires any, remain separate proof-driven changes rather than benchmark-specific rules.

## Impact

Existing touch sites include `benches/gpu/tile_bench.cu` (independent M/N/K and sampled
reference checks), `gen_tiles.py`, `chain_bench.cu` and the bundle/run/report tooling.
Exact generator and composed-workload changes await the source-grounded readiness
inspection. PyTorch is a benchmark-environment dependency only; record its actual
version, compiler mode, precision settings and generated-kernel evidence.

The current five-profile plus independent-confirmation baseline remains frozen at
`b4152c5`. Its source, binaries, bundle inputs and manifests must not change. This
extension has separate inputs/results and does not redefine the existing acceptance
scope. Preparation may inspect existing code and use separate temporary fixtures;
repository benchmark changes follow completion of the frozen hardware gates.

Initial model-shape references are the official
[DeepSeek-V3.2 configuration](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/raw/main/config.json)
and [Qwen3.5-397B-A17B configuration](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/raw/main/config.json).
These establish dimensions and declared dtypes, not a claim that this benchmark
implements either model. Exact retrieved configurations, revisions and hashes are
preserved in `benches/results/gpu/workloads/model-shapes/receipt.json`.
