# Philosophy

**A dataflow programming language.** You write what the data is and how it flows; the compiler works out how
it runs: on the CPU, on the GPU, and across both. No threads, streams, kernels, tile sizes or intrinsics appear
in the source.

The [README](../README.md#the-ideas) says this in plain words; here it is with the decisions behind it.

## The idea

- **The language is the flow.** A program states data and how it flows, `data -> transform -> result;`, not the
  control that moves it. A `map` says every cell is independent, a `fold` says how values combine, a binding says
  who reads what.
- **Data is category-theoretic.** Values are objects, transforms are morphisms, a program is their composition.
  The compiler's IR is that graph, and its rewrites are the category's laws
  ([`docs/spec/category-ir.md`](spec/category-ir.md)).
- **Expressing the flow beats expressing control.** Because the source states only what depends on what, the
  compiler derives what runs in parallel, what merges into one kernel, what stays in registers, shared memory or
  cache, and what moves between devices, from facts on the graph (independence, reuse, death points, footprints)
  against the hardware's properties. Control written by hand fixes each of those decisions for one machine.

## What follows from it

- **Two kinds of facts, reconciled by the compiler.** The dataflow's facts (what each value reads, when it dies,
  what is independent) and the hardware's facts, as abstract compute and placement: where bytes can live (capacities,
  bandwidths, latencies, the cache a part really keeps), what computes (matrix and vector operations as shapes, with
  the place each operand must be in), what moves bytes, and how work is dispatched, each fact measured, queried or
  declared, never a part's name. Compilation places the dataflow on the machine with one cost model and one search,
  then lowers it to the hardware's language; the few hardware-specific steps that cannot be facts come last and are
  kept to a minimum. Adding a chip, a vendor or an ASIC is describing it as facts, not writing a new backend's worth
  of heuristics ([ADR-0050](decisions/ADR-0050-the-machine-model-placement-by-one-cost-model.md); its first
  step, matrix products, is in progress — today rules over the same physical properties decide).
- **No runtime graph recording.** Which work merges into one kernel, vertically along a chain and horizontally
  across independent branches, is decided at compile time from the dataflow. Nothing is traced or captured while
  the program runs.
- **Correct numerics: the flow and the bound, not bit-identity.** Every result is within an error bound stated before
  the program runs, at full precision ([ADR-0040](decisions/ADR-0040-default-float-face.md)). The compiler orders
  and combines operations freely for speed and placement, so parallel sums may differ in the last bits between runs —
  correctness is a correct flow of operations, not a repeated bit pattern
  ([ADR-0049](decisions/ADR-0049-correctness-is-the-flow-and-the-bound.md)). Bit-reproducible runs exist as a
  development mode for testing and debugging. Reduced precision is opt-in, per product.
- **Readable, idiomatic source.** Programs read like the math they compute. If natural code compiles worse than a
  hand-shaped version, the compiler is fixed, not the program.
