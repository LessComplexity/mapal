# Plan S49 — a linear NN at scale, and two nets at once

Written: 2026-08-09 · component: cross (mapal-ir `path_plan` · backend-llvm · mapal-rt)
Requested by Sapir: *"linear NN, 3–4 layers, numpy on CPU vs mapal, on the arch box and
this machine; measure how it performs in scale; and run 2 nets at the same time — 2-layer,
2 inputs — to see how mapal gives an advantage as everything is auto-parallelised."*

## 1. Why model this categorically first

A linear NN forward pass is a **chain in `Alg`** (FRAMEWORK §4.1): `t_to(layerᵢ) =
t_from(layerᵢ₊₁)`. That single equation is the whole of "3–4 layers", and it is also the
reason the depth-4 net cannot overlap with itself — a chain has no two independent arrows.

Two nets is the **coproduct** of two such chains over disjoint data: no morphism of chain A
reads an object of chain B. So `depends-on` has two connected components, and in
`path_plan`'s task DAG both roots have `deps_left == 0` and are seeded together
(`mapal_par_launch` seeds *every* initially-ready task, `mapal-rt/src/lib.rs:891`). The
"auto-parallelism" claim is therefore not a new feature to build — it is a **prediction the
existing machinery already makes**, and the only honest thing to do is measure whether it
holds.

That gives the experiment its control. Same total FLOPs, two dependency structures:

| program | shape | independent chains | can overlap |
| --- | --- | --- | --- |
| `nn4` | X·W₁·W₂·W₃·W₄ | 1 (depth 4) | no — by construction |
| `nn2x2` | (X₁·W₁ᵃ·W₂ᵃ) and (X₂·W₁ᵇ·W₂ᵇ) | 2 (depth 2 each) | yes |
| `nn2` | X·W₁·W₂ | 1 (depth 2) | half the work — the unit |

`nn4` is the **negative control** (rule 22): same 4 matmuls, zero available concurrency. If
`nn2x2` gains and `nn4` does not, the gain is the DAG and not the machine.

## 2. Model

`Dat` — `M(d) = [f32; d*d]`, one object per activation/weight. `Trn` — `layer : M × M → M`,
the map+fold matmul site (`tile_site` recognises it; the corpus twin is
`benches/shapes/attn_256.mapal`, which is already a *chained* pair). `Loc` — host spine +
pool lanes (M4 Pro: 10P+4E; i9: 32 threads). `Trm` — the frame slot each task writes and
its dependent reads; `mapal_par_dep` edges are the reified dependency.

Morphisms, `nn2x2`:

| Morphism | Signature | Partiality | Semantics |
| --- | --- | --- | --- |
| `h1a` | `M × M → M` | Total | `x_a · w1a` |
| `h2a` | `M × M → M` | Total | `h1a · w2a` |
| `h1b` | `M × M → M` | Total | `x_b · w1b` |
| `h2b` | `M × M → M` | Total | `h1b · w2b` |

Composition rule: `{h1a,h2a} ∩ {h1b,h2b} = ∅` on reads — the disjointness that makes the
two chains a coproduct. Verified by emission (two independent `mapal_par_dep` chains, no
cross edge), not asserted.

## 3. What gets measured

**Metric — overlap factor** `Ω = 2·T(nn2) / T(nn2x2)`, per leg.
`Ω = 1` ⇒ the two nets ran serially. `Ω = 2` ⇒ perfect overlap. Reported per size, per leg,
per machine. numpy (sequential source) is pinned at `Ω ≈ 1` **by construction** — that is
the baseline the claim is against, and it is what a person writing the obvious numpy gets.

**Scale sweep** `d ∈ {128, 256, 512, 1024, 2048}`, square: batch = width = `d`, every layer
`d×d`. One axis, trivial FLOP accounting (`nn4`, `nn2x2`: `8·d³`; `nn2`: `4·d³`). Square is
a simplification, stated rather than hidden: a real net has batch ≠ width.

**Legs** — mapal `conf`/`fma` × `1t`/`par`, numpy `1t`/`mt`. The `fma` (`--contract`) face
is the like-for-like one against BLAS (S36c: every pre-S36c Mapal number was the FMA-free
conformance face while numpy got FMA from BLAS); both are reported.

**Machines** — M4 Pro locally (`--target=apple-m4-sme`, `clang -march=armv8-a+sme2`) and
the arch i9-14900F over the `i9_ladder.sh` split (emit + `.ll → .o` here, link and run
there; the box has no clang), `--target=raptorlake`, pinned `taskset`.

## 4. Value gate before any timing

Inputs are **positive small integers** (`(t·a + b) % 9 + 1`), not the tree's usual
`% 101 - 50`. Reason: a depth-4 f32 chain over mean-zero data cancels catastrophically, so
mapal's k-order fold and BLAS's blocked order could disagree by percent while both are
correct — a gate that cannot distinguish a real defect from summation order is not a gate.
All-positive removes cancellation; agreement then lands near f32 epsilon and the gate has
teeth. Magnitude check: `d=2048` reaches ~5.5e16, `d=4096` ~8.8e17, both far under f32 max.

Gate: mapal-conf is the reference; numpy and mapal-fma compared by relative error, and the
**observed maximum is reported**, never just "passed".

## 5. Steps

1. `benches/nn/nn_gen.py` — emit `nn4_{d}`, `nn2_{d}`, `nn2x2_{d}` `.mapal` sources.
2. `benches/nn/nn_numpy.py` — same generator arithmetic, same printed values, `iter ms=`.
3. Emission check: each program's tile-site count == its layer count; `nn2x2` shows two
   independent dep chains. A program whose sites were not recognised is measuring the
   scalar fallback and must be a hard error, not a slow row.
4. `benches/nn/nn_ab.sh` — local, interleaved, value-gated, through `perflock.sh`.
5. `benches/nn/nn_i9.sh` — the remote split, same structure.
6. Report: absolute ms + GF/s per cell, `Ω` per leg, both machines.

## 6. What building found — the benchmark could not be run as written

**Every chained bulk program in this tree has been emitting fully sequential code, and
still would be.** `nn2`/`nn4`/`nn2x2` emitted **zero** `mapal_par_*` calls at every size;
so does `benches/shapes/attn_256.mapal`, the project's own chained-matmul corpus shape.
A single matmul emits 70.

Root cause, read off the plan dump rather than guessed (`nn2x2_128`, pre-fix):

```
task  8: Seq x16  deps=[…, 11, 14]   16 capture Pairs @476 @633 @791 @950
task 11: Split    deps=[0, 8]        layer-1 map @476
task 14: Split    deps=[0, 8]        layer-1 map @791
```

`path_plan` grouped scalar morphisms into **undirected** connected components of the
"shares an object in any role" graph. The four matmul sites each build a capture tuple,
and all four tuples capture **the same `k` range** — so one shared *read* fused four
independent capture groups into one task. Then layer 2's capture group reads layer 1's
output, so that single task is **after** the layer-1 maps while every map is **before** it:
`8 → 11 → 8`. A cycle.

Nothing crashes. `crates/backends/llvm/src/lib.rs:261` filters a cyclic plan out with
`path_plan_is_acyclic` and emits the sequential form — so the symptom is *silently lost
parallelism*, and `debug_assert_eq!(task_topo.len(), tasks.len())` (`algo.rs:2050`), which
does detect it, is compiled out of every release build. It fires in debug: `left: 6,
right: 11`.

**Fix (`algo.rs`, one adjacency expression).** Component adjacency is dataflow plus
co-writing, never sibling-reading:

| edge | keep? | why |
| --- | --- | --- |
| `in_edges(source)` | ✔ | who produced my input — a chain |
| `out_edges(target)` | ✔ | who reads my output — a chain |
| `in_edges(target)` | ✔ | co-writers of one object are **one** producer (product slots) |
| `out_edges(source)` | ✘ | sibling *readers* of one object have no order between them |

Two morphisms that merely read one value are siblings, not a chain; merging them merges
work with no order between it, which is what let one task straddle a bulk task. Post-fix
`nn2x2_128`'s plan is acyclic and reads exactly as §2 predicts: `8→14→9→15` and
`10→18→11→19`, **no edge between the two chains**.

This is not confined to the new sources — `example_fanout`'s golden moved from one task to
two, which is the same defect in the tree's own example corpus.

## 7. What would refute the claim

`Ω ≈ 1` for mapal-par at every size — the DAG exists but the pool does not overlap the
chains. That is a real possible outcome and gets published as one if it happens.
Second refutation: `Ω > 1` for numpy too (BLAS internal threading absorbing the second net),
which would make the advantage a threading artifact rather than a scheduling one.
