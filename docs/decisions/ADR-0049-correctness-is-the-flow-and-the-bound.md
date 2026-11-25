# ADR-0049: Correctness is the flow and the bound — bit-identity is a development tool, not a language promise

Date: 2026-11-26 · Status: **accepted** (decided by the user).
Builds on: ADR-0040 (the default float face: full precision, a-priori bound, order free). Amends the framing of
ADR-0040 D1/D3 (the exact / contract / fixed faces), ADR-0042 D-items and ADR-0046's contract ("bit for bit on the
exact faces").

## Context

The tensor-tile stream-K walk (`mma-walk`, 2026-11-25) combines partial sums with atomic adds in arrival order. Floating
point addition is not associative, so the last bits of some outputs differ between runs, and greedy decoding of the
readable Llama in the bf16 class produced different token sequences on repeated runs at 512 tokens — every one within
ADR-0040's a-priori bound. The question was whether to make the default run-to-run deterministic (a fixed combine order
with a workspace, at some cost) or to keep free order.

The user's decision (2026-11-26): "correctness is achieved not by a deterministic result, but a correct flow of
operations"; "this should override the rule of bit-identical, this is not something that is good to pursue … it allows
for more optimization and correctness where it matters; determinism should be in development, not in a language that
tries to create optimal, correct implementation and placement based on dataflow". Parallel reductions behave this way
in every language and on every device; LLM serving engines are not run-to-run deterministic either. On hardware, forcing
an order over dynamically scheduled units costs buffers and ordering logic (a fixed, wired reduction tree is
deterministic for free and needs no rule).

## Decision

**D1 — What correct means.** A Mapal program is correct when its flow of operations is the program's dataflow and every
value lies within the stated a-priori bound (ADR-0040 D2, ADR-0042/0046 contracts on the default face). That is the
language's whole numerical promise.

**D2 — Order is the compiler's.** The compiler orders, splits, merges and combines operations freely for performance
and placement, including combines whose order depends on run-time arrival (atomics, stream-K, split-K, persistent
walks). Results may differ in the last bits between runs and between targets. No optimization is refused, priced up or
restricted to reproduce bits, and no target (CPU, GPU, ASIC) is required to provide deterministic reductions.

**D3 — Determinism is a development tool.** The exact, contract and fixed modes (ADR-0040 D1, D3) remain, renamed in
the documentation as **development modes**: they exist to test and debug — the interpreter as the oracle, compiled
orders replayed bit for bit, generated-program value gates, reproducing a failure. They are not part of the language's
contract and are never a goal the compiler optimizes for. ADR-0042's and ADR-0046's "bit for bit on the exact faces"
clauses describe these development modes.

**D4 — Claims.** Documentation, the README and presentations describe Mapal's numerics as "correct within a stated
bound; operations ordered freely; bit-reproducible runs available as a development mode" — never "identical bits on
every backend" as a language property.

## Alternatives rejected

- **Run-to-run deterministic default** (fixed combine order through a workspace): trades speed and memory on every
  parallel combine for a property that does not make results more correct; on hardware it costs ordering logic.
- **Bit-identity across targets as a language guarantee** (the earlier framing of the exact faces): constrains
  placement and realization choices per target and pushes the language toward one fixed execution order — the opposite
  of deriving the best schedule from the dataflow.

## What would falsify it

A class of programs whose results, within the stated bound, are still wrong in a way users must care about (e.g. a
bound too loose to be meaningful for a common workload) — then the fix is a tighter bound or a better realization, not
a determinism rule.

## The interpreter

Unchanged: it remains the reference semantics and the development oracle (exact mode, replayed orders). The default
face's value gates keep checking the bound, never bits.
