# Allocation latency after unused-output elimination

Forced-stream realization, before (`predce`) vs after (`fs`): identical PTX, host objects differ only in the omitted allocations/frees. Ratio = before/after on the program's own iteration clock (>1: after faster), `s1 / s2` = two sessions on different boxes, 12 interleaved repetitions per arm and face. Sign: beyond 1 ± the session's largest unchanged-code `null/on` deviation in both sessions; `*` disjoint ranges in both.

| Part | Sessions | Affected cells, iteration clock (+/−/cells) | Affected cells, request clock incl. alloc+free (+/−/cells) | Unchanged controls (+/−/cells) | Session floor s1 / s2 |
| --- | --- | ---: | ---: | ---: | ---: |
| RTX 3090 | 2 boxes each, merged | 12/0/14 | 9/0/14 | 1/0/18 | 0.050 / 0.025 |
| RTX 4090 | 1 box each | 7/0/14 | 3/0/14 | 0/0/18 | 0.190 / 0.189 |
| H100 | PCIe; SXM (3 boxes, merged) | 5/0/14 | 9/0/14 | 0/0/18 | 0.109 / 0.034 |
| RTX 5090 | 1 box each | 11/0/14 | 11/0/14 | 0/0/18 | 0.038 / 0.012 |

Allocation requests per process (traced): independent 7 → 6, diamond 8 → 7, two-MLP 14 → 12; twin 6 → 6 and single-chain MLP 1 → 1 (controls, byte-identical host objects). The RTX 3090 control that reads + is `twin_512` Exact, 1.06 / 1.13, on identical binaries.

| Cell (Contract = c) | RTX 3090 | RTX 4090 | H100 | RTX 5090 |
|---|---|---|---|---|
| diamond_1024 c | 1.33 / 1.13 + | 1.18 / 1.27 0 | 1.15 / 3.17 + | 1.20 / 1.20 + |
| diamond_1024 | 1.10 / 1.11 + | 1.17 / 1.19 0 | 1.11 / 4.02 + | 1.15 / 1.03 + |
| diamond_512 c | 1.53 / 1.13 + | 1.00 / 1.00 0 | 1.06 / 1.15 0 | 0.97 / 1.01 0 |
| diamond_512 | 1.84 / 1.09 + | 0.98 / 0.93 0 | 0.99 / 1.10 0 | 0.99 / 1.01 0 |
| indep_1024 c | 1.37 / 1.17 + | 1.23 / 1.22 +* | 1.22 / 0.40 0 | 1.22 / 1.28 + |
| indep_1024 | 1.14 / 1.16 + | 1.19 / 1.19 0 | 1.22 / 2.09 + | 1.20 / 1.08 + |
| indep_512 c | 1.71 / 1.34 + | 1.41 / 1.28 + | 1.37 / 1.02 0 | 1.50 / 1.42 + |
| indep_512 | 1.25 / 1.41 + | 1.43 / 1.43 + | 1.45 / 1.55 + | 1.49 / 1.46 +* |
| mlp2_1024 c | 1.66 / 1.17 + | 1.22 / 1.21 +* | 1.19 / 0.47 0 | 1.26 / 1.34 + |
| mlp2_1024 | 2.42 / 1.15 + | 1.22 / 1.20 +* | 1.17 / 1.34 + | 1.19 / 0.86 0 |
| mlp2_2048 c | 0.76 / 1.14 0 | 1.17 / 1.18 0 | 1.10 / 1.02 0 | 1.12 / 1.23 + |
| mlp2_2048 | 1.56 / 1.10 + | 1.17 / 1.15 0 | 1.06 / 1.13 0 | 1.10 / 1.09 + |
| mlp2_512 c | 1.22 / 1.16 + | 1.21 / 1.21 + | 1.20 / 0.83 0 | 1.16 / 1.25 + |
| mlp2_512 | 0.94 / 1.20 0 | 1.20 / 1.23 + | 1.20 / 0.40 0 | 1.19 / 1.16 + |

Shipped policy (streams OFF on every measured part): the emitted text is unchanged by the elimination, and the arena still reserves the omitted raw outputs' bytes (`px_mlp2_128` 525,312 B before and after; 459,776 B with a `store_out` member filter, not applied).

[Raw results and per-part tables](../../../../benches/results/gpu/s62/runtime-measurements/) · [Allocation counts](dead-output-allocation.md) · [All memory results](README.md)
