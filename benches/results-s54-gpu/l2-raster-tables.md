# S54 `l2-raster`: grid order and L2 window, five parts

Branch `s54/l2-raster`, 2026-10-21. vast.ai, one box per session, every box destroyed. Ratios are medians; kernel ratios are normalized by each bench's own cuBLAS leg. Session files: `l2-raster-<part>[-<session>].txt`. Static counts: `l2-raster-fusion-report.txt`.

## Shipped table (`GPU_RUNGS_OFF`)

| part | `raster` (derived: grouped where one block's panels leave the L2) | `l2-gate` | `l2-window` |
|---|---|---|---|
| RTX 3090 | **OFF**: no face faster in both shipped-kernel sessions | ON | **OFF**: no probe operand fits 6 MB, unmeasured |
| A100 | ON (gate closed at every size: row-major) | ON: ungated grouping never faster | **OFF** |
| RTX 4090 | ON (gate closed: row-major) | ON: ungated never faster | **OFF**: costs up to 7% on contract |
| H100 PCIe | ON (gate closed: row-major) | ON: ungated never faster | **ON contract**, OFF exact |
| RTX 5090 | ON | **OFF on contract** (grouped at 4096 and 8192, FMA and TF32); ON exact | **OFF** |

Grouping now ships on one part, the RTX 5090, and only on the contract faces. The L2 window ships on one part and face, H100 PCIe contract.

## The derivation and where it was wrong

`Gpu::raster` picks h = the power of two dividing `gy` that minimizes the panels a wave reads, `h·bm + ⌈W/h⌉·bn`. It keeps row-major while one block's panels over k, `(bm+bn)·k·elem`, fit `l2_bytes`.

| part | L2 | W (blocks/wave) | derived h at 2048 / 4096 / 8192 | ungated h | shipped h |
|---|---:|---:|---|---|---|
| RTX 3090 | 6 MB | 164 | 1 / 1 / 16 | 16 / 16 / 16 | 1 (table) |
| A100 | 40 MB | 216 | 1 / 1 / 1 | 16 / 16 / 16 | 1 |
| RTX 4090 | 72 MB | 256 | 1 / 1 / 1 | 1 / 16 / 16 | 1 |
| H100 PCIe | 50 MB | 228 (FMA), 114 (wgmma) | 1 / 1 / 1 | 16 (FMA), 8 (wgmma) | 1 |
| RTX 5090 | 96 MB | 340 | 1 / 1 / 1 | 1 / 16 / 16 | 16 at 4096/8192 contract (table) |

- **The gate predicted the 3090 and missed.** The 3090's grouped 8192 kernel measured 0.986–1.029 (exact), 0.992–1.017 (FMA) and 0.956–0.991 (TF32) against row-major across both shipped-kernel sessions.
- **It excluded the 5090, which gains.** Ungated grouping on the 5090 at 8192 measured TF32 1.031 / 1.054 / 1.014 in three sessions.
- The plan's literal formula, `(h·bm + wave_cols·bn)·k·elem ≤ l2_bytes/2`, has no solution on any part at 8192, so it was not built.

## Grouped vs row-major, every session (>1 = grouped faster)

`*` marks disjoint ranges. Registers are grouped/row-major, with `+N` bytes spilled. The sessions are labeled by kernel variant:

- **pre-fix**: the first build. The remap's known-bits analysis turned address `add`s into `or`s, which cost the H100 FMA tile 18 registers.
- **shipped**: the opaque remap, which is what ships.
- **epilogue**: the shipped remap plus an epilogue re-read of `ctaid`. It was reverted because the 5090 FMA tile spilled 16 B.

| part | session (variant) | arm | n | exact | contract (FMA) | TF32 |
|---|---|---|---:|---:|---:|---:|
| RTX 3090 | run1 (pre-fix) | raster | 8192 | 0.987 r128+24/r128 | 1.068 r218/r218 | 0.987 r195/r194 |
| RTX 3090 | run2 (shipped) | raster | 8192 | 0.986 r128+24/r128 | 0.992 r218/r218 | 0.991 r195/r194 |
| RTX 3090 | run3 (shipped) | raster | 8192 | 1.029 r128+24/r128 | 1.017 r218/r218 | 0.956 r195/r194 |
| RTX 3090 | run4 (epilogue) | raster | 8192 | 1.028 r128+24/r128 | 1.132 r218/r218 | 0.922 r191/r194 |
| RTX 3090 | run2 (shipped) | ungated | 2048 / 4096 | 0.976* / 0.976 | 1.001 / 0.998 | 0.997 / 0.999 |
| RTX 3090 | run3 (shipped) | ungated | 2048 / 4096 | 0.977 / 0.980 | 0.997 / 0.992 | 1.019 / 1.002 |
| A100 | run2 (shipped) | ungated | 2048 / 4096 / 8192 | 0.996 / 1.000 / 0.998 | 0.956* / 0.970 / 0.962 (r137/r128+24) | 0.995 / 1.006 / 0.977 |
| A100 | run3 (shipped) | ungated | 2048 / 4096 / 8192 | 0.995 / 1.000 / 0.999 | 0.957* / 0.970* / 0.966* | 1.000 / 0.994 / 1.009 |
| A100 | run4 (shipped) | ungated | 2048 / 4096 / 8192 | 0.990* / 1.000 / 0.998 | 0.957* / 0.970* / 0.966* | 0.997* / 0.995 / 1.008* |
| A100 | run1, confirm (pre-fix); run5 (epilogue) | ungated | 8192 | 0.993 / 0.993* / 0.995 | 0.965 / 0.970* / 0.966* | 0.968 / 1.004 / 1.002 |
| RTX 4090 | run2 (shipped) | ungated | 4096 / 8192 | 0.973 / 0.977 (r255 both) | 1.000 / 1.007 | 0.995 / 0.994 |
| RTX 4090 | run4 (epilogue) | ungated | 4096 / 8192 | 0.953 / 0.907 (r255+24/r255) | 0.997 / 1.017 | 1.002 / 1.011 |
| H100 PCIe | run2 (shipped) | ungated | 2048 / 4096 / 8192 | 0.973* / 0.974* / 0.973* (r128+8/r128) | 1.006 / 1.003 / 0.996 (r127 both) | 0.996 / 1.003 / 1.000 |
| H100 PCIe | run3 (shipped) | ungated | 2048 / 4096 / 8192 | 0.979* / 0.973* / 0.965 (r128+8/r128) | 1.002 / 1.002 / 1.002 (r127 both) | 1.002 / 0.883 / 1.006 |
| H100 PCIe | run1 (pre-fix) | ungated | 2048 / 4096 / 8192 | 0.977* / 0.973* / 0.962 | 0.788* / 0.772* / 0.747* (r145/r127) | 0.996 / 1.036 / 1.001 |
| RTX 5090 | run2 (shipped) | ungated | 4096 / 8192 | 1.009 / 1.019 | 1.007 / 1.020 | 0.999 / 1.031 |
| RTX 5090 | run3 (shipped) | ungated | 4096 / 8192 | 1.005 / 1.052 | 0.995 / 1.043 | 0.997 / 1.054 |
| RTX 5090 | final (shipped + table) | raster | 4096 / 8192 | 0.987 / 0.999 | 0.973 / 1.003 | 1.001 / 1.014 |
| RTX 5090 | run4, run5 (epilogue) | ungated | 8192 | 1.027 / 1.038 | 0.969 / 0.992 (r128+16/r128) | 1.027 / 1.061 |
| RTX 5090 | run1 (pre-fix) | ungated | 8192 | 0.994 | 0.935 (r128+16/r128) | 1.014 |

Raw milliseconds and min–max ranges are in the session files (`python3 benches/gpu/table.py <file>`).

## `l2-window`: traced kernel time, window off / on (>1 = window faster)

| part | probes × faces | sessions | range | > 1.01 in every session |
|---|---|---:|---|---|
| A100 | mlp, mlp2, diamond at 2048 × 2 (none fits at 4096) | 6 | 0.926–1.025 | none |
| RTX 4090 | 2048, 4096 × 2 | 2 | 0.933*–1.040 | none; contract costs 6–7% (diamond_2048_c 0.933*/0.939*, mlp2_2048_c 0.938*/0.949) |
| H100 PCIe | 2048 × 2 | 3 | 1.000–1.016* | mlp2_2048_c 1.016*/1.012*/1.015*; exact 1.000–1.008* |
| RTX 5090 | 2048, 4096 × 2 | 5 | 0.976–1.005 | none |
| RTX 3090 | — | — | — | no probe operand fits 6 MB |

| H100 PCIe probe | face | run1 | run2 | run3 |
|---|---|---:|---:|---:|
| diamond_2048 | contract | 1.005 | 1.006* | 1.007* |
| diamond_2048 | exact | 1.007* | 1.004* | 1.008* |
| mlp2_2048 | contract | 1.016* | 1.012* | 1.015* |
| mlp2_2048 | exact | 1.000 | 1.002 | 1.000 |
| mlp_2048 | contract | 1.016* | 1.007* | 1.011* |
| mlp_2048 | exact | 1.000 | 1.000 | 1.001 |

Device facts from the boxes, in bytes:

| part | `l2_bytes` | persisting set-aside | largest window |
|---|---:|---:|---:|
| RTX 3090 | 6291456 | 4325376 | 134213632 |
| A100 SXM4 | 41943040 | 26214400 | 134213632 |
| RTX 4090 | 75497472 | 51904512 | 134213632 |
| H100 PCIe | 52428800 | 32768000 | 134217728 |
| RTX 5090 | 100663296 | 62914560 | 134217728 |

## Kernels as shipped vs cuBLAS / CUTLASS (>1 = Mapal faster)

`†` marks rows the table changes: on the 3090 the row-major arm, on the 5090 the grouped arm.

| n | RTX 3090 exact / contract / TF32 (run2) | A100 (run3) | RTX 4090 (run2) | H100 PCIe (run2) | RTX 5090 (run3) |
|---:|---|---|---|---|---|
| 1024 | 0.62/0.55 · 0.86/0.78 · 0.89/0.89 | 0.34/0.51 · 0.63/0.94 · 0.57/0.60 | 0.63/1.08 · 0.79/1.35 · 0.78/1.21 | 0.46/0.62 · 0.65/0.88 · 0.57/0.80 | 0.84/1.05 · 0.97/1.25 · 0.79/1.37 |
| 2048 | 0.50/0.58 · 0.88/1.02 · 0.91/0.92 | 0.42/0.49 · 0.80/0.95 · 0.69/0.71 | 0.62/0.60 · 1.06/1.03 · 0.89/0.90 | 0.48/0.61 · 0.84/1.07 · 0.61/0.95 | 0.59/0.64 · 0.87/0.96 · 0.82/0.87 |
| 4096 | 0.58/0.59 · 0.97/0.99 · 0.99/0.93 | 0.47/0.49 · 0.91/0.96 · 0.66/0.72 | 0.59/0.60 · 1.05/1.06 · 0.91/0.93 | 0.59/0.63 · 1.03/1.10 · 0.67/1.08 | 0.59/0.64 · 0.91/0.99† · 0.89/0.88† |
| 8192 | 0.53/0.61† · 0.91/0.99† · 0.91/0.90† | 0.50/0.50 · 0.96/0.96 · 0.65/0.73 | 0.56/0.57 · 1.03/1.04 · 0.93/0.95 | 0.53/0.64 · 0.95/1.13 · 0.56/1.19 | 0.59/0.71 · 0.89/1.06† · 0.84/0.92† |

Each cell is vs cuBLAS / vs CUTLASS, for exact · contract · TF32.

- **Losses:**
  - The exact face is 0.34–0.84× cuBLAS everywhere.
  - TF32 is 0.56–0.69× on the A100 and H100.
  - Contract at 1024 is 0.63–0.97×.
  - A100 contract is 0.63–0.96×.
- **The 3090 at 8192:** the row-major contract kernel is 0.91× on run2's box and 0.79× on run3's and run4's (the baseline's 0.84 drop). Grouping measured +1.7% (run3) and +13.2% (run4, the epilogue variant) on those slow boxes, and −0.8% on run2's.

## Multi-computation probes vs the library legs (contract face)

| part (session) | probe @ 4096 | Mapal iter ms | Mapal kernel ms | best library ms | lib / Mapal iter | lib / Mapal kernel |
|---|---|---:|---:|---|---:|---:|
| RTX 3090 (run2) | mlp / mlp2 / diamond | 33.88 / 77.62 / 46.31 | 11.72 / 23.19 / 17.43 | lt 11.965 / lt2 24.114 / streams 18.506 | 0.35 / 0.31 / 0.40 | 1.02 / 1.04 / 1.06 |
| A100 (run3) | mlp / mlp2 / diamond | 41.97 / 83.20 / 55.00 | 20.39 / 35.06 / 29.73 | lt 14.520 / streams 29.044 / lt 21.779 | 0.35 / 0.35 / 0.40 | 0.71 / 0.83 / 0.73 |
| RTX 4090 (run2) | mlp / mlp2 / diamond | 40.74 / 91.10 / 53.18 | 4.79 / 9.50 / 7.20 | lt 5.102 / lt2 10.081 / cutlass 7.679 | 0.13 / 0.11 / 0.14 | 1.06 / 1.06 / 1.07 |
| H100 PCIe (run2) | mlp / mlp2 / diamond | 66.29 / 51.42 / 79.26 | 8.39 / 16.77 / 12.56 | lt 8.611 / lt2 15.540 / streams 12.215 | 0.13 / 0.30 / 0.15 | 1.03 / 0.93 / 0.97 |
| RTX 5090 (final, grouped) | mlp / mlp2 / diamond | 19.12 / 40.26 / 22.84 | 4.21 / 8.35 / 6.31 | lt 4.168 / lt2 8.049 / streams 6.291 | 0.22 / 0.20 / 0.28 | 0.99 / 0.96 / 1.00 |

- **Whole chain:** 2.5–9× slower than the best library chain on every part. The loss is the host maps and copy-outs, which other S54 items address; this item moves no `Trm`.
- **5090 at 4096, all 11 device probes, grouped vs row-major:** traced kernel time 0.996–1.008, so grouping has no effect on the chains.

## Sessions

| part | file | GPU | vast host | kernels | differential | trap-before-site |
|---|---|---|---|---|---:|---:|
| RTX 3090 | `l2-raster-rtx-3090.txt` | RTX 3090 | 452472 (TH) | pre-fix | 80/80 | 4/4 |
| RTX 3090 | `-run2` | RTX 3090 | 24953 (US) | shipped | 80/80 | 4/4 |
| RTX 3090 | `-run3` | RTX 3090 | 443829 (US) | shipped | 80/80 | 4/4 |
| RTX 3090 | `-run4` (recovered from the box's stdout; the box died during the profile) | RTX 3090 | 443829 (US) | epilogue | 80/80 | 4/4 |
| A100 | `l2-raster-a100.txt`, `-confirm` | A100-SXM4-40GB | 80738 (NL), 591700 (US) | pre-fix | 80/80, 80/80 | 4/4, 4/4 |
| A100 | `-run2` | A100-SXM4-40GB | 224078 (SI) | shipped | 80/80 | 4/4 |
| A100 | `-run3`, `-run4` | A100-SXM4-40GB | 195410 (US), the same machine | shipped | 80/80, 80/80 | 4/4, 4/4 |
| A100 | `-run5` | A100-SXM4-40GB | 195410 (US) | epilogue | 80/80 | 4/4 |
| RTX 4090 | `-run2` | RTX 4090 | 87485 (ES) | shipped | 80/80 | 4/4 |
| RTX 4090 | `-run4` | RTX 4090 | 96250 (ES) | epilogue | 80/80 | **3/4** |
| H100 PCIe | `l2-raster-h100-pcie.txt` | H100 PCIe | 415974 (US) | pre-fix | 80/80 | 4/4 |
| H100 PCIe | `-run2` | H100 PCIe | 670790 (ZA) | shipped | 80/80 | 4/4 |
| H100 PCIe | `-run3` | H100 PCIe | 364425 | shipped | 80/80 | 4/4 |
| RTX 5090 | `l2-raster-rtx-5090.txt` | RTX 5090 | 229807 (EE) | pre-fix | 80/80 | 4/4 |
| RTX 5090 | `-run2`, `-run3` | RTX 5090 | 95374 (VN), 43746 (US) | shipped | 80/80, 80/80 | 4/4, 4/4 |
| RTX 5090 | `-run4`, `-run5` | RTX 5090 | 80057 (KR), 154140 (ZA) | epilogue | 80/80, 80/80 | 4/4, 4/4 |
| RTX 5090 | `-final` | RTX 5090 | 127708 (KR) | shipped + table | 46/46 | 2/2 |

Differential coverage:

- Every program is checked on both faces and every arm against the interpreter oracle.
- `deep_4096x1024x8192` covers the grouped order on the 3090 (h = 16, two groups).
- `wide_8192x1024x1024` covers it on the 5090 in the final session.
- `tile_bench` checks every kernel arm bit-exact (exact, contract) or within the TF32 bound.

**Trap failure, 4090 run4:** `e2e_trap.off-l2-window` exited 102, not 101, after printing the trap. That arm's host object is s54-base's text. The likely cause is a race between the eager-init thread and the trap's exit.
