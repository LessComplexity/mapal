# retain-resident (S54 item 5) — decode_1024_256 on five parts, two boxes each

Files: `retain-resident-<part>.txt` (session 1: full bundle, differential 71/71 on every part) and
`retain-resident-<part>-confirm.txt` (session 2, another host: decode programs, differential 38/38).
Static and emulated counts: `retain-resident-fusion-report.txt`.

## Each rung OFF / ON (both faces pooled; > 1 = the rung pays)

`loop` = traced process wall − that process's `load_ms` (context + module, 170–790 ms, ±200 ms
between processes on the H100s), the decision metric; `wall` = untraced process wall.

| part | rung | session 1 loop | session 1 wall | session 2 loop | session 2 wall | ships |
|---|---|---:|---:|---:|---:|---|
| rtx-3090 | retain-resident | 1.053 | 1.054 | 1.035 | 1.045 | ON |
| rtx-3090 | delta-h2d | 1.143 | 1.113 | 1.210 | 1.169 | ON |
| rtx-3090 | lane-stride | 1.565 | 1.397 | 1.445 | 1.309 | ON |
| a100 | retain-resident | 1.091 | 1.070 | 1.037 | 1.030 | ON |
| a100 | delta-h2d | 1.135 | 1.120 | 1.095 | 1.026 | ON |
| a100 | lane-stride | 1.317 | 1.259 | 1.420 | 1.214 | ON |
| rtx-4090 | retain-resident | 1.025 | 1.011 | 0.981 | 0.988 | **OFF** (sign changes) |
| rtx-4090 | delta-h2d | 1.169 | 1.131 | 1.203 | 1.090 | ON |
| rtx-4090 | lane-stride | 1.608 | 1.430 | 1.590 | 1.465 | ON |
| h100-pcie | retain-resident | 1.055 | 1.057 | 1.086 | 0.865 | ON (wall: load noise) |
| h100-pcie | delta-h2d | 1.402 | 1.221 | 1.095 | 0.935 | ON (wall: load noise) |
| h100-pcie | lane-stride | 1.398 | 1.189 | 1.295 | 0.977 | ON (wall: load noise) |
| rtx-5090 | retain-resident | 1.008 | 1.022 | 1.016 | 1.005 | **OFF** (≤ 1%) |
| rtx-5090 | delta-h2d | 1.237 | 1.188 | 1.279 | 1.191 | ON |
| rtx-5090 | lane-stride | 1.577 | 1.405 | 1.582 | 1.362 | ON |

All three off is the s54-base placement: loop OFF/ON 2.21 / 2.29 (3090), 1.78 / 1.99 (A100),
2.20 / 2.35 (4090), 2.70 / 1.38 (H100), 2.31 / 2.47 (5090); process wall 1.88 / 1.94, 1.65 / 1.63,
1.85 / 1.93, 1.87 / 1.14, 1.94 / 1.96.

## Against the libraries

| session | Mapal wall s | load ms | Mapal loop ms | s54-base wall s | base loop ms | cuBLAS KV cache, device ms | eager + cache, wall ms | eager, no cache, wall ms | best library / Mapal loop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rtx-3090 | 0.508 | 174 | 366 | 0.955 | 807 | 29.7 | 28.6 | 49.0 | 0.078 |
| rtx-3090-confirm | 0.558 | 211 | 400 | 1.081 | 917 | 28.6 | 28.3 | 39.8 | 0.071 |
| a100 | 1.014 | 217 | 869 | 1.671 | 1547 | 84.6 | 85.7 | 72.7 | 0.084 |
| a100-confirm | 0.859 | 340 | 567 | 1.399 | 1130 | 64.7 | 63.9 | 53.1 | 0.094 |
| rtx-4090 | 0.510 | 194 | 352 | 0.942 | 774 | 29.1 | 28.7 | 29.7 | 0.081 |
| rtx-4090-confirm | 0.554 | 192 | 386 | 1.070 | 907 | 29.1 | 29.5 | 30.4 | 0.075 |
| h100-pcie | 0.958 | 391 | 517 | 1.788 | 1397 | 25.2 | 24.7 | 43.9 | 0.048 |
| h100-pcie-confirm | 1.348 | 408 | 597 | 1.539 | 824 | 21.1 | 21.3 | 40.1 | 0.035 |
| rtx-5090 | 0.511 | 194 | 358 | 0.989 | 827 | 51.7 | 52.1 | 42.9 | 0.120 |
| rtx-5090-confirm | 0.520 | 214 | 341 | 1.020 | 843 | 48.8 | 49.9 | 41.1 | 0.120 |

Mapal = the shipped table (retain-resident ON where it ships), both faces pooled. Every library
leg wins, by 8–28× on the loop. Per step Mapal's device work is four launches and three 1 KB copies
(traced: h2d 7–20, kernel 75–190, d2h 8–23 ms over 1024 steps); the rest of the loop is host time,
in which the host IR copies the carried `x` (1 MB) through the loop-state products four times per
step (`memcpy` merge → `x`, `x` → state, two first-class aggregate copies state → route → merge).
