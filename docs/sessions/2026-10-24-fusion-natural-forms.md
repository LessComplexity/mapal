# 2026-10-24 — fusion-natural-forms: the four-part GPU sweep

Branch `fusion-natural-forms` from the integration at `332d20f` (generic-fusion + natural-forms); last code commit `20cfffe`; this log is in the docs commit after it.
Not pushed. Ran unattended (the user asked for the complete sweep, up to 7 h). Tables:
[evidence README](../performance/evidence/README.md#fusion-natural-forms-sweep-the-four-parts-on-hardware); raw:
[`benches/results/evidence/fusion-natural-forms/`](../../benches/results/evidence/fusion-natural-forms/).

## Steps

| step | result |
| --- | --- |
| library legs for the chains (integration finding 7) | chain_bench `attn_<T>` (cuBLAS eager, cuBLAS + warp-per-row softmax) and `smlp_<M>_<d>` (eager, cuBLASLt RELU_BIAS, CUTLASS LinearCombinationRelu), any-shape GEMM helpers; torch_bench attention, smlp, FIR→ew→rowsum, ew→conv→ew. chain_bench compiled in the local `nvidia/cuda:12.8.1-devel` image; the torch legs matched the exact-face oracles on CPU torch (maxdev ≤ 7.8e-6) — `0f3a3f8` |
| bundle | `integrated_bundle.sh` with the default GEMM cells and MLP programs, the 23-program suite and 96 rung arms per part (`fuse`, `window`, `range-map`, all three; `lane-stride` on attention), 4 min; host IR / PTX byte-identical to the integrator's bundle2 on the sampled stems; the pt rows of the static intent check identical (2223 / 2444 / 2883 / 2737) |
| main boxes, four in parallel | H100 done in 20 min, 5090 26 min; the 4090 took five rentals (a 40 KB/s upload on a host at load 23, stopped; two created stopped; one upload failed five times); the 3090 three (ssh lost; never left image loading) |
| attention library legs wrong on the first H100 / 5090 boxes | the T×T `p` buffer reused main()'s d×d operand `p`; `B()` handed back the small buffer (gate FAIL, values 1024 / nan). Fixed and guarded (`233c49d`); the 4090 / 3090 boxes uploaded after the fix; H100 / 5090 re-taken with the attention programs alone (`attn-confirm/`, the H100 on the same host) |
| second host per part (`confirm/`) | 4090 Ryzen 9 5950X; 5090 Ryzen 9 7950X (a first 5090 on a 7945HX lost clock at loop_4096 rep 2: kernels 2.01 → 6.76 ms, kept in `confirm-throttled/`); 3090 S65's own host (24953 / 52126, EPYC 7702, 350 W; a first 3090 on an EPYC 7B12 ran at ~27 % speed, `confirm-throttled/`); H100 S65's own host (670790 / 149333, EPYC 7713) |
| decision | `fuse` OFF on all four parts (`GPU_RUNGS_OFF` × 4, `a5a5368`); `window`, `range-map`, 2-D `lane-stride` ON |
| end check | the build with the rows: 25 suite programs × 4 parts = the measured build (fused programs = their `.off-fuse` arm), host IR and PTX (`endcheck-fuse-off.txt`) |
| tests | workspace release 1254 passed, 0 failed, 3 ignored (93 binaries); `generic_fusion.rs` forces `fuse` on |
| harness fixes found on the way | `vast_part.sh` searched only the 20 cheapest offers, so a `CPU_NAME` filter found nothing (now 64); the torch wheel download timed out on the 3090 main box (pip `--timeout 120 --retries 10`) |

## Decisions

| rung | rtx-4090 | rtx-5090 | h100-pcie | rtx-3090 | main box: ON slower / inside null / faster, of the programs it changes | second host |
| --- | --- | --- | --- | --- | --- | --- |
| `fuse` | OFF | OFF | OFF | OFF | 4 / 2 / 0, 5 / 1 / 0, 4 / 2 / 0, 6 / 0 / 0 (of 6) | 6 / 0 / 0, 4 / 2 / 0, 6 / 0 / 0, 6 / 0 / 0 |
| `window` | ON | ON | ON | ON | ON faster 2 / 2 on every part (9.7–34.4×) | ON faster everywhere (11.5–47.9×) |
| `range-map` | ON | ON | ON | ON | decode ON faster (2.50–4.39×) | ON faster (2.35–4.24×) |
| `lane-stride` (2-D) | ON | ON | ON | ON | attention ON faster 4 / 4 (106–693×) | ON faster (110–436×) |

Why `fuse` loses: the unit (`mx:rowfold; ex:cell; den:rowfold; y|p:cell`) runs one thread per row — 1024–4096
threads, 32–128 blocks — walking the row's T cells for each member and evaluating `ex` twice; the OFF arm runs `ex` and
`y|p` as cell kernels over T² threads. Traced, the unit is 1.79–4.82× the four kernels it replaces on every part and
program; the d = 128 attention unit carries 6 `div.rn.f32` against 2 and 4 multiplies by 0.125 at d = 64 (÷11.3 against ÷8.0, PTX of the unit kernel).

## Rentals

| instance | part | box | offer, host / machine, host CPU, place | outcome | destroyed (confirmed absent) |
|---|---|---|---|---|---|
| 57541722 | rtx-4090 | main | 54483245, 115630 / 136786, AMD EPYC 7K62 48-Core Processor, Poland, PL | stopped (slow upload, host load 23) | yes |
| 57541723 | h100-pcie | main | 46716863, 415974 / 81035, AMD EPYC 9354 32-Core Processor, , US | DONE | yes |
| 57541724 | rtx-5090 | main | 50338900, 553680 / 148415, AMD EPYC 7K62 48-Core Processor, Beijing, CN | DONE | yes |
| 57541725 | rtx-3090 | main | 56377758, 410852 / 57628, Xeon® E5-2673 v4 , China, CN | ssh lost mid-run (box_run exit 255) | yes |
| 57542138 | rtx-3090 | main | 54681931, 483833 / 113571, Xeon® E5-2673 v4 , California, US | never reached running | yes |
| 57543748 | rtx-4090 | main | 51194881, 655981 / 148197, Xeon® Platinum 8352V , Quebec, CA | start failed | yes |
| 57543751 | rtx-4090 | main | 53473111, 366851 / 150707, AMD EPYC 9684X 96-Core Processor, California, US | start failed | yes |
| 57543756 | rtx-4090 | main | 57531516, 55116 / 10577, AMD EPYC 7402 24-Core Processor, Vietnam, VN | bundle upload failed 5 times | yes |
| 57544369 | rtx-3090 | main | 55676191, 443829 / 84216, AMD EPYC 7K62 48-Core Processor, California, US | DONE | yes |
| 57544551 | rtx-4090 | main | 45147855, 229807 / 113799, AMD EPYC 7532 32-Core Processor, Estonia, EE | DONE | yes |
| 57544627 | h100-pcie | attn-confirm | 46716863, 415974 / 81035, AMD EPYC 9354 32-Core Processor, , US | DONE | yes |
| 57544628 | rtx-5090 | attn-confirm | 45089955, 203531 / 144253, AMD EPYC 7543 32-Core Processor, Bulgaria, BG | DONE | yes |
| 57547762 | rtx-5090 | confirm-throttled | 56721658, 617499 / 147959, AMD Ryzen 9 7945HX with Radeon Graphics, New Jersey, US | DONE | yes |
| 57548252 | rtx-3090 | confirm-throttled | 51025618, 116365 / 33953, AMD EPYC 7B12 64-Core Processor, Taiwan, TW | DONE | yes |
| 57549653 | rtx-4090 | confirm | 53475231, 206415 / 139350, AMD Ryzen 9 5950X 16-Core Processor, Washington, US | DONE | yes |
| 57549771 | rtx-5090 | confirm | 51586720, 236375 / 51364, AMD Ryzen 9 7950X 16-Core Processor, Hungary, HU | DONE | yes |
| 57551147 | rtx-3090 | confirm | 56607982 (named: S65 host 24953 / 52126), AMD EPYC 7702 64-Core Processor, Oregon, US | DONE | yes |
| 57553934 | h100-pcie | confirm | 57077524 (named: S65 host 670790 / 149333), AMD EPYC 7713 64-Core Processor, South Africa, ZA | DONE | yes |

Protected, never touched: 57065574, 57100404, 57126023, 57133432, 57285249, 57415164, 57450102 (`instances-before.txt`,
`instances-after.txt`). Credit 30.15 → 23.77 (6.38 spent).

## Commits

| commit | what |
| --- | --- |
| `0f3a3f8` | bench: library legs for the generic-fusion chains |
| `233c49d` | fix(bench): attention's p buffer was the operand p |
| `a4040d8` | evidence(fusion-natural-forms): H100 PCIe and RTX 5090 box runs |
| `e4d6457` | evidence(fusion-natural-forms): attention re-take, H100 PCIe and RTX 5090 |
| `8fef884` | evidence(fusion-natural-forms): RTX 4090 box run |
| `1e77740` | evidence(fusion-natural-forms): RTX 3090 box run |
| `bb9e362` | evidence(fusion-natural-forms): RTX 5090 confirm box (Ryzen 9 7945HX host) |
| `b13b6ee` | evidence(fusion-natural-forms): the 5090 confirm box throttled mid-run |
| `a5a5368` | perf(gpu): fuse ships OFF on all four measured parts |
| `08f4fa5` | evidence(fusion-natural-forms): RTX 3090 confirm box (EPYC 7B12, 250 W) |
| `28df78c` | evidence(fusion-natural-forms): the slow 3090 confirm box to confirm-throttled/ |
| `8a5cf52` | evidence(fusion-natural-forms): RTX 4090 confirm box (Ryzen 9 5950X, 375 W) |
| `53e3ac6` | evidence(fusion-natural-forms): RTX 5090 confirm box (Ryzen 9 7950X, 575 W) |
| `f970650` | evidence(fusion-natural-forms): RTX 3090 confirm box on S65's own host (EPYC 7702, 350 W) |
| `1abed1f` | evidence(fusion-natural-forms): H100 PCIe confirm box on S65's own host (EPYC 7713) |
| `20cfffe` | perf(gpu): the fuse rows cite the second hosts; sweep tables |
| (this) | docs: the sweep in the evidence README, next-session, this log |
