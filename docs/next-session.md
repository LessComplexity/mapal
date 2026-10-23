# Continue from S64 evidence pass

Branch `s64/evidence` (cut from `s63/default-float` @ `c0103c5`), worktree `.claude/worktrees/wf_ac66b133-4ba-2`. Not pushed.
Results: [evidence README](performance/evidence/README.md) · raw: [`benches/results/evidence/s64/`](../benches/results/evidence/s64/) ·
checker: `benches/gpu/evidence.py`, intent: `benches/gpu/intent.py`, re-judge: `benches/gpu/rejudge.py`.

## State

| item | state |
| --- | --- |
| evidence lines | every device kernel carries `; mapal.plan …` (geometry, fill key, face, index width, combine, kernel rungs ON); a GPU build's host module ends with `; mapal.host …` (profiles, face, device / host sites). Comments only: `llc` drops them |
| remarks | `MAPAL_EVIDENCE=1 mapal build` and every bundle save opt / llc / clang optimization remarks beside the outputs |
| checker | `evidence.py static|facts|check|diff|jit`; a bundle writes `<part>/evidence/{facts.json,evidence.tsv}`, a box run `evidence-<part>.tsv` (JIT registers / local bytes from the harnesses) and `jit …` lines (driver image vs ptxas) |
| rule (proposed for CLAUDE.md, not written: agents may not edit it) | a bundle or box run without `evidence.tsv` is not a result; `GPU_RUNGS_OFF` rows cite the evidence row that explains them |
| rows | 92 in-scope rows carry their S64 status in the cite (judge.tsv row N); 16 A100 rows unverified (no box). Changed: h100-pcie `warp-fill` Under/Contract → `Fill::Partial` (≥ 1 warp per scheduler); end check on the H100: 128×2048×7168 0.3090 vs 0.5381 ms, MLP kernels 0.1514 / 0.0329 vs 0.3823 / 0.1035 ms. The same split on the 3090 was measured and not taken (128×2048×7168 default 1.179, `_rc` 1.058: the filled tile lowers the split count) |
| rentals | none live. Protected, never touch: 57065574, 57100404, 57126023, 57133432, 57285249 (57415164 is another session's) |

## Findings, open

| finding | evidence | next |
| --- | --- | --- |
| loop-carried arrays are memcpy'd on the host every iteration (5 × the array per trip) although the device holds them | host IR of `pt_loop_2048`; host residue 34.49 ms of 38.46 (4090), 25.77 / 35.00 (H100); loopmm / rt_mlp2 / decode the same; CPU build of `pt_loop_4096` overflows a 64 MB stack (rc 139) | drop the host shadow of a resident carried array (resident.rs) |
| softmax_rows: the normalize runs on the host behind a 16 / 64 MB copy-out | trace: d2h 1.37 ms + host 4.39 ms of 6.21 (4090, n=2048) | a device map for the row-normalize (devmap after rowfold) |
| conv (`k / kk`, `k % kk` taps) is never a device site | `pt_conv_2048_5`: no kernel; 1.2274 ms vs torch 0.0891 (4090) | `gpu_site` encoding for the 2-D window |
| 5090: the driver JIT (610.57.04, CUDA 13.3) is not ptxas 12.8 | `jit …: 0 identical, 3 differ`; default 16×1024 kernel 19 vs 33 LDL/STL; `exec` local 152 vs ptxas 208 B | static SASS for sm_120 from a 13.x ptxas wheel |
| spills on shipped kernels | 5090 16×1024 and 16×2048 (default, `_c`, `_rc`; 16×1024: 208 B at 255 regs, 32-thread blocks, bk 256), 4096³ `_r`/`_x`; 3090 128×2048 `_rc`, 16×1024 `_r`/`_x` (24–140 B) | the panel-regs / bk choice on 1-warp blocks |
| prefetch not realized: the next panel's LDG sits after the math, its STS reads it at the next trip's top | H100 16×18432 (all faces), 4096 `_r`/`_x`; 3090 16×1024 `_r`/`_x`, 4096 `_r`/`_x` (`sass.hot.ldg_cover_min = 0`) | schedule the load before the math in the emitted IR |
| unlaunched kernels in device modules (the ordered twin of a split / stream-K site) | `module:on` mismatches on all four parts | emit only the realization the launch uses |
| value gate: the relative bound fails under cancellation | `pt_rowsum_2048` default 38.058914 vs exact 38.090485 (rel 8.288e-04 > 4.885e-04) on every part, CPU build identical | bound against Σ|terms| (needs an `_xabs` oracle) |
| rows `open (ON faster …)` | 4090 grid-share Under/Exact (128×2048 0.887/0.928), 5090 streamk Under (128×2048 0.861/0.857), H100 streamk Filled (16×18432 0.748/0.749 vs 4096³ 1.134), 3090 asyncstage (128×2048_c 0.830/0.805), 3090 streamk Under (16×1024 0.862/0.865), H100/3090 grid-share on MLPs (0.991, 0.952) | class split by a fact the static evidence names (H100 streamk: ordered waves 0.24 vs 4.49) |
| 3090 warp-fill at 0 warps per scheduler: MLP 0.527 / 0.478, 128×2048×7168 `_c` 0.523, but default 1.179 / `_rc` 1.058 | the split count falls from 8 to 2 (atomic) and 4 to 2 (workspace) under the filled tile | let `split_count` keep the grid's fill, then split the class |
| row 93 (3090 splitk Filled/Exact) met the change rule this session (ON/OFF 0.904/0.890, null 0.011; resident warps 3.12 → 12.49) | S62 measured 1.134/1.126 on another 3090 | a third 3090 before any change (sign unstable) |

## Resume commands

```sh
cd /Volumes/LessComplex/Personal/mapal/.claude/worktrees/wf_ac66b133-4ba-2
git status --short && git log --oneline -8
python3 benches/gpu/evidence_report.py benches/results/evidence/s64            # every table
python3 benches/gpu/evidence.py diff <bundle>/<part> <stemA> <stemB>           # the re-judge instrument
SUITE="…" ARMS="$(python3 benches/gpu/rejudge.py arms h100-pcie)" benches/gpu/integrated_bundle.sh <out> h100-pcie:sm_90a
MAPAL_REQUIRE_CLANG=1 cargo test --workspace --release --no-fail-fast -- --test-threads=1
vastai show instances   # expect only the protected ids
```

Box env: `PROFILE=1 E2E=0 TIMING=0 PROBE_REPS=7 RUN_MIN=55` through `benches/gpu/vast_part.sh`; torch 2.11.0+cu128 (the
newest wheel for the image's Python 3.10). Static SASS: `benches/gpu/cudatool.sh` (docker, pypi wheels ptxas 12.8.93,
cuobjdump / nvdisasm 13.4.92). One rented 3090 host lacked AVX2: the x86-64-v3 CPU-only build printed nothing there.
