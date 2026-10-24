# Continue from S65 residency

Branch `s65/residency` (cut from `s64/evidence` @ `5d594e9`), worktree `.claude/worktrees/wf_ee07f04a-dfc-2`. Not pushed.
Results: [evidence README, S65](performance/evidence/README.md) · raw: [`benches/results/evidence/s65/`](../benches/results/evidence/s65/) ·
tables: `benches/gpu/evidence_report.py`, `benches/gpu/movement_report.py`, `benches/gpu/s65_arms.py`.

## State

| item | state |
| --- | --- |
| rule R | in the existing storage layer (`func/frame.rs` host, `func/resident.rs` device); `byref.rs` deleted. New IR query: `LastUsePlan::view_pos` / `view_death` / `view_dead_after` (`mapal-ir/src/algo.rs`, tested in `algos.rs`). Rung `by-ref` (OFF arm = the S64 build) |
| movement | host memcpy 7–11 → 0 in loop / loopmm / decode / rt_mlp2 on CPU and GPU, four parts; per-trip copies = the facts' (`; mapal.move`); traced crossings = predicted (`by-ref:runtime`, 0 mismatch) |
| GPU_RUNGS_OFF | 4090 / 5090 `retain-resident` rows removed (ON 1.214 / 1.167, explained by the one-row K/V site); H100 row kept (inside null 0.094) |
| property tests | carried loops 320 programs / 434 loops (copies 292 = required 292); device loops 64 programs / 37 device-carried states |
| diff | source +1271 / −55 lines vs `5d594e9` (the byref.rs version: +925 / −40) — larger: the IR query, general views, the parallel-entry race fix, N-buffer coloring |
| rentals | none live. Protected, never touch: 57065574, 57100404, 57126023, 57133432, 57285249, 57415164, 57450102 |

## Findings, open

| finding | evidence | next |
| --- | --- | --- |
| 3090 `splitk-atomic` slower on loopmm 1024³ (Filled) | 1.6191 vs 1.4133 ms, 1.144 (1.124..1.186), null 0.059; atomic S=2 + zero-fill vs S=1; decode / 128×2048 ON faster | split the class by a fact (per-trip zero-fill on a carried output) |
| H100 PCIe box noise on rt_mlp2 | shipped vs null 20.54 / 12.93 ms (same build) | re-measure on another H100 |
| rt_mlp2 5.07 / 4.89 / 6.43 / 3.50× PyTorch compile (4090 / 5090 / H100 / 3090) | kernel 2.83 ms, copies 4.12 ms (8 MB host-made request inputs per trip, required by the facts) on the 4090 | make the request inputs on the device when their producer allows it |
| decode 1.64–3.23× the cache harness (4090 2.49, 5090 1.94, H100 3.23, 3090 1.64) | 4090 kernel 30.24 ms over 1024 steps, 4096 launches | launch count (fuse the per-step sites) |
| S64 open items (spills 5090 / 3090, prefetch H100 / 3090, unlaunched twins, H100 grid-share, 5090 JIT) | unchanged; [S64 table](performance/evidence/README.md#s64) | as S64 |

## Resume commands

```sh
cd /Volumes/LessComplex/Personal/mapal/.claude/worktrees/wf_ee07f04a-dfc-2
git status --short && git log --oneline -10
python3 benches/gpu/evidence_report.py benches/results/evidence/s65
python3 benches/gpu/movement_report.py benches/results/evidence/s65
python3 benches/gpu/s65_arms.py benches/results/evidence/s65/rtx-4090/results-rtx-4090.txt
MAPAL_REQUIRE_CLANG=1 cargo test --workspace --release --no-fail-fast -- --test-threads=1
vastai show instances   # expect only the protected ids
```

Box env: `PROFILE=1 E2E=0 TIMING=0 PROBE_REPS=9 RUN_MIN=60` through `benches/gpu/vast_part.sh`, 30 reps.
