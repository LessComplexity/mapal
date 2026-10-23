# 2026-10-23 — S62: six optimization items integrated and measured on four parts

Branch `s62-integrated` cut from `4e1102c`; final measured build `6a0e3a0`. Not pushed.
`s54-gpu-fusion` and `oainotes.md` untouched. Full tables: [current results](../performance/gpu/current-results.md#s62-integrated-6a0e3a0);
gaps and designs: [S62 gap map](../performance/gpu/disassembly/s62-gap-map.md).

## Items

| item | branch @ commit | merge | verdict | kept | discarded |
| --- | --- | --- | --- | --- | --- |
| geometry | s62/geometry @ 580f672 | 40f858c | partial | row-fit, wave-gate, grid-share, panel-regs, fill-rows, with per-part OFF rows | — |
| asyncstage | s62/asyncstage @ be89bc7 | ee5770d | partial | cell-fill | asyncstage (OFF on all four parts) |
| splitk | s62/splitk @ 6c3162c | c0eaa84 | keep | reassoc face split-K (ADR-0040), 3090 Filled/Exact and Under/Exact OFF | — |
| cpugpu | s62/cpugpu @ f8d7272 | b94177c | keep | hostopt, place | — |
| runtime | s62/runtime @ 2b25583 | 52c57e0 | keep | dead-output allocation elimination, arena, resident, devgen | streams stays OFF |
| cli | s62/cli @ 8f1ae02 | 0601a93 | keep | `mapal build/run/dump-ir/test`, `flags::apply` | — |

## Integration fixes

| commit | fix |
| --- | --- |
| 9a9cc20 | cell-fill × splitk: per-cell store omitted the `z·rows·c` slab offset, every partition wrote slab 0; caught by value gates on 5090/4090/H100/3090 (incl. shipped 3090 `_rc` 16×1024×4096); regression `cell_tile_split_stores_into_its_slab` |
| 6a0e3a0 | CLI read the device arch from the `<gpu>+<host>` profile name (panic); now the GPU half |
| 6a0e3a0 | tests re-pinned: asyncstage tests reach cell-fill with fill-rows/row-fit off; all-off test turns off all seven S62 tile rungs; wave-gate on → cell tile; splitk S pins on pre-S62 tiles; CLI GPU test `--gpu-off=place` |

Merge conflicts resolved: tile scalar path only for the cell-fill tile (`tn < lanes`), same for the column-major store check;
kernel list keeps splitk's split kernels and cpugpu's placement filter; probe loop keeps cpugpu's value gate plus runtime's SUBMIT pass;
emit flags moved into `flags::apply`.

## Numbers (kernel ms, integrated vs 4e1102c, int/cuBLAS)

| part | 16×2048×7168 E / C | 16×18432×7168 E / C | 4096³ C | MLP base/int E / C |
| --- | --- | --- | --- | --- |
| 5090 | 0.0922 / 0.0819 (base 0.2560 / 0.2621); 1.80 / 1.60 | 0.3379 / 0.4588; 1.00 / 1.36 | 2.3204; 1.08 | 0.52, 0.51 / 0.68, 0.88 |
| 4090 | 0.1055 / 0.0993 (0.5153 / 0.5161); 1.10 / 1.03 | 0.6176 / 0.6185; 0.97 / 0.97 | 2.9194; 0.99 | 0.96, 0.96 / 0.44, 0.62 |
| H100 PCIe | 0.1417 / 0.1305 (0.5270 / 0.8100); 2.06 / 1.89 | 0.5252 / 0.4826; 1.43 / 1.31 | 3.6354; 0.96 | 1.00, 1.00 / 0.77, 1.25 |
| 3090 | 0.1516 / 0.1526 (0.4526 / 0.6605); 1.64 / 1.66 | 0.8161 / 0.9206; 1.01 / 1.13 | 8.6518; 0.90 | 0.47, 0.84 / 0.99, 0.99 |

Disassembly ranks nine gaps by attributed ms: G1 FMUL+FADD on E/R 17.2459, G2 grid underfill 2.9733, G3 unpipelined mainloop 2.1950,
G4 per-face tile 1.5094, G5 lazy loading 1.2275, G6 split wave quantization 0.2893, G7 Exact MLP rungs 0.1124, G8 DRAM residual 0.0872,
G9 per-launch driver calls 0.0276.

## Open

| issue | measured |
| --- | --- |
| residual MLP single iteration slower than base | 5090 E 0.51–0.52, C 0.68–0.88; 4090 C 0.44–0.62; 3090 E 0.47–0.84; H100 C d256 0.77; gap is outside the kernels; the gap map attributes 1.2275 ms over 16 arms to lazy loading (G5) |
| grid-share × splitk | grid-share out faster on 8 of 11 reassoc cells, slower on 3 (1.05–1.14) |
| H100 16×1024 reassoc+contract after row-fit flip | 0.0383 vs 0.0366 ms (noise 0.002) |
| wave-gate on reassoc faces | off/on 0.40–0.95 on 16×1024 |
| OpenSpec | `optimize-ordered-smallm-gpu` 2.1 open: 16K controls not run |

## Tests

| suite | result |
| --- | --- |
| `cargo test -p mapal-backend-llvm` | 258 passed, 0 failed, 1 ignored |
| `cargo test -p mapal-cli` | 4 passed |
| `MAPAL_REQUIRE_CLANG=1 cargo test --workspace --release --no-fail-fast -- --test-threads=1` | 1217 passed, 0 failed, 1 ignored (after the final rung flips; `benches/results/gpu/s62/integrated/workspace-tests.log`) |

## Rentals (all destroyed with `-y`)

| scope | created | ids |
| --- | --- | --- |
| integration | 7 | 57304329 (5090), 57304326 (4090, upload failed), 57305611 (4090), 57304327 (H100 PCIe), 57304328 (3090), 57306700 (H100 PCIe confirm), 57306701 (3090 confirm) |
| geometry | 14 | 57293231, 57294918, 57294920, 57294919, 57297489, 57297492, 57297490, 57297491, 57298823, 57298826, 57298825, 57298824, 57300259, 57301011 |
| asyncstage | 11 | 57284956, 57284957, 57284958, 57284997, 57284959, 57287180, 57287181, 57287605, 57287679, 57288872, 57289286 |
| splitk | 4 | 57280970, 57280973, 57280971, 57280972 |
| cpugpu | 11 | 57281973, 57281974, 57281975, 57282035, 57283515, 57289275, 57289289, 57286720, 57286721, 57287944, 57293236 |
| runtime | 43 | listed in `benches/results/gpu/s62/runtime-measurements/` (`instances.json`) |
| cli | 0 | — |

At 2026-10-23T02:21:50Z `vastai show instances` listed only the protected 57065574, 57100404, 57126023, 57133432, 57285249.

## Resume

```sh
cd /Volumes/LessComplex/Personal/mapal/.claude/worktrees/wf_a3ec2a1a-376-1
git log --oneline -8
cat docs/next-session.md docs/performance/gpu/disassembly/s62-gap-map.md
python3 benches/gpu/integrated_report.py benches/results/gpu/s62/integrated/*/results-*.txt
supercharge-drift . && openspec validate --all --strict
```
