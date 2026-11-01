# the roofline dispatch threshold, built and measured

Historical provenance: S50e.

Date: 2026-08-10 · plan: `components/runtime/plans/plan-s50-scheduler-handoff.md`
Predecessor: `s50d-dont-dispatch-what-cannot-be-shared.md`, which shipped the constant-free half
and recorded this half as a decision.
Raw: `benches/results/cpu/scheduling/integrated-dispatch/` (§5, the table to quote) and `benches/results/cpu/scheduling/dispatch-threshold/`.

## 1. The rule

A task is registered `kind = 0` — one slice, which `mapal-rt` runs on the calling thread
instead of queueing — when its estimated **serial** time is under what one dispatch costs.

```
est_ns = max( 2·n·k / flops_per_ns ,  bytes / bytes_per_ns )
inline iff est_ns < dispatch_ns
```

**No ABI change and no runtime rule.** `kind = 0` already means one slice, and the
single-slice inline path shipped in S50d. So the whole decision lives in
`backends/llvm/src/func/drive.rs`, per backend, and `mapal-ir` never learns it.

Applied in two places, because a packed tile site is two tasks: the matmul (compute term
dominates) and the B pack, which is a pure copy and has **no compute term at all**.

### Why a roofline and not an operation count

| task | est. flops | measured serial | achieved |
| --- | ---: | ---: | ---: |
| `nn4_256` layer | 3.4e7 | 68 µs | 495 GF/s |
| `conv2d_1024` | 1.9e7 | 250 µs | 76 GF/s |

Half the flops, 3.7× the time. They must go opposite ways, so no threshold on flops exists.
The memory term is what separates them.

### Why only a recognised tile site is estimated

The first build also estimated elementwise maps, as `3·n·width` bytes at the streaming rate,
and got two of fourteen shapes wrong **in the expensive direction**: `transpose_512`
(estimated 25 µs, real serial 142 µs, parallel is 1.85×) because a permutation does not run
at streaming rate, and `fir_65536` (estimated 6 µs, real 91 µs) because a 64-tap window reads
far more than three values per output.

A wrong verdict is not symmetric — dispatching something small costs one wave, inlining
something large costs the whole parallel speedup — so the estimator only decides where it has
a basis. A tile record carries `rows`, `c`, `k` and `elem`, which is exactly what the roofline
needs; without one, the task is dispatched as before.

## 2. The three constants, and which are measured

`TargetProfile::dispatch: Option<DispatchCost>`. **`None` means "dispatch everything" — the
behaviour every profile had before this existed — so no part can be made worse by a number
nobody took.** Only `apple-m4-sme` carries `Some`, and `native` inherits it only when the
detected part is the one it was measured on (SME with a 64-byte SVL).

| field | value | measured how |
| --- | ---: | --- |
| `flops_per_ns` | 1000 | `nn4_1024` at `MAPAL_PAR=1` runs 986 GF/s, `nn4_2048` 939 |
| `bytes_per_ns` | 127 | `saxpy_1048576` at `MAPAL_PAR=1` moves 12.58 MB in 0.0991 ms |
| `dispatch_ns` | 8300 | see below |

`raptorlake` is **owed** its numbers and does not have them: the arch box was unreachable this
session, so it keeps `None` and the i9 is unchanged by this work. `generic`, `apple-m`,
`zen3` and `cuda-ada` are `None` on purpose — verified: **0 of 174 cells move under
`--target=generic`.**

### `dispatch_ns` is not the parked-wave figure, and getting that wrong cost three cells

The first version used **30 µs**, from `handoff_floor`'s parked wave (22.6 µs at 8 executors,
41.2 at 13). That over-priced a dispatch, because P0/P0b/P2/P1 and the single-slice inline
rule had already made a real dispatch far cheaper than a cold park. Three cells regressed:

| shape | at 30 µs | at 8.3 µs |
| --- | ---: | ---: |
| `conv2d_512` | **0.805×** | 0.963× |
| `matmul512` | **0.944×** | 1.019× |
| `nn4_512` | **0.945×** | 1.006× |

The replacement is measured on a program the runtime actually ran: `nn4_128` opens 8
dispatches and runs **0.1449 ms** with them against **0.0788 ms** with all eight inlined, so
one dispatch costs `(144.9 - 78.8) / 8 = 8.3 us`. (Both absolutes are from the 81-cycle A/B
run; the same cell reads 0.0810 in the §5 run, which moves the answer to 7.9 us — inside the
4.2-16.5 us gap either way, which is why the gap is what gets quoted and not the point.)

The estimates the constant has to separate
are 4.2 µs (`nn4_128` layer), 4.1 µs (its pack), 16.5 µs (`nn4_512`/`matmul512` pack), 21 µs
(`conv2d_512`), 34 µs (`nn4_256` layer) and 84 µs (`conv2d_1024`) — a clear gap between 4.2
and 16.5, so any value in it answers every measured cell the same way.

**The lesson is that this constant tracks the runtime, not the machine.** It was correct when
`handoff_floor` measured it and wrong four changes later. It is derived from a program rather
than a microbenchmark for exactly that reason.

## 3. Verdicts against measurement

`par gain` is `1t / par` — under 1 means serial was faster and the task should be inlined.

| shape | verdict | par gain | |
| --- | --- | ---: | --- |
| `nn4_128` | INLINE | 0.55 | ✓ |
| `nn4_256` | dispatch | 0.95 | marginal either way |
| `nn4_512` | dispatch | 2.13 | ✓ |
| `nn4_1024` | dispatch | 2.94 | ✓ |
| `nn4_2048` | dispatch | 3.57 | ✓ |
| `conv2d_512` | dispatch | 1.03 | ✓ |
| `conv2d_1024` | dispatch | 3.33 | ✓ |
| `matmul512` | dispatch | 2.06 | ✓ |
| `matmul1024` | dispatch | 2.75 | ✓ |
| `transpose_512` | dispatch | 1.85 | ✓ |
| `fir_65536` | dispatch | 1.38 | ✓ |

Ten of eleven; `nn4_256` measures 0.95 and is within noise of parity either way.

## 4. Measured — interleaved A/B, same runtime both arms, 81 cycles

*(Source: `roof_ab.sh`, threshold arm against no-threshold arm. These are A/B RATIOS from one
run. The absolute ms in §5 come from a different, later run and will not match to the last
digit — see the note there.)*

| shape | par | | shape | par |
| --- | ---: | --- | --- | ---: |
| `nn2_128` | **2.041×** | | `matmul512` | 1.019× |
| `nn4_128` | **1.784×** | | `nn4_512` | 1.006× |
| `nn2x2_128` | **1.416×** | | `matmul1024` | 1.007× |
| `nn2x2_256` | **1.130×** | | `conv2d_1024` | 1.004× |
| `nn4_256` | **1.093×** | | `matmul4096`, `nn4_2048` | 1.000 |

`conv2d_512` reads 0.963 on a cell whose pre-arm spread is 371% — noise.
**Gate: 1056 passed / 0 failed**, differential oracle included; two new pins
(`tiny_tile_site_is_inlined_when_the_part_has_measured_costs`,
`the_dispatch_threshold_is_what_separates_the_two_profiles_here`); fmt clean.

Two existing tests were re-scoped rather than deleted:
`packed_wrapper_dispatches_the_b_pack_across_the_pool` now pins `generic` by name (S46's rule
— it is a structural test and the default target is `native`), and
`sme_profile_is_byte_identical_to_apple_m_wherever_the_rung_declines` became
`..._differs_from_apple_m_only_in_dispatch_kind_...`, because `apple-m4-sme` is now `apple-m`
plus **two** declared capabilities. It normalises exactly the `kind` argument and demands
byte identity everywhere else, so an SME change that leaked into a declining shape still
fails.

## 5. Cumulative: S49 → now, M4 Pro — THE TABLE TO QUOTE

**One run, `benches/results/cpu/scheduling/integrated-dispatch/`, 41 interleaved cycles**, mapal and numpy alternating
inside it. mapal is the FMA face at `MAPAL_PAR=14`, median; numpy is the better of its
1-thread and multi-thread legs. `spread` is `(max-min)/min` over that run's mapal samples,
and it is printed because it decides which rows are worth reading at all.

| shape | d | mapal | spread | numpy | **mapal × np** | S49 | **vs S49** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nn2 | 128 | 0.0406 | 20% | 0.0136 | 0.33 | 0.1720 | **4.24×** |
| nn4 | 128 | 0.0810 | 20% | 0.0234 | 0.29 | 0.3636 | **4.49×** |
| nn2x2 | 128 | 0.0758 | 10% | 0.0230 | 0.30 | 0.2301 | **3.03×** |
| nn2 | 256 | 0.1370 | 41% | 0.0484 | 0.35 | 0.2422 | 1.77× |
| nn4 | 256 | 0.2613 | 64% | 0.0919 | 0.35 | 0.4751 | 1.82× |
| nn2x2 | 256 | 0.1917 | 63% | 0.0910 | 0.47 | 0.3161 | 1.65× |
| nn2 | 512 | 0.4400 | 72% | 0.3254 | 0.74 | 0.4982 | 1.13× |
| nn4 | 512 | 0.6333 | 63% | 0.6450 | **1.02** | 0.8718 | 1.38× |
| nn2x2 | 512 | 0.4726 | 27% | 0.6438 | **1.36** | 0.5712 | 1.21× |
| nn2 | 1024 | 1.4583 | 6% | 2.1709 | **1.49** | 1.5915 | 1.09× |
| nn4 | 1024 | 2.9301 | 8% | 3.7635 | **1.28** | 3.1681 | 1.08× |
| nn2x2 | 1024 | 2.7707 | 6% | 4.0635 | **1.47** | 2.8298 | 1.02× |
| nn2 | 2048 | 10.1295 | 4% | 13.5052 | **1.33** | 10.3757 | 1.02× |
| nn4 | 2048 | 20.3625 | 5% | 24.9696 | **1.23** | 20.6452 | 1.01× |
| nn2x2 | 2048 | 19.6773 | 3% | 25.6455 | **1.30** | 19.9572 | 1.01× |

**READ THE SPREAD COLUMN.** d >= 1024 spreads 3-8% and those rows are solid. d = 256-512
spreads 27-72%, so nothing under ~1.3x there is a result. d = 128 spreads 10-20% at ~80 us,
which is exactly why the same `nn4_128` cell reads **0.0788, 0.0810 and 0.0827** across the
three runs quoted in this document — one measurement, three samples, all inside its own band.
Quote this section; §2 and §4 keep their own absolutes only because what those sections are
about is the RATIO within a single run.

**d=128 is 3.1–4.4× faster than it was**, and the standing against numpy there went 0.06–0.10×
to 0.28–0.34× — still a loss, and now a **compute** loss rather than a scheduling one: at
d=128 the parallel time (0.0827 ms for `nn4`) is within 20% of the one-thread time
(0.0698 ms), so there is almost no dispatch left to remove. What remains is the single-thread
deficit, which at d=128 is 0.31× of numpy.

## 6. Open

- **`raptorlake` owes its three numbers.** Box unreachable; the i9 is unchanged by this work
  and the rule is inert there.
- `dispatch_ns` tracks the runtime and will need re-measuring whenever the handoff changes
  again. The derivation (`nn4_128` with and without its dispatches) is one command.
- **The threshold only decides for recognised tile sites — extending it to elementwise maps
  is the open item, and it is open because the obvious estimate is measurably wrong.**
  Estimating a map as `3 * n * width` bytes at the streaming rate fails two ways:

  | shape | access pattern | what the estimate got wrong | est vs real |
  | --- | --- | --- | ---: |
  | `fir_65536` | **windowed** — each output reads a 64-tap window of consecutive inputs, the next output the same window shifted by one | the BYTE COUNT: 65 reads per output, not 3 | 6 µs vs 91 µs |
  | `transpose_512` | **scattered** — `out[j*n+i] = in[i*n+j]`, so walking one array in order strides the other by a whole row and every read lands in a different cache line | the RATE: ~15 B/ns achieved, not the 127 B/ns a streaming kernel reaches | 25 µs vs 142 µs |

  Both are UNDER-estimates, so both would have been inlined, and both have real parallel
  speedups (1.85x and 1.38x) that inlining would have thrown away. A tile record supplies
  `rows`/`c`/`k`, which is exactly the reads-per-output and reuse information these lack.
  What is needed is a per-element read count and an achieved-rate class the emitter can
  derive from the graph — `elem_plan` already records what `out[i]` IS, which is the place to
  look first.
- The one-thread deficit is now the dominant term at every size — 0.28–0.35× of numpy below
  d=512 on the M4, 0.87× on the i9. S48's standing P0, untouched.
