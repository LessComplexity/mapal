//! The emitter's command-line flags, shared by the `emit` dev example and the
//! `mapal` CLI so every rung's OFF arm is reachable from the product binary.

use crate::{EmitOpts, GpuRungs, MovePanel, Order};

/// Apply one emitter flag to `opts`. `Ok(false)` = not an emitter flag (the
/// caller owns it); `Err` = an emitter flag with a malformed value.
pub fn apply(opts: &mut EmitOpts, a: &str) -> Result<bool, String> {
    match a {
        "--perf" => opts.perf_timing = true,
        "--no-tile" => opts.tiling = false,
        "--no-pack" => opts.packing = false,
        // The KC k-panel nest is default-OFF (a measured 3x loss locally at
        // 1024 f32, S29); this is how a box run opts in.
        "--kc" => opts.kc_nest = true,
        // ADR-0040 D1: the float face's two axes. `fma` steps and the free
        // order are the default; `--contract` survives as the steps flag.
        "--contract" => opts.contract = true,
        "--no-contract" => opts.contract = false,
        // fold-algebra OFF arms: no product–accumulate fusion from the step
        // algebra (host pairs, device strided tiles); no `reassoc` on fold steps.
        "--no-fma-pairs" => opts.gpu.pairs = false,
        "--no-fold-regroup" => opts.gpu.regroup = false,
        // compile-time-graph OFF arm: no row-block unit kernels (tile sites
        // keep their own launches).
        "--no-row-block" => opts.gpu.row_block = false,
        // exp (ADR-0042) OFF arms: the oracle's steps on the default face;
        // no rescale law in `fold_combine` (a fact arm).
        "--no-exp-hw" => opts.gpu.off.exp_hw = true,
        "--no-exp-rescale" => opts.gpu.off.exp_rescale = true,
        // flash-attention: the stream fact arm.
        "--no-stream" => opts.gpu.off.stream_fact = true,
        "--no-guard-box" => opts.gpu.off.guard_box = true,
        "--no-finite" => opts.gpu.off.finite = true,
        // Measurement only: a stream group's rows a block and window keys.
        _ if a.starts_with("--stream=") => {
            let v = &a["--stream=".len()..];
            let parts: Vec<Option<u64>> = v.split('x').map(|x| x.parse::<u64>().ok()).collect();
            let f = match parts[..] {
                [Some(b), Some(k)] => (b, k, 0, 0),
                [Some(b), Some(k), Some(r)] => (b, k, r, 0),
                [Some(b), Some(k), Some(r), Some(s)] => (b, k, r, s),
                _ => return Err(format!("--stream expects <b>x<kb>[x<rt>[x<s>]], got `{a}`")),
            };
            opts.gpu.stream_force = Some(f);
        }
        // Measurement only: a row-block unit's rows (0 = derive), the rung
        // table and the cost test aside. `--gpu-on=row-block` runs the rule
        // (cost-gated) where the part's table keeps it off.
        _ if a.starts_with("--row-block=") => {
            let Ok(b) = a["--row-block=".len()..].parse::<u64>() else {
                return Err(format!("--row-block expects a row count, got `{a}`"));
            };
            opts.gpu.row_block_b = b;
        }
        "--exact" => {
            opts.contract = false;
            opts.order = Order::Sequential;
        }
        _ if a.starts_with("--row-stage=") => {
            let v = &a["--row-stage=".len()..];
            let parsed = v.split_once('x').and_then(|(kc, rest)| {
                let (s, p) = rest.split_once('p').unwrap_or((rest, "1"));
                Some((kc.parse().ok()?, s.parse().ok()?, p.parse().ok()?))
            });
            let Some(f) = parsed else {
                return Err(format!(
                    "--row-stage expects <kc>x<stages>[p<passes>], got `{a}`"
                ));
            };
            opts.gpu.row_stage_force = Some(f);
        }
        // Measurement only: `horizontal` groups without the occupancy and
        // wave tests.
        "--hz=force" => opts.gpu.hz_force = true,
        "--calls=force" => opts.gpu.calls_force = true,
        // Measurement only: `s` blocks a row of a flow unit the slicing
        // admits (`row-slices`), the prices aside.
        _ if a.starts_with("--row-slices=") => {
            let Ok(s) = a["--row-slices=".len()..].parse::<u64>() else {
                return Err(format!("--row-slices expects a block count, got `{a}`"));
            };
            opts.gpu.row_slices_s = s;
        }
        _ if a.starts_with("--order=") => {
            opts.order = match &a["--order=".len()..] {
                "sequential" => Order::Sequential,
                "fixed" => Order::Fixed,
                "free" => Order::Free,
                v => {
                    return Err(format!(
                        "--order expects sequential, fixed or free, got `{v}`"
                    ));
                }
            };
        }
        // The move-panel traversal. **Deduced by default (S45)** — the
        // geometry comes from `mapal_ir::MoveSite` and the machine facts
        // from the target profile, so there is nothing to pass. This flag
        // survives only as a MEASUREMENT override: `off` is an A/B's OFF
        // arm, `<W>:<B>` forces S44's hand-typed behaviour for a block
        // sweep. A malformed value is an error rather than a silent no-op,
        // for the same reason `--target=` is.
        _ if a.starts_with("--move-panel=") => {
            let spec = &a["--move-panel=".len()..];
            let parsed = if spec == "off" {
                Some(MovePanel::Off)
            } else {
                spec.split_once(':')
                    .and_then(|(w, b)| Some((w.parse::<u64>().ok()?, b.parse::<u64>().ok()?)))
                    .filter(|&(w, b)| w > 0 && b > 0)
                    .map(|(w, b)| MovePanel::Force(w, b))
            };
            let Some(mp) = parsed else {
                return Err(format!(
                    "--move-panel expects `off` or <W>:<B> with both positive, got `{spec}`"
                ));
            };
            opts.move_panel = mp;
        }
        // S52/S53: measurement-only arms for the GPU rungs (an A/B needs
        // the same compiler with one rung flipped against the profile's
        // table). Unknown names are errors.
        _ if a.starts_with("--gpu-off=") || a.starts_with("--gpu-on=") => {
            let (flag, list) = a.split_once('=').expect("matched on `=`");
            let mask = if flag == "--gpu-on" {
                &mut opts.gpu.on
            } else {
                &mut opts.gpu.off
            };
            // An S53 item name stands for all of its rungs; S54's
            // `arch-select` is the table rung, `arch-rungs`.
            for rung in list.split(',').flat_map(|r| match r {
                "fma-regs" => vec!["idx32", "affine", "pad", "wide-fma"],
                "arch-select" => vec!["arch-rungs"],
                r => vec![r],
            }) {
                let Some(r) = mask.rung(rung) else {
                    return Err(format!(
                        "{flag}: unknown rung `{rung}` ({}, fma-regs, arch-select)",
                        GpuRungs::NAMES.join(", ")
                    ));
                };
                *r = true;
            }
        }
        // ADR-0041 D7: `--precision=tf32` is gone — a program admits TF32
        // products with `: tf32 | f32` on the body (`-> fold: tf32 | f32 { … }`).
        "--precision=tf32" => {
            return Err(
                "--precision=tf32 was removed (ADR-0041 D7): write `: tf32 | f32` on the \
                        fold, map, loop, seq or fn body"
                    .into(),
            );
        }
        // Measurement only: force split-K's partition count (0 = derive), on
        // `splitk` and `splitk-atomic`.
        _ if a.starts_with("--splitk=") => {
            let Ok(s) = a["--splitk=".len()..].parse::<u64>() else {
                return Err(format!("--splitk expects a partition count, got `{a}`"));
            };
            opts.gpu.splitk = s;
        }
        // ADR-0034 D1: one geometry knob of the GPU part's table, for the
        // offline search's arms (`benches/gpu/knobs.json`). Repeatable.
        _ if a.starts_with("--geom=") => {
            let spec = &a["--geom=".len()..];
            let knob = spec.split_once('=').and_then(|(k, v)| {
                let i = crate::profile::Geom::NAMES.iter().position(|n| *n == k)?;
                Some((i, v.parse::<u64>().ok()?))
            });
            let Some((i, v)) = knob else {
                return Err(format!(
                    "--geom expects <knob>=<count> with a knob of {}, got `{spec}`",
                    crate::profile::Geom::NAMES.join(", ")
                ));
            };
            opts.geom[i] = Some(v);
        }
        // plan-cost: one plan rate of the GPU part, for the fit harness's
        // arms (`benches/gpu/plancost.py`). Repeatable.
        _ if a.starts_with("--cost=") => {
            let spec = &a["--cost=".len()..];
            let rate = spec.split_once('=').and_then(|(k, v)| {
                let i = crate::profile::DeviceCost::NAMES
                    .iter()
                    .position(|n| *n == k)?;
                Some((i, v.parse::<u64>().ok()?))
            });
            let Some((i, v)) = rate else {
                return Err(format!(
                    "--cost expects <rate>=<value> with a rate of {}, got `{spec}`",
                    crate::profile::DeviceCost::NAMES.join(", ")
                ));
            };
            opts.cost[i] = Some(v);
        }
        // plan-cost: the candidate plans' `; mapal.cand` lines.
        "--plan-cands" => opts.gpu.cands = true,
        // per-site-choice: the most candidates a site's table holds (1 = the
        // rules' launch alone, the emission before candidates).
        _ if a.starts_with("--cands=") => {
            let n = a["--cands=".len()..]
                .parse::<u64>()
                .ok()
                .filter(|&n| n >= 1);
            let Some(n) = n else {
                return Err(format!("--cands expects a count >= 1, got `{a}`"));
            };
            opts.gpu.site_cands = n;
        }
        // Machine facts by name (plan-s31-target-profiles); never probed
        // unless the name is `native`. A box run names its machine.
        _ if a.starts_with("--target=") => {
            opts.target = a["--target=".len()..].to_owned().leak();
        }
        // S62 `hostopt`: the CPU profile a GPU target's host code is emitted
        // against (`native` by default).
        _ if a.starts_with("--host=") => {
            opts.host = a["--host=".len()..].to_owned().leak();
        }
        _ => return Ok(false),
    }
    Ok(true)
}
