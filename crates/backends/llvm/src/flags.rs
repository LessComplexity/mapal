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
        "--exact" => {
            opts.contract = false;
            opts.order = Order::Sequential;
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
