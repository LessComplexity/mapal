//! Emit textual LLVM IR for a `.mapal` file (ADR-0020; dev tool — the future
//! `mapal build` embryo). Usage: `cargo run -p mapal-backend-llvm --example emit -- <file.mapal> [-] [--perf] [--no-tile] [--no-pack] [--kc] [--contract] [--rewrite] [--target=<name>] [--device-out=<file.ll>]`
//! (writes `<file>.ll` next to the source, `<file>_perf.ll` with `--perf`, or
//! stdout with `-`; `--rewrite` rewrites the lowered IR before emission).

use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!(
            "usage: emit <file.mapal> [-] [--perf] [--no-tile] [--no-pack] [--kc] [--contract] [--rewrite] [--target=<name>] [--device-out=<file.ll>]"
        );
        return ExitCode::from(2);
    };
    let mut to_stdout = false;
    let mut rewrite = false;
    let mut device_out: Option<String> = None;
    let mut opts = mapal_backend_llvm::EmitOpts::default();
    for a in args {
        match a.as_str() {
            "-" => to_stdout = true,
            "--perf" => opts.perf_timing = true,
            "--no-tile" => opts.tiling = false,
            "--no-pack" => opts.packing = false,
            // The KC k-panel nest is default-OFF (a measured 3x loss locally at
            // 1024 f32, S29); this is how a box run opts in.
            "--kc" => opts.kc_nest = true,
            "--contract" => opts.contract = true,
            "--rewrite" => rewrite = true,
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
                    Some(mapal_backend_llvm::MovePanel::Off)
                } else {
                    spec.split_once(':')
                        .and_then(|(w, b)| Some((w.parse::<u64>().ok()?, b.parse::<u64>().ok()?)))
                        .filter(|&(w, b)| w > 0 && b > 0)
                        .map(|(w, b)| mapal_backend_llvm::MovePanel::Force(w, b))
                };
                let Some(mp) = parsed else {
                    eprintln!(
                        "--move-panel expects `off` or <W>:<B> with both positive, got `{spec}`"
                    );
                    return ExitCode::from(2);
                };
                opts.move_panel = mp;
            }
            // Machine facts by name (plan-s31-target-profiles): `generic` is
            // the default and reproduces today's literals; `apple-m` and
            // `zen3` differ. Never probed — a box run names its machine.
            // S52: where the device (nvptx64) translation unit goes on a GPU
            // profile. Required when one is produced — a GPU program whose
            // kernels were silently dropped would fail only at launch.
            // S52: measurement-only OFF arms for the GPU rungs (an A/B needs
            // the same compiler with one rung off). Unknown names are errors.
            _ if a.starts_with("--gpu-off=") => {
                for rung in a["--gpu-off=".len()..].split(',') {
                    match rung {
                        "interleave" => opts.gpu.interleave = false,
                        "unroll" => opts.gpu.unroll = false,
                        "min-blocks" => opts.gpu.min_blocks = false,
                        "warp" => opts.gpu.warp = false,
                        "prefetch" => opts.gpu.prefetch = false,
                        "mma" => opts.gpu.mma = false,
                        "async" => opts.gpu.cp_async = false,
                        "wide" => opts.gpu.wide = false,
                        "ldmatrix" => opts.gpu.ldmatrix = false,
                        other => {
                            eprintln!(
                                "--gpu-off: unknown rung `{other}` (interleave, unroll, min-blocks, warp, prefetch, mma, async, wide, ldmatrix)"
                            );
                            return ExitCode::from(2);
                        }
                    }
                }
            }
            // A reduced-precision face (ADR-0032 D1): changes values by the
            // format's bound and admits the tensor cores.
            "--precision=tf32" => opts.reduced = Some(mapal_backend_llvm::Reduced::Tf32),
            _ if a.starts_with("--device-out=") => {
                device_out = Some(a["--device-out=".len()..].to_owned());
            }
            _ if a.starts_with("--target=") => {
                opts.target = a["--target=".len()..].to_owned().leak();
            }
            other => {
                eprintln!(
                    "unknown flag: {other} (usage: emit <file.mapal> [-] [--perf] [--no-tile] [--no-pack] [--kc] [--contract] [--rewrite] [--target=<name>] [--device-out=<file.ll>])"
                );
                return ExitCode::from(2);
            }
        }
    }
    let src = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("read {path}: {e}");
            return ExitCode::from(1);
        }
    };
    let po = mapal_syntax::parse(&src);
    if !po.diagnostics.is_empty() {
        eprintln!("parse diagnostics: {:?}", po.diagnostics);
        return ExitCode::from(1);
    }
    let ir = match mapal_lower::lower(&src, &po.program) {
        Ok(ir) => ir,
        Err(d) => {
            eprintln!("lower: {d:?}");
            return ExitCode::from(1);
        }
    };
    let ir = if rewrite {
        let r = mapal_rewrite::rewrite(ir);
        r.ir
    } else {
        ir
    };
    match mapal_backend_llvm::emit_detailed(&ir, &opts) {
        Ok(em) => {
            match (&em.device, &device_out) {
                (Some(dev), Some(p)) => std::fs::write(p, dev).expect("write device .ll"),
                (Some(_), None) => {
                    eprintln!(
                        "emit: target `{}` offloads tile sites; pass --device-out=<file.ll>",
                        em.profile
                    );
                    return ExitCode::from(2);
                }
                (None, _) => {}
            }
            let ll = em.text;
            if to_stdout {
                print!("{ll}");
            } else {
                let stem = path.strip_suffix(".mapal").unwrap_or(&path);
                let out = if opts.perf_timing {
                    format!("{stem}_perf.ll")
                } else {
                    format!("{stem}.ll")
                };
                std::fs::write(&out, ll).expect("write .ll");
                eprintln!("wrote {out}");
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("emit: {e:?}");
            ExitCode::from(1)
        }
    }
}
