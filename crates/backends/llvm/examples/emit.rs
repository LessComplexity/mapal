//! Emit textual LLVM IR for a `.mapal` file (ADR-0020; dev tool — the future
//! `mapal build` embryo). Usage: `cargo run -p mapal-backend-llvm --example emit -- <file.mapal> [-] [--perf] [--no-tile] [--no-pack] [--kc] [--contract] [--rewrite] [--target=<name>] [--host=<name>] [--device-out=<file.ll>] [--precision=tf32|reassoc] [--splitk=<S>]`
//! (writes `<file>.ll` next to the source, `<file>_perf.ll` with `--perf`, or
//! stdout with `-`; `--rewrite` rewrites the lowered IR before emission).

use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!(
            "usage: emit <file.mapal> [-] [--perf] [--no-tile] [--no-pack] [--kc] [--contract] [--rewrite] [--target=<name>] [--host=<name>] [--device-out=<file.ll>] [--precision=tf32|reassoc] [--splitk=<S>]"
        );
        return ExitCode::from(2);
    };
    let mut to_stdout = false;
    let mut rewrite = false;
    let mut device_out: Option<String> = None;
    let mut host_given = false;
    let mut opts = mapal_backend_llvm::EmitOpts::default();
    for a in args {
        host_given |= a.starts_with("--host=");
        match mapal_backend_llvm::flags::apply(&mut opts, &a) {
            Ok(true) => continue,
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::from(2);
            }
            Ok(false) => {}
        }
        match a.as_str() {
            "-" => to_stdout = true,
            "--rewrite" => rewrite = true,
            // S52: where the device (nvptx64) translation unit goes on a GPU
            // profile. Required when one is produced — a GPU program whose
            // kernels were silently dropped would fail only at launch.
            _ if a.starts_with("--device-out=") => {
                device_out = Some(a["--device-out=".len()..].to_owned());
            }
            other => {
                eprintln!(
                    "unknown flag: {other} (usage: emit <file.mapal> [-] [--perf] [--no-tile] [--no-pack] [--kc] [--contract] [--rewrite] [--target=<name>] [--host=<name>] [--device-out=<file.ll>] [--precision=tf32|reassoc] [--splitk=<S>])"
                );
                return ExitCode::from(2);
            }
        }
    }
    if host_given
        && mapal_backend_llvm::target_profile(opts.target).is_some_and(|p| p.gpu().is_none())
    {
        eprintln!(
            "--host names the host of a GPU target; `{}` is a CPU profile",
            opts.target
        );
        return ExitCode::from(2);
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
