//! The GPU emulator harness the integration tests share (S54 `retain_delta`,
//! S62 `splitk`): the device unit compiled for the host and run thread by
//! thread through `crates/mapal-rt/gpu/mapal_gpu_emu.c`.

use std::process::Command;

/// The device unit compiled for the host: special registers become the
/// emulator's `emu_*` (`mapal_gpu_emu.c`), global and shared pointers generic
/// (shared memory is `emu_smem`, `barrier0` is `emu_barrier`), the PTX-only
/// inline asm (`griddepcontrol`, the raster's opaque `mov`) dropped or an
/// identity, and an atomic's device scope the host's. `cp.async` is not
/// emulated: tests that reach a staged kernel turn `async` off.
pub fn host_device(dev: &str) -> String {
    let mut out = String::new();
    for line in dev.lines() {
        let nvvm_decl = line.starts_with("declare")
            && (line.contains("@llvm.nvvm.") || line.contains("cp.async"));
        if line.starts_with("target ")
            || line.starts_with("@mapal_smem")
            || nvvm_decl
            || line.contains("asm sideeffect \"griddepcontrol")
        {
            continue;
        }
        let mut l = line
            .replace("ptx_kernel ", "")
            .replace("addrspace(1) ", "")
            .replace("addrspace(3) ", "")
            .replace("@mapal_smem", "@emu_smem")
            .replace("call void @llvm.nvvm.barrier0()", "call void @emu_barrier()")
            .replace(" syncscope(\"device\")", "");
        for (ty, bits, c) in [("i32", 32, "r"), ("i64", 64, "l")] {
            let asm = format!("call {ty} asm \"mov.b{bits} $0, $1;\", \"={c},{c}\"({ty} ");
            if let Some(i) = l.find(&asm) {
                let v = l[i + asm.len()..].trim_end_matches(')').to_string();
                l = format!("{}add {ty} {v}, 0", &l[..i]);
            }
        }
        // A generic-to-global cast is an identity once both are generic.
        if let Some((lhs, rest)) = l.split_once(" = addrspacecast ptr ")
            && let Some(v) = rest.strip_suffix(" to ptr")
        {
            l = format!("{lhs} = getelementptr i8, ptr {v}, i64 0");
        }
        for (r, e) in [
            ("tid.x", "tid_x"),
            ("ctaid.x", "ctaid_x"),
            ("ctaid.y", "ctaid_y"),
        ] {
            l = l.replace(
                &format!("@llvm.nvvm.read.ptx.sreg.{r}()"),
                &format!("@emu_{e}()"),
            );
        }
        out.push_str(&l);
        out.push('\n');
    }
    out + "declare i32 @emu_tid_x()\ndeclare i32 @emu_ctaid_x()\ndeclare i32 @emu_ctaid_y()\n\
           declare void @emu_barrier()\n@emu_smem = external global [0 x i8], align 16\n"
}

/// Build and run `host` + `dev` against the emulated device; its stdout.
pub fn run_emulated(clang: &str, host: &str, dev: &str) -> String {
    let dir = tempfile::tempdir().unwrap();
    let p = |n: &str| dir.path().join(n);
    std::fs::write(p("h.ll"), host).unwrap();
    std::fs::write(p("d.ll"), host_device(dev)).unwrap();
    let emu = format!(
        "{}/../../mapal-rt/gpu/mapal_gpu_emu.c",
        env!("CARGO_MANIFEST_DIR")
    );
    let ok = Command::new(clang)
        .args(["-O1", "-Wno-override-module", "-c"])
        .arg(p("d.ll"))
        .arg("-o")
        .arg(p("d.o"))
        .status()
        .unwrap()
        .success();
    assert!(ok, "device unit does not compile for the host");
    let mut link = Command::new(clang);
    link.args(["-O1", "-Wno-override-module"])
        .arg(p("h.ll"))
        .arg(p("d.o"))
        .arg(&emu)
        .arg(rt_lib())
        .arg("-o")
        .arg(p("x"));
    if cfg!(target_os = "linux") {
        // The emulator finds kernels by name with `dlsym`.
        link.args(["-rdynamic", "-ldl", "-lm"]);
    }
    assert!(link.status().unwrap().success(), "link failed");
    let out = Command::new(p("x")).output().unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

pub fn clang() -> Option<String> {
    if let Ok(cc) = std::env::var("CC") {
        return Some(cc);
    }
    let found = Command::new("which")
        .arg("clang")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
    assert!(
        !(found.is_none() && std::env::var("MAPAL_REQUIRE_CLANG").is_ok_and(|v| v != "0")),
        "MAPAL_REQUIRE_CLANG is set but clang was not found"
    );
    found
}

pub fn rt_lib() -> std::path::PathBuf {
    static BUILD: std::sync::Once = std::sync::Once::new();
    BUILD.call_once(|| {
        let ok = Command::new("cargo")
            .args(["build", "-p", "mapal-rt"])
            .status()
            .is_ok_and(|s| s.success());
        assert!(ok, "cargo build -p mapal-rt failed");
    });
    format!(
        "{}/../../../target/debug/libmapal_rt.a",
        env!("CARGO_MANIFEST_DIR")
    )
    .into()
}
