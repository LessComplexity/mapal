//! The GPU emulator harness the integration tests share (S54 `retain_delta`,
//! S62 `splitk`): the device unit compiled for the host and run thread by
//! thread through `crates/mapal-rt/gpu/mapal_gpu_emu.c`.

use std::process::{Command, Stdio};
use std::sync::{Condvar, Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant};

/// Child processes (clang, test binaries) this test binary runs at once, across
/// all of its tests: a quarter of the CPUs this process may run on (a child
/// runs a 2-lane pool, an emulated block's threads beside it; at half the
/// CPUs the 32-CPU pin ran its cores at 89–93 C), or `MAPAL_TEST_JOBS`.
///
/// Unbounded, each sweep ran `available_parallelism` workers, each starting a
/// binary whose runtime pool was itself `available_parallelism` wide: 14 × 14
/// runnable threads on 14 cores — the runtime's own oversubscription cliff
/// (`mapal-rt` `spin_iters`, 1500x at 14 executors) — so programs ran into the
/// 15 s timeout with every core pegged and the machine unusable.
pub fn job_slots() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("MAPAL_TEST_JOBS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| n >= 1)
            .unwrap_or_else(|| {
                let cpus = std::thread::available_parallelism().map_or(4, |n| n.get());
                (cpus / 4).max(1)
            })
    })
}

/// Children in flight, bounded by [`job_slots`].
static IN_FLIGHT: (Mutex<usize>, Condvar) = (Mutex::new(0), Condvar::new());

/// Hold one of [`job_slots`] while a child process runs.
pub struct Slot;

pub fn slot() -> Slot {
    let (m, cv) = &IN_FLIGHT;
    let guard = m.lock().unwrap_or_else(PoisonError::into_inner);
    let mut used = cv
        .wait_while(guard, |n| *n >= job_slots())
        .unwrap_or_else(PoisonError::into_inner);
    *used += 1;
    Slot
}

impl Drop for Slot {
    fn drop(&mut self) {
        let (m, cv) = &IN_FLIGHT;
        *m.lock().unwrap_or_else(PoisonError::into_inner) -= 1;
        cv.notify_one();
    }
}

/// The device unit compiled for the host: special registers become the
/// emulator's `emu_*` (`mapal_gpu_emu.c`), global and shared pointers generic
/// (shared memory is `emu_smem`, `barrier0` is `emu_barrier`), the PTX-only
/// inline asm (`griddepcontrol`, the raster's opaque `mov`) dropped or an
/// identity, and an atomic's device scope the host's. `cp.async` is the
/// emulator's deferred per-thread copy queue (`emu_cp_async`, applied at
/// `emu_cp_wait`), so a missing wait reads stale shared memory.
pub fn host_device(dev: &str) -> String {
    let mut out = String::new();
    // tensor-cores: an `ldmatrix` operand's 32-bit shared address, back to the
    // pointer it was cast from (`emu_ldsm` reads the emulator's shared memory).
    let mut shared32: std::collections::HashMap<String, String> = Default::default();
    // kernel-merging `calls`: the grid barrier is the emulator's
    // (`mapal_gsync` in `mapal_gpu_emu.c`), its PTX body dropped.
    let mut in_gsync = false;
    // regressions: every pointer alignment a kernel declares (`align N %p`,
    // the facts its vector loads are formed from) is checked at the kernel's
    // entry on every emulated launch — a claim that does not hold aborts the
    // run (on the device it is a misaligned-address fault).
    let mut entry_checks: Vec<String> = Vec::new();
    for line in dev.lines() {
        if !entry_checks.is_empty() {
            let label = !line.starts_with(' ') && line.ends_with(':');
            if label {
                out.push_str(line);
                out.push('\n');
            }
            for c in entry_checks.drain(..) {
                out.push_str(&c);
                out.push('\n');
            }
            if label {
                continue;
            }
        }
        if line.starts_with("define ptx_kernel ") {
            let mut rest = line;
            while let Some(i) = rest.find(" align ") {
                rest = &rest[i + " align ".len()..];
                let n: String = rest.chars().take_while(char::is_ascii_digit).collect();
                let name: String = rest[n.len()..]
                    .trim_start()
                    .chars()
                    .take_while(|c| *c != ',' && *c != ')')
                    .collect();
                if n.parse::<u64>().is_ok_and(|n| n > 1) && name.starts_with('%') {
                    entry_checks.push(format!("  call void @emu_align(ptr {name}, i64 {n})"));
                }
            }
        }
        if line.starts_with("define internal void @mapal_gsync(") {
            in_gsync = true;
            continue;
        }
        if in_gsync {
            in_gsync = line != "}";
            continue;
        }
        let nvvm_decl = line.starts_with("declare")
            && (line.contains("@llvm.nvvm.") || line.contains("cp.async"));
        if line.starts_with("target ")
            || line.starts_with("@mapal_smem")
            || nvvm_decl
            || line.contains("asm sideeffect \"griddepcontrol")
            // TMA `bulk-stage`: the proxy and barrier-init fences order
            // nothing in the emulator's one memory.
            || line.contains("asm sideeffect \"fence.")
            // hopper-wgmma-tma: register reallocation and the warpgroup's
            // operand fence order nothing in the emulator.
            || line.contains("asm sideeffect \"setmaxnreg.")
            || line.contains("asm sideeffect \"wgmma.fence.")
        {
            continue;
        }
        let mut l = line
            .replace("ptx_kernel ", "")
            // hopper-wgmma-tma: a tensor map by value is the emulator's map
            // by pointer (`emu_tmaps`).
            .replace(" byval([128 x i8]) align 64 \"nvvm.grid_constant\"", "")
            .replace("addrspace(1) ", "")
            .replace("addrspace(3) ", "")
            .replace("@mapal_smem", "@emu_smem")
            .replace(
                "call void @llvm.nvvm.barrier0()",
                "call void @emu_barrier()",
            )
            // kernel-merging `horizontal`: a member's barrier of its own threads.
            .replace(
                "call void @llvm.nvvm.barrier.sync.cnt(i32 0, ",
                "call void @emu_barrier_n(",
            )
            // kernel-merging `calls`: the tile boundary's own barrier.
            .replace(
                "call void @llvm.nvvm.barrier.sync.cnt(i32 1, ",
                "call void @emu_barrier_k(i32 1, ",
            )
            .replace(" syncscope(\"device\")", "");
        for (ty, bits, c) in [("i32", 32, "r"), ("i64", 64, "l")] {
            let asm = format!("call {ty} asm \"mov.b{bits} $0, $1;\", \"={c},{c}\"({ty} ");
            if let Some(i) = l.find(&asm) {
                let v = l[i + asm.len()..].trim_end_matches(')').to_string();
                l = format!("{}add {ty} {v}, 0", &l[..i]);
            }
        }
        // gapfix `stream-ld`: a streaming load (`ld.global.cs.nc`, the L2's
        // eviction priority) is a plain load of the same words.
        if let Some(i) = l.find(" asm \"ld.global.cs.nc.")
            && let Some(call) = l[..i].rfind("call ")
            && let Some((_, p)) = l[i..].rsplit_once("(ptr ")
        {
            let ty = l[call + "call ".len()..i].to_string();
            let p = p.trim_end_matches(')');
            l = format!("{}load {ty}, ptr {p}", &l[..call]);
        }
        // tile-efficiency T2: `cp.async.{ca,cg}.shared.global.<n>(dst, src)`
        // queues an `n`-byte copy; commit and wait drive the queue.
        if let Some(i) = l.find("@llvm.nvvm.cp.async.")
            && let Some(j) = l[i..].find(".shared.global.")
        {
            let rest = &l[i + j + ".shared.global.".len()..];
            if let Some((n, args)) = rest.split_once('(') {
                let args = args.trim_end_matches(')');
                l = format!("{}@emu_cp_async({args}, i32 {n})", &l[..i]);
            }
        }
        // tensor-cores (ADR-0043 D6): the 16-bit matrix instructions as the
        // emulator's warp meets; a struct result rebuilt from `emu_frag`.
        if let Some((lhs, rest)) = l.split_once(" = ptrtoint ptr ")
            && let Some(p) = rest.strip_suffix(" to i32")
        {
            shared32.insert(lhs.trim().to_string(), p.to_string());
        }
        // ADR-0043 slice 3: the device's fp8 roundings (`crate::narrow`).
        for f in ["e4m3", "e5m2"] {
            let asm = format!("asm \"cvt.rn.satfinite.{f}x2.f32 $0, $1, $2;\", \"=h,f,f\"(");
            l = l.replace(&asm, &format!("@emu_cvt_{f}x2("));
            let wid = format!("asm \"cvt.rn.f16x2.{f}x2 $0, $1;\", \"=r,h\"(");
            l = l.replace(&wid, &format!("@emu_cvt_f16x2_{f}x2("));
        }
        for (f, e) in [("bf16", "bf16x2"), ("f16", "f16x2")] {
            let asm = format!("asm \"cvt.rn.{f}x2.f32 $0, $1, $2;\", \"=r,f,f\"(");
            l = l.replace(&asm, &format!("@emu_cvt_{e}("));
        }
        if let Some(i) = l.find("asm sideeffect \"ldmatrix.sync.aligned.m8n8.x")
            && let Some((lhs, _)) = l.split_once(" = call ")
        {
            let n: usize = l[i + "asm sideeffect \"ldmatrix.sync.aligned.m8n8.x".len()..][..1]
                .parse()
                .expect("x1/x2/x4");
            let trans = u8::from(l[i..].contains(".trans"));
            let a = l
                .rsplit_once("(i32 ")
                .expect("address")
                .1
                .trim_end_matches(')');
            let p = shared32.get(a).expect("the address's pointer").clone();
            let lhs = lhs.trim();
            let ty = vec!["i32"; n].join(", ");
            let mut text = format!("  call void @emu_ldsm(ptr {p}, i32 {n}, i32 {trans})\n");
            let mut prev = "poison".to_string();
            for j in 0..n {
                let (x, s) = (format!("{lhs}_x{j}"), format!("{lhs}_s{j}"));
                let s = if j + 1 == n { lhs.to_string() } else { s };
                text += &format!(
                    "  {x} = call i32 @emu_frag(i32 {j})\n  {s} = insertvalue {{ {ty} }} {prev}, i32 {x}, {j}\n"
                );
                prev = s;
            }
            l = text.trim_end().to_string();
        }
        // TMA `bulk-stage` tests: the TF32 tile's rounding and its
        // `m16n8k8` as the same warp meet (`emu_mma16`'s format 2).
        l = l
            .replace(
                "asm \"cvt.rna.tf32.f32 $0, $1;\", \"=r,r\"(",
                "@emu_cvt_tf32(",
            )
            .replace(
                "asm \"cvt.rna.tf32.f32 $0, $1;\", \"=r,f\"(",
                "@emu_cvt_tf32f(",
            );
        for (f, f16) in [("bf16", 0), ("f16", 1), ("tf32", 2)] {
            let shape = if f16 == 2 { "m16n8k8" } else { "m16n8k16" };
            let asm = format!("asm \"mma.sync.aligned.{shape}.row.col.f32.{f}.{f}.f32 ");
            if let Some(i) = l.find(&asm)
                && let Some((lhs, _)) = l.split_once(" = call ")
                && let Some(j) = l[i..].find("\"(")
            {
                let args = l[i + j + 2..].trim_end_matches(')').to_string();
                let lhs = lhs.trim();
                let ty = "{ float, float, float, float }";
                let mut text = format!("  call void @emu_mma16(i32 {f16}, {args})\n");
                let mut prev = "poison".to_string();
                for q in 0..4 {
                    let (x, s) = (format!("{lhs}_x{q}"), format!("{lhs}_s{q}"));
                    let s = if q == 3 { lhs.to_string() } else { s };
                    text += &format!(
                        "  {x} = call float @emu_fragf(i32 {q})\n  {s} = insertvalue {ty} {prev}, float {x}, {q}\n"
                    );
                    prev = s;
                }
                l = text.trim_end().to_string();
            }
        }
        // hopper-wgmma-tma: RNE TF32 rounding; `wgmma` as the warpgroup's
        // meet (`emu_wgmma`: accumulators in by `emu_acc_set`, out by
        // `emu_fragf`); its groups (`emu_wg_commit`, `emu_wg_wait`).
        for pre in ["asm sideeffect", "asm"] {
            l = l.replace(
                &format!("{pre} \"cvt.rn.tf32.f32 $0, $1;\", \"=r,f\"("),
                "@emu_cvt_rn_tf32f(",
            );
        }
        l = l.replace(
            "call void asm sideeffect \"wgmma.commit_group.sync.aligned;\", \"~{memory}\"()",
            "call void @emu_wg_commit()",
        );
        if let Some(i) = l.find("call void asm sideeffect \"wgmma.wait_group.sync.aligned ") {
            let rest = &l[i + "call void asm sideeffect \"wgmma.wait_group.sync.aligned ".len()..];
            let n: String = rest.chars().take_while(char::is_ascii_digit).collect();
            l = format!("{}call void @emu_wg_wait(i32 {n})", &l[..i]);
        }
        if let Some(i) = l.find("asm sideeffect \"wgmma.mma_async.sync.aligned.m64n")
            && let Some((lhs, _)) = l.split_once(" = call ")
            && let Some((_, args)) = l.rsplit_once("\"(")
        {
            let n: usize = l[i + "asm sideeffect \"wgmma.mma_async.sync.aligned.m64n".len()..]
                .split('k')
                .next()
                .and_then(|x| x.parse().ok())
                .expect("m64nNk8");
            let args: Vec<&str> = args.trim_end_matches(')').split(", ").collect();
            let lhs = lhs.trim();
            let ty = format!("{{ {} }}", vec!["float"; n / 2].join(", "));
            let mut text = String::new();
            for (q, v) in args[5..].iter().enumerate() {
                text += &format!("  call void @emu_acc_set(i32 {q}, {v})\n");
            }
            text += &format!(
                "  call void @emu_wgmma(i32 {n}, {})\n",
                args[..5].join(", ")
            );
            let mut prev = "poison".to_string();
            for q in 0..n / 2 {
                let (x, s) = (format!("{lhs}_x{q}"), format!("{lhs}_s{q}"));
                let s = if q + 1 == n / 2 { lhs.to_string() } else { s };
                text += &format!(
                    "  {x} = call float @emu_fragf(i32 {q})\n  {s} = insertvalue {ty} {prev}, float {x}, {q}\n"
                );
                prev = s;
            }
            l = text.trim_end().to_string();
        }
        // hopper-wgmma-tma: a tensor box (`emu_bulk_tensor`: its map, five
        // coordinates, its barrier) and a counted arrival (`emu_mbar_arrive`).
        if l.contains("asm sideeffect \"cp.async.bulk.tensor.")
            && let Some(i) = l.find("call ")
            && let Some((_, args)) = l.rsplit_once("\"(")
        {
            let args: Vec<String> = args
                .trim_end_matches(')')
                .split(", ")
                .map(String::from)
                .collect();
            let ptr = |a: &str| {
                let v = a.strip_prefix("i32 ").expect("a shared address");
                format!("ptr {}", shared32.get(v).expect("its pointer"))
            };
            let mut c: Vec<String> = args[2..args.len() - 1].to_vec();
            c.resize(5, "i32 0".into());
            l = format!(
                "{}call void @emu_bulk_tensor({}, {}, {}, {})",
                &l[..i],
                ptr(&args[0]),
                args[1],
                c.join(", "),
                ptr(&args[args.len() - 1])
            );
        }
        // TMA `bulk-stage`: the transaction barriers and the bulk copies as
        // the emulator's (`emu_mbar_*`, `emu_bulk`: a copy is applied when a
        // thread waits on its barrier); a 32-bit shared address back to its
        // pointer.
        for (marker, func, ret) in [
            ("mbarrier.init.shared", "emu_mbar_init", "void"),
            ("mbarrier.arrive.expect_tx", "emu_mbar_expect", "void"),
            ("cp.async.bulk.shared", "emu_bulk", "void"),
            ("mbarrier.try_wait.parity", "emu_mbar_try", "i32"),
            (
                "mbarrier.arrive.shared::cta.b64 _",
                "emu_mbar_arrive",
                "void",
            ),
        ] {
            if l.contains(marker)
                && let Some(i) = l.find("call ")
                && let Some((_, args)) = l.rsplit_once("\"(")
            {
                let args: Vec<String> = args
                    .trim_end_matches(')')
                    .split(", ")
                    .map(
                        |a| match a.strip_prefix("i32 ").and_then(|v| shared32.get(v)) {
                            Some(p) => format!("ptr {p}"),
                            None => a.to_string(),
                        },
                    )
                    .collect();
                l = format!("{}call {ret} @{func}({})", &l[..i], args.join(", "));
            }
        }
        // f32-tensor-cores `split-tile`: the guard's `vote.sync.any` as the
        // emulator's warp meet.
        if let Some(i) =
            l.find("asm sideeffect \"{ .reg .pred p, q; setp.ne.u32 p, $1, 0; vote.sync.any.pred")
            && let Some((_, arg)) = l.rsplit_once("\"(")
        {
            l = format!("{}@emu_vote_any({arg}", &l[..i]);
        }
        // ADR-0042: the hardware `exp` form's `ex2.approx` is the host's `exp2f`.
        l = l.replace("@llvm.nvvm.ex2.approx.ftz.f(", "@emu_ex2(");
        l = l
            .replace("@llvm.nvvm.cp.async.commit.group()", "@emu_cp_commit()")
            .replace("@llvm.nvvm.cp.async.wait.group(", "@emu_cp_wait(");
        // A generic-to-global cast is an identity once both are generic.
        if let Some((lhs, rest)) = l.split_once(" = addrspacecast ptr ")
            && let Some(v) = rest.strip_suffix(" to ptr")
        {
            l = format!("{lhs} = getelementptr i8, ptr {v}, i64 0");
        }
        // `cell-lanes`: a warp's butterfly shuffle meets its 31 lanes in
        // the emulator (`emu_shfl_bfly`; the kernel runs threaded).
        if l.contains("@llvm.nvvm.shfl.sync.bfly.f32(i32 -1, ") {
            l = l
                .replace("@llvm.nvvm.shfl.sync.bfly.f32(i32 -1, ", "@emu_shfl_bfly(")
                .replace(", i32 31)", ")");
        }
        for (r, e) in [
            ("laneid", "laneid"),
            ("nctaid.x", "nctaid_x"),
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
    // A module with shuffles runs its cell launches threaded.
    let lanes = if out.contains("@emu_shfl_bfly(") {
        "@emu_threaded = global i32 1\n"
    } else {
        ""
    };
    // The entry checks' callee: abort where the address is not a multiple.
    let align = "define internal void @emu_align(ptr %p, i64 %a) {\n  %i = ptrtoint ptr %p to i64\n  \
                 %m = urem i64 %i, %a\n  %z = icmp eq i64 %m, 0\n  br i1 %z, label %ok, label %bad\nbad:\n  \
                 call void @abort()\n  unreachable\nok:\n  ret void\n}\n";
    let abort = match out.contains("declare void @abort()") {
        true => "",
        false => "declare void @abort()\n",
    };
    out + lanes
        + align
        + abort
        + "declare i32 @emu_tid_x()\ndeclare i32 @emu_ctaid_x()\ndeclare i32 @emu_ctaid_y()\n\
           declare i32 @emu_nctaid_x()\ndeclare void @emu_barrier_k(i32, i32)\ndeclare void @mapal_gsync(ptr)\n\
           declare i32 @emu_laneid()\ndeclare float @emu_shfl_bfly(float, i32)\n\
           declare void @emu_barrier()\ndeclare void @emu_barrier_n(i32)\n@emu_smem = external global [0 x i8], align 16\n\
           declare void @emu_cp_async(ptr, ptr, i32)\ndeclare void @emu_cp_commit()\n\
           declare void @emu_cp_wait(i32)\ndeclare float @emu_ex2(float)\n\
           declare void @emu_ldsm(ptr, i32, i32)\ndeclare i32 @emu_frag(i32)\ndeclare float @emu_fragf(i32)\n\
           declare i32 @emu_cvt_bf16x2(float, float)\ndeclare i32 @emu_cvt_f16x2(float, float)\n\
           declare i16 @emu_cvt_e4m3x2(float, float)\ndeclare i16 @emu_cvt_e5m2x2(float, float)\n\
           declare i32 @emu_cvt_f16x2_e4m3x2(i16)\ndeclare i32 @emu_cvt_f16x2_e5m2x2(i16)\n\
           declare void @emu_mma16(i32, i32, i32, i32, i32, i32, i32, float, float, float, float)\n\
           declare i32 @emu_cvt_tf32(i32)\ndeclare i32 @emu_cvt_tf32f(float)\ndeclare i32 @emu_vote_any(i32)\n\
           declare void @emu_mbar_init(ptr, i32, i32)\ndeclare void @emu_mbar_expect(ptr, i32, i32)\n\
           declare void @emu_bulk(ptr, ptr, i32, ptr, i32)\ndeclare i32 @emu_mbar_try(ptr, i32)\n\
           declare i32 @emu_cvt_rn_tf32f(float)\ndeclare void @emu_acc_set(i32, float)\n\
           declare void @emu_wgmma(i32, i32, i32, i32, i32, i64)\ndeclare void @emu_wg_commit()\n\
           declare void @emu_wg_wait(i32)\ndeclare void @emu_mbar_arrive(ptr)\n\
           declare void @emu_bulk_tensor(ptr, i64, i32, i32, i32, i32, i32, ptr)\n"
}

/// Emulation runs every device thread on the host, so it gets 8x the native timeout.
const EMU_TIMEOUT_SECS: u64 = 120;

/// Build and run `host` + `dev` against the emulated device; its stdout.
#[allow(dead_code)]
pub fn run_emulated(clang: &str, host: &str, dev: &str) -> String {
    run_emulated_totals(clang, host, dev).0
}

/// [`run_emulated`], and the run's device traffic (`emu_total`: uploads,
/// copy-outs, launches and their bytes) as `[h2d, d2h, launches, bytes_in,
/// bytes_out]`.
#[allow(dead_code)]
pub fn run_emulated_totals(clang: &str, host: &str, dev: &str) -> (String, [u64; 5]) {
    let (out, err) = build_emulated(clang, host, dev).run(&[]);
    (out, totals(&err))
}

/// The emulator compiled once — per clang (its `--version`) and the text of
/// `mapal_gpu_emu.c` + `mapal_choice.h` — into the tests' target tmpdir;
/// every emulated program links that object. Compiling the emulator inside
/// each program's link was ~0.25 s of every build: about two thirds of a
/// generated-program test's clang time (generic_fusion: 1082 clang runs,
/// 197 s). The object is the same code the link compiled (`-O1`, no other
/// flag), so every program is the binary it was.
fn emu_obj(clang: &str) -> std::path::PathBuf {
    static OBJS: Mutex<Vec<(String, std::path::PathBuf)>> = Mutex::new(Vec::new());
    let mut objs = OBJS.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some((_, o)) = objs.iter().find(|(c, _)| c == clang) {
        return o.clone();
    }
    let dir = format!("{}/../../mapal-rt/gpu", env!("CARGO_MANIFEST_DIR"));
    let version = Command::new(clang)
        .arg("--version")
        .output()
        .expect("clang --version")
        .stdout;
    let mut key: u64 = 0xcbf2_9ce4_8422_2325;
    for f in ["mapal_gpu_emu.c", "mapal_choice.h"] {
        let text = std::fs::read(format!("{dir}/{f}")).expect("emulator source");
        for b in clang.bytes().chain(version.iter().copied()).chain(text) {
            key = (key ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    let obj = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("emu-{key:016x}.o"));
    if !obj.exists() {
        // Another test binary may be compiling it too: each writes its own
        // file, the rename is atomic.
        let tmp = obj.with_extension(format!("{}.o", std::process::id()));
        let _slot = slot();
        let ok = Command::new(clang)
            .args(["-O1", "-c"])
            .arg(format!("{dir}/mapal_gpu_emu.c"))
            .arg("-o")
            .arg(&tmp)
            .status()
            .is_ok_and(|s| s.success());
        assert!(ok, "the emulator does not compile");
        std::fs::rename(&tmp, &obj).expect("emulator object");
    }
    objs.push((clang.to_string(), obj.clone()));
    obj
}

/// `ld.lld` is on the path (`MAPAL_EMU_LLD=0`: the default linker).
fn lld() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("MAPAL_EMU_LLD").map_or(true, |v| v != "0")
            && Command::new("ld.lld")
                .arg("--version")
                .output()
                .is_ok_and(|o| o.status.success())
    })
}

/// A program built against the emulated device, run as often as a test
/// needs (per-site-choice: once per candidate index).
#[allow(dead_code)]
pub struct Emulated {
    dir: tempfile::TempDir,
}

/// Build `host` + `dev` against the emulated device.
#[allow(dead_code)]
pub fn build_emulated(clang: &str, host: &str, dev: &str) -> Emulated {
    let dir = tempfile::tempdir().unwrap();
    let p = |n: &str| dir.path().join(n);
    std::fs::write(p("h.ll"), host).unwrap();
    std::fs::write(p("d.ll"), host_device(dev)).unwrap();
    let emu = emu_obj(clang);
    let _slot = slot();
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
        // lld where installed: the same program, linked in half the time
        // (the debug runtime archive: 54 -> 26 ms a link).
        if lld() {
            link.arg("-fuse-ld=lld");
        }
    }
    assert!(link.status().unwrap().success(), "link failed");
    Emulated { dir }
}

impl Emulated {
    /// One run with `env` set: its stdout and stderr. Time-boxed: an
    /// emulated program that never finishes fails the test instead of
    /// pinning a core until someone kills it.
    #[allow(dead_code)]
    pub fn run(&self, env: &[(&str, &str)]) -> (String, String) {
        let _slot = slot();
        let mut child = Command::new(self.dir.path().join("x"))
            .env("MAPAL_PAR", "2")
            .env("MAPAL_SPIN", "0")
            .envs(env.iter().copied())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let start = Instant::now();
        while child.try_wait().unwrap().is_none() {
            if start.elapsed() > Duration::from_secs(EMU_TIMEOUT_SECS) {
                let _ = child.kill();
                panic!("emulated run timed out after {EMU_TIMEOUT_SECS} s");
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        let out = child.wait_with_output().unwrap();
        assert!(
            out.status.success(),
            "{}: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        );
        (
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    }

    /// [`Emulated::run`] once per environment (the emulator's modes), the
    /// runs side by side — each holds one of [`job_slots`] — their stdout
    /// and stderr in `envs`' order.
    #[allow(dead_code)]
    pub fn run_each<'e, E>(&self, envs: &[E]) -> Vec<(String, String)>
    where
        E: AsRef<[(&'e str, &'e str)]> + Sync,
    {
        std::thread::scope(|s| {
            let hs: Vec<_> = envs
                .iter()
                .map(|e| s.spawn(move || self.run(e.as_ref())))
                .collect();
            hs.into_iter()
                .map(|h| h.join().unwrap_or_else(|p| std::panic::resume_unwind(p)))
                .collect()
        })
    }
}

/// A run's `emu_total` line as `[h2d, d2h, launches, bytes_in, bytes_out]`.
#[allow(dead_code)]
pub fn totals(err: &str) -> [u64; 5] {
    let mut totals = [0u64; 5];
    if let Some(l) = err.lines().find_map(|l| l.strip_prefix("emu_total ")) {
        for (t, kv) in totals.iter_mut().zip(l.split_whitespace()) {
            *t = kv
                .split_once('=')
                .and_then(|(_, v)| v.parse().ok())
                .unwrap_or(0);
        }
    }
    totals
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
