//! S54 `epilogue` (R1, plan-s54 §4.1): a per-cell consumer of a device site's
//! output runs in that site's store. Pinned on the emitted text — which sites
//! take a group, which values are stored, what still crosses the bus — for the
//! multi-computation shapes of `benches/gpu/gen_tiles.py`; the values are the
//! box differential's job (both faces, ON and OFF arms, the interpreter oracle).

use mapal_backend_llvm::{Emission, EmitOpts, GpuArms, Reduced, emit_detailed};
use mapal_ir::CategoryIr;

const N: u64 = 64;

fn lower_src(src: &str) -> CategoryIr {
    let po = mapal_syntax::parse(src);
    assert!(po.diagnostics.is_empty(), "parse: {:?}", po.diagnostics);
    let ir = mapal_lower::lower(src, &po.program).unwrap_or_else(|d| panic!("lower: {d:?}"));
    mapal_rewrite::rewrite(ir).ir
}

fn emit_on(src: &str, target: &'static str, contract: bool, mut gpu: GpuArms) -> Emission {
    // R1 alone (S54 integration): R2 (`devmap`) would place the per-cell maps
    // R1 declines; its own tests pin that.
    *gpu.off.rung("devmap").expect("a rung") = true;
    emit_detailed(
        &lower_src(src),
        &EmitOpts {
            host: "generic",
            target,
            contract,
            gpu,
            ..EmitOpts::exact()
        },
    )
    .expect("emits")
}

fn emit(src: &str, gpu: GpuArms) -> Emission {
    emit_on(src, "rtx-4090", false, gpu)
}

fn on() -> GpuArms {
    GpuArms::default()
}

fn off() -> GpuArms {
    let mut a = GpuArms::default();
    *a.off.rung("epilogue").expect("a rung") = true;
    a
}

fn mapal_main(ll: &str) -> &str {
    let start = ll
        .find("define internal void @mapal_main(")
        .expect("mapal_main");
    let end = start + ll[start..].find("\n}\n").expect("mapal_main end") + 3;
    &ll[start..end]
}

/// The runtime calls of `mapal_main` in order: `launch`, `launchv <n>`,
/// `gen`, `h2d <bytes>`, `d2h <bytes>` (alloc and free dropped).
fn transfers(ll: &str) -> Vec<String> {
    mapal_main(ll)
        .lines()
        .filter_map(|l| {
            let at = l.find("@mapal_gpu_")?;
            let op = &l[at + "@mapal_gpu_".len()..];
            let name = &op[..op.find('(')?];
            let last = op
                .rsplit("i64 ")
                .next()
                .map(|s| s.trim_end_matches(')').to_owned())
                .unwrap_or_default();
            match name {
                "h2d" | "d2h" => Some(format!("{name} {last}")),
                "launchv" => {
                    let n = op.split(", i32 ").nth(2).expect("nptr");
                    let n = n.split(',').next().expect("nptr");
                    Some(format!("launchv {n}"))
                }
                "launch" | "gen" => Some(name.to_owned()),
                _ => None,
            }
        })
        .collect()
}

/// The pointers each `launchv` passes, `"0"` for an unstored site output.
fn launchv_ptrs(ll: &str) -> Vec<Vec<String>> {
    let main = mapal_main(ll);
    let mut out = Vec::new();
    let mut cur: Vec<String> = Vec::new();
    for l in main.lines() {
        if l.contains("x i64], ptr %p") && l.contains("getelementptr") {
            continue;
        }
        if let Some(v) = l.trim().strip_prefix("store i64 ")
            && l.contains(", ptr %t")
        {
            cur.push(v.split(',').next().expect("value").to_owned());
        }
        if l.contains("@mapal_gpu_launchv(") {
            out.push(std::mem::take(&mut cur));
        }
    }
    out
}

fn site(out: &str, a: &str, b: &str) -> String {
    format!(
        "    ta -> map {{ t ->
        t / {N} -> i;
        t % {N} -> j;
        (0.0, krange) -> fold {{ acc, k -> acc + {a}[i * {N} + k] * {b}[k * {N} + j] }}
    }} -> {out};\n"
    )
}

fn operand(name: &str, mul: u32) -> String {
    format!(
        "    ta -> map {{ t -> (t * {mul} + 13) % 101 - 50 -> widen_f32 -> v; v / 7.0 }} -> {name};\n"
    )
}

/// A program over `N × N` cells: `ops` closed-form operands, the body, and
/// the two corners of each output printed (one-element copies).
fn program(ops: &[&str], body: &str, outs: &[&str]) -> String {
    let mut s = format!(
        "fn main() {{\n    {} -> iota -> ta;\n    {N} -> iota -> krange;\n",
        N * N
    );
    for (i, o) in ops.iter().enumerate() {
        s.push_str(&operand(o, 7 + 2 * i as u32));
    }
    s.push_str("    krange -> map { t -> t % 13 - 6 -> widen_f32 -> v; v / 5.0 } -> bias;\n");
    s.push_str(body);
    for o in outs {
        s.push_str(&format!(
            "    {o}[0] -> println;\n    {o}[{}] -> println;\n",
            N * N - 1
        ));
    }
    s + "}\n"
}

fn relu(e: &str) -> String {
    format!("{e} -> u; (u > 0.0) -> {{ -true-> u; -false-> 0.0; }} -> r; r")
}

fn epi() -> String {
    program(
        &["a", "b"],
        &(site("c", "a", "b") + "    c -> map { v -> v * 0.5 + 1.0 } -> d;\n"),
        &["d"],
    )
}

fn bias() -> String {
    program(
        &["a", "b"],
        &(site("c", "a", "b") + &format!("    ta -> map {{ t -> c[t] + bias[t % {N}] }} -> d;\n")),
        &["d"],
    )
}

#[test]
fn epi_runs_in_the_store_and_c_never_exists() {
    let e = emit(&epi(), on());
    // One launch, the two corners of `d`: nothing else crosses.
    assert_eq!(
        transfers(&e.text),
        ["gen", "gen", "launchv 4", "d2h 4", "d2h 4"]
    );
    // `c` has no reader outside the group: its pointer is 0, the kernel has
    // no store to `%o`, and the host map loop is gone.
    assert_eq!(launchv_ptrs(&e.text)[0][2], "0");
    let dev = e.device.expect("a device unit");
    assert!(dev.contains("%y0") && dev.contains("call float @fn"));
    assert!(!mapal_main(&e.text).contains("call float @fn5("));
}

/// S54 integration: one launch never passes two objects on the same arena
/// bytes. A stored member is written by its site's kernel while it reads the
/// site's operands, so it is born at the site (the box differential caught
/// `d` on `a`'s offset: `px_bias_100`'s checksum raced).
#[test]
fn a_launch_never_passes_two_objects_on_the_same_arena_bytes() {
    for src in [bias(), epi()] {
        let mut arms = GpuArms::default();
        arms.on.arena = true;
        let ll = emit_on(&src, "rtx-4090", false, arms).text;
        let main = mapal_main(&ll);
        let (mut bases, mut off, mut slot) = (Vec::new(), Vec::new(), Vec::new());
        let lookup = |v: &[(String, u64)], k: &str| v.iter().find(|(n, _)| n == k).map(|x| x.1);
        for l in main.lines().map(str::trim) {
            if let Some((lhs, _)) = l.split_once(" = call i64 @mapal_gpu_arena(") {
                bases.push(lhs.to_owned());
            } else if let Some((lhs, rhs)) = l.split_once(" = add i64 ")
                && let Some((b, k)) = rhs.split_once(", ")
                && bases.iter().any(|x| x == b)
            {
                off.push((lhs.to_owned(), k.parse::<u64>().expect("a literal offset")));
            } else if let Some(rest) = l.strip_prefix("store i64 ")
                && let Some((v, s)) = rest.split_once(", ptr ")
                && let Some(o) = lookup(&off, v)
            {
                slot.push((s.to_owned(), o));
            } else if let Some((lhs, s)) = l.split_once(" = load i64, ptr ")
                && let Some(o) = lookup(&slot, s)
            {
                off.push((lhs.to_owned(), o));
            }
        }
        for ptrs in launchv_ptrs(&ll) {
            let mut offs: Vec<u64> = ptrs.iter().filter_map(|p| lookup(&off, p)).collect();
            let n = offs.len();
            offs.sort_unstable();
            offs.dedup();
            assert_eq!(offs.len(), n, "{ptrs:?}\n{main}");
            assert!(n >= 3, "the arena holds the launch's objects: {ptrs:?}");
        }
    }
}

#[test]
fn bias_reads_its_vector_in_gmem() {
    let e = emit(&bias(), on());
    // a, b and the bias are built on the device; the bias is the one operand.
    assert_eq!(
        transfers(&e.text),
        ["gen", "gen", "gen", "launchv 5", "d2h 4", "d2h 4"]
    );
    let dev = e.device.expect("a device unit");
    // `c[t]` arrives as a one-cell buffer: the body's device variant.
    assert!(dev.contains("_cell({ ptr, ptr, i32 }"), "{dev}");
}

#[test]
fn indep_joins_the_later_site_and_reads_the_earlier_one() {
    let src = program(
        &["a", "b", "e", "f"],
        &(site("c", "a", "b")
            + &site("d", "e", "f")
            + "    ta -> map { t -> c[t] + d[t] } -> g;\n"),
        &["g"],
    );
    let t = transfers(&emit(&src, on()).text);
    assert_eq!(t.iter().filter(|x| *x == "launch").count(), 1, "{t:?}");
    assert!(t.contains(&"launchv 5".to_owned()), "c, g: {t:?}");
    assert_eq!(t.iter().filter(|x| x.starts_with("d2h")).count(), 2);
    assert!(!t.iter().any(|x| x.starts_with("h2d")));
}

#[test]
fn mlp2_two_groups_store_only_what_the_next_site_reads() {
    let src = program(
        &["a", "b", "e", "f", "g", "p"],
        &(site("c1", "a", "b")
            + &format!(
                "    ta -> map {{ t -> {} }} -> h1;\n",
                relu(&format!("c1[t] + bias[t % {N}]"))
            )
            + &site("d1", "h1", "e")
            + &site("c2", "f", "g")
            + &format!(
                "    ta -> map {{ t -> {} }} -> h2;\n",
                relu(&format!("c2[t] + bias[t % {N}]"))
            )
            + &site("d2", "h2", "p")),
        &["d1", "d2"],
    );
    let e = emit(&src, on());
    let t = transfers(&e.text);
    assert_eq!(t.iter().filter(|x| *x == "launch").count(), 2, "{t:?}");
    assert_eq!(t.iter().filter(|x| *x == "launchv 5").count(), 2, "{t:?}");
    assert_eq!(
        t.iter()
            .filter(|x| x.starts_with("d2h"))
            .collect::<Vec<_>>(),
        ["d2h 4"; 4]
    );
    assert!(!t.iter().any(|x| x.starts_with("h2d")));
    for p in launchv_ptrs(&e.text) {
        assert_eq!(p[2], "0", "c1 / c2 are never stored");
    }
}

#[test]
fn fanout_stores_the_shared_output_once() {
    let src = program(
        &["a", "b", "e"],
        &(site("c", "a", "b")
            + "    c -> map { v -> v * 0.5 + 1.0 } -> h;\n"
            + &site("d", "c", "e")),
        &["c", "h", "d"],
    );
    let e = emit(&src, on());
    let t = transfers(&e.text);
    assert_eq!(t.iter().filter(|x| x.starts_with("launch")).count(), 2);
    assert_eq!(
        t.iter().filter(|x| x.starts_with("d2h")).count(),
        6,
        "{t:?}"
    );
    // `c` has two outside readers (the second site, the host): stored.
    assert_ne!(launchv_ptrs(&e.text)[0][2], "0");
}

#[test]
fn loop_carries_the_activation_on_the_device() {
    let n2 = N * N;
    let src = format!(
        "fn main() {{
    {n2} -> iota -> ta;
    {N} -> iota -> krange;
{}{}    mut x: [f32; {n2}] <- a;
    mut it: i32 <- 0;
    loop {{
        (it < 3) -> {{
            -true-> {{
{}                y -> map {{ v -> {} }} -> x;
                it + 1 -> it;
                -> loop;
            }}
            -false-> x -> out;
        }}
    }}
    out[0] -> println;
}}
",
        operand("a", 7),
        operand("w", 9),
        site("y", "x", "w").replace("    ", "                "),
        relu("v")
    );
    let e = emit(&src, on());
    let main = mapal_main(&e.text);
    let lv = main
        .find("@mapal_gpu_launchv(")
        .expect("the relu runs in y's store");
    let back = main[lv..].find("br label").expect("the back edge") + lv;
    let body = &main[lv..back];
    assert!(
        !body.contains("@mapal_gpu_d2h(") && !body.contains("@mapal_gpu_h2d("),
        "nothing crosses inside the loop:\n{body}"
    );
    assert_eq!(launchv_ptrs(&e.text)[0][2], "0", "y is never stored");
    // The back edge swaps the carry's two device buffers.
    assert_eq!(body.matches("store i64 %t").count(), 2, "{body}");
}

#[test]
fn off_arm_is_the_host_map_behind_a_copy_out() {
    let e = emit(&epi(), off());
    assert_eq!(transfers(&e.text), ["gen", "gen", "launch", "d2h 16384"]);
    assert!(!e.text.contains("launchv"));
    let dev = e.device.expect("a device unit");
    assert!(!dev.contains("_cell") && !dev.contains("%y0"));
}

#[test]
fn programs_without_a_per_cell_consumer_are_unchanged() {
    let chain = program(
        &["a", "b", "e"],
        &(site("c", "a", "b") + &site("d", "c", "e")),
        &["d"],
    );
    for p in [chain] {
        let (a, b) = (emit(&p, on()), emit(&p, off()));
        assert_eq!(a.text, b.text);
        assert_eq!(a.device, b.device);
    }
}

#[test]
fn a_trap_capable_or_float_mod_consumer_stays_on_the_host() {
    // `bias[t % 65]` over 64 cells: not proven, the cell can trap.
    let trap = program(
        &["a", "b"],
        &(site("c", "a", "b") + "    ta -> map { t -> c[t] + bias[t % 65] } -> d;\n"),
        &["d"],
    );
    // A float `%`: the host's `fmodf` is not NVPTX's `frem`.
    let fmod = program(
        &["a", "b"],
        &(site("c", "a", "b") + "    c -> map { v -> v % 0.7 } -> d;\n"),
        &["d"],
    );
    for p in [trap, fmod] {
        let e = emit(&p, on());
        assert!(
            !e.text.contains("launchv"),
            "{}",
            transfers(&e.text).join(" ")
        );
    }
}

#[test]
fn a_read_off_the_cell_is_a_kernel_boundary() {
    // `c[t / N]` reads another block's cell: it needs every block's store.
    let src = program(
        &["a", "b"],
        &(site("c", "a", "b") + &format!("    ta -> map {{ t -> c[t] + c[t / {N}] }} -> d;\n")),
        &["d"],
    );
    assert!(!emit(&src, on()).text.contains("launchv"));
}

#[test]
fn the_exact_face_device_unit_has_no_contraction() {
    for p in [epi(), bias()] {
        let dev = emit(&p, on()).device.expect("a device unit");
        assert!(!dev.contains(" contract "), "no contract flag");
        assert!(!dev.contains("call float @llvm.fma"), "no fused step");
        // The contract face fuses the site's step only, never a body's op.
        let dev = emit_on(&p, "rtx-4090", true, on())
            .device
            .expect("a device unit");
        assert!(dev.contains("call float @llvm.fma"));
        assert!(!dev.contains(" contract "));
    }
}

#[test]
fn every_kernel_takes_the_epilogue() {
    // TF32: `mma.sync` (4090) and Hopper's `wgmma` (H100 PCIe, sm_90a).
    for target in ["rtx-4090", "h100-pcie"] {
        let e = emit_detailed(
            &lower_src(&bias()),
            &EmitOpts {
                host: "generic",
                target,
                contract: true,
                reduced: Some(Reduced::Tf32),
                gpu: on(),
                ..EmitOpts::exact()
            },
        )
        .expect("emits");
        let dev = e.device.expect("a device unit");
        assert!(e.text.contains("@mapal_gpu_launchv("), "{target}");
        assert!(dev.contains("%y0") && dev.contains("_cell("), "{target}");
    }
    // An indivisible site: the one-thread-per-cell kernel.
    let odd = bias()
        .replace(&N.to_string(), "60")
        .replace(&(N * N).to_string(), "3600");
    let e = emit(&odd, on());
    assert!(e.text.contains("@mapal_gpu_launchv("));
    assert!(e.device.expect("a device unit").contains("%y0"));
}
