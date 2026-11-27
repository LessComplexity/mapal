//! Golden IR for the packed pool wrapper, inlined tiny tiles and GPU time brackets.

use super::*;

/// plan-s43-parallel-bpack: the B pack is dispatched across the pool, exactly
/// once, and the matmul still waits for it.
///
/// THIS IS THE ONLY TEST THAT COVERS THE PACKED PARALLEL WRAPPER. The insta
/// goldens do not: `parallel_matmul_cap`'s program has no packing site, and
/// `tile_nest_shape*` snapshot only `@task{n}_slice` — the function this change
/// deliberately leaves byte-identical. So a regression here moves no snapshot.
///
/// The assertion that matters most is `slice_elems != 0`. It is the silent
/// failure mode: `mapal-rt::slice_ranges` falls back to `min(T, ceil(n/GRAIN))`
/// with `GRAIN = 4096`, and at a few hundred j-tiles that is ONE slice — the
/// pack would go back to running serially with every other gate still green.
#[test]
fn packed_wrapper_dispatches_the_b_pack_across_the_pool() {
    let ir = mapal_rewrite::rewrite(lower_src(TILE_MATMUL_SRC)).ir;
    // `generic` BY NAME (S46). This is a structural pin, and the default target
    // is `native`; on a part that carries measured dispatch economics the
    // plan-s50 P3 threshold correctly INLINES this 4x4 matmul, which is the
    // subject of `tiny_tile_site_is_inlined_when_the_part_has_measured_costs`
    // below. `generic` has no measured economics, so it dispatches everything
    // and this test sees the shape it is about.
    let ll = emit_with_opts(
        &ir,
        &EmitOpts {
            host: "generic",
            target: "generic",
            ..EmitOpts::exact()
        },
    )
    .unwrap();

    let pack = regex_lite_find(&ll, "_pack, i64 ")
        .expect("the packed wrapper must register a parallel pack task (@task{n}_pack)");
    // `call void @mapal_par_task(ptr %h, i32 0, i32 KIND, ptr @task{n}_pack,
    //                            i64 TILES, i32 rank, i64 SLICE, i32 OVERSUB, i32 0)`
    let args: Vec<&str> = pack
        .rsplit_once('(')
        .expect("a call")
        .1
        .trim_end_matches(further_paren)
        .split(", ")
        .collect();
    assert_eq!(
        args[2], "i32 1",
        "the pack must be a SPLIT dispatch:\n{pack}"
    );
    let tiles: u64 = args[4]
        .trim_start_matches("i64 ")
        .parse()
        .expect("tile count");
    let slice_elems: u64 = args[6]
        .trim_start_matches("i64 ")
        .parse()
        .expect("slice_elems");
    assert!(tiles > 0, "the pack's n is the j-tile count:\n{pack}");
    assert!(
        slice_elems > 0 && slice_elems <= tiles,
        "slice_elems=0 silently collapses the pack to ONE slice (GRAIN=4096):\n{pack}"
    );

    // plan-s50 P0: the pack and the matmul are two tasks in the OUTER run,
    // joined by a dep edge — NOT a wrapper opening two nested runs. The nested
    // form cost 36.8 us per inner run at 14 lanes, per site, per invocation,
    // inside the timed region (`mapal-rt/examples/dispatch_cost.rs`).
    assert_eq!(
        ll.matches("@mapal_par_begin").count(),
        2,
        "one declaration and ONE run — a nested `mapal_par_begin` is the \
         per-invocation cost this shape exists to remove:\n{ll}"
    );
    let pack_task: u32 = args[1]
        .trim_start_matches("i32 ")
        .parse()
        .expect("the pack's task index");
    // `@task{n}_pack` names the matmul task it feeds.
    let matmul_task: u32 = pack
        .rsplit_once("@task")
        .expect("the pack body is named after its site")
        .1
        .split_once("_pack")
        .expect("_pack suffix")
        .0
        .parse()
        .expect("the matmul's task index");
    assert_ne!(pack_task, matmul_task, "the pack is its own task:\n{pack}");
    assert!(
        ll.contains(&format!(
            "call void @mapal_par_dep(ptr %h, i32 {pack_task}, i32 {matmul_task})"
        )),
        "the dep edge IS the happens-before from every pack store to every \
         matmul load — it replaced the nested run's `finish`:\n{ll}"
    );
    // The matmul is now dispatched by the outer run, so it must be a SPLIT with
    // the region's own sizing; registering it `kind = 0` would run the whole
    // matmul on one lane with every gate still green.
    let matmul = regex_lite_find(&ll, "_slice, i64 ").expect("the matmul registration");
    let margs: Vec<&str> = matmul
        .rsplit_once('(')
        .expect("a call")
        .1
        .trim_end_matches(further_paren)
        .split(", ")
        .collect();
    assert_eq!(
        margs[2], "i32 1",
        "the matmul must stay a SPLIT dispatch:\n{matmul}"
    );
    assert!(
        margs[6].trim_start_matches("i64 ").parse::<u64>().unwrap() > 0,
        "the matmul keeps its region quantum:\n{matmul}"
    );
    // The pack body is a real loop over [%lo, %hi), not the old [0, tiles).
    assert!(
        ll.contains("store i64 %lo, ptr") && ll.contains(", %hi\n"),
        "the pack function must walk the j-tile range it was handed:\n{ll}"
    );
}

/// plan-s50 P3: a task that cannot pay for its own dispatch is registered
/// `kind = 0`, which `mapal-rt` runs on the calling thread instead of queueing.
///
/// `TILE_MATMUL_SRC` is 4x4 — sixteen output elements over a reduction of four.
/// Dispatching it costs a wave (30 us on the measured part) to buy one lane's
/// share of ~128 flops. Measured at scale, the same mistake costs 1.8x: `nn4` at
/// d=128 runs 0.0790 ms on one thread and 0.1448 ms on fourteen.
///
/// The negative control is the point of the second half: a profile with no
/// MEASURED dispatch economics must keep dispatching, so no part can be made
/// worse by a number nobody took.
///
/// **The locator changed in S51.** This used to find the tile task by the
/// `_slice` suffix, which only a PACKED site has — so it was keyed on an
/// artifact of packing rather than on the thing under test. plan-s51 H1 made
/// packing conditional on size, and a 4x4's working set is 192 bytes against a
/// 16 MB L2, so on `apple-m4-sme` there is no `_slice` function any more. The
/// region quantum is the property that actually says "this is the bulk tile
/// task", on either profile and whether or not it packs.
#[test]
fn tiny_tile_site_is_inlined_when_the_part_has_measured_costs() {
    let ir = mapal_rewrite::rewrite(lower_src(TILE_MATMUL_SRC)).ir;
    // The BULK TILE task, on either profile: of the registrations that carry a
    // region quantum (argument 7 — a scalar/Seq task carries `i64 0`), the one
    // with the largest. On `generic` that separates the matmul (64) from the
    // b-pack task the same site registers (1); on `apple-m4-sme` the site does
    // not pack at this size and the matmul is the only one.
    let tile_kind = |target: &'static str| -> String {
        let ll = emit_with_opts(
            &ir,
            &EmitOpts {
                host: "generic",
                target,
                ..EmitOpts::exact()
            },
        )
        .unwrap();
        let quantum = |l: &str| -> u64 {
            l.split(", ")
                .nth(6)
                .and_then(|a| a.trim_start_matches("i64 ").parse().ok())
                .unwrap_or(0)
        };
        let call = ll
            .lines()
            .filter(|l| l.contains("call void @mapal_par_task("))
            .max_by_key(|l| quantum(l))
            .expect("a registered task");
        assert!(
            quantum(call) > 0,
            "no task carries a region quantum, so none of them is the tile \
             site:\n{ll}"
        );
        call.split(", ")
            .nth(2)
            .expect("the kind argument")
            .to_string()
    };
    assert_eq!(
        tile_kind("apple-m4-sme"),
        "i32 0",
        "a 4x4 matmul must not be handed to the pool on a part whose dispatch \
         cost is known"
    );
    assert_eq!(
        tile_kind("generic"),
        "i32 1",
        "a profile with no measured dispatch economics must dispatch exactly as \
         it did before the threshold existed"
    );
}

/// plan-s53 `clock-fence` on a GPU profile: the GPU e2e program
/// (`benches/gpu/gen_tiles.py e2e`, n = 64) brackets exactly its site. S52's
/// host read `t1` BEFORE the tile call and `t0` before the generation, so the
/// program's `iter ms` timed the generation. Now the generation precedes the
/// first read, and each read waits for the device first — a launch that
/// returns early must not let the clock close over it.
#[test]
fn gpu_time_bracket_is_time_tile_sync_time() {
    let e2e = |n: u32| {
        format!(
            "fn main() {{
    {nn} -> iota -> ta;
    ta -> map {{ t -> (t * 7 + 13) % 101 - 50 -> widen<f32> }} -> a;
    ta -> map {{ t -> (t * 7 + 57) % 101 - 50 -> widen<f32> }} -> b;
    {n} -> iota -> krange;
    () -> time -> t0;
    ta -> map {{ t ->
        t / {n} -> i;
        t % {n} -> j;
        (0.0, krange) -> fold {{ acc, k -> acc + a[i * {n} + k] * b[k * {n} + j] }}
    }} -> c;
    () -> time -> t1;
    c[0] -> println;
    \"iter ms=\" -> print;
    t1 - t0 -> println;
}}
",
            nn = n * n
        )
    };
    let at_arms = |src: &str, target: &'static str, gpu: GpuArms| {
        let ir = mapal_rewrite::rewrite(lower_src(src)).ir;
        emit_with_opts(
            &ir,
            &EmitOpts {
                host: "generic",
                target,
                gpu,
                ..EmitOpts::exact()
            },
        )
        .unwrap()
    };
    // The rules' launch (`--cands=1`): a table's call brackets the same way.
    let rules = GpuArms {
        site_cands: 1,
        ..GpuArms::default()
    };
    let at = |src: &str, target: &'static str| at_arms(src, target, rules);
    // `resident` (shipped): the operands are built on the device and each read
    // waits for every launch before it, so the bracket is the site's launch.
    let ll = at(&e2e(64), "h100-pcie");
    let host = mapal_main(&ll);
    let calls: Vec<&str> = host
        .lines()
        .filter_map(|l| {
            [
                "@mapal_gpu_gen(",
                "@mapal_time_ms(",
                "@mapal_gpu_launch", // `_l2(` too: a window changes no order
                "@mapal_gpu_sync(",
            ]
            .into_iter()
            .find(|c| l.contains(c))
        })
        .collect();
    assert_eq!(
        calls,
        [
            "@mapal_gpu_gen(",
            "@mapal_gpu_gen(",
            "@mapal_gpu_sync(",
            "@mapal_time_ms(",
            "@mapal_gpu_launch",
            "@mapal_gpu_sync(",
            "@mapal_time_ms("
        ],
        "gen -> sync -> time -> launch -> sync -> time:\n{host}"
    );
    // The S52 round trip (`resident` off): the host generation precedes t0.
    let mut off = GpuArms::default();
    *off.off.rung("resident").unwrap() = true;
    let ll = at_arms(&e2e(64), "h100-pcie", off);
    let host = mapal_main(&ll);
    let calls: Vec<&str> = host
        .lines()
        .filter_map(|l| {
            [
                "@fn",
                "@mapal_time_ms(",
                "@mapal_gpu_tile(",
                "@mapal_gpu_sync(",
            ]
            .into_iter()
            .find(|c| l.contains(c))
        })
        .collect();
    let first_read = calls.iter().position(|&c| c == "@mapal_time_ms(").unwrap();
    assert!(
        calls[..first_read].contains(&"@fn"),
        "the generation runs before t0:\n{host}"
    );
    assert_eq!(
        calls[first_read - 1..],
        [
            "@mapal_gpu_sync(",
            "@mapal_time_ms(",
            "@mapal_gpu_tile(",
            "@mapal_gpu_sync(",
            "@mapal_time_ms("
        ],
        "time -> tile -> sync -> time:\n{host}"
    );
    assert!(ll.contains("declare void @mapal_gpu_sync()"), "{ll}");
    // No device site, nothing to wait for: a CPU profile, and a GPU profile on
    // a module whose `time` brackets host work only, emit no sync at all.
    let host_only = "fn main() {\n    () -> time -> t0;\n    \
                     () -> time -> t1;\n    t1 - t0 -> println;\n}\n";
    for ll in [at(&e2e(64), "generic"), at(host_only, "h100-pcie")] {
        assert!(!ll.contains("@mapal_gpu_sync"), "{ll}");
    }
}

pub(crate) fn further_paren(c: char) -> bool {
    c == ')'
}

/// The first line containing `needle`. Kept local so the test adds no dep.
pub(crate) fn regex_lite_find<'a>(hay: &'a str, needle: &str) -> Option<&'a str> {
    hay.lines().find(|l| l.contains(needle))
}

/// `benches/gpu/gen_tiles.py`'s `mm_<n>` timing site.
pub(crate) fn gpu_mm(n: u64) -> CategoryIr {
    let src = format!(
        "fn main() {{
    {nn} -> iota -> ta;
    ta -> map {{ t -> (t * 7 + 13) % 101 - 50 -> widen<f32> }} -> a;
    ta -> map {{ t -> (t * 7 + 57) % 101 - 50 -> widen<f32> }} -> b;
    {n} -> iota -> krange;
    ta -> map {{ t ->
        t / {n} -> i;
        t % {n} -> j;
        (0.0, krange) -> fold {{ acc, k -> acc + a[i * {n} + k] * b[k * {n} + j] }}
    }} -> c;
    c[0] -> println;
}}
",
        nn = n * n
    );
    mapal_rewrite::rewrite(lower_src(&src)).ir
}

/// S53 `arch-rungs`: the rtx-4090's table takes `cp.async` only where the site
/// under-fills the part (1024²: one warp per scheduler) — not at 4096², the
/// v6 row where the async arm cost 8% — and the `arch-rungs` OFF arm restores
/// S52's every-rung-on kernel.
#[test]
fn rtx_4090_streams_with_cp_async_only_under_filled() {
    let device = |n, arms: mapal_backend_llvm::GpuArms| {
        let opts = EmitOpts {
            host: "generic",
            target: "rtx-4090",
            contract: true,
            gpu: arms,
            ..EmitOpts::exact()
        };
        mapal_backend_llvm::emit_detailed(&gpu_mm(n), &opts)
            .expect("emit")
            .device
            .expect("a GPU site")
    };
    let streams = |d: String| d.contains("call void @llvm.nvvm.cp.async");
    // The rules' kernel (`--cands=1`): a table also holds the `+async` one.
    let rules = mapal_backend_llvm::GpuArms {
        site_cands: 1,
        ..Default::default()
    };
    assert!(!streams(device(4096, rules)));
    assert!(streams(device(1024, rules)));
    let mut off = rules;
    *off.off.rung("arch-rungs").expect("a rung") = true;
    assert!(streams(device(4096, off)));
}
