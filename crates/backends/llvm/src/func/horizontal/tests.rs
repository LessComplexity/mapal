//! Unit tests of the horizontal kernel groups: which grids merge, residency, merged text.

use super::device::*;
use super::emit::*;
use super::*;

fn gpu() -> &'static Gpu {
    crate::profile::TargetProfile::named("rtx-5080")
        .and_then(|p| p.gpu())
        .expect("rtx-5080 has a GPU")
}

fn shape(blocks: u64, block: u64, smem: u64, per_sm: u64) -> Shape {
    Shape {
        blocks,
        block,
        smem,
        per_sm,
        args: 3,
    }
}

#[test]
fn small_grids_merge_and_full_last_waves_do_not() {
    let g = gpu();
    // Three one-block launches: one wave together.
    let tiny = [
        shape(1, 256, 0, 6),
        shape(2, 32, 66048, 1),
        shape(1, 256, 0, 6),
    ];
    assert!(side_by_side(&tiny, g).is_ok());
    // Two cells of 4096 blocks: their last waves (64 each) fit one.
    let w = g.sms * 6;
    let big = [shape(4096, 256, 0, 6), shape(4096, 256, 0, 6)];
    assert_eq!(side_by_side(&big, g), Ok(w));
    // Two grids whose last waves are full: no wave saved.
    let full = [shape(w, 256, 0, 6), shape(w, 256, 0, 6)];
    assert_eq!(side_by_side(&full, g), Err(Refuse::Waves));
    // A member's residency falls at the merged footprint: refused.
    let occ = [shape(4 * w, 256, 0, 6), shape(1, 256, 90 * 1024, 1)];
    assert_eq!(side_by_side(&occ, g), Err(Refuse::Occupancy));
    // Arguments over one launch's.
    let many = [Shape { args: 9, ..tiny[0] }, Shape { args: 8, ..tiny[2] }];
    assert_eq!(side_by_side(&many, g), Err(Refuse::Args));
}

#[test]
fn merged_residency_keeps_each_members_registers() {
    let g = gpu();
    // A 128-thread tile at 4 blocks / SM (512 threads' registers) in a
    // 256-thread grid: 2 blocks / SM.
    let s = [shape(10, 128, 0, 4), shape(10, 256, 0, 6)];
    assert_eq!(merged_per_sm(&s, g), 2);
}

/// wgmma-16bit: a tensor map taken by value (`grid_constant`) becomes
/// the device function's pointer to the merged kernel's parameter — a
/// byval argument of an internal function would be a local copy.
#[test]
fn a_tensor_map_parameter_becomes_its_address() {
    let k = "define ptx_kernel void @mapal_tile_00000000000000cc(ptr addrspace(1) %a, ptr byval([128 x i8]) align 64 \"nvvm.grid_constant\" %tm0) \"nvvm.maxntid\"=\"256\" {\n\
             entry:\n  ret void\n}\n";
    let f = device_fn(k, "mapal_tile_00000000000000cc", None);
    assert!(f.starts_with("define internal void @hz.mapal_tile_00000000000000cc(ptr addrspace(1) %a, ptr readonly align 64 %tm0, i32 %hz.bx, i32 %hz.by) alwaysinline {"), "{f}");
}

#[test]
fn a_kernel_becomes_a_device_function_of_its_block() {
    let k = "define ptx_kernel void @mapal_tile_00000000000000aa(ptr addrspace(1) noalias readonly align 16 %a, i64 %c0) \"nvvm.maxntid\"=\"32\" \"nvvm.minctasm\"=\"2\" {\n\
             entry:\n  %v1 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n  %v2 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()\n  call void @llvm.nvvm.barrier0()\n  ret void\n}\n";
    let f = device_fn(k, "mapal_tile_00000000000000aa", Some(32));
    assert!(f.starts_with("define internal void @hz.mapal_tile_00000000000000aa(ptr addrspace(1) noalias readonly align 16 %a, i64 %c0, i32 %hz.bx, i32 %hz.by) alwaysinline {"));
    assert!(f.contains("%v1 = add i32 %hz.bx, 0") && f.contains("%v2 = add i32 %hz.by, 0"));
    assert!(f.contains("barrier.sync.cnt(i32 0, i32 32)") && !f.contains("barrier0"));
    let h = parse_head(k.lines().next().unwrap());
    assert_eq!(h.minctasm, Some(2));
    assert_eq!(h.params[0].0, "ptr addrspace(1)");
    assert_eq!(h.params[1].0, "i64");
    let spec = HzSpec {
        name: "mapal_tile_00000000000000bb".into(),
        block: 256,
        members: vec![
            ("mapal_tile_00000000000000aa".into(), 2, 1, 32),
            ("mapal_tile_00000000000000aa".into(), 1, 3, 32),
        ],
        level: vec![0, 0],
        zero: vec![None, None],
    };
    let m = merged_kernel(&spec, &[&h, &h]);
    assert!(m.contains("@mapal_tile_00000000000000bb(ptr addrspace(1) noalias readonly align 16 %h0.a, i64 %h0.c0, ptr addrspace(1) noalias readonly align 16 %h1.a, i64 %h1.c0)"));
    // minctasm 2 at 32 threads: 64 threads, under one 256-thread block.
    assert!(m.contains("\"nvvm.maxntid\"=\"256\" \"nvvm.minctasm\"=\"1\""));
    assert!(m.contains("icmp ult i32 %hz.b, 2") && m.contains("sub i32 %hz.b, 2"));
    assert!(m.contains("icmp ult i32 %hz.t, 32"));
}

#[test]
fn a_member_launched_nowhere_else_leaves_only_its_device_function() {
    let k = |n: &str| {
        format!(
            "; cell\n; mapal.plan kernel={n} kind=cell uniform=0\ndefine ptx_kernel void @{n}(ptr addrspace(1) %o) \"nvvm.maxntid\"=\"256\" {{\nentry:\n  ret void\n}}\n"
        )
    };
    let (a, b) = ("mapal_tile_000000000000000a", "mapal_tile_000000000000000b");
    let module = format!("; head\n{}\n{}", k(a), k(b));
    let spec = HzSpec {
        name: "mapal_tile_000000000000000c".into(),
        block: 256,
        members: vec![(a.into(), 4, 1, 256), (b.into(), 2, 1, 256)],
        level: vec![0, 0],
        zero: vec![None, None],
    };
    let live: std::collections::BTreeSet<String> = [b.to_string()].into();
    let out = compose(module, &[spec], &live);
    assert!(
        !out.contains(&format!("void @{a}(")) && !out.contains(&format!("kernel={a} kind=cell"))
    );
    assert!(out.contains(&format!("define ptx_kernel void @{b}(")));
    assert!(out.contains(&format!("define internal void @hz.{a}(")));
    assert!(out.contains(&format!("define internal void @hz.{b}(")));
    assert!(out.contains("define ptx_kernel void @mapal_tile_000000000000000c(ptr addrspace(1) %h0.o, ptr addrspace(1) %h1.o)"));
    assert_eq!(out.matches("; cell\n").count(), 1);
}

#[test]
fn launch_lines_parse() {
    let Some(Parsed::Site {
        global, nptr, out, ..
    }) = parse_launch(
        "call void @mapal_gpu_site(i64 -5, i32 1, i32 0, i32 4, ptr @mapal_main.cands152, i32 3, ptr %p153, i32 2, ptr null)",
    )
    else {
        panic!("site")
    };
    assert_eq!((global.as_str(), nptr, out), ("@mapal_main.cands152", 3, 2));
    assert!(matches!(
        parse_launch(
            "call void @mapal_gpu_launchv(i64 7, i32 0, i32 6, ptr %dv1, i32 4, i32 1, i32 256, i32 0) ; unit"
        ),
        Some(Parsed::Launchv { nptr: 6, .. })
    ));
    // A dynamic grid, a split's tag, a stream: not taken.
    assert!(parse_launch("call void @mapal_gpu_launchv(i64 7, i32 0, i32 6, ptr %dv1, i32 %t9, i32 1, i32 256, i32 0)").is_none());
    assert!(parse_launch("call void @mapal_gpu_launch(i64 7, i64 %a, i64 %b, i64 %o, i32 4, i32 2, i32 256, i32 0) ; splitk=2 atomic").is_none());
    assert!(parse_launch("call void @mapal_gpu_launchv(i64 7, i32 1, i32 3, ptr %p, i32 4, i32 1, i32 256, i32 0)").is_none());
    let rows = table_rows(
        "@g = private unnamed_addr constant [2 x %mapal_cand] [%mapal_cand { i64 -3, i64 0, i64 0, i64 0, i32 1, i32 1, i32 256, i32 0, i32 0, i32 0 }, %mapal_cand { i64 4, i64 0, i64 1024, i64 0, i32 2, i32 2, i32 32, i32 66048, i32 0, i32 0 }]\n",
        "@g",
    )
    .unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!((rows[1].zero, rows[1].gx, rows[1].smem), (1024, 2, 66048));
}
