//! S54 `rowfold` (plan-s54 §4.4): a row-local in-order fold is a device site
//! when the array it reads is a device site's output, and the producing site
//! stores that output in the order the fold reads it when the fold is its only
//! reader (`rowfold-layout`). The recognizer, the placement and the layout
//! are pinned here; the values are the box differential's job (and the CPU
//! emulation of the device unit the builder ran before renting).

use mapal_backend_llvm::{EmitOpts, GpuArms, emit_detailed};
use mapal_ir::{CategoryIr, Operation, RowFoldOp, RowFoldSite, TileRead};

const N: u64 = 64;

fn lower_src(src: &str) -> CategoryIr {
    let po = mapal_syntax::parse(src);
    assert!(po.diagnostics.is_empty(), "parse: {:?}", po.diagnostics);
    let ir = mapal_lower::lower(src, &po.program).unwrap_or_else(|d| panic!("lower: {d:?}"));
    mapal_rewrite::rewrite(ir).ir
}

/// `c = a·b` (n × n) on the device, then `tail`.
fn program(tail: &str) -> String {
    format!(
        "fn main() {{
    {nn} -> iota -> ta;
    {N} -> iota -> krange;
    ta -> map {{ t -> (t * 7 + 13) % 101 - 50 -> widen_f32 -> v; v / 7.0 }} -> a;
    ta -> map {{ t -> (t * 11 + 57) % 103 - 51 -> widen_f32 -> v; v / 9.0 }} -> b;
    ta -> map {{ t ->
        t / {N} -> i;
        t % {N} -> j;
        (0.0, krange) -> fold {{ acc, k -> acc + a[i * {N} + k] * b[k * {N} + j] }}
    }} -> c;
{tail}}}
",
        nn = N * N
    )
}

/// A per-row fold of `c` with this fold body (`e` is the read).
fn fold(seed: &str, body: &str, out: &str) -> String {
    format!(
        "    krange -> map {{ i -> ({seed}, krange) -> fold {{ acc, j -> c[i * {N} + j] -> e; {body} }} }} -> {out};\n"
    )
}

fn rowsum(extra: &str) -> String {
    program(&format!(
        "{}    r[0] -> println;\n    r[{}] -> println;\n{extra}",
        fold("0.0", "acc + e", "r"),
        N - 1
    ))
}

fn sites(src: &str) -> Vec<RowFoldSite> {
    let ir = lower_src(src);
    ir.row_fold(ir.entry()).values().cloned().collect()
}

#[test]
fn row_fold_recognizes_a_row_sum() {
    let s = sites(&rowsum(""));
    assert_eq!(s.len(), 1, "the row sum, and not the tile site: {s:?}");
    let s = &s[0];
    assert_eq!((s.rows, s.k), (N, N));
    assert_eq!(
        s.a,
        TileRead {
            slot: s.a.slot,
            base: 0,
            ci: N,
            ck: 1,
            clane: 0,
            ksplit: None
        }
    );
    assert_eq!(
        s.op,
        RowFoldOp::Arith {
            op: Operation::Add,
            acc_first: true
        }
    );
    assert_eq!(s.elem, mapal_ir::Ty::f32());
}

#[test]
fn row_fold_recognizes_a_running_max_in_source_order() {
    let src = program(&format!(
        "{}    m[0] -> println;\n",
        fold(
            "-1000000000.0",
            "(e > acc) -> { -true-> e; -false-> acc; }",
            "m"
        )
    ));
    let s = sites(&src);
    assert_eq!(s.len(), 1, "{s:?}");
    assert_eq!(
        s[0].op,
        RowFoldOp::Select {
            cmp: Operation::Gt,
            acc_lhs: false,
            acc_if_true: false
        }
    );
}

#[test]
fn row_fold_refuses_a_product_and_a_second_read() {
    for body in [
        // a product of the read: a second step
        "acc + e * 2.0",
        // a second read of the same array
        "acc + e + c[j * 64 + i]",
        // the read twice
        "acc + e * e",
    ] {
        let src = program(&format!("{}    r[0] -> println;\n", fold("0.0", body, "r")));
        assert!(sites(&src).is_empty(), "{body}");
    }
    // The tile site (`acc + a·b`) is `tile_plan`'s, never a row fold.
    assert!(sites(&program("    c[0] -> println;\n")).is_empty());
}

fn emit(src: &str, mut gpu: GpuArms, contract: bool) -> (String, String) {
    // R0 for folds alone (S54 integration): R1 (`epilogue`) and R2 (`devmap`)
    // would place these programs' per-cell maps too; their own tests pin that.
    for r in ["epilogue", "devmap"] {
        *gpu.off.rung(r).expect("a rung") = true;
    }
    let opts = EmitOpts {
        target: "rtx-4090",
        gpu,
        contract,
        ..EmitOpts::default()
    };
    let e = emit_detailed(&lower_src(src), &opts).expect("emits");
    (e.text, e.device.unwrap_or_default())
}

fn off(rung: &str) -> GpuArms {
    let mut a = GpuArms::default();
    *a.off.rung(rung).expect("a rung") = true;
    a.off.fill_rows = true; // as in `forced`
    a
}

/// Both rungs forced on: the placement under test, whatever the part's
/// measured table ships (`rowfold-layout` ships OFF where it was measured).
/// S62 `fill-rows` held off: it gives this site a one-row register tile, and
/// a column-major store needs whole vectors of rows (`col_store`).
fn forced() -> GpuArms {
    let mut a = GpuArms::default();
    a.on.rowfold = true;
    a.on.rowfold_layout = true;
    a.off.fill_rows = true;
    a
}

/// `mapal_gpu_*` calls of the host text, in order, SSA names dropped.
fn calls(ll: &str) -> Vec<String> {
    ll.lines()
        .filter(|l| l.contains("call ") && l.contains("@mapal_gpu_"))
        .map(|l| {
            let at = l.find("@mapal_gpu_").unwrap() + "@mapal_gpu_".len();
            let op = &l[at..l[at..].find('(').unwrap() + at];
            match op {
                "d2h" => format!(
                    "d2h {}",
                    l.rsplit("i64 ").next().unwrap().trim_end_matches(')')
                ),
                _ => op.to_owned(),
            }
        })
        // An arena member (S54 R6) is an allocation too.
        .filter(|c| !["free", "alloc", "arena"].iter().any(|p| c.starts_with(p)) && c != "init")
        .collect()
}

/// The launch ids of the host text, in order.
fn launch_ids(ll: &str) -> Vec<String> {
    ll.lines()
        .filter_map(|l| {
            let at = l.find("@mapal_gpu_launch(i64 ")? + "@mapal_gpu_launch(i64 ".len();
            Some(l[at..at + l[at..].find(',')?].to_owned())
        })
        .collect()
}

#[test]
fn a_fold_over_a_device_output_is_a_device_site() {
    for contract in [false, true] {
        let (on, dev) = emit(&rowsum(""), forced(), contract);
        // Two closed-form operands built on the device, the site, the fold,
        // then the two printed cells — one element each; `c` never crosses.
        assert_eq!(
            calls(&on),
            ["gen", "gen", "launch", "launch", "d2h 4", "d2h 4"],
            "contract {contract}"
        );
        assert!(dev.contains("%accn = fadd float %acc, %v"));
        assert!(!dev.contains("contract"), "the step is never contracted");
        // OFF: the host folds `c` behind its whole copy-out.
        let (off_ll, off_dev) = emit(&rowsum(""), off("rowfold"), contract);
        assert_eq!(
            calls(&off_ll),
            ["gen", "gen", "launch", &format!("d2h {}", N * N * 4)]
        );
        assert!(!off_dev.contains("%accn"));
    }
}

#[test]
fn the_layout_is_the_only_readers_order() {
    // The fold is `c`'s only reader: the site's kernel stores column-major (a
    // different kernel, so a different id) and the fold reads `c[i + k·rows]`.
    let (on, dev) = emit(&rowsum(""), forced(), false);
    let (row, row_dev) = emit(&rowsum(""), off("rowfold-layout"), false);
    let (ids, row_ids) = (launch_ids(&on), launch_ids(&row));
    assert_ne!(ids[0], row_ids[0], "the site stores in another order");
    assert_ne!(ids[1], row_ids[1], "the fold reads in another order");
    assert!(
        dev.contains(&format!("%kk = mul i64 %k, {N}")) && dev.contains("%ri = mul i64 %i, 1\n")
    );
    assert!(
        row_dev.contains("%kk = mul i64 %k, 1\n")
            && row_dev.contains(&format!("%ri = mul i64 %i, {N}"))
    );
    // A second reader of `c` (one host element): the order is no longer the
    // fold's to choose — row-major, the arm's text exactly.
    let two = rowsum("    c[5] -> println;\n");
    let (on2, dev2) = emit(&two, forced(), false);
    let (off2, off_dev2) = emit(&two, off("rowfold-layout"), false);
    assert_eq!(on2, off2);
    assert_eq!(dev2, off_dev2);
    assert_eq!(
        launch_ids(&on2).len(),
        2,
        "the fold still runs on the device"
    );
}

#[test]
fn a_fold_over_a_host_array_stays_on_the_host() {
    // `mx` folds the device output `c` (device); `den` folds `ex`, a per-cell
    // map the host computes (host: moving `ex` in would be the copy the rule
    // removes); `c` has two readers, so no layout.
    let src = program(&format!(
        "{}    ta -> map {{ t -> c[t] - mx[t / {N}] }} -> ex;
    krange -> map {{ i -> (0.0, krange) -> fold {{ acc, j -> ex[i * {N} + j] -> e; acc + e }} }} -> den;
    den[0] -> println;\n",
        fold(
            "-1000000000.0",
            "(e > acc) -> { -true-> e; -false-> acc; }",
            "mx"
        )
    ));
    assert_eq!(sites(&src).len(), 2, "both are row folds");
    let (on, dev) = emit(&src, forced(), false);
    assert_eq!(launch_ids(&on).len(), 2, "the site and `mx`");
    // `mx` crosses back whole (the host's `ex` reads it): its copy keeps no
    // pointer to the host slot, so the host loop that reads it still
    // vectorizes; `c`'s copy is today's text.
    let d2h: Vec<&str> = on
        .lines()
        .filter(|l| l.contains("call void @mapal_gpu_d2h("))
        .collect();
    assert_eq!(d2h.len(), 2, "{d2h:?}");
    assert!(!d2h[0].contains("nocapture") && d2h[1].contains("(ptr nocapture "));
    assert!(dev.contains("%c = fcmp ogt float %v, %acc"));
    assert!(dev.contains("%accn = select i1 %c, float %v, float %acc"));
    let (off_ll, _) = emit(&src, off("rowfold-layout"), false);
    assert_eq!(on, off_ll, "two readers: no layout");
}
