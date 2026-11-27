//! Per-cell kernels: variants, `cell_kernel_text`, `CellGeo` and launched-kernel scans.

use super::*;

/// A cell kernel's launch: one thread per cell, [`NAIVE_THREADS`] per block
/// (a one-load, one-store-per-thread kernel is bandwidth-bound at any block
/// size a part accepts — the `devgen` reason), as many blocks as cover `n`.
pub(crate) fn cell_grid(n: u64) -> (u64, u64) {
    (n.div_ceil(NAIVE_THREADS), NAIVE_THREADS)
}

/// The `Fold` a tile site's map `m` evaluates per cell: the one fold of its
/// body (`CategoryIr::tile_site` admits exactly one).
pub(super) fn site_fold(ir: &CategoryIr, m: MorphismId) -> (FuncId, MorphismId) {
    let Operation::Map { body, .. } = ir.morphism(m).expect("site morphism").op else {
        unreachable!("a tile site is a map");
    };
    let fold = ir
        .func(body)
        .expect("map body")
        .morphisms
        .iter()
        .copied()
        .find(|&x| matches!(ir.morphism(x).map(|g| g.op), Some(Operation::Fold { .. })))
        .expect("a tile site's body folds");
    (body, fold)
}

/// `cell-law`'s device variant of body `name` (`FnEmit::cell_vals`).
/// bf16-class `device-trap`: a cell body's device variant whose unproven
/// index reads record the trap (`@dev_trap_record`) and read element 0.
pub(crate) fn trap_variant(name: &str) -> String {
    format!("{name}_trap")
}

/// bf16-class `device-trap`: the module's trap word (`i64::MAX`: none; else
/// `topo << 8 | kind`, the least recorded) and the record a trap-checking
/// body calls — the runtime reads the word at the host's next checkpoint
/// (`mapal_gpu_trap_arm`, `mapal_gpu.c`).
pub(super) const DEVICE_TRAP: &str = "\n@mapal_dev_trap = addrspace(1) global i64 9223372036854775807, align 8\n\
define internal void @dev_trap_record(i64 %topo, i32 %kind) {\n\
entry:\n\
  %k = zext i32 %kind to i64\n\
  %t = shl i64 %topo, 8\n\
  %v = or i64 %t, %k\n\
  %o = atomicrmw min ptr addrspace(1) @mapal_dev_trap, i64 %v monotonic, align 8\n\
  ret void\n\
}\n";

pub(crate) fn cell_variant(name: &str) -> String {
    format!("{name}_cell")
}

/// The device variant of body `name` that reads the captured arrays in
/// `slots` as one-cell buffers (`FnEmit::cell_slots`: an S54 epilogue member,
/// a unit's member): `<name>_u<slots>`; `name` itself for no slot. Its own
/// name per slot set, apart from [`cell_variant`]'s: the inliner's copies of
/// a map share one body, so one body can be a cell-law cell here and an
/// epilogue member there — two texts, two signatures (precision: the shared
/// body's two `_cell` texts were one symbol, and the cell kernel called the
/// member's one-argument text with its forwarded value: a poison pointer).
pub(crate) fn slots_variant(name: &str, slots: &[u32]) -> String {
    if slots.is_empty() {
        return name.to_string();
    }
    let s: Vec<String> = slots.iter().map(u32::to_string).collect();
    format!("{name}_u{}", s.join("_"))
}

/// `tile-post`'s epilogue variant of map body `name` (`FnEmit::post_fold`).
pub(crate) fn post_variant(name: &str) -> String {
    format!("{name}_post")
}

/// `cell-lanes`' device variant of body `name` (`FnEmit::lanes`).
pub(crate) fn lanes_variant(name: &str) -> String {
    format!("{name}_lanes")
}

/// `out[i] = body(caps…, x[i])`, one thread per cell (`devmap`): the host's
/// body text runs on the device, so the cell is the host's bit for bit. The
/// parameters are the output, then the mapped array when it is loaded (`elem`
/// is its element type; `None` forms the element from the index, an `iota`),
/// then each captured array the cell loads — generic pointers to the body, as
/// the host passes them by reference — or uniform it reads. A capture
/// [`CapArg::Law`] is not a parameter: `cell-law` evaluates `law(i)` and
/// `callee` (the body's variant) takes the value after its input. A
/// [`CapArg::Uniform`] arrives as the 8-byte argument `mapal_gpu_launchv`
/// passes, its low bits the host's value.
pub(crate) fn cell_kernel_text(
    id: u64,
    callee: &str,
    n: u64,
    out: &str,
    arg: &str,
    elem: Option<&str>,
    caps: &[CapArg],
    g: CellGeo,
) -> String {
    let mut params = vec!["ptr addrspace(1) %o".to_string()];
    if elem.is_some() {
        params.push("ptr addrspace(1) %x".into());
    }
    params.extend(caps.iter().enumerate().filter_map(|(k, c)| match c {
        CapArg::Ptr => Some(format!("ptr addrspace(1) %c{k}")),
        CapArg::Uniform(_) => Some(format!("i64 %c{k}")),
        CapArg::Dev(_) => Some(format!("ptr addrspace(1) %c{k}")),
        CapArg::Law(..) | CapArg::Trip(..) => None,
    }));
    if g.range.is_some() {
        params.push("i64 %rs".into());
        params.push("i64 %rn".into());
    }
    if g.zero > 0 {
        params.extend(super::super::zero::zero_params(g.zero));
    }
    let mut body = String::new();
    let e_ty = match elem {
        Some(t) => {
            body.push_str(&format!(
                "  %xp = getelementptr inbounds {t}, ptr addrspace(1) %x, i64 %i\n  %e = load {t}, ptr addrspace(1) %xp\n"
            ));
            t
        }
        None => {
            body.push_str("  %e = trunc i64 %i to i32\n");
            "i32"
        }
    };
    let mut call_arg = if caps.is_empty() {
        format!("{e_ty} %e")
    } else {
        let mut prev = "poison".to_string();
        for (k, c) in caps.iter().enumerate() {
            match c {
                CapArg::Law(..) => continue, // the variant never reads the pointer
                CapArg::Ptr => body.push_str(&format!(
                    "  %g{k} = addrspacecast ptr addrspace(1) %c{k} to ptr\n  %a{k} = insertvalue {arg} {prev}, ptr %g{k}, {k}\n"
                )),
                CapArg::Uniform(t) => body.push_str(&format!(
                    "{}  %a{k} = insertvalue {arg} {prev}, {t} %g{k}, {k}\n",
                    uniform_from_i64(&format!("%g{k}"), &format!("%c{k}"), t)
                )),
                // `scalar-fold`: the number lives in device memory.
                CapArg::Dev(t) => body.push_str(&format!(
                    "  %g{k} = load {t}, ptr addrspace(1) %c{k}\n  %a{k} = insertvalue {arg} {prev}, {t} %g{k}, {k}\n"
                )),
                // `trip-batch`: the counter is the cell's own trip, `i ÷ d − b`.
                CapArg::Trip(t, d, b) => body.push_str(&format!(
                    "  %q{k} = sdiv i64 %i, {d}\n  %w{k} = sub i64 %q{k}, {b}\n{}  %a{k} = insertvalue {arg} {prev}, {t} %g{k}, {k}\n",
                    uniform_from_i64(&format!("%g{k}"), &format!("%w{k}"), t)
                )),
            }
            prev = format!("%a{k}");
        }
        body.push_str(&format!(
            "  %a = insertvalue {arg} {prev}, {e_ty} %e, {}\n",
            caps.len()
        ));
        format!("{arg} %a")
    };
    for (k, c) in caps.iter().enumerate() {
        if let CapArg::Law(law, t) = c {
            body.push_str(&format!("  %v{k} = call {t} @{law}(i32 %e)\n"));
            call_arg.push_str(&format!(", {t} %v{k}"));
        }
    }
    // `range-cell`: thread `k` is cell `rs + step·k` of the interval's `rn`
    // cells. `cell-lanes`: a cell is `lanes` consecutive threads (the body's
    // lane variant splits its fold over them), lane 0 stores.
    let (k, bound) = match g.range {
        None => ("%i", n.to_string()),
        Some(_) => ("%k", "%rn".to_string()),
    };
    // `piece-delta`: cell `k` of the range is `(k ÷ w)·step + k % w` past `rs`.
    let offset = |step: u64| match g.piece {
        0 | 1 => format!("  %ks = mul i64 %k, {step}\n  %i = add i64 %rs, %ks\n"),
        w => format!(
            "  %kq = udiv i64 %k, {w}\n  %kr = urem i64 %k, {w}\n  %kqs = mul i64 %kq, {step}\n  \
             %ks = add i64 %kqs, %kr\n  %i = add i64 %rs, %ks\n"
        ),
    };
    let (gid, at) = match (g.lanes, g.range) {
        (1, None) => ("%i".to_string(), String::new()),
        (1, Some(step)) => ("%k".to_string(), offset(step)),
        (l, None) => ("%g".to_string(), format!("  %i = udiv i64 %g, {l}\n")),
        (l, Some(step)) => (
            "%g".to_string(),
            format!("  %k = udiv i64 %g, {l}\n{}", offset(step)),
        ),
    };
    // Lanes: the bound test is on the cell, after the division.
    // `retain-cell` (`clip`): the cell itself is tested against the extent
    // too, so its index is formed first.
    let (head, guard) = if g.clip && g.range.is_some() {
        (
            at.clone(),
            format!(
                "  %in0 = icmp ult i64 {k}, {bound}\n  %inc = icmp ult i64 %i, {n}\n  %in = and i1 %in0, %inc\n"
            ),
        )
    } else if g.lanes == 1 {
        (
            String::new(),
            format!("  %in = icmp ult i64 {k}, {bound}\n"),
        )
    } else {
        (at.clone(), format!("  %in = icmp ult i64 {k}, {bound}\n"))
    };
    let at = if g.lanes == 1 && !(g.clip && g.range.is_some()) {
        at
    } else {
        String::new()
    };
    let store = if g.lanes == 1 {
        String::new()
    } else {
        format!(
            "  %ln = urem i64 %g, {l}\n  %l0 = icmp eq i64 %ln, 0\n  br i1 %l0, label %st, label %done\nst:\n",
            l = g.lanes
        )
    };
    format!(
        "define ptx_kernel void @{name}({params}) \"nvvm.maxntid\"=\"{NAIVE_THREADS}\" {{\n\
         entry:\n\
         \x20 %t = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n\
         \x20 %b = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n\
         \x20 %t64 = zext i32 %t to i64\n\
         \x20 %b64 = zext i32 %b to i64\n\
         \x20 %base = mul i64 %b64, {NAIVE_THREADS}\n\
         \x20 {gid} = add i64 %base, %t64\n\
         {head}{guard}\
         \x20 br i1 %in, label %body, label %done\n\
         body:\n\
         {at}{body}\
         \x20 %v = call {out} @{callee}({call_arg})\n\
         {store}\
         \x20 %p = getelementptr inbounds {out}, ptr addrspace(1) %o, i64 %i\n\
         \x20 store {out} %v, ptr addrspace(1) %p\n\
         \x20 br label %done\n\
         done:\n\
         {tail}\
         \x20 ret void\n\
         }}\n",
        name = kernel_name(id),
        params = params.join(", "),
        tail = super::super::zero::zero_tail(g.zero, NAIVE_THREADS),
    )
}

/// One capture slot of a cell kernel ([`cell_kernel_text`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum CapArg {
    /// A captured array, by device pointer.
    Ptr,
    /// `cell-law`: `(law fn, element type)`, evaluated in the cell.
    Law(String, String),
    /// A number capture of LLVM type `ty`, by value.
    Uniform(String),
    /// `scalar-fold`: a number capture of type `ty` in a one-element device
    /// buffer (a whole fold's result), by pointer.
    Dev(String),
    /// `trip-batch`: the loop counter, of type `ty`, as the cell's own trip
    /// `i ÷ d − b` (a batched row write: row `t + b` is trip `t`'s).
    Trip(String, u64, i64),
}

/// A cell kernel's geometry ([`cell_kernel_text`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CellGeo {
    /// `range-cell`: the interval's step; the kernel takes its start and
    /// count by value (`%rs`, `%rn`).
    pub range: Option<u64>,
    /// `piece-delta`: each `step` position is a piece of this many cells
    /// (0 or 1: one cell). Thread `k` is cell `rs + (k ÷ w)·step + k % w`.
    pub piece: u64,
    /// `cell-lanes`: threads a cell (1: one thread a cell).
    pub lanes: u64,
    /// kernel-merging `zero-fold`: the later sites' outputs this kernel
    /// zeroes ([`super::super::zero::zero_tail`]).
    pub zero: usize,
    /// caching-memory `retain-cell`: a range cell past the extent stores
    /// nothing (a delta's window may reach past the array's end).
    pub clip: bool,
}

impl CellGeo {
    #[cfg(test)]
    pub(crate) const ONE: CellGeo = CellGeo {
        range: None,
        piece: 0,
        lanes: 1,
        zero: 0,
        clip: false,
    };

    /// `(blocks, threads)` for `cells` cells.
    pub(crate) fn grid(self, cells: u64) -> (u64, u64) {
        cell_grid(cells.saturating_mul(self.lanes.max(1)))
    }
}

/// `dst = src` narrowed from the 8-byte launch argument to `ty` (`i64`
/// passes as is; a float is its bit pattern).
pub(crate) fn uniform_from_i64(dst: &str, src: &str, ty: &str) -> String {
    match ty {
        "i64" => format!("  {dst} = bitcast i64 {src} to i64\n"),
        "double" => format!("  {dst} = bitcast i64 {src} to double\n"),
        "float" => {
            format!("  {dst}.b = trunc i64 {src} to i32\n  {dst} = bitcast i32 {dst}.b to float\n")
        }
        t => format!("  {dst} = trunc i64 {src} to {t}\n"),
    }
}

/// The host's 8-byte launch argument for a uniform `val` of LLVM type `ty`
/// (the inverse of [`uniform_from_i64`]): the instructions, then the operand.
pub(crate) fn uniform_to_i64(dst: &str, val: &str, ty: &str) -> String {
    match ty {
        "i64" => format!("{dst} = bitcast i64 {val} to i64"),
        "double" => format!("{dst} = bitcast double {val} to i64"),
        "float" => {
            format!("{dst}.b = bitcast float {val} to i32\n  {dst} = zext i32 {dst}.b to i64")
        }
        t => format!("{dst} = zext {t} {val} to i64"),
    }
}

/// A `devmap` cell for the device unit: the functions its kernel calls
/// (`(name, emitted text)`), then the kernel text.
pub(crate) type CellKernel = (Vec<(String, String)>, String);

/// The kernels the host's launch sites name, by symbol: every
/// `mapal_gpu_{tile,launch,launch_l2,launchv}(id, …)` loads `mapal_tile_<id>`,
/// every `mapal_gpu_gen(id, …)` loads `mapal_gen_<id>` (`mapal_gpu.c`). A
/// device kernel outside this set has no use and is not emitted.
pub(crate) fn launched_kernels(host: &str) -> std::collections::BTreeSet<String> {
    let mut live = std::collections::BTreeSet::new();
    for (i, pat) in host.match_indices("call void @mapal_gpu_") {
        let rest = &host[i + pat.len()..];
        let Some((entry, args)) = rest.split_once("(i64 ") else {
            continue;
        };
        let prefix = match entry {
            "tile" | "launch" | "launch_l2" | "launchv" | "launch_coop" => "mapal_tile",
            "gen" => "mapal_gen",
            _ => continue,
        };
        let lit: String = args
            .chars()
            .take_while(|c| *c == '-' || c.is_ascii_digit())
            .collect();
        let id = lit
            .parse::<u64>()
            .or_else(|_| lit.parse::<i64>().map(|v| v as u64));
        if let Ok(id) = id {
            live.insert(format!("{prefix}_{id:016x}"));
        }
    }
    // per-site-choice: every kernel a site's table names
    // (`%mapal_cand { i64 <id>, i64 <reduce id>, … }`).
    for (i, pat) in host.match_indices("%mapal_cand { i64 ") {
        let mut f = host[i + pat.len()..].split(", ");
        let main = f.next();
        let red = f.next().and_then(|r| r.strip_prefix("i64 "));
        for lit in [main, red].into_iter().flatten() {
            if let Ok(id) = lit.parse::<i64>()
                && id != 0
            {
                live.insert(kernel_name(id as u64));
            }
        }
    }
    live
}

/// The symbol a kernel's text defines (its first `void @…(`).
pub(super) fn kernel_symbol(text: &str) -> &str {
    text.split_once("void @")
        .and_then(|(_, r)| r.split_once('('))
        .map_or("?", |(n, _)| n)
}
