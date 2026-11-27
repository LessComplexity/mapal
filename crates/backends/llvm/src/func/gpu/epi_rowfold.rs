//! The epilogue kernel records, generated-kernel ids and the row-fold kernel text.

use super::*;

/// **S54 R1, the device half: the per-cell consumers a site's store
/// evaluates** (`resident.rs:Epi`), in the terms the kernel needs. Group value
/// 0 is the site's accumulator, `k + 1` call `k`'s result.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct EpiKernel {
    /// The site's own output is stored (a reader outside the group).
    pub store_out: bool,
    /// Device pointers to complete operands, after `a, b, out` (`%x<i>`).
    pub caps: usize,
    pub calls: Vec<EpiCall>,
    /// device-geometry `tile-post`: the site's own body after its fold,
    /// `ret @callee(arg, acc)` — its result is group value 0.
    pub post: Option<EpiCall>,
    /// `tile-post`: the LLVM types of the numbers the epilogue reads by
    /// value (`%u<j>`, 8-byte arguments after the stored pointers).
    pub uniforms: Vec<String>,
}

// The kernel id hashes this text: a group without a post keeps the id it
// had before `tile-post` existed.
impl std::fmt::Debug for EpiKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("EpiKernel");
        d.field("store_out", &self.store_out)
            .field("caps", &self.caps)
            .field("calls", &self.calls);
        if let Some(p) = &self.post {
            d.field("post", p);
        }
        if !self.uniforms.is_empty() {
            d.field("uniforms", &self.uniforms);
        }
        d.finish()
    }
}

/// One member: `ret @callee(arg)`, its input assembled field by field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct EpiCall {
    pub callee: String,
    /// The body's input type: a scalar element, or a `{ … }` struct.
    pub arg: String,
    pub ret: String,
    pub fields: Vec<EpiArg>,
    /// Stored (`%y<j>`, in call order among the stored).
    pub store: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EpiArg {
    /// Group value `k`, by value (a mapped element).
    Value(usize),
    /// Group value `k` through a one-cell buffer: a captured array the body
    /// reads at `t`, which its device variant reads at 0.
    Cell(usize),
    /// Operand pointer `%x<i>`.
    Cap(usize),
    /// The cell index `t`, `i32`.
    Index,
    /// `tile-post`: an array only the site's fold reads — the epilogue
    /// variant never touches it (`ptr null`).
    Null,
    /// `tile-post`: uniform `j` (`%uv<j>`), by value.
    Uniform(usize),
}

/// A grouped site for the device unit: its function and morphism (its plain
/// kernel is not emitted), geometry and epilogue.
pub(crate) struct EpiSite {
    pub f: mapal_ir::FuncId,
    pub m: mapal_ir::MorphismId,
    pub site: TileSite,
    pub k: EpiKernel,
    /// plan-cost `linear` (§4a concat): the site table's terms and each
    /// term's partitions (`z = term·S + part`); 0 on an epilogue group.
    pub terms: u64,
    pub s: u64,
}

pub(crate) fn kernel_name(id: u64) -> String {
    format!("mapal_tile_{id:016x}")
}

/// A `devgen` kernel's id: FNV-1a over its body's name, length and profile —
/// the host passes it, the device unit names the kernel `mapal_gen_<id>`.
pub(crate) fn gen_id(body: &str, n: u64, profile: &TargetProfile) -> u64 {
    format!("gen|{body}|{n}|{}", profile.name)
        .bytes()
        .fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
        })
}

/// A `devgen` producer for the device unit: `(id, body fn name, element
/// type, the body's own emitted text)`.
pub(crate) type GenKernel = (u64, String, String, String);

/// `out[i] = body(i)`, one thread per element: the host's own body text runs
/// on the device (a clean leaf fn is scalar IR with no host symbol), so the
/// element is the host's bit for bit.
pub(super) fn gen_kernel_text(id: u64, callee: &str, elem: &str) -> String {
    format!(
        "define ptx_kernel void @mapal_gen_{id:016x}(ptr addrspace(1) %o, i64 %n) \
         \"nvvm.maxntid\"=\"{GEN_THREADS}\" {{\n\
         entry:\n\
         \x20 %t = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n\
         \x20 %b = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n\
         \x20 %t64 = zext i32 %t to i64\n\
         \x20 %b64 = zext i32 %b to i64\n\
         \x20 %base = mul i64 %b64, {GEN_THREADS}\n\
         \x20 %i = add i64 %base, %t64\n\
         \x20 %in = icmp ult i64 %i, %n\n\
         \x20 br i1 %in, label %body, label %done\n\
         body:\n\
         \x20 %x = trunc i64 %i to i32\n\
         \x20 %v = call {elem} @{callee}(i32 %x)\n\
         \x20 %p = getelementptr inbounds {elem}, ptr addrspace(1) %o, i64 %i\n\
         \x20 store {elem} %v, ptr addrspace(1) %p\n\
         \x20 br label %done\n\
         done:\n\
         \x20 ret void\n\
         }}\n"
    )
}

/// A device row fold's kernel id (S54 `rowfold`): FNV-1a over the site (its
/// read already re-addressed for a column-major producer) and the profile.
/// Named `mapal_tile_<id>` like a site's, so the runtime's `launch` finds it.
pub(crate) fn rowfold_id(site: &RowFoldSite, profile: &TargetProfile, lanes: u64) -> u64 {
    let l = if lanes > 1 {
        format!("|lanes={lanes}")
    } else {
        String::new()
    };
    format!("rowfold|{site:?}|{}{l}", profile.name)
        .bytes()
        .fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
        })
}

/// A cell kernel's id (`devmap`): FNV-1a over its body's name, extent and
/// profile. The kernel is named like a site's (`mapal_tile_<id>`), which is
/// how `mapal_gpu_launchv` finds it; the `cell` key keeps the two apart.
pub(crate) fn cell_id(body: &str, n: u64, profile: &TargetProfile) -> u64 {
    format!("cell|{body}|{n}|{}", profile.name)
        .bytes()
        .fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
        })
}

/// A row fold's launch `(blocks, threads)`: one thread per row, and the rows
/// spread over every SM — a block of `rows / sms` threads rounded up to a
/// warp, at most [`NAIVE_THREADS`]. A fold has one thread per row and no more
/// (ADR-0028 D4), so few threads that the SMs they land on, not the block
/// size, bound how many loads are in flight.
pub(crate) fn rowfold_grid(site: &RowFoldSite, profile: &TargetProfile, lanes: u64) -> (u64, u64) {
    let g = profile.gpu().expect("a device fold has a GPU profile");
    if lanes > 1 {
        // `fold-lanes`: a warp a row, whole warps per block.
        return (
            site.rows.saturating_mul(lanes).div_ceil(NAIVE_THREADS),
            NAIVE_THREADS,
        );
    }
    let threads = site
        .rows
        .div_ceil(g.sms)
        .next_multiple_of(g.warp)
        .min(NAIVE_THREADS);
    (site.rows.div_ceil(threads), threads)
}

/// `o[i] = fold_k (seed, ⊕) a[base + ci·i + ck·k]`, one thread per row, `k`
/// in order: the host fold's chain, so its bits. The step is the host's own
/// instruction (`fadd`/`fsub`/`fmul`, or `fcmp` + `select` for a running
/// max/min) in the source's operand order, never `contract`ed. The kernel
/// takes the site ABI's three pointers; `%b` is unused.
/// recurrence-precision item 2 `fold-lanes`: threads a row of a device row
/// fold folds over — a warp where the order is free (the default face,
/// ADR-0040), the step regroups (`+`, `×`, running max/min) over `f32`, the
/// fold is at least a warp long, plain (no window, no tap split) and its
/// rows leave the part's resident threads idle. 1 otherwise — and 1 over a
/// column-major operand (`colmajor`: its producing tile stores in the fold's
/// order, `rowfold-layout`): one thread a row already reads consecutive
/// addresses across a warp, where lanes along `k` would read `rows` apart
/// (the rtx-4090 measurement that once kept `fold-lanes` off there).
pub(crate) fn rowfold_lanes(
    site: &RowFoldSite,
    colmajor: bool,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    order: Order,
) -> u64 {
    let regroups = matches!(
        site.op,
        RowFoldOp::Arith {
            op: Operation::Add | Operation::Mul,
            ..
        } | RowFoldOp::Select { .. }
    );
    match host_rungs(profile, contract, arms).fold_lanes
        && !colmajor
        && regroups
        && site.w == 1
        && site.a.ksplit.is_none()
    {
        true => warp_lanes(profile, order, &site.elem, site.k, site.rows),
        false => 1,
    }
}

/// The lane gate `fold-lanes` and `cell-lanes` share: a warp per unit (a row,
/// a cell) where the order is free (ADR-0040), the fold is f32 and at least a
/// warp long, and `units` warps fit the part's resident threads. 1 otherwise.
pub(crate) fn warp_lanes(
    profile: &TargetProfile,
    order: Order,
    elem: &Ty,
    k: u64,
    units: u64,
) -> u64 {
    let Some(g) = profile.gpu() else { return 1 };
    let fits = units.saturating_mul(g.warp) <= g.sms.saturating_mul(g.max_threads_per_sm);
    match order == Order::Free && g.warp == 32 && *elem == Ty::f32() && k >= g.warp && fits {
        true => g.warp,
        false => 1,
    }
}

/// A device row fold's kernel: its name, and its `; mapal.plan` line + text.
pub(super) fn rowfold_entry(
    s: &RowFoldSite,
    colmajor: bool,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
    order: Order,
) -> (String, String) {
    let lanes = rowfold_lanes(s, colmajor, profile, contract, arms, order);
    let name = kernel_name(rowfold_id(s, profile, lanes));
    let (gx, block) = rowfold_grid(s, profile, lanes);
    let l = if lanes > 1 {
        format!(" lanes={lanes}")
    } else {
        String::new()
    };
    let text = format!(
        "; mapal.plan kernel={name} kind=rowfold rows={} k={} gx={gx} block={block}{l}\n",
        s.rows, s.k
    ) + &rowfold_kernel_text(&name, s, profile, lanes);
    (name, text)
}

/// `fold-lanes` (`lanes > 1`): a warp a row — lane `l` folds `k = l, l +
/// lanes, …` from the step's identity (a running max/min's from the seed),
/// a butterfly of shuffles combines the partials, `⊕` the seed once, lane 0
/// stores.
pub(super) fn rowfold_kernel_text(
    name: &str,
    s: &RowFoldSite,
    profile: &TargetProfile,
    lanes: u64,
) -> String {
    let (_, threads) = rowfold_grid(s, profile, lanes);
    let t = lower_ty(&s.elem).expect("float elem lowers");
    let e = elem_bytes_of(&s.elem);
    let seed = const_literal(&s.seed);
    let step = match s.op {
        RowFoldOp::Arith { op, acc_first } => {
            let code = match op {
                Operation::Add => "fadd",
                Operation::Sub => "fsub",
                _ => "fmul",
            };
            let (x, y) = if acc_first {
                ("%acc", "%v")
            } else {
                ("%v", "%acc")
            };
            format!("  %accn = {code} {t} {x}, {y}\n")
        }
        RowFoldOp::Select {
            cmp,
            acc_lhs,
            acc_if_true,
        } => {
            let pred = match cmp {
                Operation::Lt => "olt",
                Operation::Gt => "ogt",
                Operation::Le => "ole",
                _ => "oge",
            };
            let (x, y) = if acc_lhs {
                ("%acc", "%v")
            } else {
                ("%v", "%acc")
            };
            let (p, q) = if acc_if_true {
                ("%acc", "%v")
            } else {
                ("%v", "%acc")
            };
            format!("  %c = fcmp {pred} {t} {x}, {y}\n  %accn = select i1 %c, {t} {p}, {t} {q}\n")
        }
    };
    let (rows, k, ci, base) = (s.rows, s.k, s.a.ci, s.a.base);
    // generic-fusion `window`: the cell's row and lane `(t÷w, t%w)`, and the
    // fold's taps `(k÷div, k%div)`; the plain row fold's text otherwise.
    let row = if s.w > 1 {
        format!(
            "  %iq = udiv i64 %i, {w}\n  %il = urem i64 %i, {w}\n  %rq = mul i64 %iq, {ci}\n  \
             %rl = mul i64 %il, {cl}\n  %ri = add i64 %rq, %rl\n",
            w = s.w,
            cl = s.a.clane
        )
    } else {
        format!("  %ri = mul i64 %i, {ci}\n")
    };
    let taps = match s.a.ksplit {
        Some(ks) => format!(
            "  %kq = udiv i64 %k, {d}\n  %kr = urem i64 %k, {d}\n  %kq1 = mul i64 %kq, {cq}\n  \
             %kr1 = mul i64 %kr, {cr}\n  %kk = add i64 %kq1, %kr1\n",
            d = ks.div,
            cq = ks.cq,
            cr = ks.cr
        ),
        None => format!("  %kk = mul i64 %k, {}\n", s.a.ck),
    };
    // `fold-lanes`: the row is `g ÷ lanes`, the lane `g % lanes`; each lane
    // walks `k ≡ lane` from the identity, then the warp combines.
    let (gid, k0, acc0, kstep, fin, store) = if lanes > 1 {
        let comb = |a: &str, b: &str, r: &str| match s.op {
            RowFoldOp::Arith { op, .. } => {
                format!(
                    "  {r} = {} {t} {a}, {b}\n",
                    if op == Operation::Mul { "fmul" } else { "fadd" }
                )
            }
            RowFoldOp::Select {
                cmp,
                acc_lhs,
                acc_if_true,
            } => {
                let pred = match cmp {
                    Operation::Lt => "olt",
                    Operation::Gt => "ogt",
                    Operation::Le => "ole",
                    _ => "oge",
                };
                let (x, y) = if acc_lhs { (a, b) } else { (b, a) };
                let (p, q) = if acc_if_true { (a, b) } else { (b, a) };
                format!(
                    "  {r}c = fcmp {pred} {t} {x}, {y}\n  {r} = select i1 {r}c, {t} {p}, {t} {q}\n"
                )
            }
        };
        let (identity, seeded) = match s.op {
            RowFoldOp::Arith {
                op: Operation::Mul, ..
            } => ("1.0".to_string(), true),
            RowFoldOp::Arith { .. } => ("-0.0".to_string(), true),
            RowFoldOp::Select { .. } => (seed.clone(), false),
        };
        let mut tree = String::new();
        let mut cur = "%accn".to_string();
        let mut off = lanes / 2;
        while off >= 1 {
            tree.push_str(&format!(
                "  %sh{off} = call float @llvm.nvvm.shfl.sync.bfly.f32(i32 -1, float {cur}, i32 {off}, i32 31)\n"
            ));
            let r = format!("%cb{off}");
            tree.push_str(&comb(&cur, &format!("%sh{off}"), &r));
            cur = r;
            off /= 2;
        }
        if seeded {
            let first = matches!(
                s.op,
                RowFoldOp::Arith {
                    acc_first: true,
                    ..
                }
            );
            let (a, b) = if first {
                (seed.as_str(), cur.as_str())
            } else {
                (cur.as_str(), seed.as_str())
            };
            tree.push_str(&comb(a, b, "%tot"));
            cur = "%tot".into();
        }
        (
            format!(
                "  %g = add i64 %row0, %t64\n  %i = udiv i64 %g, {lanes}\n  %ln = urem i64 %g, {lanes}\n"
            ),
            "%ln".to_string(),
            identity,
            lanes,
            "uge",
            format!(
                "{tree}  %l0 = icmp eq i64 %ln, 0\n  br i1 %l0, label %st, label %done\nst:\n  \
                 %q = getelementptr inbounds {t}, ptr addrspace(1) %o, i64 %i\n  \
                 store {t} {cur}, ptr addrspace(1) %q, align {e}\n"
            ),
        )
    } else {
        (
            "  %i = add i64 %row0, %t64\n".to_string(),
            "0".to_string(),
            seed.clone(),
            1,
            "eq",
            format!(
                "  %q = getelementptr inbounds {t}, ptr addrspace(1) %o, i64 %i\n  \
                 store {t} %accn, ptr addrspace(1) %q, align {e}\n"
            ),
        )
    };
    format!(
        "define ptx_kernel void @{name}(\
         ptr addrspace(1) noalias readonly align {e} %a, \
         ptr addrspace(1) noalias readonly align {e} %b, \
         ptr addrspace(1) noalias align {e} %o) \"nvvm.maxntid\"=\"{threads}\" {{\n\
         entry:\n\
         \x20 %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n\
         \x20 %bid = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n\
         \x20 %t64 = zext i32 %tid to i64\n\
         \x20 %b64 = zext i32 %bid to i64\n\
         \x20 %row0 = mul i64 %b64, {threads}\n\
         {gid}\
         \x20 %in = icmp ult i64 %i, {rows}\n\
         \x20 br i1 %in, label %body, label %done\n\
         body:\n\
         {row}\
         \x20 %r0 = add i64 %ri, {base}\n\
         \x20 br label %kloop\n\
         kloop:\n\
         \x20 %k = phi i64 [{k0}, %body], [%kn, %kloop]\n\
         \x20 %acc = phi {t} [{acc0}, %body], [%accn, %kloop]\n\
         {taps}\
         \x20 %x = add i64 %r0, %kk\n\
         \x20 %p = getelementptr inbounds {t}, ptr addrspace(1) %a, i64 %x\n\
         \x20 %v = load {t}, ptr addrspace(1) %p, align {e}\n\
         {step}\
         \x20 %kn = add i64 %k, {kstep}\n\
         \x20 %fin = icmp {fin} i64 %kn, {k}\n\
         \x20 br i1 %fin, label %store, label %kloop\n\
         store:\n\
         {store}\
         \x20 br label %done\n\
         done:\n\
         \x20 ret void\n\
         }}\n"
    )
}
