//! the GPU realization of a tile site (plan-s41 steps 2–5, S52): the same
//! `Trn` at a second `Loc`.
//!
//! One cohesive unit, selected in one place (`emit_tiled_map`), never an `if`
//! inside a shared emitter (plan-s41 §2.2 rule 2). It has two halves:
//!
//! - **host** — [`FnEmit::emit_tiled_map_gpu`]: at the site, the host moves the
//!   two operands to the device, launches, and moves the output back, through
//!   one runtime call (`mapal_gpu_tile`, `crates/mapal-rt/gpu/mapal_gpu.c`).
//!   Those are the plan's `h2d` / `launch` / `d2h` transmissions.
//! - **device** — [`device_module`]: one `nvptx64` kernel per site, in a second
//!   translation unit that `llc` turns into PTX.
//!
//! **Values are the oracle's.** Every output cell accumulates its own products
//! in increasing `k`, one thread, one register, exactly the interpreter's left
//! fold — the tiles change *where* operands come from (shared memory, then
//! registers), never the order a cell combines them. The conformance face emits
//! a separate `fmul` and `fadd` (PTX `mul.rn` / `add.rn`, which `ptxas` may not
//! fuse); `--contract` emits `llvm.fma` per step, the CPU contract face's fused
//! form.
//!
//! The geometry is [`TargetProfile::gpu_tile`]'s: derived from device facts and
//! the site's extents, never a literal here.

use super::*;
use crate::Reduced;
use crate::profile::{
    GPU_VEC_BYTES, Gpu, GpuMmaTile, GpuTile, GpuWgmmaTile, MMA_K, MMA_M, MMA_N, WARPGROUP_WARPS,
    WGMMA_ATOM_ROWS, WGMMA_K, WGMMA_M, WGMMA_SWIZZLE_BYTES,
};

/// Bytes one shared-memory wavefront serves: 32 banks × 4 B. An ISA rule on
/// every NVIDIA part since Kepler.
const GPU_WAVEFRONT_BYTES: u64 = 128;

/// The device data layout and triple `clang --target=nvptx64-nvidia-cuda`
/// emits (LLVM 22).
const DEVICE_PREAMBLE: &str = "\
target datalayout = \"e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64\"\n\
target triple = \"nvptx64-nvidia-cuda\"\n\
\n\
@mapal_smem = external addrspace(3) global [0 x i8], align 16\n\
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n\
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n\
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()\n\
declare void @llvm.nvvm.barrier0()\n\
declare void @llvm.nvvm.cp.async.cg.shared.global.16(ptr addrspace(3), ptr addrspace(1))\n\
declare void @llvm.nvvm.cp.async.ca.shared.global.4(ptr addrspace(3), ptr addrspace(1))\n\
declare void @llvm.nvvm.cp.async.ca.shared.global.8(ptr addrspace(3), ptr addrspace(1))\n\
declare void @llvm.nvvm.cp.async.commit.group()\n\
declare void @llvm.nvvm.cp.async.wait.group(i32 immarg)\n\
declare float @llvm.fma.f32(float, float, float)\n\
declare double @llvm.fma.f64(double, double, double)\n";

/// The host-side runtime entry, declared only when some site called it.
pub(crate) const GPU_DECLS: &str =
    "declare void @mapal_gpu_tile(i64, ptr, i64, ptr, i64, ptr, i64, i32, i32, i32, i32)\n";

/// Threads per block of the one-thread-per-cell kernel.
/// ponytail: a literal, not derived — the fallback is a correctness path; derive
/// it when an untiled site turns out to matter for speed.
const NAIVE_THREADS: u64 = 256;

/// Can this site run on the GPU at all? Any float site whose reads are plain
/// affine in `(i, lane, k)`. The `k÷div` decomposition (conv taps) has no
/// device encoding yet and stays on the host.
pub(crate) fn gpu_site(site: &TileSite, profile: &TargetProfile) -> bool {
    profile.gpu().is_some()
        && is_float(&site.elem)
        && site.a.ksplit.is_none()
        && site.b.ksplit.is_none()
        && site.k > 0
        && site.rows * site.c > 0
}

/// Which kernel a GPU site gets, and how it is launched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Launch {
    pub tiled: Option<GpuTile>,
    /// The tensor-core kernel, on a reduced face where the part has one.
    pub mma: Option<GpuMmaTile>,
    /// Hopper's asynchronous warpgroup tensor-core kernel (TF32 face, sm_90a).
    pub wgmma: Option<GpuWgmmaTile>,
    pub gx: u64,
    pub gy: u64,
    pub block: u64,
    pub smem: u64,
}

/// The staged kernel needs the separable shape the CPU's register-blocked rung
/// cashes (`b.ci == 0`): `a` ignores the lane and `b` ignores the row, so a
/// block's row panel of `a` and lane panel of `b` are each read once per `k`
/// step and shared by every cell of the tile. Strides are free — each operand
/// is staged with vector loads when its fast axis is unit-stride and aligned
/// ([`vector_a`], [`vector_b`]) and element by element otherwise. Anything else
/// takes the one-thread-per-cell kernel.
pub(crate) fn launch(
    site: &TileSite,
    profile: &TargetProfile,
    rungs: crate::GpuRungs,
    reduced: Option<Reduced>,
) -> Launch {
    let e = elem_bytes_of(&site.elem);
    let shaped = site.a.clane == 0 && site.b.ci == 0;
    // Tensor cores: only on a face that admits them (TF32), f32 operands both
    // streamed with 16-byte copies, on a part with `mma.sync`.
    let tensor = rungs.mma
        && reduced == Some(Reduced::Tf32)
        && site.elem == Ty::f32()
        && shaped
        && vector_a(site)
        && vector_b(site);
    let wgmma = (tensor && rungs.wgmma)
        .then(|| {
            let (r, c, k) = (site.rows, site.c, site.k);
            let overlap = rungs.wgmma_overlap;
            profile
                .gpu_wgmma_tile(r, c, k, rungs.wgmma_reg_stage, overlap)
                .or_else(|| profile.gpu_wgmma_tile(r, c, k, false, overlap))
        })
        .flatten();
    if let Some(w) = wgmma {
        let warp = profile.gpu().expect("gpu").warp;
        return Launch {
            tiled: None,
            mma: None,
            wgmma: Some(w),
            gx: site.c / w.bn(),
            gy: site.rows / w.bm(),
            block: w.threads(warp),
            smem: w.smem_bytes(),
        };
    }
    let mma = tensor
        .then(|| profile.gpu_mma_tile(site.rows, site.c, site.k, rungs.wide))
        .flatten();
    if let Some(m) = mma {
        let warp = profile.gpu().expect("gpu").warp;
        return Launch {
            tiled: None,
            mma: Some(m),
            wgmma: None,
            gx: site.c / m.bn(),
            gy: site.rows / m.bm(),
            block: m.threads(warp),
            smem: m.smem_bytes(),
        };
    }
    let tiled = shaped
        .then(|| profile.gpu_tile(site.rows, site.c, site.k, &site.elem))
        .flatten()
        .map(|g| {
            let streamed = rungs
                .cp_async
                .then(|| profile.gpu_tile_async(g, site.k, &site.elem))
                .flatten();
            match streamed {
                Some(a) => a,
                None if rungs.prefetch => prefetch_depth(g, e),
                None => g,
            }
        });
    match tiled {
        Some(g) => Launch {
            tiled: Some(g),
            mma: None,
            wgmma: None,
            gx: site.c / g.bn(),
            gy: site.rows / g.bm(),
            block: g.threads(),
            smem: g.smem_bytes(e),
        },
        None => Launch {
            tiled: None,
            mma: None,
            wgmma: None,
            gx: (site.rows * site.c).div_ceil(NAIVE_THREADS),
            gy: 1,
            block: NAIVE_THREADS,
            smem: 0,
        },
    }
}

/// The prefetch rung holds the next panel in registers while the current one
/// is computed, so its depth is bounded by registers, not shared memory: one
/// panel's elements per thread, `(bm + bn)·bk / threads`, may take at most one
/// fragment pair's worth (`tm + tn`) — the budget the register tile already
/// spends on operands. Halve `bk` until it fits; a halved power of two still
/// divides `k` and still loads in whole vectors per thread.
fn prefetch_depth(g: GpuTile, elem_bytes: u64) -> GpuTile {
    let lanes = GPU_VEC_BYTES / elem_bytes;
    let regs = |bk: u64| (g.bm() + g.bn()) * bk * elem_bytes.div_ceil(4) / g.threads();
    let whole = |bk: u64| {
        (g.bm() * bk / lanes).is_multiple_of(g.threads())
            && (bk * g.bn() / lanes).is_multiple_of(g.threads())
    };
    let mut bk = g.bk;
    while regs(bk) > g.tm + g.tn && bk / 2 >= lanes && whole(bk / 2) {
        bk /= 2;
    }
    GpuTile { bk, ..g }
}

fn aligned(site: &TileSite, x: u64) -> bool {
    x.is_multiple_of(GPU_VEC_BYTES / elem_bytes_of(&site.elem))
}

/// `a`'s panel rows are contiguous along `k` and vector-aligned.
fn vector_a(site: &TileSite) -> bool {
    site.a.ck == 1 && aligned(site, site.a.ci) && aligned(site, site.a.base)
}

/// `b`'s panel rows are contiguous along the lane and vector-aligned.
fn vector_b(site: &TileSite) -> bool {
    site.b.clane == 1 && aligned(site, site.b.ck) && aligned(site, site.b.base)
}

fn elem_bytes_of(elem: &Ty) -> u64 {
    match elem {
        Ty::Float { bits } => u64::from(*bits) / 8,
        _ => 4,
    }
}

/// A stable id for a site's kernel: FNV-1a over everything the kernel text
/// depends on. The host passes it to the runtime, which looks the kernel up as
/// `mapal_tile_<id>` — so the two translation units agree without sharing state.
pub(crate) fn kernel_id(
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    reduced: Option<Reduced>,
) -> u64 {
    let key = format!("{site:?}|{}|{contract}|{reduced:?}", profile.name);
    key.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

pub(crate) fn kernel_name(id: u64) -> String {
    format!("mapal_tile_{id:016x}")
}

/// The device translation unit: one kernel per GPU tile site in `ir`, in name
/// order. `None` when no site offloads (every CPU profile, and a GPU profile
/// on a program with no tile site).
pub(crate) fn device_module(
    ir: &CategoryIr,
    profile: &TargetProfile,
    contract: bool,
    rungs: crate::GpuRungs,
    reduced: Option<Reduced>,
) -> Option<String> {
    let mut kernels = std::collections::BTreeMap::new();
    for (f, _) in ir.funcs() {
        for (_, site) in ir.tile_plan(f).sites.iter() {
            if gpu_site(site, profile) {
                let name = kernel_name(kernel_id(site, profile, contract, reduced));
                kernels
                    .entry(name.clone())
                    .or_insert_with(|| kernel_text(&name, site, profile, contract, rungs, reduced));
            }
        }
    }
    if kernels.is_empty() {
        return None;
    }
    let mut out = String::from("; mapal-backend-llvm device module\n");
    out.push_str(DEVICE_PREAMBLE);
    let mut md = String::new();
    for (i, k) in kernels.values().enumerate() {
        out.push('\n');
        if k.contains("!@LOOP@") {
            let n = 2 * i;
            out.push_str(&k.replace("!@LOOP@", &format!("!{n}")));
            md.push_str(&format!(
                "!{n} = distinct !{{!{n}, !{}}}\n!{} = !{{!\"llvm.loop.unroll.full\"}}\n",
                n + 1,
                n + 1
            ));
        } else {
            out.push_str(k);
        }
    }
    out.push('\n');
    out.push_str(&md);
    // A `wgmma` kernel's swizzle atoms start on 1024-byte boundaries.
    if out.contains("wgmma.mma_async") {
        out = out.replacen("[0 x i8], align 16", "[0 x i8], align 1024", 1);
    }
    Some(out)
}

fn kernel_text(
    name: &str,
    site: &TileSite,
    profile: &TargetProfile,
    contract: bool,
    rungs: crate::GpuRungs,
    reduced: Option<Reduced>,
) -> String {
    let l = launch(site, profile, rungs, reduced);
    let mut k = Kernel {
        out: String::new(),
        next: 0,
        t: lower_ty(&site.elem).expect("float elem lowers"),
        site,
        contract,
        rungs,
        profile_gpu: *profile.gpu().expect("a GPU site has a GPU profile"),
        async_tail: None,
        sync_tail: None,
    };
    match (l.wgmma, l.mma, l.tiled) {
        (Some(w), _, _) => k.wgmma(name, w),
        (None, Some(m), _) => k.mma(name, m),
        (None, None, Some(g)) => k.tiled(name, g),
        (None, None, None) => k.naive(name, l.block),
    }
    k.out
}

/// String builder for one kernel — the device twin of [`FnEmit`]'s
/// `line`/`tmp`, kept separate because a kernel shares no state with the host
/// function around the site.
struct Kernel<'s> {
    out: String,
    next: u32,
    /// `float` or `double`.
    t: String,
    site: &'s TileSite,
    contract: bool,
    rungs: crate::GpuRungs,
    profile_gpu: Gpu,
    /// The panel loop's induction values, threaded from its head to its latch.
    async_tail: Option<(String, String, u64)>,
    sync_tail: Option<(String, String)>,
}

impl Kernel<'_> {
    fn tmp(&mut self) -> String {
        self.next += 1;
        format!("%v{}", self.next)
    }

    fn line(&mut self, s: impl AsRef<str>) {
        self.out.push_str("  ");
        self.out.push_str(s.as_ref());
        self.out.push('\n');
    }

    /// `dst = op i64 lhs, rhs`, returning `dst`.
    fn op(&mut self, code: &str, lhs: &str, rhs: impl std::fmt::Display) -> String {
        let t = self.tmp();
        self.line(format!("{t} = {code} i64 {lhs}, {rhs}"));
        t
    }

    fn sreg(&mut self, reg: &str) -> String {
        let r32 = self.tmp();
        self.line(format!("{r32} = call i32 @llvm.nvvm.read.ptx.sreg.{reg}()"));
        let r = self.tmp();
        self.line(format!("{r} = zext i32 {r32} to i64"));
        r
    }

    /// `ptr addrspace(AS) gep T, base, idx`.
    fn gep(&mut self, space: u32, base: &str, idx: &str) -> String {
        let p = self.tmp();
        self.line(format!(
            "{p} = getelementptr inbounds {}, ptr addrspace({space}) {base}, i64 {idx}",
            self.t
        ));
        p
    }

    /// One accumulation step, in the oracle's operand order:
    /// `acc ⊕ (a ⊗ b)` with each side placed as the source wrote it.
    fn step(&mut self, acc: &str, a: &str, b: &str) -> String {
        let t = self.t.clone();
        let r = self.tmp();
        if self.contract {
            let bits = if t == "float" { 32 } else { 64 };
            self.line(format!(
                "{r} = call {t} @llvm.fma.f{bits}({t} {a}, {t} {b}, {t} {acc})"
            ));
            return r;
        }
        let (x, y) = if self.site.mul_a_first {
            (a, b)
        } else {
            (b, a)
        };
        let p = self.tmp();
        self.line(format!("{p} = fmul {t} {x}, {y}"));
        let (x, y) = if self.site.add_acc_first {
            (acc, p.as_str())
        } else {
            (p.as_str(), acc)
        };
        self.line(format!("{r} = fadd {t} {x}, {y}"));
        r
    }

    /// `blocks` > 0 also emits `.minnctapersm`: the geometry assumed that many
    /// resident blocks per SM when it sized the register tile and the panels,
    /// so ptxas is held to the register budget that makes it true (without
    /// it, sm_90 allocated 166 registers and one block fit — S52 v1).
    fn header(&mut self, name: &str, threads: u64, align: u64, blocks: u64) {
        let min_blocks = if blocks > 0 {
            format!(" \"nvvm.minctasm\"=\"{blocks}\"")
        } else {
            String::new()
        };
        self.out.push_str(&format!(
            "define ptx_kernel void @{name}(\
             ptr addrspace(1) noalias readonly align {align} %a, \
             ptr addrspace(1) noalias readonly align {align} %b, \
             ptr addrspace(1) noalias align {align} %o) \"nvvm.maxntid\"=\"{threads}\"{min_blocks} {{\n\
             entry:\n"
        ));
    }

    /// One thread per output cell, `k` walked in order — the correctness path
    /// for any affine site the tiled geometry does not cover.
    fn naive(&mut self, name: &str, threads: u64) {
        let s = self.site;
        let t = self.t.clone();
        let seed = const_literal(&s.seed);
        let e = elem_bytes_of(&s.elem);
        self.header(name, threads, e, 0);
        let tid = self.sreg("tid.x");
        let bx = self.sreg("ctaid.x");
        let cell0 = self.op("mul", &bx, threads);
        let cell = self.op("add", &cell0, &tid);
        let inside = self.tmp();
        self.line(format!("{inside} = icmp ult i64 {cell}, {}", s.rows * s.c));
        self.line(format!("br i1 {inside}, label %body, label %done"));
        self.out.push_str("body:\n");
        let i = self.op("udiv", &cell, s.c);
        let lane = self.op("urem", &cell, s.c);
        let base = |k: &mut Self, r: &mapal_ir::TileRead| {
            let x = k.op("mul", &i, r.ci);
            let y = k.op("mul", &lane, r.clane);
            let z = k.op("add", &x, &y);
            k.op("add", &z, r.base)
        };
        let abase = base(self, &s.a);
        let bbase = base(self, &s.b);
        self.line("br label %kloop");
        self.out.push_str("kloop:\n");
        let (kv, acc, kn) = (self.tmp(), self.tmp(), self.tmp());
        self.line(format!("{kv} = phi i64 [0, %body], [{kn}, %kloop]"));
        // The back-edge value is the step's result, known only after the body
        // is emitted; patched in below (the marker cannot collide with a `%v`).
        self.line(format!(
            "{acc} = phi {t} [{seed}, %body], [@ACC_NEXT@, %kloop]"
        ));
        let load = |k: &mut Self, arr: &str, base: &str, ck: u64| {
            let off = k.op("mul", &kv, ck);
            let idx = k.op("add", base, &off);
            let p = k.gep(1, arr, &idx);
            let v = k.tmp();
            k.line(format!("{v} = load {t}, ptr addrspace(1) {p}, align {e}"));
            v
        };
        let av = load(self, "%a", &abase, s.a.ck);
        let bv = load(self, "%b", &bbase, s.b.ck);
        let accn = self.step(&acc, &av, &bv);
        self.out = self.out.replace("@ACC_NEXT@", &accn);
        self.line(format!("{kn} = add i64 {kv}, 1"));
        let fin = self.tmp();
        self.line(format!("{fin} = icmp eq i64 {kn}, {}", s.k));
        self.line(format!("br i1 {fin}, label %store, label %kloop"));
        self.out.push_str("store:\n");
        let p = self.gep(1, "%o", &cell);
        self.line(format!("store {t} {accn}, ptr addrspace(1) {p}, align {e}"));
        self.line("br label %done");
        self.out.push_str("done:\n  ret void\n}\n");
    }

    /// Issue panel `tile` (an i64 value) into stage `stage` (an i64 value):
    /// one 16-byte `cp.async` per `(is_a, global index at k0 = 0, shared
    /// word in stage 0)` copy.
    fn issue16(
        &mut self,
        copies: &[(bool, String, String)],
        tile: &str,
        stage: &str,
        bk: u64,
        stage_words: u64,
    ) {
        let s = self.site;
        let k0 = self.op("mul", tile, bk);
        let ka = self.op("mul", &k0, s.a.ck);
        let kb = self.op("mul", &k0, s.b.ck);
        let so = self.op("mul", stage, stage_words);
        for (is_a, gsrc, sdst) in copies {
            let (arr, off) = if *is_a { ("%a", &ka) } else { ("%b", &kb) };
            let gi = self.op("add", gsrc, off);
            let gp = self.gep(1, arr, &gi);
            let si = self.op("add", sdst, &so);
            let sp = self.gep(3, "@mapal_smem", &si);
            self.line(format!(
                "call void @llvm.nvvm.cp.async.cg.shared.global.16(ptr addrspace(3) {sp}, ptr addrspace(1) {gp})"
            ));
        }
    }

    /// `cvt.rn.tf32.f32`: round to nearest, ties to even — ADR-0039's
    /// `to_tf32`. `wgmma` itself truncates the low 13 bits (PTX ISA), so an
    /// operand it reads must already be rounded. `sideeffect` pins the write
    /// before the `wgmma.fence` that orders it.
    fn round_tf32(&mut self, x: &str, pinned: bool) -> String {
        let r = self.tmp();
        let se = if pinned { " sideeffect" } else { "" };
        self.line(format!(
            "{r} = call i32 asm{se} \"cvt.rn.tf32.f32 $0, $1;\", \"=r,f\"(float {x})"
        ));
        r
    }

    /// Round the four floats `v` to TF32 (RNE) and store them at shared word
    /// `si` — what `wgmma` reads must already be rounded.
    fn store_rounded(&mut self, v: &str, si: &str) {
        let sp = self.gep(3, "@mapal_smem", si);
        let mut r = "poison".to_string();
        for j in 0..4 {
            let x = self.tmp();
            self.line(format!("{x} = extractelement <4 x float> {v}, i64 {j}"));
            let y = self.round_tf32(&x, false);
            let nr = self.tmp();
            self.line(format!(
                "{nr} = insertelement <4 x i32> {r}, i32 {y}, i64 {j}"
            ));
            r = nr;
        }
        self.line(format!(
            "store <4 x i32> {r}, ptr addrspace(3) {sp}, align 16"
        ));
    }

    /// `ld.global` this thread's `a` chunks of panel `tile` (an i64 value).
    fn load_a(&mut self, copies: &[(bool, String, String)], tile: &str, bk: u64) -> Vec<String> {
        let k0 = self.op("mul", tile, bk);
        let ka = self.op("mul", &k0, self.site.a.ck);
        let mut vals = Vec::new();
        for (_, gsrc, _) in copies.iter().filter(|c| c.0) {
            let gi = self.op("add", gsrc, &ka);
            let gp = self.gep(1, "%a", &gi);
            let v = self.tmp();
            self.line(format!(
                "{v} = load <4 x float>, ptr addrspace(1) {gp}, align 16"
            ));
            vals.push(v);
        }
        vals
    }

    /// Round and store `vals` (from [`Self::load_a`]) into stage `stage`.
    fn store_a(
        &mut self,
        copies: &[(bool, String, String)],
        vals: &[String],
        stage: &str,
        stage_words: u64,
    ) {
        let so = self.op("mul", stage, stage_words);
        for ((_, _, sdst), v) in copies.iter().filter(|c| c.0).zip(vals) {
            let si = self.op("add", sdst, &so);
            self.store_rounded(v, &si);
        }
    }

    fn asm(&mut self, text: &str) {
        self.line(format!(
            "call void asm sideeffect \"{text}\", \"~{{memory}}\"()"
        ));
    }

    /// **Hopper TF32 through `wgmma` (S53 `wgmma-hopper`).** The site's
    /// product is computed transposed, `cᵀ = bᵀ·aᵀ` ([`GpuWgmmaTile`]): warpgroup
    /// `wg` of block `(bx, by)` owns `c` columns `bx·bn + 64·wg ..+64` (the
    /// instruction's M) × rows `by·n ..+n` (its N).
    ///
    /// - `a` panels are K-major in 128-byte swizzle rows (the descriptor's
    ///   mode 1; SBO one 8-row atom). `wgmma` reads shared memory as it is,
    ///   truncating, so every `a` word is rounded (RNE) once before it lands:
    ///   with `reg_stage`, `ld.global` → `cvt.rn` → `st.shared`, the next
    ///   panel held in registers across the MMA issue and stored while the
    ///   group computes; without, `cp.async` and an in-place pass over the
    ///   chunks this thread copied. Either is fenced to the async proxy before
    ///   the panel barrier.
    /// - `b` panels land row-major, rows padded; each thread loads its `bᵀ`
    ///   fragments (`a0..a3` = (g, t), (g+8, t), (g, t+4), (g+8, t+4) of the
    ///   warp's 16 rows, PTX ISA m64nNk8) and rounds them in the load.
    /// - Pipeline: panel `it + stages − 2` is copied into the stage panel
    ///   `it − 2` used — the group issued at `it − 1` may still be reading
    ///   its own. Without `overlap` the wait for that group sits after the
    ///   barrier, right before its fragment registers are rewritten; with it,
    ///   `wait_group 1` before the barrier retires group `it − 2` only, so
    ///   group `it − 1` computes while group `it` is issued.
    ///
    /// Accumulator `d[4j + 2h + e]` holds `cᵀ(16w + g + 8h, 8j + 2t + e)`;
    /// each warp-wide store fills four whole 32-byte sectors of `c`.
    fn wgmma(&mut self, name: &str, g: GpuWgmmaTile) {
        let s = self.site;
        let seed = const_literal(&s.seed);
        let warp = self.profile_gpu.warp;
        let threads = g.threads(warp);
        let (bm, bn, bk, ldb, n) = (g.bm(), g.bn(), g.bk, g.ldb(), g.n);
        let (stage_words, a_words) = (g.stage_words(), g.a_words());
        let nk = s.k / bk;
        let dist = g.stages - 2;
        let blocks = if self.rungs.min_blocks {
            g.blocks_per_sm
        } else {
            0
        };
        self.header(name, threads, GPU_VEC_BYTES, blocks);
        let smem = "@mapal_smem";

        let tid = self.sreg("tid.x");
        let bx = self.sreg("ctaid.x");
        let by = self.sreg("ctaid.y");
        let wg = self.op("udiv", &tid, WARPGROUP_WARPS * warp);
        let warp_id = self.op("udiv", &tid, warp);
        let w = self.op("urem", &warp_id, WARPGROUP_WARPS);
        let lane = self.op("urem", &tid, warp);
        let grp = self.op("udiv", &lane, 4);
        let t4 = self.op("urem", &lane, 4);
        let row0 = self.op("mul", &by, bm);
        let col0 = self.op("mul", &bx, bn);

        let accs = (n / 2) as usize;
        let acc = self.tmp();
        self.line(format!("{acc} = alloca [{accs} x float], align 16"));
        let acc_slot = |k: &mut Self, idx: usize| {
            let p = k.tmp();
            k.line(format!(
                "{p} = getelementptr inbounds [{accs} x float], ptr {acc}, i64 0, i64 {idx}"
            ));
            p
        };
        for idx in 0..accs {
            let p = acc_slot(self, idx);
            self.line(format!("store float {seed}, ptr {p}, align 4"));
        }

        // Copies, in words: `a` row m, 16-byte chunk kc — atom column
        // ka = kc / 8, chunk c = kc % 8 — lands at ka·bm·128 + m·128 +
        // (c ⊕ m%8)·16 bytes (`Swizzle<3,4,3>`): eight lanes read one whole
        // 128-byte line and write eight distinct bank groups. `b` as the mma
        // kernel stages it.
        let kcs = bk * 4 / GPU_VEC_BYTES;
        let row_chunks = WGMMA_SWIZZLE_BYTES / GPU_VEC_BYTES;
        let row_w = WGMMA_SWIZZLE_BYTES / 4;
        let mut copies: Vec<(bool, String, String)> = Vec::new();
        for p in 0..bm * kcs / threads {
            let li = self.op("add", &tid, p * threads);
            let kc = self.op("urem", &li, kcs);
            let m = self.op("udiv", &li, kcs);
            let grow = self.op("add", &row0, &m);
            let gr = self.op("mul", &grow, s.a.ci);
            let gk = self.op("mul", &kc, 4);
            let g0 = self.op("add", &gr, &gk);
            let gsrc = self.op("add", &g0, s.a.base);
            let ka = self.op("udiv", &kc, row_chunks);
            let c = self.op("urem", &kc, row_chunks);
            let r8 = self.op("urem", &m, WGMMA_ATOM_ROWS);
            let sw = self.op("xor", &c, &r8);
            let sa = self.op("mul", &ka, bm * row_w);
            let sr = self.op("mul", &m, row_w);
            let sc = self.op("mul", &sw, 4);
            let sdst = self.op("add", &sa, &sr);
            let sdst = self.op("add", &sdst, &sc);
            copies.push((true, gsrc, sdst));
        }
        let bc = bn / 4;
        for p in 0..bk * bc / threads {
            let li = self.op("add", &tid, p * threads);
            let kr = self.op("udiv", &li, bc);
            let nc0 = self.op("urem", &li, bc);
            let nc = self.op("mul", &nc0, 4);
            let gk = self.op("mul", &kr, s.b.ck);
            let gc = self.op("add", &col0, &nc);
            let g0 = self.op("add", &gk, &gc);
            let gsrc = self.op("add", &g0, s.b.base);
            let sk = self.op("mul", &kr, ldb);
            let sn = self.op("add", &sk, &nc);
            let sdst = self.op("add", &sn, a_words);
            copies.push((false, gsrc, sdst));
        }
        // With `reg_stage` only `b` is copied asynchronously; `a` goes
        // `ld.global` → round → `st.shared`, one panel held in registers.
        let async_copies: Vec<_> = copies
            .iter()
            .filter(|c| !(g.reg_stage && c.0))
            .cloned()
            .collect();
        for st in 0..dist.min(nk) {
            let st = st.to_string();
            if g.reg_stage {
                let vals = self.load_a(&copies, &st, bk);
                self.store_a(&copies, &vals, &st, stage_words);
            }
            self.issue16(&async_copies, &st, &st, bk, stage_words);
            self.line("call void @llvm.nvvm.cp.async.commit.group()");
        }
        for _ in nk..dist {
            self.line("call void @llvm.nvvm.cp.async.commit.group()");
        }
        // This thread's bᵀ fragment origin inside a stage: row k = t, column
        // n = 64·wg + 16·w + g.
        let fr = self.op("mul", &t4, ldb);
        let fc = self.op("mul", &wg, WGMMA_M);
        let fw = self.op("mul", &w, WGMMA_M / WARPGROUP_WARPS);
        let fo = self.op("add", &fr, &fc);
        let fo = self.op("add", &fo, &fw);
        let fo = self.op("add", &fo, &grp);
        let b_org = self.op("add", &fo, a_words);
        self.line("br label %kt");

        self.out.push_str("kt:\n");
        let (it, itn) = (self.tmp(), self.tmp());
        let latch = if g.reg_stage { "%stored" } else { "%issued" };
        self.line(format!("{it} = phi i64 [0, %entry], [{itn}, {latch}]"));
        if g.overlap {
            // Group `it − 2` done here, in every warpgroup by the barrier
            // below: its stage may be refilled and its fragment registers
            // reused, while group `it − 1` keeps computing.
            self.asm("wgmma.wait_group.sync.aligned 1;");
        }
        self.line(format!(
            "call void @llvm.nvvm.cp.async.wait.group(i32 {})",
            dist - 1
        ));
        let cst = self.op("urem", &it, g.stages);
        let cbase = self.op("mul", &cst, stage_words);
        // Off `reg_stage`: round this thread's own landed `a` chunks of panel
        // `it` in place, once.
        for (_, _, sdst) in copies.iter().filter(|c| c.0 && !g.reg_stage) {
            let si = self.op("add", sdst, &cbase);
            let sp = self.gep(3, smem, &si);
            let v = self.tmp();
            self.line(format!(
                "{v} = load <4 x float>, ptr addrspace(3) {sp}, align 16"
            ));
            self.store_rounded(&v, &si);
        }
        // Generic-proxy writes of `a` (the rounding, or `reg_stage`'s stores
        // at the end of the previous iteration) become visible to `wgmma`.
        self.asm("fence.proxy.async.shared::cta;");
        self.line("call void @llvm.nvvm.barrier0()");
        let nxt = self.op("add", &it, dist);
        let more = self.tmp();
        self.line(format!("{more} = icmp ult i64 {nxt}, {nk}"));
        self.line(format!("br i1 {more}, label %issue, label %issued"));
        self.out.push_str("issue:\n");
        let nst = self.op("urem", &nxt, g.stages);
        let loaded = if g.reg_stage {
            self.load_a(&copies, &nxt, bk)
        } else {
            Vec::new()
        };
        self.issue16(&async_copies, &nxt, &nst, bk, stage_words);
        self.line("br label %issued");
        self.out.push_str("issued:\n");
        let mut staged = Vec::new();
        for v in &loaded {
            let p = self.tmp();
            self.line(format!(
                "{p} = phi <4 x float> [{v}, %issue], [poison, %kt]"
            ));
            staged.push(p);
        }
        self.line("call void @llvm.nvvm.cp.async.commit.group()");
        if !g.overlap {
            // The previous group done: its fragment registers may be rewritten.
            self.asm("wgmma.wait_group.sync.aligned 0;");
        }
        let fb = self.op("add", &cbase, &b_org);
        let mut frags = Vec::new();
        for kk in 0..bk / WGMMA_K {
            let mut f = Vec::new();
            for off in [0, 8, 4 * ldb, 4 * ldb + 8] {
                let i = self.op("add", &fb, kk * WGMMA_K * ldb + off);
                let p = self.gep(3, smem, &i);
                let x = self.tmp();
                self.line(format!("{x} = load float, ptr addrspace(3) {p}, align 4"));
                f.push(self.round_tf32(&x, true));
            }
            frags.push(f);
        }
        self.asm("wgmma.fence.sync.aligned;");
        // Descriptor of this stage's `a` panel; k8 step `kk` starts in atom
        // column `kk·32 / 128`, `kk·32 % 128` bytes into its swizzled rows (the
        // hardware applies the XOR to the address it forms).
        let ap = self.gep(3, smem, &cbase);
        let a32 = self.tmp();
        self.line(format!("{a32} = ptrtoint ptr addrspace(3) {ap} to i32"));
        let a64 = self.tmp();
        self.line(format!("{a64} = zext i32 {a32} to i64"));
        let enc = self.op("lshr", &a64, 4);
        let enc = self.op("and", &enc, 0x3fff);
        let desc0 = self.op("or", &enc, g.desc_fields());
        let k_off = |kk: u64| {
            let b = kk * WGMMA_K * 4;
            (b / WGMMA_SWIZZLE_BYTES * bm * WGMMA_SWIZZLE_BYTES + b % WGMMA_SWIZZLE_BYTES) >> 4
        };
        let sty = format!("{{ {} }}", vec!["float"; accs].join(", "));
        let mut text = format!("wgmma.mma_async.sync.aligned.m64n{n}k8.f32.tf32.tf32 {{");
        text.push_str(
            &(0..accs)
                .map(|i| format!("${i}"))
                .collect::<Vec<_>>()
                .join(","),
        );
        text.push_str(&format!(
            "}}, {{${},${},${},${}}}, ${}, 1, 1, 1;",
            accs,
            accs + 1,
            accs + 2,
            accs + 3,
            accs + 4
        ));
        let mut cons = vec!["=f".to_string(); accs];
        cons.extend(["r", "r", "r", "r", "l"].map(String::from));
        cons.extend((0..accs).map(|i| i.to_string()));
        let cons = cons.join(",");
        for (kk, f) in frags.iter().enumerate() {
            let desc = self.op("add", &desc0, k_off(kk as u64));
            let mut args = format!(
                "i32 {}, i32 {}, i32 {}, i32 {}, i64 {desc}",
                f[0], f[1], f[2], f[3]
            );
            let mut slots = Vec::new();
            for q in 0..accs {
                let p = acc_slot(self, q);
                let v = self.tmp();
                self.line(format!("{v} = load float, ptr {p}, align 4"));
                args.push_str(&format!(", float {v}"));
                slots.push(p);
            }
            let r = self.tmp();
            self.line(format!(
                "{r} = call {sty} asm sideeffect \"{text}\", \"{cons}\"({args})"
            ));
            for (q, p) in slots.iter().enumerate() {
                let d = self.tmp();
                self.line(format!("{d} = extractvalue {sty} {r}, {q}"));
                self.line(format!("store float {d}, ptr {p}, align 4"));
            }
        }
        self.asm("wgmma.commit_group.sync.aligned;");
        if g.reg_stage {
            // While the group computes: panel `it + dist` into the stage the
            // group waited above (`it − 2`) read.
            self.line(format!("br i1 {more}, label %store, label %stored"));
            self.out.push_str("store:\n");
            let sst = self.op("urem", &nxt, g.stages);
            self.store_a(&copies, &staged, &sst, stage_words);
            self.line("br label %stored");
            self.out.push_str("stored:\n");
        }
        self.line(format!("{itn} = add i64 {it}, 1"));
        let fin = self.tmp();
        self.line(format!("{fin} = icmp eq i64 {itn}, {nk}"));
        self.line(format!("br i1 {fin}, label %done, label %kt"));

        self.out.push_str("done:\n");
        self.asm("wgmma.wait_group.sync.aligned 0;");
        let er = self.op("mul", &t4, 2);
        let er = self.op("add", &er, &row0);
        let er = self.op("mul", &er, s.c);
        let ec = self.op("add", &col0, &fc);
        let ec = self.op("add", &ec, &fw);
        let ec = self.op("add", &ec, &grp);
        let ebase = self.op("add", &er, &ec);
        for j in 0..n / 8 {
            for h in 0..2u64 {
                for e in 0..2u64 {
                    let q = (4 * j + 2 * h + e) as usize;
                    let p = acc_slot(self, q);
                    let x = self.tmp();
                    self.line(format!("{x} = load float, ptr {p}, align 4"));
                    let oi = self.op("add", &ebase, (8 * j + e) * s.c + 8 * h);
                    let op = self.gep(1, "%o", &oi);
                    self.line(format!("store float {x}, ptr addrspace(1) {op}, align 4"));
                }
            }
        }
        self.out.push_str("  ret void\n}\n");
    }

    /// The tensor-core kernel (TF32 face): block `(bx, by)` owns a `bm × bn`
    /// output tile; warp `(wr, wc)` a `wm × wn` tile of `mt × nt` m16n8 tiles
    /// whose f32 accumulators live in registers. The reduction is staged in
    /// `stages` shared-memory buffers filled by `cp.async` — panel `it + stages
    /// - 1` is in flight while panel `it` is consumed — and each `bk` panel is
    /// `bk / 8` steps of `mma.sync.m16n8k8`, operands rounded with `cvt.rna`.
    ///
    /// Fragment layouts are the PTX ISA's for m16n8k8 (`group = lane / 4`,
    /// `t = lane % 4`): A `a0..a3` = (g, t), (g+8, t), (g, t+4), (g+8, t+4);
    /// B `b0, b1` = (k t, n g), (k t+4, n g); C `c0..c3` = (g, 2t), (g, 2t+1),
    /// (g+8, 2t), (g+8, 2t+1).
    fn mma(&mut self, name: &str, g: GpuMmaTile) {
        let s = self.site;
        let seed = const_literal(&s.seed);
        let warp = self.profile_gpu.warp;
        let threads = g.threads(warp);
        let (bm, bn, bk, lda, ldb) = (g.bm(), g.bn(), g.bk, g.lda(), g.ldb());
        let (mt, nt) = (g.mt as usize, g.nt as usize);
        let stage_words = g.stage_words();
        let nk = s.k / bk;
        let blocks = if self.rungs.min_blocks {
            g.blocks_per_sm
        } else {
            0
        };
        self.header(name, threads, GPU_VEC_BYTES, blocks);
        let smem = "@mapal_smem";

        let tid = self.sreg("tid.x");
        let bx = self.sreg("ctaid.x");
        let by = self.sreg("ctaid.y");
        let warp_id = self.op("udiv", &tid, warp);
        let lane = self.op("urem", &tid, warp);
        let wr = self.op("udiv", &warp_id, g.warps_x);
        let wc = self.op("urem", &warp_id, g.warps_x);
        let grp = self.op("udiv", &lane, 4);
        let t4 = self.op("urem", &lane, 4);
        let row0 = self.op("mul", &by, bm);
        let col0 = self.op("mul", &bx, bn);

        let accs = mt * nt * 4;
        let acc = self.tmp();
        self.line(format!("{acc} = alloca [{accs} x float], align 16"));
        let acc_slot = |k: &mut Self, idx: usize| {
            let p = k.tmp();
            k.line(format!(
                "{p} = getelementptr inbounds [{accs} x float], ptr {acc}, i64 0, i64 {idx}"
            ));
            p
        };
        for idx in 0..accs {
            let p = acc_slot(self, idx);
            self.line(format!("store float {seed}, ptr {p}, align 4"));
        }

        // cp.async coordinates: (global index at k0 = 0, shared word in stage 0),
        // one per 16-byte chunk this thread copies, A then B.
        let mut copies: Vec<(bool, String, String)> = Vec::new();
        let (ac, bc) = (bk / 4, bn / 4);
        for p in 0..bm * bk / 4 / threads {
            let li = self.op("add", &tid, p * threads);
            let r = self.op("udiv", &li, ac);
            let kc0 = self.op("urem", &li, ac);
            let kc = self.op("mul", &kc0, 4);
            let grow = self.op("add", &row0, &r);
            let gr = self.op("mul", &grow, s.a.ci);
            let g0 = self.op("add", &gr, &kc);
            let gsrc = self.op("add", &g0, s.a.base);
            let sr = self.op("mul", &r, lda);
            let sdst = self.op("add", &sr, &kc);
            copies.push((true, gsrc, sdst));
        }
        for p in 0..bk * bn / 4 / threads {
            let li = self.op("add", &tid, p * threads);
            let kr = self.op("udiv", &li, bc);
            let nc0 = self.op("urem", &li, bc);
            let nc = self.op("mul", &nc0, 4);
            let gk = self.op("mul", &kr, s.b.ck);
            let gc = self.op("add", &col0, &nc);
            let g0 = self.op("add", &gk, &gc);
            let gsrc = self.op("add", &g0, s.b.base);
            let sk = self.op("mul", &kr, ldb);
            let sn = self.op("add", &sk, &nc);
            let sdst = self.op("add", &sn, bm * lda);
            copies.push((false, gsrc, sdst));
        }
        let issue = |k: &mut Self, tile: &str, stage: &str| {
            k.issue16(&copies, tile, stage, bk, stage_words)
        };
        for st in 0..(g.stages - 1).min(nk) {
            issue(self, &st.to_string(), &st.to_string());
            self.line("call void @llvm.nvvm.cp.async.commit.group()");
        }
        for _ in nk..g.stages - 1 {
            self.line("call void @llvm.nvvm.cp.async.commit.group()");
        }
        // Per-thread fragment origins inside a stage.
        let a_r = self.op("mul", &wr, g.wm());
        let a_r = self.op("add", &a_r, &grp);
        let a_r = self.op("mul", &a_r, lda);
        let a_org = self.op("add", &a_r, &t4);
        let b_r = self.op("mul", &t4, ldb);
        let b_c = self.op("mul", &wc, g.wn());
        let b_o = self.op("add", &b_r, &b_c);
        let b_o = self.op("add", &b_o, &grp);
        let b_org = self.op("add", &b_o, bm * lda);
        self.line("br label %kt");

        self.out.push_str("kt:\n");
        let (it, itn) = (self.tmp(), self.tmp());
        self.line(format!("{it} = phi i64 [0, %entry], [{itn}, %issued]"));
        self.line(format!(
            "call void @llvm.nvvm.cp.async.wait.group(i32 {})",
            g.stages - 2
        ));
        self.line("call void @llvm.nvvm.barrier0()");
        let nxt = self.op("add", &it, g.stages - 1);
        let more = self.tmp();
        self.line(format!("{more} = icmp ult i64 {nxt}, {nk}"));
        self.line(format!("br i1 {more}, label %issue, label %issued"));
        self.out.push_str("issue:\n");
        let nst = self.op("urem", &nxt, g.stages);
        issue(self, &nxt, &nst);
        self.line("br label %issued");
        self.out.push_str("issued:\n");
        self.line("call void @llvm.nvvm.cp.async.commit.group()");
        let cst = self.op("urem", &it, g.stages);
        let cbase = self.op("mul", &cst, stage_words);
        let a_base = self.op("add", &cbase, &a_org);
        let b_base = self.op("add", &cbase, &b_org);
        let lds_tf32 = |k: &mut Self, base: &str, off: u64| {
            let i = k.op("add", base, off);
            let p = k.gep(3, smem, &i);
            let x = k.tmp();
            k.line(format!("{x} = load float, ptr addrspace(3) {p}, align 4"));
            let r = k.tmp();
            k.line(format!(
                "{r} = call i32 asm \"cvt.rna.tf32.f32 $0, $1;\", \"=r,f\"(float {x})"
            ));
            r
        };
        // `ldmatrix.x4` (rung `ldmatrix`): the four 8×8 b16 matrices of one
        // m16k8 TF32 fragment — rows 0–7 / 8–15 × k 0–3 / 4–7 — are exactly
        // a0..a3 when each 32-bit element is read as a b16 pair. Lane `l`
        // supplies the row address of matrix `l / 8`, row `l % 8`.
        let lm_base = if self.rungs.ldmatrix {
            let q = self.op("udiv", &lane, 8);
            let i = self.op("urem", &lane, 8);
            let qr = self.op("urem", &q, 2);
            let qk = self.op("udiv", &q, 2);
            let r = self.op("mul", &qr, 8);
            let r = self.op("add", &r, &i);
            let wrow = self.op("mul", &wr, g.wm());
            let r = self.op("add", &r, &wrow);
            let r = self.op("mul", &r, lda);
            let kq = self.op("mul", &qk, 4);
            let o = self.op("add", &r, &kq);
            Some(self.op("add", &cbase, &o))
        } else {
            None
        };
        for kk in 0..bk / MMA_K {
            let k8 = kk * MMA_K;
            let mut af = Vec::new();
            for mi in 0..mt as u64 {
                let r = mi * MMA_M * lda + k8;
                if let Some(lb) = &lm_base {
                    let i = self.op("add", lb, r);
                    let p = self.gep(3, smem, &i);
                    let a32 = self.tmp();
                    self.line(format!("{a32} = ptrtoint ptr addrspace(3) {p} to i32"));
                    let m = self.tmp();
                    self.line(format!(
                        "{m} = call {{ i32, i32, i32, i32 }} asm sideeffect \"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{$0,$1,$2,$3}}, [$4];\", \"=r,=r,=r,=r,r\"(i32 {a32})"
                    ));
                    let mut q = Vec::new();
                    for j in 0..4 {
                        let x = self.tmp();
                        self.line(format!(
                            "{x} = extractvalue {{ i32, i32, i32, i32 }} {m}, {j}"
                        ));
                        let r = self.tmp();
                        self.line(format!(
                            "{r} = call i32 asm \"cvt.rna.tf32.f32 $0, $1;\", \"=r,r\"(i32 {x})"
                        ));
                        q.push(r);
                    }
                    af.push([q[0].clone(), q[1].clone(), q[2].clone(), q[3].clone()]);
                    continue;
                }
                af.push([
                    lds_tf32(self, &a_base, r),
                    lds_tf32(self, &a_base, r + 8 * lda),
                    lds_tf32(self, &a_base, r + 4),
                    lds_tf32(self, &a_base, r + 8 * lda + 4),
                ]);
            }
            let mut bf = Vec::new();
            for ni in 0..nt as u64 {
                let c = k8 * ldb + ni * MMA_N;
                bf.push([
                    lds_tf32(self, &b_base, c),
                    lds_tf32(self, &b_base, c + 4 * ldb),
                ]);
            }
            for mi in 0..mt {
                for ni in 0..nt {
                    let base = (mi * nt + ni) * 4;
                    let mut cur = Vec::new();
                    for q in 0..4 {
                        let p = acc_slot(self, base + q);
                        let v = self.tmp();
                        self.line(format!("{v} = load float, ptr {p}, align 4"));
                        cur.push((p, v));
                    }
                    let r = self.tmp();
                    let [a0, a1, a2, a3] = &af[mi];
                    let [b0, b1] = &bf[ni];
                    self.line(format!(
                        "{r} = call {{ float, float, float, float }} asm \"mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {{$0,$1,$2,$3}}, {{$4,$5,$6,$7}}, {{$8,$9}}, {{$10,$11,$12,$13}};\", \"=f,=f,=f,=f,r,r,r,r,r,r,0,1,2,3\"(i32 {a0}, i32 {a1}, i32 {a2}, i32 {a3}, i32 {b0}, i32 {b1}, float {}, float {}, float {}, float {})",
                        cur[0].1, cur[1].1, cur[2].1, cur[3].1
                    ));
                    for (q, (p, _)) in cur.iter().enumerate() {
                        let d = self.tmp();
                        self.line(format!(
                            "{d} = extractvalue {{ float, float, float, float }} {r}, {q}"
                        ));
                        self.line(format!("store float {d}, ptr {p}, align 4"));
                    }
                }
            }
        }
        self.line(format!("{itn} = add i64 {it}, 1"));
        let fin = self.tmp();
        self.line(format!("{fin} = icmp eq i64 {itn}, {nk}"));
        self.line(format!("br i1 {fin}, label %done, label %kt"));

        // C fragments: (g, 2t..2t+1) and (g+8, 2t..2t+1) of every m16n8 tile.
        self.out.push_str("done:\n");
        let orow = self.op("mul", &wr, g.wm());
        let orow = self.op("add", &orow, &row0);
        let orow = self.op("add", &orow, &grp);
        let ocol = self.op("mul", &wc, g.wn());
        let ocol = self.op("add", &ocol, &col0);
        let t2 = self.op("mul", &t4, 2);
        let ocol = self.op("add", &ocol, &t2);
        for mi in 0..mt {
            for half in 0..2u64 {
                let r = self.op("add", &orow, mi as u64 * MMA_M + half * 8);
                let rr = self.op("mul", &r, s.c);
                let rc = self.op("add", &rr, &ocol);
                for ni in 0..nt {
                    let base = (mi * nt + ni) * 4 + half as usize * 2;
                    let mut vec = "poison".to_string();
                    for j in 0..2 {
                        let p = acc_slot(self, base + j);
                        let x = self.tmp();
                        self.line(format!("{x} = load float, ptr {p}, align 4"));
                        let nv = self.tmp();
                        self.line(format!(
                            "{nv} = insertelement <2 x float> {vec}, float {x}, i64 {j}"
                        ));
                        vec = nv;
                    }
                    let oi = self.op("add", &rc, ni as u64 * MMA_N);
                    let op = self.gep(1, "%o", &oi);
                    self.line(format!(
                        "store <2 x float> {vec}, ptr addrspace(1) {op}, align 8"
                    ));
                }
            }
        }
        self.out.push_str("  ret void\n}\n");
    }

    /// The tiled kernel: block `(bx, by)` owns the `bm × bn` output tile at
    /// `(by·bm, bx·bn)`; thread `(trow, tcol)` owns a `tm × tn` register tile
    /// inside it. Per `bk` step of the reduction, the block stages
    ///
    /// - `A[rows][k0 .. k0+bk]` **transposed** into `As[kk][m]`, so a thread's
    ///   `tm` rows at one `kk` are contiguous (one vector read each), and
    /// - `B[k0 .. k0+bk][cols]` as `Bs[kk][n]`,
    ///
    /// with `ld.global.nc.v4`, then a barrier (the morphism that completes the
    /// `g2s` transmission — plan-s41 §1.4), then every thread folds `bk` steps
    /// into its registers, then a barrier before the panels are overwritten.
    fn tiled(&mut self, name: &str, g: GpuTile) {
        let s = self.site;
        let t = self.t.clone();
        let seed = const_literal(&s.seed);
        let e = elem_bytes_of(&s.elem);
        let lanes = GPU_VEC_BYTES / e;
        let vt = format!("<{lanes} x {t}>");
        let (bm, bn, bk, tm, tn, threads) = (g.bm(), g.bn(), g.bk, g.tm, g.tn, g.threads());
        let (kv, nv) = (bk / lanes, bn / lanes);

        let blocks = if self.rungs.min_blocks {
            g.blocks_per_sm
        } else {
            0
        };
        self.header(name, threads, GPU_VEC_BYTES, blocks);
        let tid = self.sreg("tid.x");
        let bx = self.sreg("ctaid.x");
        let by = self.sreg("ctaid.y");
        // Warp shape (S52 rung `warp`): lay a warp's 32 lanes out as
        // `warp / wx` rows × `wx` columns of threads, `wx` = the lanes one
        // shared-memory wavefront serves with vector loads (128 B / 16 B = 8).
        // A warp then reads 4 + 8 distinct thread fragments per `k` step
        // instead of 2 + 16 — two-thirds of the bytes. Off arm / a block too
        // narrow for it: row-major threads.
        let gpu = self.profile_gpu;
        let wx = GPU_WAVEFRONT_BYTES / GPU_VEC_BYTES;
        let wy = gpu.warp / wx;
        let (trow, tcol) = if self.rungs.warp && g.tx % wx == 0 && g.ty % wy == 0 && g.tx > wx {
            let warp = self.op("udiv", &tid, gpu.warp);
            let lane = self.op("urem", &tid, gpu.warp);
            let warps_x = g.tx / wx;
            let wr = self.op("udiv", &warp, warps_x);
            let wc = self.op("urem", &warp, warps_x);
            let lr = self.op("udiv", &lane, wx);
            let lc = self.op("urem", &lane, wx);
            let r0 = self.op("mul", &wr, wy);
            let c0 = self.op("mul", &wc, wx);
            (self.op("add", &r0, &lr), self.op("add", &c0, &lc))
        } else {
            (self.op("udiv", &tid, g.tx), self.op("urem", &tid, g.tx))
        };
        let row0 = self.op("mul", &by, bm);
        let col0 = self.op("mul", &bx, bn);
        let smem = "@mapal_smem";
        // `sa`: words between two `k` rows of the transposed A panel; `bs_off`:
        // where B starts inside a stage. Async staging pads A's rows and keeps
        // `stages` buffers; register staging has one unpadded buffer.
        let streamed = g.stages >= 2;
        let sa = g.sa();
        let bs_off = bk * sa;
        let stage_elems = g.stage_elems();

        let accs = (tm * tn) as usize;
        let acc = self.tmp();
        self.line(format!(
            "{acc} = alloca [{accs} x {t}], align {GPU_VEC_BYTES}"
        ));
        let acc_slot = |k: &mut Self, idx: usize| {
            let p = k.tmp();
            k.line(format!(
                "{p} = getelementptr inbounds [{accs} x {t}], ptr {acc}, i64 0, i64 {idx}"
            ));
            p
        };
        for idx in 0..accs {
            let p = acc_slot(self, idx);
            self.line(format!("store {t} {seed}, ptr {p}, align {e}"));
        }

        // Loop-invariant load coordinates, one set per load this thread issues:
        // `(global index at k0 = 0, shared index)`. A vector operand moves
        // `lanes` elements along its fast axis per load; a strided one moves one
        // element, consecutive threads taking consecutive panel rows/lanes.
        let (a_vec, b_vec) = (vector_a(s), vector_b(s));
        // Async staging copies `a` element by element (its panel is transposed
        // in shared memory, which a 16-byte copy cannot do): `lanes` lanes run
        // along `k` within one row, then rows — four lanes read one 16-byte
        // global run, a warp covers 8 rows, and `pad_a` keeps those 32 writes
        // on 32 banks.
        let a_elementwise = streamed || !a_vec;
        let mut a_src = Vec::new();
        let a_loads = if a_elementwise {
            bm * bk / threads
        } else {
            bm * kv / threads
        };
        for p in 0..a_loads {
            let li = self.op("add", &tid, p * threads);
            let (r, kc) = if streamed {
                let j = self.op("urem", &li, lanes);
                let q = self.op("udiv", &li, lanes);
                let r = self.op("urem", &q, bm);
                let c = self.op("udiv", &q, bm);
                let kc = self.op("mul", &c, lanes);
                (r, self.op("add", &kc, &j))
            } else if a_vec {
                let r = self.op("udiv", &li, kv);
                let kc0 = self.op("urem", &li, kv);
                (r, self.op("mul", &kc0, lanes))
            } else {
                let kc = self.op("udiv", &li, bm);
                (self.op("urem", &li, bm), kc)
            };
            let grow = self.op("add", &row0, &r);
            let gr = self.op("mul", &grow, s.a.ci);
            let gk = self.op("mul", &kc, s.a.ck);
            let g0 = self.op("add", &gr, &gk);
            let gsrc = self.op("add", &g0, s.a.base);
            let skc = self.op("mul", &kc, sa);
            let sdst = self.op("add", &skc, &r);
            a_src.push((gsrc, sdst));
        }
        let mut b_src = Vec::new();
        let b_loads = if b_vec {
            bk * nv / threads
        } else {
            bk * bn / threads
        };
        for p in 0..b_loads {
            let li = self.op("add", &tid, p * threads);
            let (kr, nc) = if b_vec {
                let kr = self.op("udiv", &li, nv);
                let nc0 = self.op("urem", &li, nv);
                (kr, self.op("mul", &nc0, lanes))
            } else {
                let kr = self.op("udiv", &li, bn);
                (kr, self.op("urem", &li, bn))
            };
            let gk = self.op("mul", &kr, s.b.ck);
            let gc = self.op("add", &col0, &nc);
            let gl = self.op("mul", &gc, s.b.clane);
            let g0 = self.op("add", &gk, &gl);
            let gsrc = self.op("add", &g0, s.b.base);
            let sk = self.op("mul", &kr, bn);
            let sn = self.op("add", &sk, &nc);
            let sdst = self.op("add", &sn, bs_off);
            b_src.push((gsrc, sdst));
        }
        // Bank geometry: a thread's register tile is `tm/lanes × tn/lanes`
        // vector chunks spaced one thread-row of chunks apart (`ty·lanes` rows,
        // `tx·lanes` columns), not one contiguous `tm × tn` block. Consecutive
        // threads then read consecutive 16-byte chunks of a panel row, which is
        // what a shared-memory wavefront serves without a conflict — and their
        // output stores coalesce. The CPU's move-panel rung answers the same
        // question for cache sets; here the geometry is the 32 × 4 B banks.
        // (Off arm: one contiguous `tm × tn` block per thread, chunk = lanes.)
        let (a_step, b_step, a_chunk, b_chunk) = if self.rungs.interleave {
            (lanes, lanes, g.ty * lanes, g.tx * lanes)
        } else {
            (tm, tn, lanes, lanes)
        };
        let a_frag0 = self.op("mul", &trow, a_step);
        let b_frag0 = self.op("mul", &tcol, b_step);

        // --- panel staging: global → (registers) → shared memory
        let a_ty = if a_vec { vt.clone() } else { t.clone() };
        let b_ty = if b_vec { vt.clone() } else { t.clone() };
        let a_al = if a_vec { GPU_VEC_BYTES } else { e };
        let b_al = if b_vec { GPU_VEC_BYTES } else { e };
        let load_panels = |k: &mut Self, k0: &str| -> Vec<String> {
            let ka = k.op("mul", k0, s.a.ck);
            let kb = k.op("mul", k0, s.b.ck);
            let mut vals = Vec::new();
            for (gsrc, _) in &a_src {
                let gi = k.op("add", gsrc, &ka);
                let gp = k.gep(1, "%a", &gi);
                let v = k.tmp();
                k.line(format!(
                    "{v} = load {a_ty}, ptr addrspace(1) {gp}, align {a_al}"
                ));
                vals.push(v);
            }
            for (gsrc, _) in &b_src {
                let gi = k.op("add", gsrc, &kb);
                let gp = k.gep(1, "%b", &gi);
                let v = k.tmp();
                k.line(format!(
                    "{v} = load {b_ty}, ptr addrspace(1) {gp}, align {b_al}"
                ));
                vals.push(v);
            }
            vals
        };
        let store_panels = |k: &mut Self, vals: &[String]| {
            for ((_, sdst), v) in a_src.iter().zip(vals) {
                if !a_vec {
                    let sp = k.gep(3, smem, sdst);
                    k.line(format!("store {t} {v}, ptr addrspace(3) {sp}, align {e}"));
                    continue;
                }
                // `a` is staged transposed: one vector along k scatters to
                // `lanes` panel rows.
                for j in 0..lanes {
                    let x = k.tmp();
                    k.line(format!("{x} = extractelement {vt} {v}, i64 {j}"));
                    let si = k.op("add", sdst, j * sa);
                    let sp = k.gep(3, smem, &si);
                    k.line(format!("store {t} {x}, ptr addrspace(3) {sp}, align {e}"));
                }
            }
            for ((_, sdst), v) in b_src.iter().zip(&vals[a_src.len()..]) {
                let sp = k.gep(3, smem, sdst);
                k.line(format!(
                    "store {b_ty} {v}, ptr addrspace(3) {sp}, align {b_al}"
                ));
            }
        };
        let tys: Vec<&String> = a_src
            .iter()
            .map(|_| &a_ty)
            .chain(b_src.iter().map(|_| &b_ty))
            .collect();

        // `cbase`: the stage the compute reads (async), else the one buffer.
        let (cbase, pre) = if streamed {
            // Issue panel `tile` into stage `stage`: one `cp.async` per unit.
            let issue = |k: &mut Self, tile: &str, stage: &str| {
                let k0 = k.op("mul", tile, bk);
                let ka = k.op("mul", &k0, s.a.ck);
                let kb = k.op("mul", &k0, s.b.ck);
                let so = k.op("mul", stage, stage_elems);
                for (gsrc, sdst) in &a_src {
                    let gi = k.op("add", gsrc, &ka);
                    let gp = k.gep(1, "%a", &gi);
                    let si = k.op("add", sdst, &so);
                    let sp = k.gep(3, smem, &si);
                    k.line(format!(
                        "call void @llvm.nvvm.cp.async.ca.shared.global.{e}(ptr addrspace(3) {sp}, ptr addrspace(1) {gp})"
                    ));
                }
                for (gsrc, sdst) in &b_src {
                    let gi = k.op("add", gsrc, &kb);
                    let gp = k.gep(1, "%b", &gi);
                    let si = k.op("add", sdst, &so);
                    let sp = k.gep(3, smem, &si);
                    let (kind, n) = if b_vec {
                        ("cg", GPU_VEC_BYTES)
                    } else {
                        ("ca", e)
                    };
                    k.line(format!(
                        "call void @llvm.nvvm.cp.async.{kind}.shared.global.{n}(ptr addrspace(3) {sp}, ptr addrspace(1) {gp})"
                    ));
                }
            };
            let nk = s.k / bk;
            for st in 0..(g.stages - 1).min(nk) {
                issue(self, &st.to_string(), &st.to_string());
                self.line("call void @llvm.nvvm.cp.async.commit.group()");
            }
            for _ in nk..g.stages - 1 {
                self.line("call void @llvm.nvvm.cp.async.commit.group()");
            }
            self.line("br label %kt");
            self.out.push_str("kt:\n");
            let (it, itn) = (self.tmp(), self.tmp());
            self.line(format!("{it} = phi i64 [0, %entry], [{itn}, %ktl]"));
            self.line(format!(
                "call void @llvm.nvvm.cp.async.wait.group(i32 {})",
                g.stages - 2
            ));
            // One barrier per panel: panel `it` is visible to every thread, and
            // every thread is done with the stage the next issue overwrites.
            self.line("call void @llvm.nvvm.barrier0()");
            let nxt = self.op("add", &it, g.stages - 1);
            let more = self.tmp();
            self.line(format!("{more} = icmp ult i64 {nxt}, {nk}"));
            self.line(format!("br i1 {more}, label %issue, label %issued"));
            self.out.push_str("issue:\n");
            let nst = self.op("urem", &nxt, g.stages);
            issue(self, &nxt, &nst);
            self.line("br label %issued");
            self.out.push_str("issued:\n");
            self.line("call void @llvm.nvvm.cp.async.commit.group()");
            let cst = self.op("urem", &it, g.stages);
            let cbase = self.op("mul", &cst, stage_elems);
            self.line("br label %kk");
            self.async_tail = Some((it, itn, nk));
            (cbase, "issued")
        } else {
            // Prefetch rung: panel 0 is loaded before the loop; inside it, the
            // registers holding panel `k0` are stored, and panel `k0 + bk` is
            // requested before the compute starts, so its latency hides behind
            // `bk` steps of arithmetic. The last trip re-reads its own panel
            // (in bounds, never stored) rather than branching.
            let first = if self.rungs.prefetch {
                load_panels(self, "0")
            } else {
                Vec::new()
            };
            self.line("br label %kt");
            self.out.push_str("kt:\n");
            let (k0, k0n) = (self.tmp(), self.tmp());
            self.line(format!("{k0} = phi i64 [0, %entry], [{k0n}, %ktl]"));
            if self.rungs.prefetch {
                let held: Vec<String> = first.iter().map(|_| self.tmp()).collect();
                for (i, (h, f)) in held.iter().zip(&first).enumerate() {
                    self.line(format!(
                        "{h} = phi {} [{f}, %entry], [@NEXT{i}@, %ktl]",
                        tys[i]
                    ));
                }
                store_panels(self, &held);
                self.line("call void @llvm.nvvm.barrier0()");
                let nk = self.op("add", &k0, bk);
                let more = self.tmp();
                self.line(format!("{more} = icmp ult i64 {nk}, {}", s.k));
                let kl = self.tmp();
                self.line(format!("{kl} = select i1 {more}, i64 {nk}, i64 {k0}"));
                let next = load_panels(self, &kl);
                for (i, n) in next.iter().enumerate() {
                    self.out = self.out.replace(&format!("@NEXT{i}@"), n);
                }
            } else {
                let vals = load_panels(self, &k0);
                store_panels(self, &vals);
                self.line("call void @llvm.nvvm.barrier0()");
            }
            self.line("br label %kk");
            self.sync_tail = Some((k0, k0n));
            ("0".to_string(), "kt")
        };

        // --- bk register steps over the staged panels
        self.out.push_str("kk:\n");
        let (kk, kkn) = (self.tmp(), self.tmp());
        self.line(format!("{kk} = phi i64 [0, %{pre}], [{kkn}, %kk]"));
        let frag = |k: &mut Self, row: u64, first: &str, n: u64, off: u64, chunk: u64| {
            let r = k.op("mul", &kk, row);
            let r = k.op("add", &r, first);
            let r = k.op("add", &r, off);
            let r = k.op("add", &r, &cbase);
            let mut vals = Vec::new();
            for v in 0..n / lanes {
                let si = k.op("add", &r, v * chunk);
                let sp = k.gep(3, smem, &si);
                let x = k.tmp();
                k.line(format!(
                    "{x} = load {vt}, ptr addrspace(3) {sp}, align {GPU_VEC_BYTES}"
                ));
                for j in 0..lanes {
                    let y = k.tmp();
                    k.line(format!("{y} = extractelement {vt} {x}, i64 {j}"));
                    vals.push(y);
                }
            }
            vals
        };
        let av = frag(self, sa, &a_frag0, tm, 0, a_chunk);
        let bv = frag(self, bn, &b_frag0, tn, bs_off, b_chunk);
        for i in 0..tm as usize {
            for j in 0..tn as usize {
                let p = acc_slot(self, i * tn as usize + j);
                let cur = self.tmp();
                self.line(format!("{cur} = load {t}, ptr {p}, align {e}"));
                let r = self.step(&cur, &av[i], &bv[j]);
                self.line(format!("store {t} {r}, ptr {p}, align {e}"));
            }
        }
        self.line(format!("{kkn} = add i64 {kk}, 1"));
        let kkd = self.tmp();
        self.line(format!("{kkd} = icmp eq i64 {kkn}, {bk}"));
        // Fully unrolled: every shared-memory offset becomes an immediate.
        let unroll = if self.rungs.unroll {
            ", !llvm.loop !@LOOP@"
        } else {
            ""
        };
        self.line(format!("br i1 {kkd}, label %ktl, label %kk{unroll}"));

        self.out.push_str("ktl:\n");
        let ktd = self.tmp();
        if let Some((it, itn, nk)) = self.async_tail.take() {
            // No barrier here: the next panel's barrier orders the overwrite.
            self.line(format!("{itn} = add i64 {it}, 1"));
            self.line(format!("{ktd} = icmp eq i64 {itn}, {nk}"));
        } else {
            let (k0, k0n) = self.sync_tail.take().expect("sync panel loop");
            self.line("call void @llvm.nvvm.barrier0()");
            self.line(format!("{k0n} = add i64 {k0}, {bk}"));
            self.line(format!("{ktd} = icmp eq i64 {k0n}, {}", s.k));
        }
        self.line(format!("br i1 {ktd}, label %done, label %kt"));

        // --- one vector store per `lanes` columns of each owned row
        self.out.push_str("done:\n");
        for i in 0..tm {
            let r = self.op("add", &row0, &a_frag0);
            let r = self.op("add", &r, (i / lanes) * a_chunk + i % lanes);
            let rr = self.op("mul", &r, s.c);
            let c0 = self.op("add", &rr, &col0);
            let c0 = self.op("add", &c0, &b_frag0);
            for v in 0..tn / lanes {
                let mut vec = "poison".to_string();
                for j in 0..lanes {
                    let p = acc_slot(self, (i * tn + v * lanes + j) as usize);
                    let x = self.tmp();
                    self.line(format!("{x} = load {t}, ptr {p}, align {e}"));
                    let nv = self.tmp();
                    self.line(format!("{nv} = insertelement {vt} {vec}, {t} {x}, i64 {j}"));
                    vec = nv;
                }
                let oi = self.op("add", &c0, v * b_chunk);
                let op = self.gep(1, "%o", &oi);
                self.line(format!(
                    "store {vt} {vec}, ptr addrspace(1) {op}, align {GPU_VEC_BYTES}"
                ));
            }
        }
        self.out.push_str("  ret void\n}\n");
    }
}

impl<'a> FnEmit<'a> {
    /// Host half: hand the site's operands and output to the runtime, which
    /// copies them in, launches [`kernel_name`]`(id)`, and copies the output
    /// back before returning — so everything after the site sees the values a
    /// host loop would have stored.
    pub(super) fn emit_tiled_map_gpu(
        &mut self,
        source: ObjectId,
        target: ObjectId,
        site: &TileSite,
    ) {
        let src_ty = self.obj_ty(source);
        let bytes = |ty: &Ty| {
            let (_, n) = array_parts(ty);
            n * elem_bytes_of(&site.elem)
        };
        let a_bytes = bytes(src_ty.component_ty(site.a.slot).expect("tile a array"));
        let b_bytes = bytes(src_ty.component_ty(site.b.slot).expect("tile b array"));
        let o_bytes = bytes(&self.obj_ty(target));
        let a_ptr = self
            .array_operand_ptr(source, Some(site.a.slot))
            .expect("tile a ptr");
        let b_ptr = self
            .array_operand_ptr(source, Some(site.b.slot))
            .expect("tile b ptr");
        let o_ptr = self.slot(target).expect("tile output slot");
        // LLVM reads an `i64` literal as signed; the runtime prints the same
        // 64 bits back as hex, so the two units agree on the name.
        let id = kernel_id(site, self.profile, self.contract, self.reduced) as i64;
        let l = launch(site, self.profile, self.gpu_rungs, self.reduced);
        self.line(format!(
            "call void @mapal_gpu_tile(i64 {id}, ptr {a_ptr}, i64 {a_bytes}, ptr {b_ptr}, \
             i64 {b_bytes}, ptr {o_ptr}, i64 {o_bytes}, i32 {}, i32 {}, i32 {}, i32 {})",
            l.gx, l.gy, l.block, l.smem
        ));
    }
}
