//! `Kernel` stream-K walk, ticket and finish steps.

use super::*;

impl Kernel<'_> {
    /// **S63 `streamk`'s walk** (ADR-0040 D4): block `ctaid.x` owns units
    /// `[cta·iters, min((cta+1)·iters, tiles·P))` of the `tiles × panels`
    /// space, `P = panels`. Emits the entry's branch into the piece loop's
    /// head `seg`, and there the current piece: tile `it / P` (in the grid's
    /// raster order), panels `[p0, p1)`, `p1 = min(P, p0 + (end − it))`.
    /// Every block of the grid is resident (`ctas ≤ wave`), so the units a
    /// block walks are its whole share; a tile cut between blocks combines in
    /// the output ([`Kernel::store_o`]). The bounds sit inside the site's
    /// proved panels: the reads are the ordered kernel's.
    pub(super) fn segment(&mut self, iters: u64, panels: u64, bk: u64) -> Segment {
        let ix = self.idx;
        let (gx, h) = self.raster;
        let total = gx * self.gy_out * panels;
        let cta = self.sreg("ctaid.x");
        let (start, end0, limit) = match self.whole {
            0 => {
                let start = self.op("mul", &cta, iters);
                let end0 = self.op("add", &start, iters);
                (start, end0, total.to_string())
            }
            // ADR-0050 C5 hybrid: the walk takes the raster order's first
            // `tail = tiles − whole` tiles (units `[0, tail·P)`, so the zero
            // pass is the output's prefix those tiles' rows span), the whole
            // waves the rest: block `cta < whole` folds tile `tail + cta`
            // whole, a later one walks `iters` units from `(cta − whole)·
            // iters`, never past `tail·P`.
            whole => {
                let tail = total / panels - whole;
                let ordered = self.tmp();
                self.line(format!("{ordered} = icmp ult {ix} {cta}, {whole}"));
                let past = self.tmp();
                self.line(format!(
                    "{past} = select i1 {ordered}, {ix} {whole}, {ix} {cta}"
                ));
                let t0 = self.tmp();
                self.line(format!("{t0} = sub {ix} {past}, {whole}"));
                let walk = self.op("mul", &t0, iters);
                let tile = self.plus(cta.clone(), tail);
                let own = self.op("mul", &tile, panels);
                let start = self.tmp();
                self.line(format!(
                    "{start} = select i1 {ordered}, {ix} {own}, {ix} {walk}"
                ));
                let step = self.tmp();
                self.line(format!(
                    "{step} = select i1 {ordered}, {ix} {panels}, {ix} {iters}"
                ));
                let end0 = self.tmp();
                self.line(format!("{end0} = add {ix} {start}, {step}"));
                let limit = self.tmp();
                self.line(format!(
                    "{limit} = select i1 {ordered}, {ix} {total}, {ix} {}",
                    tail * panels
                ));
                (start, end0, limit)
            }
        };
        let past = self.tmp();
        self.line(format!("{past} = icmp ult {ix} {end0}, {limit}"));
        let end = self.tmp();
        self.line(format!(
            "{end} = select i1 {past}, {ix} {end0}, {ix} {limit}"
        ));
        self.line("br label %seg");
        self.out.push_str("seg:\n");
        let it = self.tmp();
        self.line(format!(
            "{it} = phi {ix} [{start}, %entry], [@SEG_NEXT@, %segl]"
        ));
        let tile = self.op("udiv", &it, panels);
        let p0 = self.op("urem", &it, panels);
        let left = self.tmp();
        self.line(format!("{left} = sub {ix} {end}, {it}"));
        let p1a = self.op("add", &p0, &left);
        let fits = self.tmp();
        self.line(format!("{fits} = icmp ult {ix} {p1a}, {panels}"));
        let p1 = self.tmp();
        self.line(format!(
            "{p1} = select i1 {fits}, {ix} {p1a}, {ix} {panels}"
        ));
        let bx = self.op("urem", &tile, gx);
        let by = self.op("udiv", &tile, gx);
        let (bx, by) = if h < 2 {
            (bx, by)
        } else {
            self.line(format!("; raster: {h} block-rows per group"));
            let (c, r) = raster_block(self, &bx, &by, gx, h);
            (self.opaque(&c), self.opaque(&r))
        };
        let kbeg = self.op("mul", &p0, bk);
        let kend = self.op("mul", &p1, bk);
        let base = self.op("mul", &tile, panels);
        let next = self.op("add", &base, &p1);
        self.p0 = Some(p0.clone());
        self.head = "seg";
        Segment {
            tile,
            bx,
            by,
            kbeg,
            kend,
            next,
            end,
            p0,
            p1,
        }
    }

    /// **sweep-runtime-2 `streamk-epi`: is the piece its tile's whole
    /// fold?** The walk names the blocks that share tile `t`: `c₀ = ⌊t·P /
    /// iters⌋ … c₁ = ⌊((t+1)·P − 1) / iters⌋`. One block (`c₀ = c₁`): the
    /// piece is the tile, and its sums are whole in registers. Shared: the
    /// tile's workspace slot is `c₀` — a block starts at most one tile it
    /// does not finish, so the slots of shared tiles are distinct and the
    /// workspace is `ctas` tiles, not the output. Branches to `st_whole` /
    /// `st_part`.
    pub(super) fn ticket_split(
        &mut self,
        sg: &Segment,
        panels: u64,
        (bm, bn): (u64, u64),
    ) -> TicketSplit {
        let ix = self.idx;
        let iters = self.streamk.expect("a stream-K kernel");
        let first = self.op("mul", &sg.tile, panels);
        let (c0, c1) = match self.whole {
            0 => {
                let c0 = self.op("udiv", &first, iters);
                let lastu = self.plus(first, panels - 1);
                (c0, self.op("udiv", &lastu, iters))
            }
            // ADR-0050 C5 hybrid: tile `t ≥ tail` is block `t − tail`'s
            // alone; a walked tile's blocks count from `whole` over the
            // walk's units `[0, tail·P)` ([`Kernel::segment`]).
            whole => {
                let tail = self.raster.0 * self.gy_out - whole;
                let ordered = self.tmp();
                self.line(format!("{ordered} = icmp uge {ix} {}, {tail}", sg.tile));
                let own0 = self.tmp();
                self.line(format!(
                    "{own0} = select i1 {ordered}, {ix} {}, {ix} {tail}",
                    sg.tile
                ));
                let own = self.tmp();
                self.line(format!("{own} = sub {ix} {own0}, {tail}"));
                let q0 = self.op("udiv", &first, iters);
                let w0 = self.plus(q0, whole);
                let lastu = self.plus(first.clone(), panels - 1);
                let q1 = self.op("udiv", &lastu, iters);
                let w1 = self.plus(q1, whole);
                let (c0, c1) = (self.tmp(), self.tmp());
                self.line(format!("{c0} = select i1 {ordered}, {ix} {own}, {ix} {w0}"));
                self.line(format!("{c1} = select i1 {ordered}, {ix} {own}, {ix} {w1}"));
                (c0, c1)
            }
        };
        let shared = self.tmp();
        self.line(format!("{shared} = sub {ix} {c1}, {c0}"));
        let whole = self.tmp();
        self.line(format!("{whole} = icmp eq {ix} {c1}, {c0}"));
        let base = self.op("mul", &c0, bm * bn);
        self.line(format!("br i1 {whole}, label %st_whole, label %st_part"));
        let units = self.raster.0 * self.gy_out * panels - self.whole * panels;
        let ctas = self.whole + units.div_ceil(iters);
        TicketSplit {
            base,
            slot: c0,
            shared,
            ctas,
        }
    }

    /// **sweep-runtime-2 `streamk-epi`: the shared tile's ticket.** After a
    /// piece of a shared tile has added its partials into the tile's
    /// workspace slot (`%w`, zeroed by the runtime), its block takes one
    /// ticket: release fence, barrier, thread 0's `atomicrmw add` (device
    /// scope, acq_rel) on the slot's `i32` past the slots, the old count
    /// broadcast through shared memory. The block that draws the last ticket
    /// (`c₁ − c₀`, [`Kernel::ticket_split`]) holds the whole sums: after an
    /// acquire fence it reads the slot's `bm × bn` cells (volatile: from L2,
    /// not a stale L1 line), `lanes` consecutive cells a thread a step, and
    /// evaluates the group's members on them ([`Kernel::store_cells`], the
    /// ordered kernel's store). Every other block goes on to its next piece.
    /// The tiled kernel's tiles divide the site, so no cell is guarded.
    pub(super) fn ticket(
        &mut self,
        sp: &TicketSplit,
        (bm, bn, threads): (u64, u64, u64),
        (row0, col0, tid): (&str, &str, &str),
    ) {
        let ix = self.idx;
        let s = self.site;
        let (t, e) = (self.t.clone(), elem_bytes_of(&s.elem));
        let base = (sp.ctas * bm * bn * e).div_ceil(GPU_VEC_BYTES) * GPU_VEC_BYTES / 4;
        self.line("; streamk-epi: the tile's ticket");
        self.line("fence syncscope(\"device\") release");
        self.line("call void @llvm.nvvm.barrier0()");
        let lead = self.tmp();
        self.line(format!("{lead} = icmp eq {ix} {tid}, 0"));
        self.line(format!("br i1 {lead}, label %tk_draw, label %tk_join"));
        self.out.push_str("tk_draw:\n");
        let slot64 = if ix == "i32" {
            let z = self.tmp();
            self.line(format!("{z} = zext nneg i32 {} to i64", sp.slot));
            z
        } else {
            sp.slot.clone()
        };
        let ti = self.tmp();
        self.line(format!("{ti} = add nuw nsw i64 {slot64}, {base}"));
        let tp = self.tmp();
        self.line(format!(
            "{tp} = getelementptr inbounds i32, ptr addrspace(1) %w, i64 {ti}"
        ));
        let old = self.tmp();
        self.line(format!(
            "{old} = atomicrmw add ptr addrspace(1) {tp}, i32 1 syncscope(\"device\") acq_rel, align 4"
        ));
        self.line(format!(
            "store i32 {old}, ptr addrspace(3) @mapal_smem, align 4"
        ));
        self.line("br label %tk_join");
        self.out.push_str("tk_join:\n");
        self.line("call void @llvm.nvvm.barrier0()");
        let drawn = self.tmp();
        self.line(format!(
            "{drawn} = load i32, ptr addrspace(3) @mapal_smem, align 4"
        ));
        // The next piece's panels overwrite the slot only after every
        // thread has read it.
        self.line("call void @llvm.nvvm.barrier0()");
        let shared32 = if ix == "i32" {
            sp.shared.clone()
        } else {
            let z = self.tmp();
            self.line(format!("{z} = trunc i64 {} to i32", sp.shared));
            z
        };
        let last = self.tmp();
        self.line(format!("{last} = icmp eq i32 {drawn}, {shared32}"));
        self.line(format!("br i1 {last}, label %tk_epi, label %tk_end"));
        self.out.push_str("tk_epi:\n");
        self.line("fence syncscope(\"device\") acquire");
        self.line("br label %tk_cell");
        let lanes = GPU_VEC_BYTES / e;
        let lanes = match bn % lanes == 0 && (bm * bn) % (threads * lanes) == 0 {
            true => lanes,
            false => 1,
        };
        let steps = bm * bn / (threads * lanes);
        self.out.push_str("tk_cell:\n");
        let (j, jn) = (self.tmp(), self.tmp());
        self.line(format!("{j} = phi {ix} [0, %tk_epi], [{jn}, %tk_next]"));
        let q0 = self.op("mul", &j, threads);
        let q1 = self.op("add", &q0, tid);
        let q = self.op("mul", &q1, lanes);
        let r = self.op("udiv", &q, bn);
        let cc = self.op("urem", &q, bn);
        let gr = self.op("add", row0, &r);
        let gc = self.op("add", col0, &cc);
        let oi = match s.radix.as_deref().and_then(|x| x.out.as_ref()) {
            Some(o) => self.radix_at(o, &gr, &gc),
            None => {
                let rr = self.op("mul", &gr, s.c);
                self.op("add", &rr, &gc)
            }
        };
        let wi = self.op("add", &sp.base, &q);
        let p = self.gep(1, "%w", &wi);
        let xs: Vec<String> = if lanes == 1 {
            let x = self.tmp();
            self.line(format!(
                "{x} = load volatile {t}, ptr addrspace(1) {p}, align {e}"
            ));
            vec![x]
        } else {
            let vt = format!("<{lanes} x {t}>");
            let v = self.tmp();
            self.line(format!(
                "{v} = load volatile {vt}, ptr addrspace(1) {p}, align {}",
                lanes * e
            ));
            (0..lanes)
                .map(|l| {
                    let x = self.tmp();
                    self.line(format!("{x} = extractelement {vt} {v}, i64 {l}"));
                    x
                })
                .collect()
        };
        self.store_cells(&xs, &oi);
        self.line("br label %tk_next");
        self.out.push_str("tk_next:\n");
        self.line(format!("{jn} = add {ix} {j}, 1"));
        let more = self.tmp();
        self.line(format!("{more} = icmp ult {ix} {jn}, {steps}"));
        self.line(format!("br i1 {more}, label %tk_cell, label %tk_end"));
        self.out.push_str("tk_end:\n");
    }

    /// The kernel's end: `ret`, or in a stream-K kernel the piece loop's
    /// latch — the next piece starts at the unit after this one's last.
    pub(super) fn finish(&mut self, seg: Option<&Segment>) {
        let Some(sg) = seg else {
            self.out.push_str("  ret void\n}\n");
            return;
        };
        let ix = self.idx;
        self.line("br label %segl");
        self.out.push_str("segl:\n");
        let more = self.tmp();
        self.line(format!("{more} = icmp ult {ix} {}, {}", sg.next, sg.end));
        self.line(format!("br i1 {more}, label %seg, label %exit"));
        self.out.push_str("exit:\n  ret void\n}\n");
        self.out = self.out.replace("@SEG_NEXT@", &sg.next);
    }
}

/// sweep-runtime-2 `streamk-epi`: a shared tile's workspace slot
/// ([`Kernel::ticket_split`]).
pub(super) struct TicketSplit {
    /// The slot's first workspace cell (`c₀ · bm · bn`).
    pub(super) base: String,
    /// The slot (`c₀`, the tile's first block): its ticket's index.
    pub(super) slot: String,
    /// The tile's blocks less one: the last ticket's number.
    pub(super) shared: String,
    /// The walk's blocks: the workspace's slots.
    pub(super) ctas: u64,
}

/// One stream-K piece's values ([`Kernel::segment`]).
pub(super) struct Segment {
    /// The piece's tile, in the walk's linear order (before `raster`).
    pub(super) tile: String,
    pub(super) bx: String,
    pub(super) by: String,
    pub(super) kbeg: String,
    pub(super) kend: String,
    /// The first unit after the piece, and the block's end.
    pub(super) next: String,
    pub(super) end: String,
    /// The piece's panels `[p0, p1)` of its tile (bf16 tensor-tile
    /// `mma-walk`: the tensor tile's pipeline runs over them).
    pub(super) p0: String,
    pub(super) p1: String,
}
