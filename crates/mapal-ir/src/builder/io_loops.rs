//! `FnBuilder` effects and control: print, clock, read, output, loops and `finish`.

use super::*;

impl FnBuilder<'_> {
    /// `print` — raw, no trailing newline (ADR-0015): emits `Print { newline: false }`.
    pub fn print(
        &mut self,
        token: ObjectId,
        value: ObjectId,
        loc: SourceLoc,
    ) -> Result<ObjectId, IrError> {
        self.print_with(token, value, false, loc)
    }

    /// `println` — appends a trailing `"\n"` (ADR-0015): emits `Print { newline: true }`.
    pub fn println(
        &mut self,
        token: ObjectId,
        value: ObjectId,
        loc: SourceLoc,
    ) -> Result<ObjectId, IrError> {
        self.print_with(token, value, true, loc)
    }

    /// `Print { newline }` (DESIGN §5.1): `(IoToken, P) → IoToken`. Builds the internal
    /// `(IoToken, P)` pair (the only place `Str` may be a pair component, I9s).
    pub(super) fn print_with(
        &mut self,
        token: ObjectId,
        value: ObjectId,
        newline: bool,
        loc: SourceLoc,
    ) -> Result<ObjectId, IrError> {
        self.check_obj(token)?;
        self.check_obj(value)?;
        let tty = self.ty_of(token);
        let vty = self.ty_of(value);
        if tty != Ty::IoToken {
            return Err(IrError::TypeMismatch {
                expected: Ty::IoToken,
                found: tty,
                loc,
            });
        }
        if !vty.is_printable() {
            return Err(IrError::TypeMismatch {
                expected: Ty::Str,
                found: vty,
                loc,
            });
        }
        // The (IoToken, Str|N|Bool) pair — Str allowed here only (I9s).
        let pair_ty = Ty::Tuple(vec![Ty::IoToken, vty.clone()]);
        self.b.intake_ty(&pair_ty)?;
        let pair = self
            .b
            .mint_object(self.f, pair_ty, None, ObjectKind::Temporary, None, loc);
        self.b.add_edge(
            self.f,
            token,
            pair,
            Operation::Pair { slot: 0, arity: 2 },
            loc,
        );
        self.b.add_edge(
            self.f,
            value,
            pair,
            Operation::Pair { slot: 1, arity: 2 },
            loc,
        );
        // Result is a fresh IoToken.
        let out = self
            .b
            .mint_object(self.f, Ty::IoToken, None, ObjectKind::Temporary, None, loc);
        self.b
            .add_edge(self.f, pair, out, Operation::Print { newline }, loc);
        Ok(out)
    }

    /// `time` (plan-time-builtin): `IoToken → (IoToken × f64)`. The source is
    /// the bare token (no internal pair — unlike `print`, there is no value
    /// operand); the target is a fresh `(IoToken, f64)` pair the caller projs
    /// apart (token rebind, milliseconds value).
    pub fn time_ms(&mut self, token: ObjectId, loc: SourceLoc) -> Result<ObjectId, IrError> {
        self.check_obj(token)?;
        let tty = self.ty_of(token);
        if tty != Ty::IoToken {
            return Err(IrError::TypeMismatch {
                expected: Ty::IoToken,
                found: tty,
                loc,
            });
        }
        let pair_ty = Ty::Tuple(vec![Ty::IoToken, Ty::f64()]);
        self.b.intake_ty(&pair_ty)?;
        let out = self
            .b
            .mint_object(self.f, pair_ty, None, ObjectKind::Temporary, None, loc);
        self.b.add_edge(self.f, token, out, Operation::TimeMs, loc);
        Ok(out)
    }

    /// `read` (ADR-0045): `(IoToken × Str) → (IoToken × [T; n])`. Builds the
    /// internal `(IoToken, Str)` pair exactly as `print` does (the only
    /// product a `Str` may join, I9s); the target is a fresh
    /// `(IoToken, array)` pair the caller projs apart (token rebind, array
    /// value). `array` must be a one-dimensional array of a readable element
    /// type ([`crate::read::readable`]); `path` a `Str` constant.
    pub fn read(
        &mut self,
        token: ObjectId,
        path: ObjectId,
        array: &Ty,
        loc: SourceLoc,
    ) -> Result<ObjectId, IrError> {
        self.check_obj(token)?;
        self.check_obj(path)?;
        let tty = self.ty_of(token);
        if tty != Ty::IoToken {
            return Err(IrError::TypeMismatch {
                expected: Ty::IoToken,
                found: tty,
                loc,
            });
        }
        let pty = self.ty_of(path);
        if pty != Ty::Str {
            return Err(IrError::TypeMismatch {
                expected: Ty::Str,
                found: pty,
                loc,
            });
        }
        if !crate::read::readable(array) {
            return Err(IrError::InvalidRead);
        }
        let src_ty = Ty::Tuple(vec![Ty::IoToken, Ty::Str]);
        let pair_ty = Ty::Tuple(vec![Ty::IoToken, array.clone()]);
        self.b.intake_ty(&src_ty)?;
        self.b.intake_ty(&pair_ty)?;
        let src = self
            .b
            .mint_object(self.f, src_ty, None, ObjectKind::Temporary, None, loc);
        self.b.add_edge(
            self.f,
            token,
            src,
            Operation::Pair { slot: 0, arity: 2 },
            loc,
        );
        self.b.add_edge(
            self.f,
            path,
            src,
            Operation::Pair { slot: 1, arity: 2 },
            loc,
        );
        let out = self
            .b
            .mint_object(self.f, pair_ty, None, ObjectKind::Temporary, None, loc);
        self.b.add_edge(self.f, src, out, Operation::Read, loc);
        Ok(out)
    }

    /// Bare `x -> ret` / `x -> ret.k` for an EXISTING object (DESIGN §10, D6):
    /// `Output` (full) or a `Pair` slot edge.
    pub fn output(
        &mut self,
        value: ObjectId,
        slot: Option<u32>,
        loc: SourceLoc,
    ) -> Result<(), IrError> {
        self.check_obj(value)?;
        let vty = self.ty_of(value);
        let ret = self.output_id();
        let out_ty = self.output_ty();
        match slot {
            None => {
                if vty != out_ty {
                    return Err(IrError::RetTypeMismatch);
                }
                self.b.add_edge(self.f, value, ret, Operation::Output, loc);
            }
            Some(k) => {
                if out_ty.product_arity().is_none() {
                    return Err(IrError::RetNotProduct);
                }
                // u64 arity capped to u32: an array of size > u32::MAX is not
                // slot-addressable; the truncated arity would let an OOB slot
                // pass `k < arity` (reviewer F4/SND-3). Reject instead.
                let arity = out_ty.product_arity_u32().ok_or(IrError::SlotOutOfRange)?;
                if k >= arity {
                    return Err(IrError::SlotOutOfRange);
                }
                let comp_ty = out_ty.component_ty(k).unwrap().clone();
                if vty != comp_ty {
                    return Err(IrError::RetTypeMismatch);
                }
                self.b
                    .add_edge(self.f, value, ret, Operation::Pair { slot: k, arity }, loc);
            }
        }
        Ok(())
    }

    /// Begin a loop (DESIGN §10): mints the `LoopMerge` (ty = `init.ty`) and the
    /// `LoopEnter` edge from `init`.
    pub fn begin_loop(&mut self, init: ObjectId, loc: SourceLoc) -> Result<LoopHandle, IrError> {
        self.check_obj(init)?;
        let u_ty = self.ty_of(init);
        self.b.intake_ty(&u_ty)?;
        let merge =
            self.b
                .mint_object(self.f, u_ty.clone(), None, ObjectKind::LoopMerge, None, loc);
        self.b
            .add_edge(self.f, init, merge, Operation::LoopEnter, loc);
        let fc = &mut self.b.funcs[self.f];
        fc.open_loops += 1;
        fc.loop_counts.push((merge, 0, 0));
        Ok(LoopHandle { merge, u_ty })
    }

    /// Bump the `(back, exit)` tally for an open loop's merge.
    pub(super) fn bump_loop(&mut self, merge: ObjectId, back: bool) {
        let fc = &mut self.b.funcs[self.f];
        if let Some(entry) = fc.loop_counts.iter_mut().find(|(m, _, _)| *m == merge) {
            if back {
                entry.1 += 1;
            } else {
                entry.2 += 1;
            }
        }
    }

    /// Read the `(back, exit)` tally for a merge.
    pub(super) fn loop_tally(&self, merge: ObjectId) -> (u32, u32) {
        self.b.funcs[self.f]
            .loop_counts
            .iter()
            .find(|(m, _, _)| *m == merge)
            .map(|(_, b, e)| (*b, *e))
            .unwrap_or((0, 0))
    }

    /// The `LoopMerge` object of an open loop.
    pub fn merge_of(&self, lh: &LoopHandle) -> ObjectId {
        lh.merge
    }

    /// Record a back edge (DESIGN §10): packs `(next_state, cond)` into a route
    /// object, then `LoopBack` → merge. `next_state.ty == U`, `cond: Bool`.
    pub fn loop_back(
        &mut self,
        lh: &LoopHandle,
        next_state: ObjectId,
        cond: ObjectId,
        loc: SourceLoc,
    ) -> Result<(), IrError> {
        self.check_obj(next_state)?;
        self.check_obj(cond)?;
        let sty = self.ty_of(next_state);
        let cty = self.ty_of(cond);
        if sty != lh.u_ty {
            return Err(IrError::TypeMismatch {
                expected: lh.u_ty.clone(),
                found: sty,
                loc,
            });
        }
        if cty != Ty::Bool {
            return Err(IrError::TypeMismatch {
                expected: Ty::Bool,
                found: cty,
                loc,
            });
        }
        let route_ty = Ty::Tuple(vec![lh.u_ty.clone(), Ty::Bool]);
        self.b.intake_ty(&route_ty)?;
        let route = self
            .b
            .mint_object(self.f, route_ty, None, ObjectKind::Temporary, None, loc);
        self.b.add_edge(
            self.f,
            next_state,
            route,
            Operation::Pair { slot: 0, arity: 2 },
            loc,
        );
        self.b.add_edge(
            self.f,
            cond,
            route,
            Operation::Pair { slot: 1, arity: 2 },
            loc,
        );
        self.b
            .add_edge(self.f, route, lh.merge, Operation::LoopBack, loc);
        self.bump_loop(lh.merge, true);
        Ok(())
    }

    /// Record an exit edge (DESIGN §10): packs `(value, cond)` into a route
    /// object, then `LoopExit` → a fresh exit object (or Ret per `dest`).
    pub fn loop_exit(
        &mut self,
        lh: &LoopHandle,
        value: ObjectId,
        cond: ObjectId,
        dest: Dest,
        loc: SourceLoc,
    ) -> Result<ObjectId, IrError> {
        self.check_obj(value)?;
        self.check_obj(cond)?;
        let vty = self.ty_of(value);
        let cty = self.ty_of(cond);
        if cty != Ty::Bool {
            return Err(IrError::TypeMismatch {
                expected: Ty::Bool,
                found: cty,
                loc,
            });
        }
        let route_ty = Ty::Tuple(vec![vty.clone(), Ty::Bool]);
        self.b.intake_ty(&route_ty)?;
        let route = self
            .b
            .mint_object(self.f, route_ty, None, ObjectKind::Temporary, None, loc);
        self.b.add_edge(
            self.f,
            value,
            route,
            Operation::Pair { slot: 0, arity: 2 },
            loc,
        );
        self.b.add_edge(
            self.f,
            cond,
            route,
            Operation::Pair { slot: 1, arity: 2 },
            loc,
        );
        self.b.intake_ty(&vty)?;
        let plan = self.dest_target(&dest, &vty, loc)?;
        let out = self.emit_to_dest(plan, route, Operation::LoopExit, loc);
        self.bump_loop(lh.merge, false);
        Ok(out)
    }

    /// Close a loop (DESIGN §10): requires ≥1 back + ≥1 exit; consumes handle.
    pub fn end_loop(&mut self, lh: LoopHandle) -> Result<(), IrError> {
        let (backs, exits) = self.loop_tally(lh.merge);
        if backs == 0 {
            return Err(IrError::LoopWithoutBack);
        }
        if exits == 0 {
            return Err(IrError::LoopWithoutExit);
        }
        self.b.funcs[self.f].open_loops -= 1;
        Ok(())
    }

    /// Finish the function (DESIGN §10): local I-RET completeness + no open
    /// loops. The seal pass re-checks everything globally.
    pub fn finish(self) -> Result<(), IrError> {
        let fc = &self.b.funcs[self.f];
        if fc.open_loops > 0 {
            return Err(IrError::OpenLoop);
        }
        // Local I-RET pre-check (seal re-derives, but report early).
        let ret = fc.def.output;
        let out_ty = self.b.objects[ret].ty.clone();
        let mut full = 0usize;
        let mut slots: Vec<u32> = Vec::new();
        for &m in &self.b.in_edges[ret] {
            match self.b.morphisms[m].op {
                Operation::Pair { slot, .. } => slots.push(slot),
                _ => full += 1,
            }
        }
        if full > 0 && !slots.is_empty() {
            return Err(IrError::RetMixedWriters);
        }
        if !slots.is_empty() {
            let arity = out_ty.product_arity().ok_or(IrError::RetNotProduct)?;
            slots.sort_unstable();
            for w in slots.windows(2) {
                if w[0] == w[1] {
                    return Err(IrError::RetSlotConflict);
                }
            }
            // Completeness: every slot `0..arity` must have a distinct writer.
            // `slots` is `u32`-valued and now deduplicated; a product whose arity
            // exceeds the writer count (incl. any > u32::MAX-arity array, which
            // is not slot-addressable anyway) is missing slots. Comparing counts
            // avoids iterating a u64 range (F4/SND-3).
            if (slots.len() as u64) != arity {
                return Err(IrError::RetSlotMissing);
            }
            for (k, &slot) in slots.iter().enumerate() {
                if slot as u64 != k as u64 {
                    return Err(IrError::RetSlotMissing);
                }
            }
        } else if full == 0 && out_ty != Ty::Unit {
            return Err(IrError::RetSlotMissing);
        }
        self.b.funcs[self.f].finished = true;
        Ok(())
    }
}
