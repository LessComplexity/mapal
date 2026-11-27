//! The walker's blocks, statements, chains and stages.

use super::*;

impl<'a> Walker<'a> {
    // --- blocks, statements, chains ----------------------------------------

    pub(super) fn block_items(&mut self, b: &Block) -> Ty {
        self.block_items_from(b, &Ty::Unknown)
    }

    /// A block's items; a headless chain (no head expression) starts from
    /// `seed` (a `seq`'s input, ADR-0019 pin a).
    pub(super) fn block_items_from(&mut self, b: &Block, seed: &Ty) -> Ty {
        for item in &b.items {
            match item {
                BlockItem::Stmt(s) => match &s.kind {
                    StmtKind::Chain(c) => {
                        self.chain_from(c, seed);
                    }
                    StmtKind::Bind(bs) => {
                        if let Some(ix) = &bs.index {
                            self.expr(ix);
                            self.expr(&bs.value);
                            let text = self.name(bs.name);
                            if let Some((id, _)) = self.lookup_local(text) {
                                self.occ(bs.name.span, id, Role::Write);
                            }
                        } else {
                            let v = self.expr(&bs.value);
                            let ty = match &bs.ty {
                                Some(t) => self.ty(t),
                                None => v,
                            };
                            self.bind(
                                bs.name,
                                DefKind::Local,
                                bs.mut_span.is_some(),
                                ty,
                                s.span,
                                None,
                            );
                        }
                    }
                    StmtKind::Loop(l) => {
                        self.compute(&l.body);
                        self.push(FrameKind::Loop, l.body.span.end);
                        self.block_items(&l.body);
                        self.pop();
                    }
                    StmtKind::Error => {}
                },
                BlockItem::Arm(arm) => {
                    self.arm(arm, Ty::Unknown, false);
                }
            }
        }
        match &b.tail {
            Some(t) => self.chain_from(t, seed),
            None => Ty::Unknown,
        }
    }

    /// `seq { … }` (ADR-0019): the block lowers in the enclosing scope (pin b),
    /// so its bindings stay visible after it; headless statements and the
    /// tail start from the seq's input (pin a).
    pub(super) fn seq(&mut self, b: &Block, input: &Ty) -> Ty {
        self.compute(b);
        let top = std::mem::replace(&mut self.top, false);
        let t = self.block_items_from(b, input);
        self.top = top;
        t
    }

    /// A nested block in its own frame.
    pub(super) fn block(&mut self, b: &Block, kind: FrameKind) -> Ty {
        self.compute(b);
        let top = std::mem::replace(&mut self.top, false);
        self.push(kind, b.span.end);
        let t = self.block_items(b);
        self.pop();
        self.top = top;
        t
    }

    pub(super) fn chain(&mut self, c: &Chain) -> Ty {
        self.chain_from(c, &Ty::Unknown)
    }

    /// A chain; with no head it starts from `seed`.
    pub(super) fn chain_from(&mut self, c: &Chain, seed: &Ty) -> Ty {
        let mut wire = Wire {
            ty: match &c.head {
                Some(h) => self.expr(h),
                None => seed.clone(),
            },
            head: c.head.as_ref(),
        };
        let exit = self.exit_chain == Some(c as *const Chain);
        let n = c.stages.len();
        for (i, st) in c.stages.iter().enumerate() {
            let exit_bind = exit && i + 1 == n;
            let ty = self.stage(st, &wire, exit_bind, c.span);
            wire = Wire { ty, head: None };
        }
        wire.ty
    }

    pub(super) fn stage(
        &mut self,
        st: &Stage,
        wire: &Wire,
        exit_bind: bool,
        chain_span: SourceLoc,
    ) -> Ty {
        let target = if exit_bind {
            self.loop_exit_frame()
        } else {
            None
        };
        match &st.kind {
            StageKind::Expr(e) => match &e.kind {
                ExprKind::Var(n) => self.bare_stage(*n, wire, target, chain_span),
                ExprKind::Generic { name, args } => self.generic_stage(*name, args, wire),
                _ => self.expr(e),
            },
            StageKind::Bind { mut_span, name, ty } => {
                let t = match ty {
                    Some(t) => self.ty(t),
                    None => wire.ty.clone(),
                };
                if self.top {
                    self.global_bind(*name, &t);
                } else {
                    self.bind(
                        *name,
                        DefKind::Local,
                        mut_span.is_some(),
                        t.clone(),
                        chain_span,
                        target,
                    );
                }
                t
            }
            StageKind::Ret { .. } | StageKind::LoopJump | StageKind::Error(_) => Ty::Unknown,
            StageKind::OpShorthand { expr } => {
                let t = self.expr(expr);
                if wire.ty.is_known() {
                    wire.ty.clone()
                } else {
                    t
                }
            }
            StageKind::Guard(arms) => {
                let routing = self.frames.iter().any(|f| f.kind == FrameKind::Loop)
                    && arms.iter().any(|a| arm_routes(&a.payload));
                let mut out = Ty::Unknown;
                for a in arms {
                    let t = self.arm(a, wire.ty.clone(), routing);
                    if !out.is_known() {
                        out = t;
                    }
                }
                out
            }
            StageKind::Fanout { branches, .. } => {
                let mut ts = Vec::new();
                // The branches join: their bindings are visible after the
                // fanout (`6 -> { -> square -> sq; -> double -> db; }`).
                for b in branches {
                    ts.push(self.chain(b));
                }
                Ty::Tuple(ts)
            }
            StageKind::MapFold { op, params, body } => self.map_fold(*op, params, body, wire),
            StageKind::SeqBlock(b) => self.seq(b, &wire.ty),
            StageKind::StmtBlock(b) => self.block(b, FrameKind::Block),
        }
    }

    pub(super) fn global_bind(&mut self, name: Name, ty: &Ty) {
        if let Some(&id) = self.global_sites.get(&name.span.start) {
            let first = self.defs[id].name_span == name.span;
            self.occ(name.span, id, if first { Role::Decl } else { Role::Write });
            if first && ty.is_known() {
                self.defs[id].ty = ty.clone();
            }
        }
    }

    /// `-> name` with no annotation: rebind, call, builtin or fresh binding.
    pub(super) fn bare_stage(
        &mut self,
        n: Name,
        wire: &Wire,
        target: Option<usize>,
        chain_span: SourceLoc,
    ) -> Ty {
        let text = self.name(n);
        if let Some((id, frame)) = self.lookup_local(text) {
            let mutable = self.defs[id].mutable;
            if !mutable && self.is_capture(frame) {
                // ADR-0027: a fresh body-local shadow of a read capture.
                return {
                    self.bind(
                        n,
                        DefKind::Local,
                        false,
                        wire.ty.clone(),
                        chain_span,
                        target,
                    );
                    wire.ty.clone()
                };
            }
            self.occ(n.span, id, Role::Write);
            return if wire.ty.is_known() {
                wire.ty.clone()
            } else {
                self.defs[id].ty.clone()
            };
        }
        if let Some(&id) = self.fns.get(text) {
            self.occ(n.span, id, Role::Call);
            return self.defs[id].ty.clone();
        }
        if is_builtin_name(text) {
            self.lex(n.span, Class::Builtin);
            return builtin_result(text, None, wire, self.src);
        }
        if self.top {
            self.global_bind(n, &wire.ty);
        } else {
            self.bind(
                n,
                DefKind::Local,
                false,
                wire.ty.clone(),
                chain_span,
                target,
            );
        }
        wire.ty.clone()
    }

    /// ADR-0051 D2: a stage with explicit generic arguments, `x -> to<bf16>`,
    /// `(w, x) -> project<4096, 2048>` — a call of a function or a builtin.
    pub(super) fn generic_stage(&mut self, n: Name, args: &[SizeExpr], wire: &Wire) -> Ty {
        for a in args {
            self.size_expr(a);
        }
        let text = self.name(n);
        if let Some(&id) = self.fns.get(text) {
            self.occ(n.span, id, Role::Call);
            return self.defs[id].ty.clone();
        }
        if is_builtin_name(text) {
            self.lex(n.span, Class::Builtin);
            let first = args.first().map(|a| self.text(a.span));
            return builtin_result(text, first, wire, self.src);
        }
        Ty::Unknown
    }

    pub(super) fn arm(&mut self, arm: &GuardArm, _input: Ty, routing: bool) -> Ty {
        let exit = routing && !arm_routes(&arm.payload);
        let saved = self.exit_chain;
        if exit {
            self.exit_chain = final_chain(&arm.payload).map(|c| c as *const Chain);
        }
        let t = match &arm.payload {
            ArmPayload::Chain(c) => {
                let top = std::mem::replace(&mut self.top, false);
                self.push(FrameKind::Block, arm.span.end);
                let t = self.chain(c);
                self.pop();
                self.top = top;
                t
            }
            ArmPayload::Block(b) => self.block(b, FrameKind::Block),
        };
        self.exit_chain = saved;
        t
    }

    pub(super) fn map_fold(
        &mut self,
        op: CollOp,
        params: &[Name],
        body: &Block,
        wire: &Wire,
    ) -> Ty {
        // The parameters' types: a map's element; a fold's accumulator (the
        // seed) and element.
        let (seed, elem) = match (op, &wire.ty) {
            (CollOp::Fold, Ty::Tuple(ts)) if ts.len() == 2 => {
                (Some(ts[0].clone()), elem_of(&ts[1]))
            }
            (CollOp::Fold, _) => (None, Ty::Unknown),
            (CollOp::Map, t) => (None, elem_of(t)),
        };
        let mut tys: Vec<Ty> = Vec::new();
        if let Some(s) = &seed {
            tys.push(s.clone());
        }
        let rest = params.len().saturating_sub(tys.len());
        match &elem {
            Ty::Tuple(es) if rest > 1 && es.len() == rest => tys.extend(es.iter().cloned()),
            e => tys.push(e.clone()),
        }
        self.compute(body);
        let top = std::mem::replace(&mut self.top, false);
        self.push(FrameKind::Body, body.span.end);
        let frame = self.frames.len() - 1;
        for (i, p) in params.iter().enumerate() {
            let t = tys.get(i).cloned().unwrap_or(Ty::Unknown);
            let id = self.bind(*p, DefKind::BodyParam, false, t, p.span, Some(frame));
            if id != usize::MAX {
                self.defs[id].visible = SourceLoc::new(p.span.start, body.span.end);
            }
        }
        let bt = self.block_items(body);
        self.pop();
        self.top = top;
        match op {
            CollOp::Map => Ty::Array {
                elem: Box::new(bt),
                len: match &wire.ty {
                    Ty::Array { len, .. } => len.clone(),
                    _ => "n".into(),
                },
            },
            CollOp::Fold => match seed {
                Some(s) if s.is_known() => s,
                _ => bt,
            },
        }
    }
}
