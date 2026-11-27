//! The walker's item pass: declarations of functions, types, fields and globals.

use super::*;

impl<'a> Walker<'a> {
    // --- items --------------------------------------------------------------

    pub(super) fn program(&mut self, p: &Program) {
        // Pass A: the program-wide names (order-independent, ADR-0048 D2).
        for item in &p.items {
            match item {
                Item::Fn(fd) => self.declare_fn(fd),
                Item::Type(td) => self.declare_type(td),
                _ => {}
            }
        }
        for item in &p.items {
            if let Item::Bind(tb) = item {
                self.declare_globals(&tb.chain, tb.span);
            }
        }
        for item in &p.items {
            if let Item::Type(td) = item {
                self.declare_fields(td);
            }
        }
        // The signatures' types, from their annotations: a call reads its
        // callee's result type wherever the two are declared.
        for item in &p.items {
            if let Item::Fn(fd) = item {
                self.declare_signature(fd);
            }
        }
        // The top-level values' types: a chain may read a value a later chain
        // binds (`cfg.scale` above `-> cfg;`), so walk them for their types
        // alone until no type changes (each round settles one more link of a
        // dependency chain; lowering rejects a cycle, L1807).
        let tops = p
            .items
            .iter()
            .filter(|i| matches!(i, Item::Bind(_)))
            .count();
        for _ in 0..tops.min(16) {
            let before: Vec<Ty> = self
                .globals
                .values()
                .map(|&g| self.defs[g].ty.clone())
                .collect();
            self.quietly(|w| w.top_chains(p));
            let after: Vec<Ty> = self
                .globals
                .values()
                .map(|&g| self.defs[g].ty.clone())
                .collect();
            if before == after {
                break;
            }
        }
        // Pass B: bodies. Top-level chains first so globals carry their types
        // into the functions that read them.
        self.top_chains(p);
        for item in &p.items {
            if let Item::Fn(fd) = item {
                self.function(fd);
            }
        }
    }

    pub(super) fn declare_fn(&mut self, fd: &FnDecl) {
        let name = self.name(fd.name).to_string();
        if name.is_empty() || self.fns.contains_key(&name) {
            return;
        }
        let id = self.def(Def {
            name: name.clone(),
            kind: DefKind::Function,
            name_span: fd.name.span,
            span: fd.span,
            ty: Ty::Unknown,
            mutable: false,
            container: None,
            visible: self.whole(),
        });
        self.occ(fd.name.span, id, Role::Decl);
        self.fns.insert(name, id);
    }

    /// A function's type is its declared result (`-> T`, else `()`).
    pub(super) fn declare_signature(&mut self, fd: &FnDecl) {
        let Some(&fid) = self.fns.get(self.name(fd.name)) else {
            return;
        };
        if self.defs[fid].span != fd.span {
            return; // a duplicate declaration: the first one owns the name
        }
        let ret = self.quietly(|w| {
            fd.ret_ty
                .as_ref()
                .map(|t| w.ty(t))
                .unwrap_or(Ty::Tuple(Vec::new()))
        });
        self.defs[fid].ty = ret;
    }

    /// Walk every top-level chain (ADR-0048 D1: their bindings are globals).
    pub(super) fn top_chains(&mut self, p: &Program) {
        for item in &p.items {
            if let Item::Bind(tb) = item {
                self.top = true;
                self.frames.push(Frame {
                    names: BTreeMap::new(),
                    kind: FrameKind::Root,
                    end: tb.span.end,
                });
                self.chain(&tb.chain);
                self.frames.clear();
                self.top = false;
            }
        }
    }

    /// Run `f` for the types it settles alone: the definitions, occurrences
    /// and names it records are dropped (the real walk records them once).
    pub(super) fn quietly<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> R {
        let (defs, occs, lexical, puns) = (
            self.defs.len(),
            self.occs.len(),
            self.lexical.len(),
            self.puns.len(),
        );
        let r = f(self);
        self.defs.truncate(defs);
        self.occs.truncate(occs);
        self.lexical.truncate(lexical);
        self.puns.truncate(puns);
        r
    }

    pub(super) fn declare_type(&mut self, td: &TypeDecl) {
        let name = self.name(td.name).to_string();
        if name.is_empty() || self.types.contains_key(&name) {
            return;
        }
        let id = self.def(Def {
            name: name.clone(),
            kind: DefKind::Type,
            name_span: td.name.span,
            span: td.span,
            ty: Ty::Named {
                name: name.clone(),
                args: Vec::new(),
            },
            mutable: false,
            container: None,
            visible: self.whole(),
        });
        self.occ(td.name.span, id, Role::Decl);
        self.types.insert(name, id);
    }

    pub(super) fn declare_fields(&mut self, td: &TypeDecl) {
        let tname = self.name(td.name).to_string();
        let Some(&tid) = self.types.get(&tname) else {
            return;
        };
        if self.defs[tid].span != td.span {
            return; // a duplicate declaration: the first one owns the name
        }
        self.sizes = self.size_params(&td.generics, tid, td.span);
        let mut fields = BTreeMap::new();
        for f in &td.fields {
            let ty = self.ty(&f.ty);
            let fname = self.name(f.name).to_string();
            let id = self.def(Def {
                name: fname.clone(),
                kind: DefKind::Field,
                name_span: f.name.span,
                span: f.span,
                ty,
                mutable: false,
                container: Some(tid),
                visible: td.span,
            });
            self.occ(f.name.span, id, Role::Decl);
            fields.entry(fname).or_insert(id);
        }
        self.fields.insert(tname, fields);
        self.sizes.clear();
    }

    /// The declared generic list: a bare name is a size parameter, `name:
    /// kind` a format parameter (ADR-0051 D1; the kind reads as a primitive).
    pub(super) fn size_params(
        &mut self,
        generics: &[GenericParam],
        owner: DefId,
        scope: SourceLoc,
    ) -> BTreeMap<String, DefId> {
        let mut out = BTreeMap::new();
        for g in generics {
            let text = self.name(g.name).to_string();
            let (kind, ty) = match g.kind {
                None => (DefKind::SizeParam, Ty::Prim("i32".into())),
                Some(k) => {
                    self.lex(k.span, Class::Primitive);
                    (DefKind::FormatParam, Ty::Prim(self.name(k).to_string()))
                }
            };
            let id = self.def(Def {
                name: text.clone(),
                kind,
                name_span: g.name.span,
                span: g.span,
                ty,
                mutable: false,
                container: Some(owner),
                visible: scope,
            });
            self.occ(g.name.span, id, Role::Decl);
            out.insert(text, id);
        }
        out
    }

    /// ADR-0048 D1: the names a top-level chain binds.
    pub(super) fn declare_globals(&mut self, chain: &Chain, span: SourceLoc) {
        for st in &chain.stages {
            let (name, mutable) = match &st.kind {
                StageKind::Bind { mut_span, name, .. } => (*name, mut_span.is_some()),
                StageKind::Expr(Expr {
                    kind: ExprKind::Var(n),
                    ..
                }) if self.binds_bare(*n) => (*n, false),
                _ => continue,
            };
            let text = self.name(name).to_string();
            if let Some(&prev) = self.globals.get(&text) {
                self.global_sites.insert(name.span.start, prev);
                continue;
            }
            let id = self.def(Def {
                name: text.clone(),
                kind: DefKind::Global,
                name_span: name.span,
                span,
                ty: Ty::Unknown,
                mutable,
                container: None,
                visible: self.whole(),
            });
            self.globals.insert(text, id);
            self.global_sites.insert(name.span.start, id);
        }
    }

    pub(super) fn binds_bare(&self, n: Name) -> bool {
        let t = self.name(n);
        !self.fns.contains_key(t) && !is_builtin_name(t) && !is_scalar(t)
    }

    pub(super) fn function(&mut self, fd: &FnDecl) {
        let fname = self.name(fd.name);
        let Some(&fid) = self.fns.get(fname) else {
            return;
        };
        self.sizes = self.size_params(&fd.generics, fid, fd.span);
        self.frames.push(Frame {
            names: BTreeMap::new(),
            kind: FrameKind::Root,
            end: fd.span.end,
        });
        for p in &fd.params {
            let ty = self.ty(&p.ty);
            let text = self.name(p.name).to_string();
            let id = self.def(Def {
                name: text.clone(),
                kind: DefKind::Param,
                name_span: p.name.span,
                span: p.span,
                ty,
                mutable: p.mut_span.is_some(),
                container: Some(fid),
                visible: SourceLoc::new(p.name.span.start, fd.span.end),
            });
            self.occ(p.name.span, id, Role::Decl);
            self.frames[0].names.insert(text, id);
        }
        // The result type is the signature's (`declare_signature`); walk the
        // annotation for its names' occurrences.
        if let Some(t) = &fd.ret_ty {
            self.ty(t);
        }
        self.compute(&fd.body);
        self.block_items(&fd.body);
        self.frames.clear();
        self.sizes.clear();
    }
}
