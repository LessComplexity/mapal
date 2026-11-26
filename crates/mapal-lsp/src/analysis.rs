//! Name resolution over the parse tree: every definition, every occurrence of
//! a name, and a light structural type for each value — the facts behind
//! go-to-definition, references, rename, hover, completion and semantic tokens.
//!
//! The walk mirrors lowering's scoping rules (`mapal-lower` `scope`, `emit`,
//! `globals`, `typing`) instead of inventing its own:
//!
//! - A function body is one root frame holding the parameters; `loop`, guard
//!   arms and block stages push child frames (a fanout's branches do not: they
//!   join; nor does a `seq`: it lowers in the enclosing scope, ADR-0019 pin b);
//!   a `map` / `fold` body pushes a frame that is a capture boundary (ADR-0027).
//! - Every function's signature type is known before any body is walked, and
//!   every top-level value's type before any chain reads it (ADR-0048 D2:
//!   program-wide names are order-independent), so a call to a function
//!   declared further down, or a top-level value bound further down, still
//!   carries its type into field access, hover and completion.
//! - A bare-name stage `-> name` resolves innermost-first: a `mut` local is a
//!   rebind of that local; an immutable capture is a fresh shadow in the body;
//!   then a function is a call, a builtin is the builtin, and anything else is
//!   a fresh binding (LD1). A typed stage `-> name: T` always binds.
//! - A statement `mut x: T <- e` / `x <- e` always binds; `c[i] <- v` updates
//!   (rebinds) the existing `c` (ADR-0021).
//! - A routing guard's exit arm binds its final `-> name` into the scope that
//!   encloses the loop (DESIGN §8.5), so the exit value is visible after it.
//! - A name read resolves locals first, then the function's size parameters
//!   (ADR-0047), then top-level bindings (ADR-0048), then functions.
//! - Top-level chains bind program-wide names (ADR-0048 D1): a bare name that
//!   is no function and no builtin binds, so does a typed stage.
//!
//! The tree may come from a file mid-edit: error nodes are skipped and the
//! rest still resolves (the parser always returns a whole tree).

use std::collections::BTreeMap;

use mapal_syntax::{
    ArmPayload, Block, BlockItem, Chain, CollOp, Diagnostic, Expr, ExprKind, FnDecl, GuardArm,
    Item, MemberField, Name, Program, SizeExpr, SizeKind, SourceLoc, Stage, StageKind, StmtKind,
    TokenKind, TyKind, TypeDecl,
};

pub type DefId = usize;

/// What a name denotes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DefKind {
    Function,
    Type,
    /// A field of a `type` (`_` included, ADR-0048 D5).
    Field,
    /// A top-level binding (ADR-0048 D1).
    Global,
    /// A local binding (`-> name`, `x <- e`).
    Local,
    /// A function parameter.
    Param,
    /// A `map` / `fold` body parameter.
    BodyParam,
    /// A size parameter of a function or a type (ADR-0047 / ADR-0048 D4).
    SizeParam,
}

impl DefKind {
    pub fn describe(self) -> &'static str {
        match self {
            DefKind::Function => "function",
            DefKind::Type => "type",
            DefKind::Field => "field",
            DefKind::Global => "top-level value",
            DefKind::Local => "local",
            DefKind::Param => "parameter",
            DefKind::BodyParam => "body parameter",
            DefKind::SizeParam => "size parameter",
        }
    }
}

/// A structural type, as far as the source states or implies it. Sizes are
/// kept as the source wrote them (`tokens * hidden_size`).
#[derive(Clone, PartialEq, Debug)]
pub enum Ty {
    Prim(String),
    Named { name: String, args: Vec<String> },
    Array { elem: Box<Ty>, len: String },
    Tuple(Vec<Ty>),
    Unknown,
}

impl Ty {
    pub fn render(&self) -> String {
        match self {
            Ty::Prim(p) => p.clone(),
            Ty::Named { name, args } if args.is_empty() => name.clone(),
            Ty::Named { name, args } => format!("{name}<{}>", args.join(", ")),
            Ty::Array { elem, len } => format!("[{}; {len}]", elem.render()),
            Ty::Tuple(ts) if ts.is_empty() => "()".into(),
            Ty::Tuple(ts) => format!(
                "({})",
                ts.iter().map(Ty::render).collect::<Vec<_>>().join(", ")
            ),
            Ty::Unknown => "_".into(),
        }
    }

    pub fn is_known(&self) -> bool {
        !matches!(self, Ty::Unknown)
    }

    fn named(&self) -> Option<&str> {
        match self {
            Ty::Named { name, .. } => Some(name),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Def {
    pub name: String,
    pub kind: DefKind,
    /// The name at its definition (the selection range).
    pub name_span: SourceLoc,
    /// The whole defining construct (a function, a statement, a field).
    pub span: SourceLoc,
    pub ty: Ty,
    pub mutable: bool,
    /// A field's type, a parameter's or size parameter's function or type.
    pub container: Option<DefId>,
    /// Where the name can be written and mean this definition.
    pub visible: SourceLoc,
}

/// How an occurrence uses its definition.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role {
    Decl,
    Read,
    /// A rebind or element update of a `mut` binding.
    Write,
    Call,
    TypeUse,
}

#[derive(Clone, Copy, Debug)]
pub struct Occ {
    pub span: SourceLoc,
    pub def: DefId,
    pub role: Role,
}

/// The highlight class of a name, the vocabulary `editors/test/classes.tsv`
/// maps to Vim groups, TextMate scopes and LSP semantic tokens.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Class {
    FunctionDeclaration,
    Function,
    Builtin,
    TypeDeclaration,
    Type,
    Primitive,
    SizeDeclaration,
    Size,
    Global,
    Binding,
    Rebind,
    ParameterDeclaration,
    Parameter,
    FieldDeclaration,
    Field,
    Variable,
}

impl Class {
    pub fn name(self) -> &'static str {
        match self {
            Class::FunctionDeclaration => "function-declaration",
            Class::Function => "function",
            Class::Builtin => "builtin",
            Class::TypeDeclaration => "type-declaration",
            Class::Type => "type",
            Class::Primitive => "primitive",
            Class::SizeDeclaration => "size-declaration",
            Class::Size => "size",
            Class::Global => "global",
            Class::Binding => "binding",
            Class::Rebind => "rebind",
            Class::ParameterDeclaration => "parameter-declaration",
            Class::Parameter => "parameter",
            Class::FieldDeclaration => "field-declaration",
            Class::Field => "field",
            Class::Variable => "variable",
        }
    }
}

/// One file's analysis. Pure: a function of the text alone (ADR-0008 (d)).
pub struct Analysis {
    pub text: String,
    pub program: Program,
    pub parse_diagnostics: Vec<Diagnostic>,
    pub defs: Vec<Def>,
    pub occs: Vec<Occ>,
    /// Builtin and primitive-type names (no definition in the file).
    pub lexical: Vec<(SourceLoc, Class)>,
    /// Struct-literal puns `T { name }`: one name, a field and a value.
    pub puns: Vec<SourceLoc>,
}

impl Analysis {
    pub fn new(text: &str) -> Analysis {
        let po = mapal_syntax::parse(text);
        let tokens = mapal_syntax::lex(text).tokens;
        let mut w = Walker::new(text, &tokens);
        w.program(&po.program);
        let Walker {
            defs,
            occs,
            lexical,
            puns,
            ..
        } = w;
        let mut a = Analysis {
            text: text.to_string(),
            program: po.program,
            parse_diagnostics: po.diagnostics,
            defs,
            occs,
            lexical,
            puns,
        };
        a.occs.sort_by_key(|o| (o.span.start, o.span.end));
        a.lexical.sort_by_key(|(s, _)| (s.start, s.end));
        a
    }

    pub fn slice(&self, span: SourceLoc) -> &str {
        self.text
            .get(span.start as usize..span.end as usize)
            .unwrap_or("")
    }

    /// The occurrence whose name contains `offset` (its end included, so a
    /// cursor just after a name still finds it).
    pub fn occ_at(&self, offset: u32) -> Option<&Occ> {
        let hit = |o: &&Occ| o.span.start <= offset && offset < o.span.end;
        let touch = |o: &&Occ| o.span.start <= offset && offset <= o.span.end;
        self.occs
            .iter()
            .find(hit)
            .or_else(|| self.occs.iter().find(touch))
    }

    /// A builtin or primitive name at `offset`.
    pub fn lexical_at(&self, offset: u32) -> Option<(SourceLoc, Class)> {
        self.lexical
            .iter()
            .find(|(s, _)| s.start <= offset && offset <= s.end)
            .copied()
    }

    pub fn def_at(&self, offset: u32) -> Option<DefId> {
        self.occ_at(offset).map(|o| o.def)
    }

    /// Every occurrence of `def`, its declaration included, in source order.
    pub fn occurrences(&self, def: DefId) -> impl Iterator<Item = &Occ> {
        self.occs.iter().filter(move |o| o.def == def)
    }

    /// The highlight class of an occurrence.
    pub fn class_of(&self, occ: &Occ) -> Option<Class> {
        let d = &self.defs[occ.def];
        let decl = occ.role == Role::Decl;
        Some(match d.kind {
            DefKind::Function if decl => Class::FunctionDeclaration,
            DefKind::Function => Class::Function,
            DefKind::Type if decl => Class::TypeDeclaration,
            DefKind::Type => Class::Type,
            // `_` reads as the default interpretation, a lexical class.
            DefKind::Field if d.name == "_" => return None,
            DefKind::Field if decl => Class::FieldDeclaration,
            DefKind::Field => Class::Field,
            DefKind::SizeParam if decl => Class::SizeDeclaration,
            DefKind::SizeParam => Class::Size,
            _ if decl && matches!(d.kind, DefKind::Param | DefKind::BodyParam) => {
                Class::ParameterDeclaration
            }
            _ if decl => Class::Binding,
            _ if occ.role == Role::Write => Class::Rebind,
            DefKind::Global => Class::Global,
            DefKind::Param | DefKind::BodyParam => Class::Parameter,
            _ => Class::Variable,
        })
    }

    /// Every classified name in source order: definitions' occurrences and the
    /// builtin / primitive names.
    pub fn classified(&self) -> Vec<(SourceLoc, Class)> {
        let mut out: Vec<(SourceLoc, Class)> = self
            .occs
            .iter()
            .filter_map(|o| self.class_of(o).map(|c| (o.span, c)))
            .chain(self.lexical.iter().copied())
            .collect();
        out.sort_by_key(|(s, _)| (s.start, s.end));
        out.dedup_by_key(|(s, _)| s.start);
        out
    }

    /// The definitions visible at `offset`, innermost (latest) first per name.
    pub fn visible_at(&self, offset: u32) -> Vec<DefId> {
        let mut by_name: BTreeMap<&str, DefId> = BTreeMap::new();
        for (id, d) in self.defs.iter().enumerate() {
            if d.kind == DefKind::Field {
                continue;
            }
            if !(d.visible.start <= offset && offset <= d.visible.end) {
                continue;
            }
            match by_name.get(d.name.as_str()) {
                Some(&prev) if self.defs[prev].visible.start > d.visible.start => {}
                _ => {
                    by_name.insert(&d.name, id);
                }
            }
        }
        by_name.into_values().collect()
    }

    /// The fields of a type by name (through nothing: the type's own).
    pub fn fields_of(&self, ty: DefId) -> Vec<DefId> {
        self.defs
            .iter()
            .enumerate()
            .filter(|(_, d)| d.kind == DefKind::Field && d.container == Some(ty))
            .map(|(i, _)| i)
            .collect()
    }

    pub fn type_named(&self, name: &str) -> Option<DefId> {
        self.defs
            .iter()
            .position(|d| d.kind == DefKind::Type && d.name == name)
    }

    /// The type definition a value's type names (through arrays' elements).
    pub fn type_def_of(&self, ty: &Ty) -> Option<DefId> {
        match ty {
            Ty::Named { name, .. } => self.type_named(name),
            Ty::Array { elem, .. } => self.type_def_of(elem),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// The walk.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum FrameKind {
    Root,
    Block,
    Loop,
    /// A `map` / `fold` body: names below it are captures.
    Body,
}

struct Frame {
    names: BTreeMap<String, DefId>,
    kind: FrameKind,
    end: u32,
}

struct Walker<'a> {
    src: &'a str,
    tokens: &'a [mapal_syntax::Token],
    defs: Vec<Def>,
    occs: Vec<Occ>,
    lexical: Vec<(SourceLoc, Class)>,
    puns: Vec<SourceLoc>,
    fns: BTreeMap<String, DefId>,
    types: BTreeMap<String, DefId>,
    fields: BTreeMap<String, BTreeMap<String, DefId>>,
    globals: BTreeMap<String, DefId>,
    /// Top-level binding stages by the name's start offset → the global.
    global_sites: BTreeMap<u32, DefId>,
    /// The enclosing function's or type's size parameters.
    sizes: BTreeMap<String, DefId>,
    frames: Vec<Frame>,
    /// Walking a top-level chain (its bindings are globals).
    top: bool,
    /// The final chain of a routing guard's exit arm being walked.
    exit_chain: Option<*const Chain>,
}

/// A stage's input: its type and, when it is the chain's head, the head.
#[derive(Clone)]
struct Wire<'e> {
    ty: Ty,
    head: Option<&'e Expr>,
}

impl<'a> Walker<'a> {
    fn new(src: &'a str, tokens: &'a [mapal_syntax::Token]) -> Self {
        Walker {
            src,
            tokens,
            defs: Vec::new(),
            occs: Vec::new(),
            lexical: Vec::new(),
            puns: Vec::new(),
            fns: BTreeMap::new(),
            types: BTreeMap::new(),
            fields: BTreeMap::new(),
            globals: BTreeMap::new(),
            global_sites: BTreeMap::new(),
            sizes: BTreeMap::new(),
            frames: Vec::new(),
            top: false,
            exit_chain: None,
        }
    }

    fn text(&self, span: SourceLoc) -> &'a str {
        self.src
            .get(span.start as usize..span.end as usize)
            .unwrap_or("")
    }

    fn name(&self, n: Name) -> &'a str {
        self.text(n.span)
    }

    fn whole(&self) -> SourceLoc {
        SourceLoc::new(0, self.src.len() as u32)
    }

    fn def(&mut self, d: Def) -> DefId {
        self.defs.push(d);
        self.defs.len() - 1
    }

    fn occ(&mut self, span: SourceLoc, def: DefId, role: Role) {
        if !span.is_empty() {
            self.occs.push(Occ { span, def, role });
        }
    }

    fn lex(&mut self, span: SourceLoc, class: Class) {
        if !span.is_empty() {
            self.lexical.push((span, class));
        }
    }

    // --- items --------------------------------------------------------------

    fn program(&mut self, p: &Program) {
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

    fn declare_fn(&mut self, fd: &FnDecl) {
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
    fn declare_signature(&mut self, fd: &FnDecl) {
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
    fn top_chains(&mut self, p: &Program) {
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
    fn quietly<R>(&mut self, f: impl FnOnce(&mut Self) -> R) -> R {
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

    fn declare_type(&mut self, td: &TypeDecl) {
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

    fn declare_fields(&mut self, td: &TypeDecl) {
        let tname = self.name(td.name).to_string();
        let Some(&tid) = self.types.get(&tname) else {
            return;
        };
        if self.defs[tid].span != td.span {
            return; // a duplicate declaration: the first one owns the name
        }
        self.sizes = self.size_params(&td.sizes, tid, td.span);
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

    fn size_params(
        &mut self,
        sizes: &[Name],
        owner: DefId,
        scope: SourceLoc,
    ) -> BTreeMap<String, DefId> {
        let mut out = BTreeMap::new();
        for s in sizes {
            let text = self.name(*s).to_string();
            let id = self.def(Def {
                name: text.clone(),
                kind: DefKind::SizeParam,
                name_span: s.span,
                span: s.span,
                ty: Ty::Prim("i32".into()),
                mutable: false,
                container: Some(owner),
                visible: scope,
            });
            self.occ(s.span, id, Role::Decl);
            out.insert(text, id);
        }
        out
    }

    /// ADR-0048 D1: the names a top-level chain binds.
    fn declare_globals(&mut self, chain: &Chain, span: SourceLoc) {
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

    fn binds_bare(&self, n: Name) -> bool {
        let t = self.name(n);
        !self.fns.contains_key(t) && !mapal_lower::is_builtin(t) && !is_scalar(t)
    }

    fn function(&mut self, fd: &FnDecl) {
        let fname = self.name(fd.name);
        let Some(&fid) = self.fns.get(fname) else {
            return;
        };
        self.sizes = self.size_params(&fd.sizes, fid, fd.span);
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

    // --- types --------------------------------------------------------------

    fn ty(&mut self, t: &mapal_syntax::Ty) -> Ty {
        match &t.kind {
            TyKind::Named(n) => self.type_name(*n, Vec::new()),
            TyKind::Generic { name, args } => {
                let args = args
                    .iter()
                    .map(|a| {
                        self.size_expr(a);
                        self.text(a.span).to_string()
                    })
                    .collect();
                self.type_name(*name, args)
            }
            TyKind::Tuple(ts) => Ty::Tuple(ts.iter().map(|t| self.ty(t)).collect()),
            TyKind::Array {
                elem,
                len_span,
                size,
                ..
            } => {
                let elem = Box::new(self.ty(elem));
                let len = match size {
                    Some(s) => {
                        self.size_expr(s);
                        self.text(s.span).to_string()
                    }
                    None => self.text(*len_span).to_string(),
                };
                Ty::Array { elem, len }
            }
            TyKind::Dynamic(elem) => Ty::Array {
                elem: Box::new(self.ty(elem)),
                len: "?".into(),
            },
            TyKind::Error => Ty::Unknown,
        }
    }

    fn type_name(&mut self, n: Name, args: Vec<String>) -> Ty {
        let text = self.name(n);
        if is_scalar(text) {
            self.lex(n.span, Class::Primitive);
            return Ty::Prim(text.to_string());
        }
        if let Some(&id) = self.types.get(text) {
            self.occ(n.span, id, Role::TypeUse);
        }
        Ty::Named {
            name: text.to_string(),
            args,
        }
    }

    fn size_expr(&mut self, s: &SizeExpr) {
        match &s.kind {
            SizeKind::Int(_) => {}
            SizeKind::Name(n) => {
                let text = self.name(*n);
                if let Some(&id) = self.sizes.get(text) {
                    self.occ(n.span, id, Role::Read);
                } else if let Some(&id) = self.globals.get(text) {
                    self.occ(n.span, id, Role::Read);
                }
            }
            SizeKind::Binary { lhs, rhs, .. } => {
                self.size_expr(lhs);
                self.size_expr(rhs);
            }
        }
    }

    /// ADR-0041 D2: the precision members of a body's `: compute`.
    fn compute(&mut self, b: &Block) {
        let Some(p) = b.compute else {
            return;
        };
        let spans: Vec<SourceLoc> = self
            .tokens
            .iter()
            .filter(|t| {
                t.kind == TokenKind::Ident
                    && t.span.start >= p.span.start
                    && t.span.end <= p.span.end
            })
            .map(|t| t.span)
            .collect();
        for s in spans {
            self.lex(s, Class::Primitive);
        }
    }

    // --- scopes -------------------------------------------------------------

    fn push(&mut self, kind: FrameKind, end: u32) {
        self.frames.push(Frame {
            names: BTreeMap::new(),
            kind,
            end,
        });
    }

    fn pop(&mut self) {
        self.frames.pop();
    }

    fn lookup_local(&self, name: &str) -> Option<(DefId, usize)> {
        self.frames
            .iter()
            .enumerate()
            .rev()
            .find_map(|(i, f)| f.names.get(name).map(|&id| (id, i)))
    }

    fn is_capture(&self, frame: usize) -> bool {
        self.frames[frame + 1..]
            .iter()
            .any(|f| f.kind == FrameKind::Body)
    }

    /// Bind a fresh local `name` into frame `target` (the innermost when
    /// `None`).
    fn bind(
        &mut self,
        name: Name,
        kind: DefKind,
        mutable: bool,
        ty: Ty,
        span: SourceLoc,
        target: Option<usize>,
    ) -> DefId {
        let text = self.name(name).to_string();
        let Some(last) = self.frames.len().checked_sub(1) else {
            return usize::MAX;
        };
        let idx = target.unwrap_or(last).min(last);
        let id = self.def(Def {
            name: text.clone(),
            kind,
            name_span: name.span,
            span,
            ty,
            mutable,
            container: None,
            visible: SourceLoc::new(name.span.start, self.frames[idx].end.max(name.span.end)),
        });
        self.occ(name.span, id, Role::Decl);
        self.frames[idx].names.insert(text, id);
        id
    }

    /// The frame a routing guard's exit binding lands in: the one enclosing
    /// the innermost loop.
    fn loop_exit_frame(&self) -> Option<usize> {
        self.frames
            .iter()
            .rposition(|f| f.kind == FrameKind::Loop)
            .map(|i| i.saturating_sub(1))
    }

    // --- blocks, statements, chains ----------------------------------------

    fn block_items(&mut self, b: &Block) -> Ty {
        self.block_items_from(b, &Ty::Unknown)
    }

    /// A block's items; a headless chain (no head expression) starts from
    /// `seed` (a `seq`'s input, ADR-0019 pin a).
    fn block_items_from(&mut self, b: &Block, seed: &Ty) -> Ty {
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
    fn seq(&mut self, b: &Block, input: &Ty) -> Ty {
        self.compute(b);
        let top = std::mem::replace(&mut self.top, false);
        let t = self.block_items_from(b, input);
        self.top = top;
        t
    }

    /// A nested block in its own frame.
    fn block(&mut self, b: &Block, kind: FrameKind) -> Ty {
        self.compute(b);
        let top = std::mem::replace(&mut self.top, false);
        self.push(kind, b.span.end);
        let t = self.block_items(b);
        self.pop();
        self.top = top;
        t
    }

    fn chain(&mut self, c: &Chain) -> Ty {
        self.chain_from(c, &Ty::Unknown)
    }

    /// A chain; with no head it starts from `seed`.
    fn chain_from(&mut self, c: &Chain, seed: &Ty) -> Ty {
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

    fn stage(&mut self, st: &Stage, wire: &Wire, exit_bind: bool, chain_span: SourceLoc) -> Ty {
        let target = if exit_bind {
            self.loop_exit_frame()
        } else {
            None
        };
        match &st.kind {
            StageKind::Expr(e) => match &e.kind {
                ExprKind::Var(n) => self.bare_stage(*n, wire, target, chain_span),
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

    fn global_bind(&mut self, name: Name, ty: &Ty) {
        if let Some(&id) = self.global_sites.get(&name.span.start) {
            let first = self.defs[id].name_span == name.span;
            self.occ(name.span, id, if first { Role::Decl } else { Role::Write });
            if first && ty.is_known() {
                self.defs[id].ty = ty.clone();
            }
        }
    }

    /// `-> name` with no annotation: rebind, call, builtin or fresh binding.
    fn bare_stage(
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
        if mapal_lower::is_builtin(text) {
            self.lex(n.span, Class::Builtin);
            return builtin_result(text, wire, self.src);
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

    fn arm(&mut self, arm: &GuardArm, _input: Ty, routing: bool) -> Ty {
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

    fn map_fold(&mut self, op: CollOp, params: &[Name], body: &Block, wire: &Wire) -> Ty {
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

    // --- expressions --------------------------------------------------------

    fn expr(&mut self, e: &Expr) -> Ty {
        match &e.kind {
            ExprKind::Int(_) => Ty::Prim("i32".into()),
            ExprKind::Float => Ty::Prim("f32".into()),
            ExprKind::Str => Ty::Prim("str".into()),
            ExprKind::Bool(_) => Ty::Prim("bool".into()),
            ExprKind::Unit => Ty::Tuple(Vec::new()),
            ExprKind::Hole | ExprKind::Error => Ty::Unknown,
            ExprKind::Var(n) => self.read(*n),
            ExprKind::Unary { op, operand, .. } => {
                let t = self.expr(operand);
                match op {
                    mapal_syntax::UnOp::Not => Ty::Prim("bool".into()),
                    mapal_syntax::UnOp::Neg => t,
                }
            }
            ExprKind::Binary { op, lhs, rhs, .. } => {
                use mapal_syntax::BinOp::*;
                let l = self.expr(lhs);
                let r = self.expr(rhs);
                match op {
                    Eq | Ne | Lt | Gt | Le | Ge | And | Or => Ty::Prim("bool".into()),
                    _ if l.is_known() && !matches!(&l, Ty::Prim(p) if p == "i32") => l,
                    _ if r.is_known() => r,
                    _ => l,
                }
            }
            ExprKind::Member { base, field } => {
                let b = self.expr(base);
                match field {
                    MemberField::Named(f) => self.field_access(&b, *f),
                    MemberField::Index { value, .. } => match &b {
                        Ty::Tuple(ts) => ts.get(*value as usize).cloned().unwrap_or(Ty::Unknown),
                        _ => Ty::Unknown,
                    },
                }
            }
            ExprKind::Index { base, index } => {
                let b = self.expr(base);
                self.expr(index);
                self.elem_through_default(&b)
            }
            ExprKind::Tuple(es) => Ty::Tuple(es.iter().map(|e| self.expr(e)).collect()),
            ExprKind::Array(es) => {
                let ts: Vec<Ty> = es.iter().map(|e| self.expr(e)).collect();
                Ty::Array {
                    elem: Box::new(ts.into_iter().next().unwrap_or(Ty::Unknown)),
                    len: es.len().to_string(),
                }
            }
            ExprKind::Struct { name, fields } => {
                let tname = self.name(*name).to_string();
                if let Some(&id) = self.types.get(&tname) {
                    self.occ(name.span, id, Role::TypeUse);
                }
                for f in fields {
                    match &f.value {
                        Some(v) => {
                            let fname = self.name(f.name);
                            if let Some(&fid) = self.fields.get(&tname).and_then(|m| m.get(fname)) {
                                self.occ(f.name.span, fid, Role::Read);
                            }
                            self.expr(v);
                        }
                        // A pun `{ keys }` reads the local of that name and
                        // names the field: both occurrences, the local's first
                        // (definition goes to the value, as in Rust).
                        None => {
                            self.read(f.name);
                            let fname = self.name(f.name);
                            let field = self.fields.get(&tname).and_then(|m| m.get(fname)).copied();
                            if let Some(fid) = field {
                                self.occ(f.name.span, fid, Role::Read);
                            }
                            self.puns.push(f.name.span);
                        }
                    }
                }
                Ty::Named {
                    name: tname,
                    args: Vec::new(),
                }
            }
            ExprKind::Call { callee, args } => {
                let t = self.expr(callee);
                for a in args {
                    self.expr(a);
                }
                t
            }
            ExprKind::Question(inner) => self.expr(inner),
        }
    }

    fn read(&mut self, n: Name) -> Ty {
        let text = self.name(n);
        if let Some((id, _)) = self.lookup_local(text) {
            self.occ(n.span, id, Role::Read);
            return self.defs[id].ty.clone();
        }
        if let Some(&id) = self.sizes.get(text) {
            self.occ(n.span, id, Role::Read);
            return Ty::Prim("i32".into());
        }
        if let Some(&id) = self.globals.get(text) {
            self.occ(n.span, id, Role::Read);
            return self.defs[id].ty.clone();
        }
        if let Some(&id) = self.fns.get(text) {
            self.occ(n.span, id, Role::Call);
            return self.defs[id].ty.clone();
        }
        if mapal_lower::is_builtin(text) {
            self.lex(n.span, Class::Builtin);
        }
        Ty::Unknown
    }

    /// `base.f`: the field of the base's type, else of its `_` (ADR-0048 D5:
    /// a type with a default interpretation reads through it).
    fn field_access(&mut self, base: &Ty, f: Name) -> Ty {
        let fname = self.name(f);
        let mut cur = base.clone();
        for _ in 0..8 {
            let Some(tname) = cur.named().map(str::to_string) else {
                return Ty::Unknown;
            };
            let Some(fields) = self.fields.get(&tname) else {
                return Ty::Unknown;
            };
            if let Some(&fid) = fields.get(fname) {
                self.occ(f.span, fid, Role::Read);
                return self.defs[fid].ty.clone();
            }
            match fields.get("_") {
                Some(&d) => cur = self.defs[d].ty.clone(),
                None => return Ty::Unknown,
            }
        }
        Ty::Unknown
    }

    fn elem_through_default(&self, t: &Ty) -> Ty {
        let mut cur = t.clone();
        for _ in 0..8 {
            match &cur {
                Ty::Array { elem, .. } => return (**elem).clone(),
                Ty::Named { name, .. } => match self.fields.get(name).and_then(|m| m.get("_")) {
                    Some(&d) => cur = self.defs[d].ty.clone(),
                    None => return Ty::Unknown,
                },
                _ => return Ty::Unknown,
            }
        }
        Ty::Unknown
    }
}

fn is_scalar(t: &str) -> bool {
    mapal_lower::SCALAR_TYPES.contains(&t)
}

fn elem_of(t: &Ty) -> Ty {
    match t {
        Ty::Array { elem, .. } => (**elem).clone(),
        _ => Ty::Unknown,
    }
}

/// Whether a guard arm ends in `-> loop` (lowering's `arm_is_routing`).
fn arm_routes(p: &ArmPayload) -> bool {
    final_chain(p).is_some_and(|c| {
        matches!(
            c.stages.last(),
            Some(Stage {
                kind: StageKind::LoopJump,
                ..
            })
        )
    })
}

/// An arm's final chain: the chain payload, or a block's tail, else its last
/// chain statement (lowering's `loop_exit_bind_names`).
fn final_chain(p: &ArmPayload) -> Option<&Chain> {
    match p {
        ArmPayload::Chain(c) => Some(c),
        ArmPayload::Block(b) => b.tail.as_ref().or_else(|| {
            b.items.iter().rev().find_map(|it| match it {
                BlockItem::Stmt(s) => match &s.kind {
                    StmtKind::Chain(c) => Some(c),
                    _ => None,
                },
                _ => None,
            })
        }),
    }
}

/// The value a builtin stage produces from its input.
fn builtin_result(name: &str, wire: &Wire, src: &str) -> Ty {
    let text = |e: &Expr| {
        src.get(e.span.start as usize..e.span.end as usize)
            .unwrap_or("n")
            .to_string()
    };
    let scalar = |p: &str| match &wire.ty {
        Ty::Array { len, .. } => Ty::Array {
            elem: Box::new(Ty::Prim(p.into())),
            len: len.clone(),
        },
        _ => Ty::Prim(p.into()),
    };
    match name {
        "iota" => Ty::Array {
            elem: Box::new(Ty::Prim("i32".into())),
            len: wire.head.map(text).unwrap_or_else(|| "n".into()),
        },
        "fill" => match (&wire.ty, wire.head.map(|h| &h.kind)) {
            (Ty::Tuple(ts), Some(ExprKind::Tuple(es))) if ts.len() == 2 && es.len() == 2 => {
                Ty::Array {
                    elem: Box::new(ts[0].clone()),
                    len: text(&es[1]),
                }
            }
            (Ty::Tuple(ts), _) if ts.len() == 2 => Ty::Array {
                elem: Box::new(ts[0].clone()),
                len: "n".into(),
            },
            _ => Ty::Unknown,
        },
        "zip" => match &wire.ty {
            Ty::Tuple(ts) => {
                let len = match ts.first() {
                    Some(Ty::Array { len, .. }) => len.clone(),
                    _ => "n".into(),
                };
                Ty::Array {
                    elem: Box::new(Ty::Tuple(ts.iter().map(elem_of).collect())),
                    len,
                }
            }
            _ => Ty::Unknown,
        },
        "enumerate" => match &wire.ty {
            Ty::Array { elem, len } => Ty::Array {
                elem: Box::new(Ty::Tuple(vec![Ty::Prim("i32".into()), (**elem).clone()])),
                len: len.clone(),
            },
            _ => Ty::Unknown,
        },
        "exp" | "sqrt" | "sin" | "cos" => wire.ty.clone(),
        n if n.starts_with("widen_") => scalar(&n["widen_".len()..]),
        n if n.starts_with("to_") => scalar(&n["to_".len()..]),
        _ => Ty::Unknown,
    }
}
