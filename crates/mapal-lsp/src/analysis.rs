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
    ArmPayload, Block, BlockItem, Chain, CollOp, Diagnostic, Expr, ExprKind, FnDecl, GenericParam,
    GuardArm, Item, MemberField, Name, Program, SizeExpr, SizeKind, SourceLoc, Stage, StageKind,
    StmtKind, TokenKind, TyKind, TypeDecl,
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
    /// A format parameter of a function or a type, `name: kind` (ADR-0051 D1).
    FormatParam,
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
            DefKind::FormatParam => "format parameter",
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
            // A format parameter is a generic parameter like a size: the
            // editors' one class for both (ADR-0051 D1).
            DefKind::SizeParam | DefKind::FormatParam if decl => Class::SizeDeclaration,
            DefKind::SizeParam | DefKind::FormatParam => Class::Size,
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
    /// The enclosing function's or type's generic parameters: sizes and
    /// formats (ADR-0051 D1).
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

/// A stage builtin, or a removed conversion name (ADR-0051 D6): reserved, so
/// never a fresh binding — the compiler reports its use (L1713) and the
/// editors show it as the builtin it was.
pub fn is_builtin_name(name: &str) -> bool {
    mapal_lower::is_builtin(name) || mapal_lower::removed_builtin(name).is_some()
}

/// The value a builtin stage produces from its input; `explicit` is the first
/// explicit generic argument's text (`to<bf16>`: `bf16`, ADR-0051 D2).
fn builtin_result(name: &str, explicit: Option<&str>, wire: &Wire, src: &str) -> Ty {
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
        // ADR-0051 D4: the target is the first generic argument (else inferred
        // from the context, which the analysis does not model).
        "to" | "widen" => match explicit {
            Some(t) => scalar(t),
            None => Ty::Unknown,
        },
        n => match mapal_lower::removed_builtin(n) {
            Some(r) => {
                let t = r
                    .split_once('<')
                    .map_or("", |(_, t)| t.trim_end_matches('>'));
                scalar(t)
            }
            None => Ty::Unknown,
        },
    }
}

mod chains;
mod exprs;
mod items;
mod scopes;
