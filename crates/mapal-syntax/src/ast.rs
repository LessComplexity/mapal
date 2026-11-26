//! The Mapal-Core parse tree (DESIGN §15): a **thin** tree — syntax + names +
//! spans + literal values + flags only. No name resolution, no type inference,
//! no effect classification (C10; architecture.md §2.2.2).
//!
//! `Debug` is derived on every node so tests can snapshot the structured
//! values; there is deliberately **no** `Display` impl anywhere in this crate
//! (C3 / I5 / J5) — rendering belongs to `mapal-cli`. Every node carries a
//! [`SourceLoc`] span (C12); `Name` is a bare span (text via `&source[span]`,
//! the same single-source-of-truth rule as tokens).
//!
//! See DESIGN §15 for the type catalogue, §14 for the grammar each node maps
//! to, and §16 for the rejected-but-kept forms (`Call`, `Question`, `Dynamic`,
//! `StmtBlock`, `LoopLabel::Custom`, `GuardDiscr::OutOfCore`).

use crate::diag::Diagnostic;
use crate::loc::SourceLoc;

/// The result of parsing: the program tree plus the merged lex+parse
/// diagnostics, stably sorted by `(span.start, span.end)` (DESIGN §15/§18).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ParseOutput {
    pub program: Program,
    pub diagnostics: Vec<Diagnostic>,
}

/// `program := item* EOF` (DESIGN §14.1).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Program {
    pub items: Vec<Item>,
    pub span: SourceLoc,
}

/// A top-level item: a function or type declaration, or a top-level binding
/// (ADR-0048 D1) (DESIGN §14.1). `Error` holds a recovery region (e.g. a
/// top-level statement that is not a chain, ⟦P0012⟧).
// One `FnDecl` per function and a handful per file: boxing it would cost every
// consumer a deref for no measurable saving.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Item {
    Fn(FnDecl),
    Type(TypeDecl),
    /// ADR-0048 D1: `2048 -> hidden_size;` — a chain at the top level whose
    /// binding stages name program-wide immutable values.
    Bind(TopBind),
    Error(SourceLoc),
}

/// ADR-0048 D1: a top-level binding statement — one chain (`;`-terminated)
/// whose `-> name` stages bind program-wide values. Order-independent, never
/// `mut`, never effectful (lowering checks: L1801–L1808).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TopBind {
    pub chain: Chain,
    pub span: SourceLoc,
}

/// `fn-decl := 'fn' IDENT generics? '(' params? ')' ( '->' type )? block`
/// (DESIGN §14.1; `generics := '<' generic (',' generic)* ','? '>'`,
/// `generic := IDENT (':' IDENT)?` — ADR-0047 D1, ADR-0051 D1).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FnDecl {
    pub name: Name,
    /// ADR-0047 / ADR-0051: the declared generic list `<rows, cache: float>`
    /// in declared order — size parameters and format parameters; empty for
    /// a fixed function.
    pub generics: Vec<GenericParam>,
    pub params: Vec<Param>,
    pub ret_ty: Option<Ty>,
    pub body: Block,
    pub span: SourceLoc,
}

/// ADR-0051 D1: one entry of a declared generic list — a size parameter
/// (`rows`, no kind) or a format parameter with its kind (`cache: float`).
/// The kind's name is kept as written (lowering checks it: `float`,
/// `narrow`, `int`, `number`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GenericParam {
    pub name: Name,
    /// `Some(kind)` for a format parameter; `None` for a size parameter.
    pub kind: Option<Name>,
    pub span: SourceLoc,
}

impl GenericParam {
    /// Whether this is a size parameter (a bare name, ADR-0047).
    pub fn is_size(&self) -> bool {
        self.kind.is_none()
    }
}

/// `compute := ':' member ('|' member)*` on anything that opens a body — a
/// step `-> fold: tf32 | f32 { … }`, a block `loop: tf32 | f32 { … }` /
/// `seq: … { … }`, a function after its output `fn f(a) -> T: tf32 | f32 { … }`
/// (ADR-0041 D2): the products that body admits (D1), nested bodies included
/// unless they carry their own (innermost wins; `: f32` restores the default).
/// ADR-0043 D4: the narrow members `bf16` / `f16` (multiplicands rounded to
/// the format, f32 accumulation).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Precision {
    pub f32: bool,
    pub tf32: bool,
    pub bf16: bool,
    pub f16: bool,
    pub span: SourceLoc,
}

/// `param := 'mut'? IDENT ':' type` (DESIGN §14.1).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Param {
    pub mut_span: Option<SourceLoc>,
    pub name: Name,
    pub ty: Ty,
    pub span: SourceLoc,
}

/// `type-decl := ('type' | category) IDENT '{' field-list? '}'` (DESIGN §14.1).
/// A `category` keyword in this position is recovered as `type` with no new
/// diagnostic (L0004 already emitted; C14).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TypeDecl {
    pub name: Name,
    /// ADR-0048 D4 / ADR-0051 D1: the declared generic list `type
    /// Matrix<rows, columns, element: float>`; empty for a fixed type.
    pub generics: Vec<GenericParam>,
    pub fields: Vec<Field>,
    pub span: SourceLoc,
}

/// `field := IDENT ':' type` (DESIGN §14.1).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Field {
    pub name: Name,
    pub ty: Ty,
    pub span: SourceLoc,
}

/// An identifier occurrence: a bare span (text via `&source[span]`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Name {
    pub span: SourceLoc,
}

/// A type with its span (DESIGN §14.2 / §15).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Ty {
    pub kind: TyKind,
    pub span: SourceLoc,
}

/// The shape of a type (DESIGN §14.2). Size arguments (`Matrix<64, n>`,
/// ADR-0048 D4) are `Generic`; arguments that are not size expressions keep
/// their base `Named` (⟦P0103⟧); `Dynamic` (`[T]`, ⟦P0104⟧) is kept for
/// span-precise rejection.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum TyKind {
    /// `i32`, `Pixel`, …; also the kept base of ⟦P0103⟧ generics.
    Named(Name),
    /// ADR-0048 D4: a size-generic type at its sizes, `Matrix<rows, 64 * 4>`
    /// — one size expression per declared size parameter.
    Generic {
        name: Name,
        args: Vec<SizeExpr>,
    },
    Tuple(Vec<Ty>),
    Array {
        elem: Box<Ty>,
        /// The length when written as one integer literal (`[T; 4096]`); 0
        /// when `size` holds an expression.
        len: u64,
        len_span: SourceLoc,
        /// ADR-0047 D2: a length written as a size expression
        /// (`[T; rows * columns]`, `[T; 64 * 64]`); `None` for a literal.
        size: Option<SizeExpr>,
    },
    /// `[T]` — kept, P0104 reported.
    Dynamic(Box<Ty>),
    Error,
}

/// ADR-0047 D2: an array length over integer literals and the function's
/// size parameters — `size := term (('+' | '-') term)*`, `term := atom ('*'
/// atom)*`, `atom := INT | IDENT | '(' size ')'`. Parentheses only group.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SizeExpr {
    pub kind: SizeKind,
    pub span: SourceLoc,
}

/// The shape of a [`SizeExpr`].
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum SizeKind {
    Int(u64),
    /// A size parameter (resolved by lowering; an undeclared one is L1701).
    Name(Name),
    Binary {
        op: SizeOp,
        lhs: Box<SizeExpr>,
        rhs: Box<SizeExpr>,
    },
}

/// The operators a size expression admits: the ones whose unknown operand is
/// recovered exactly from the result (ADR-0047 D2; `/` and `%` are P0119).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SizeOp {
    Add,
    Sub,
    Mul,
}

/// `block := '{' block-item* tail? '}'` (DESIGN §14.3).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Block {
    pub items: Vec<BlockItem>,
    pub tail: Option<Chain>,
    pub span: SourceLoc,
    /// The `: compute` written on whatever opened this body (ADR-0041 D2).
    pub compute: Option<Precision>,
}

/// A block item is a statement or a guard arm; mixing the two draws P0006
/// (DESIGN §14.3).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum BlockItem {
    Stmt(Stmt),
    Arm(GuardArm),
}

/// A statement with its span (DESIGN §15).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Stmt {
    pub kind: StmtKind,
    pub span: SourceLoc,
}

/// The shape of a statement (DESIGN §14.3 / §15). `Error` marks a recovery
/// region.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum StmtKind {
    Chain(Chain),
    /// `place <- expr`.
    Bind(BindStmt),
    Loop(LoopStmt),
    /// Recovery region.
    Error,
}

/// `loop-stmt := 'loop' block | IDENT block` (DESIGN §14.3). A `Custom` label
/// means P0110 was reported (ADR-0011).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct LoopStmt {
    pub label: LoopLabel,
    pub body: Block,
    pub span: SourceLoc,
}

/// The loop label: the keyword `loop`, or a custom identifier (⟦P0110⟧).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum LoopLabel {
    Loop(SourceLoc),
    /// Custom ⇒ P0110 was reported.
    Custom(Name),
}

/// `bind-stmt := 'mut'? IDENT ('[' expr ']')? (':' type)? '<-' expr`
/// (DESIGN §14.3). The optional `[' expr ']` is the element-update target
/// (`c[i] <- x`, ADR-0021): an *indexed* bind is a rebind (never a fresh
/// shadow). `index.is_some()` excludes `mut` (P0013) and a type annotation
/// (P0014); a nested target `c[i][j]` draws P0015.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BindStmt {
    pub mut_span: Option<SourceLoc>,
    pub name: Name,
    /// `Some` for the element-update sugar `c[i] <- x` (ADR-0021).
    pub index: Option<Expr>,
    pub ty: Option<Ty>,
    pub value: Expr,
    pub span: SourceLoc,
}

/// `chain := expr stage* | stage+` — a headed or headless flat ordered stage
/// list (DESIGN §14.3; chains never nest arrows, C11).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Chain {
    pub head: Option<Expr>,
    pub stages: Vec<Stage>,
    pub span: SourceLoc,
}

/// A single `-> …` stage: the arrow's span, the stage kind, and the whole-stage
/// span (DESIGN §15).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Stage {
    pub arrow_span: SourceLoc,
    pub kind: StageKind,
    pub span: SourceLoc,
}

/// The classified body of a stage (DESIGN §14.3/§14.4 / §15).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum StageKind {
    /// Targets, tuple stages, ADR-0005 general expression stages.
    Expr(Expr),
    /// `-> x: i32` / `-> mut y` — a binding stage.
    Bind {
        mut_span: Option<SourceLoc>,
        name: Name,
        ty: Option<Ty>,
    },
    /// `-> ret` / `-> ret.0`.
    Ret {
        proj: Option<(u64, SourceLoc)>,
    },
    /// `-> loop` — back-edge jump, innermost loop (ADR-0011).
    LoopJump,
    /// Hole-expression: contains exactly one `Expr::Hole` leaf, as the leftmost
    /// leaf. `-> + 5` ⇒ `Binary(Add, Hole, Int 5)`; `-> * 2 + 1` ⇒
    /// `Binary(Add, Binary(Mul, Hole, Int 2), Int 1)`.
    OpShorthand {
        expr: Expr,
    },
    /// Ordered guard arms; lowering selects Phi vs Trace routing (§4.4/§4.5).
    Guard(Vec<GuardArm>),
    /// Fanout branches are headless chains.
    Fanout {
        kind: FanoutKind,
        branches: Vec<Chain>,
    },
    /// `map`/`fold` with an inline op-block (ADR-0009).
    MapFold {
        op: CollOp,
        params: Vec<Name>,
        body: Block,
    },
    /// `seq { … }` — an ordered statement block in stage position (ADR-0019).
    /// The body is the ordinary block production (statements + optional tail),
    /// **not** a fanout: `seq` is Mapal-Core's keyword-marked block stage. Guard
    /// arms are illegal in it (a clean guard token → stray-guard P0004).
    SeqBlock(Block),
    /// ⟦P0115⟧ anonymous block stage — kept.
    StmtBlock(Block),
    Error(SourceLoc),
}

/// The kind of a fanout stage (DESIGN §14.4). `Void` ⇒ P0113 reported.
/// `seq` is no longer a fanout kind (ADR-0019): it is `StageKind::SeqBlock`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum FanoutKind {
    Plain,
    Void(SourceLoc),
}

/// The collection operator of a `map`/`fold` stage (DESIGN §15).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CollOp {
    Map,
    Fold,
}

/// A guard arm: the discriminant, its span, the payload, and the whole-arm
/// span (DESIGN §14.3 / §15).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct GuardArm {
    pub discr: GuardDiscr,
    pub discr_span: SourceLoc,
    pub payload: ArmPayload,
    pub span: SourceLoc,
}

/// The discriminant of a guard arm (DESIGN §15). `OutOfCore` ⇒ P0106 (pattern
/// arm).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GuardDiscr {
    True,
    False,
    Default,
    Int(u64),
    /// ⇒ P0106 (pattern guard arm).
    OutOfCore,
}

/// A guard-arm payload: a chain or a plain payload block (DESIGN §14.3).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ArmPayload {
    Chain(Chain),
    Block(Block),
}

/// An expression with its span (DESIGN §14.6 / §15).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: SourceLoc,
}

/// The shape of an expression (DESIGN §14.6 / §15). `Call`, `Question`, and
/// `Error` are rejected-but-kept or recovery forms (C13).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ExprKind {
    /// Value clamped like L0008 (re-parse of span digits; no new diagnostic, C14).
    Int(u64),
    /// Value is type-directed (f32 vs f64) — parsed later from the span.
    Float,
    /// Unescaping is the consumer's job (`unescape_string`).
    Str,
    Bool(bool),
    /// The Unit literal `()` (plan-time-builtin): the only legal empty-paren
    /// form; its one sanctioned use is the `() -> time` chain head.
    Unit,
    Var(Name),
    /// The piped value inside an `OpShorthand` rhs — never constructible from
    /// ordinary expression syntax.
    Hole,
    Unary {
        op: UnOp,
        op_span: SourceLoc,
        operand: Box<Expr>,
    },
    Binary {
        op: BinOp,
        op_span: SourceLoc,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    Member {
        base: Box<Expr>,
        field: MemberField,
    },
    Index {
        base: Box<Expr>,
        index: Box<Expr>,
    },
    Tuple(Vec<Expr>),
    Array(Vec<Expr>),
    Struct {
        name: Name,
        fields: Vec<FieldInit>,
    },
    /// ADR-0051 D2: a stage's explicit generic arguments — `to<bf16>`,
    /// `project<4096, 2048>` — a prefix of the callee's declared list, each a
    /// size expression (a format is a bare name: `bf16`, or the caller's own
    /// format parameter). Built only in stage position.
    Generic {
        name: Name,
        args: Vec<SizeExpr>,
    },
    /// ⟦P0108⟧ — kept for precision.
    Call {
        callee: Box<Expr>,
        args: Vec<Expr>,
    },
    /// ⟦P0101⟧ — kept.
    Question(Box<Expr>),
    Error,
}

/// A member-access field: a named field or a tuple projection (`x.0`) (DESIGN §15).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum MemberField {
    Named(Name),
    Index { value: u64, span: SourceLoc },
}

/// A struct field initializer; `value: None` is a pun shorthand (DESIGN §14.6).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FieldInit {
    pub name: Name,
    pub value: Option<Expr>,
    pub span: SourceLoc,
}

/// A unary operator (DESIGN §15). Not in the §3.6 table; binds tighter than
/// `*`, looser than postfix (W15).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UnOp {
    Neg,
    Not,
}

/// A binary operator (DESIGN §15; §14.6 precedence levels 3–7).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    And,
    Or,
}
