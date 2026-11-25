//! The graph data structures (DESIGN §2, §4, §5, §6) and the read-only API.
//!
//! Storage is `slotmap` only — no `HashMap` anywhere (D2 / I12): SlotMap and
//! SecondaryMap iterate in key (= insertion) order, so every dump, topo order,
//! and snapshot is deterministic. Edge `Vec`s are in insertion order, the
//! stable tie-break for topo (DESIGN §13).
//!
//! Fields are `pub(crate)`: downstream crates and integration tests construct
//! graphs only through the builder; in-crate unit tests may corrupt a graph to
//! negative-test [`crate::validate`].

use slotmap::{SecondaryMap, SlotMap};

use crate::loc::SourceLoc;
use crate::ty::{NarrowFloat, Ty, Value};

slotmap::new_key_type! {
    /// Stable key for an [`Object`].
    pub struct ObjectId;
    /// Stable key for a [`Morphism`].
    pub struct MorphismId;
    /// Stable key for a [`FuncDef`].
    pub struct FuncId;
}

/// The kind of an [`Object`] (DESIGN §4; exactly category-ir §3.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ObjectKind {
    /// A function input (one per function; product ty when multi-param).
    Parameter,
    /// The default kind — a computed intermediate.
    Temporary,
    /// A literal: `value: Some(v)`, in-degree 0 (I7).
    Constant,
    /// The function's output object (one per function).
    Return,
    /// The `U` of `Tr^U` — receives one `LoopEnter` + ≥1 `LoopBack` (E1).
    LoopMerge,
}

/// A node in the dataflow graph (DESIGN §4).
#[derive(Clone, Debug, PartialEq)]
pub struct Object {
    pub id: ObjectId,
    pub ty: Ty,
    /// `Some` ⇔ `kind == Constant` (I7); then `value.ty() == ty`.
    pub value: Option<Value>,
    pub kind: ObjectKind,
    /// Surface variable name, for dumps/debug (DESIGN §4, D4).
    pub name: Option<String>,
    pub loc: SourceLoc,
}

/// The Core operation set (DESIGN §5; the ADR-0013 delta from category-ir §3.3).
///
/// Omitted vs spec §3.3: `Identity` (`Output` is the single sanctioned
/// identity-shaped exception), `Const` (constants are objects), `Trace` (the
/// cycle is the trace), and all out-of-Core variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Operation {
    // structural
    /// Slot-injection into a product object (LC-4 reading of spec `Pair`).
    Pair {
        slot: u32,
        arity: u32,
    },
    /// π_index out of a Tuple/Struct.
    Proj {
        index: u32,
    },
    // arithmetic (binary ops take the 2-tuple product as source)
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    /// Unary; IEEE fneg ≠ `0 − x` (parse tree has `UnOp::Neg`) — ADR-0013.
    Neg,
    // comparison / logic
    Eq,
    Neq,
    Lt,
    Gt,
    Le,
    Ge,
    And,
    Or,
    Not,
    // selection
    /// `(T × T × Bool) → T`; first-class per CHANGES §1.2.
    Phi,
    // calls & collections
    /// Named functions only.
    Call(FuncId),
    /// `[T; n] → [U; n]`; `body: T → U` (ADR-0009/LC-2). With `captures > 0`
    /// (ADR-0027), the source is the product `(c₁…cₖ, [T; n])` — the first
    /// `captures` components are enclosing values broadcast to every element —
    /// and the body input is `(c₁…cₖ, T)`.
    Map {
        body: FuncId,
        /// Leading capture components of the source product (ADR-0027). `0`
        /// for every pre-ADR-0027 program: source is the bare array.
        captures: u32,
    },
    /// `(Acc × [T; n]) → Acc`; `body: (Acc × T) → Acc`. With `captures > 0`
    /// (ADR-0027): source `(c₁…cₖ, Acc, [T; n])`, body input `(c₁…cₖ, Acc, T)`.
    Fold {
        body: FuncId,
        /// Leading capture components of the source product (ADR-0027).
        captures: u32,
    },
    /// `([T; n] × I) → T`; OOB = trap in Core (ADR-0013).
    Index,
    /// `([A; n] × [B; n]) → [(A, B); n]` — the canonical iso `Aⁿ × Bⁿ ≅ (A×B)ⁿ`
    /// (ADR-0018). Source is the 2-tuple product (Pair-then-primitive, as for
    /// every binary op); both arrays share size `n`. Pure — no token.
    Zip,
    /// `[A; n] → [(i32, A); n]` (ADR-0018): pairs each element with its index,
    /// index pinned `i32` with the extra bound `n ≤ i32::MAX`. Pure — no token.
    Enumerate,
    /// `n → [i32; n]` of `0..n-1` (ADR-0029): the source is a `Constant`
    /// integer count (builder-minted; validate ties its value to the target's
    /// static size — the static-n rule). Element type pinned `i32` (v1), so
    /// `n ≤ i32::MAX` (the Enumerate bound). Pure — no token; trap-free.
    Iota,
    /// `T → [T; n]` with every element the source value (ADR-0029): the count
    /// is **type-carried** by the target `[T; n]` (deduced, not stored — no
    /// count operand). Pure — no token; trap-free.
    Fill,
    /// `(Array{T,n} × I × T) → Array{T,n}` (ADR-0021): a fresh array with slot
    /// `i` replaced. Source is the internal 3-tuple product. `I` is exactly
    /// `Index`'s integer-scalar set; OOB (`i < 0 ∨ i ≥ n`) = trap (`IndexOob`,
    /// the same class as `Index`). Pure — no token; may-trap.
    Update,
    // effects (§8)
    /// `(IoToken × P) → IoToken`, `P` printable — the effectful op. `newline: true`
    /// is `println` (appends `\n`), `false` is `print` (raw) — ADR-0015.
    Print {
        newline: bool,
    },
    /// `IoToken → (IoToken × f64)` — the effectful monotonic-clock read
    /// (plan-time-builtin). The target is the `(token, milliseconds)` pair;
    /// token-threaded exactly like `Print`, so it is never reordered, folded,
    /// CSE'd, or DCE'd.
    TimeMs,
    /// `(IoToken × Str) → (IoToken × [T; n])` — the effectful file read
    /// (ADR-0045): the source pairs the token with the path literal (a `Str`
    /// constant, the print pair's shape); the target pairs the next token
    /// with the array, whose type the binding's annotation gave — `n`
    /// little-endian elements of `T ∈ {i32, i64, u8, f32, f64, bf16, f16}`,
    /// exactly `n · |T|` bytes. Token-threaded like `Print` / `TimeMs`, so it
    /// is never reordered, folded, CSE'd or DCE'd; the array it yields is an
    /// ordinary immutable value. A relative path resolves per
    /// [`crate::read::resolve`] against [`CategoryIr::data_dir`].
    Read,
    // loops (§7) — the inline-cycle realization of `Tr^U`
    /// `U → U`, init value → LoopMerge.
    LoopEnter,
    /// `(U × Bool) → U`, route → LoopMerge; THE back edge; fires on `true`.
    LoopBack,
    /// `(B × Bool) → B`, route → exit object; fires on `false`.
    LoopExit,
    /// The only identity-shaped morphism (D6): bare `x -> ret` / `x -> ret.k`.
    Output,
    /// Explicit numeric widening (ADR-0029): one of `i32→i64`, `i32→f32`,
    /// `i32→f64`, or `f32→f64`; ADR-0043: `bf16|f16 → f32|f64` (exact).
    /// Pure — no token; trap-free.
    Widen,
    /// ADR-0043: `f32 → Narrow(fmt)` — the operand rounded once to
    /// the target's narrow format, to nearest with ties to even
    /// ([`crate::narrow::round`]): the `to_bf16` / `to_f16` builtins. Pure —
    /// no token; total and trap-free.
    Narrow,
    /// A math builtin on a float scalar, `Float{b} → Float{b}` (ADR-0046;
    /// `exp` ADR-0042): the `x -> exp` / `sqrt` / `sin` / `cos` pipeline
    /// builtins. Pure — no token; total and trap-free. Its value is
    /// [`crate::math`]'s reference on the exact faces and within its
    /// contract on the default face.
    Math(crate::math::MathFn),
}

impl Operation {
    /// ADR-0042's `exp`, now one member of [`Operation::Math`] (every reader
    /// of the operation keeps its meaning).
    #[allow(non_upper_case_globals)]
    pub const Exp: Operation = Operation::Math(crate::math::MathFn::Exp);
}

/// A directed dataflow edge (DESIGN §5). Exactly one source, one target (I1).
#[derive(Clone, Debug, PartialEq)]
pub struct Morphism {
    pub id: MorphismId,
    pub source: ObjectId,
    pub target: ObjectId,
    pub op: Operation,
    pub loc: SourceLoc,
}

/// ADR-0041 D1: the product precisions a float `Mul` admits — a set, not a
/// preference. `{f32}` is the default face (ADR-0040); `tf32` admits operands
/// rounded to 10 explicit significand bits (accumulation stays f32). A set's
/// semantics is its weakest member's bound (D3); stronger members are always
/// admitted. ADR-0043 D4: the narrow members `bf16` / `f16` admit
/// multiplicands rounded to that format (accumulation stays f32).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Precision {
    pub f32: bool,
    pub tf32: bool,
    pub bf16: bool,
    pub f16: bool,
}

impl Precision {
    pub const F32: Self = Self {
        f32: true,
        tf32: false,
        bf16: false,
        f16: false,
    };

    /// The weakest member admits TF32 operands.
    pub fn admits_tf32(self) -> bool {
        self.tf32
    }

    /// ADR-0043 D4: the 16-bit member a product admits, the weakest first
    /// (`bf16`, 8 significand bits, before `f16`'s 11): multiplicands rounded
    /// to it, accumulated in f32.
    pub fn half(self) -> Option<NarrowFloat> {
        if self.bf16 {
            Some(NarrowFloat::Bf16)
        } else if self.f16 {
            Some(NarrowFloat::F16)
        } else {
            None
        }
    }

    /// The weakest member's multiplicand unit roundoff `u_t` (ADR-0041 D3):
    /// `bf16` 2⁻⁸, `f16` / `tf32` 2⁻¹¹; `None` for `{f32}` (exact products
    /// of f32 operands, one rounding each).
    pub fn unit(self) -> Option<f64> {
        match (self.half(), self.tf32) {
            (Some(n), _) => Some(crate::narrow::NarrowForm::of(n).u),
            (None, true) => Some(2f64.powi(-11)),
            (None, false) => None,
        }
    }

    /// Some member rounds a multiplicand (`tf32`, `bf16`, `f16`).
    pub fn reduced(self) -> bool {
        self.unit().is_some()
    }
}

impl Default for Precision {
    fn default() -> Self {
        Self::F32
    }
}

/// The kind of a [`FuncDef`] (DESIGN §6).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FuncKind {
    /// A surface function; the only `Call` target.
    Named,
    /// A `Map` body block: `T → U`.
    MapBody,
    /// A `Fold` body block: `(Acc, T) → Acc`.
    FoldBody,
}

/// A function: one input object, one output object, and its morphism set
/// (DESIGN §6). `morphisms` is in insertion order (a valid construction order);
/// topo is recomputed on demand (D5).
#[derive(Clone, Debug, PartialEq)]
pub struct FuncDef {
    pub name: String,
    pub kind: FuncKind,
    pub input: ObjectId,
    pub output: ObjectId,
    pub morphisms: Vec<MorphismId>,
    pub loc: SourceLoc,
}

/// The sealed category (DESIGN §2). Append-only during construction, then
/// immutable; produced only by [`crate::IrBuilder::seal`].
pub struct CategoryIr {
    pub(crate) objects: SlotMap<ObjectId, Object>,
    pub(crate) morphisms: SlotMap<MorphismId, Morphism>,
    pub(crate) out_edges: SecondaryMap<ObjectId, Vec<MorphismId>>,
    pub(crate) in_edges: SecondaryMap<ObjectId, Vec<MorphismId>>,
    pub(crate) owner: SecondaryMap<ObjectId, FuncId>,
    pub(crate) funcs: SlotMap<FuncId, FuncDef>,
    pub(crate) entry: FuncId,
    /// Stored facts ([`crate::facts`]); not part of the graph's identity.
    pub(crate) facts: crate::facts::Facts,
    /// ADR-0041 D1: each float product's admitted set, where it is not `{f32}`.
    pub(crate) precision: SecondaryMap<MorphismId, Precision>,
    /// ADR-0045 D5: the directory a relative `read` path resolves against —
    /// the program's source directory, recorded by the driver at compile
    /// time (`None`: the working directory at run time).
    pub(crate) data_dir: Option<String>,
}

impl std::fmt::Debug for CategoryIr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CategoryIr")
            .field("objects", &self.objects)
            .field("morphisms", &self.morphisms)
            .field("out_edges", &self.out_edges)
            .field("in_edges", &self.in_edges)
            .field("owner", &self.owner)
            .field("funcs", &self.funcs)
            .field("entry", &self.entry)
            .finish()
    }
}

impl CategoryIr {
    /// The object for `id`, or `None` if it does not belong to this graph.
    pub fn object(&self, id: ObjectId) -> Option<&Object> {
        self.objects.get(id)
    }

    /// The morphism for `id`, or `None`.
    pub fn morphism(&self, id: MorphismId) -> Option<&Morphism> {
        self.morphisms.get(id)
    }

    /// The precisions product `m` admits (ADR-0041 D1); `{f32}` for every
    /// morphism no `: compute` covers.
    pub fn precision(&self, m: MorphismId) -> Precision {
        self.precision.get(m).copied().unwrap_or_default()
    }

    /// All objects, in deterministic (insertion) order.
    pub fn objects(&self) -> impl Iterator<Item = (ObjectId, &Object)> {
        self.objects.iter()
    }

    /// All morphisms, in deterministic (insertion) order.
    pub fn morphisms(&self) -> impl Iterator<Item = (MorphismId, &Morphism)> {
        self.morphisms.iter()
    }

    /// Edges whose source is `id`, in insertion order. Empty if unknown.
    pub fn out_edges(&self, id: ObjectId) -> &[MorphismId] {
        self.out_edges.get(id).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Edges whose target is `id`, in insertion order. Empty if unknown.
    pub fn in_edges(&self, id: ObjectId) -> &[MorphismId] {
        self.in_edges.get(id).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// The function `id` belongs to. Panics only on a foreign id (unreachable
    /// for a sealed graph's own objects).
    pub fn owner(&self, id: ObjectId) -> FuncId {
        self.owner[id]
    }

    /// The function `id` belongs to, or `None` if unowned (used by
    /// [`crate::validate`] to certify the I6 ownership clause).
    pub fn try_owner(&self, id: ObjectId) -> Option<FuncId> {
        self.owner.get(id).copied()
    }

    /// The function definition for `id`, or `None`.
    pub fn func(&self, id: FuncId) -> Option<&FuncDef> {
        self.funcs.get(id)
    }

    /// All functions, in declaration (insertion) order.
    pub fn funcs(&self) -> impl Iterator<Item = (FuncId, &FuncDef)> {
        self.funcs.iter()
    }

    /// The entry function (DESIGN §2). Always `FuncKind::Named`.
    pub fn entry(&self) -> FuncId {
        self.entry
    }

    /// ADR-0045 D5: the directory a relative `read` path resolves against,
    /// when the driver recorded one ([`Self::with_data_dir`]).
    pub fn data_dir(&self) -> Option<&str> {
        self.data_dir.as_deref()
    }

    /// The same graph with its `read` base directory recorded (ADR-0045 D5):
    /// the driver passes the program's source directory; a rewrite carries
    /// it to the graph it rebuilds. Not part of the graph's identity.
    pub fn with_data_dir(mut self, dir: Option<String>) -> Self {
        self.data_dir = dir;
        self
    }
}
