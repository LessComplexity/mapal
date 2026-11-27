//! Fixture of the typing-table golden test: small type builders and a sealed graph to check edges against.

use super::*;

pub(super) const L: SourceLoc = SourceLoc { start: 0, end: 0 };

pub(super) fn int(bits: u8) -> Ty {
    Ty::Int { bits, signed: true }
}
pub(super) fn f32t() -> Ty {
    Ty::Float { bits: 32 }
}
pub(super) fn tup(ts: &[Ty]) -> Ty {
    Ty::Tuple(ts.to_vec())
}
pub(super) fn arr(t: Ty, n: u64) -> Ty {
    Ty::Array {
        elem: Box::new(t),
        size: n,
    }
}
pub(super) fn strct(fields: &[(&str, Ty)]) -> Ty {
    Ty::Struct {
        name: "P".to_string(),
        fields: fields
            .iter()
            .map(|(n, t)| (n.to_string(), t.clone()))
            .collect(),
    }
}

pub(super) fn mk_obj(
    objects: &mut SlotMap<ObjectId, Object>,
    ty: Ty,
    kind: ObjectKind,
) -> ObjectId {
    objects.insert_with_key(|id| Object {
        id,
        ty,
        value: None,
        kind,
        name: None,
        loc: L,
    })
}
pub(super) fn mk_func(
    objects: &mut SlotMap<ObjectId, Object>,
    funcs: &mut SlotMap<FuncId, FuncDef>,
    kind: FuncKind,
    in_ty: Ty,
    out_ty: Ty,
) -> FuncId {
    let input = mk_obj(objects, in_ty, ObjectKind::Parameter);
    let output = mk_obj(objects, out_ty, ObjectKind::Return);
    funcs.insert(FuncDef {
        name: "f".to_string(),
        kind,
        input,
        output,
        morphisms: Vec::new(),
        loc: L,
    })
}

pub(super) struct Fx {
    pub(super) ir: CategoryIr,
    pub(super) merge: ObjectId, // a LoopMerge-kind object (valid loop-edge target)
    pub(super) plain: ObjectId, // a Temporary object (non-merge target; source is unused)
    pub(super) call_f: FuncId,  // Named:    i32 -> f32
    pub(super) map_f: FuncId,   // MapBody:  i32 -> f32
    pub(super) fold_f: FuncId,  // FoldBody: (f32, i32) -> f32
    pub(super) map_caps_f: FuncId, // MapBody (ADR-0027): (i32, i32) -> f32 (1 capture)
    pub(super) fold_caps_f: FuncId, // FoldBody (ADR-0027): (i32, f32, i32) -> f32 (1 capture)
    pub(super) mid: MorphismId,
}

pub(super) fn fixture() -> Fx {
    let mut objects = SlotMap::with_key();
    let mut funcs = SlotMap::with_key();
    let merge = mk_obj(&mut objects, int(32), ObjectKind::LoopMerge);
    let plain = mk_obj(&mut objects, int(32), ObjectKind::Temporary);
    let call_f = mk_func(&mut objects, &mut funcs, FuncKind::Named, int(32), f32t());
    let map_f = mk_func(&mut objects, &mut funcs, FuncKind::MapBody, int(32), f32t());
    let fold_f = mk_func(
        &mut objects,
        &mut funcs,
        FuncKind::FoldBody,
        tup(&[f32t(), int(32)]),
        f32t(),
    );
    let map_caps_f = mk_func(
        &mut objects,
        &mut funcs,
        FuncKind::MapBody,
        tup(&[int(32), int(32)]),
        f32t(),
    );
    let fold_caps_f = mk_func(
        &mut objects,
        &mut funcs,
        FuncKind::FoldBody,
        tup(&[int(32), f32t(), int(32)]),
        f32t(),
    );
    // Minted from a throwaway map: `edge_type_ok` never reads `m.id` (or
    // `m.source` — sty/tty are passed explicitly), only `m.op`/`m.target`.
    let mut ms: SlotMap<MorphismId, ()> = SlotMap::with_key();
    let mid = ms.insert(());
    let ir = CategoryIr {
        objects,
        morphisms: SlotMap::with_key(),
        out_edges: SecondaryMap::new(),
        in_edges: SecondaryMap::new(),
        owner: SecondaryMap::new(),
        facts: crate::facts::Facts::new(funcs.keys()),
        funcs,
        entry: call_f,
        precision: SecondaryMap::new(),
        data_dir: None,
    };
    Fx {
        ir,
        merge,
        plain,
        call_f,
        map_f,
        fold_f,
        map_caps_f,
        fold_caps_f,
        mid,
    }
}

pub(super) fn check(fx: &Fx, op: Operation, sty: Ty, tty: Ty, target: ObjectId) -> bool {
    let m = Morphism {
        id: fx.mid,
        source: fx.plain,
        target,
        op,
        loc: L,
    };
    edge_type_ok(&fx.ir, &m, &sty, &tty)
}
