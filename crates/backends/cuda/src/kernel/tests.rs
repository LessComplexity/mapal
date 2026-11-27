//! Unit tests of the device kernel emitter.

use super::*;

use mapal_ir::{Dest, IrBuilder};

const L: SourceLoc = SourceLoc { start: 0, end: 0 };

fn lower_src(src: &str) -> CategoryIr {
    let po = mapal_syntax::parse(src);
    assert!(po.diagnostics.is_empty(), "parse: {:?}", po.diagnostics);
    mapal_lower::lower(src, &po.program).unwrap_or_else(|d| panic!("lower: {d:?}"))
}

fn emit_src(src: &str) -> String {
    crate::emit(&lower_src(src)).unwrap()
}

fn build_example(name: &str) -> CategoryIr {
    let path = format!(
        "{}/../../../examples/{name}.mapal",
        env!("CARGO_MANIFEST_DIR")
    );
    let src = std::fs::read_to_string(&path).unwrap();
    lower_src(&src)
}

/// The fnames map, built exactly as lib.rs's emit does.
fn fnames_of(ir: &CategoryIr) -> SecondaryMap<FuncId, String> {
    let mut fnames: SecondaryMap<FuncId, String> = SecondaryMap::new();
    let entry = ir.entry();
    for (ord, (id, _)) in ir.funcs().enumerate() {
        fnames.insert(
            id,
            if id == entry {
                "mapal_main".to_string()
            } else {
                format!("fn{ord}")
            },
        );
    }
    fnames
}

/// The device section of an emitted module (twin definitions) — from the
/// first `__device__` to the first `__global__` (kernels follow it).
fn twin_slice(cu: &str) -> &str {
    let start = cu.find("static __device__").expect("a device twin");
    let end = cu[start..]
        .find("__global__")
        .map(|e| start + e)
        .unwrap_or(cu.len());
    &cu[start..end]
}

/// The qualifier of the one fn named `name`.
fn qual_of(ir: &CategoryIr, quals: &Qualifiers, name: &str) -> FnQual {
    for (id, fd) in ir.funcs() {
        if fd.name == name {
            return quals.get(id);
        }
    }
    panic!("no fn named {name}");
}

mod abi_traps;
mod cells_captures;
mod shapes_inline;
