//! ADR-0045: the `read` builtin's semantics, shared by every consumer that
//! runs it on the host — the interpreter (the oracle) calls [`load`]; the
//! native runtime (`mapal-rt`, which links into every emitted program and so
//! takes no dependency on this crate) carries the same resolution and the
//! same messages, and its tests replay [`resolve`] and [`size_message`]
//! against its own.
//!
//! - **Types** (D2): arrays of `i32`, `i64`, `u8`, `f32`, `f64`, `bf16`,
//!   `f16`, and arrays / structs / tuples of those at any depth with no
//!   padding — the file is the value's bytes, leaves in order ([`readable`]).
//! - **File format** (D3): raw little-endian elements, exactly `n · |T|`
//!   bytes; any other length, and a file that cannot be opened, is a read
//!   failure: exit [`EXIT_CODE`] with [`size_message`] / [`open_message`] on
//!   stderr, identical on every backend.
//! - **Path** (D5): [`resolve`] — absolute as written; a relative path
//!   against `$MAPAL_DATA_DIR` when set and non-empty, else against the
//!   directory the driver recorded at compile time (the program's source
//!   directory, [`crate::CategoryIr::data_dir`]), else the working directory.

use std::path::{Path, PathBuf};

use crate::ty::{NarrowFloat, Ty, Value};

/// The environment variable that overrides the recorded data directory.
pub const DATA_DIR_ENV: &str = "MAPAL_DATA_DIR";

/// The exit code of a failed read (traps exit 101, GPU runtime failures 102).
pub const EXIT_CODE: i32 = 103;

/// A `read` target: an array or a product (struct, tuple) whose every leaf
/// is a readable scalar ([`elem_bytes`]), at any depth — arrays of arrays,
/// arrays of structs, structs of arrays — laid out with no padding (each
/// leaf at an offset that is a multiple of its own width, every product's
/// size a multiple of its widest leaf): the file is the value's bytes, leaves
/// in order, which is also how the value is stored. A bare scalar is not a
/// target (a file holds an array or a record).
pub fn readable(ty: &Ty) -> bool {
    matches!(ty, Ty::Array { .. } | Ty::Tuple(_) | Ty::Struct { .. }) && layout(ty).is_some()
}

/// `(bytes, widest leaf)` of a readable value with no padding, else `None`.
fn layout(ty: &Ty) -> Option<(u64, u64)> {
    match ty {
        Ty::Array { elem, size } => {
            let (b, a) = layout(elem)?;
            (*size >= 1).then_some(())?;
            Some((b.checked_mul(*size)?, a))
        }
        Ty::Tuple(_) | Ty::Struct { .. } => {
            let comps: Vec<&Ty> = match ty {
                Ty::Tuple(ts) => ts.iter().collect(),
                Ty::Struct { fields, .. } => fields.iter().map(|(_, t)| t).collect(),
                _ => unreachable!("a product"),
            };
            let (mut at, mut widest) = (0u64, 0u64);
            for c in comps {
                let (b, a) = layout(c)?;
                (at % a == 0).then_some(())?;
                at = at.checked_add(b)?;
                widest = widest.max(a);
            }
            (widest > 0 && at % widest == 0).then_some((at, widest))
        }
        scalar => {
            let w = elem_bytes(scalar)?;
            Some((w, w))
        }
    }
}

/// The bytes one element of `elem` occupies in a file, when it is readable.
pub fn elem_bytes(elem: &Ty) -> Option<u64> {
    match elem {
        Ty::Int {
            bits: 32,
            signed: true,
        }
        | Ty::Float { bits: 32 } => Some(4),
        Ty::Int {
            bits: 64,
            signed: true,
        }
        | Ty::Float { bits: 64 } => Some(8),
        Ty::Int {
            bits: 8,
            signed: false,
        } => Some(1),
        Ty::Narrow(NarrowFloat::Bf16 | NarrowFloat::F16) => Some(2),
        _ => None,
    }
}

/// The exact file length a `read` of `ty` requires: its leaves' bytes
/// (`n · |T|` for an array of scalars).
pub fn file_bytes(ty: &Ty) -> Option<u64> {
    readable(ty).then(|| layout(ty).map(|(b, _)| b))?
}

/// The leaves of a readable `ty` in file order, as `(scalar type, count)`
/// runs: an array of scalars is one run, an array of products repeats its
/// element's runs.
pub fn leaf_runs(ty: &Ty, out: &mut Vec<(Ty, u64)>) {
    match ty {
        Ty::Array { elem, size } if elem_bytes(elem).is_some() => {
            out.push(((**elem).clone(), *size))
        }
        Ty::Array { elem, size } => {
            for _ in 0..*size {
                leaf_runs(elem, out);
            }
        }
        Ty::Tuple(ts) => ts.iter().for_each(|t| leaf_runs(t, out)),
        Ty::Struct { fields, .. } => fields.iter().for_each(|(_, t)| leaf_runs(t, out)),
        scalar => out.push((scalar.clone(), 1)),
    }
}

/// Resolve the path literal `path` (D5): absolute as written; otherwise
/// joined to `$MAPAL_DATA_DIR` when it is set and non-empty, else to
/// `data_dir` (the source directory recorded at compile time), else left
/// relative (the working directory).
pub fn resolve(path: &str, data_dir: Option<&str>) -> PathBuf {
    let env = std::env::var(DATA_DIR_ENV).ok();
    resolve_with(path, env.as_deref(), data_dir)
}

/// [`resolve`] with the environment value given — the rule itself, which
/// the runtime's tests replay case by case.
pub fn resolve_with(path: &str, env: Option<&str>, data_dir: Option<&str>) -> PathBuf {
    let p = Path::new(path);
    if p.is_absolute() {
        return p.to_path_buf();
    }
    match env.filter(|e| !e.is_empty()).or(data_dir) {
        Some(base) => Path::new(base).join(p),
        None => p.to_path_buf(),
    }
}

/// The message of a read whose file has the wrong length.
pub fn size_message(path: &Path, expected: u64, actual: u64) -> String {
    format!(
        "mapal: read failed: {}: expected {expected} bytes, found {actual}",
        path.display()
    )
}

/// The message of a read whose file cannot be opened or read.
pub fn open_message(path: &Path, err: &std::io::Error) -> String {
    format!("mapal: read failed: {}: {err}", path.display())
}

/// The oracle's read: the file's bytes, or the failure's message.
pub fn load(path: &str, data_dir: Option<&str>, expected: u64) -> Result<Vec<u8>, String> {
    let full = resolve(path, data_dir);
    let meta = std::fs::metadata(&full).map_err(|e| open_message(&full, &e))?;
    if meta.is_dir() {
        let e = std::io::Error::from(std::io::ErrorKind::IsADirectory);
        return Err(open_message(&full, &e));
    }
    if meta.len() != expected {
        return Err(size_message(&full, expected, meta.len()));
    }
    let bytes = std::fs::read(&full).map_err(|e| open_message(&full, &e))?;
    if bytes.len() as u64 != expected {
        return Err(size_message(&full, expected, bytes.len() as u64));
    }
    Ok(bytes)
}

/// The elements of `bytes` as values of `elem` (little-endian).
pub fn decode(bytes: &[u8], elem: &Ty) -> Vec<Value> {
    let w = elem_bytes(elem).expect("a readable element type") as usize;
    bytes
        .chunks_exact(w)
        .map(|c| match elem {
            Ty::Int { bits: 32, .. } => Value::I32(i32::from_le_bytes(c.try_into().unwrap())),
            Ty::Int { bits: 64, .. } => Value::I64(i64::from_le_bytes(c.try_into().unwrap())),
            Ty::Int { bits: 8, .. } => Value::U8(c[0]),
            Ty::Float { bits: 32 } => Value::F32(f32::from_le_bytes(c.try_into().unwrap())),
            Ty::Float { bits: 64 } => Value::F64(f64::from_le_bytes(c.try_into().unwrap())),
            Ty::Narrow(n) => Value::Narrow(*n, u16::from_le_bytes(c.try_into().unwrap())),
            _ => unreachable!("decode of an unreadable element type"),
        })
        .collect()
}

/// The little-endian bytes of `values` — the file a test or a generator
/// writes for a `read` to load back.
pub fn encode(values: &[Value]) -> Vec<u8> {
    let mut out = Vec::new();
    for v in values {
        match v {
            Value::I32(x) => out.extend_from_slice(&x.to_le_bytes()),
            Value::I64(x) => out.extend_from_slice(&x.to_le_bytes()),
            Value::U8(x) => out.push(*x),
            Value::F32(x) => out.extend_from_slice(&x.to_le_bytes()),
            Value::F64(x) => out.extend_from_slice(&x.to_le_bytes()),
            Value::Narrow(_, b) => out.extend_from_slice(&b.to_le_bytes()),
            Value::Bool(_) | Value::Str(_) | Value::Unit => panic!("encode of an unreadable value"),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readable_types() {
        let arr = |e: Ty| Ty::Array {
            elem: Box::new(e),
            size: 3,
        };
        for e in [
            Ty::i32(),
            Ty::i64(),
            Ty::u8(),
            Ty::f32(),
            Ty::f64(),
            Ty::Narrow(NarrowFloat::Bf16),
            Ty::Narrow(NarrowFloat::F16),
        ] {
            assert!(readable(&arr(e)));
        }
        assert!(!readable(&arr(Ty::Bool)));
        assert!(!readable(&arr(Ty::Narrow(NarrowFloat::F8e4m3))));
        assert!(!readable(&Ty::f32()));
        assert_eq!(file_bytes(&arr(Ty::f64())), Some(24));
        // Nested values: the file is the value's leaves, in order, unpadded.
        assert!(readable(&arr(arr(Ty::f32()))));
        assert_eq!(file_bytes(&arr(arr(Ty::f32()))), Some(36));
        assert!(readable(&arr(Ty::Tuple(vec![Ty::f32(), Ty::f32()]))));
        let layer = Ty::Struct {
            name: "Layer".into(),
            fields: vec![
                ("gain".into(), arr(Ty::Narrow(NarrowFloat::Bf16))),
                ("weight".into(), arr(arr(Ty::Narrow(NarrowFloat::Bf16)))),
            ],
        };
        assert_eq!(file_bytes(&layer), Some(6 + 18));
        assert_eq!(file_bytes(&arr(layer.clone())), Some(3 * 24));
        let mut runs = Vec::new();
        leaf_runs(&arr(layer), &mut runs);
        assert_eq!(runs.len(), 3 * 4);
        // A leaf off its width's alignment (an f32 after one u8), or a
        // record whose size is not a multiple of its widest leaf: padding.
        assert!(!readable(&Ty::Tuple(vec![Ty::u8(), Ty::f32()])));
        assert!(!readable(&arr(Ty::Tuple(vec![Ty::f32(), Ty::u8()]))));
        assert!(readable(&Ty::Tuple(vec![
            Ty::f32(),
            Ty::u8(),
            Ty::u8(),
            Ty::u8(),
            Ty::u8()
        ])));
        assert!(!readable(&arr(Ty::Bool)));
    }

    #[test]
    fn resolution() {
        assert_eq!(
            resolve_with("/abs/w.bin", Some("/env"), Some("/src")),
            PathBuf::from("/abs/w.bin")
        );
        assert_eq!(
            resolve_with("w/x.bin", Some("/env"), Some("/src")),
            PathBuf::from("/env/w/x.bin")
        );
        assert_eq!(
            resolve_with("w/x.bin", Some(""), Some("/src")),
            PathBuf::from("/src/w/x.bin")
        );
        assert_eq!(
            resolve_with("w/x.bin", None, Some("/src")),
            PathBuf::from("/src/w/x.bin")
        );
        assert_eq!(
            resolve_with("w/x.bin", None, None),
            PathBuf::from("w/x.bin")
        );
    }

    #[test]
    fn round_trip() {
        let vs = vec![Value::F32(1.5), Value::F32(-0.0), Value::F32(f32::INFINITY)];
        let b = encode(&vs);
        assert_eq!(b.len(), 12);
        assert_eq!(decode(&b, &Ty::f32()), vs);
        let n = vec![Value::Narrow(NarrowFloat::Bf16, 0x3f80)];
        assert_eq!(decode(&encode(&n), &Ty::Narrow(NarrowFloat::Bf16)), n);
    }
}
