//! One-line descriptions of the names a file does not define: the stage
//! builtins and the primitive types (hover and completion details).

/// The builtin's shape and meaning, or `None` for a name that is no builtin.
pub fn builtin(name: &str) -> Option<&'static str> {
    Some(match name {
        "print" => "`x -> print` — write the value (no newline); an effect (ADR-0015)",
        "println" => "`x -> println` — write the value and a newline; an effect (ADR-0015)",
        "time" => {
            "`() -> time` — read the clock (a later reading minus an earlier one is the elapsed time); an effect"
        }
        "read" => {
            "`\"file.bin\" -> read -> name: T` — read a file's bytes as `T`, the path relative to the \
             source file (or `MAPAL_DATA_DIR`); an effect (ADR-0045)"
        }
        "zip" => "`(a, b) -> zip` — the arrays' elements paired (ADR-0018)",
        "enumerate" => "`a -> enumerate` — each element with its index (ADR-0018)",
        "iota" => "`n -> iota` — the indices `[0, 1, …, n - 1]` (ADR-0031)",
        "fill" => "`(value, n) -> fill` — `n` copies of `value` (ADR-0031)",
        "exp" => "`x -> exp` — e^x, within its stated bound (ADR-0042)",
        "sqrt" => "`x -> sqrt` — the square root (ADR-0046)",
        "sin" => "`x -> sin` — the sine, radians (ADR-0046)",
        "cos" => "`x -> cos` — the cosine, radians (ADR-0046)",
        "to" => {
            "`x -> to<target>` — the value rounded to nearest of `target` (ADR-0051 D4: `f32` → a \
             narrow format, ADR-0043's contract, fp8 saturating; an `int` → a `float`); `target` \
             may be inferred from the binding's type"
        }
        "widen" => {
            "`x -> widen<target>` — the value as `target`, exactly (ADR-0051 D4: every value of the \
             source is one of `target`, or the facts prove it is); `target` may be inferred"
        }
        "widen_i64" => "removed (ADR-0051 D6): write `widen<i64>`",
        "widen_f32" => {
            "removed (ADR-0051 D6): write `widen<f32>` (or `to<f32>` where it may round)"
        }
        "widen_f64" => "removed (ADR-0051 D6): write `widen<f64>`",
        "to_bf16" => "removed (ADR-0051 D6): write `to<bf16>`",
        "to_f16" => "removed (ADR-0051 D6): write `to<f16>`",
        "to_f8e4m3" => "removed (ADR-0051 D6): write `to<f8e4m3>`",
        "to_f8e5m2" => "removed (ADR-0051 D6): write `to<f8e5m2>`",
        _ => return None,
    })
}

pub fn primitive(name: &str) -> Option<&'static str> {
    Some(match name {
        "i32" => "32-bit signed integer",
        "i64" => "64-bit signed integer",
        "u8" => "8-bit unsigned integer",
        "f32" => "32-bit IEEE float",
        "f64" => "64-bit IEEE float",
        "bf16" => "bfloat16 storage (ADR-0043); also a `: compute` member (ADR-0041)",
        "f16" => "IEEE half storage (ADR-0043); also a `: compute` member",
        "f8e4m3" => "fp8 e4m3 storage (ADR-0043)",
        "f8e5m2" => "fp8 e5m2 storage (ADR-0043)",
        "bool" => "`true` or `false`",
        "tf32" => "a `: compute` member: products may round their multiplicands to tf32 (ADR-0041)",
        _ => return None,
    })
}
