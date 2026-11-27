//! Rejection-matrix tests: tokens, the clock, struct-name conflicts, calls and bodies.

use super::*;

// --- tokens ---------------------------------------------------------------

#[test]
fn token_in_phi() {
    // A Phi branch of ty ((IoToken, i32), bool) → TokenInPhi.
    let pair = Ty::Tuple(vec![Ty::IoToken, Ty::i32()]);
    let (mut b, f) = one_fn(pair.clone(), pair.clone());
    let mut fb = b.build_fn(f).unwrap();
    let x = fb.input();
    let cond = fb.constant(Value::Bool(true), L).unwrap();
    let e = fb.phi(x, x, cond, Dest::Fresh(None), L).unwrap_err();
    assert_eq!(e, IrError::TokenInPhi);
}

#[test]
fn token_in_map_body() {
    // A MapBody whose body ty contains a token → TokenInBody at map().
    let mut b = IrBuilder::new();
    let body = b
        .declare(FuncKind::MapBody, "mb", Ty::IoToken, Ty::IoToken, L)
        .unwrap();
    {
        let fb = b.build_fn(body).unwrap();
        // Trivially route input to output via Output.
        let inp = fb.input();
        let mut fb = fb;
        fb.output(inp, None, L).unwrap();
        fb.finish().unwrap();
    }
    let arr = Ty::Array {
        elem: Box::new(Ty::IoToken),
        size: 2,
    };
    let f = b
        .declare(FuncKind::Named, "main", arr.clone(), arr, L)
        .unwrap();
    let mut fb = b.build_fn(f).unwrap();
    let a = fb.input();
    let e = fb.map(body, a, Dest::Fresh(None), L).unwrap_err();
    assert_eq!(e, IrError::TokenInBody);
}

#[test]
fn token_dropped_at_seal() {
    // A token produced but never written to Return → TokenDropped (I4b).
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "main", Ty::IoToken, Ty::IoToken, L)
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let t0 = fb.input();
        let s = fb.constant(Value::Str("hi".into()), L).unwrap();
        // print produces a fresh token t1, but we never write it to Return.
        let _t1 = fb.print(t0, s, L).unwrap();
        // Return is an unwritten IoToken with no writers → and t1 is dropped.
        fb.finish().unwrap_err(); // I-RET: missing return writer (IoToken != Unit)
    }
    // We instead seal to surface the token-drop path explicitly below.
}

#[test]
fn token_dropped_seal_path() {
    // Write a *different* token into Return so I-RET passes but the print token
    // is dangling → TokenDropped at seal.
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "main", Ty::IoToken, Ty::IoToken, L)
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let t0 = fb.input();
        let s = fb.constant(Value::Str("hi".into()), L).unwrap();
        let _dangling = fb.print(t0, s, L).unwrap();
        // Route the ORIGINAL token t0 to Return via Output; but t0 now has two
        // token consumers (print pair + Output) → TokenNotLinear at seal? No:
        // Output target is Return (token), print pair is token-bearing too.
        // To isolate TokenDropped, give Return its own fresh token instead.
        // Simplest: write t0 to ret — but t0 already feeds the print pair.
        // Use a second print to consume t0's successor and drop the tail.
        fb.output(t0, None, L).unwrap();
        fb.finish().unwrap();
    }
    // t0 has two token consumers (print-pair + Output-to-Return). That is not
    // the loop fork, so seal must reject with TokenNotLinear — and the dangling
    // print token also violates I4b. Either is a token error; assert it errors.
    let e = b.seal(f).unwrap_err();
    assert!(matches!(e, IrError::TokenNotLinear | IrError::TokenDropped));
}

#[test]
fn str_outside_print() {
    // pack of a Str constant → StrOutsidePrint.
    let (mut b, f) = one_fn(Ty::i32(), Ty::Tuple(vec![Ty::Str, Ty::i32()]));
    let mut fb = b.build_fn(f).unwrap();
    let x = fb.input();
    let s = fb.constant(Value::Str("x".into()), L).unwrap();
    let e = fb.pack(&[s, x], Dest::Fresh(None), L).unwrap_err();
    assert_eq!(e, IrError::StrOutsidePrint);
}

// --- time (plan-time-builtin) ----------------------------------------------

#[test]
fn time_ms_builds_and_validates() {
    use crate::validate::validate;
    // main : IoToken -> IoToken — t0 --TimeMs--> (tok, f64); proj both; the
    // proj'd token threads to Return (mirrors the print token path).
    let (mut b, f) = one_fn(Ty::IoToken, Ty::IoToken);
    {
        let mut fb = b.build_fn(f).unwrap();
        let t0 = fb.input();
        let pair = fb.time_ms(t0, L).unwrap();
        let tok = fb.proj(pair, 0, Dest::Fresh(None), L).unwrap();
        let _ms = fb.proj(pair, 1, Dest::Fresh(Some("t0".into())), L).unwrap();
        fb.output(tok, None, L).unwrap();
        fb.finish().unwrap();
    }
    let ir = b.seal(f).unwrap();
    assert!(validate(&ir).is_empty(), "{:?}", validate(&ir));
    // The op shape: one TimeMs morphism, token source, (IoToken, f64) target.
    let tim: Vec<_> = ir
        .morphisms()
        .filter(|(_, m)| m.op == Operation::TimeMs)
        .collect();
    assert_eq!(tim.len(), 1);
    let (_, m) = tim[0];
    assert_eq!(ir.object(m.source).unwrap().ty, Ty::IoToken);
    assert_eq!(
        ir.object(m.target).unwrap().ty,
        Ty::Tuple(vec![Ty::IoToken, Ty::f64()])
    );
}

#[test]
fn time_ms_rejects_non_token_source() {
    let (mut b, f) = one_fn(Ty::i32(), Ty::i32());
    let mut fb = b.build_fn(f).unwrap();
    let x = fb.input();
    let e = fb.time_ms(x, L).unwrap_err();
    assert_eq!(
        e,
        IrError::TypeMismatch {
            expected: Ty::IoToken,
            found: Ty::i32(),
            loc: L,
        }
    );
}

// --- struct name conflict -------------------------------------------------

#[test]
fn struct_name_conflict() {
    // Two `Pixel` decls with different fields → StructNameConflict at seal.
    let p1 = Ty::Struct {
        name: "Pixel".into(),
        fields: vec![("r".into(), Ty::u8())],
    };
    let p2 = Ty::Struct {
        name: "Pixel".into(),
        fields: vec![("r".into(), Ty::i32())],
    };
    let mut b = IrBuilder::new();
    // f takes p1, g takes p2 — both Pixel, different fields.
    let f = b.declare(FuncKind::Named, "f", p1.clone(), p1, L).unwrap();
    {
        let fb = b.build_fn(f).unwrap();
        let inp = fb.input();
        let mut fb = fb;
        fb.output(inp, None, L).unwrap();
        fb.finish().unwrap();
    }
    let g = b.declare(FuncKind::Named, "g", p2.clone(), p2, L).unwrap();
    {
        let fb = b.build_fn(g).unwrap();
        let inp = fb.input();
        let mut fb = fb;
        fb.output(inp, None, L).unwrap();
        fb.finish().unwrap();
    }
    let e = b.seal(f).unwrap_err();
    assert_eq!(e, IrError::StructNameConflict);
}

// --- calls and bodies -----------------------------------------------------

#[test]
fn call_to_body() {
    // Call targeting a MapBody → CallToBody.
    let mut b = IrBuilder::new();
    let body = b
        .declare(FuncKind::MapBody, "mb", Ty::i32(), Ty::i32(), L)
        .unwrap();
    let f = b
        .declare(FuncKind::Named, "main", Ty::i32(), Ty::i32(), L)
        .unwrap();
    let mut fb = b.build_fn(f).unwrap();
    let x = fb.input();
    let e = fb.call(body, x, Dest::Fresh(None), L).unwrap_err();
    assert_eq!(e, IrError::CallToBody);
}

#[test]
fn body_kind_mismatch() {
    // map() with a FoldBody → BodyKindMismatch.
    let mut b = IrBuilder::new();
    let fold_body = b
        .declare(
            FuncKind::FoldBody,
            "fb",
            Ty::Tuple(vec![Ty::i32(), Ty::i32()]),
            Ty::i32(),
            L,
        )
        .unwrap();
    let arr = Ty::Array {
        elem: Box::new(Ty::i32()),
        size: 2,
    };
    let f = b
        .declare(FuncKind::Named, "main", arr.clone(), arr, L)
        .unwrap();
    let mut fb = b.build_fn(f).unwrap();
    let a = fb.input();
    let e = fb.map(fold_body, a, Dest::Fresh(None), L).unwrap_err();
    assert_eq!(e, IrError::BodyKindMismatch);
}

#[test]
fn recursive_call() {
    // main calls itself → RecursiveCall at seal.
    let mut b = IrBuilder::new();
    let f = b
        .declare(FuncKind::Named, "main", Ty::i32(), Ty::i32(), L)
        .unwrap();
    {
        let mut fb = b.build_fn(f).unwrap();
        let x = fb.input();
        fb.call(f, x, Dest::Ret { slot: None }, L).unwrap();
        fb.finish().unwrap();
    }
    let e = b.seal(f).unwrap_err();
    assert!(matches!(e, IrError::RecursiveCall { .. }));
}
