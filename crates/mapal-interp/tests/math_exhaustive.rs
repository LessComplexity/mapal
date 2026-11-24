//! ADR-0046 D3/D4, measured exhaustively (this crate's tests build at
//! `opt-level = 3`): every f32 through `S_32` / `C_32` against the f64
//! truth — the default face's bound everywhere and the worst ulps — and
//! every non-negative f32 through `sqrt_def` (the plain-operation digit
//! recurrence) against the correctly rounded `sqrt`. Minutes on one core:
//! `#[ignore]`d, run with `cargo test -p mapal-interp --test math_exhaustive
//! -- --ignored` (eight threads).

use mapal_ir::math::{self, MathFn, Native};

fn chunks(f: impl Fn(u32, u32) -> (u64, f64) + Sync) -> (u64, f64) {
    let n = 8u64;
    let step = (1u64 << 32) / n;
    std::thread::scope(|s| {
        let hs: Vec<_> = (0..n)
            .map(|i| {
                let f = &f;
                s.spawn(move || f((i * step) as u32, ((i + 1) * step - 1) as u32))
            })
            .collect();
        hs.into_iter()
            .map(|h| h.join().unwrap())
            .fold((0u64, 0f64), |a, b| (a.0 + b.0, a.1.max(b.1)))
    })
}

#[test]
#[ignore]
fn sin_cos_every_f32_within_the_contract() {
    for f in [MathFn::Sin, MathFn::Cos] {
        let (bad, worst) = chunks(|lo, hi| {
            let (mut bad, mut worst) = (0u64, 0f64);
            for b in lo..=hi {
                let x = f32::from_bits(b);
                let mut n = Native::<32>;
                let got = f64::from(f32::from_bits(
                    math::reference(f, &mut n, u64::from(b)) as u32
                ));
                let t = math::truth(f, f64::from(x));
                if !math::within(f, 32, f64::from(x), got, t, 0.0) {
                    bad += 1;
                }
                if t.is_finite() && t != 0.0 && x.is_finite() && x != 0.0 {
                    let ulp = 2f64.powi(t.abs().log2().floor() as i32 - 23);
                    worst = worst.max((got - t).abs() / ulp);
                }
            }
            (bad, worst)
        });
        eprintln!("{f:?} every f32: outside the contract {bad}, worst {worst:.4} ulp");
        assert_eq!(bad, 0, "{f:?}");
        assert!(worst < 2.0, "{f:?} {worst}");
    }
}

#[test]
#[ignore]
fn sqrt_def_every_f32_is_correctly_rounded() {
    let (bad, _) = chunks(|lo, hi| {
        let mut bad = 0u64;
        for b in lo..=hi {
            let x = f32::from_bits(b);
            let mut n = Native::<32>;
            let got = f32::from_bits(math::sqrt_def(&mut n, u64::from(b)) as u32);
            let want = x.sqrt();
            if got.to_bits() != want.to_bits() && !(got.is_nan() && want.is_nan()) {
                bad += 1;
            }
        }
        (bad, 0.0)
    });
    eprintln!("sqrt_def every f32: {bad} differ from the correctly rounded sqrt");
    assert_eq!(bad, 0);
}
