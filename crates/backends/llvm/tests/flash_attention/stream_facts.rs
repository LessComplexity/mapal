//! The stream fact on attention programs, streamed values and pricing on every part.

use super::*;

/// `pt_attn_1024_64`: one stream group over the scores' axis — the scores
/// produced window by window, the row max a fold whose running state is the
/// shift `ex` reads (a rescale), the sum a fold rescaled by it, `p` reading
/// the sum as a quotient factor, `P·V` consuming the windows with both laws;
/// the scores and probabilities dissolved. The fact arm off: no group; the
/// rescale off (`exp_rescale`): the max is a phase, refused.
#[test]
fn stream_fact_on_attention() {
    let ir = lower(&program("pt_attn_1024_64"));
    let f = ir.entry();
    let sp = ir.stream_plan(f);
    assert_eq!(sp.groups.len(), 1, "{sp:?}");
    let g = &sp.groups[0];
    assert_eq!(g.axis, 1024);
    assert_eq!(g.passes, 1);
    let unit = &ir.unit_plan(f).groups[g.group];
    let name = |k: usize| {
        ir.object(unit.members[k].out)
            .and_then(|o| o.name.clone())
            .unwrap_or_default()
    };
    let roles: Vec<(String, &StreamRole)> = g
        .roles
        .iter()
        .enumerate()
        .map(|(k, r)| (name(k), r))
        .collect();
    let role = |n: &str| roles.iter().find(|(x, _)| x == n).map(|x| x.1).unwrap();
    assert_eq!(role("s"), &StreamRole::Produce);
    let k_mx = roles.iter().position(|(x, _)| x == "mx").unwrap();
    let k_den = roles.iter().position(|(x, _)| x == "den").unwrap();
    match role("ex") {
        StreamRole::Cell { subst } => assert_eq!(subst[..], [(1, k_mx, DepLaw::Rescale)]),
        r => panic!("{r:?}"),
    }
    match role("p") {
        StreamRole::Cell { subst } => {
            assert_eq!(subst[..], [(1, k_den, DepLaw::Factor { div: true })])
        }
        r => panic!("{r:?}"),
    }
    match role("den") {
        StreamRole::Fold { laws } => {
            assert_eq!(laws.len(), 1);
            assert_eq!((laws[0].value, laws[0].law), (k_mx, DepLaw::Rescale));
        }
        r => panic!("{r:?}"),
    }
    match role("o") {
        StreamRole::Consume { laws } => {
            let l: Vec<(usize, DepLaw)> = laws.iter().map(|l| (l.value, l.law)).collect();
            assert!(l.contains(&(k_mx, DepLaw::Rescale)), "{l:?}");
            assert!(l.contains(&(k_den, DepLaw::Factor { div: true })), "{l:?}");
        }
        r => panic!("{r:?}"),
    }
    let dissolved: Vec<String> = g.dissolved.iter().map(|&k| name(k)).collect();
    assert_eq!(dissolved, ["s", "ex", "p"]);
    assert_eq!(g.dissolved_bytes, 3 * 1024 * 1024 * 4);
    // Stored: the second query reads it.
    assert_eq!(ir.stream_plan(f), sp);
    // The fact arm off, and the rescale off.
    let off = ir.refacted(mapal_ir::FactArms {
        stream: false,
        ..ir.fact_arms()
    });
    assert!(off.stream_plan(off.entry()).groups.is_empty());
    let off = ir.refacted(mapal_ir::FactArms {
        exp_rescale: false,
        ..ir.fact_arms()
    });
    let sp = off.stream_plan(off.entry());
    assert!(sp.groups.is_empty());
    assert_eq!(
        sp.refused.iter().map(|r| r.1).collect::<Vec<_>>(),
        ["phase"]
    );
}

/// The streamed unit's emission and values on `pt_attn_1024_64` scaled down
/// (`px_attn_64_16`'s shape): one kernel where there were three, nothing
/// axis-wide stored, the values the interpreter's within the bound in every
/// emulator mode; the OFF arm (`--gpu-off=stream`) is the split emission.
#[test]
fn streamed_attention_values() {
    let src = program("px_attn_64_16");
    let ir = lower(&src);
    let rr = run(&ir, 2_000_000_000);
    let Outcome::Done(_) = rr.outcome else {
        panic!("interp")
    };
    let on = emission(&ir, Some((16, 16, 0, 0, 0)), &[]);
    let lines = stream_lines(&on);
    assert_eq!(lines.len(), 1, "{}", on.device.as_deref().unwrap_or(""));
    assert_eq!(field(&lines[0], "store"), "o");
    assert_eq!(field(&lines[0], "dissolved"), "s,ex,p");
    assert_eq!(field(&lines[0], "windows"), "4");
    let off = emission(&ir, Some((16, 16, 0, 0, 0)), &["stream"]);
    assert!(stream_lines(&off).is_empty());
    let Some(clang) = emu::clang() else {
        return;
    };
    for em in [&on, &off] {
        let ex = emu::build_emulated(&clang, &em.text, em.device.as_deref().unwrap_or(""));
        for (mode, (got, _)) in MODES.iter().zip(ex.run_each(&MODES)) {
            assert!(
                within(&got, &rr.output),
                "{mode:?}: {got:?} vs {:?}",
                rr.output
            );
        }
    }
}

/// review-fixes: **the stream is priced on every part** — a part whose
/// profile measured only some plan rates (the 3090, A100, H100 SXM and PCIe)
/// prices the streamed unit on the rates its properties derive
/// (`TargetProfile::priced`) instead of keeping the split realization
/// unweighed (`stream=s:unpriced`, sweep-attention's replay). The 2048-token
/// prefill streams on each by its price; the scaled-down attention's values
/// under each part's own rule match the interpreter in every emulator mode.
#[test]
fn stream_priced_on_every_part() {
    let on = |ir: &CategoryIr, target: &'static str| {
        let mut gpu = GpuArms::default();
        gpu.on.flow = true;
        for r in ["place", "pdl", "streams", "pinned", "horizontal", "calls"] {
            *gpu.off.rung(r).expect("rung") = true;
        }
        let opts = EmitOpts {
            host: "generic",
            target,
            gpu,
            ..EmitOpts::default()
        };
        emit_detailed(ir, &opts).expect("emits")
    };
    let pre = lower(&program("pt_llm_pre_t2048"));
    // attention-chains: the window's products and their shared loads priced
    // half in series at the broadcast-load rate — the A100's 64 FP32 lanes
    // no longer hide the loads of a one-row-a-thread geometry (measured 1.5×
    // a four-row one): its streams hold four rows a thread, as the GDDR
    // parts' did.
    let a128 = lower(&program("pt_attn_2048_128"));
    for (ir, part) in [(&pre, "a100"), (&a128, "a100")] {
        let em = on(ir, part);
        let l = stream_lines(&em);
        assert_eq!(l.len(), 1, "{part}: {}", em.device.as_deref().unwrap_or(""));
        assert!(field(&l[0], "stream").contains(":rt4"), "{part}: {}", l[0]);
    }
    let small = lower(&program("px_attn_64_16"));
    let rr = run(&small, 2_000_000_000);
    let Outcome::Done(_) = rr.outcome else {
        panic!("interp")
    };
    let clang = emu::clang();
    for part in ["rtx-3090", "a100", "h100", "h100-pcie"] {
        let em = on(&pre, part);
        let dev = em.device.as_deref().unwrap_or("");
        assert!(!dev.contains("stream=s:unpriced"), "{part}");
        assert_eq!(stream_lines(&em).len(), 1, "{part}: the prefill streams");
        let Some(clang) = clang.as_deref() else {
            continue;
        };
        let em = on(&small, part);
        let ex = emu::build_emulated(clang, &em.text, em.device.as_deref().unwrap_or(""));
        for (mode, (got, _)) in MODES.iter().zip(ex.run_each(&MODES)) {
            assert!(
                within(&got, &rr.output),
                "{part} {mode:?}: {got:?} vs {:?}",
                rr.output
            );
        }
    }
}
