//! **The scratchpad latency probe measures one use's latency alone**
//! (cost-rental review): `place_calib.cu`'s `lds_chase_k<KIND>` — the
//! `ld.shared.u32`, `ld.shared.v4.u32` and `ldmatrix.x4` one-use latencies
//! every part's facts carry — compiled for each part's SASS family is a pure
//! pointer chase: each load's address register is the previous load's
//! destination, no instruction between them writes it (no branch on the
//! kind, no address arithmetic), the wide loads stay wide, and the loop is
//! unrolled. The old probe branched on a run-time `kind` inside the chain
//! and moved the address by `v * zero` (an `IMAD` a step), and its
//! `lane * 16` words put a 4-way bank conflict on `ld.shared.u32`: the
//! RTX 5080 measured 68–69 clocks where the chase measures 28–40.
//!
//! Skipped where `nvcc` or `cuobjdump` is missing (the CUDA toolkit).

use std::path::Path;
use std::process::Command;

/// A SASS line's instruction text (between its offset and its `;`).
fn instr(line: &str) -> Option<&str> {
    Some(
        line.split("/*")
            .nth(1)?
            .split_once("*/")?
            .1
            .split(';')
            .next()?
            .trim(),
    )
}

/// `(opcode, destination, address)` of a SASS line's shared-memory load.
fn shared_load(line: &str) -> Option<(String, String, String)> {
    let (op, rest) = instr(line)?.split_once(' ')?;
    if !op.starts_with("LDS") {
        return None;
    }
    let (dst, addr) = rest.split_once(',')?;
    let addr = addr.trim().strip_prefix('[')?.strip_suffix(']')?;
    Some((
        op.to_string(),
        dst.trim().to_string(),
        addr.trim().to_string(),
    ))
}

/// The destination register of a SASS instruction line (its first operand,
/// past a predicate).
fn dest(line: &str) -> Option<String> {
    let mut words = instr(line)?.split_whitespace();
    let mut op = words.next()?;
    if op.starts_with('@') {
        op = words.next()?;
    }
    let _ = op;
    Some(words.next()?.trim_end_matches(',').to_string())
}

#[test]
fn place_calib_chase_is_pure() {
    let have = |t: &str| Command::new(t).arg("--version").output().is_ok();
    if !have("nvcc") || !have("cuobjdump") {
        eprintln!("skipped: nvcc / cuobjdump missing");
        return;
    }
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../benches/local/place_calib.cu");
    let dir = std::env::temp_dir().join(format!("mapal-chase-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // The SASS families of the parts whose facts carry the probe's rows.
    for arch in ["sm_80", "sm_86", "sm_89", "sm_90", "sm_120"] {
        let cubin = dir.join(format!("{arch}.cubin"));
        let out = Command::new("nvcc")
            .args([
                "-cubin",
                &format!("-arch={arch}"),
                "-std=c++17",
                "-diag-suppress",
                "550",
            ])
            .arg(&src)
            .arg("-o")
            .arg(&cubin)
            .output()
            .expect("nvcc runs");
        assert!(
            out.status.success(),
            "{arch}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let sass = Command::new("cuobjdump")
            .arg("-sass")
            .arg(&cubin)
            .output()
            .expect("cuobjdump runs");
        let sass = String::from_utf8_lossy(&sass.stdout).to_string();
        // Each instantiation's function text.
        let mut kinds = 0;
        for f in sass.split("Function : ").skip(1) {
            let name = f.lines().next().unwrap_or("");
            let Some(kind) = ["ILi0E", "ILi1E", "ILi2E"]
                .iter()
                .position(|k| name.contains("lds_chase_k") && name.contains(k))
            else {
                continue;
            };
            kinds += 1;
            let want = ["LDS", "LDS.128", "LDSM.16.M88.4"][kind];
            let lines: Vec<&str> = f.lines().collect();
            let loads: Vec<(usize, (String, String, String))> = lines
                .iter()
                .enumerate()
                .filter_map(|(i, l)| shared_load(l).map(|x| (i, x)))
                .collect();
            assert!(
                loads.len() >= 8,
                "{arch} {name}: the chase is unrolled ({} loads)",
                loads.len()
            );
            let mut chained = 0;
            for w in loads.windows(2) {
                let ((i, (op, d, _)), (j, (_, _, a))) = (&w[0], &w[1]);
                assert_eq!(op, want, "{arch} {name}: the load is the probe's own width");
                let between = &lines[i + 1..*j];
                if between.iter().any(|l| l.contains("BRA")) {
                    continue; // a block boundary (the unrolled body's ends)
                }
                assert_eq!(
                    a, d,
                    "{arch} {name}: the next address is the loaded word ({})",
                    lines[*j]
                );
                assert!(
                    between
                        .iter()
                        .all(|l| dest(l).as_deref() != Some(d.as_str())),
                    "{arch} {name}: nothing between two uses writes the chain's register"
                );
                chained += 1;
            }
            assert!(chained >= 7, "{arch} {name}: a chain of loads ({chained})");
        }
        assert_eq!(kinds, 3, "{arch}: the three chase kernels");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
