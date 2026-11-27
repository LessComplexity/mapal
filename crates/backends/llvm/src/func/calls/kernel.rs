//! the persistent kernel text: grid barrier, declarations and `persistent_kernel`

/// **The grid barrier** between two levels of a persistent kernel (`calib.cu
/// gsync`'s measured form): the block meets, its thread 0 adds one arrival
/// to the kernel's counter with release semantics at GPU scope — block 0
/// adds `0x80000000 − (n − 1)`, every other block 1, so the counter's top bit
/// flips exactly when all `n` blocks arrived (its low 31 bits start at zero
/// and every complete barrier restores them) — and spins on an acquire load
/// until the bit differs from what it read; then the block meets again. The
/// acquire at GPU scope and the block meet make every write of the level
/// before visible to every thread of the level after (the PTX memory
/// model's release / acquire pattern, cumulative through `bar.sync`).
pub(crate) const GSYNC: &str = "define internal void @mapal_gsync(ptr addrspace(1) %c) alwaysinline {
entry:
  call void @llvm.nvvm.barrier0()
  %t = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %lead = icmp eq i32 %t, 0
  br i1 %lead, label %arrive, label %done
arrive:
  %b = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %n = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()
  %first = icmp eq i32 %b, 0
  %rest = sub i32 -2147483647, %n
  %inc = select i1 %first, i32 %rest, i32 1
  %old = call i32 asm sideeffect \"atom.add.release.gpu.global.u32 $0, [$1], $2;\", \"=r,l,r,~{memory}\"(ptr addrspace(1) %c, i32 %inc)
  br label %spin
spin:
  %cur = call i32 asm sideeffect \"ld.acquire.gpu.global.u32 $0, [$1];\", \"=r,l,~{memory}\"(ptr addrspace(1) %c)
  %x = xor i32 %old, %cur
  %flip = icmp slt i32 %x, 0
  br i1 %flip, label %done, label %spin
done:
  call void @llvm.nvvm.barrier0()
  ret void
}
";

/// The intrinsics a persistent kernel calls, declared when the module has
/// none of its own.
pub(super) const DECLS: [&str; 5] = [
    "declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()",
    "declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()",
    "declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()",
    "declare void @llvm.nvvm.barrier0()",
    "declare void @llvm.nvvm.barrier.sync.cnt(i32, i32)",
];

/// Each of [`DECLS`] the module `text` lacks.
pub(in crate::func) fn missing_decls(text: &str) -> String {
    DECLS
        .iter()
        .filter(|d| !text.contains(*d))
        .map(|d| format!("{d}\n"))
        .collect()
}

/// **The persistent kernel of `spec`** (module doc): level by level, each
/// zero pass due before the next level as grid-stride stores, then the
/// level's members over their concatenated block ranges — block `i` on
/// persistent block `i mod G`, each member a device function of its own
/// block coordinates (a member of fewer threads than the block on its own
/// warps, its barriers counted), a block barrier (id 1, apart from the
/// members' id 0) between two blocks' bodies: the next body reuses the
/// shared memory, and a padded member's idle warps wait there — then the
/// grid barrier. Parameters lose `noalias` / `readonly`: a later level reads,
/// through another parameter, what an earlier one wrote (never through the
/// non-coherent read-only path).
pub(in crate::func) fn persistent_kernel(
    spec: &super::super::horizontal::HzSpec,
    heads: &[&super::super::horizontal::Head],
) -> String {
    let b = spec.block;
    let mut params = Vec::new();
    for (i, h) in heads.iter().enumerate() {
        for (_, full, name) in &h.params {
            let stem = name.trim_start_matches('%');
            let at = full.len() - name.len();
            let ty = full[..at]
                .replace("noalias ", "")
                .replace("readonly ", "")
                .replace("writeonly ", "");
            params.push(format!("{ty}%h{i}.{stem}"));
        }
    }
    let minct = spec
        .members
        .iter()
        .zip(heads)
        .filter_map(|(m, h)| h.minctasm.map(|c| (c * m.3).div_ceil(b)))
        .max();
    let mut attrs = format!("\"nvvm.maxntid\"=\"{b}\"");
    if let Some(c) = minct {
        attrs += &format!(" \"nvvm.minctasm\"=\"{c}\"");
    }
    let levels = spec.level.iter().copied().max().unwrap_or(0) + 1;
    let members: Vec<String> = spec
        .members
        .iter()
        .zip(&spec.level)
        .map(|(m, l)| format!("{}:{}x{}x{}:l{l}", m.0, m.1, m.2, m.3))
        .collect();
    let name = &spec.name;
    let mut s = format!(
        "; mapal.plan kernel={name} kind=calls block={b} levels={levels} members={}\n\
         @{name}.gsync = internal addrspace(1) global i32 0, align 4\n\
         define ptx_kernel void @{name}({}) {attrs} {{\n\
         entry:\n\
         \x20 %cl.b = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n\
         \x20 %cl.t = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n\
         \x20 %cl.g = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()\n\
         \x20 %cl.gb = mul i32 %cl.g, {b}\n\
         \x20 %cl.b0 = mul i32 %cl.b, {b}\n\
         \x20 %cl.gt = add i32 %cl.b0, %cl.t\n\
         \x20 br label %cl.l0\n",
        members.join(","),
        params.join(", ")
    );
    for l in 0..levels {
        s += &format!("cl.l{l}:\n");
        let mut cur = format!("cl.l{l}");
        // The zero passes of the next level's atomic members.
        for (k, (i, (bytes, p))) in spec
            .zero
            .iter()
            .enumerate()
            .filter(|&(i, _)| spec.level[i] == l + 1)
            .filter_map(|(i, z)| z.map(|z| (i, z)))
            .enumerate()
        {
            let stem = heads[i].params[p].2.trim_start_matches('%');
            let (w, store) = match bytes {
                x if x % 16 == 0 => (
                    16,
                    "store <4 x i32> zeroinitializer, ptr addrspace(1) %P, align 16",
                ),
                x if x % 4 == 0 => (4, "store i32 0, ptr addrspace(1) %P, align 4"),
                _ => (1, "store i8 0, ptr addrspace(1) %P, align 1"),
            };
            let n = bytes / w;
            let t = format!("{l}_{k}");
            s += &format!(
                "  br label %cl.zh{t}\n\
                 cl.zh{t}:\n\
                 \x20 %cl.zi{t} = phi i32 [ %cl.gt, %{cur} ], [ %cl.zn{t}, %cl.zs{t} ]\n\
                 \x20 %cl.zc{t} = icmp ult i32 %cl.zi{t}, {n}\n\
                 \x20 br i1 %cl.zc{t}, label %cl.zs{t}, label %cl.ze{t}\n\
                 cl.zs{t}:\n\
                 \x20 %cl.zo{t} = zext i32 %cl.zi{t} to i64\n\
                 \x20 %cl.zb{t} = mul i64 %cl.zo{t}, {w}\n\
                 \x20 %cl.zp{t} = getelementptr i8, ptr addrspace(1) %h{i}.{stem}, i64 %cl.zb{t}\n\
                 \x20 {}\n\
                 \x20 %cl.zn{t} = add i32 %cl.zi{t}, %cl.gb\n\
                 \x20 br label %cl.zh{t}\n\
                 cl.ze{t}:\n",
                store.replace("%P", &format!("%cl.zp{t}"))
            );
            cur = format!("cl.ze{t}");
        }
        let lm: Vec<usize> = (0..spec.members.len())
            .filter(|&i| spec.level[i] == l)
            .collect();
        if !lm.is_empty() {
            let total: u64 = lm
                .iter()
                .map(|&i| spec.members[i].1 * spec.members[i].2)
                .sum();
            // The member selection for block index `idx` (labels and values
            // tagged `tag`), every path on to `next`.
            let dispatch = |tag: &str, idx: &str, next: &str| -> String {
                let mut d = String::new();
                let mut start = 0u64;
                for (k, &i) in lm.iter().enumerate() {
                    let (mname, gx, gy, bi) = &spec.members[i];
                    let end = start + gx * gy;
                    d += &format!("cl.m{tag}_{k}:\n");
                    if k + 1 < lm.len() {
                        d += &format!(
                            "  %cl.s{tag}_{k} = icmp ult i32 {idx}, {end}\n  br i1 %cl.s{tag}_{k}, label %cl.r{tag}_{k}, label %cl.m{tag}_{}\n",
                            k + 1
                        );
                    } else {
                        d += &format!("  br label %cl.r{tag}_{k}\n");
                    }
                    d += &format!("cl.r{tag}_{k}:\n");
                    if *bi < b {
                        d += &format!(
                            "  %cl.p{tag}_{k} = icmp ult i32 %cl.t, {bi}\n  br i1 %cl.p{tag}_{k}, label %cl.q{tag}_{k}, label %{next}\ncl.q{tag}_{k}:\n"
                        );
                    }
                    let args: Vec<String> = heads[i]
                        .params
                        .iter()
                        .map(|(ty, _, pn)| format!("{ty} %h{i}.{}", pn.trim_start_matches('%')))
                        .chain([
                            format!("i32 %cl.ux{tag}_{k}"),
                            format!("i32 %cl.uy{tag}_{k}"),
                        ])
                        .collect();
                    d += &format!(
                        "  %cl.o{tag}_{k} = sub i32 {idx}, {start}\n\
                         \x20 %cl.ux{tag}_{k} = urem i32 %cl.o{tag}_{k}, {gx}\n\
                         \x20 %cl.uy{tag}_{k} = udiv i32 %cl.o{tag}_{k}, {gx}\n\
                         \x20 call void @hz.{mname}({})\n\
                         \x20 br label %{next}\n",
                        args.join(", ")
                    );
                    start = end;
                }
                d
            };
            // The first block of the level straight-line (the common case:
            // a level of at most `G` blocks runs one each) — a body inside a
            // loop of unknown trips compiles to fewer loads in flight
            // (decode's rows: 10.8 us against 4.2 alone, `kernel-merging.md`
            // Calls) — then the loop over the rest, when the grid is smaller.
            s += &format!(
                "  %cl.c{l} = icmp ult i32 %cl.b, {total}\n\
                 \x20 br i1 %cl.c{l}, label %cl.mf{l}_0, label %cl.e{l}\n"
            );
            s += &dispatch(&format!("f{l}"), "%cl.b", &format!("cl.fx{l}"));
            s += &format!(
                "cl.fx{l}:\n\
                 \x20 call void @llvm.nvvm.barrier.sync.cnt(i32 1, i32 {b})\n\
                 \x20 %cl.r{l} = add i32 %cl.b, %cl.g\n\
                 \x20 br label %cl.h{l}\n\
                 cl.h{l}:\n\
                 \x20 %cl.i{l} = phi i32 [ %cl.r{l}, %cl.fx{l} ], [ %cl.n{l}, %cl.x{l} ]\n\
                 \x20 %cl.k{l} = icmp ult i32 %cl.i{l}, {total}\n\
                 \x20 br i1 %cl.k{l}, label %cl.m{l}_0, label %cl.e{l}\n"
            );
            s += &dispatch(&l.to_string(), &format!("%cl.i{l}"), &format!("cl.x{l}"));
            s += &format!(
                "cl.x{l}:\n\
                 \x20 call void @llvm.nvvm.barrier.sync.cnt(i32 1, i32 {b})\n\
                 \x20 %cl.n{l} = add i32 %cl.i{l}, %cl.g\n\
                 \x20 br label %cl.h{l}\n\
                 cl.e{l}:\n"
            );
        }
        if l + 1 < levels {
            s += &format!(
                "  call void @mapal_gsync(ptr addrspace(1) @{name}.gsync)\n  br label %cl.l{}\n",
                l + 1
            );
        } else {
            s += "  ret void\n";
        }
    }
    s + "}\n"
}
