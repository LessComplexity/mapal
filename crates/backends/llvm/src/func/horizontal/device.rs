//! `horizontal` device side: the merged kernels' specs, heads and composition.

// ---- device ---------------------------------------------------------------------------------------

/// A merged kernel as the host's `; mapal.hz` line names it: its block, and
/// each member's kernel, grid and block, in block-range order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct HzSpec {
    pub name: String,
    pub block: u64,
    pub members: Vec<(String, u64, u64, u64)>,
    /// kernel-merging `calls`: each member's level (all 0: side by side).
    pub level: Vec<u32>,
    /// `calls`: a member's zero pass — its output's bytes and parameter.
    pub zero: Vec<Option<(u64, usize)>>,
}

impl HzSpec {
    pub(crate) fn persistent(&self) -> bool {
        self.level.iter().any(|&l| l > 0)
    }
}

/// Every merged kernel the host text names (once each).
pub(crate) fn hz_specs(host: &str) -> Vec<HzSpec> {
    let mut out: Vec<HzSpec> = Vec::new();
    for line in host.lines() {
        let Some(rest) = line.trim().strip_prefix("; mapal.hz kernel=") else {
            continue;
        };
        let mut f = rest.split_whitespace();
        let name = f.next().unwrap_or_default().to_string();
        let mut block = 0;
        let mut members = Vec::new();
        let (mut level, mut zero) = (Vec::new(), Vec::new());
        for kv in f {
            if let Some(b) = kv.strip_prefix("block=") {
                block = b.parse().unwrap_or(0);
            } else if let Some(ms) = kv.strip_prefix("members=") {
                for m in ms.split(',') {
                    let p: Vec<&str> = m.split(':').collect();
                    if let [n, gx, gy, b, ref rest @ ..] = p[..] {
                        members.push((
                            n.to_string(),
                            gx.parse().unwrap_or(0),
                            gy.parse().unwrap_or(0),
                            b.parse().unwrap_or(0),
                        ));
                        // `calls`: `l<level>`, then `z<bytes>@<param>`.
                        let l = rest
                            .iter()
                            .find_map(|x| x.strip_prefix('l'))
                            .and_then(|v| v.parse().ok())
                            .unwrap_or(0);
                        let z = rest.iter().find_map(|x| {
                            let (b, p) = x.strip_prefix('z')?.split_once('@')?;
                            Some((b.parse().ok()?, p.parse().ok()?))
                        });
                        level.push(l);
                        zero.push(z);
                    }
                }
            }
        }
        if !out.iter().any(|s| s.name == name) {
            out.push(HzSpec {
                name,
                block,
                members,
                level,
                zero,
            });
        }
    }
    out
}

/// The kernels the merged kernels call (they must be in the module to be
/// composed).
pub(crate) fn hz_members(specs: &[HzSpec]) -> std::collections::BTreeSet<String> {
    specs
        .iter()
        .flat_map(|s| s.members.iter().map(|m| m.0.clone()))
        .collect()
}

/// A kernel's parameters `(type, full text, name)` and its `nvvm` bounds.
pub(crate) struct Head {
    pub params: Vec<(String, String, String)>,
    pub minctasm: Option<u64>,
}

pub(crate) fn parse_head(define: &str) -> Head {
    let open = define.find('(').expect("a define's parameters");
    let close = define.rfind(") ").expect("a define's attributes");
    let ps = &define[open + 1..close];
    let params = ps
        .split(", ")
        .filter(|p| !p.is_empty())
        .map(|p| {
            let name = p.rsplit(' ').next().expect("a named parameter").to_string();
            let ty = match p.strip_prefix("ptr addrspace(") {
                Some(r) => format!("ptr addrspace({}", &r[..=r.find(')').expect("addrspace")]),
                None => p.split(' ').next().expect("a type").to_string(),
            };
            (ty, p.to_string(), name)
        })
        .collect();
    let attr = |k: &str| {
        define[close..]
            .split(&format!("\"nvvm.{k}\"=\""))
            .nth(1)
            .and_then(|r| r.split('"').next())
            .and_then(|v| v.parse().ok())
    };
    Head {
        params,
        minctasm: attr("minctasm"),
    }
}

/// Kernel `name`'s text as a device function of its block coordinates:
/// `@hz.<name>(<params>, i32 %hz.bx, i32 %hz.by)` (not a `mapal_tile_` symbol:
/// no launch loads it), inlined into each merged
/// kernel; its `ctaid` reads are the arguments, and (`pad`: a merged block
/// larger than its own `block`) its barriers count its own threads.
pub(crate) fn device_fn(text: &str, name: &str, pad: Option<u64>) -> String {
    let (define, body) = text.split_once('\n').expect("a define line");
    let open = define.find('(').expect("parameters");
    let close = define.rfind(") ").expect("attributes");
    // A tensor map the kernel takes by value (`grid_constant`, a TMA
    // kernel's) is the merged kernel's parameter: the member receives its
    // address — a byval argument of an internal function is a local copy,
    // which the bulk-copy engine cannot read (wgmma-16bit: an H100 run of a
    // merged warp-specialized member faulted, `illegal memory access`).
    let ps = define[open + 1..close].replace(
        " byval([128 x i8]) align 64 \"nvvm.grid_constant\"",
        " readonly align 64",
    );
    let ps = ps.as_str();
    let sep = if ps.is_empty() { "" } else { ", " };
    let mut out = format!(
        "define internal void @hz.{name}({ps}{sep}i32 %hz.bx, i32 %hz.by) alwaysinline {{\n"
    );
    let mut b = body
        .replace(
            "call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()",
            "add i32 %hz.bx, 0",
        )
        .replace(
            "call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()",
            "add i32 %hz.by, 0",
        );
    if let Some(n) = pad {
        b = b.replace(
            "call void @llvm.nvvm.barrier0()",
            &format!("call void @llvm.nvvm.barrier.sync.cnt(i32 0, i32 {n})"),
        );
    }
    out.push_str(&b);
    out
}

/// The merged kernel of `spec`: block `b` of the grid runs the member whose
/// block range holds it, at its own coordinates (`x` fastest, the order the
/// hardware issues a grid's blocks in); a member of fewer threads than the
/// block runs on its own warps, the others exit.
pub(super) fn merged_kernel(spec: &HzSpec, heads: &[&Head]) -> String {
    let b = spec.block;
    let mut params = Vec::new();
    for (i, h) in heads.iter().enumerate() {
        for (_, full, name) in &h.params {
            let stem = name.trim_start_matches('%');
            let at = full.len() - name.len();
            params.push(format!("{}%h{i}.{stem}", &full[..at]));
        }
    }
    // Each member keeps its own register cap: `minctasm·block` threads a
    // member asked of an SM, in blocks of the merged size.
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
    let members: Vec<String> = spec
        .members
        .iter()
        .map(|m| format!("{}:{}x{}x{}", m.0, m.1, m.2, m.3))
        .collect();
    let mut s = format!(
        "; mapal.plan kernel={} kind=hz block={b} members={}\n\
         define ptx_kernel void @{}({}) {attrs} {{\n\
         entry:\n\
         \x20 %hz.b = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n\
         \x20 %hz.t = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n\
         \x20 br label %hz.m0\n",
        spec.name,
        members.join(","),
        spec.name,
        params.join(", ")
    );
    let n = spec.members.len();
    let mut start = 0u64;
    for (i, ((name, gx, gy, bi), h)) in spec.members.iter().zip(heads).enumerate() {
        let end = start + gx * gy;
        s += &format!("hz.m{i}:\n");
        if i + 1 < n {
            s += &format!(
                "  %hz.c{i} = icmp ult i32 %hz.b, {end}\n  br i1 %hz.c{i}, label %hz.r{i}, label %hz.m{}\n",
                i + 1
            );
        } else {
            s += &format!("  br label %hz.r{i}\n");
        }
        s += &format!("hz.r{i}:\n");
        if *bi < b {
            s += &format!(
                "  %hz.p{i} = icmp ult i32 %hz.t, {bi}\n  br i1 %hz.p{i}, label %hz.g{i}, label %hz.end\nhz.g{i}:\n"
            );
        }
        let args: Vec<String> = h
            .params
            .iter()
            .map(|(ty, _, name)| format!("{ty} %h{i}.{}", name.trim_start_matches('%')))
            .chain([format!("i32 %hz.x{i}"), format!("i32 %hz.y{i}")])
            .collect();
        s += &format!(
            "  %hz.l{i} = sub i32 %hz.b, {start}\n\
             \x20 %hz.x{i} = urem i32 %hz.l{i}, {gx}\n\
             \x20 %hz.y{i} = udiv i32 %hz.l{i}, {gx}\n\
             \x20 call void @hz.{name}({})\n\
             \x20 br label %hz.end\n",
            args.join(", ")
        );
        start = end;
    }
    s + "hz.end:\n  ret void\n}\n"
}

/// The device module with every merged kernel of `specs`: each member kernel
/// once as a device function (its standalone kernel kept only when a host
/// launch still names it, `live`), then the merged kernels.
pub(crate) fn compose(
    module: String,
    specs: &[HzSpec],
    live: &std::collections::BTreeSet<String>,
) -> String {
    if specs.is_empty() {
        return module;
    }
    let mut out = module;
    let mut heads: std::collections::BTreeMap<String, Head> = std::collections::BTreeMap::new();
    let mut fns = String::new();
    let mut padded = false;
    for name in hz_members(specs) {
        let head = format!("define ptx_kernel void @{name}(");
        let start = out
            .find(&head)
            .unwrap_or_else(|| panic!("merged member {name} is in the device module"));
        let end = start + out[start..].find("\n}\n").expect("a kernel's end") + 3;
        let text = out[start..end].to_string();
        let own = specs
            .iter()
            .flat_map(|s| s.members.iter().map(move |m| (s.block, m)))
            .filter(|(_, m)| m.0 == name)
            .map(|(b, m)| (b, m.3))
            .collect::<Vec<_>>();
        let pad = own.iter().any(|&(b, bi)| b > bi) && text.contains("@llvm.nvvm.barrier0()");
        padded |= pad;
        heads.insert(
            name.clone(),
            parse_head(text.lines().next().expect("define")),
        );
        fns.push('\n');
        fns.push_str(&device_fn(&text, &name, pad.then(|| own[0].1)));
        if !live.contains(&name) {
            // Its facts lines go with it (the merged kernel's line names it).
            let mut from = start;
            for tag in [format!("; mapal.plan kernel={name}"), "; cell".to_string()] {
                let prev = out[..from.saturating_sub(1)]
                    .rfind('\n')
                    .map_or(0, |i| i + 1);
                let line = &out[prev..from.saturating_sub(1)];
                if from > 0 && (line == tag || line.starts_with(&format!("{tag} "))) {
                    from = prev;
                }
            }
            out.replace_range(from..end, "");
        }
    }
    if padded && !out.contains("declare void @llvm.nvvm.barrier.sync.cnt(i32, i32)") {
        out.push_str("\ndeclare void @llvm.nvvm.barrier.sync.cnt(i32, i32)\n");
    }
    out.push_str(&fns);
    if specs.iter().any(HzSpec::persistent) {
        let decls = super::calls::missing_decls(&out);
        out.push('\n');
        out.push_str(&decls);
        out.push_str(super::calls::GSYNC);
    }
    for spec in specs {
        let hs: Vec<&Head> = spec.members.iter().map(|m| &heads[&m.0]).collect();
        out.push('\n');
        if spec.persistent() {
            out.push_str(&super::calls::persistent_kernel(spec, &hs));
        } else {
            out.push_str(&merged_kernel(spec, &hs));
        }
    }
    out
}
