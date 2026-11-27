//! `horizontal` emission: the per-group state, the launch lines it parses and the host half of a merged launch.
use super::*;

// ---- emission -------------------------------------------------------------------------------------

/// The emission's state of every group: what each member's launch left.
#[derive(Clone, Debug, Default)]
pub(crate) struct HzEmit {
    /// The member being emitted: `(group, member)`.
    pub cur: Option<(usize, usize)>,
    /// A retained site's whole-site arm: launched in place, never captured.
    pub standalone: bool,
    groups: Vec<HzState>,
}

#[derive(Clone, Debug, Default)]
pub(super) struct HzState {
    caps: Vec<Option<Cap>>,
    /// A retained member's flag (`alloca i1`): its row arm ran this trip.
    flags: Vec<Option<String>>,
    /// A member's emission held a launch the merge cannot take: its own
    /// launch ran in place, the others launch alone at the group's point.
    aborted: bool,
}

/// One member's deferred launch: its pointer array, and its table or kernel.
#[derive(Clone, Debug)]
pub(super) struct Cap {
    arr: String,
    nptr: usize,
    kind: CapKind,
    /// The launch as the member would issue it (re-issued when not merged).
    call: String,
}

#[derive(Clone, Debug)]
pub(super) enum CapKind {
    Table {
        global: String,
        rows: Vec<Row>,
        flags: u32,
        out: usize,
    },
    Fixed(Row),
}

/// A launch's kernel and geometry (a `%mapal_cand` row's fields).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Row {
    pub(super) id: u64,
    pub(super) zero: u64,
    pub(super) ws: u64,
    pub(super) gx: u64,
    pub(super) gy: u64,
    pub(super) block: u64,
    pub(super) smem: u64,
}

/// A runtime launch line's parts: a table (`mapal_gpu_site`), a pointer
/// array (`launchv`), or three pointers by value (`launch`).
pub(super) enum Parsed {
    Site {
        global: String,
        flags: u32,
        nptr: usize,
        arr: String,
        out: usize,
    },
    Launchv {
        row: Row,
        nptr: usize,
        arr: String,
    },
    Launch {
        row: Row,
        args: [String; 3],
    },
}

pub(super) fn int(s: &str, ty: &str) -> Option<i64> {
    s.trim().strip_prefix(ty)?.trim().parse().ok()
}

/// The fields of `call void @mapal_gpu_<entry>(…)` (a ` ; unit` tag allowed,
/// any other tag — a split's — not).
pub(super) fn call_fields<'s>(line: &'s str, entry: &str) -> Option<Vec<&'s str>> {
    let line = line.trim();
    let line = line.strip_suffix(" ; unit").unwrap_or(line);
    let rest = line.strip_prefix(&format!("call void @mapal_gpu_{entry}("))?;
    let inner = rest.strip_suffix(')')?;
    (!inner.contains(';')).then(|| inner.split(", ").collect())
}

pub(super) fn parse_launch(line: &str) -> Option<Parsed> {
    if let Some(f) = call_fields(line, "site") {
        let [_, flags, stream, _, global, nptr, arr, out, _] = f[..] else {
            return None;
        };
        if int(stream, "i32")? != 0 {
            return None;
        }
        return Some(Parsed::Site {
            global: global.strip_prefix("ptr ")?.to_string(),
            flags: int(flags, "i32")? as u32,
            nptr: int(nptr, "i32")? as usize,
            arr: arr.strip_prefix("ptr ")?.to_string(),
            out: int(out, "i32")? as usize,
        });
    }
    let geom = |f: &[&str]| -> Option<[u64; 4]> {
        let v: Vec<u64> = f
            .iter()
            .map(|x| int(x, "i32").map(|v| v as u64))
            .collect::<Option<_>>()?;
        v.try_into().ok()
    };
    if let Some(f) = call_fields(line, "launchv") {
        let [id, stream, nptr, arr, ..] = f[..] else {
            return None;
        };
        let [gx, gy, block, smem] = geom(f.get(4..8)?)?;
        if f.len() != 8 || int(stream, "i32")? != 0 {
            return None;
        }
        return Some(Parsed::Launchv {
            row: Row {
                id: int(id, "i64")? as u64,
                zero: 0,
                ws: 0,
                gx,
                gy,
                block,
                smem,
            },
            nptr: int(nptr, "i32")? as usize,
            arr: arr.strip_prefix("ptr ")?.to_string(),
        });
    }
    if let Some(f) = call_fields(line, "launch") {
        if f.len() != 8 {
            return None;
        }
        let [gx, gy, block, smem] = geom(&f[4..8])?;
        let arg = |x: &str| x.strip_prefix("i64 ").map(str::to_string);
        return Some(Parsed::Launch {
            row: Row {
                id: int(f[0], "i64")? as u64,
                zero: 0,
                ws: 0,
                gx,
                gy,
                block,
                smem,
            },
            args: [arg(f[1])?, arg(f[2])?, arg(f[3])?],
        });
    }
    None
}

/// The rows of table `global` in `tables` (`%mapal_cand { i64 id, i64 rid,
/// i64 zero, i64 ws, i32 gx, i32 gy, i32 block, i32 smem, … }`).
pub(super) fn table_rows(tables: &str, global: &str) -> Option<Vec<Row>> {
    let line = tables
        .lines()
        .find(|l| l.starts_with(&format!("{global} = ")))?;
    line.split("%mapal_cand { ")
        .skip(1)
        .map(|r| {
            let f: Vec<&str> = r.split(" }").next()?.split(", ").collect();
            let i = |k: usize, ty: &str| int(f.get(k)?, ty);
            Some(Row {
                id: i(0, "i64")? as u64,
                zero: i(2, "i64")? as u64,
                ws: i(3, "i64")? as u64,
                gx: i(4, "i32")? as u64,
                gy: i(5, "i32")? as u64,
                block: i(6, "i32")? as u64,
                smem: i(7, "i32")? as u64,
            })
        })
        .collect()
}

/// A merged kernel's id: its members' kernels and grids.
pub(super) fn hz_id(rows: &[Row]) -> u64 {
    let key: Vec<String> = rows
        .iter()
        .map(|r| format!("{:016x}:{}x{}x{}", r.id, r.gx, r.gy, r.block))
        .collect();
    gpu::fnv(&format!("hz|{}", key.join("|")))
}

impl FnEmit<'_> {
    /// `line`'s hook while member `(g, i)` is emitted: a launch the merge
    /// takes is captured (`true`: not emitted here); any other runtime call
    /// aborts the group — the member's captured launch is issued in place,
    /// before it (`false`: the caller emits the line).
    pub(crate) fn hz_take(&mut self, s: &str) -> bool {
        let Some((g, i)) = self.hz.cur else {
            return false;
        };
        if self.hz.standalone || !s.contains("call void @mapal_gpu_") {
            return false;
        }
        if self.hz.groups.len() <= g {
            self.hz.groups.resize(g + 1, HzState::default());
        }
        let k = self.resident.as_ref().expect("plan").hz[g].members.len();
        let st = &mut self.hz.groups[g];
        if st.caps.len() < k {
            st.caps.resize(k, None);
            st.flags.resize(k, None);
        }
        let parsed = (!s.contains('\n'))
            .then(|| parse_launch(s))
            .flatten()
            .filter(|_| self.hz.groups[g].caps[i].is_none() && !self.hz.groups[g].aborted);
        let Some(p) = parsed else {
            // Not a launch the merge takes: this member's launch runs here.
            self.hz_abort(g, i);
            return false;
        };
        let cap = match p {
            Parsed::Site {
                global,
                flags,
                nptr,
                arr,
                out,
            } => {
                let Some(rows) = table_rows(&self.cand_tables, &global) else {
                    self.hz_abort(g, i);
                    return false;
                };
                Cap {
                    arr,
                    nptr,
                    kind: CapKind::Table {
                        global,
                        rows,
                        flags,
                        out,
                    },
                    call: s.trim().to_string(),
                }
            }
            Parsed::Launchv { row, nptr, arr } => Cap {
                arr,
                nptr,
                kind: CapKind::Fixed(row),
                call: s.trim().to_string(),
            },
            Parsed::Launch { row, args } => {
                // Three pointers by value: kept in an array, so a launch in
                // a branch arm reaches the group's point.
                let arr = format!("%hzp{}", self.fresh());
                self.allocas
                    .push_str(&format!("  {arr} = alloca [3 x i64]\n"));
                for (k, a) in args.iter().enumerate() {
                    let p = self.tmp();
                    self.body.push_str(&format!(
                        "  {p} = getelementptr [3 x i64], ptr {arr}, i64 0, i64 {k}\n  store i64 {a}, ptr {p}\n"
                    ));
                }
                Cap {
                    arr: arr.clone(),
                    nptr: 3,
                    kind: CapKind::Fixed(row),
                    call: format!(
                        "call void @mapal_gpu_launchv(i64 {}, i32 0, i32 3, ptr {arr}, i32 {}, i32 {}, i32 {}, i32 {})",
                        row.id as i64, row.gx, row.gy, row.block, row.smem
                    ),
                }
            }
        };
        if self.resident.as_ref().expect("plan").hz[g].branchy[i] {
            let flag = match self.hz.groups[g].flags[i].clone() {
                Some(f) => f,
                None => {
                    let f = format!("%hzf{}", self.fresh());
                    self.allocas
                        .push_str(&format!("  {f} = alloca i1\n  store i1 false, ptr {f}\n"));
                    self.hz.groups[g].flags[i] = Some(f.clone());
                    f
                }
            };
            self.body
                .push_str(&format!("  store i1 true, ptr {flag}\n"));
        }
        self.hz.groups[g].caps[i] = Some(cap);
        true
    }

    /// Member `i` of group `g` launches in place: its captured launch is
    /// issued here, and in a `calls` group every earlier member's before it
    /// (they produce what it reads); the rest launch alone.
    pub(super) fn hz_abort(&mut self, g: usize, i: usize) {
        self.hz.cur = None;
        let group = &self.resident.as_ref().expect("plan").hz[g];
        let persistent = group.persistent();
        // kernel-merging `calls`: the last member's own launch was taken and
        // what follows it (its output's copy-out) reads what the group
        // writes: the group launches here, before it.
        if persistent
            && i + 1 == group.members.len()
            && !self.hz.groups[g].aborted
            && self.hz.groups[g].caps[i].is_some()
        {
            self.hz_launch(g);
            return;
        }
        self.hz.groups[g].aborted = true;
        let from = if persistent { 0 } else { i };
        for k in from..=i {
            if let Some(c) = self.hz.groups[g].caps[k].take() {
                let flag = self.hz.groups[g].flags[k].clone();
                self.hz_issue(&c, flag.as_deref());
            }
        }
    }

    /// Issue a captured launch on its own (under its flag when it ran in a
    /// branch arm).
    pub(super) fn hz_issue(&mut self, c: &Cap, flag: Option<&str>) {
        let Some(f) = flag else {
            self.line(&c.call);
            return;
        };
        let (on, next) = (self.label(), self.label());
        let v = self.tmp();
        self.line(format!("{v} = load i1, ptr {f}"));
        self.line(format!("br i1 {v}, label %{on}, label %{next}"));
        self.label_line(&on);
        self.line(&c.call);
        self.line(format!("br label %{next}"));
        self.label_line(&next);
    }

    /// The merged tuples of the captured launches: each member's unsplit
    /// candidates (a fixed launch its own), every tuple whose block and shared
    /// memory a launch takes, in order of their ranks' sum, at most `--cands`.
    pub(super) fn hz_tuples(&self, group: &HzGroup, caps: &[&Cap]) -> Vec<Vec<Row>> {
        let gpu = self.profile.gpu().expect("a GPU profile");
        // kernel-merging `calls`: a member whose zero pass the persistent
        // kernel runs one level before it takes its atomic candidates too.
        let opts: Vec<Vec<(usize, Row)>> = caps
            .iter()
            .enumerate()
            .map(|(i, c)| match &c.kind {
                CapKind::Table { rows, .. } => rows
                    .iter()
                    .copied()
                    .enumerate()
                    .filter(|(_, r)| {
                        r.ws == 0 && (r.zero == 0 || group.zero_ok.get(i).copied().unwrap_or(false))
                    })
                    .collect(),
                CapKind::Fixed(r) => vec![(0, *r)],
            })
            .collect();
        if opts.iter().any(Vec::is_empty) {
            return Vec::new();
        }
        let mut all: Vec<(Vec<usize>, Vec<Row>)> = vec![(Vec::new(), Vec::new())];
        for o in &opts {
            all = all
                .into_iter()
                .flat_map(|(ks, rs)| {
                    o.iter().map(move |&(k, r)| {
                        let (mut ks, mut rs) = (ks.clone(), rs.clone());
                        ks.push(k);
                        rs.push(r);
                        (ks, rs)
                    })
                })
                .collect();
        }
        all.retain(|(_, rs)| {
            let b = rs.iter().map(|r| r.block).max().unwrap_or(0);
            b <= gpu.max_threads_per_block
                && rs.iter().all(|r| r.block % gpu.warp == 0)
                && rs.iter().map(|r| r.smem).max().unwrap_or(0) <= gpu.smem_per_block
        });
        all.sort_by_key(|(ks, _)| (ks.iter().sum::<usize>(), ks.clone()));
        let cap = usize::try_from(self.gpu_arms.site_cands)
            .unwrap_or(usize::MAX)
            .max(1);
        all.into_iter().take(cap).map(|(_, rs)| rs).collect()
    }

    /// At group `g`'s last member: the merged launch over every member's
    /// captured pointers (when every member's launch was captured and a tuple
    /// fits; under the retained members' flags), each member's own launch
    /// otherwise.
    pub(crate) fn hz_launch(&mut self, g: usize) {
        let group = self.resident.as_ref().expect("plan").hz[g].clone();
        let k = group.members.len();
        let st = match self.hz.groups.get_mut(g) {
            Some(s) => std::mem::take(s),
            None => HzState::default(),
        };
        let caps: Vec<Option<Cap>> = (0..k).map(|i| st.caps.get(i).cloned().flatten()).collect();
        let flags: Vec<Option<String>> =
            (0..k).map(|i| st.flags.get(i).cloned().flatten()).collect();
        let complete: Option<Vec<&Cap>> = (!st.aborted)
            .then(|| caps.iter().map(Option::as_ref).collect::<Option<Vec<_>>>())
            .flatten();
        let tuples = complete
            .as_ref()
            .map(|c| self.hz_tuples(&group, c))
            .unwrap_or_default();
        let fallback = |e: &mut Self| {
            for (c, f) in caps.iter().zip(&flags) {
                if let Some(c) = c {
                    e.hz_issue(c, f.as_deref());
                }
            }
        };
        if tuples.is_empty() {
            fallback(self);
        } else {
            let all_flags: Vec<&String> = flags.iter().flatten().collect();
            let arms = (!all_flags.is_empty()).then(|| {
                let mut acc = String::new();
                for f in &all_flags {
                    let v = self.tmp();
                    self.line(format!("{v} = load i1, ptr {f}"));
                    acc = if acc.is_empty() {
                        v
                    } else {
                        let a = self.tmp();
                        self.line(format!("{a} = and i1 {acc}, {v}"));
                        a
                    };
                }
                let (m, f, end) = (self.label(), self.label(), self.label());
                self.line(format!("br i1 {acc}, label %{m}, label %{f}"));
                self.label_line(&m);
                (f, end)
            });
            let caps: Vec<&Cap> = complete.expect("complete");
            self.hz_merged(&group, &caps, &tuples);
            if let Some((f, end)) = arms {
                self.line(format!("br label %{end}"));
                self.label_line(&f);
                fallback(self);
                self.line(format!("br label %{end}"));
                self.label_line(&end);
            }
            // A table no other launch issues is not the module's.
            if flags.iter().all(Option::is_none) {
                for c in &caps {
                    if let CapKind::Table { global, .. } = &c.kind {
                        self.cand_tables = self
                            .cand_tables
                            .lines()
                            .filter(|l| !l.starts_with(&format!("{global} = ")))
                            .map(|l| format!("{l}\n"))
                            .collect();
                    }
                }
            }
        }
        for f in flags.iter().flatten() {
            self.line(format!("store i1 false, ptr {f}"));
        }
    }

    /// The merged launch: the members' pointers in one array, then the
    /// tuples' table (or the one kernel).
    pub(super) fn hz_merged(&mut self, group: &HzGroup, caps: &[&Cap], tuples: &[Vec<Row>]) {
        let np: usize = caps.iter().map(|c| c.nptr).sum();
        let arr = format!("%hzv{}", self.fresh());
        self.allocas
            .push_str(&format!("  {arr} = alloca [{np} x i64]\n"));
        let mut off = 0;
        let mut out = None;
        let mut rerun = group.rerun;
        for c in caps {
            for p in 0..c.nptr {
                let (s, v, d) = (self.tmp(), self.tmp(), self.tmp());
                self.line(format!(
                    "{s} = getelementptr [{n} x i64], ptr {a}, i64 0, i64 {p}",
                    n = c.nptr,
                    a = c.arr
                ));
                self.line(format!("{v} = load i64, ptr {s}"));
                self.line(format!(
                    "{d} = getelementptr [{np} x i64], ptr {arr}, i64 0, i64 {}",
                    off + p
                ));
                self.line(format!("store i64 {v}, ptr {d}"));
            }
            if let CapKind::Table { flags, out: o, .. } = &c.kind {
                rerun &= flags & 1 != 0;
                out.get_or_insert(off + o);
            }
            off += c.nptr;
        }
        let persistent = group.persistent();
        let mut rows = Vec::new();
        for t in tuples {
            let block = t.iter().map(|r| r.block).max().expect("members");
            let smem = t.iter().map(|r| r.smem).max().expect("members");
            if persistent {
                // kernel-merging `calls`: one persistent kernel, at most the
                // largest level's blocks (the runtime caps it at the loaded
                // kernel's co-resident grid).
                let members: Vec<String> = t
                    .iter()
                    .zip(caps)
                    .zip(&group.level)
                    .map(|((r, c), l)| {
                        let z = match (&c.kind, r.zero) {
                            (CapKind::Table { out, .. }, z) if z > 0 => format!(":z{z}@{out}"),
                            _ => String::new(),
                        };
                        format!(
                            "{}:{}:{}:{}:l{l}{z}",
                            gpu::kernel_name(r.id),
                            r.gx,
                            r.gy,
                            r.block
                        )
                    })
                    .collect();
                let id = gpu::fnv(&format!("calls|{}", members.join("|")));
                let mut per_level: std::collections::BTreeMap<u32, u64> = Default::default();
                for (r, &l) in t.iter().zip(&group.level) {
                    *per_level.entry(l).or_default() += r.gx * r.gy;
                }
                let gx = per_level.values().copied().max().unwrap_or(1);
                self.line(format!(
                    "; mapal.hz kernel={} block={block} smem={smem} grid={gx} members={}",
                    gpu::kernel_name(id),
                    members.join(",")
                ));
                rows.push((id, gx, block, smem));
                continue;
            }
            let id = hz_id(t);
            let gx: u64 = t.iter().map(|r| r.gx * r.gy).sum();
            let members: Vec<String> = t
                .iter()
                .map(|r| format!("{}:{}:{}:{}", gpu::kernel_name(r.id), r.gx, r.gy, r.block))
                .collect();
            self.line(format!(
                "; mapal.hz kernel={} block={block} smem={smem} members={}",
                gpu::kernel_name(id),
                members.join(",")
            ));
            rows.push((id, gx, block, smem));
        }
        if rows.len() == 1 {
            let (id, gx, block, smem) = rows[0];
            let entry = if persistent { "launch_coop" } else { "launchv" };
            self.line(format!(
                "call void @mapal_gpu_{entry}(i64 {}, i32 0, i32 {np}, ptr {arr}, i32 {gx}, i32 1, i32 {block}, i32 {smem})",
                id as i64
            ));
            return;
        }
        let global = format!("@{}.hz{}", self.fnames[self.f], self.fresh());
        let consts: Vec<String> = rows
            .iter()
            .map(|&(id, gx, block, smem)| {
                format!(
                    "%mapal_cand {{ i64 {}, i64 0, i64 0, i64 0, i32 {gx}, i32 1, i32 {block}, i32 {smem}, i32 0, i32 0 }}",
                    id as i64
                )
            })
            .collect();
        self.cand_tables.push_str(&format!(
            "{global} = private unnamed_addr constant [{} x %mapal_cand] [{}]\n",
            consts.len(),
            consts.join(", ")
        ));
        let coop = if persistent {
            super::calls::SITE_COOP
        } else {
            0
        };
        self.line(format!(
            "call void @mapal_gpu_site(i64 {}, i32 {}, i32 0, i32 {}, ptr {global}, i32 {np}, ptr {arr}, i32 {}, ptr null)",
            rows[0].0 as i64,
            u32::from(rerun) | coop,
            rows.len(),
            out.unwrap_or(0)
        ));
    }
}
