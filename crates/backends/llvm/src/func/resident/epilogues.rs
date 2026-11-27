//! epilogues, post folds, the site table, absorbed-cell pricing and delta uploads

use super::*;

impl<'a> FnEmit<'a> {
    /// **S54 R1: the per-cell morphisms each device site's store evaluates.**
    ///
    /// Sites in execution order; a `Map` joins site `s`'s group when its
    /// [`CellSite`](mapal_ir::CellSite) reads a group value (the site's output,
    /// or a member's) only at `t`, and every other operand is complete in
    /// `Gmem` when `s` launches — produced by an earlier launch, or on the host
    /// before `s` (then copied in or built on the device, as a site operand
    /// is) — at a footprint that is a function of `t` alone (`Same`, `Row`,
    /// `Col`, `Const`). Law 1 of the placement model: an operand is read only
    /// where it is complete. Members run where the site runs (the same loop
    /// phase, never a guard arm), their bodies are device leaves, and their
    /// cells are the site's (`extent = rows · c`). A chain composes: a member
    /// of a member joins. Each group value is stored only when a reader
    /// outside the group exists — the use count; the rest live in a register.
    pub(in crate::func) fn epilogues(&self) -> SecondaryMap<MorphismId, Epi> {
        let mut out: SecondaryMap<MorphismId, Epi> = SecondaryMap::new();
        let sites = self.device_sites();
        let cells = self.ir.cell_plan(self.f);
        if sites.is_empty() || cells.is_empty() {
            return out;
        }
        let region = self.regions();
        let pos = |m: MorphismId| self.lup.position(m).unwrap_or(u32::MAX);
        let target = |m: MorphismId| self.ir.morphism(m).expect("morphism resolves").target;
        let mut order: Vec<MorphismId> = sites.keys().collect();
        order.sort_by_key(|&m| pos(m));
        // The launch after which a device-made object is complete in `Gmem`.
        let mut made: SecondaryMap<ObjectId, MorphismId> = SecondaryMap::new();
        for &s in &order {
            made.insert(target(s), s);
        }
        let mut taken: SecondaryMap<MorphismId, ()> = SecondaryMap::new();
        // plan-cost `linear` (§4a concat): a sum of device tile sites over one
        // `k` range (`linear_plan`) is one site table launch where it beats
        // the sites' own plans (D1): the latest term launches it; the other
        // terms and the sum emit nothing.
        let mut tabled: SecondaryMap<MorphismId, ()> = SecondaryMap::new();
        for (sum, ls) in self.ir.linear_plan(self.f).iter() {
            let Some((anchor, epi)) = self.site_table(sum, ls, &sites, &region) else {
                continue;
            };
            for &t in &ls.terms {
                tabled.insert(t, ());
            }
            taken.insert(sum, ());
            made.insert(target(sum), anchor);
            out.insert(anchor, epi);
        }
        for &s in &order {
            if tabled.contains_key(s) {
                continue;
            }
            // A row fold's store is a scalar per row: no group (S54 `rowfold`).
            let DevSite::Tile(site) = &sites[s] else {
                continue;
            };
            let l = gpu::launch(site, self.profile, self.contract, self.gpu_arms);
            let Some(&rs) = region.get(s) else {
                continue;
            };
            // A retained output (S53) updates one row a trip: its group's
            // members would keep the full launch's stale rows. A mixed-radix
            // site stores digit by digit: no epilogue (`radix-tile`).
            // A `tile-post` radix site's store evaluates its epilogue at each
            // cell's address (its index `t`).
            if !l.rungs.epilogue
                || self.retain.contains_key(target(s))
                || (site.radix.is_some() && site.post.is_none())
            {
                continue;
            }
            let c = target(s);
            let n = array_len(&self.obj_ty(c));
            let before = |p: MorphismId| {
                pos(p) < pos(s) && region.get(p).is_some_and(|&r| r == rs || r.0.is_none())
            };
            // `tile-post`: the site's body after its fold is the site's value.
            let post = match &site.post {
                Some(p) => match self.post_of(s, p) {
                    Some(x) => Some(x),
                    None => continue,
                },
                None => None,
            };
            let mut values = vec![c];
            let mut chain: Vec<Member> = Vec::new();
            loop {
                // Complete before `s` launches, where `s` runs.
                let complete = |o: ObjectId| match made.get(o) {
                    Some(&p) => p != s && before(p),
                    None => matches!(self.ir.in_edges(o), [p] if before(*p)),
                };
                let mut cands: Vec<MorphismId> = values
                    .iter()
                    .flat_map(|&v| self.cell_readers(v))
                    .filter(|&m| !taken.contains_key(m) && cells.contains_key(m))
                    .collect();
                cands.sort_by_key(|&m| pos(m));
                cands.dedup();
                let admit = |mc: MorphismId| -> Option<Member> {
                    let morph = self.ir.morphism(mc)?;
                    let Operation::Map { body, captures } = morph.op else {
                        return None;
                    };
                    let cell = &cells[mc];
                    let in_ty = self.obj_ty(self.ir.func(body)?.input);
                    // ponytail: the site's store takes no scalar argument and
                    // walks no nested fold; such a member stays its own cell.
                    if !cell.uniform.is_empty()
                        || !cell.nested.is_empty()
                        || region.get(mc) != Some(&rs)
                        || cell.extent != n
                        || !is_float(array_elem(&self.obj_ty(morph.target)))
                        || !self.device_leaf(body)
                    {
                        return None;
                    }
                    let mut srcs = Vec::new();
                    for slot in 0..=captures {
                        let arr = if captures == 0 {
                            morph.source
                        } else {
                            self.pair_source(morph.source, slot)?
                        };
                        let feet: Vec<Footprint> = cell
                            .reads
                            .iter()
                            .filter(|r| r.slot == slot)
                            .map(|r| r.foot)
                            .collect();
                        let src = match values.iter().position(|&v| v == arr) {
                            // A group value read off its own cell needs every
                            // block's store first: a kernel boundary (R3).
                            Some(k) if feet.iter().all(|&f| f == Footprint::Same) => {
                                EpiSrc::Value(k)
                            }
                            Some(_) => return None,
                            None if slot == captures && feet.is_empty() => EpiSrc::Index,
                            // ponytail: a mapped array outside the group is one
                            // more load per cell; admit it when a program has one.
                            None if slot == captures => return None,
                            None if feet.contains(&Footprint::Other) || !complete(arr) => {
                                return None;
                            }
                            None => EpiSrc::Gmem(arr),
                        };
                        if captures > 0 && crate::ty::erased_index(&in_ty, slot) != Some(slot) {
                            return None;
                        }
                        srcs.push(src);
                    }
                    srcs.iter()
                        .any(|s| matches!(s, EpiSrc::Value(_)))
                        .then_some(Member {
                            m: mc,
                            body,
                            out: morph.target,
                            store: false,
                            srcs,
                        })
                };
                let Some(mb) = cands.into_iter().find_map(admit) else {
                    break;
                };
                // `a, b, out`, every operand and every member's output.
                let mut trial = Epi {
                    post: post.clone(),
                    store_out: true,
                    chain: chain.clone(),
                    terms: Vec::new(),
                    split: 1,
                };
                trial.chain.push(mb.clone());
                if 3 + trial.caps().len() + trial.chain.len() + trial.uniforms().len()
                    > gpu::LAUNCHV_MAX
                {
                    break;
                }
                taken.insert(mb.m, ());
                values.push(mb.out);
                made.insert(mb.out, s);
                chain.push(mb);
            }
            if chain.is_empty() && post.is_none() {
                continue;
            }
            // plan-cost (design #10): absorbing the members keeps the site
            // unsplit; leaving them their own cell launches frees its fold to
            // split. The cheaper by `site_ns` and `op_ns` (D1); the absorbed
            // group where the part has no plan rates (D8). A `tile-post`
            // site's group stands either way (its value is the epilogue's).
            if !chain.is_empty() && !self.absorb_pays(site, &l, n, &chain) {
                for mb in &chain {
                    taken.remove(mb.m);
                    made.remove(mb.out);
                }
                chain.clear();
                if post.is_none() {
                    continue;
                }
            }
            out.insert(
                s,
                Epi {
                    post,
                    store_out: true,
                    chain,
                    terms: Vec::new(),
                    split: 1,
                },
            );
        }
        // Stores from the use count, once every group is known (a later
        // group's operand is an outside reader of an earlier group's value).
        for (s, g) in out.iter_mut() {
            if !g.terms.is_empty() {
                g.chain[0].store = true;
                continue;
            }
            let ms: Vec<MorphismId> = g.chain.iter().map(|mb| mb.m).collect();
            g.store_out = self.read_outside(target(s), &ms);
            for mb in &mut g.chain {
                mb.store = self.read_outside(mb.out, &ms);
            }
        }
        out
    }

    /// `tile-post`: the epilogue of tile site `s` — its map body after the
    /// fold, fed slot by slot (`Post::srcs`): `None` when the body is no
    /// device leaf or its input erases a slot.
    pub(super) fn post_of(&self, s: MorphismId, p: &mapal_ir::TilePost) -> Option<Post> {
        let morph = self.ir.morphism(s)?;
        let Operation::Map { body, captures } = morph.op else {
            return None;
        };
        let in_ty = self.obj_ty(self.ir.func(body)?.input);
        if !self.device_leaf(body) {
            return None;
        }
        let mut srcs = Vec::new();
        let mut u = 0;
        for slot in 0..=captures {
            if crate::ty::erased_index(&in_ty, slot) != Some(slot) {
                return None;
            }
            srcs.push(if slot == captures {
                Some(EpiSrc::Index)
            } else if p.uniform.contains(&slot) {
                u += 1;
                Some(EpiSrc::Uniform(u - 1))
            } else if p.reads.contains(&slot) {
                Some(EpiSrc::Gmem(self.pair_source(morph.source, slot)?))
            } else {
                None
            });
        }
        Some(Post {
            body,
            fold: p.fold,
            srcs,
        })
    }

    /// plan-cost `linear`: the site table of linear site `ls` (summing map
    /// `sum`) — its launching term and group — when every term is a device
    /// tile site of one kernel shape, all in one region with the sum, the
    /// order is free, and [`gpu::concat_plan`] costs less than the terms' own
    /// picks plus the latest term's store reading the others' outputs.
    pub(super) fn site_table(
        &self,
        sum: MorphismId,
        ls: &mapal_ir::LinearSite,
        sites: &SecondaryMap<MorphismId, DevSite>,
        region: &SecondaryMap<MorphismId, (Option<ObjectId>, bool)>,
    ) -> Option<(MorphismId, Epi)> {
        if !self.gpu_arms.free {
            return None;
        }
        let tile = |t: MorphismId| match sites.get(t) {
            Some(DevSite::Tile(x)) => Some(x),
            _ => None,
        };
        let pos = |m: MorphismId| self.lup.position(m).unwrap_or(u32::MAX);
        let anchor = *ls.terms.iter().max_by_key(|&&t| pos(t))?;
        let site = tile(anchor)?;
        let shape = |x: &TileSite| {
            let mut y = x.clone();
            (y.a.slot, y.b.slot) = (0, 0);
            y
        };
        let target = |m: MorphismId| self.ir.morphism(m).expect("morphism resolves").target;
        for &t in ls.terms.iter().chain([&sum]) {
            let ok = region.get(t).is_some() && region.get(t) == region.get(anchor);
            if !ok || self.retain.contains_key(target(t)) {
                return None;
            }
        }
        let n = ls.terms.len() as u64;
        let (split, table) = gpu::concat_plan(
            site,
            self.profile,
            self.contract,
            self.gpu_arms,
            n,
            ls.bound,
        )?;
        let mut today = 0.0;
        for &t in &ls.terms {
            let x = tile(t)?;
            if shape(x) != shape(site) {
                return None;
            }
            let l = gpu::launch(x, self.profile, self.contract, self.gpu_arms);
            let cands = gpu::plan_cands(
                &l,
                x,
                self.profile,
                self.contract,
                self.gpu_arms,
                crate::Order::Free,
            );
            // The latest term keeps the sum as its epilogue: unsplit.
            let own = match t == anchor {
                true => cands.iter().find(|c| c.combine == gpu::Combine::None),
                false => cands.first(),
            };
            today += own?.cost.ns;
        }
        let bytes = (n - 1) * site.rows * site.c * crate::profile::elem_bytes(&site.elem);
        today +=
            self.profile.op_ns(f64::INFINITY, bytes)? - self.profile.op_ns(f64::INFINITY, 0)?;
        if table.ns >= today {
            return None;
        }
        let morph = self.ir.morphism(sum)?;
        let Operation::Map { body, captures } = morph.op else {
            return None;
        };
        let mut srcs = vec![EpiSrc::Value(0); captures as usize];
        srcs.push(EpiSrc::Index);
        Some((
            anchor,
            Epi {
                post: None,
                store_out: false,
                chain: vec![Member {
                    m: sum,
                    body,
                    out: morph.target,
                    store: true,
                    srcs,
                }],
                terms: ls.terms.iter().copied().filter(|&t| t != anchor).collect(),
                split,
            },
        ))
    }

    /// plan-cost: does site `site` (launch `l`, `n` cells) evaluating `chain`
    /// in its store beat its cheapest split plus each member as its own cell
    /// launch? Absorbed: the unsplit candidate, each member's operand loads
    /// added to its store. Split: the split candidate, then per member one
    /// dependent op ([`TargetProfile::op_ns`]) reading its operands and the
    /// value(s) the group kept in registers and storing its output. True where
    /// the free order has no split candidate or the part no plan rates.
    pub(super) fn absorb_pays(
        &self,
        site: &TileSite,
        l: &gpu::Launch,
        n: u64,
        chain: &[Member],
    ) -> bool {
        let order = if self.gpu_arms.free {
            crate::Order::Free
        } else {
            crate::Order::Sequential
        };
        if !l.rungs.plan_cost {
            return true;
        }
        let cands = gpu::plan_cands(l, site, self.profile, self.contract, self.gpu_arms, order);
        let (Some(pick), Some(unsplit)) = (
            cands.first(),
            cands.iter().find(|c| c.combine == gpu::Combine::None),
        ) else {
            return true;
        };
        if pick.combine == gpu::Combine::None {
            return true;
        }
        let e = crate::profile::elem_bytes(&site.elem);
        let loads = |mb: &Member| {
            mb.srcs
                .iter()
                .filter(|s| !matches!(s, EpiSrc::Index))
                .count() as u64
        };
        let gmem = |mb: &Member| {
            mb.srcs
                .iter()
                .filter(|s| matches!(s, EpiSrc::Gmem(_)))
                .count() as u64
        };
        let mut absorbed = unsplit.cost.ns;
        let mut split = pick.cost.ns;
        let mut prev = pick.cost.kernel_ns;
        for mb in chain {
            absorbed += self
                .profile
                .op_ns(f64::INFINITY, gmem(mb) * n * e)
                .unwrap_or(0.0)
                - self.profile.op_ns(f64::INFINITY, 0).unwrap_or(0.0);
            let Some(op) = self.profile.op_ns(prev, (loads(mb) + 1) * n * e) else {
                return true;
            };
            split += op;
            prev = op;
        }
        absorbed <= split
    }

    /// The `upload`s a loop's iteration writes only rows of (`delta-h2d`): the
    /// value `write_delta` says crosses the back edge, or a `Same` image of it
    /// — and (caching-memory `retain-same`) the uploads `unchanged_plan` has
    /// the same on every trip. Its producer runs every iteration, inside the
    /// loop, so its device copy is never freed there and holds last
    /// iteration's value.
    pub(super) fn delta_uploads(
        &self,
        upload: &SecondaryMap<ObjectId, ()>,
        same_on: bool,
    ) -> SecondaryMap<ObjectId, DeltaUp> {
        let mut out = SecondaryMap::new();
        for d in self.ir.write_delta(self.f).values() {
            let Delta::Rows {
                stride,
                start,
                next,
            } = *d
            else {
                continue;
            };
            // The counter is a `Proj` of the loop's merge.
            let merge = self.ir.in_edges(start.var).first();
            // (A range wider than the array is the whole upload.)
            if let Some(merge) = merge.and_then(|&m| self.ir.morphism(m)).map(|m| m.source)
                && upload.contains_key(next)
                && stride <= array_len(&self.obj_ty(next))
            {
                out.insert(
                    next,
                    DeltaUp {
                        merge,
                        start,
                        stride,
                        same: false,
                    },
                );
            }
        }
        let up = self.ir.unchanged_plan(self.f);
        if same_on {
            for o in up.same.keys().filter(|&o| upload.contains_key(o)) {
                if let Some(&merge) = up.loop_of.get(o) {
                    out.insert(
                        o,
                        DeltaUp {
                            merge,
                            start: Affine {
                                var: merge,
                                a: 0,
                                b: 0,
                            },
                            stride: 0,
                            same: true,
                        },
                    );
                }
            }
        }
        out
    }
}
