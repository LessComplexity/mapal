//! Where a site runs (host or device), the device-made facts and the tile/post site plans.

use super::*;

/// Threads per block of a `devgen` kernel — the runtime's `MAPAL_GEN_THREADS`
/// launches it with exactly this many, so the two must agree.
/// ponytail: the naive kernel's literal, for the same reason (a one-element-
/// per-thread fill is bandwidth-bound at any block size a part accepts).
pub(super) const GEN_THREADS: u64 = NAIVE_THREADS;

/// Threads per block of the one-thread-per-cell kernel.
/// ponytail: a literal, not derived — the fallback is a correctness path; derive
/// it when an untiled site turns out to matter for speed.
pub(crate) const NAIVE_THREADS: u64 = 256;

/// Can this site run on the GPU at all? Any float site whose reads are affine
/// in `(i, lane, k)`, or (generic-fusion `window`) in the fold's derived axes
/// `(k÷div, k%div)` — conv taps, which the one-thread-per-cell kernel
/// addresses. `window` off: those stay on the host.
pub(crate) fn gpu_site(site: &TileSite, profile: &TargetProfile, window: bool) -> bool {
    profile.gpu().is_some()
        && is_float(&site.elem)
        && (window || (site.a.ksplit.is_none() && site.b.ksplit.is_none()))
        && site.k > 0
        && site.rows * site.c > 0
        && uniform_aligned(site)
}

/// long-prefill `uniform-base`: a site whose operand address takes terms of a
/// number its map captures ([`mapal_ir::RadixTile::ua`]) — its launch moves
/// the operand's pointer by them, so it launches on its own.
pub(crate) fn uniform_site(site: &TileSite) -> bool {
    site.radix.as_deref().is_some_and(|r| r.uniform())
}

/// The pointer a launch moves by a captured number keeps the alignment every
/// kernel's widest load of that operand assumes: a vector of `lanes =
/// 16 / e` elements in the operand's own storage — 16 bytes of an `f32`
/// operand ([`GPU_VEC_BYTES`]); a narrow operand's register-staged vector is
/// the same `lanes` elements of its format (`<4 x bfloat>`, 8 bytes, ADR-0043),
/// and the kernels that copy 16 narrow bytes at once (`narrow-async`'s
/// `cp.async`, the 16-bit `mma` panels, `vec-cells`) take no mixed-radix site.
/// Each term's step, in the operand's bytes, is whole such vectors. Otherwise
/// the site is no device site.
pub(super) fn uniform_aligned(site: &TileSite) -> bool {
    let lanes = GPU_VEC_BYTES / elem_bytes_of(&site.elem);
    site.radix.as_deref().is_none_or(|r| {
        [(&r.ua, &site.a), (&r.ub, &site.b)]
            .iter()
            .all(|(ds, read)| {
                let e = read.elem_bytes(&site.elem);
                ds.iter().all(|d| (d.coef * e).is_multiple_of(lanes * e))
            })
    })
}

/// **S62 `place`: the tile sites of `f` that run on the device** — of
/// `plan`, the sites as the caller walks them. One answer for every reader
/// (the host walk, the residency plan, retention, row folds, the device
/// unit): a site one of them placed and another did not would be a kernel
/// reading a buffer nobody wrote — the [`super::super::packing_site`] lesson.
///
/// In walk order, a capable site ([`gpu_site`]) is a device site when an
/// operand is device-produced (`fed`: the host placement would copy it out —
/// the resident plan's whole point) or when its roofline says the device
/// finishes first ([`TargetProfile::device_pays`]: `2·rows·c·k` flops over
/// [`crate::profile::tile_bytes`], the dispatch rule's arithmetic). Without
/// measured costs every capable site is placed: the S52–S54 rule.
pub(crate) fn placement(
    ir: &CategoryIr,
    f: FuncId,
    plan: &TilePlan,
    profile: &TargetProfile,
    window: bool,
    carry: bool,
) -> SecondaryMap<MorphismId, ()> {
    let mut dev = SecondaryMap::new();
    if profile.gpu().is_none() {
        return dev;
    }
    for m in ir.topo_order(f) {
        let Some(site) = plan.sites.get(m).filter(|s| gpu_site(s, profile, window)) else {
            continue;
        };
        let src = ir.morphism(m).expect("site resolves").source;
        let fed = [site.a.slot, site.b.slot].into_iter().any(|k| {
            super::super::pair_source_ir(ir, src, k)
                .is_some_and(|o| device_made(ir, &dev, o, carry))
        });
        let flops = 2u64
            .saturating_mul(site.rows)
            .saturating_mul(site.c)
            .saturating_mul(site.k);
        let bytes = crate::profile::site_bytes(site);
        if fed || profile.device_pays(flops, bytes) {
            dev.insert(m, ());
        }
    }
    dev
}

/// `o` is produced on the device: by a placed site, or by a map or fold over
/// an array that is (a per-cell kernel, a row fold, an epilogue member — the
/// runs `devmap`, `rowfold` and `epilogue` keep beside their producer).
/// ponytail: a map that reads a device array but stays on the host counts too;
/// that over-approximation only ever keeps a reader on the device (today).
///
/// `carry` (onto-device `carry-made`): a loop's carried view is made where the
/// next state replacing it is (the back edge's slot), and a loop's exit value
/// where its payload is — a value the device made last trip is device-made.
/// A cycle through the back edge alone proves nothing (each object is asked
/// once per query).
pub(crate) fn device_made(
    ir: &CategoryIr,
    dev: &SecondaryMap<MorphismId, ()>,
    o: ObjectId,
    carry: bool,
) -> bool {
    device_made_at(ir, dev, o, carry, None, &mut Vec::new())
}

/// `assume`: a carried view taken as device-made (the induction step of
/// [`device_made`]'s carried rule).
pub(super) fn device_made_at(
    ir: &CategoryIr,
    dev: &SecondaryMap<MorphismId, ()>,
    o: ObjectId,
    carry: bool,
    assume: Option<ObjectId>,
    seen: &mut Vec<ObjectId>,
) -> bool {
    if assume == Some(o) {
        return true;
    }
    if seen.contains(&o) || seen.len() > 4096 {
        return false;
    }
    seen.push(o);
    if carry && let Some(next) = carried_from(ir, o) {
        if device_made_at(ir, dev, next, carry, assume, seen) {
            return true;
        }
        // residency `carry-induct`: a carried view the loop enters with on
        // the device (its init device-made) whose next state the device
        // makes from it (the view assumed device-made, every other operand
        // as found) is device-made on every trip, by induction over trips.
        // A loop state written in place each trip (a KV cache the prompt
        // filled on the device) is anchored by its init, not by the cycle.
        return assume.is_none()
            && carried_init(ir, o).is_some_and(|i| {
                device_made_at(ir, dev, i, carry, None, &mut Vec::new())
                    && device_made_at(ir, dev, next, carry, Some(o), &mut Vec::new())
            });
    }
    let Some(&p) = ir.in_edges(o).first() else {
        return false;
    };
    if dev.contains_key(p) {
        return true;
    }
    let morph = ir.morphism(p).expect("morphism resolves");
    if !matches!(morph.op, Operation::Map { .. } | Operation::Fold { .. }) {
        return false;
    }
    let pairs: Vec<ObjectId> = ir
        .in_edges(morph.source)
        .iter()
        .map(|&q| ir.morphism(q).expect("morphism resolves"))
        .filter(|q| matches!(q.op, Operation::Pair { .. }))
        .map(|q| q.source)
        .collect();
    if pairs.is_empty() {
        device_made_at(ir, dev, morph.source, carry, assume, seen)
    } else {
        pairs
            .into_iter()
            .any(|x| device_made_at(ir, dev, x, carry, assume, seen))
    }
}

/// The value a loop view `o` enters the loop with: the init's slot of a
/// canonical loop's merge projection, or the init of a one-value loop.
pub(super) fn carried_init(ir: &CategoryIr, o: ObjectId) -> Option<ObjectId> {
    let obj = ir.object(o)?;
    let (merge, k) = if obj.kind == mapal_ir::ObjectKind::LoopMerge {
        (o, None)
    } else {
        let &p = ir.in_edges(o).first()?;
        let morph = ir.morphism(p)?;
        match morph.op {
            Operation::Proj { index }
                if ir.object(morph.source)?.kind == mapal_ir::ObjectKind::LoopMerge =>
            {
                (morph.source, Some(index))
            }
            _ => return None,
        }
    };
    let plan = ir.loop_plan(ir.owner(merge), merge)?;
    match k {
        Some(k) => super::super::pair_source_ir(ir, plan.init, k),
        None => Some(plan.init),
    }
}

/// The value a loop view `o` carries in from the trip before (the back
/// edge's next-state slot of a canonical loop: `Proj{k}` of its merge, or the
/// merge of a one-value loop), or the payload a loop's exit value is.
pub(super) fn carried_from(ir: &CategoryIr, o: ObjectId) -> Option<ObjectId> {
    let obj = ir.object(o)?;
    let (merge, k) = if obj.kind == mapal_ir::ObjectKind::LoopMerge {
        (o, None)
    } else {
        let &p = ir.in_edges(o).first()?;
        let morph = ir.morphism(p)?;
        match morph.op {
            Operation::Proj { index }
                if ir.object(morph.source)?.kind == mapal_ir::ObjectKind::LoopMerge =>
            {
                (morph.source, Some(index))
            }
            Operation::LoopExit => return super::super::pair_source_ir(ir, morph.source, 0),
            _ => return None,
        }
    };
    let plan = ir.loop_plan(ir.owner(merge), merge)?;
    let next = super::super::pair_source_ir(ir, plan.back_route, 0)?;
    match k {
        Some(k) => super::super::pair_source_ir(ir, next, k),
        None => Some(next),
    }
}

/// The tile sites of `f` a GPU profile places (S54 `lane-stride`): the CPU's
/// [`CategoryIr::tile_plan`] plus the one-row strided-lane sites
/// ([`CategoryIr::tile_plan_strided`]) that are device sites — the
/// one-thread-per-cell kernel addresses any affine read. One answer for the
/// host walk and the device unit.
pub(crate) fn tile_plan(
    ir: &CategoryIr,
    f: FuncId,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
) -> TilePlan {
    let rungs = host_rungs(profile, contract, arms);
    if profile.gpu().is_none() {
        return ir.tile_plan(f);
    }
    let strict = ir.tile_plan(f);
    let mut plan = if rungs.lane_stride {
        ir.tile_plan_strided(f)
    } else {
        strict.clone()
    };
    // recurrence-precision `radix-tile`: a mixed-radix site where the staged
    // tile takes it (its kernels address digits; the naive kernel too, but a
    // site the tile refuses keeps the `nest-cell` it had).
    if rungs.radix_tile {
        for (m, s) in ir
            .tile_plan_radix(f)
            .into_iter()
            .filter(|(_, s)| s.post.is_none())
        {
            let l = launch(&s, profile, contract, arms);
            let staged = l.tiled.is_some() || l.mma.is_some();
            if staged && gpu_site(&s, profile, rungs.window) {
                plan.sites.insert(m, s);
            }
        }
    }
    // device-geometry `tile-post`: a sum of products whose body goes on past
    // its fold is a site whose store evaluates the rest (`post_sites`).
    for (m, s) in post_sites(ir, f, profile, contract, arms) {
        plan.sites.insert(m, s);
    }
    // ADR-0043: a sum of products over narrow storage (`tile_plan_narrow`),
    // where the device loads the format (the epilogue ones in `post_sites`).
    if rungs.narrow_tile {
        for (m, s) in ir.tile_plan_narrow(f).sites.iter() {
            let strided = s.a.clane > 1 || s.b.clane > 1;
            // model-ready `stride-lanes`: one row over a k-major `b` (a
            // 1-row projection) shares no panel — its nest cell's warp a
            // row reads `b` coalesced, the one-thread kernel `k` apart.
            let lone_k_major = rungs.stride_lanes && row_group(s) == 1 && b_k_major(s);
            if s.post.is_none()
                && !lone_k_major
                && (rungs.lane_stride || !strided)
                && gpu_site(s, profile, rungs.window)
            {
                plan.sites.insert(m, s.clone());
            }
        }
    }
    // S62: an extra is a site only where it is placed on the device.
    let dev = placement(ir, f, &plan, profile, rungs.window, rungs.carry_made);
    let extra: Vec<MorphismId> = plan
        .sites
        .iter()
        .filter(|&(m, _)| !strict.sites.contains_key(m) && !dev.contains_key(m))
        .map(|(m, _)| m)
        .collect();
    for m in extra {
        plan.sites.remove(m);
    }
    plan
}

/// A mixed-radix site whose `k` is not cut: every one but a `tile-post`
/// group's, whose partitions the staged and one-thread kernels offset
/// through its `k` digits (`z·kext` added before them) and whose
/// candidates are the ordered kernel and the workspace split
/// ([`group_order`]: no atomic, no stream-K).
pub(super) fn radix_whole(site: &TileSite) -> bool {
    site.radix.is_some() && site.post.is_none()
}

/// device-geometry `head-group`: the rows a tile block may span — a batch's
/// (the least stride of `b`'s row digits, which every block-row start then
/// leaves unchanged within the block), else all of them.
pub(crate) fn row_group(site: &TileSite) -> u64 {
    site.radix
        .as_deref()
        .and_then(|r| r.b.as_ref())
        .and_then(|b| b.i.iter().map(|d| d.stride).min())
        .unwrap_or(site.rows)
}

/// device-geometry `tile-post`: `b` walks the fold contiguously (its unit
/// `k` step is one element) and its lanes not — the k-major layout a warp
/// folding one cell across its lanes reads coalesced (`cell-lanes`), where a
/// tile's lanes read it one element a row.
pub(super) fn b_k_major(site: &TileSite) -> bool {
    let unit = |ds: &[mapal_ir::AxisDigit]| ds.iter().any(|d| d.stride == 1 && d.coef == 1);
    match site.radix.as_deref().and_then(|r| r.b.as_ref()) {
        Some(b) => unit(&b.k) && !unit(&b.lane),
        None => site.b.ck == 1 && site.b.clane != 1,
    }
}

/// device-geometry `tile-post`: the [`CategoryIr::tile_plan_post`] sites a
/// GPU build realizes — each one's epilogue evaluated in its store, which
/// exists exactly where an epilogue group can (`resident.rs:epilogues`): a
/// resident function (no retained site unless `retain-resident`), the
/// `epilogue` rung at the site's fill, the map outside every gated guard
/// arm, an `i32` cell index, numbers read by value that are the host's
/// (none a whole fold makes), a `b` panel shared by more than one row or
/// read along its lanes ([`b_k_major`]), a body the device runs with the host's text,
/// and its operand pointers within one launch's arguments. A site one of
/// these refuses stays the per-cell kernel it was.
pub(crate) fn post_sites(
    ir: &CategoryIr,
    f: FuncId,
    profile: &TargetProfile,
    contract: bool,
    arms: crate::GpuArms,
) -> Vec<(MorphismId, TileSite)> {
    let rungs = host_rungs(profile, contract, arms);
    if profile.gpu().is_none()
        || !rungs.tile_post
        || !rungs.resident
        || !rungs.epilogue
        || !(rungs.retain_resident
            || super::super::retain::retained_sites(ir, f, profile, rungs).is_empty())
    {
        return Vec::new();
    }
    let gated: Vec<MorphismId> = ir
        .guard_plan(f)
        .into_iter()
        .filter(mapal_ir::GuardSite::gated)
        .flat_map(|g| g.on_true.own.into_iter().chain(g.on_false.own))
        .collect();
    let mut out = Vec::new();
    // A mixed-radix post site (`head-group` attention) where the staged tile
    // takes it, as `radix-tile`'s sites.
    let radix: Vec<(MorphismId, TileSite)> = match rungs.radix_tile {
        true => ir
            .tile_plan_radix(f)
            .into_iter()
            .filter(|(_, s)| s.post.is_some())
            .filter(|(_, s)| {
                let l = launch(s, profile, contract, arms);
                l.tiled.is_some() || l.mma.is_some()
            })
            .collect(),
        false => Vec::new(),
    };
    let posts = ir.tile_plan_post(f);
    let whole = ir.whole_fold(f);
    // ADR-0043: the narrow-operand sums of products with an epilogue.
    let narrow = match rungs.narrow_tile {
        true => ir.tile_plan_narrow(f),
        false => TilePlan {
            sites: SecondaryMap::new(),
        },
    };
    let narrow_posts = narrow.sites.iter().filter(|(_, s)| s.post.is_some());
    for (m, s) in posts
        .sites
        .iter()
        .chain(radix.iter().map(|(m, s)| (*m, s)))
        .chain(narrow_posts)
    {
        let Some(p) = &s.post else { continue };
        let morph = ir.morphism(m).expect("site resolves");
        let Operation::Map { body, captures } = morph.op else {
            continue;
        };
        let strided = s.a.clane > 1 || s.b.clane > 1;
        // A number the epilogue reads by value is the host's: one a whole
        // fold makes stays in its device buffer (`scalar-fold`), read by the
        // per-cell kernel there — a by-value read would wait on the device
        // each trip.
        let dev_number = p.uniform.iter().any(|&slot| {
            super::super::pair_source_ir(ir, morph.source, slot)
                .and_then(|o| ir.in_edges(o).first().copied())
                .is_some_and(|q| whole.contains_key(q))
        });
        let index_i32 = super::super::pair_source_ir(ir, morph.source, captures)
            .and_then(|o| ir.object(o))
            .is_some_and(|o| matches!(&o.ty, Ty::Array { elem, .. } if **elem == Ty::i32()));
        // One row a `b` panel (nothing to share it with) over a k-major `b`:
        // the per-cell kernel's warp a cell reads `b` coalesced and skips the
        // cells a guard masks; the tile would read it a row an element.
        let lone_k_major = row_group(s) == 1 && b_k_major(s);
        if gated.contains(&m)
            || dev_number
            || lone_k_major
            || (strided && !rungs.lane_stride)
            || !index_i32
            || !device_body(ir, body, 0)
            || 3 + p.reads.len() + 1 + p.uniform.len() > LAUNCHV_MAX
            || !gpu_site(s, profile, rungs.window)
            || !launch(s, profile, contract, arms).rungs.epilogue
        {
            continue;
        }
        out.push((m, s.clone()));
    }
    out
}

/// A body the device runs with the host's text, as far as the graph tells:
/// no `Call`, no float `%` (NVPTX's `frem` is not the host's `fmodf`), every
/// nested body the same. The emitter's `device_leaf` adds the function
/// attributes (clean) it alone knows.
pub(super) fn device_body(ir: &CategoryIr, body: FuncId, depth: u32) -> bool {
    let Some(fd) = ir.func(body) else {
        return false;
    };
    depth < 8
        && fd.morphisms.iter().all(|&g| {
            let Some(x) = ir.morphism(g) else {
                return false;
            };
            let float_mod = x.op == Operation::Mod
                && ir
                    .object(x.source)
                    .and_then(|o| o.ty.component_ty(0).cloned())
                    .is_some_and(|t| is_float(&t));
            !float_mod
                && match x.op {
                    Operation::Call(_) => false,
                    Operation::Map { body, .. } | Operation::Fold { body, .. } => {
                        device_body(ir, body, depth + 1)
                    }
                    _ => true,
                }
        })
}
