//! **compile-time-graph B: launch regions.** One trip of a device loop's
//! advance cone is a static sequence of device ops, known at compile time: the
//! emitter brackets it (`mapal_gpu_region_begin` / `_end`) and tags each op
//! with its index in the region and the earlier ops it depends on — read off
//! the dataflow: an op waits for an earlier one that writes bytes it reads
//! (RAW), reads bytes it writes (WAR) or writes bytes it writes (WAW); its zero
//! pass waits only for the earlier readers and writers of its output. Bytes are
//! an object's device buffer: a carried state's views, next value and exit
//! share one; arena members share bytes where their offsets overlap. The
//! runtime records the trip's launches and zero passes as graph nodes with
//! those edges and replays them as one CUDA graph (`mapal_gpu.c`).

use super::*;

/// The most ops a region tags (a dependency is one bit of an `i64`).
const REGION_OPS: usize = 64;

impl<'a> FnEmit<'a> {
    /// `m` launches device work in the resident plan (a unit, cell, row fold,
    /// device build or tile site that is not another op's member).
    fn device_op(&self, m: MorphismId) -> bool {
        let Some(p) = &self.resident else {
            return false;
        };
        if p.skip.contains_key(m) {
            return false;
        }
        p.units.contains_key(m)
            || p.cells.contains_key(m)
            || p.rowfold.contains_key(m)
            || p.devgen.contains_key(m)
            || self.device.contains_key(m)
    }

    /// The arrays a morphism reads: its source, or each slot of the product
    /// it reads.
    fn op_inputs(&self, m: MorphismId) -> Vec<ObjectId> {
        let mo = self.ir.morphism(m).expect("morphism resolves");
        let slots = match mo.op {
            Operation::Map { captures, .. } | Operation::Fold { captures, .. } => captures + 1,
            _ => 0,
        };
        let mut out: Vec<ObjectId> = (0..=slots)
            .filter_map(|s| self.pair_source(mo.source, s))
            .collect();
        if out.is_empty() {
            out.push(mo.source);
        }
        out.retain(|&o| matches!(self.obj_ty(o), Ty::Array { .. }));
        out
    }

    /// `(reads, writes)` of device op `m`: a unit's members' inputs and stored
    /// values, a site's and its epilogue chain's, else the op's own.
    fn op_access(&self, m: MorphismId) -> (Vec<ObjectId>, Vec<ObjectId>) {
        let plan = self.resident.as_ref().expect("a region is resident");
        let target = self.ir.morphism(m).expect("morphism resolves").target;
        if let Some(u) = plan.units.get(m) {
            let outs: Vec<ObjectId> = u.members.iter().map(|mb| mb.out).collect();
            let reads = u
                .members
                .iter()
                .flat_map(|mb| self.op_inputs(mb.m))
                .filter(|o| !outs.contains(o))
                .collect();
            let writes = u
                .members
                .iter()
                .filter(|mb| mb.store)
                .map(|mb| mb.out)
                .collect();
            return (reads, writes);
        }
        let mut reads = self.op_inputs(m);
        let mut writes = vec![target];
        if let Some(g) = self.groups.get(m) {
            for mb in &g.chain {
                reads.extend(self.op_inputs(mb.m));
                writes.push(mb.out);
            }
            reads.retain(|o| !writes.contains(o));
        }
        (reads, writes)
    }

    /// The buffer an object's bytes live in: a carried state's views, next
    /// value, exit and merge share the carry's.
    fn bytes_key(&self, o: ObjectId) -> ObjectId {
        let plan = self.resident.as_ref().expect("a region is resident");
        plan.carries
            .iter()
            .find(|c| c.views.contains(&o) || c.exit == Some(o) || c.next == o || c.merge == o)
            .map_or(o, |c| c.key())
    }

    /// Do `x` and `y` share device bytes (one buffer, or overlapping arena
    /// offsets)?
    fn same_bytes(&self, x: ObjectId, y: ObjectId) -> bool {
        if self.bytes_key(x) == self.bytes_key(y) {
            return true;
        }
        let plan = self.resident.as_ref().expect("a region is resident");
        let Some(a) = &plan.arena else {
            return false;
        };
        match (a.offset.get(x), a.offset.get(y)) {
            (Some(&ox), Some(&oy)) => {
                let (bx, by) = (
                    self.array_bytes(&self.obj_ty(x)),
                    self.array_bytes(&self.obj_ty(y)),
                );
                ox < oy + by && oy < ox + bx
            }
            _ => false,
        }
    }

    /// Open the launch region of a loop's trip over its advance cone: tag
    /// every device op with its compile-time dependencies. Nothing is emitted
    /// under the OFF arm (`--no-launch-graph`), without a resident plan, for a
    /// trip whose plan moves bytes across the bus or copies on the host (a host
    /// step inside would end the recording: the trip is not one schedule), for
    /// a trip of fewer than two ops (no launch to save) or more than
    /// [`REGION_OPS`], or where the part's graph facts say one graph launch
    /// with every node patched issues no faster than the ops one by one.
    pub(crate) fn region_open(&mut self, plan: &mapal_ir::LoopPlan) {
        if !self.gpu_arms.graph || self.resident.is_none() || self.region.is_some() {
            return;
        }
        let (copy, h2d, d2h, _) = self.trip_moves(plan);
        if copy + h2d + d2h > 0 {
            return;
        }
        let merge = plan.merge;
        let ops: Vec<MorphismId> = plan
            .advance_order
            .iter()
            .copied()
            .filter(|&m| !self.gated.contains_key(m) && self.device_op(m))
            .collect();
        if ops.len() < 2 || ops.len() > REGION_OPS {
            return;
        }
        // The part's graph facts (calib `graph`): the region pays where one
        // graph launch with every node patched issues faster than the ops one
        // by one. A part without them takes every region.
        if let Some(d) = self.profile.gpu().and_then(|g| g.cost)
            && d.graph_launch_ns > 0
        {
            let n = ops.len() as u64;
            if d.graph_launch_ns + n * d.set_params_ns >= n * d.issue_ns {
                return;
            }
        }
        let acc: Vec<(Vec<ObjectId>, Vec<ObjectId>)> =
            ops.iter().map(|&m| self.op_access(m)).collect();
        let hit = |xs: &[ObjectId], ys: &[ObjectId]| {
            xs.iter()
                .any(|&x| ys.iter().any(|&y| self.same_bytes(x, y)))
        };
        let mut tags = SecondaryMap::new();
        let mut edges = 0;
        for (j, &m) in ops.iter().enumerate() {
            let (mut deps, mut out) = (0u64, 0u64);
            let (rj, wj) = (&acc[j].0, &acc[j].1);
            for (i, (ri, wi)) in acc.iter().enumerate().take(j) {
                if hit(wi, rj) || hit(ri, wj) || hit(wi, wj) {
                    deps |= 1 << i;
                    edges += 1;
                }
                if hit(ri, wj) || hit(wi, wj) {
                    out |= 1 << i;
                }
            }
            tags.insert(m, (j as u32, deps, out));
        }
        let key = format!("region|{}|{merge:?}", self.fnames[self.f]);
        let rid = key.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
        }) as i64;
        self.line(format!(
            "; mapal.region id={rid} ops={} edges={edges}",
            ops.len()
        ));
        self.line(format!("call void @mapal_gpu_region_begin(i64 {rid})"));
        self.region = Some((rid as u64, tags));
    }

    /// Close the open region: the runtime replays the trip's graph.
    pub(crate) fn region_close(&mut self) {
        if let Some((rid, _)) = self.region.take() {
            self.line(format!(
                "call void @mapal_gpu_region_end(i64 {})",
                rid as i64
            ));
        }
    }

    /// Before device op `m` of the open region: its tag. `true` when one was
    /// emitted (the op is then untagged after it).
    pub(super) fn region_tag(&mut self, m: MorphismId) -> bool {
        let Some(&(op, deps, out)) = self.region.as_ref().and_then(|(_, t)| t.get(m)) else {
            return false;
        };
        self.line(format!(
            "call void @mapal_gpu_tag(i32 {op}, i64 {}, i64 {})",
            deps as i64, out as i64
        ));
        true
    }

    /// After a tagged op: later device work of the trip is untagged (it waits
    /// for everything before it) until the next tag.
    pub(super) fn region_untag(&mut self) {
        self.line("call void @mapal_gpu_tag(i32 -1, i64 0, i64 0)".to_string());
    }
}
