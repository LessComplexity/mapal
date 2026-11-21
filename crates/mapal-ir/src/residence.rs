//! **`residence_plan`: where a value's bytes live between its producer and its
//! last reader.** A value's bytes stay in a cache the whole device shares (the
//! GPU's L2) when everything live from the moment it is made to the moment
//! its last reader runs fits there; else its reader fetches them from DRAM. A
//! value with no array bytes (a number) lives on chip. The fact stores, per
//! array object, its bytes and that working set: the peak, over the positions
//! from its producer to its death ([`CategoryIr::last_use_plan`]), of the
//! bytes of every array live at the position — each storage class
//! ([`CategoryIr::access_plan`]: a carried buffer and its views) counted once,
//! live from its first producer to its last death; an index domain (`Iota`)
//! has none (its readers evaluate the index). Arrays count whole (a
//! partial reader's footprint is not subtracted), so the working set is an
//! upper bound and a value placed in L2 by it is in L2. The capacity is the
//! part's ([`ResidencePlan::at`]): the backend's `TargetProfile`.

use slotmap::SecondaryMap;

use crate::graph::{CategoryIr, FuncId, ObjectId, Operation};
use crate::ty::Ty;

/// Where a value's bytes live between its producer and its last reader.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Residence {
    /// No array bytes: a number in registers.
    Chip,
    /// The working set fits the shared cache.
    L2,
    /// It does not: the reader fetches from DRAM.
    Dram,
}

/// One function's residence fact ([`CategoryIr::residence_plan`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ResidencePlan {
    bytes: SecondaryMap<ObjectId, u64>,
    working: SecondaryMap<ObjectId, u64>,
}

impl ResidencePlan {
    /// The array bytes of `o` (0: a number, a product, a view's own).
    pub fn bytes(&self, o: ObjectId) -> u64 {
        self.bytes.get(o).copied().unwrap_or(0)
    }

    /// The peak bytes live while `o` waits for its last reader (0 for an
    /// object without array bytes).
    pub fn working_set(&self, o: ObjectId) -> u64 {
        self.working.get(o).copied().unwrap_or(0)
    }

    /// Where `o`'s bytes live on a part whose shared cache holds `cache`
    /// bytes.
    pub fn at(&self, o: ObjectId, cache: u64) -> Residence {
        match self.working.get(o) {
            None => Residence::Chip,
            Some(&w) if w <= cache => Residence::L2,
            Some(_) => Residence::Dram,
        }
    }
}

impl CategoryIr {
    /// **The residence fact of `f`** (module doc): stored on the graph, built
    /// from `last_use_plan` and `access_plan`.
    pub fn residence_plan(&self, f: FuncId) -> ResidencePlan {
        self.fact(f, |s| &s.residence_plan, || self.residence_plan_compute(f))
    }

    fn residence_plan_compute(&self, f: FuncId) -> ResidencePlan {
        let mut plan = ResidencePlan::default();
        if self.func(f).is_none() {
            return plan;
        }
        let lup = self.last_use_plan(f);
        let acc = self.access_plan(f);
        let array_bytes = |o: ObjectId| match self.object(o).map(|x| &x.ty) {
            Some(Ty::Array { elem, size }) => match **elem {
                Ty::Float { bits } | Ty::Int { bits, .. } => u64::from(bits).div_ceil(8) * size,
                Ty::Narrow(n) => n.bytes() * size,
                Ty::Bool => *size,
                _ => 0,
            },
            _ => 0,
        };
        // Each class: its bytes and live interval `[birth, death]` over the
        // objects of the class that some morphism writes (or that enter the
        // function: born at 0).
        let mut class: SecondaryMap<ObjectId, (u64, u32, u32)> = SecondaryMap::new();
        let mut members: Vec<ObjectId> = Vec::new();
        let objects = self.func_objects(f);
        // An escaping value lives to the function's end.
        let end = objects
            .iter()
            .flat_map(|&o| self.in_edges(o))
            .filter_map(|&m| lup.position(m))
            .max()
            .unwrap_or(0);
        for o in objects {
            let b = array_bytes(o);
            if b == 0 {
                continue;
            }
            // An index domain (`Iota`) has no bytes: its readers evaluate
            // the index.
            if self
                .in_edges(o)
                .iter()
                .any(|&m| self.morphisms[m].op == Operation::Iota)
            {
                continue;
            }
            let made: Vec<u32> = self
                .in_edges(o)
                .iter()
                .filter(|&&m| !acc.writes(m).is_empty())
                .filter_map(|&m| lup.position(m))
                .collect();
            let entry = self.in_edges(o).is_empty();
            if made.is_empty() && !entry {
                continue;
            }
            let birth = made.iter().copied().min().unwrap_or(0);
            let death = if lup.escapes(o) {
                end
            } else {
                lup.death(o)
                    .unwrap_or(birth)
                    .max(lup.view_death(o).unwrap_or(0))
                    .max(birth)
            };
            let c = acc.class(o);
            let e = class
                .entry(c)
                .expect("class key")
                .or_insert((0, birth, death));
            e.0 = e.0.max(b);
            e.1 = e.1.min(birth);
            e.2 = e.2.max(death);
            plan.bytes.insert(o, b);
            members.push(o);
        }
        let n = class.values().map(|c| c.2 as usize + 1).max().unwrap_or(0);
        // The bytes live at each position, then a sparse table of maxima.
        let mut diff = vec![0i128; n + 1];
        for &(b, lo, hi) in class.values() {
            diff[lo as usize] += i128::from(b);
            diff[hi as usize + 1] -= i128::from(b);
        }
        let mut live = Vec::with_capacity(n);
        let mut run = 0i128;
        for d in diff.iter().take(n) {
            run += d;
            live.push(run.max(0) as u64);
        }
        let mut table = vec![live];
        let mut w = 1;
        while 2 * w <= n {
            let prev = table.last().expect("a level");
            let next = (0..=n - 2 * w).map(|i| prev[i].max(prev[i + w])).collect();
            table.push(next);
            w *= 2;
        }
        let peak = |lo: usize, hi: usize| {
            let k = (usize::BITS - 1 - (hi - lo + 1).leading_zeros()) as usize;
            table[k][lo].max(table[k][hi + 1 - (1 << k)])
        };
        for o in members {
            let (_, lo, hi) = class[acc.class(o)];
            plan.working.insert(o, peak(lo as usize, hi as usize));
        }
        plan
    }
}
