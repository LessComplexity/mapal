# Runtime — suggestions

CT-derived only. Each cites the rule it applies.

## 1. `Work` should not carry an `Arc<Run>` — §5 deduce, don't store

Every `WorkItem` clones an `Arc`, so a dispatch of S slices pays 2S atomic refcount
operations to re-state a fact the run already knows: the lane is executing *this* run.
The run outlives every item it queued (`mapal_par_finish` is `help_until(remaining == 0)`),
so the lifetime is already implied by the protocol and the refcount is a stored copy of
a deduced morphism.

## 2. `take_any`'s miss path is the last O(N) — §4.5 Law 4

A failed steal locks every lane's `Mutex<VecDeque>` in turn. There is no shared record
of which lanes hold work, so each thread rediscovers it by polling all of them —
FRAMEWORK §5 "one source of truth for shared structure" failing on the *idle* set.
`Pool::pending` (added S50 P1) is that record for the spin loop; the steal path does not
use it yet.

## 3. The ARCHITECTURE §4 composition rules want pins — §6.6

Rule 1 (unlock before publish) was violated for the whole of S50 P2 and surfaced only
because P1 made the host fast enough to observe it. A rule that can only be caught by an
unrelated performance change is an undocumented exception waiting to rot. Each of the six
is testable from `mapal-rt`'s own suite.

## 4. `path_plan_is_acyclic` should become unnecessary — §6.6

`llvm/src/lib.rs:258` carries a `ponytail:` note saying to delete the fallback "when
`path_plan`'s total-DAG contract is enforced before backend entry". S49 removed the
sibling-merge cycle class but not the general one, and a cyclic plan is still silently
demoted to sequential emission — the failure mode that hid every chained matmul for
nine sessions. Either enforce the contract or make the demotion loud.

## 5. The dispatch verdict is compile-time; the cost that decides it is not — §4.2

`DispatchCost::dispatch_ns` is a property of the *runtime*, measured on a program, and
it is consumed by the *emitter*. That is a `TrnLoc` fact crossing into a compile-time
decision with no mechanism keeping the two in agreement — exactly the drift shape §6.3
exists to prevent. A runtime self-calibration at pool creation, published where the next
build can read it, would make it a measurement instead of a written-down number.
