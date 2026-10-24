"""S64 evidence pass — the intent registry: per rung, what the emitted IR must become.

Data only; `evidence.py check` evaluates it. A rung's `on` checks run on a kernel whose plan line lists
it (`; mapal.plan … rungs_on=`), its `off` checks where it is absent or the bundle arm forced it off;
`applies` (or `applies_on` / `applies_off`) gates by what the emitter decided (`plan.*`: kind, face,
geometry, stages, combine). A check is `(label, expression)` over the facts of one kernel:
  plan.*     the emitter's `; mapal.plan` line          ptx.*, ptx.hot.*   PTX counts, whole / hottest loop
  ptxas.*    regs spill_st spill_ld stack static_smem   sass.*, sass.hot.* cuobjdump -sass, whole / hot loop
  occ.*      blocks_per_sm warps_per_sm waves resident_warps_per_sm sms (occ.py, the part's limits)
  remarks.opt.* / remarks.llc.*  opt / llc optimization remarks for the kernel (full_unroll, insns, …)
  launch.*   the stem's `.launch` line (S, combine, reduce)   exec.*  the box's JIT image (regs local smem)
  table.rungs_on  the part's GPU_RUNGS_OFF table at the kernel's fill and face, under the bundle arm
Host rungs (`host=True`) run once per stem over `host.*` (`; mapal.host`, `mapal_gpu_*` call counts,
`host.obj.*` mnemonics of the host object, `host.remarks.*`) and `dev.*` (kernel kinds).
A missing fact is `n/a`, never a pass. A rung with no static signature says so (`note`) and has no
checks: its evidence is the arm's diff (`evidence.py diff`), not a predicate here.
"""

TILE = "plan.kind == 'tile'"
SCALAR = "plan.kind in ('tile', 'naive')"
ANY = "plan.kind in ('tile', 'naive', 'mma', 'wgmma')"

INTENT = {
    # ---- what every scalar / tensor kernel must keep ------------------------------------------------
    'face-fma': dict(applies=SCALAR + " and plan.face == 'OrderedFma'", on=[
        ('PTX hot loop: fma.rn, no separate mul', 'ptx.hot.fma > 0 and ptx.hot.mul == 0'),
        ('SASS hot loop: FFMA, no FMUL', 'sass.hot.ffma > 0 and sass.hot.fmul == 0')]),
    'face-separate': dict(applies=SCALAR + " and plan.face == 'SeparateMulAdd'", on=[
        ('PTX hot loop: mul.rn + add.rn pairs, no fma', 'ptx.hot.fma == 0 and ptx.hot.mul > 0 and ptx.hot.mul == ptx.hot.add'),
        ('SASS hot loop: FMUL + FADD, never fused by ptxas', 'sass.hot.ffma == 0 and sass.hot.fmul > 0')]),
    'face-registers': dict(applies=TILE, on=[
        ('every tm*tn accumulator live across the k loop (SROA kept them in registers)', 'sass.hot.acc >= plan.tm * plan.tn'),
        ('no spill', 'ptxas.spill_st == 0 and ptxas.spill_ld == 0'),
        ('the derived blocks/SM are resident at ptxas registers + smem', 'occ.blocks_per_sm >= plan.blocks_per_sm')]),
    'exec': dict(applies=ANY + " and exec.regs >= 0", on=[
        ('JIT image registers == ptxas -v', 'exec.regs == ptxas.regs'),
        ('JIT image local bytes == ptxas stack frame', 'exec.local == ptxas.stack')]),
    'module': dict(applies="plan.kind != 'gen'", on=[
        ('every kernel in the device module is launched by the host', 'mod.launched > 0')]),
    'arch-rungs': dict(applies=ANY, on=[
        ("the kernel's rungs are the part's measured table at its fill and face", 'set(plan.rungs_on) == set(table.rungs_on)')],
        off=[('every rung on', 'set(plan.rungs_on) == set(table.rungs_on)')]),
    # ---- FMA tile rungs ------------------------------------------------------------------------------
    'unroll': dict(applies=TILE + ' and plan.bk > 1', on=[
        ('LLVM fully unrolled the bk-step loop (loop-unroll FullyUnrolled = bk)', 'remarks.opt.full_unroll == plan.bk'),
        ('one hot-loop trip holds >= bk k-steps', 'sass.hot.ksteps >= plan.bk')],
        off=[('LLVM did not fully unroll it', 'remarks.opt.full_unroll != plan.bk'),
             ('fewer than bk k-steps per trip', 'sass.hot.ksteps < plan.bk')]),
    'prefetch': dict(applies=TILE + ' and plan.stages == 1 and sass.hot.ldg > 0', on=[
        ("the next panel's LDGs are covered by this panel's math", 'sass.hot.ldg_cover_min > 0')],
        off=[('panel LDGs consumed before the math', 'sass.hot.ldg_cover_min <= 0')]),
    'async': dict(applies_on=TILE + ' and plan.stages >= 2', on=[
        ('cp.async + wait_group in PTX', 'ptx.cp_async > 0 and ptx.cp_async_wait > 0'),
        ('LDGSTS in the hot loop, no register-staged STS', 'sass.hot.ldgsts > 0 and sass.hot.sts == 0')],
        # asyncstage streams through cp.async whatever `async` says (stages > 2 implies it).
        applies_off=TILE + ' and plan.stages < 2', off=[('register staging: no cp.async', 'ptx.cp_async == 0 and sass.ldgsts == 0')]),
    'asyncstage': dict(applies_on=TILE + ' and plan.stages > 2', on=[
        ('stages-2 groups left in flight at the wait', 'ptx.wait_n_max == plan.stages - 2'),
        ('LDGSTS', 'sass.ldgsts > 0')]),
    'interleave': dict(applies_on=TILE + ' and plan.tm * plan.tn > 1', on=[
        ('128-bit shared fragment loads', 'sass.hot.lds128 > 0')]),
    'min-blocks': dict(applies="plan.kind in ('tile', 'mma', 'wgmma')", on=[
        ('.minnctapersm = the derived blocks/SM', 'ptx.minnctapersm == plan.blocks_per_sm'),
        ('ptxas met it without spilling', 'ptxas.spill_st == 0')],
        off=[('no .minnctapersm', 'ptx.minnctapersm == 0')]),
    'warp': dict(applies=TILE, note='lane-to-thread map only: no static SASS signature (diff the arm)'),
    # `fill` (profile.rs gpu_tile_with): shrink the tile until the grid gives every scheduler two warps,
    # down to the floor (tn = one vector, tm = one vector or, under fill-rows, one row).
    'warp-fill': dict(applies_on=TILE, on=[
        ('every warp scheduler holds two warps, or the tile is at the fill floor',
         'occ.resident_warps_per_sm >= 8 or (plan.tn <= 4 and plan.tm <= 4)')]),
    'fill-rows': dict(applies=TILE, note='the fill floor one row lower; fires only where the vector floor fails (diff the arm)'),
    # PTX keeps shared addresses 64-bit (ptxas narrows them): the realized index width is SASS's.
    'idx32': dict(applies_on=SCALAR + " and plan.idx == 'i32'", on=[
        ('hot loop: at most one 64-bit address IMAD.WIDE per global access',
         'sass.hot.imad_wide <= sass.hot.ldg + sass.hot.ldgsts')]),
    # NVPTX re-bases a long run of immediate offsets (`[%rd84]`, then `[%rd86+…]`), so a PTX base count
    # does not separate the rung (S64 c/: 3 bases with it ON, 2 with it OFF on the same kernels).
    'affine': dict(applies=TILE, note='one base register per operand: PTX base counts do not separate it (diff the arm)'),
    'pad': dict(applies=TILE, note='bank geometry: no static SASS signature (diff the arm)'),
    'wide-fma': dict(applies_on=TILE + ' and plan.held > 0', on=[
        ('the whole-file tile: its accumulators live in the hot loop', 'sass.hot.acc >= plan.tm * plan.tn'),
        ('ptxas gave the thread what the tile holds', 'ptxas.regs >= plan.held'),
        ('no spill', 'ptxas.spill_st == 0')]),
    'raster': dict(applies=TILE, note='block-id remap: no static SASS signature (diff the arm)'),
    'l2-gate': dict(applies=TILE, note='decides raster: no static SASS signature'),
    'row-fit': dict(applies_on=TILE, on=[('the block rows fit the site', 'plan.ty * plan.tm <= plan.rows')]),
    'wave-gate': dict(applies=TILE, note='chooses the kernel kind: no static SASS signature (diff the arm)'),
    'grid-share': dict(applies_on=TILE, on=[
        ('residency sized to the grid share', 'plan.blocks_per_sm <= max(1, ceil((plan.ctas if plan.combine == "streamk" '
         'else plan.gx * plan.gy) / occ.sms))')]),
    'panel-regs': dict(applies_on=TILE + ' and plan.stages == 1 and plan.held == 0', on=[
        ('the prefetch panel fits the register file: no spill', 'ptxas.spill_st == 0')]),
    'cell-fill': dict(applies_on=TILE + ' and plan.tm * plan.tn == 1', on=[
        ('one output per thread, staged through shared memory', 'sass.hot.lds > 0')]),
    # ---- fold realizations (ADR-0040 D4) --------------------------------------------------------------
    'splitk': dict(applies_on="plan.combine == 'splitk'", on=[
        ('S partial folds, then one reduce launch', "launch.S == plan.S and launch.combine == 'splitk'"),
        ('no atomics', 'ptx.atom_add + ptx.red_add == 0')]),
    'splitk-atomic': dict(applies_on="plan.combine == 'atomic'", on=[
        ('one red.global.add per accumulator', 'ptx.atom_add + ptx.red_add >= (plan.tm * plan.tn if plan.kind == "tile" else 1)'),
        ('no workspace, no reduce launch', "launch.combine == 'atomic'")]),
    'streamk': dict(applies_on="plan.combine == 'streamk'", on=[
        ('one wave of blocks', 'plan.ctas <= occ.blocks_per_sm * occ.sms'),
        ('pieces combine by atomic add', 'ptx.atom_add + ptx.red_add > 0'),
        ('the launch is stream-K', "launch.combine == 'streamk'")]),
    # ---- tensor-core kernels ---------------------------------------------------------------------------
    'mma': dict(applies="plan.kind == 'mma'", on=[
        ('mma.sync in PTX', 'ptx.mma > 0'), ('HMMA in SASS', 'sass.hmma > 0')]),
    'wide': dict(applies="plan.kind == 'mma'", note='warp-tile size: no static SASS predicate (diff the arm)'),
    'ldmatrix': dict(applies="plan.kind == 'mma'", on=[
        ('ldmatrix in PTX', 'ptx.ldmatrix > 0'), ('LDSM in SASS', 'sass.ldsm > 0')],
        off=[('no ldmatrix / LDSM', 'ptx.ldmatrix == 0 and sass.ldsm == 0')]),
    'wgmma-hopper': dict(applies="plan.kind == 'wgmma'", on=[
        ('wgmma.mma_async in PTX', 'ptx.wgmma > 0'), ('HGMMA in SASS', 'sass.hgmma > 0')]),
    'pdl': dict(applies=ANY + " and sm in ('sm_90a', 'sm_120')", on=[
        ('griddepcontrol at kernel entry', 'ptx.griddep > 0')],
        off=[('no griddepcontrol', 'ptx.griddep == 0')]),
    # ---- host rungs (per stem) ---------------------------------------------------------------------------
    'resident': dict(host=True, applies='host.device_sites > 0', on=[
        ('no S52 round trip (mapal_gpu_tile)', 'host.calls.tile == 0')],
        off=[('every site a round trip', 'host.calls.tile > 0')]),
    'devgen': dict(host=True, applies='dev.gen > 0', on=[
        ('the host launches every closed-form producer on the device', 'host.calls.gen >= dev.gen')]),
    # Loop-carried objects keep their own allocations (`arena` lays out straight-line objects only).
    'arena': dict(host=True, applies='host.loop == 0 and host.calls.alloc + host.calls.arena > 0', on=[
        ('straight-line objects in one per-function arena', 'host.calls.arena > 0')],
        off=[('per-object allocations', 'host.calls.arena == 0')]),
    'eager-init': dict(host=True, applies='host.device_sites > 0', on=[
        ("one init, main's first call", 'host.calls.init == 1')]),
    'epilogue': dict(host=True, applies='dev.epi > 0', on=[
        ('epilogue kernels launched over n pointers', 'host.calls.launchv > 0')]),
    'devmap': dict(host=True, applies='dev.cell > 0', on=[
        ('cell kernels launched over n pointers', 'host.calls.launchv > 0')]),
    'streams': dict(host=True, note='stream ids in launchv: diff the arm (`.fs`)'),
    'splitk-atomic-zero': dict(host=True, applies="launch.combine in ('atomic', 'streamk')", on=[
        ('the output is zeroed before its partials add', 'host.calls.zero >= 1')]),
    # Only where LLVM vectorized host code at all (a program whose every site is on the device has none).
    'hostopt': dict(host=True, applies='host.remarks.vectorized + host.remarks.slp > 0', on=[
        ('the host object uses 256-bit AVX2 vectors (the x86-64-v3 profile)', 'host.obj.ymm_ops > 0')],
        off=[('generic x86-64: no 256-bit vectors', 'host.obj.ymm_ops == 0')]),
    'place': dict(host=True, note='site placement: diff the `.off-place` arm (device_sites, launches)'),
    # S65 `by-ref` (rule R): each program loop's `; mapal.move` line is the facts' derivation of what crosses
    # per trip; `host.move` is what the host IR does from the loop header to its latch (evidence.py
    # move_facts). Any copy the facts do not require is a mismatch. CPU builds included (no bus: only copies).
    'by-ref': dict(host=True, applies='host.move.loops > 0 and host.move.carried > 0', on=[
        ('every carried array travels by reference', 'host.move.by_value == 0'),
        ('no host copy per trip beyond the facts', 'host.move.copy_b == host.move.intent_host_copy'),
        ("per-trip uploads are the facts' (a host producer, a device reader)",
         'host.move.h2d_b == host.move.intent_h2d'),
        ("per-trip copy-outs are the facts' (a device producer, a host observer)",
         'host.move.d2h_b == host.move.intent_d2h')],
        off=[('the by-value products copy the state every trip', 'host.move.copy_b > 0')]),
    # S65 rule R for folds: a row fold over any device output is a device site (no whole copy-out to fold it).
    'rowfold': dict(host=True, applies='dev.rowfold > 0', on=[
        ('each placed row fold is launched', 'host.calls.launch >= dev.rowfold')]),
    # `flow` (func/flow.rs): a unit kernel holds placed folds and cells whose footprints stay in
    # one row of its unit; `; mapal.flow` says what it keeps (no store, no pointer), stores and reads.
    'flow': dict(applies="plan.kind == 'flow'", on=[
        ('kept members are not kernel parameters: the pointers are the stored members and the operands',
         'ptx.params == flow.n_store + flow.n_gmem and flow.ll_params == ptx.params'),
        ('a stored member is stored, a kept one is not: st.global iff something is stored',
         '(ptx.st_global > 0) == (flow.n_store > 0)'),
        ('no shared memory beyond the plan (the members run in registers)', 'ptxas.static_smem == flow.smem'),
        ('no spill', 'ptxas.spill_st == 0 and ptxas.spill_ld == 0'),
        ('every member body inlined: no call left in PTX', 'ptx.call == 0')]),
    'flow-host': dict(host=True, applies='host.flow.units > 0', on=[
        ('one launch per unit (its members launch nothing of their own)', 'host.flow.unit_calls == host.flow.units'),
        ('every unit kernel in the device module', 'dev.flow == host.flow.units'),
        ("per-trip copy-outs are the facts' (a kept member never crosses)", 'host.move.d2h_b == host.move.intent_d2h')],
        applies_off='dev.cell + dev.rowfold > 0', off=[('no unit kernel', 'dev.flow == 0')]),
    # generic-fusion `window` (design §4/§7): a site whose reads are affine in the fold's derived axes (k÷div, k%div:
    # a 2-D convolution's taps) is a device site; the naive kernel walks k in order with constant divisors. Only a
    # window site's plan line carries `taps=` (the kernel exists only with the rung on: `.off-window` keeps it host).
    'window': dict(applies="plan.kind == 'naive' and plan.taps != ''", on=[
        ('no integer divide or remainder left: k÷div, k%div fold to induction variables', 'ptx.div_rem == 0'),
        ('LLVM fully unrolled the constant-trip tap walk (loop-unroll FullyUnrolled)', 'remarks.opt.full_unroll >= 1'),
        ('straight-line taps in SASS: one FFMA (or FMUL) per tap', 'sass.ffma + sass.fmul >= plan.k')]),
    # generic-fusion natural forms `range-map`: a map whose cells off a row / range / column / cell are a capture's
    # own runs in place over that set (`in_place_source`); its movement is the by-ref rows' (`host.move`).
    'range-map': dict(host=True, note='work per trip, not bytes: diff the `.off-range-map` arm (host insns, loops)'),
    'cpu-tiling': dict(host=True, applies='host.cpu == 1 and host.obj.insns > 0', on=[
        ('the CPU build vectorizes on ymm', 'host.obj.ymm_ops > 0'),
        ('the CPU build has the parallel entry', 'host.par > 0')]),
}
