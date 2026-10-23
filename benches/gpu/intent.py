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
        applies_off=TILE, off=[('register staging: no cp.async', 'ptx.cp_async == 0 and sass.ldgsts == 0')]),
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
    'warp-fill': dict(applies_on=TILE, on=[
        ('every warp scheduler holds two warps', 'occ.resident_warps_per_sm >= 8')]),
    'fill-rows': dict(applies_on=TILE, on=[
        ('every warp scheduler holds two warps', 'occ.resident_warps_per_sm >= 8')]),
    'idx32': dict(applies_on=SCALAR + " and plan.idx == 'i32'", on=[
        ('hot loop: at most one 64-bit integer op per global access (the address)',
         'ptx.hot.i64_arith <= ptx.hot.ld_global + ptx.hot.ld_global_v2 + ptx.hot.ld_global_v4 + ptx.hot.cp_async + ptx.hot.st_global')]),
    'affine': dict(applies_on=TILE, on=[
        ('one base register per operand in the hot loop', 'ptx.hot.distinct_load_bases <= 2')],
        applies_off=TILE, off=[('several base registers', 'ptx.hot.distinct_load_bases > 2')]),
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
        ('residency sized to the grid share', 'plan.blocks_per_sm <= max(1, ceil(plan.gx * plan.gy / occ.sms))')]),
    'panel-regs': dict(applies_on=TILE + ' and plan.stages == 1 and plan.held == 0', on=[
        ('the prefetch panel fits the register file: no spill', 'ptxas.spill_st == 0')]),
    'cell-fill': dict(applies_on=TILE + ' and plan.tm * plan.tn == 1', on=[
        ('one output per thread, staged through shared memory', 'sass.hot.lds > 0')]),
    # ---- fold realizations (ADR-0040 D4) --------------------------------------------------------------
    'splitk': dict(applies_on="plan.combine == 'splitk'", on=[
        ('S partial folds, then one reduce launch', "launch.S == plan.S and launch.combine == 'splitk'"),
        ('no atomics', 'ptx.atom_add + ptx.red_add == 0')]),
    'splitk-atomic': dict(applies_on="plan.combine == 'atomic'", on=[
        ('one red.global.add per accumulator', 'ptx.atom_add + ptx.red_add >= plan.tm * plan.tn'),
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
    'arena': dict(host=True, applies='host.calls.alloc + host.calls.arena > 0', on=[
        ('one arena per function, no per-object cuMemAlloc', 'host.calls.alloc == 0')],
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
    'hostopt': dict(host=True, applies='host.obj.insns > 0', on=[
        ('the host object uses 256-bit AVX2 vectors (the x86-64-v3 profile)', 'host.obj.ymm_ops > 0')],
        off=[('generic x86-64: no 256-bit vectors', 'host.obj.ymm_ops == 0')]),
    'place': dict(host=True, note='site placement: diff the `.off-place` arm (device_sites, launches)'),
    'cpu-tiling': dict(host=True, applies='host.cpu == 1 and host.obj.insns > 0', on=[
        ('the CPU build vectorizes on ymm', 'host.obj.ymm_ops > 0'),
        ('the CPU build has the parallel entry', 'host.par > 0')]),
}
