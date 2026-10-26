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
    # cpu-gpu-boundary (runtime): kernel loading, allocation and upload waits leave the timed path.
    'preload': dict(host=True, applies='host.calls.init > 0', on=[
        ('every kernel resolved at module load (init flag 8)', 'host.init_flags & 8 > 0')],
        off=[('each kernel resolved at its first launch', 'host.init_flags & 8 == 0')]),
    'entry-arena': dict(host=True, applies='host.calls.arena > 0', applies_off='host.calls.alloc + host.calls.arena > 0', on=[
        ('the arena made in the entry block, nowhere else', 'host.arena_entry == host.calls.arena > 0')],
        off=[('every member asks the arena (no base loaded from the entry arena)', 'host.arena_load == 0')]),
    'async-h2d': dict(host=True, applies='host.calls.h2d + host.calls.h2d_async > 0', on=[
        ('no upload waits for the device', 'host.calls.h2d == 0')],
        off=[('every upload synchronous', 'host.calls.h2d_async == 0')]),
    'stage': dict(host=True, applies='host.calls.h2d_async > 0', on=[
        ('the page-locked staging ring (init flag 16)', 'host.init_flags & 16 > 0')],
        applies_off='host.calls.init > 0', off=[('no staging ring', 'host.init_flags & 16 == 0')],
        note='staged uploads: `staged=` in mapal_gpu_trace_total'),
    # ON fires only where streams are open and one chain reads an upload (tests/boundary.rs
    # chain_h2d_on_unseen_programs checks each `h2d_on(k)` has a launch on stream k).
    'chain-h2d': dict(host=True, applies_off='host.calls.launchv > 0', off=[
        ('every upload on the legacy stream', 'host.calls.h2d_on == 0')],
        note='ON: `h2d_on` calls where one chain reads an upload: diff the arm'),
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
        ('shared memory is the launch\'s (tree partials, on-chip rows): none static', 'ptxas.static_smem == 0'),
        ('a row of more than one cell or fold step is a block of its lanes', 'flow.lanes == 1 or plan.block == flow.lanes'),
        ('a tree fold or an on-chip row synchronizes the block: BAR iff shared memory', '(sass.bar > 0) == (flow.smem > 0)'),
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
    # recurrence-precision `radix-tile`: a map/fold whose reads are affine per digit of the cell and tap indices
    # (constant div/mod splits: a convolution's implicit GEMM, `mapal_ir::RadixTile`) is the staged FMA tile; its plan
    # line names the digit-affine addresses (`radix=b,out`). `.off-radix-tile`: the nest cell (no smem, one thread per
    # output walking the taps).
    'radix-tile': dict(applies="plan.kind == 'tile' and plan.radix != ''", on=[
        ('operands staged through shared memory: ld.shared in the hot loop', 'ptx.hot.ld_shared + ptx.hot.ld_shared_v4 > 0'),
        ('every digit a constant divisor: no div/rem instruction left', 'ptx.div_rem == 0'),
        ('a register tile per thread: >= tm*tn*bk FP ops per panel trip', 'ptx.hot.fma + ptx.hot.mul >= plan.tm * plan.tn * plan.bk'),
        ('no spill', 'ptxas.spill_st == 0 and ptxas.spill_ld == 0')]),
    # recurrence-precision `tf32-tile` (ADR-0041 D6): a site whose products admit TF32 (`; mapal.plan … set=tf32|f32`,
    # `precision_plan.set`) takes the tensor-core kernel where its shape fits. `.off-tf32-tile`: the f32 FFMA tile.
    'tf32-tile': dict(applies_on="plan.set == 'tf32|f32' and plan.kind in ('mma', 'wgmma')", on=[
        ('tensor-core instructions in PTX: mma.sync or wgmma', 'ptx.mma + ptx.wgmma > 0'),
        ('HMMA / HGMMA in SASS', 'sass.hmma + sass.hgmma > 0')],
        applies_off="plan.set == 'tf32|f32'", off=[
        ('the f32 FFMA tile: no mma.sync / wgmma', "ptx.mma == 0 and ptx.wgmma == 0 and plan.kind == 'tile'")]),
    # generic-fusion natural forms `range-map`: a map whose cells off a row / range / column / cell are a capture's
    # own runs in place over that set (`in_place_source`); its movement is the by-ref rows' (`host.move`).
    'range-map': dict(host=True, note='work per trip, not bytes: diff the `.off-range-map` arm (host insns, loops)'),
    # cpu-gpu-boundary 1b `uniform`: a per-cell map with number captures (a loop counter, a fold's result) is a
    # cell, each uniform a by-value i64 kernel argument; a closed form of (cell, uniforms) is built where its
    # readers run. `.off-uniform` builds it on the host and uploads it per trip (decode q, rt_mlp2 a/f).
    'uniform': dict(host=True, applies='dev.cell_u > 0', on=[
        ('every uniform cell is launched, its uniforms by-value `i64` parameters', 'host.calls.launchv >= dev.cell_u'),
        ("per-trip uploads are the facts' (a uniform cell's output never crosses)",
         'host.move.h2d_b == host.move.intent_h2d')],
        applies_off='dev.cell > 0', off=[('no cell kernel takes a uniform', 'dev.cell_u == 0')]),
    # cpu-gpu-boundary 1a `nest-cell`: a map whose body folds over captured arrays (a guarded dot product: causal /
    # prefill `s`, `o`) is a one-thread-per-cell kernel running the host's body text once its operands are on the
    # device; the K/V rows a host score loop read no longer cross. `.off-nest-cell`: the map on the host.
    'nest-cell': dict(host=True, applies='dev.cell_nest > 0', on=[
        ('each nest cell walks its fold in the kernel (a PTX loop, or >= work-1 FP ops unrolled)',
         'dev.cell_nest_walk == dev.cell_nest'),
        ("per-trip copy-outs are the facts' (a device producer, a host observer)",
         'host.move.d2h_b == host.move.intent_d2h')],
        applies_off='dev.cell > 0', off=[('no nest cell', 'dev.cell_nest == 0')]),
    # item 5 `tree-gate` (under `flow`): a lone fold over a column-major operand keeps its rowfold kernel.
    'tree-gate': dict(host=True, note='unit count: diff the `.off-tree-gate` arm (rtx-4090 contract rowsum: 0 vs 1 unit)'),
    # 1a `retained-fold`: standalone row folds stay device sites in a function with retained sites (prefill pmx/pden);
    # `.off-retained-fold` restores S65's veto (fewer `dev.rowfold`). Decided by the H100 retain-resident decode ABBA.
    'retained-fold': dict(host=True, applies='dev.rowfold > 0', on=[
        ('each placed row fold is launched', 'host.calls.launch >= dev.rowfold')],
        note='rowfold count: diff the `.off-retained-fold` arm'),
    # recurrence-precision 1b `range-cell`: a piecewise map written in place over a device-carried state (a row, a
    # range, a column) is a cell over its interval in the carry's buffer: the written row never crosses per trip.
    'range-cell': dict(host=True, applies='dev.cell_range > 0', on=[
        ('each range cell is launched (its interval start and count by value)', 'host.calls.launchv >= dev.cell_range'),
        ("per-trip uploads are the facts' (no written row crosses)", 'host.move.h2d_b == host.move.intent_h2d')],
        applies_off='dev.cell > 0', off=[('no range cell', 'dev.cell_range == 0')]),
    # 1b `scalar-fold`: a whole fold of a device array is a one-row row fold into a one-element device buffer; a
    # cell reading it loads it there (a pointer, not a by-value uniform): no host observes inside the trip.
    'scalar-fold': dict(host=True, applies='dev.cell_dev > 0', on=[
        ('each placed row fold (whole folds included) is launched', 'host.calls.launch >= dev.rowfold'),
        ("per-trip copy-outs are the facts' (a device number never crosses)", 'host.move.d2h_b == host.move.intent_d2h')],
        applies_off='dev.cell > 0', off=[('no cell reads a device number', 'dev.cell_dev == 0')]),
    # 1a `trip-batch`: a loop whose first trip interval skips (`trip_split`) enters at that interval's last trip,
    # its batched row writes made first by one launch of the range cell's batch variant (each cell its own trip).
    'trip-batch': dict(host=True, applies='dev.cell_batch > 0', on=[
        ('each batch variant is launched (once per loop entry)', 'host.calls.launchv >= dev.cell_batch'),
        ("the loop enters at the interval's last trip (`; mapal.batch enter=`)", 'host.batch > 0')],
        applies_off='dev.cell_range > 0', off=[
        ('no batch variant, every trip runs', 'dev.cell_batch == 0 and host.batch == 0')],
        note='trips run: diff the `.off-trip-batch` arm (loop entry counter, launches per run)'),
    # item 2 `cell-lanes`: on the default face a nest cell whose cells leave resident threads idle folds over a warp's
    # lanes (lane l takes every 32nd step) and combines by a butterfly of shuffles, lane 0 storing.
    'cell-lanes': dict(host=True, applies='dev.cell_lanes > 0', on=[
        ('PTX: each lane cell combines by >= 5 butterfly shuffles (log2 32)', 'dev.cell_lanes_shfl == dev.cell_lanes')],
        applies_off='dev.cell_nest > 0', off=[('every nest cell one thread a cell', 'dev.cell_lanes == 0')]),
    # item 2 `fold-lanes`: on the default face a device row fold (a whole fold: one row) is a warp a row, lane l taking
    # every 32nd k, combined by a butterfly of shuffles; off: one thread a row walking k in order.
    'fold-lanes': dict(host=True, applies='dev.rowfold_lanes > 0', on=[
        ('PTX: each lane row fold combines by >= 5 butterfly shuffles (log2 32)',
         'dev.rowfold_lanes_shfl == dev.rowfold_lanes')],
        applies_off='dev.rowfold > 0', off=[('every row fold one thread a row', 'dev.rowfold_lanes == 0')]),
    # site-geometry `fill-wave` (C9, C10, C6c): a tile site whose grid holds fewer warps than the fill rule's target
    # (`Geom::resident` = 2 a scheduler) cuts its free fold into the fewest partitions that reach it, inside the blocks
    # the part holds at the tile's own registers and threads; the split kernel's panels are sized for the split grid.
    # `.off-fill-wave`: the largest split inside the unsplit grid's wave (grid-share's residency: GEMV, attn P·V).
    'fill-wave': dict(applies="plan.kind == 'tile' and plan.combine in ('atomic', 'splitk')", on=[
        ('the split grid reaches 2 warps a scheduler on every SM, or holds the whole wave',
         'plan.gx * plan.gy * plan.S * ((plan.block + 31) // 32) >= 8 * occ.sms'
         ' or 2 * plan.gx * plan.gy * plan.S > occ.blocks_per_sm * occ.sms'),
        ('one wave: the split grid is resident at ptxas registers + smem',
         'plan.gx * plan.gy * plan.S <= occ.blocks_per_sm * occ.sms'),
        ('no spill', 'ptxas.spill_st == 0 and ptxas.spill_ld == 0')]),
    # site-geometry `row-split` (D1): a retained site's one-row pass takes the fold chooser — the row zeroed and one
    # launch whose partials add into it (a part without the atomic split: the workspace split and its reduce) —
    # instead of one block over the row (decode: grid 1x1x256 per trip).
    'row-split': dict(host=True, applies='dev.naive + dev.tile > 0', on=[
        ('a split launch zeroes its output first (atomic) or reduces a workspace (splitk)',
         'host.calls.zero >= 1 or host.calls.alloc >= 1')],
        note='row launch geometry: diff the `.off-row-split` arm (S CTAs per trip vs 1)'),
    # site-geometry `fit-first` (R1): an FMA tile whose thread grid outgrows its rows reshapes (`ty` halved, `tx`
    # doubled, threads kept) when that alone gives every SM a block, before the fill step halves its threads (convmc on
    # the 4090: 784 blocks of 2 warps -> 196 of 8). `.off-fit-first`: the shrink first.
    'fit-first': dict(applies=TILE + " and plan.ty * plan.tm <= plan.rows and plan.block >= 128", on=[
        ('the thread grid fits the rows and every SM has a block',
         'plan.ty * plan.tm <= plan.rows and plan.gx * plan.gy >= occ.sms'),
        ('no spill', 'ptxas.spill_st == 0 and ptxas.spill_ld == 0')]),
    # site-geometry B4 / ADR-0034 D1: a `--geom=` candidate (knobs.json: tm, tn, warps, bk, stages; wgmma wg_n,
    # wg_groups) replaces the derived tile where the site takes it; every candidate is checked as its plan line says.
    'geom': dict(applies="plan.kind in ('tile', 'wgmma')", on=[
        ('stages >= 2 stream with cp.async, register staging has none',
         "plan.kind != 'tile' or (plan.stages >= 2) == (ptx.cp_async > 0)"),
        ('the derived blocks/SM are resident at ptxas registers + smem', 'occ.blocks_per_sm >= plan.blocks_per_sm'),
        ('no spill', 'ptxas.spill_st == 0 and ptxas.spill_ld == 0')],
        note='candidate vs table row: diff the arm (`--geom=<knob>=<value>`, emission hash)'),
    'cpu-tiling': dict(host=True, applies='host.cpu == 1 and host.obj.insns > 0', on=[
        ('the CPU build vectorizes on ymm', 'host.obj.ymm_ops > 0'),
        ('the CPU build has the parallel entry', 'host.par > 0')]),
}
