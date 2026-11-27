//! [`GpuRungs`](super::GpuRungs)'s remaining fields, in declaration order:
//! the onto-device rounds (`cond_bounds`) through model-ready placement
//! (`machine_place`).

/// Appends this part's fields to the field list it is handed and passes the
/// list to the next macro in `[..]` (the chain `super` starts).
macro_rules! device_rounds {
    ([$next:ident $(, $rest:ident)*] $($fields:tt)*) => {
        $next! {
            [$($rest),*]
            $($fields)*
            /// onto-device `cond-bounds`: `bounds_proof` ranges an index by the
            /// conditions that hold where it runs (a guard arm's, a loop's continue
            /// and exit conditions, a `Phi`'s per arm) and ranges loop-carried
            /// counters from `loop_plan` (init, affine step, continue condition), so
            /// a guarded read (RoPE's rotate_half partner) or a counter-indexed read
            /// (a decode step's position) carries no trap and its site is device
            /// capable. Off (`--gpu-off=cond-bounds`): the graph's facts are rebuilt
            /// under plain interval ranges (`mapal_ir::FactArms`), the proof before it.
            pub cond_bounds: bool,
            /// onto-device `carry-made`: a loop-carried value is device-made when the
            /// next state that replaces it is (and a loop's exit value when its
            /// payload is) — `place`'s "an operand is device-produced" rule sees
            /// through the back edge, so a step's small maps over a device-carried
            /// state (decode's RMSNorm) stay beside it instead of pulling the state
            /// to the host every trip. Off (`--gpu-off=carry-made`): the rule stops at
            /// a loop view, the placement before it.
            pub carry_made: bool,
            /// onto-device `host-par`: a GPU build's host map with the CPU build's
            /// split shape (clean body, static extent, run once, no device-only
            /// operand) runs as one split task on the pool where the host's measured
            /// dispatch cost says it pays (`func/hostpar.rs`), the CPU build's rule.
            /// Off (`--gpu-off=host-par`): the host walk runs it on one core.
            pub host_par: bool,
            /// device-geometry `tile-post`: a map whose body is a sum of products
            /// followed by elementwise terms of the same cell (a residual, a scale, a
            /// mask: [`mapal_ir::TilePost`]) is a tile site whose store evaluates the
            /// body after its fold (the epilogue, `func/resident.rs:epilogues`) —
            /// where its `b` panel serves more than one row or is read along its
            /// lanes: a lone row over a k-major `b` (a decode step's masked scores)
            /// stays the warp-a-cell kernel (`func/gpu.rs:post_sites`).
            /// Off (`--gpu-off=tile-post`): the map is a per-cell kernel (one thread
            /// a cell, its fold serial).
            pub tile_post: bool,
            /// device-geometry `row-tree` (a fact arm, `mapal_ir::FactArms`): a
            /// per-cell map whose body folds one row of an array beside folds that
            /// load nothing (RMSNorm's sum of squares, then its Heron root) splits
            /// that fold over a warp's lanes (`cell-lanes`). Off
            /// (`--gpu-off=row-tree`): only a body with one nested fold does; such a
            /// row is one thread's serial fold.
            pub row_tree: bool,
            /// device-geometry `piece-delta` (a fact arm, `mapal_ir::FactArms`): a
            /// map writing one row of each sequence of a `[B][L][…]` state,
            /// `((t ÷ d) % s == r) ? new : x[t]`, is piecewise ([`mapal_ir::Outside`]
            /// `w > 1`, [`mapal_ir::Delta::Pieces`]): on the device a range cell
            /// over its B pieces, in the state's buffer (`range-cell`). Off
            /// (`--gpu-off=piece-delta`): the map rewrites the whole state each trip.
            pub piece_delta: bool,
            /// device-geometry `head-group` (a fact arm, `mapal_ir::FactArms`): a
            /// mixed-radix tile site may have batch digits — digits of the map index
            /// both operands read (the query heads of one KV head, a sequence) —
            /// as the most significant row digits; a block's rows lie in one batch,
            /// so its `b` panel (one KV head's K or V rows) is loaded once for the
            /// group's query heads. Off (`--gpu-off=head-group`): such a map is a
            /// per-cell kernel (each query head re-reads its KV head's rows).
            pub head_group: bool,
            /// exp (ADR-0042 D5): on the default face `exp` takes the fastest form
            /// the bound admits — the host's contracted polynomial (vectorizes), the
            /// device's `ex2.approx` after a Cody–Waite reduction where the part's
            /// measured `ex2` error meets the bound (`profile::Caps::ex2`). Off
            /// (`--gpu-off=exp-hw`, `--no-exp-hw`): the oracle's steps on every face.
            pub exp_hw: bool,
            /// math builtins (ADR-0046 D10): on the default face every math builtin
            /// takes the cheapest realization its contract admits from the side's
            /// declared units and definitions (`sqrt.approx`, the `ex2` form, a
            /// target's routine). Off (`--gpu-off=math-hw`, `--no-math-hw`): the exact
            /// face's choice on every face — the reference, or an `Exact` unit (IEEE
            /// `sqrt`).
            pub math_hw: bool,
            /// exp (ADR-0042 D7, a fact arm, `mapal_ir::FactArms`): `fold_combine`'s
            /// rescale law — a value read only through `exp` of a difference lets a
            /// block's partial take its own shift (online softmax in `row-slices`).
            /// Off (`--gpu-off=exp-rescale`, `--no-exp-rescale`): such a value is a
            /// phase.
            pub exp_rescale: bool,
            /// half-precision (ADR-0043): a sum of products whose operand is the exact
            /// widening of a `bf16` / `f16` array element (`tile_plan_narrow`) is a
            /// device tile site whose kernel loads the 2-byte format and widens it.
            /// Off (`--gpu-off=narrow-tile`): such a map is a per-cell kernel (or the
            /// host's loop).
            pub narrow_tile: bool,
            /// large-gemm `streamk-fill`: stream-K walks the site's tiles over the
            /// slots its own launched grid fills — a grid whose blocks at their own
            /// geometry give the part fewer warps than [`profile::Gpu::fill_warps`]
            /// (a skinny site: 16 one-warp tiles) is no stream-K site; it takes the
            /// `fill-wave` split, which reaches that target. Off
            /// (`--gpu-off=streamk-fill`): stream-K wherever the fill class of the
            /// site's base grid allows it, at one block an SM.
            pub streamk_fill: bool,
            /// large-gemm `k-fill`: an under-filled site (no warp per scheduler at
            /// the base grid) whose free-order fold the `fill-wave` split can cut
            /// into enough partitions keeps the full register tile and its threads
            /// and fills the part along `k`, instead of shrinking the tile
            /// (`warp-fill`) or its threads — a tile below `8 × 8` issues a shared
            /// load per fewer than 16 FFMA. Off (`--gpu-off=k-fill`): the shrunk
            /// tile.
            pub k_fill: bool,
            /// long-prefill `uniform-base` (a fact arm, `mapal_ir::FactArms`): a
            /// mixed-radix tile read may take terms of a number its map captures (a
            /// loop counter slicing a head group or a weight panel); the launch
            /// moves the operand's pointer by them. Off (`--gpu-off=uniform-base`):
            /// such a map is a per-cell kernel.
            pub uniform_base: bool,
            /// long-prefill `carry-take`: a loop carry whose init's device copy is
            /// valid and dies at the loop's entry (`LastUsePlan`) takes that buffer
            /// as its first state, host-built or not — the S65 move of a
            /// device-built init, widened. Off (`--gpu-off=carry-take`): a host-built
            /// init is uploaded again into the carry's own buffer.
            pub carry_take: bool,
            /// long-prefill `slice-launches`: a sliced unit's phases may meet at
            /// kernel boundaries — one launch a phase — where the part's measured
            /// launches (`launch_at` × phases) undercut one cooperative launch and
            /// its grid barriers (outside a loop with the process's first
            /// cooperative launch, `coop_first_ns`). Off
            /// (`--gpu-off=slice-launches`): the cooperative launch only.
            pub slice_launches: bool,
            /// caching-memory `unchanged` (a fact arm, `mapal_ir::FactArms`): the
            /// trip deltas of `unchanged_plan` reach past `write_delta`'s carried
            /// arrays — through every map whose cells read rows of a changed array
            /// (a row fold, a per-row scale, a projection, a rotation) — so the
            /// retained sites and cells below see them. Off
            /// (`--gpu-off=unchanged`): `write_delta`'s deltas only.
            pub unchanged: bool,
            /// caching-memory `retain-cell`: a device cell map inside a loop whose
            /// output differs from its previous trip's only in a row delta
            /// (`unchanged_plan`) launches over that delta on every trip after the
            /// loop's first, in its own persisting buffer. Off
            /// (`--gpu-off=retain-cell`): every trip launches the whole map.
            pub retain_cell: bool,
            /// caching-memory `retain-pieces`: a retained tile site whose operand
            /// changes in one row of every sequence (`Delta::Pieces`) computes those
            /// rows as one site over the gathered rows into a compact buffer, then
            /// places them (`mapal_gpu_copy2d`). Off (`--gpu-off=retain-pieces`):
            /// such a site runs whole every trip.
            pub retain_pieces: bool,
            /// caching-memory `retain-same`: a device cell map inside a loop whose
            /// output is the same on every trip (`unchanged_plan`'s `same`: every
            /// input unchanged — a transform of a carried array the loop never
            /// writes) launches on the loop's first trip only, in its own persisting
            /// buffer. Off (`--gpu-off=retain-same`): every trip launches it.
            pub retain_same: bool,
            /// caching-memory `cone-span`, under `entry-arena`: a buffer freed only at
            /// exit lives in the arena from its first writer (a loop's body: its
            /// outermost loop's entry, the `LoopEnter`'s position; an epilogue's site,
            /// a unit's tile site, a kernel its zero pass rides) to exit — placed
            /// first-fit among the straight-line members' intervals (a member a
            /// carry took: to exit), so it takes the bytes of what died before (a
            /// prebuilt cache's inputs). Off (`--gpu-off=cone-span`): every such
            /// buffer lives over the whole function, after the members.
            pub cone_span: bool,
            /// caching-memory `arena-fit`: the arena's members are also placed
            /// largest first (then in birth order), first-fit, and that arena kept
            /// where it is smaller and `zero-fold` folds as many zero passes under
            /// it — largest first packs tighter where small early buffers would push
            /// a large late one above them, birth order where it does not (and a
            /// tighter packing may put a member between a site's zero and the
            /// site). Off (`--gpu-off=arena-fit`): birth order only.
            pub arena_fit: bool,
            /// tensor-cores (ADR-0043 D4 / D6): a tile site whose product admits a
            /// 16-bit member (`: bf16 | f32`, `: f16 | f32`) takes the part's
            /// matrix instruction for that format (`mma.sync.m16n8k16`, `Caps::matrix`):
            /// 16-byte `cp.async` panels of a narrow operand, an f32 operand rounded
            /// once as its panel is staged, `ldmatrix` fragments, f32 accumulators.
            /// Off (`--gpu-off=half-tile`): the f32 / narrow FMA tile it also admits.
            pub half_tile: bool,
            /// tensor-cores: a narrow (`bf16` / `f16`) operand's FMA-tile panel is
            /// copied in its own format by 16-byte `cp.async` and widened as each
            /// fragment is read from shared memory, on the `cp.async` path the rules
            /// pick (`async`). Off (`--gpu-off=narrow-async`): the register-staged
            /// tile that widens each panel as it copies it (slice 1).
            pub narrow_async: bool,
            /// tensor-cores `staged-regs`: a register-staged FMA tile whose panel
            /// (every copy's value live from load to store) and tile exceed the
            /// thread's registers is refused — the spilling geometry (the block
            /// decode's one-row narrow sites, 516 registers at `bk` 128). Off
            /// (`--gpu-off=staged-regs`): the rules' tile however large its panel.
            pub staged_regs: bool,
            /// tensor-cores `half-fill`: the 16-bit tile is taken where the site's
            /// flops per operand byte reach the part's FFMA balance or its own grid
            /// fills the warp schedulers; elsewhere the FMA tile, which splits `k`
            /// to fill the part. Off (`--gpu-off=half-fill`): the 16-bit tile
            /// wherever it fits.
            pub half_fill: bool,
            /// tensor-cores `narrow-radix` (a fact arm, `mapal_ir::FactArms`): a
            /// mixed-radix tile site may read the exact widening of a narrow element
            /// (attention over a `bf16` KV cache). Off (`--gpu-off=narrow-radix`):
            /// such a map is a per-cell kernel.
            pub narrow_radix: bool,
            /// gemv-stream `vec-cells`: the one-thread-per-cell kernel of a streaming
            /// site (flops per operand byte below the part's FFMA balance) gives a
            /// thread the adjacent cells one 16-byte load of `b` covers — `16 / e`
            /// lanes, where the layout facts prove the load (`b` unit-stride along
            /// the lane, its base and `k` stride whole 16-byte granules, the lanes a
            /// whole number of vectors) and `a` is the same for every lane: one
            /// `ld.global.nc.v4` and one `a` load per `16 / e` FMAs, the `k` steps'
            /// loads of one trip issued before their FMAs. Off
            /// (`--gpu-off=vec-cells`): one cell a thread, scalar loads.
            pub vec_cells: bool,
            /// gemv-stream `stream-fly`: a streaming one-thread-per-cell site's `k`
            /// split is the fewest partitions whose grid puts a block on every SM
            /// with the part's measured in-flight knee in flight on each
            /// ([`profile::Gpu::stream_fly_per_sm`],
            /// [`profile::TargetProfile::split_stream`]). Off
            /// (`--gpu-off=stream-fly`): the warp fill (`fill-wave`) alone.
            pub stream_fly: bool,
            /// regressions `vec-spread`: the vector cells (`vec-cells`) are taken
            /// where the part's DRAM share an SM asks for the grid's spread only
            /// when their grid holds the SMs as evenly as one cell a thread at the
            /// same split ([`rules::vec_cells_pay`]). Off (`--gpu-off=vec-spread`):
            /// the vector cells at any spread.
            pub vec_spread: bool,
            /// gemv-stream `bulk-stage` (TMA): the warp `mma` tile's panels are
            /// staged by the part's bulk-copy engine ([`profile::Caps::bulk`]) where
            /// its geometry covers every panel row ([`profile::BulkCopy::box_fits`]):
            /// one warp issues a row copy each (`cp.async.bulk`), a stage's
            /// transaction barrier (`mbarrier.expect_tx`) counts its bytes, and every
            /// thread waits on it (`mbarrier.try_wait.parity`) instead of on its own
            /// `cp.async` groups. Off (`--gpu-off=bulk-stage`): every thread's 16-byte
            /// `cp.async` copies.
            pub bulk_stage: bool,
            /// hopper-wgmma-tma R2 (`wgmma-ws`): the Hopper TF32 site as a
            /// warp-specialized `wgmma` kernel — one producer warpgroup copying each
            /// stage as whole tensor boxes (`cp.async.bulk.tensor`, a host-encoded
            /// map per operand) onto a full barrier and rounding the `aᵀ` box once,
            /// consumer warpgroups releasing each stage on an empty barrier once the
            /// group reading it completed (`wgmma.wait_group`)
            /// ([`profile::TargetProfile::gpu_wgmma_ws_tile`]). Off
            /// (`--gpu-off=wgmma-ws`): the S53 kernel.
            pub wgmma_ws: bool,
            /// hopper-wgmma-tma R3 (`ws-regs`): the warp-specialized kernel moves
            /// registers from its producer to its consumers (`setmaxnreg`) where the
            /// part reallocates ([`profile::Caps::reg_realloc`]). Off: every thread
            /// keeps the launch's even share (a smaller consumer tile).
            pub ws_regs: bool,
            /// hopper-wgmma-tma R1 (`tma-digits`): the `b` panel of a stage as ONE
            /// tensor box — its lane axis split into digits `lane = 32·q + r` (a
            /// 3-d map: the 32-word swizzle row, `k`, then `q` at 128 bytes), which
            /// lands the `bn/32` column slabs one after another exactly as `bn/32`
            /// 2-d boxes do: 2 copies a stage instead of `1 + bn/32` (the engine's
            /// measured cost is per copy, `tma_copy_ns`). Off: a 2-d box per slab.
            pub tma_digits: bool,
            /// wgmma-16bit (`wgmma16`): **a 16-bit tile on the part's warpgroup
            /// product** — where the part lists `wgmma.m64nNk16` for the site's
            /// format ([`profile::Caps::matrix_group`]) at a measured rate above its
            /// warp `mma` ([`crate::rules::wgmma16_pays`]), the bf16 / f16 site takes
            /// the warp-specialized tile family of `wgmma-ws` with the format a
            /// parameter: `c = a·b` as stored, `a` from a K-major descriptor (stored
            /// in the format) or registers (f32, rounded in its fragment load), `b`
            /// from a descriptor in either major, TMA boxes on an mbarrier ring
            /// ([`profile::TargetProfile::gpu_wgmma_ws_tile_of`]). Off
            /// (`--gpu-off=wgmma16`): the warp `mma.sync.m16n8k16` tile.
            pub wgmma16: bool,
            /// flash-attention `stream`: a row-block group `stream_plan` proves one
            /// pass over windows of its axis (attention's `q·Kᵀ → softmax → P·V`)
            /// runs as one kernel walking the axis window by window — each fold's
            /// state carried by its law, the axis-wide values never stored, each
            /// operand window read once a block — where the plan model prices it
            /// below the split launches (`func/stream.rs`). Off
            /// (`--gpu-off=stream`): the split realization, the emission before it.
            pub stream: bool,
            /// flash-attention `stream` (a fact arm, `mapal_ir::FactArms::stream`):
            /// `stream_plan` forms its groups. Off (`--gpu-off=stream-fact`,
            /// `--no-stream`): no stream group.
            pub stream_fact: bool,
            /// flash-attention `split-kv` (R3): a stream unit whose row blocks leave
            /// the part's SMs idle splits each row block's windows over `s` blocks;
            /// the row block's last block to finish combines the partial states by
            /// their laws (a ticket: a fence and one atomic, no second launch). Off
            /// (`--gpu-off=split-kv`): one block a row block.
            pub split_kv: bool,
            /// sweep-attention-2 `stream-mma`: a stream unit whose products each
            /// admit a 16-bit member the part has a warp matrix product for (its
            /// `m16n8k16` geometry, the accumulator meeting the member's `u_a`) folds
            /// them on it — the threads in the accumulator's fragment layout (two
            /// rows and a quad's columns a thread), the row operand's fragments held
            /// in registers, the producer's window cells passed to the consumer as
            /// its `a` fragments — where the plan model prices it below the FFMA
            /// stream. Off (`--gpu-off=stream-mma`): the FFMA stream.
            pub stream_mma: bool,
            /// sweep-attention-2 `kmajor-b`: a register-staged tile whose `b` is
            /// unit-stride along `k` and not along its lanes (a decode step's keys,
            /// `K[key][u]`) copies its panel along those rows — a vector of `k` a
            /// copy, a key's row coalesced, scattered into the `[k][n]` panel the
            /// fragments read — instead of one element a copy down a column (each
            /// thread a different key's row). Off (`--gpu-off=kmajor-b`): the
            /// element copies.
            pub kmajor_b: bool,
            /// flash-attention F5 (a fact arm, `mapal_ir::FactArms::guard_box`): a
            /// stream group's masked producer has its mask's form, and a wholly
            /// masked window is skipped where F6 allows. Off (`--gpu-off=guard-box`,
            /// `--no-guard-box`): every window computed.
            pub guard_box: bool,
            /// mask-frontend (a fact arm, `mapal_ir::FactArms::guard_terms`): a
            /// mask's side through a captured array — its element law, else a
            /// per-row load — or a per-launch number. Off (`--gpu-off=guard-terms`):
            /// digit forms of the cell index alone.
            pub guard_terms: bool,
            /// mask-frontend (a fact arm, `mapal_ir::FactArms::walk_order`): a
            /// stream group's blocks walk its rows in the seed's tile order. Off
            /// (`--gpu-off=walk-order`): in the rows' own order.
            pub walk_order: bool,
            /// mask-frontend `stream-numbers`: a streamed unit's epilogue numbers
            /// are launch arguments. Off (`--gpu-off=stream-numbers`): a unit whose
            /// epilogue reads a number after its fold is no stream (as before).
            pub stream_numbers: bool,
            /// bf16-class `finite-check`: a stream group's masked windows are
            /// skipped where a consumer's column operand is finite only at run time
            /// (`mapal_ir::StreamGuard::check`) — the kernel reads the skipped
            /// windows' column elements and runs those windows where one is not
            /// finite. Off (`--gpu-off=finite-check`): such a group skips none.
            pub finite_check: bool,
            /// bf16-class `narrow-window`: a stream group whose column operand is
            /// stored bf16 (a bf16 KV cache) streams on the matrix product, its
            /// windows holding the operand's own bytes (half an f32 window's) and
            /// its fragments taking them as they stand. Off
            /// (`--gpu-off=narrow-window`): such a group is no stream.
            pub narrow_window: bool,
            /// bf16-class `device-trap`: a map whose only possible trap is an index
            /// `bounds_proof` cannot prove (`mapal_ir::CategoryIr::trap_cell_plan`:
            /// an embedding's row through a run-time token id) may run as a device
            /// cell — each such read checked in the kernel (out of bounds: the trap
            /// recorded in the module's word, element 0 read), the trap raised at
            /// the next host checkpoint (a print, the clock, a trap, the exit).
            /// Off (`--gpu-off=device-trap`): such a map runs on the host.
            pub device_trap: bool,
            /// flash-attention F7 (a fact arm, `mapal_ir::FactArms::finite`):
            /// `finite_plan` proves values finite. Off (`--gpu-off=finite`,
            /// `--no-finite`): nothing is, so no masked window is skipped.
            pub finite: bool,
            /// sweep-attention `unit-uniform` (a fact arm,
            /// `mapal_ir::FactArms::unit_uniform`): a tile whose operand address a
            /// captured number moves (a loop counter slicing a KV head group) joins
            /// `unit_plan`'s extended groups, the unit's launch moving that pointer
            /// — attention inside a head-group loop streams. Off
            /// (`--gpu-off=unit-uniform`): such a tile launches alone, as before.
            pub unit_uniform: bool,
            /// f32-tensor-cores R-T1 `mma-combine`: a tensor-core tile site (TF32,
            /// 16-bit, split) cuts its `k` into `S` pieces whose partials add into
            /// the zeroed output (`splitk-atomic`) or a workspace (`splitk`) — the
            /// split the FMA tile takes (ADR-0040 D4), now on the `mma.sync` kernels:
            /// each piece's `cp.async` pipeline over its own `k / S`, the seed on
            /// piece 0. Off (`--gpu-off=mma-combine`): tensor tiles never split.
            pub mma_combine: bool,
            /// f32-tensor-cores R-F `split-tile`: a `{f32}` sum of products on the
            /// default face (free order) takes a split realization on the part's
            /// matrix instruction (`mapal_ir::split`: tf32x3, narrow1) where its
            /// coefficient meets D2 at the site's `K` and the part's matrix rate pays
            /// for the form's passes against its FFMA rate. Off (`--gpu-off=split-tile`):
            /// the FFMA tile (the emission before it); on (`--gpu-on=split-tile`):
            /// the split wherever the bound admits it, past the rate rule.
            pub split_tile: bool,
            /// tensor-cores-2 `split-issue`: a split tile's realization at its
            /// issue rate — each f32 element cut by integer rounding with the value
            /// classes left to the guard (which now also takes an element whose
            /// leading part rounds to infinity), and each tile's zeroed chain
            /// carried over the k-steps the bound admits before its round-to-nearest
            /// add ([`func::gpu::split_flush`]). Off (`--gpu-off=split-issue`): the
            /// 5fcb401b realization (`cvt.rna`, the overflow select, a flush every
            /// k-step) — the A100's tf32x3 loser (1612 SASS instructions a panel of
            /// 96 `HMMA` against 792).
            pub split_issue: bool,
            /// tensor-cores-2 `split-least`: an f32 × bf16 product takes the split
            /// form of fewest part products its bound admits at the site's `K` —
            /// bf16x2n (two bf16 parts of the f32 operand, split error `2⁻¹⁶`)
            /// before bf16x3n (three, exact). Off (`--gpu-off=split-least`):
            /// bf16x3n wherever the split is taken.
            pub split_least: bool,
            /// f32-tensor-cores R-T2 `mma-depth`: a tensor-core tile takes the panel
            /// depth keeping the most operand bytes in flight a block at its
            /// residency (`profile::mma_depth_pick`: `bk` 16 × 5 stages against `bk`
            /// 32 × 2) instead of the deepest panel two stages fit (S52). The rules
            /// ship it off on every part (it lost every measured site); `--gpu-on=
            /// mma-depth` measures it.
            pub mma_depth: bool,
            /// gapfix tensor-tile (C3) `mma-fly`: where a tensor tile's panel loads
            /// outlast the products that cover them (an operand past the L2 keep,
            /// fragments read as stored, the S52 panel's DRAM time within its
            /// products' but not their latency: `profile::TargetProfile::gpu_mma_fly`)
            /// its ring keeps the part's in-flight bytes
            /// (`profile::Gpu::stream_fly_per_sm`) past its wait on every SM — the
            /// residency, panel depth and stages of `profile::mma_fly_pick`;
            /// elsewhere the S52 panel. Off (`--gpu-off=mma-fly`): the S52 panel on
            /// every site.
            pub mma_fly: bool,
            /// sweep-gemm `round-store` (ADR-0041 D5 `round_at` over arrays): an f32
            /// array a map produces whose every element read is a multiplicand of a
            /// product admitting `bf16` ([`mapal_ir::PrecisionPlan::store`]) is
            /// stored rounded once by its producer ([`CategoryIr::narrow_stored`]):
            /// its tile panels copy 2-byte elements (`ldmatrix`, no conversion per
            /// fragment, half the shared memory). Default (order-free) face only.
            /// Off (`--gpu-off=round-store`): each reader rounds its copy.
            pub round_store: bool,
            /// sweep-gemm-2 `warm-zero`: a module whose launches zero an output
            /// (an atomic split's or stream-K's zero op, `mapal_gpu_zero`) has the
            /// runtime run one zero of a scratch buffer on the `eager-init` thread,
            /// before the clock: the driver initializes its memset path on the
            /// process's first fill of 8 KiB or more (CUDA 13, rtx-5080: 13–17 µs
            /// that call against 2–4 µs after), which a one-call program otherwise
            /// pays inside its timed region. Off: the first zero pays it.
            pub warm_zero: bool,
            /// sweep-gemm-2 `streamk-price`: on a part whose rates price a stream-K
            /// fixup (`DeviceCost`: the FMA tile rate, the `red` rate, the zero
            /// pass), a site's stream-K is weighed against the S63 count by those
            /// prices ([`crate::func::gpu`] `streamk_priced`: the pieces' partials and
            /// the zero pass against the panels the walk saves) — under-filled sites
            /// too, whose class ships `streamk` off (site-level wave quantization).
            /// Off (`--gpu-off=streamk-price`): the panel count (`iters + 1 < split`)
            /// in the classes that ship `streamk`.
            pub streamk_price: bool,
            /// sweep-runtime-2 `streamk-epi`: an epilogue group (S54, no `tile-post`
            /// body) whose site the rules walk by stream-K keeps it — every piece
            /// adds its partials into a zeroed workspace and the CTA that completes a
            /// tile (a per-tile ticket: the stream-K walk names how many CTAs share
            /// it) evaluates the group's members on the whole sums. Off
            /// (`--gpu-off=streamk-epi`): the group's ordered kernel, as before (a
            /// measurement flag; the stream-K rule decides where it applies).
            pub streamk_epi: bool,
            /// bf16 tensor-tile `mma-walk`: a 16-bit tensor tile (no split form)
            /// whose grid rounds over the SMs' matrix pipes (`⌈tiles / sms⌉` tiles
            /// on the busiest SM) cost more than the same products spread evenly is
            /// walked by stream-K over every resident slot at its whole warp tile
            /// (`func::gpu::mma_walk`, priced by `func::gpu::MmaWalk`); the shared
            /// tiles' partials add into the zeroed output, or an epilogue group's
            /// workspace and ticket (`streamk-epi`). Off (`--gpu-off=mma-walk`): the
            /// ordered tile or its `k` cut, as before (a measurement flag; the price
            /// decides where it applies). `--gpu-on=mma-walk` walks every site the
            /// walk admits, priced or not (the measurement and emulator arm).
            pub mma_walk: bool,
            /// hopper-launch `tma-warm`: a module carrying a kernel that reads
            /// tensor maps (the `<kernel>_tmap` recipes the runtime finds in its
            /// text at load) has the runtime launch its own one-box TMA kernel once
            /// on the `eager-init` thread, before the clock: the driver readies a
            /// context's tensor-copy path at the first launch of a kernel holding a
            /// `cp.async.bulk.tensor` (CUDA 13: 1.34–1.49 ms that host call on the
            /// RTX 5080, against 3–4 µs after; on the H100 the 120–270 µs a
            /// `wgmma-ws` call spent outside its kernel, cross-arch 5fcb401b), which
            /// a one-call program otherwise pays inside its timed region. Host only
            /// (the module's flag word); off (`--gpu-off=tma-warm`): the first TMA
            /// launch pays it.
            pub tma_warm: bool,
            /// hopper-launch `ws-persist`: the warp-specialized `wgmma` kernel as a
            /// persistent grid — `sms × blocks_per_sm` CTAs each walking the tiles
            /// `cta, cta + ctas, …` in raster order, the stage ring and its barrier
            /// phases running on across tiles, so a CTA's producer copies the next
            /// tile's boxes under this tile's epilogue and the pipeline fills once
            /// a CTA — where [`crate::rules::ws_persist_pays`] prices the fills and
            /// epilogues the walk overlaps above its tail. The rules ship it off (its
            /// H100 measurement is mixed, the separating fact open: `rules::rungs`);
            /// `--gpu-on=ws-persist` measures the walk. Off: a CTA a tile.
            pub ws_persist: bool,
            /// gaps-disasm C1 `load-dispatch`: the module registers its site tables
            /// before the runtime's init (`mapal_gpu_sites`), so each table's pick
            /// (where one is decided without a run: forced, stored), its candidates'
            /// kernels and shared-memory limits and a cooperative table's occupancy
            /// are resolved when the module loads — a one-call program's first site
            /// goes from its clock to its first driver call without hashing its
            /// table or scanning the wisdom (the runtime finds every preloaded kernel
            /// by its id in one hash probe either way). Off
            /// (`--gpu-off=load-dispatch`): no registry; all of it at the first call.
            pub load_dispatch: bool,
            /// gaps-disasm C1 `zero-host`: an atomic output's zero pass that no
            /// earlier kernel folded (`zero-fold`) is the runtime's zero kernel — one
            /// launch on the host — never the driver's memset API (RTX 5080: 3.7–5.9
            /// µs a call inside a one-call program's clock, against 2.5–3.9 µs a
            /// launch). Off (`--gpu-off=zero-host`): `cuMemsetD8Async`.
            pub zero_host: bool,
            /// gaps-disasm C1 `zero-hoist`: an atomic site's zero that no earlier
            /// kernel takes (`zero-fold`) is issued at the earliest point the
            /// zero-fold walk allows — right after the last step that touches the
            /// output's bytes, or the sequence's start — instead of between the
            /// site's dispatch and its launch, where its host issue delays the
            /// kernel (an arena member, in the straight-line walk; a loop trip keeps
            /// it at the site), skipped at run time where the site's decided pick
            /// adds nothing (`mapal_gpu_zero_for`). Off (`--gpu-off=zero-hoist`): at the
            /// site.
            pub zero_hoist: bool,
            /// gaps-disasm C5 `split-price`: an under-filled FMA tile whose operands
            /// stream from DRAM takes the atomic `k` split its price gives — the
            /// stream at the split grid's spread over the SMs against every
            /// partition's partial cells at the part's measured
            /// `DeviceCost::split_cell_ps` (`TargetProfile::split_priced`) — where the
            /// fill rule took the fewest partitions that reach `fill_warps`. A part
            /// without the property keeps the fill rule. Off
            /// (`--gpu-off=split-price`): the fill rule.
            pub split_price: bool,
            /// gapfix `iota-law`: an array a body fn captures whose law
            /// (`elem_plan` through captures: an `iota` passed in) is closed is
            /// computed at its read, never loaded (a norm's fold over a captured
            /// `iota`). Off: the capture is loaded.
            pub iota_law: bool,
            /// gapfix `stream-ld`: a tile operand the grid reads once whose bytes
            /// exceed what the L2 keeps (`Gpu::l2_keep_bytes`) loads with the
            /// streaming policy (`ld.global.cs`: L2 evict-first) where the part has
            /// it (`Caps::l2_evict`). Off: the default policy.
            pub stream_ld: bool,
            /// gapfix `shfl-tree`: a row unit's lane tree combines inside a warp
            /// by register shuffles and across warps through one shared exchange
            /// (one barrier), where its type is 32-bit. Off: a barrier per level.
            pub shfl_tree: bool,
            /// gapfix `kvec-swizzle`: a register-staged k-major `b` panel's words
            /// spread over the shared-memory banks by an XOR of the stored row's
            /// word group (the bank geometry: `Gpu::smem_banks`). Off: the dense
            /// panel.
            pub kvec_swizzle: bool,
            /// model-ready placement `trip-cells`: a cell a loop runs every trip is a
            /// candidate whatever its size, and an operand made once (outside every
            /// loop and guard arm) is in device memory for it (uploaded once at its
            /// producer) — the fixed point keeps it where it moves no bytes. Off
            /// (`--gpu-off=trip-cells`): a cell is a candidate only where it pays
            /// (`place`) or is fed, the placement before it.
            pub trip_cells: bool,
            /// model-ready placement `bytes-place`: a per-trip cell runs where it
            /// moves fewer bytes per trip (uploads of operands the device lacks and
            /// the copy-out of a host-read output, against the copy-out of operands
            /// only the device holds and the upload of a device-read output; a tie
            /// to the device when a device kernel reads the output). Off: every
            /// operand must already be on the device.
            pub bytes_place: bool,
            /// full-model `once-bytes`: `bytes-place`'s rule for a cell outside every
            /// loop too — it runs where it moves fewer bytes, once (a prompt's RMSNorm
            /// reads a 4 KB gain the host holds and 4 MB the device does: the gain
            /// goes up, not the 4 MB down and back). Off: such a cell needs every
            /// operand on the device already.
            pub once_bytes: bool,
            /// bf16-integrate `cand-made`: a map an operand of which a candidate
            /// device cell or site makes is a placement candidate too (S62's
            /// "operand device-produced" over the candidates, the fixed point
            /// deciding it), not only where its own roofline pays — a prompt's gather,
            /// its norm's row fold and the norm's scale are decided as one chain. Off
            /// (`--gpu-off=cand-made`): the roofline or a tile site's output alone
            /// makes a candidate (as before).
            pub cand_made: bool,
            /// full-model `once-range`: a piecewise range map (`map_law` `Outside`)
            /// outside every loop that no carry writes in place is a whole-extent
            /// device cell, run once. Off: such a map is no cell (the host runs it).
            pub once_range: bool,
            /// model-ready placement `range-chain`: a chain of in-place range maps
            /// over a carried state (a trip's appends to one cache) writes the
            /// carry's buffer link by link, each written array dead past its
            /// writer. Off: only a range map writing the next state over the view.
            pub range_chain: bool,
            /// model-ready placement `stride-lanes`: a nest cell whose fold reads an
            /// operand at unit stride along the fold (`step_algebra`'s regroup
            /// reads) takes a warp a cell — coalesced — even when its cells fill
            /// the part. Off: a warp a cell only where the cells leave threads idle.
            pub stride_lanes: bool,
            /// model-ready placement `carry-fwd`: a cell's output that is a device
            /// carry's next state forwards its pointer into the back edge — no host
            /// read there. Off: the back edge's pack is a host read.
            pub carry_fwd: bool,
            /// model-ready integration `lanes-unroll`: a warp-a-cell fold loop
            /// (`cell-lanes` / `stride-lanes`) is unrolled `NAIVE_UNROLL` trips
            /// (`llvm.loop.unroll.count`) where each lane walks at least that many —
            /// the loads in flight a lane keeps, as the stream rules price a naive
            /// thread (`NAIVE_UNROLL`), stated in the IR instead of left to `ptxas`
            /// (which unrolls the loop only when its index is loaded: an `iota-law`
            /// index left it rolled, one weight load in flight a warp). Off
            /// (`--gpu-off=lanes-unroll`): no loop metadata.
            pub lanes_unroll: bool,
            /// residency `fly-unroll`: a warp-a-cell fold whose grid holds too few
            /// warps an SM to keep the part's stream bytes in flight
            /// (`Gpu::stream_fly_per_sm`) at `NAIVE_UNROLL` loads a lane unrolls
            /// deeper — the power of two that keeps them, at most the lane's trips
            /// (a 512-row one-row projection: 6 warps an SM). Off
            /// (`--gpu-off=fly-unroll`): every lane loop `NAIVE_UNROLL`.
            pub fly_unroll: bool,
            /// ADR-0050 C4 `machine-place`: a tile site's launch is the machine
            /// model's search ([`crate::place::placed`]: the schedule the one
            /// evaluator prices lowest over every lowering's space, through the
            /// post-lowering gate, kept to the rules' launch within the model's
            /// error band), lowered by `launch_of`. Off (the default until ADR-0050
            /// C6 confirms it): the rule chain.
            pub machine_place: bool,
        }
    };
}
