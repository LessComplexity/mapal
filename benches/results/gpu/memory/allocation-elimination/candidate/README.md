# Isolated dead raw allocation candidate

Base source is git-archive `3a0d4350dc2f5b13b53ed86821cf427922fe99d9` (ordering integrated). Only `func/resident.rs` and `tests/streams.rs` differ. No shared production/docs/OpenSpec/binary edit, upload, rental or GPU execution occurred. No policy change or performance claim.

Production diff adds a three-line skip in `FnEmit::chain_prologue` when the existing selected group has `store_out=false`. Missing group remains conservative: allocate as before. `emit_site_resident` already uses that same materialization fact to pass raw pointer0 and omit its store. Launch selection, physical-writer aliases, legacy-consumer veto, arena guard, member allocations, zero slot initialization, lifetime/free logic and runtime/ABI stay unchanged. There is no hoisting and no new map/helper/pass.

Exact final formatted regression source against old production fails for both arithmetic faces: independent and diamond each have one real allocated raw slot with no pointer load/use, MLP2 each has two. The control test passes old code. Production restored in finally; final fixed streams19/0. Focused integration streams19+epilogue13+resident13+rowfold6 =51/0. Backend lib restricted run56/3 failures in unchanged native hardware detection; host-access rerun59/0. Formatting passes. Full workspace was not rerun for this isolated candidate.

`proof.json` binds28 current before/after LLVM variants (14 cases ×2 faces), exact argv and hashes. All28 deviceLLVM units are byte-identical. Every whole main body, after removing only individually proved unread old allocation instructions and normalizing SSA names/entry alloca+zero initialization, is identical. This comparison retains all other body instructions, including live allocations, pointer construction, zero raw kernel arguments, events, readback and frees, in order. All event/readback streams and free call counts remain identical. `zero-slot-ownership.json` additionally proves every free-only slot remains initialized0 with no load; runtime's zero-free early return means the prior incidental live-free wait disappears. The already integrated explicit ordering remains the actual dependency contract; GPU equivalence for this new variant remains required.

| Existing source | Before→after alloc call sites, each face | Static requested bytes omitted |
|---|---:|---:|
| No-clock omitted-raw64 |8→6|32768|
| pt512 independent |8→7|1048576|
| pt512 diamond |10→9|1048576|
| pt512 MLP2 |16→14|2097152|
| pt512 twin |8→8|0|
| pt512 single-chain MLP |0→0|0|
| pt512 stored fanout |0→0|0|
| Materialized raw / fork / capture / rowfold / legacy cell controls |unchanged|0|

The existing pt fixtures contain clocks and were emitted only: do not use them as ordinary no-clock ordering gates. Static requested sums are neither measured allocation counts nor VRAM peaks nor performance. Both stored raw host reads and stored-member reads remain explicit controls. No numerical gate changed.

Recommendation: ready for independent patch review and a separately bounded matched before/after GPU correctness/allocation run; do not integrate or claim task3.2 for DCE from this static packet alone. The prior ordering400 processes and workspace1188 pass justify preparing this candidate, not its hardware acceptance.

Resume:
```
cd /private/tmp/mapal-s58-allocation/source
CARGO_TARGET_DIR=/private/tmp/mapal-s58-allocation/target cargo test --offline --locked --release -p mapal-backend-llvm --test streams --test epilogue --test resident --test rowfold
CARGO_TARGET_DIR=/private/tmp/mapal-s58-allocation/target cargo test --offline --locked --release -p mapal-backend-llvm --lib
python3 /private/tmp/mapal-s58-allocation/evidence/proof.py
```
`emit-before` is the independently built archived-base compiler; `target/release/examples/emit` is isolated candidate compiler. `receipt.json` binds their hashes, final source inventory, patch and evidence. Preserve this directory before `/private/tmp` cleanup. No jobs remain.
