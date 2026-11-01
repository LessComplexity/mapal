Frozen host optimization evidence, no runtime or GPU execution.

Input is the already emitted RTX5090 Contract recurrent T16/d8/h16 run host
module, copied from `composition/certification-01`. Its original manifest hash
and emission command are preserved. The frozen clang22.1.8 generated optimized
host LLVM, x86_64 Linux object and assembly with `-O2`, matching the host-object
optimization level used by the existing bundle. `manifest.json` records exact
commands/tool/input/output hashes. `before.json` equals `after.json`: all328
frozen source files, four emitter/interpreter/runtime inputs, three LLVM tools
and HEAD were unchanged. Only new temporary outputs were produced; no link,
cargo, existing binary change or GPU execution occurred.

1. Host query generation simplifies around the GPU calls. Before optimization,
   `host.emitted.ll:1820` starts an8-element scalar loop, materializing arguments
   and calling `fn12` for each query element before the32-byte q upload at1843.
   After optimization, `host.O2.ll:403` computes the same eight values in two
   four-lane vectors: integer remainder/coding, `sitofp`, and multiply by1/256,
   followed by two vector stores and the q upload at422. The division by the
   constant256 is reduced to this exact power-of-two multiply. Main and helper
   scaffolding is simplified too: module allocas697→22 and function definitions
   28→1. These module totals include unused fallback helper bodies and do not
   count executed instructions or measure speedup.

2. GPU runtime boundaries remain external calls. Static call-site counts are
   unchanged:16alloc,16free,7gen,4launch,3launchv,3h2d,1d2h,1init. All seven
   static launch IDs remain in function order. In optimized LLVM the first two
   K/V launches are at398/402, the vector query calculation follows at403–420,
   and the q upload remains at422. Assembly also retains external runtime calls.
   This demonstrates ordinary CPU optimization within a mixed program; it does
   not establish runtime overlap, thread parallelism, synchronization behavior,
   dynamic call counts or a performance gain.

The object is newly generated evidence, not a linked executable. The assembly is
compiler output, not a disassembly of a running process. Line-referenced excerpts
and counts are independently recorded in `observations.json`; original and
optimized modules are both retained for inspection.
