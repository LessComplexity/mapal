# read-builtin — ADR-0045's data movement on the RTX 5080

Branch `feat/read-builtin` on `f1d3a04f`. `benches/local/read_gpu.sh` (under the bench lock): the golden program of every
element type and a failing read on the device (`mapal run --target=rtx-5080`), then a 1 GiB weight file read once
and used by every trip of a bracketed loop (`crates/mapal-cli/tests/read.rs` `gib_read_moves_once_on_the_gpu`).
Raw output stays out of git; the lines below are the test's own report.

## Intent (the facts)

The array is the file's bytes and the same on every trip (`unchanged_plan`): the bytes move once into the host array
and once into its device buffer, before the loop and before the `time` bracket; nothing per trip.

## Realized

16384 × 16384 `f32` weights (1 073 741 824 bytes) and a 16384-element start vector, a matrix-vector product per trip:

| trips | `mapal_read` 1 GiB | upload (pageable `cuMemcpyHtoD`) | uploads | bytes up | host-IR uploads per trip (`move_facts`) | run's uploads by the IR | loop (bracketed) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 111.7 ms | 41.7 ms | 2 | 1 073 807 360 | 0 | 1 073 807 360 | 2.66 ms |
| 3 | 113.1 ms | 45.0 ms | 2 | 1 073 807 360 | 0 | 1 073 807 360 | 8.22 ms |

- **One copy of the file to the device, none in the loop.** Two uploads at 1 and at 3 trips — the weights and the
  loop's start vector (the carried state's initial value, 65 536 bytes) — exactly the file's bytes plus the start;
  `evidence.py` `move_facts` finds no upload in the loop's blocks (`h2d_n` = 0 per trip) and the same bytes for the
  whole run as the trace. The upload sits before the first clock read in the host IR (asserted in
  `crates/backends/llvm/tests/read.rs` for `f32`, `bf16` and `f16` weights).
- **Host side.** `mapal_read` reads the file into the array's storage in one pass (9.6 GB/s from the page cache);
  the host IR copies no read array (`read.rs` asserts no `llvm.memcpy` names one).
- **Staged / pinned copy: not taken.** The pageable upload already runs at 25.7 GB/s (1 GiB in 41.7 ms), the bus's
  pinned rate on this part, so a pinned source saves nothing per byte and registering 1 GiB costs more than it could
  (the existing `pinned` rule: a copy that runs once is not pinned). What is left is overlap: the 41.7 ms upload
  follows the 111.7 ms read; a chunked read-and-upload of a device-only array would hide it (~27 % of the load) —
  open, it needs the read to write the device buffer, a placement the residency machinery does not express.
- **Values.** The device's result within 1e-3 relative of a host `f64` evaluation at 1 and 3 trips; the golden
  program's output equals the oracle's (`golden.out`); a 12-byte `f32.bin` where 32 are needed exits 103 with
  `mapal: read failed: …/f32.bin: expected 32 bytes, found 12` on the device build as on every other backend.
