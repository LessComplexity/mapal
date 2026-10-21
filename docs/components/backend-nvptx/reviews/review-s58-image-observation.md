# Review — S58 image-observation prototype

Scope is the standalone tooling in `benches/gpu/capture_images/`, not a CUDA runtime
or compiler change. No GPU observation has been accepted.

- [x] FRAMEWORK §4.5 law 1, local boundary: callback bytes are copied before callback
  lifetime ends; the stub destroys its source buffer and the saved copy remains equal.
- [x] Law 2: `writefile` carries exact allowlisted code/identity records from the process
  to a new directory, with exclusive creation and byte/event caps.
- [x] Law 3: every admitted image binds PTX load, actual context, context UID, correlation,
  module, function and successful submission; missing joins fail local tests.
- [x] Law 4: the output boundary is explicit. No CUDA calls or synchronization were added
  by the observer. Existing harness completion is an external contract, not inferred
  from launch success.
- [x] Law 5: admission is deduced from callback records, executable/PTX/observer hashes
  and expected grid/block/shared bytes. Independent review reproduced wrong context
  and geometry acceptance in v1 and verified rejection in v2.
- [x] Law 6: instrumented and ordinary processes remain separate realizations; every
  admitted observer packet has `timing_admissible=false`.

Real CUDA/CUPTI 12.8 headers compile with warnings as errors; eleven relocated test
groups pass. Linux linking, injection, actual resource callback representation and
executed cubin disassembly remain explicit open duties. Synthetic ELF acceptance is
not evidence that nvdisasm accepts a real GPU image. Library images are outside scope.

The implementation map, architecture measurement table/diagram and component status
match the prototype. No new production component, port or placement fact was added.
Header/package hashes, rejected packets and relocated test receipts are preserved in
`benches/results-s54-gpu/s58-executed-image-capture/`.
