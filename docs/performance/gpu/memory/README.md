# GPU memory, lifetime and ordering

[Performance index](../../README.md)

| Report | Scope |
| --- | --- |
| [Unused output allocation elimination](dead-output-allocation.md) | Integrated correction; ordinary correctness and separate allocation counts pass |
| [Allocation latency](allocation-latency.md) | Before/after under forced streams on four parts: stable gain on affected cells, no loss, controls unchanged |
| [Grouped-writer ordering and allocation development](ordering-and-allocation-development.md) | Dependency repair, emission checks and earlier correctness evidence |
| [Generated ordering investigation](ordering-investigation.md) | Host/device dependency analysis that led to the grouped-writer repair |
| [CUDA arena allocation](arena-allocation.md) | Earlier backend-cuda arena contract and measured allocation behavior |
