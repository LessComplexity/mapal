# GPU disassembly and hardware evidence

[Performance index](../../README.md)

| Report | Scope |
| --- | --- |
| [Actually executed Mapal kernels](executed-mapal-images.md) | Four Contract launches bound to captured ELF/SASS; static instructions and resources, no profiling timing |
| [Mapal, cuBLAS and CUTLASS gaps](cublas-cutlass-gap.md) | Earlier PTX/static SASS audit; actual selected library kernels remain unproved |
| [Mapal versus CUTLASS on RTX 4090](rtx4090-mapal-cutlass.md) | Historical n=4096 static assembly and hardware mapping review |
| [Accepted selection kernel evidence](selected-kernel-evidence.md) | Exact emitted kernel identities supporting frozen architecture choices |

[Existing IR facts and backend realization gaps](ir-facts-and-realization-gaps.md) maps shapes, layout, precision, fusion, lifetime and hardware limits to their actual consumers.
