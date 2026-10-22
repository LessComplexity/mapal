# GPU disassembly and hardware evidence

[Performance index](../../README.md)

| Report | Scope |
| --- | --- |
| [Actually executed library kernels](executed-library-images.md) | Fresh Mapal1/cuBLAS2/CUTLASS1 selected entries, numerical gates, qualified SASS and grounded next experiments |
| [Actually executed Mapal kernels](executed-mapal-images.md) | Four Contract launches bound to captured ELF/SASS; static instructions and resources, no profiling timing |
| [Mapal, cuBLAS and CUTLASS gaps](cublas-cutlass-gap.md) | Earlier PTX/static SASS audit, preceding the fresh selected-library capture above |
| [Mapal versus CUTLASS on RTX 4090](rtx4090-mapal-cutlass.md) | Historical n=4096 static assembly and hardware mapping review |
| [Accepted selection kernel evidence](selected-kernel-evidence.md) | Exact emitted kernel identities supporting frozen architecture choices |
| [GPU optimization questions and decisive tests](optimization-experiments.md) | Evidence-ranked experiments for geometry, submission, retention, allocation, fusion and numerical order |
| [Scheduling follows realized arithmetic](fact-driven-scheduling.md) | Proved policy/emission mismatch corrected; source tests and before/after LLVM/PTX evidence |

[Existing IR facts and backend realization gaps](ir-facts-and-realization-gaps.md) maps shapes, layout, precision, fusion, lifetime and hardware limits to their actual consumers.
