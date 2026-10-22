# Composed and large GPU workloads

[Performance index](../../README.md)

| Report | Scope |
| --- | --- |
| [Fused residual MLP: competitive libraries and same-device controls](fused-library-comparison.md) | Thirty admitted confirmation/control rows; matched library fusion deltas and finite tuning; cuBLASLt leads |
| [Residual MLP: Mapal, CUDA, cuBLAS and PyTorch](residual-mlp.md) | Ten admitted timing rows at two M128 widths; cuBLAS wins both; reset/cache and graph-mode limits stated |
| [Large matrices and MLP validation](large-matrices-and-mlp-validation.md) | Earlier 16K/model-shaped primitives and bounded composed correctness, including failures |
| [Composed benchmark preparation](composed-benchmark-preparation.md) | Historical contract, oracle and clock review; preparation alone is not timing evidence |
