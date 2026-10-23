| MLP | face | class | Mapal ms | vs cuBLAS eager | vs cuBLASLt fused | vs stock CUTLASS fused | vs tuned CUTLASS fused |
|---|---|---|---:|---|---|---|---|
| d256 | Exact | kernels only (nsys GPU, L2-warm) | 0.0228 | worse (1.75) | worse (2.09) | better (0.37) | worse (1.40) |
| d256 | Exact | warm iteration (Mapal EAGER `iter ms` / library steady events) | 0.0296 | worse (1.28) | worse (1.49) | better (0.46) | worse (1.03) |
| d256 | Exact | cold single iteration (Mapal LAZY = shipped / library first call) | 0.0669 | better (0.004) | worse (1.19) | better (0.35) | better (0.68) |
| d256 | Contract | kernels only (nsys GPU, L2-warm) | 0.0159 | worse (1.22) | worse (1.46) | better (0.26) | on-par (0.98) |
| d256 | Contract | warm iteration (Mapal EAGER `iter ms` / library steady events) | 0.0394 | worse (1.71) | worse (1.98) | better (0.61) | worse (1.37) |
| d256 | Contract | cold single iteration (Mapal LAZY = shipped / library first call) | 0.1899 | better (<0.02) | worse (3.38) | on-par (1.00) | worse (1.94) |
| d1024 | Exact | kernels only (nsys GPU, L2-warm) | 0.0785 | worse (1.96) | worse (2.01) | better (0.35) | worse (1.19) |
| d1024 | Exact | warm iteration (Mapal EAGER `iter ms` / library steady events) | 0.0822 | worse (1.16) | worse (1.21) | better (0.35) | better (0.85) |
| d1024 | Exact | cold single iteration (Mapal LAZY = shipped / library first call) | 0.1124 | better (<0.02) | better (<0.02) | better (0.37) | better (0.91) |
| d1024 | Contract | kernels only (nsys GPU, L2-warm) | 0.0564 | worse (1.41) | worse (1.44) | better (0.25) | better (0.86) |
| d1024 | Contract | warm iteration (Mapal EAGER `iter ms` / library steady events) | 0.0768 | worse (1.08) | worse (1.13) | better (0.33) | better (0.79) |
| d1024 | Contract | cold single iteration (Mapal LAZY = shipped / library first call) | 0.1750 | better (<0.02) | better (<0.02) | better (0.58) | worse (1.41) |
