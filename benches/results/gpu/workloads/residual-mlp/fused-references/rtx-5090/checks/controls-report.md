| Shape | Implementation | Wall median / p10 / p90 ms | Event median / p10 / p90 ms | Wall / Mapal | Status |
|---|---|---:|---:|---:|---|
| matrix | mapal | 0.07309 / 0.06641 / 0.102975 | 0.066976 / 0.0608 / 0.09552 | 1 | admitted |
| matrix | naive | 0.054452 / 0.05262 / 0.055644 | 0.049792 / 0.047776 / 0.05136 | 0.745 | admitted |
| matrix | cublas | 0.039069 / 0.037526 / 0.040421 | 0.034784 / 0.033248 / 0.036224 | 0.53453 | admitted |
| matrix | torch-eager | 0.115424 / 0.113 / 0.123116 | 0.101824 / 0.098688 / 0.108256 | 1.5792 | admitted |
| matrix | torch-compiled | 0.178659 / 0.171769 / 0.200893 | 0.163488 / 0.158112 / 0.187296 | 2.4444 | admitted |
| matrix-large | mapal | 0.128935 / 0.125669 / 0.134423 | 0.123008 / 0.11984 / 0.12688 | 1 | admitted |
| matrix-large | naive | 0.333153 / 0.330038 / 0.343107 | 0.328672 / 0.325728 / 0.338592 | 2.5839 | admitted |
| matrix-large | cublas | 0.09289 / 0.091998 / 0.093672 | 0.088288 / 0.087584 / 0.089248 | 0.72044 | admitted |
| matrix-large | torch-eager | 0.140602 / 0.136857 / 0.153341 | 0.123552 / 0.120256 / 0.137312 | 1.0905 | admitted |
| matrix-large | torch-compiled | 0.208715 / 0.201775 / 0.219251 | 0.191392 / 0.1848 / 0.199872 | 1.6188 | admitted |

Allocator observations (bytes; caching allocator, not all driver/private workspace):
matrix / torch-eager: {"memory_after_timing": {"allocated": 34999296, "peak_allocated": 36178944, "peak_reserved": 37748736, "reserved": 37748736}, "memory_before_timing": {"allocated": 34999296, "reserved": 37748736}}
matrix / torch-compiled: {"memory_after_timing": {"allocated": 34999296, "peak_allocated": 35523584, "peak_reserved": 138412032, "reserved": 138412032}, "memory_before_timing": {"allocated": 34999296, "reserved": 138412032}}
matrix-large / torch-eager: {"memory_after_timing": {"allocated": 55062528, "peak_allocated": 59781120, "peak_reserved": 65011712, "reserved": 65011712}, "memory_before_timing": {"allocated": 55062528, "reserved": 65011712}}
matrix-large / torch-compiled: {"memory_after_timing": {"allocated": 55062528, "peak_allocated": 57159680, "peak_reserved": 163577856, "reserved": 163577856}, "memory_before_timing": {"allocated": 55062528, "reserved": 163577856}}

Wall/Mapal>1 means that implementation is slower than Mapal under this work-region boundary.

Work-region synchronized wall/event time only; events include submission gaps and completion wakeup, not pure kernel time.
30 observations after5 warmups; empirical percentiles are descriptive, not confidence intervals.
Mapal Contract/naive use ordered fmaf gates; external FP32 reductions use predeclared componentwise budgets.
Representative values are sampled; d1024 checks four complete reference rows at declared indices.
Allocator snapshots are process caching-allocator observations; source requested-storage facts are not measured driver peaks.
Earlier admitted arms/shapes remain independently reported when a later command or shape fails.
