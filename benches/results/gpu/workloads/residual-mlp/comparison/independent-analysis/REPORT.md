| Shape | Implementation | Wall median / p10 / p90 ms | Event median / p10 / p90 ms | Wall / Mapal | Status |
|---|---|---:|---:|---:|---|
| matrix | mapal | 0.08237 / 0.07455 / 0.09523 | 0.076128 / 0.067872 / 0.08544 | 1 | admitted |
| matrix | naive | 0.053141 / 0.0528 / 0.054581 | 0.047744 / 0.04752 / 0.048992 | 0.64515 | admitted |
| matrix | cublas | 0.05623 / 0.05363 / 0.060941 | 0.050688 / 0.047968 / 0.053088 | 0.68265 | admitted |
| matrix | torch-eager | 0.209481 / 0.204521 / 0.230271 | 0.183712 / 0.18048 / 0.205952 | 2.5432 | admitted |
| matrix | torch-compiled | 0.310192 / 0.298292 / 0.326291 | 0.285408 / 0.274496 / 0.300672 | 3.7658 | admitted |
| matrix-large | mapal | 0.1381 / 0.136001 / 0.1452 | 0.131392 / 0.129568 / 0.138816 | 1 | admitted |
| matrix-large | naive | 0.339042 / 0.335702 / 0.348171 | 0.333728 / 0.330304 / 0.342752 | 2.455 | admitted |
| matrix-large | cublas | 0.102531 / 0.100081 / 0.10644 | 0.097024 / 0.094624 / 0.100768 | 0.74244 | admitted |
| matrix-large | torch-eager | 0.223571 / 0.215731 / 0.250681 | 0.197344 / 0.190976 / 0.224192 | 1.6189 | admitted |
| matrix-large | torch-compiled | 0.366742 / 0.338292 / 0.390122 | 0.33232 / 0.310016 / 0.355296 | 2.6556 | admitted |

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
