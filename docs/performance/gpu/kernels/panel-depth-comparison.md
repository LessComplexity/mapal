# Small-M tile geometry and panel comparison

This RTX 5090 comparison tests the original independent-output kernel, a cooperative
16×32 tile with K8 panels, and a deeper 8×64 tile with K16 panels. Captured artifacts
call the cooperative tile `S58` and the deeper tile `B`; those labels are provenance only.

Independent review admits all 48 records and 2,160 ordinary CUDA-event
samples: 963 sealed input files and 154 output files match. Actual kernels check 256
sampled outputs; bounded N256/N128 cases check every output. No numerical or operational
failure occurs. The identical-PTX null envelope is 0.7454%. Each range below spans two
process upper medians with 30 events per leg; it is not a confidence interval.

| M×N×K | Face | Original ms | Cooperative 16×32/K8 ms | Deeper 8×64/K16 ms | cuBLAS ms | Default CUTLASS ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 16×2048×7168 | Exact | 0.2621 | 0.3686–0.3707 | 0.3215 | 0.0532 | 0.5222–0.5243 |
| 16×2048×7168 | FMA | 0.2683–0.2703 | 0.3666–0.3686 | 0.2150 | 0.0532 | 0.5243 |
| 16×1024×4096 | Exact | 0.1556 | 0.2171 | 0.1577 | 0.0205 | 0.3052 |
| 16×1024×4096 | FMA | 0.1495 | 0.2191 | 0.1290 | 0.0205 | 0.3052 |

The deeper panel improves both original FMA cells by 1.253× and 1.159×, but Exact is 1.227× and 1.013×
slower. All four differences exceed the observed null envelope. It beats the cooperative
tile and pinned CUTLASS configuration in all four cells; cuBLAS beats it in all four.
Library arithmetic does not promise Mapal's ordered association. These four cells do not
certify whole-selector profitability. In particular, the source/PTX audit proves the deeper-panel
alternative does not retain the cooperative alternative's wide-shape routes, so these are not cumulative patches.

Runtime resource records show both cooperative alternatives have the same 64/32 total blocks
and 32 threads per block. The deeper panel changes tile aspect, increases shared bytes and
uses 103–106 physical registers versus 77–80. Local bytes are reported as zero. This rules
out more total blocks as the explanation for its gain; it does not establish traffic,
achieved occupancy, stalls or causation. Executed deeper-panel cubins are not in the capture packet.

[Full table/resources](../../../../benches/results/gpu/kernels/small-m/panel-depth-comparison/results.md)
and [independent parsed evidence](../../../../benches/results/gpu/kernels/small-m/panel-depth-comparison/independent-analysis/analysis.json)
preserve all rows and limitations. The CUDA driver API reports 13010 (13.1); that number
is not an NVIDIA driver build string.


The alternative remains isolated. The competitive CUTLASS experiment is separate from these historical default-configuration measurements.
