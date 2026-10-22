# S61 independent B comparison

Keep B isolated: both Contract cells improve over old; both Exact cells regress beyond the observed identical-PTX null envelope. B improves all four cells versus S58, but remains slower than cuBLAS in every cell. No whole-selector promotion follows.

RTX5090,170SMs, CUDA driver API version13010 (13.1). Ordinary CUDA-event kernel milliseconds; each range spans two process upper medians,30raw events per leg/process. Ranges are descriptive, not confidence intervals. Libraries use the B/B_null processes and do not promise Mapal ordered association.

| M×N×K | Face | Old ms | S58 ms | B ms | cuBLAS ms | CUTLASS ms | B vs old |
|---|---|---|---|---|---|---|---|
| 16×2048×7168 | exact | 0.2621 | 0.3686–0.3707 | 0.3215 | 0.0532 | 0.5222–0.5243 | 1.227× slower |
| 16×2048×7168 | contract | 0.2683–0.2703 | 0.3666–0.3686 | 0.2150 | 0.0532 | 0.5243 | 1.253× faster |
| 16×1024×4096 | exact | 0.1556 | 0.2171 | 0.1577 | 0.0205 | 0.3052 | 1.013× slower |
| 16×1024×4096 | contract | 0.1495 | 0.2191 | 0.1290 | 0.0205 | 0.3052 | 1.159× faster |

Independent reader verifies963 sealed input and154 output files; all48 record identities are present; no numeric failures, quarantines or operational failures;2160 event samples reparsed. Global identical-PTX null factor `1.007454342154` (about0.7454%). All four old/B differences exceed this observed envelope, including the small Qwen Exact regression.

Bounded N256/N128 gates check every output; actual N2048/N1024 shapes check256 outputs. Exact checks ordered separate multiply/add; Contract ordered FMA. No exhaustive actual-shape claim.

| Shape family / face | Old grid/block/shared/regs | S58 grid/block/shared/regs | B grid/block/shared/regs |
|---|---|---|---|
| 16×2048×7168 exact | 128x1/256/0/40 | 64x1/32/1536/77 | 32x2/32/4608/106 |
| 16×2048×7168 contract | 128x1/256/0/40 | 64x1/32/1664/79 | 32x2/32/4864/103 |
| 16×1024×4096 exact | 64x1/256/0/40 | 32x1/32/1536/80 | 16x2/32/4608/106 |
| 16×1024×4096 contract | 64x1/256/0/40 | 32x1/32/1664/79 | 16x2/32/4864/105 |

All reported local bytes are0. B and S58 have equal total block counts (64/32) and32 threads/block. B increases shared storage and physical registers; these facts do not measure achieved occupancy, traffic, stalls or explain causation. Executed B cubins were not captured.

Inputs: `/private/tmp/mapal-s60-kernel/bcomparison`. Outputs: `/private/tmp/mapal-s61-gpu/attempt-03/retrieved/trace/kernel-B`. Full independently parsed medians/raw events/library ratios/resources/provenance: `B-independent/analysis.json`.
