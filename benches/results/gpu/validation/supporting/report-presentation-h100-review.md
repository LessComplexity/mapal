# H100 presentation QA addendum

2026-10-21. Author: `/root/audit_s54_worktrees`. **No numerical mismatch or scope blocker found in the new H100 additions and totals.** Read-only review; no exporter rerun, source audit, build, GPU action or admission decision. Only this new note was written. The first presentation review remains byte-identical.

Reviewed report: `docs/performance/s54-gpu-fusion.md`, SHA256 `fd0b9f9a5b44d7a6404ef2e2d5a52fd8bf9218a9fa8ed29f7b3de4c583c7af02`.
This addendum covers only the newly added H100 kernel/program/decode rows, retention and null discussion, provenance/count claims and five-primary rollup. Later report edits are not implicitly reviewed.

| New presentation | Checked result |
|---|---|
|18 H100 shipped kernel rows |All90 timing/ratio cells match `-kernels.csv` at displayed precision. Faces map Exact→_exact, FMA/Contract→_c, requested TF32→_tf. Times are CUDA-event milliseconds; ratios are library/Mapal, above1 favoring Mapal. Raw lines4446–4456,4566–4576 and4602–4612 independently agree with the displayed4096 FMA and8192 FMA/TF32 conclusions. |
|Five Contract4096 composed rows |Ordinary Mapal iteration medians, best tested library device-event medians and leg names match `-programs.csv`. They remain in the existing contiguous25-row table under its timing-boundary explanation; no incompatible-clock speed ratio is added. |
|One H100 decode row |973ms shipped and2382ms combined-control ordinary process-wall medians; control/shipped2.448 and8/12288 allocation pair match. The row continues the existing five-row table with its header and delimiter intact. The control remains explicitly residency+retention coupled, not pure retention OFF. |
|Retention ON/OFF prose |The on-retain-resident arm has traced kernel median142.827ms versus161.283ms shipped, while ordinary process-wall medians are1007ms versus973ms. The reported ordinary ranges933–1074ms and896–1501ms match and overlap. Text correctly establishes no ordinary wall-time gain and does not convert traced kernel savings into an untraced performance claim. |
|Shared decode transmission counts |H100 agrees with the stated3,144,704/1,048,576 shipped H2D/D2H bytes; delta-H2D OFF raises H2D to1,074,790,400 with8 allocations. The combined control has4,837,081,088/2,152,726,528 bytes and12,288 allocations. These remain traced observations, not measured peak memory or isolated costs. |
|H100 nulls and interpretation |Largest normalized-null deviation is3.605183678%, rounding to3.61%, at FMA1024. FMA4096 null1.0010706668 rounds to1.00107. The4096 FMA median ratios1.033/1.145 favor Mapal on this primary. At8192, cuBLAS medians are lower and pinned CUTLASS medians higher for both FMA/TF32; FMA Mapal28.7107–33.4852ms overlaps cuBLAS27.2451–31.9374ms. No independent H100 confirmation is claimed. |
|Acceptance/provenance claims |Strict verification is pass=true with zero errors:812 differential passes,22 expected traps,1230 probes,252 e2e,70 decode and46 kernel sections. Frozen head is b4152c5. Saved metadata identifies machine147981/host3497; raw header/hardware identify H100 PCIe, sm_90a, driver610.57.04 and350W. Independent absence receipt reports confirmed_absent=true at20:37:43.372269UTC, matching the rounded presentation. |
|Independent-check and new totals |H100 receipt passes with7956 numeric fields,46 kernel rows,296 program rows,118 library rows and772 losses.46×3×30=4140 raw kernel samples. Adding to the four earlier primaries yields38577 numeric fields and19800 samples exactly. The4090 confirmation is discussed separately and is correctly excluded from these five-primary totals. |

The Markdown table blocks are intact: the several-computation table has25 data rows, including the five H100 additions; the decode table has5, including H100. The H100 kernel table is a separate18-row block. The previously fixed blank-line issue does not detach the H100 rows.

The report now accepts five primaries plus the independent4090 confirmation, consistent with the saved H100 verification and cleanup. The independent5090 confirmation remains explicitly pending. This addendum does not accept that run or infer new composed/PyTorch evidence. Existing sampled-kernel numeric-gate and library-diagnostic limitations remain unchanged.

## Binding

First review, unchanged: `/private/tmp/mapal-s56-gpu/report-presentation-review.md`, SHA256 `fcf3180f472a5a7bd4292f601e67d4f859a0975075cda515ecaedb178e91e9ac`.

H100 and rollup presentation inputs, read without regeneration:

```json
{
  "benches/results-s54-gpu/s57-integrated-h100-pcie-primary-retry2.json": "285c6a013cae57aef2c0434054892f58f60277f2ac35b518ab22382e90eb7d3b",
  "benches/results-s54-gpu/s57-integrated-h100-pcie-primary-retry2.txt": "1aec766486bd84f117245adb7b4e28576456cca43c6a3aca5e192baeef2f77ce",
  "benches/results-s54-gpu/s57-integrated-h100-pcie-primary-retry2.verification.json": "be1f44257d1ff38f27533ec95f45264fdd45be04c8268a4988e4d2e552b8202b",
  "benches/results-s54-gpu/s57-integrated-h100-pcie-primary-retry2-independent-csv-check.json": "458c403076ba8e62906d74e45d8120d50a5280a22239101c170f48fb012f05a3",
  "benches/results-s54-gpu/s57-integrated-h100-pcie-primary-retry2-kernels.csv": "9217177365cc6c24a1f14b9aaaf0e247a2d306efa5e9c491f0468bbf993d3df0",
  "benches/results-s54-gpu/s57-integrated-h100-pcie-primary-retry2-programs.csv": "c1ce718a06432c3635ad6e5d70de4633cf89929c0942c8642f09ddf998240264",
  "benches/results-s54-gpu/s57-integrated-h100-pcie-primary-retry2-nulls.csv": "3470ca489d4f6c45fb4a84892e7dc4652c9183008d2688a05729d6824be10ba3",
  "benches/results-s54-gpu/s57-integrated-h100-pcie-primary-retry2-losses.csv": "9a8b70278674dc2ffb45777ac4b887a885a9f09b7c5335f11e4d950e6c5c645b",
  "benches/results-s54-gpu/s57-integrated-rtx-3090-primary-retry2-independent-csv-check.json": "2de3f0597df9d4de7864e8332cbd39d09b37b2bc0f7d825f65a1d9b663eeaa30",
  "benches/results-s54-gpu/s57-integrated-a100-primary-independent-csv-check.json": "c8b5746e717e9d0d6cb9649d6ada99a00cb8d154b6c7a09926128e4c7f35d718",
  "benches/results-s54-gpu/s57-integrated-rtx-4090-primary-independent-csv-check.json": "cf27aff5ea02de023d240c383d3b208f1abc7d6911cbc7139ecbc48e7312fad8",
  "benches/results-s54-gpu/s57-integrated-rtx-5090-primary-retry4-independent-csv-check.json": "00d6e4656847f4fda015a2e972a53c9a1421cc667cc6c7bd1854e1a78eb84bb9"
}
```
