The blanket small-M candidate should remain isolated: three changed-PTX cells improve, four regress beyond observed null/control variation. The remaining21 admitted cells are unchanged-PTX controls; two cells fail numerical admission in both old and candidate. No universal numerical or performance acceptance follows.

234/234 record identities and raw receipts verified;106 numerical admissions,112 timed processes,28/30 timing cells admitted. All three negative controls verified. Payload2866 files, outer948 files, two harness binaries verified. Default and TF32 readbacks are0 and3; cuBLAS120804 / runtime12080. Ordinary packet elapsed140.359s within480s cap.

| Changed cell | Old ms A/B | Candidate ms A/B | Old/candidate center | Verdict |
|---|---:|---:|---:|---|
| dense M16 exact | 2.772/2.772 | 0.5673/0.5673 | 4.8863 | observed improvement beyond null/control variation |
| dense M16 contract | 2.7249/2.7423 | 0.5673/0.5673 | 4.81861 | observed improvement beyond null/control variation |
| dense M16 tf32 | 2.731/2.7369 | 0.5284/0.5263 | 5.18432 | observed improvement beyond null/control variation |
| deepseek_expert M16 exact | 0.3154/0.3154 | 0.4444/0.4444 | 0.709721 | observed regression beyond null/control variation |
| deepseek_expert M16 contract | 0.3236/0.3236 | 0.4444/0.4444 | 0.728173 | observed regression beyond null/control variation |
| qwen_expert M16 exact | 0.1823/0.1823 | 0.2597/0.258 | 0.704269 | observed regression beyond null/control variation |
| qwen_expert M16 contract | 0.1758/0.1761 | 0.258/0.258 | 0.681977 | observed regression beyond null/control variation |

Global unchanged-PTX control envelope factor1.028579. Two ABBA process observations per revision and30 raw events per process are descriptive; these envelopes are not confidence intervals.

DenseM1 requested-TF32 bounded1×512×7168 fails4×cuBLAS diagnostic in both variants (Mapal maxrel1.109e-4 vs cuBLAS2.117e-5); DeepSeek expertM1 bounded1×256×7168 similarly fails (9.399e-4 vs1.085e-4). Each reports one mismatching cell. Corresponding large timing cells are quarantined without timing. Four quarantine keys mean two bounded gates plus two dependent timing cells, not four timing cells. The new runtime/library differs from the older failing-host reference; no cross-host numerical equivalence claim follows.

Among28 admitted same-face comparisons, cuBLAS is faster beyond observed variation in25, unresolved in3, candidate wins0. CUTLASS is slower than candidate in21, faster in6, unresolved in1. These are requested-mode comparisons; library executed arithmetic is not established. SeeABSOLUTE.md for all30 sizes/faces and absolute library times.

Changed M16 Exact/Contract resources move256-thread scalar blocks to32-thread blocks, add1536/1664 shared bytes, and increase registers40→76–80; local bytes remain0. DenseTF32 uses64 threads,10752 shared bytes,48 registers instead of256/0/40. Geometry changes are measured resource facts, not a causal occupancy proof. No executed cubin/SASS or hardware counters were captured.

The runner candidate_acceptance=true checks numerical/protocol scope only and does not reject the four observed performance regressions. Keep its original field and interpretation separate from promotion.
