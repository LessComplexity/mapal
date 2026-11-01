Same-process library comparisons retain the requested numerical face; they do not prove identical library arithmetic. Exact is the ordered separate multiply/add gate; Contract is the ordered fmaf gate and permits contraction. Requested-TF32 uses the frozen 4× cuBLAS diagnostic gate, not an absolute accuracy guarantee. A/B are separate ABBA processes: old A, candidate A, candidate B (identical-PTX null), old B (identical-PTX null). No cross-face comparison is made. Nonadmitted cells retain quarantine; missing times are never zero.

Ratios are library milliseconds divided by candidate milliseconds: greater than one favors the candidate. Old/candidate ratios greater than one favor the candidate. A changed-PTX row is not a gain unless the unchanged verifier verdict exceeds the observed null/control envelope; unchanged-PTX differences are controls.

| Family / M×N×K / face | PTX | Old ms A/B | Candidate ms A/B | cuBLAS ms A/B | CUTLASS ms A/B | cuBLAS/candidate A/B | CUTLASS/candidate A/B | Verdict |
|---|---|---|---|---|---|---|---|---|
| dense / 1×18432×7168 / exact | unchanged control | 0.4687/0.4687 | 0.4669/0.4669 | 0.4659/0.4659 | 0.7516/0.7496 | 0.997858/0.997858 | 1.60977/1.60548 | unchanged PTX control; differences are not a compiler gain |
| dense / 1×18432×7168 / contract | unchanged control | 0.4649/0.4649 | 0.4649/0.4646 | 0.4659/0.4639 | 0.7516/0.7516 | 1.00215/0.998493 | 1.61669/1.61774 | unchanged PTX control; differences are not a compiler gain |
| dense / 1×18432×7168 / tf32 | unchanged control | —/— | —/— | — | — | — | — | not admitted: bounded full gate |
| dense / 16×18432×7168 / exact | changed | 2.772/2.772 | 0.5673/0.5673 | 0.4444/0.4444 | 0.7472/0.7475 | 0.78336/0.78336 | 1.31712/1.31764 | observed improvement beyond null/control variation |
| dense / 16×18432×7168 / contract | changed | 2.7249/2.7423 | 0.5673/0.5673 | 0.4444/0.4444 | 0.7475/0.7491 | 0.78336/0.78336 | 1.31764/1.32047 | observed improvement beyond null/control variation |
| dense / 16×18432×7168 / tf32 | changed | 2.731/2.7369 | 0.5284/0.5263 | 0.5059/0.5079 | 0.51/0.51 | 0.957419/0.965039 | 0.965178/0.969029 | observed improvement beyond null/control variation |
| dense / 128×18432×7168 / exact | unchanged control | 1.3251/1.3271 | 1.3248/1.3268 | 0.7229/0.7229 | 0.7967/0.7967 | 0.545667/0.544845 | 0.601374/0.600467 | unchanged PTX control; differences are not a compiler gain |
| dense / 128×18432×7168 / contract | unchanged control | 0.8684/0.8663 | 0.8684/0.8681 | 0.7209/0.7188 | 0.8005/0.7967 | 0.830147/0.828015 | 0.92181/0.917751 | unchanged PTX control; differences are not a compiler gain |
| dense / 128×18432×7168 / tf32 | unchanged control | 0.6612/0.6636 | 0.6612/0.6653 | 0.51/0.51 | 0.512/0.512 | 0.771325/0.766571 | 0.77435/0.769578 | unchanged PTX control; differences are not a compiler gain |
| deepseek_expert / 1×2048×7168 / exact | unchanged control | 0.3215/0.3215 | 0.3215/0.3236 | 0.0573/0.0573 | 0.7188/0.7209 | 0.178227/0.17707 | 2.23577/2.22775 | unchanged PTX control; differences are not a compiler gain |
| deepseek_expert / 1×2048×7168 / contract | unchanged control | 0.3052/0.3072 | 0.3052/0.3052 | 0.0573/0.0573 | 0.7209/0.7209 | 0.187746/0.187746 | 2.36206/2.36206 | unchanged PTX control; differences are not a compiler gain |
| deepseek_expert / 1×2048×7168 / tf32 | unchanged control | —/— | —/— | — | — | — | — | not admitted: bounded full gate |
| deepseek_expert / 16×2048×7168 / exact | changed | 0.3154/0.3154 | 0.4444/0.4444 | 0.0655/0.0655 | 0.7209/0.7209 | 0.14739/0.14739 | 1.62219/1.62219 | observed regression beyond null/control variation |
| deepseek_expert / 16×2048×7168 / contract | changed | 0.3236/0.3236 | 0.4444/0.4444 | 0.0655/0.0655 | 0.7206/0.7209 | 0.14739/0.14739 | 1.62151/1.62219 | observed regression beyond null/control variation |
| deepseek_expert / 16×2048×7168 / tf32 | unchanged control | 0.1882/0.1884 | 0.1884/0.1881 | 0.0594/0.0594 | 0.5018/0.5018 | 0.315287/0.315789 | 2.66348/2.66773 | unchanged PTX control; differences are not a compiler gain |
| deepseek_expert / 128×2048×7168 / exact | unchanged control | 0.6513/0.651 | 0.6554/0.651 | 0.1393/0.1393 | 0.766/0.766 | 0.212542/0.213978 | 1.16875/1.17665 | unchanged PTX control; differences are not a compiler gain |
| deepseek_expert / 128×2048×7168 / contract | unchanged control | 0.6554/0.6451 | 0.6574/0.6472 | 0.1393/0.1393 | 0.766/0.766 | 0.211895/0.215235 | 1.1652/1.18356 | unchanged PTX control; differences are not a compiler gain |
| deepseek_expert / 128×2048×7168 / tf32 | unchanged control | 0.172/0.172 | 0.172/0.172 | 0.0758/0.0758 | 0.5038/0.5038 | 0.440698/0.440698 | 2.92907/2.92907 | unchanged PTX control; differences are not a compiler gain |
| qwen_expert / 1×1024×4096 / exact | unchanged control | 0.1864/0.1884 | 0.1864/0.1864 | 0.0287/0.0287 | 0.4116/0.4116 | 0.15397/0.15397 | 2.20815/2.20815 | unchanged PTX control; differences are not a compiler gain |
| qwen_expert / 1×1024×4096 / contract | unchanged control | 0.178/0.1779 | 0.1779/0.1779 | 0.0287/0.0287 | 0.4116/0.4116 | 0.161327/0.161327 | 2.31366/2.31366 | unchanged PTX control; differences are not a compiler gain |
| qwen_expert / 1×1024×4096 / tf32 | unchanged control | 0.1782/0.1782 | 0.178/0.1782 | 0.0287/0.0287 | 0.2867/0.2867 | 0.161236/0.161055 | 1.61067/1.60887 | unchanged PTX control; differences are not a compiler gain |
| qwen_expert / 16×1024×4096 / exact | changed | 0.1823/0.1823 | 0.2597/0.258 | 0.0266/0.0266 | 0.4116/0.4116 | 0.102426/0.103101 | 1.58491/1.59535 | observed regression beyond null/control variation |
| qwen_expert / 16×1024×4096 / contract | changed | 0.1758/0.1761 | 0.258/0.258 | 0.0266/0.0266 | 0.4116/0.4116 | 0.103101/0.103101 | 1.59535/1.59535 | observed regression beyond null/control variation |
| qwen_expert / 16×1024×4096 / tf32 | unchanged control | 0.0858/0.0856 | 0.086/0.084 | 0.0266/0.0266 | 0.2888/0.2888 | 0.309302/0.316667 | 3.35814/3.4381 | unchanged PTX control; differences are not a compiler gain |
| qwen_expert / 128×1024×4096 / exact | unchanged control | 0.3564/0.3707 | 0.3625/0.3604 | 0.1352/0.1352 | 0.4198/0.4198 | 0.372966/0.375139 | 1.15807/1.16482 | unchanged PTX control; differences are not a compiler gain |
| qwen_expert / 128×1024×4096 / contract | unchanged control | 0.3604/0.3643 | 0.3645/0.3625 | 0.1352/0.1352 | 0.4198/0.4198 | 0.370919/0.372966 | 1.15171/1.15807 | unchanged PTX control; differences are not a compiler gain |
| qwen_expert / 128×1024×4096 / tf32 | unchanged control | 0.0776/0.0778 | 0.0778/0.0776 | 0.0532/0.0532 | 0.2908/0.2888 | 0.683805/0.685567 | 3.73779/3.72165 | unchanged PTX control; differences are not a compiler gain |
| square_control / 4096×4096×4096 / exact | unchanged control | 4.7186/4.7186 | 4.7186/4.7165 | 2.6225/2.6214 | 2.7105/2.7105 | 0.555779/0.555793 | 0.574429/0.574685 | unchanged PTX control; differences are not a compiler gain |
| square_control / 4096×4096×4096 / contract | unchanged control | 2.6839/2.6839 | 2.6839/2.687 | 2.6225/2.6225 | 2.7121/2.7105 | 0.977123/0.975996 | 1.01051/1.00875 | unchanged PTX control; differences are not a compiler gain |
| square_control / 4096×4096×4096 / tf32 | unchanged control | 2.1781/2.178 | 2.178/2.1791 | 1.9978/1.9978 | 1.9978/1.9978 | 0.917264/0.916801 | 0.917264/0.916801 | unchanged PTX control; differences are not a compiler gain |

Numerical admission, raw event medians, null/control noise, resources, hashes and rejected records remain in analysis.json and REPORT.md. This table adds no acceptance rule.
