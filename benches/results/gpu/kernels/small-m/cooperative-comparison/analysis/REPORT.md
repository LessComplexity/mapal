Observed 234/234 scheduled records; 106 primary numeric admissions and 112 timing processes; 28/30 cells admitted.

| Family / M / face | Old ms A/B | Candidate ms A/B | Old/candidate envelope | cuBLAS/candidate A/B | CUTLASS/candidate A/B | Verdict |
|---|---:|---:|---:|---:|---:|---|
| dense / 1 / exact | 0.4687/0.4687 | 0.4669/0.4669 | 1.0039/1.0039 | 0.9979/0.9979 | 1.6098/1.6055 | unchanged PTX control; differences are not a compiler gain |
| dense / 1 / contract | 0.4649/0.4649 | 0.4649/0.4646 | 1.0000/1.0006 | 1.0022/0.9985 | 1.6167/1.6177 | unchanged PTX control; differences are not a compiler gain |
| dense / 1 / tf32 | —/— | —/— | — | — | — | not admitted: bounded full gate |
| dense / 16 / exact | 2.7720/2.7720 | 0.5673/0.5673 | 4.8863/4.8863 | 0.7834/0.7834 | 1.3171/1.3176 | observed improvement beyond null/control variation |
| dense / 16 / contract | 2.7249/2.7423 | 0.5673/0.5673 | 4.8033/4.8340 | 0.7834/0.7834 | 1.3176/1.3205 | observed improvement beyond null/control variation |
| dense / 16 / tf32 | 2.7310/2.7369 | 0.5284/0.5263 | 5.1684/5.2003 | 0.9574/0.9650 | 0.9652/0.9690 | observed improvement beyond null/control variation |
| dense / 128 / exact | 1.3251/1.3271 | 1.3248/1.3268 | 0.9987/1.0017 | 0.5457/0.5448 | 0.6014/0.6005 | unchanged PTX control; differences are not a compiler gain |
| dense / 128 / contract | 0.8684/0.8663 | 0.8684/0.8681 | 0.9976/1.0003 | 0.8301/0.8280 | 0.9218/0.9178 | unchanged PTX control; differences are not a compiler gain |
| dense / 128 / tf32 | 0.6612/0.6636 | 0.6612/0.6653 | 0.9938/1.0036 | 0.7713/0.7666 | 0.7743/0.7696 | unchanged PTX control; differences are not a compiler gain |
| deepseek_expert / 1 / exact | 0.3215/0.3215 | 0.3215/0.3236 | 0.9935/1.0000 | 0.1782/0.1771 | 2.2358/2.2278 | unchanged PTX control; differences are not a compiler gain |
| deepseek_expert / 1 / contract | 0.3052/0.3072 | 0.3052/0.3052 | 1.0000/1.0066 | 0.1877/0.1877 | 2.3621/2.3621 | unchanged PTX control; differences are not a compiler gain |
| deepseek_expert / 1 / tf32 | —/— | —/— | — | — | — | not admitted: bounded full gate |
| deepseek_expert / 16 / exact | 0.3154/0.3154 | 0.4444/0.4444 | 0.7097/0.7097 | 0.1474/0.1474 | 1.6222/1.6222 | observed regression beyond null/control variation |
| deepseek_expert / 16 / contract | 0.3236/0.3236 | 0.4444/0.4444 | 0.7282/0.7282 | 0.1474/0.1474 | 1.6215/1.6222 | observed regression beyond null/control variation |
| deepseek_expert / 16 / tf32 | 0.1882/0.1884 | 0.1884/0.1881 | 0.9989/1.0016 | 0.3153/0.3158 | 2.6635/2.6677 | unchanged PTX control; differences are not a compiler gain |
| deepseek_expert / 128 / exact | 0.6513/0.6510 | 0.6554/0.6510 | 0.9933/1.0005 | 0.2125/0.2140 | 1.1688/1.1767 | unchanged PTX control; differences are not a compiler gain |
| deepseek_expert / 128 / contract | 0.6554/0.6451 | 0.6574/0.6472 | 0.9813/1.0127 | 0.2119/0.2152 | 1.1652/1.1836 | unchanged PTX control; differences are not a compiler gain |
| deepseek_expert / 128 / tf32 | 0.1720/0.1720 | 0.1720/0.1720 | 1.0000/1.0000 | 0.4407/0.4407 | 2.9291/2.9291 | unchanged PTX control; differences are not a compiler gain |
| qwen_expert / 1 / exact | 0.1864/0.1884 | 0.1864/0.1864 | 1.0000/1.0107 | 0.1540/0.1540 | 2.2082/2.2082 | unchanged PTX control; differences are not a compiler gain |
| qwen_expert / 1 / contract | 0.1780/0.1779 | 0.1779/0.1779 | 1.0000/1.0006 | 0.1613/0.1613 | 2.3137/2.3137 | unchanged PTX control; differences are not a compiler gain |
| qwen_expert / 1 / tf32 | 0.1782/0.1782 | 0.1780/0.1782 | 1.0000/1.0011 | 0.1612/0.1611 | 1.6107/1.6089 | unchanged PTX control; differences are not a compiler gain |
| qwen_expert / 16 / exact | 0.1823/0.1823 | 0.2597/0.2580 | 0.7020/0.7066 | 0.1024/0.1031 | 1.5849/1.5953 | observed regression beyond null/control variation |
| qwen_expert / 16 / contract | 0.1758/0.1761 | 0.2580/0.2580 | 0.6814/0.6826 | 0.1031/0.1031 | 1.5953/1.5953 | observed regression beyond null/control variation |
| qwen_expert / 16 / tf32 | 0.0858/0.0856 | 0.0860/0.0840 | 0.9953/1.0214 | 0.3093/0.3167 | 3.3581/3.4381 | unchanged PTX control; differences are not a compiler gain |
| qwen_expert / 128 / exact | 0.3564/0.3707 | 0.3625/0.3604 | 0.9832/1.0286 | 0.3730/0.3751 | 1.1581/1.1648 | unchanged PTX control; differences are not a compiler gain |
| qwen_expert / 128 / contract | 0.3604/0.3643 | 0.3645/0.3625 | 0.9888/1.0050 | 0.3709/0.3730 | 1.1517/1.1581 | unchanged PTX control; differences are not a compiler gain |
| qwen_expert / 128 / tf32 | 0.0776/0.0778 | 0.0778/0.0776 | 0.9974/1.0026 | 0.6838/0.6856 | 3.7378/3.7216 | unchanged PTX control; differences are not a compiler gain |
| square_control / 4096 / exact | 4.7186/4.7186 | 4.7186/4.7165 | 1.0000/1.0004 | 0.5558/0.5558 | 0.5744/0.5747 | unchanged PTX control; differences are not a compiler gain |
| square_control / 4096 / contract | 2.6839/2.6839 | 2.6839/2.6870 | 0.9988/1.0000 | 0.9771/0.9760 | 1.0105/1.0087 | unchanged PTX control; differences are not a compiler gain |
| square_control / 4096 / tf32 | 2.1781/2.1780 | 2.1780/2.1791 | 0.9995/1.0000 | 0.9173/0.9168 | 0.9173/0.9168 | unchanged PTX control; differences are not a compiler gain |

A/B are the two ABBA process observations, not a pooled 60-event median. Event min/median/max, host wall/RSS and GPU resource changes are in processes.csv; every raw event is retained in analysis.json.

Unchanged-PTX control variation factor: 1.0286.

Every-cell ordered proof applies only to each recorded bounded gate shape; full timing shapes check 256 sampled cells.
TF32 uses the unchanged 4x cuBLAS diagnostic gate, not an absolute accuracy guarantee; library errors remain diagnostics.
Two ABBA process observations and thirty events each are descriptive; null/control envelopes are not confidence intervals.
cuBLAS mode readback does not identify its executed kernel arithmetic. Driver JIT machine-image capture is separate.
No default switch or automated production acceptance is performed.
