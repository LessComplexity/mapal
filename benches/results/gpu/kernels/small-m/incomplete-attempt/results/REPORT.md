Observed 1/234 scheduled records; 0 primary numeric admissions and 0 timing processes; 0/30 cells admitted.
No admitted performance comparison exists; no speedup, library ranking or numeric-quarantine claim is supported.
Execution stopped: FileNotFoundError: [Errno 2] No such file or directory: '/usr/bin/time'

| Family / M / face | Old ms A/B | Candidate ms A/B | Old/candidate envelope | cuBLAS/candidate A/B | CUTLASS/candidate A/B | Verdict |
|---|---:|---:|---:|---:|---:|---|
| dense / 1 / exact | —/— | —/— | — | — | — | not admitted: bounded full gate |
| dense / 1 / contract | —/— | —/— | — | — | — | not admitted: bounded full gate |
| dense / 1 / tf32 | —/— | —/— | — | — | — | not admitted: bounded full gate |
| dense / 16 / exact | —/— | —/— | — | — | — | not admitted: bounded full gate |
| dense / 16 / contract | —/— | —/— | — | — | — | not admitted: bounded full gate |
| dense / 16 / tf32 | —/— | —/— | — | — | — | not admitted: bounded full gate |
| dense / 128 / exact | —/— | —/— | — | — | — | not admitted: bounded full gate |
| dense / 128 / contract | —/— | —/— | — | — | — | not admitted: bounded full gate |
| dense / 128 / tf32 | —/— | —/— | — | — | — | not admitted: bounded full gate |
| deepseek_expert / 1 / exact | —/— | —/— | — | — | — | not admitted: bounded full gate |
| deepseek_expert / 1 / contract | —/— | —/— | — | — | — | not admitted: bounded full gate |
| deepseek_expert / 1 / tf32 | —/— | —/— | — | — | — | not admitted: bounded full gate |
| deepseek_expert / 16 / exact | —/— | —/— | — | — | — | not admitted: bounded full gate |
| deepseek_expert / 16 / contract | —/— | —/— | — | — | — | not admitted: bounded full gate |
| deepseek_expert / 16 / tf32 | —/— | —/— | — | — | — | not admitted: bounded full gate |
| deepseek_expert / 128 / exact | —/— | —/— | — | — | — | not admitted: bounded full gate |
| deepseek_expert / 128 / contract | —/— | —/— | — | — | — | not admitted: bounded full gate |
| deepseek_expert / 128 / tf32 | —/— | —/— | — | — | — | not admitted: bounded full gate |
| qwen_expert / 1 / exact | —/— | —/— | — | — | — | not admitted: bounded full gate |
| qwen_expert / 1 / contract | —/— | —/— | — | — | — | not admitted: bounded full gate |
| qwen_expert / 1 / tf32 | —/— | —/— | — | — | — | not admitted: bounded full gate |
| qwen_expert / 16 / exact | —/— | —/— | — | — | — | not admitted: bounded full gate |
| qwen_expert / 16 / contract | —/— | —/— | — | — | — | not admitted: bounded full gate |
| qwen_expert / 16 / tf32 | —/— | —/— | — | — | — | not admitted: bounded full gate |
| qwen_expert / 128 / exact | —/— | —/— | — | — | — | not admitted: bounded full gate |
| qwen_expert / 128 / contract | —/— | —/— | — | — | — | not admitted: bounded full gate |
| qwen_expert / 128 / tf32 | —/— | —/— | — | — | — | not admitted: bounded full gate |
| square_control / 4096 / exact | —/— | —/— | — | — | — | not admitted: actual 256-sample gate |
| square_control / 4096 / contract | —/— | —/— | — | — | — | not admitted: actual 256-sample gate |
| square_control / 4096 / tf32 | —/— | —/— | — | — | — | not admitted: actual 256-sample gate |

A/B are the two ABBA process observations, not a pooled 60-event median. Event min/median/max, host wall/RSS and GPU resource changes are in processes.csv; every raw event is retained in analysis.json.

Unchanged-PTX control variation factor: not measured.

Every-cell ordered proof applies only to each recorded bounded gate shape; full timing shapes check 256 sampled cells.
TF32 uses the unchanged 4x cuBLAS diagnostic gate, not an absolute accuracy guarantee; library errors remain diagnostics.
Two ABBA process observations and thirty events each are descriptive; null/control envelopes are not confidence intervals.
cuBLAS mode readback does not identify its executed kernel arithmetic. Driver JIT machine-image capture is separate.
No default switch or automated production acceptance is performed.
