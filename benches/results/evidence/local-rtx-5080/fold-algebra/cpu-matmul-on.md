ms; par/mt: median (min); 1t: min (median)

| N | threads | Mapal fma | Mapal conf | NumPy (OpenBLAS) | NumPy / best Mapal | Mapal GFLOP/s | NumPy GFLOP/s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | whole box | 0.954 (0.795) | 1.053 (0.753) | 1.235 (0.767) | 1.29× | 2,252 | 1,739 |
| 1024 | 1 thread | 7.522 (7.584) | 7.673 (7.743) | 7.937 (8.045) | 1.06× | 286 | 271 |
| 2048 | whole box | 3.760 (3.364) | 3.861 (3.527) | 5.350 (4.744) | 1.42× | 4,570 | 3,211 |
| 4096 | whole box | 37.074 (32.452) | 35.657 (33.484) | 27.710 (26.312) | 0.78× | 3,854 | 4,960 |

(NumPy / best Mapal > 1 = Mapal faster)
