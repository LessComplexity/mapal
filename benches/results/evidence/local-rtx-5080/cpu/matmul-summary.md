ms; par/mt: median (min); 1t: min (median)

| N | threads | Mapal fma | Mapal conf | NumPy (OpenBLAS) | NumPy / best Mapal | Mapal GFLOP/s | NumPy GFLOP/s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | whole box | 0.878 (0.709) | 0.942 (0.804) | 1.082 (0.821) | 1.23× | 2,446 | 1,985 |
| 1024 | 1 thread | 7.535 (7.612) | 7.663 (7.772) | 7.902 (8.161) | 1.05× | 285 | 272 |
| 2048 | whole box | 3.604 (3.420) | 4.017 (3.819) | 5.865 (5.065) | 1.63× | 4,767 | 2,929 |
| 4096 | whole box | 35.953 (32.508) | 35.737 (33.864) | 28.689 (27.109) | 0.80× | 3,846 | 4,791 |

(NumPy / best Mapal > 1 = Mapal faster)
