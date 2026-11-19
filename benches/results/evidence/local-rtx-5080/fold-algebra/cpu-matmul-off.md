ms; par/mt: median (min); 1t: min (median)

| N | threads | Mapal fma | Mapal conf | NumPy (OpenBLAS) | NumPy / best Mapal | Mapal GFLOP/s | NumPy GFLOP/s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | whole box | 0.860 (0.731) | 0.983 (0.739) | 1.948 (0.975) | 2.27× | 2,497 | 1,102 |
| 1024 | 1 thread | 7.673 (7.780) | 7.725 (7.966) | 8.065 (8.414) | 1.05× | 280 | 266 |
| 2048 | whole box | 3.657 (3.543) | 4.000 (3.703) | 7.059 (5.391) | 1.93× | 4,698 | 2,434 |
| 4096 | whole box | 39.841 (34.832) | 37.492 (35.423) | 31.491 (28.938) | 0.84× | 3,666 | 4,364 |

(NumPy / best Mapal > 1 = Mapal faster)
