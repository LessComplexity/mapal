# GPU kernel comparisons

[Performance index](../../README.md) · [Current GPU results](../current-results.md)

| Report | Scope |
| --- | --- |
| [Tile geometry and panel comparison](panel-depth-comparison.md) | RTX 5090, two M16 shapes, Exact/FMA; deeper tile wins FMA and loses Exact versus original |
| [Cooperative small-M geometry](small-m-geometry-comparison.md) | RTX 5090 projection sweep; three improvements and four regressions, no promotion |
| [CUTLASS configuration audit](cutlass-baseline-audit.md) | Source and launch geometry audit of the historical default; competitive tuning awaits GPU execution |
| [Frozen implementation versus libraries](library-comparison.md) | Historical multi-architecture measurements with arithmetic and selection limits |
| [Integrated optimization ladder](optimization-ladder.md) | Historical per-feature timing, composition and resource comparisons |
