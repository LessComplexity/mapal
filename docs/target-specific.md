# Target-specific steps after lowering (ADR-0050 D6)

What a lowering does that is not (yet) a machine fact, keyed by properties: each entry says why it is not a fact and
what would make it one. ADR-0050: this list must shrink over time and must not grow with each new target.

**Starting size (step 1, C2 coverage test `place::tests::rules_launches_are_points_of_the_space`, every tile-site
launch of the suite on eight GPU profiles): 2 rung steps a schedule carries beyond its site's rung table, plus the
whole rung table itself.**

| step | launches carrying it (C2) | why not a fact yet | what makes it one |
| --- | ---: | --- | --- |
| `wide-fma` priced off per site (`rules::wide_fma_pays`) — the FMA tile's whole-file register rule switched by the grid's blocks against a wave | 11 112 | a tile choice made by a rung, not a coordinate of the tile | the register footprint as a schedule coordinate (`held`), the occupancy derived from it — the search then needs no rung |
| `idx32` forced on (32-bit indices where a launch's index range fits) | 463 | an index-width encoding (`ptxas` register pressure), not a machine property | a code generator's register count fed back by the post-lowering gate (C4) |
| the rung table (`rules::rungs`: `GpuRungs` over the site's fill and face) | every launch | 167 code-generation flags decided per part property and fill; the schedule names the geometry, the table the encoding | each rung either a schedule coordinate (layout, depth, traversal) or a measured codegen fact; listed as the replay retires them |

Also recorded by the coverage test (not a step, a finding): 117 launches whose rules' `blocks_per_sm` exceeds the
occupancy their declared footprint derives where `.minnctapersm` is not emitted — the rules' residency metadata, not
the hardware's (the evaluator takes the footprint's).
