# Performance evidence organization review

The result catalog now uses topic and experiment names. Reports live under `docs/performance/{gpu,cpu,compiler,matmul,methodology}`; raw artifacts live under `benches/results/{gpu,cpu,compiler}`. The migration map records old/new locations and the unchanged hashes of all 2,802 relocated raw evidence files. Historical session logs and raw receipts are not rewritten.

This changes filesystem locations of evidence (`Dat`), not compiler dataflow, GPU placement or runtime transmissions. The baseline report transformation (`Trn baseline.py`) now finds its sibling `table.py` relative to the script, independently of the input results directory. A real before/after invocation on the same five-architecture files produces byte-identical output except for the descriptive title, with both exits zero and empty stderr.

FRAMEWORK §4.5 review: evidence identity survives relocation (laws 1–2); every moved artifact has one explicit destination (law 3); no execution dependency or completion edge changes (law 4); relative links are recomputed from their new source locations, with 372 report links checked (law 5); no numerical or performance realization is promoted (law 6). Strict OpenSpec passes 7/0 and drift reports 0 dead / 575 references. Raw capture paths remain provenance; the migration map resolves them.

The user explicitly requested descriptive titles and separated folders. New reports must follow that convention. Session numbers remain only where they identify historical evidence. [Verification](../../../../benches/results/migration-check.json) records the checks; [performance index](../../../performance/README.md) is the reading entry point.
