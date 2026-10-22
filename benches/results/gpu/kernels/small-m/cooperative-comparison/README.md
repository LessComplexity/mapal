# S59 comparison evidence

The candidate remains isolated: three changed cells improve, four regress; 28/30
timing cells are admitted. See the [readable report](../../../docs/performance/s59-smallm-comparison.md),
[all absolute times](analysis/ABSOLUTE.md), [CSV](analysis/absolute-cells.csv),
[independent verdict](analysis/VERDICT.md) and [root recheck](root-verification.json).

`raw-trace.tgz` contains the complete 949-file retrieved trace, including its outer
summary, 948 hashed artifacts and both harness binaries. It contains no provider API
records. `recovery.json` identifies the ignored, private recovery archive and exact
sealed benchmark payload. Private recovery may contain credentials; do not publish it.
Cleanup and cost receipts confirm instance 57130775 is absent. Cost is estimated,
not an observed invoice. The one-use launcher is consumed.

Recheck using the existing local recovery:

```sh
python3 -B benches/results-s54-gpu/s59-smallm-comparison/analyze.py \
  --bundle /private/tmp/mapal-s58-smallm-bench \
  --results /private/tmp/mapal-s59-gpu/retrieved/trace/smallm-s59/smallm-values \
  --outer-summary /private/tmp/mapal-s59-gpu/retrieved/trace/smallm-s59/summary.json \
  --out /private/tmp/mapal-s59-independent-rerun
python3 -B benches/results-s54-gpu/s59-smallm-comparison/root-recheck.py \
  --trace /private/tmp/mapal-s59-gpu/retrieved/trace/smallm-s59 \
  --analysis benches/results-s54-gpu/s59-smallm-comparison/analysis/analysis.json \
  --out /private/tmp/mapal-s59-root-rerun.json
```

For recovery elsewhere, extract the public raw trace and the sealed payload from the
private archive into separate directories, then pass their paths explicitly. Never
rerun a rental launcher to inspect existing results. The analyst's frozen manifest
binds its original scripts/reports; the public manifest binds preserved files here.

The three derived public CSVs use LF line endings to satisfy repository whitespace
checks. Every parsed cell is unchanged; `csv-newline-normalization.json` binds original
and public hashes. Original CRLF files remain in private recovery and the analyst
frozen manifest; the public manifest binds the LF copies. Raw evidence is unchanged.
