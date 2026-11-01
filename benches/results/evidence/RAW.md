# Raw run output

Profile archives (nsys timelines, SASS, `ptxas -v`), box run logs and raw result text are not kept in git. They are
assets of the GitHub release **`evidence-raw-2026-10-29`**, one `raw-<sprint>.tar.gz` per evidence directory, with
paths inside matching `benches/results/evidence/<sprint>/…`. [`RAW-manifest.tsv`](RAW-manifest.tsv) lists every file
(path, bytes, sha256, the commit it came from, asset).

```sh
gh release download evidence-raw-2026-10-29 -p 'raw-site-geometry.tar.gz' && tar -xzf raw-site-geometry.tar.gz
```

New sweeps: `benches/gpu/archive_raw.sh <release-tag> benches/results/evidence/<sprint>` uploads the raw files and
appends their rows to the manifest; commit only the evidence tables, reports and the manifest.
