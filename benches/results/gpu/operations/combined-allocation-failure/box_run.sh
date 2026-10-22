#!/bin/bash
set -euo pipefail
cd /w
export PYTHONDONTWRITEBYTECODE=1
mkdir -p trace
finish() {
 rc=$?
 if [ ! -s results-rtx-5090.txt ]; then
  printf '{"pass":false,"timing_admitted":false,"error":"Combined adapter stopped before summary","exit":%s}\n' "$rc" > results-rtx-5090.txt
  cp results-rtx-5090.txt trace/adapter-failure.json
 fi
}
trap finish EXIT
python3 - <<'PY'
import hashlib,json,pathlib
for name,digest in json.loads(pathlib.Path('sealed-files.json').read_text()).items():
 assert hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()==digest,name
PY
mkdir -p smallm-packet
tar xzf smallm-packet.tgz -C smallm-packet
tar xzf matrix-upload.tgz
tar xzf capture-proof-source.tgz
python3 run_combined.py
