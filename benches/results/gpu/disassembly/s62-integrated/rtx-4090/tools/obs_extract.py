#!/usr/bin/env python3
"""s62 disasm (rtx-4090), on the box: one counter-free observer capture dir
(capture.cpp's events.tsv + module-N.cubin) -> the kernels launched (inside the
cudaProfilerStart/Stop region when there is one, else every launch), each joined
by symbol name to the saved module image that defines it, its resource usage
(cuobjdump -res-usage of that image) and its SASS (cuobjdump -sass of that image,
the one function's section). Writes <dir>/kernels.tsv and <dir>/sass/<n>.sass.
usage: obs_extract.py <capture_dir>"""
import re, subprocess, sys
from pathlib import Path

d = Path(sys.argv[1])
ev = (d / "events.tsv").read_text().splitlines()
unhex = lambda h: bytes.fromhex(h).decode(errors="replace") if h else ""
launches, region = [], any(l.startswith("region-start") for l in ev)
for l in ev:
    f = l.split("\t")
    # launch seq cbid enter corr rtcorr ctx ctxuid func stream grid block smem active code symbol inv rtinv (f[1]..f[17])
    if f[0] == "launch" and f[3] == "0" and f[14] == "0" and (f[13] == "1" or not region):
        launches.append({"symbol": unhex(f[15]), "grid": f[10], "block": f[11], "dyn_smem": f[12], "in_region": f[13]})
kern_act = {}
for l in ev:
    f = l.split("\t")
    if f[0] == "kernel":
        kern_act.setdefault(unhex(f[-1]), (f[6], f[7], f[8]))
cubins = sorted(d.glob("module-*.cubin"), key=lambda p: int(re.search(r"\d+", p.name).group()))
res, sass = {}, {}
for c in cubins:
    r = subprocess.run(["cuobjdump", "-res-usage", str(c)], capture_output=True, text=True, timeout=300).stdout
    for m in re.finditer(r"Function ([^\s:]+):\s*\n\s*(REG:.*)", r):
        res.setdefault(m.group(1), (c.name, m.group(2).strip()))
want = {L["symbol"] for L in launches}
for c in {res[s][0] for s in want if s in res}:
    t = subprocess.run(["cuobjdump", "-sass", str(d / c)], capture_output=True, text=True, timeout=300).stdout
    for part in re.split(r"\n(?=\s*Function : )", t):
        m = re.match(r"\s*Function : (\S+)", part)
        if m and m.group(1) in want:
            sass[m.group(1)] = part
(d / "sass").mkdir(exist_ok=True)
rows = ["n\tsymbol\tgrid\tblock\tdyn_smem\tin_region\timage\tres_usage\tsass_lines"]
for i, L in enumerate(launches):
    s = L["symbol"]
    img, ru = res.get(s, ("?", "?"))
    txt = sass.get(s, "")
    if txt:
        (d / "sass" / f"{i}.sass").write_text(txt)
    rows.append(f"{i}\t{s}\t{L['grid']}\t{L['block']}\t{L['dyn_smem']}\t{L['in_region']}\t{img}\t{ru}\t{txt.count(chr(10))}")
(d / "kernels.tsv").write_text("\n".join(rows) + "\n")
print("\n".join(rows))
