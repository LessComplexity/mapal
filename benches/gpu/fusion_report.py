#!/usr/bin/env python3
"""Static fusion report: emit every multi-computation probe for one profile and
face, and count, per program, where the work lands — from the emitted host and
device IR alone (no GPU needed):

  tile / gen / cell kernels  `define ptx_kernel` in the device module (`mapal_tile_*`, `mapal_gen_*`;
                      a `devmap` cell kernel is a `mapal_tile_*` under a `; cell` line)
  launches, gens      host call sites of `mapal_gpu_launch` / `mapal_gpu_tile` / `mapal_gpu_launchv`,
                      `mapal_gpu_gen`
  h2d, d2h            host call sites and their constant byte counts (`mapal_gpu_tile`, the
                      S52 round trip, counts as two copies in and one out)
  host elem loops     host loops of at least n·n trips — per-cell work left on the host
  over device out     host loops that read a buffer a d2h filled — a device output consumed
                      on the host (the boundary amendment 9 says should not exist)

Counts are static: a call inside a host loop (a program `loop`) counts once and
is flagged `L`.

usage: [EMIT_FLAGS="--gpu-off=<rung>"] fusion_report.py <profile> [exact|contract|tf32] [n] [stem-glob...]
       fusion_report.py --selftest
"""
import fnmatch
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EMIT = ROOT / "target/release/examples/emit"
# The pre-S54 multi-site programs (gen_tiles.py `chain`, `share`, `loop_mm`, `decode`).
EXISTING = ["mm_chain_2048", "mm_share_1024", "loop_mm_1024_8", "decode_1024_256"]


def blocks(fn_text):
    """[(label, [lines])] in textual order; the entry block is labelled ''."""
    out, label, cur = [], "", []
    for line in fn_text.splitlines():
        m = re.match(r"^(\w[\w.]*):", line)
        if m:
            out.append((label, cur))
            label, cur = m[1], []
        else:
            cur.append(line)
    out.append((label, cur))
    return out


def loops(bbs):
    """Loop bodies as index ranges: a branch back to an earlier label is a latch."""
    index = {label: i for i, (label, _) in enumerate(bbs)}
    found = []
    for i, (_, lines) in enumerate(bbs):
        for line in lines:
            for target in re.findall(r"label %([\w.]+)", line):
                h = index.get(target)
                if h is not None and h <= i:
                    trips = None
                    for hl in bbs[h][1]:
                        m = re.search(r"icmp uge i64 %\w+, (\d+)", hl)
                        if m:
                            trips = int(m[1])
                    found.append((h, i, trips))
    return found


def const(arg):
    return int(arg) if re.fullmatch(r"-?\d+", arg) else None


def analyze(host, dev, n):
    r = dict(tile=0, gen_k=0, cell_k=0, launch=0, launch_l=0, gens=0, h2d=0, h2d_b=0, d2h=0, d2h_b=0,
             elem_loops=0, dev_loops=0, unknown_bytes=False)
    r["cell_k"] = len(re.findall(r"; cell\ndefine ptx_kernel void @mapal_tile_", dev))
    r["tile"] = len(re.findall(r"define ptx_kernel void @mapal_tile_", dev)) - r["cell_k"]
    r["gen_k"] = len(re.findall(r"define ptx_kernel void @mapal_gen_", dev))

    def add(key, b):
        r[key] += 1
        if b is None:
            r["unknown_bytes"] = True
        else:
            r[key + "_b"] += b

    for fn in re.split(r"\n(?=define )", host):
        if not fn.startswith("define"):
            continue
        bbs = blocks(fn)
        lps = loops(bbs)
        in_loop = lambda i: any(h <= i <= l for h, l, _ in lps)
        filled = set()  # host buffers a d2h wrote
        for i, (_, lines) in enumerate(bbs):
            for line in lines:
                m = re.search(r"call \w+ @mapal_gpu_(\w+)\((.*)\)", line)
                if not m:
                    continue
                op, args = m[1], [a.split()[-1] for a in m[2].split(",") if a.strip()]
                if op in ("launch", "tile", "launchv"):
                    r["launch"] += 1
                    r["launch_l"] += in_loop(i)
                if op == "tile":
                    add("h2d", None if None in (const(args[2]), const(args[4])) else const(args[2]) + const(args[4]))
                    r["h2d"] += 1
                    add("d2h", const(args[6]))
                    filled.add(args[5])
                elif op == "gen":
                    r["gens"] += 1
                elif op == "h2d":
                    add("h2d", const(args[2]))
                elif op == "d2h":
                    add("d2h", const(args[3]))
                    filled.add(args[0])
        # A map body reads its arrays through an environment struct: `store ptr
        # %buf, ptr %field` with `%field = getelementptr ..., ptr %env` makes
        # `%env` hold `%buf`.
        base = dict(re.findall(r"(%\w+) = getelementptr [^,]+, ptr (%\w+)", fn))
        holds = {}
        for x, field in re.findall(r"store ptr (%\w+), ptr (%\w+)", fn):
            holds.setdefault(base.get(field, field), set()).add(x)
        reading = []
        for h, l, trips in lps:
            body = [x for _, lines in bbs[h:l + 1] for x in lines]
            if trips is not None and trips >= n * n:
                r["elem_loops"] += 1
            names = {m for x in body for m in re.findall(r"%\w+", x)}
            if any(v in filled or holds.get(v, set()) & filled for v in names):
                reading.append((h, l))
        # Innermost only: a program `loop` around a reading loop is not a second one.
        r["dev_loops"] += sum(not any((h2, l2) != (h, l) and h <= h2 and l2 <= l for h2, l2 in reading)
                              for h, l in reading)
    return r


def mb(b):
    return f"{b / 2**20:.1f} MB" if b >= 2**20 else f"{b} B"


def report(profile, face, n, globs):
    tiles = Path(tempfile.mkdtemp(prefix="fusion-"))
    subprocess.run([sys.executable, str(ROOT / "benches/gpu/gen_tiles.py"), str(tiles)], check=True,
                   stdout=subprocess.DEVNULL)
    stems = sorted(p.stem for p in tiles.glob(f"pt_*_{n}.mapal")) + EXISTING
    if globs:
        stems = [s for s in sorted(p.stem for p in tiles.glob("*.mapal")) if any(fnmatch.fnmatch(s, g) for g in globs)]
    flags = {"exact": [], "contract": ["--contract"], "tf32": ["--contract", "--precision=tf32"]}[face]
    flags += os.environ.get("EMIT_FLAGS", "").split()  # an arm: `--gpu-off=<rung>`
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    print(f"S54 static fusion report — emit --rewrite --target={profile} {' '.join(flags)} (face {face}), HEAD {head}")
    print("Static counts from the emitted IR (benches/gpu/fusion_report.py); L = call sites inside a host loop.\n")
    print("| program | tile kernels | gen kernels | cell kernels | launches | gens | h2d calls | h2d bytes | d2h calls "
          "| d2h bytes | host elem loops | host loops over device out |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s in stems:
        dev = tiles / f"{s}.dev.ll"
        dev.unlink(missing_ok=True)
        host = subprocess.run([str(EMIT), str(tiles / f"{s}.mapal"), "-", "--rewrite", f"--target={profile}",
                               f"--device-out={dev}", *flags], capture_output=True, text=True, check=True).stdout
        size = int(re.search(r"(\d+) -> iota -> krange", (tiles / f"{s}.mapal").read_text())[1])
        r = analyze(host, dev.read_text() if dev.exists() else "", size)
        q = "?" if r["unknown_bytes"] else ""
        launches = f"{r['launch']}" + (f" ({r['launch_l']} L)" if r["launch_l"] else "")
        print(f"| {s} | {r['tile']} | {r['gen_k']} | {r['cell_k']} | {launches} | {r['gens']} | {r['h2d']} | {mb(r['h2d_b'])}{q} "
              f"| {r['d2h']} | {mb(r['d2h_b'])}{q} | {r['elem_loops']} | {r['dev_loops']} |")


def selftest():
    host = """define internal void @mapal_main() {
  %o8 = call ptr @mapal_rt_alloc(i64 1048576, i64 4)
  call void @mapal_gpu_launch(i64 1, i64 %t17, i64 %t18, i64 %t20, i32 16, i32 8, i32 128, i32 0)
  call void @mapal_gpu_d2h(ptr %o8, i64 %t20, i64 0, i64 1048576)
  br label %bb22
bb22:
  %t26 = icmp uge i64 %t25, 262144
  br i1 %t26, label %bb24, label %bb23
bb23:
  %t27 = getelementptr [262144 x float], ptr %o8, i64 0, i64 %t25
  br label %bb22
bb24:
  ret void
}
"""
    dev = ("define ptx_kernel void @mapal_tile_1(\ndefine ptx_kernel void @mapal_gen_2(\n"
           "; cell\ndefine ptx_kernel void @mapal_tile_3(")
    r = analyze(host, dev, 512)
    assert (r["tile"], r["gen_k"], r["cell_k"], r["launch"], r["d2h"], r["d2h_b"], r["elem_loops"],
            r["dev_loops"]) == (1, 1, 1, 1, 1, 1048576, 1, 1), r
    print("selftest ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    else:
        a = sys.argv[1:]
        report(a[0], a[1] if len(a) > 1 else "exact", int(a[2]) if len(a) > 2 else 2048, a[3:])
