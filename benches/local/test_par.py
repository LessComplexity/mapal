#!/usr/bin/env python3
"""The workspace's test binaries side by side (cargo test runs them one after another).

usage: benches/local/test_par.py [--bins N] [--logs DIR] [--no-doc] [cargo test selection args...]
       e.g. benches/local/test_par.py --workspace          (the default selection)
            benches/local/test_par.py -p mapal-backend-llvm

Builds every selected test binary once (`cargo test --release --no-run`), then runs them N at a time
(`--bins`, default: the CPUs this process may run on / 8), longest first by the last run's times
(`target/test-par-times.tsv`), each in its package directory (as cargo does) and its own process group,
its output in `<logs>/<binary>.log`. The environment passes through (RUST_TEST_THREADS, MAPAL_TESTGEN_JOBS,
MAPAL_TEST_JOBS, MAPAL_PAR, ...): the binaries check exactly what `cargo test` runs them to check; only
their order in time differs. Doc tests run as one more job (`--no-doc`: not).

Thermal guard: above TEST_HOT C (default 85; `sensors` Tctl) every running binary's process group is
stopped until TEST_COOL C (default 78); paused seconds are reported. Exit status 1 when any binary fails.
"""
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TIMES = ROOT / "target" / "test-par-times.tsv"


def args():
    a = sys.argv[1:]
    bins, logs, doc, sel = None, None, True, []
    while a:
        x = a.pop(0)
        if x == "--bins":
            bins = int(a.pop(0))
        elif x == "--logs":
            logs = Path(a.pop(0))
        elif x == "--no-doc":
            doc = False
        else:
            sel.append(x)
    cpus = len(os.sched_getaffinity(0))
    return bins or max(1, cpus // 8), logs or ROOT / "target" / "test-par-logs", doc, sel or ["--workspace"]


def build(sel):
    cmd = ["cargo", "test", "--release", "--no-run", "--message-format=json", *sel]
    out = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, check=True, text=True).stdout
    jobs = []
    for line in out.splitlines():
        m = json.loads(line)
        if m.get("reason") != "compiler-artifact" or not m.get("executable") or not m["profile"]["test"]:
            continue
        name = f'{m["target"]["name"]}-{m["target"]["kind"][0]}'
        jobs.append((name, [m["executable"]], Path(m["manifest_path"]).parent))
    return jobs


def tctl():
    try:
        out = subprocess.run(["sensors"], stdout=subprocess.PIPE, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        if line.startswith("Tctl"):
            return float(line.split()[1].strip("+°C"))
    return None


def signal_group(p, sig):
    try:
        os.killpg(p.pid, sig)
    except ProcessLookupError:
        pass


def main():
    nbins, logs, doc, sel = args()
    logs.mkdir(parents=True, exist_ok=True)
    jobs = build(sel)
    if doc:
        jobs.append(("doctests", ["cargo", "test", "--release", "--doc", *sel], ROOT))
    last = {}
    if TIMES.exists():
        for line in TIMES.read_text().splitlines():
            k, v = line.split("\t")
            last[k] = float(v)
    # Longest first (unknown: first of all), so the longest binary is not the one left running alone.
    jobs.sort(key=lambda j: -last.get(j[0], 1e9))
    hot, cool = float(os.environ.get("TEST_HOT", 85)), float(os.environ.get("TEST_COOL", 78))
    running, results, lock = {}, [], threading.Lock()
    paused, done = [0.0], threading.Event()

    def guard():
        while not done.wait(5):
            t = tctl()
            if t is None or t <= hot:
                continue
            t0 = time.time()
            with lock:
                for p in running.values():
                    signal_group(p, signal.SIGSTOP)
            print(f"test_par: Tctl {t:.0f} C > {hot:.0f}: paused", flush=True)
            while not done.is_set() and (tctl() or 0) > cool:
                time.sleep(5)
            with lock:
                for p in running.values():
                    signal_group(p, signal.SIGCONT)
            paused[0] += time.time() - t0

    threading.Thread(target=guard, daemon=True).start()
    queue, start = list(jobs), time.time()

    def run(name, cmd, cwd):
        t0 = time.time()
        with open(logs / f"{name}.log", "w") as log:
            p = subprocess.Popen(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            with lock:
                running[name] = p
            rc = p.wait()
            with lock:
                del running[name]
        text = (logs / f"{name}.log").read_text(errors="replace")
        summary = [l for l in text.splitlines() if l.startswith("test result:")]
        n = sum(int(l.split(" passed")[0].split()[-1]) for l in summary)
        results.append((name, rc, time.time() - t0, n, summary[-1] if summary else "(no result line)"))
        print(f"test_par: {name} rc={rc} {time.time() - t0:.1f} s", flush=True)

    workers = []

    def worker():
        while True:
            with lock:
                if not queue:
                    return
                job = queue.pop(0)
            run(*job)

    for _ in range(min(nbins, len(jobs))):
        t = threading.Thread(target=worker)
        t.start()
        workers.append(t)
    for t in workers:
        t.join()
    done.set()
    wall = time.time() - start
    TIMES.parent.mkdir(parents=True, exist_ok=True)
    last.update({r[0]: r[2] for r in results})
    TIMES.write_text("".join(f"{k}\t{v:.1f}\n" for k, v in sorted(last.items())))
    failed = [r for r in results if r[1] != 0]
    passed = sum(r[3] for r in results)
    print(f"test_par: {len(results)} binaries, {passed} tests passed, {len(failed)} binaries failed, "
          f"wall {wall:.0f} s ({paused[0]:.0f} s paused), {nbins} at once; logs {logs}")
    for name, rc, _, _, line in failed:
        print(f"FAILED {name} rc={rc}: {line}  ({logs / (name + '.log')})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
