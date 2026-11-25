#!/usr/bin/env python3
"""Time examples/llama.mapal the way the competitor harness times every tool (model-compare/common.py).

One process per repetition (a Mapal program is compiled ahead of time; the weights load and upload before the
clock starts). The program prints a line per generated token: id, logit, ms since the prompt was submitted
(`() -> time` before the prompt is read, and after each token is on the host). From those:
  ttft_ms    t[0]                                  submit -> the first token on the host
  decode_ms  (t[G-1] - t[1]) / (G-2)               steady-state ms per decode step, the first excluded
  e2e_ms     t[G-1]
The first `--warmup` processes are discarded (they fill the GPU's tuning wisdom and the page cache). Peak GPU
memory: the process's used memory sampled from nvidia-smi at ~20 Hz (the competitors' `peak_proc_mb`).

Usage:
  python run_mapal.py --exe build/llama_n512 --n 512 --out mapal_n512_b1.json [--warmup 2 --reps 5]
The executable's PTX is <exe>.ptx; MAPAL_DATA_DIR defaults to ~/.cache/mapal/models/llama-3.2-1b-instruct/mapal.
"""
import argparse
import json
import os
import subprocess
import threading
import time


class Peak(threading.Thread):
    """The largest `used_memory` nvidia-smi reports for any process of `pids()`."""

    def __init__(self):
        super().__init__(daemon=True)
        self.pid = None
        self.peak_mb = 0
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set():
            if self.pid is not None:
                out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
                for line in out.splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) == 2 and parts[0] == str(self.pid) and parts[1].isdigit():
                        self.peak_mb = max(self.peak_mb, int(parts[1]))
            time.sleep(0.05)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--data", default=os.path.expanduser("~/.cache/mapal/models/llama-3.2-1b-instruct/mapal"))
    ap.add_argument("--cpus", default=None, help="taskset CPU list for the timed process")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert args.batch == 1, "the program decodes one sequence"

    env = dict(os.environ, MAPAL_PTX=args.exe + ".ptx", MAPAL_DATA_DIR=args.data)
    cmd = [os.path.abspath(args.exe)]
    if args.cpus:
        cmd = ["taskset", "-c", args.cpus] + cmd
    peak = Peak()
    peak.start()
    reps, tokens, walls = [], None, []
    for k in range(args.warmup + args.reps):
        start = time.perf_counter()
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        peak.pid = p.pid
        out, err = p.communicate()
        walls.append(time.perf_counter() - start)
        assert p.returncode == 0, f"exit {p.returncode}: {err[-2000:]}"
        rows = [line.split() for line in out.splitlines() if line.strip()]
        assert len(rows) == args.gen, f"{len(rows)} tokens, expected {args.gen}"
        t = [float(r[2]) for r in rows]
        ids = [int(r[0]) for r in rows]
        if k >= args.warmup:
            g = args.gen
            reps.append({"ttft_ms": t[0], "decode_ms": (t[g - 1] - t[1]) / (g - 2), "e2e_ms": t[g - 1],
                         "top_logits": [float(r[1]) for r in rows]})
            assert tokens is None or tokens == [ids], "greedy tokens differ between repetitions"
            tokens = [ids]
    peak.stop.set()
    result = {"tool": "mapal", "version": "examples/llama.mapal", "batch": 1, "gen": args.gen, "n": args.n,
              "reps": reps, "tokens": tokens, "extra": {"peak_proc_mb": peak.peak_mb, "wall_s": walls}}
    json.dump(result, open(args.out, "w"))
    med = lambda xs: sorted(xs)[len(xs) // 2]
    print(f"[mapal] n={args.n} b=1: ttft {med([r['ttft_ms'] for r in reps]):.2f} ms, decode "
          f"{med([r['decode_ms'] for r in reps]):.3f} ms/step, peak {peak.peak_mb} MB", flush=True)


if __name__ == "__main__":
    main()
