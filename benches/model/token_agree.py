#!/usr/bin/env python3
"""Token agreement between greedy runs, and the check that an evidence report's token claims match it.

Two measures between runs `a` and `b` of the same prompt length `n`:
  prefix  tokens equal before the first difference (the reports' "tokens = X")
  equal   positions holding the same token over the whole generation

Token files (first batch row): `hf_reference.py` / `run_mapal.py` JSON (`tokens`), or the competitor harness's
`<tool>_n<n>_b<batch>.txt` (whitespace-separated ids).

  table   --run NAME=PATTERN ... --n 128 512 2048 --out T.tsv
          every pair of runs at every length where both files exist; PATTERN has `{n}`, relative to --root.
          The TSV's header records the runs, so `table --refresh T.tsv` rebuilds it from the same raw files.
  check   --table T.tsv DOC.md ...
          every claim marker `<!-- tokens SPEC ... -->` in the docs: the numbers just before it (separated by
          `/` or `|`, bold allowed) must be, in order, the table's values for its SPECs. SPEC is
          `prefix(A,B)@N[,N...]` or `equal(A,B)@N[,N...]` (one value per N). Markers stack: each one reads the
          numbers before it with the markers removed.
"""
import argparse
import json
import re
import sys
from pathlib import Path

MARKER = re.compile(r"<!--\s*tokens\s+(.*?)-->", re.S)
SPEC = re.compile(r"(prefix|equal)\(([\w.-]+),([\w.-]+)\)@([\d,]+)")
INT = re.compile(r"(?<![\w.])\d+(?![\w.])")
SEP = re.compile(r"^[\s*/|]*$")


def load(path):
    text = Path(path).read_text()
    if text.lstrip().startswith("{"):
        tokens = json.loads(text)["tokens"]
        return tokens[0] if tokens and isinstance(tokens[0], list) else tokens
    return [int(t) for t in text.split()]


def prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def equal(a, b):
    return sum(x == y for x, y in zip(a, b))


def build_rows(runs, ns, root):
    """runs: [(name, pattern)]; one row (n, a, b, prefix, equal) per pair with both files present."""
    rows = []
    for n in ns:
        have = []
        for name, pattern in runs:
            p = Path(root) / pattern.format(n=n)
            if p.exists():
                have.append((name, load(p)))
        for i, (a, ta) in enumerate(have):
            for b, tb in have[i + 1:]:
                rows.append((n, a, b, prefix(ta, tb), equal(ta, tb)))
    return rows


def render(runs, ns, rows):
    out = ["# token agreement (benches/model/token_agree.py); greedy, first batch row"]
    out += [f"# run\t{name}\t{pattern}" for name, pattern in runs]
    out.append("# n\t" + "\t".join(map(str, ns)))
    out.append("n\ta\tb\tprefix\tequal")
    out += ["\t".join(map(str, r)) for r in rows]
    return "\n".join(out) + "\n"


def read_table(path):
    runs, ns, values = [], [], {}
    for line in Path(path).read_text().splitlines():
        f = line.split("\t")
        if line.startswith("# run\t"):
            runs.append((f[1], f[2]))
        elif line.startswith("# n\t"):
            ns = [int(x) for x in f[1:]]
        elif line and not line.startswith("#") and f[0] != "n":
            n, a, b, p, e = int(f[0]), f[1], f[2], int(f[3]), int(f[4])
            values[(n, a, b)] = values[(n, b, a)] = {"prefix": p, "equal": e}
    return runs, ns, values


def numbers_before(text):
    """The run of integers that ends `text`, separated only by `/`, `|`, `*` and space."""
    found = list(INT.finditer(text))
    run, end = [], len(text)
    for m in reversed(found):
        if not SEP.match(text[m.end():end]):
            break
        run.append(int(m.group()))
        end = m.start()
    return run[::-1]


def check_text(text, values, where="doc"):
    """[(where, line, wanted, found)] for every marker whose numbers differ; and the number of markers."""
    bad, count = [], 0
    for m in MARKER.finditer(text):
        count += 1
        line = text.count("\n", 0, m.start()) + 1
        specs = []
        for kind, a, b, ns in SPEC.findall(m.group(1)):
            specs += [(kind, a, b, int(n)) for n in ns.split(",") if n]
        wanted = []
        for kind, a, b, n in specs:
            v = values.get((n, a, b))
            wanted.append(v[kind] if v else f"no {a}/{b} at {n}")
        found = numbers_before(MARKER.sub("", text[: m.start()]))[-len(specs):] if specs else []
        if not specs or found != wanted:
            bad.append((where, line, wanted, found))
    return bad, count


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table")
    t.add_argument("--run", action="append", default=[], help="NAME=PATTERN with {n}")
    t.add_argument("--n", type=int, nargs="+", default=[128, 512, 2048])
    t.add_argument("--root", default=".")
    t.add_argument("--out")
    t.add_argument("--refresh", help="rebuild this TSV from the runs its header records")
    c = sub.add_parser("check")
    c.add_argument("--table", required=True)
    c.add_argument("docs", nargs="+")
    args = ap.parse_args()

    if args.cmd == "table":
        if args.refresh:
            runs, ns, _ = read_table(args.refresh)
            out = args.out or args.refresh
        else:
            runs, ns, out = [tuple(r.split("=", 1)) for r in args.run], args.n, args.out
        text = render(runs, ns, build_rows(runs, ns, args.root))
        if out:
            Path(out).write_text(text)
        else:
            sys.stdout.write(text)
        return 0

    _, _, values = read_table(args.table)
    total, failed = 0, []
    for doc in args.docs:
        bad, count = check_text(Path(doc).read_text(), values, doc)
        total += count
        failed += bad
    for where, line, wanted, found in failed:
        print(f"{where}:{line}: table says {wanted}, the text says {found}")
    print(f"{total} token claims checked, {len(failed)} wrong")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
