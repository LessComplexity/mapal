"""When is one implementation really faster? Shared by the CPU harnesses' summaries.

For two sets of timings (one per process, interleaved), report:
  - the ratio of medians (competitor ÷ Mapal; > 1 = Mapal faster) and its 95 % percentile-bootstrap
    interval (10,000 resamples, each side resampled independently, fixed seed);
  - the two-sided p-value of the Mann–Whitney U test (normal approximation, tie-corrected);
  - a verdict: "faster" or "slower" only when the interval excludes 1 AND p < 0.05, else
    "no clear difference".
Standard library only (no SciPy on the bench machines).
"""
import math
import random
import statistics

ALPHA = 0.05
REPS = 10_000


def median(v):
    return statistics.median(v)


def ratio_ci(num, den, reps=REPS, level=0.95, seed=20261128):
    """median(num) / median(den) and its percentile-bootstrap interval."""
    rng = random.Random(seed)
    r = median(num) / median(den)
    rs = sorted(median(rng.choices(num, k=len(num))) / median(rng.choices(den, k=len(den))) for _ in range(reps))
    return r, rs[int((1 - level) / 2 * reps)], rs[int((1 + level) / 2 * reps) - 1]


def mann_whitney_p(a, b):
    """Two-sided p-value of the Mann–Whitney U test (normal approximation with tie correction)."""
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return 1.0
    pooled = sorted([(x, 0) for x in a] + [(x, 1) for x in b])
    ranks, i = [0.0] * len(pooled), 0
    while i < len(pooled):
        j = i
        while j + 1 < len(pooled) and pooled[j + 1][0] == pooled[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        i = j + 1
    u = sum(r for r, (_, g) in zip(ranks, pooled) if g == 0) - n1 * (n1 + 1) / 2
    n = n1 + n2
    counts = {}
    for x, _ in pooled:
        counts[x] = counts.get(x, 0) + 1
    ties = sum(t ** 3 - t for t in counts.values())
    var = n1 * n2 / 12 * ((n + 1) - ties / (n * (n - 1)))
    if var <= 0:
        return 1.0
    z = (u - n1 * n2 / 2) / math.sqrt(var)
    return math.erfc(abs(z) / math.sqrt(2))


def compare(competitor, mapal):
    """The ratio competitor ÷ Mapal (medians), its interval, the p-value and the verdict."""
    r, lo, hi = ratio_ci(competitor, mapal)
    p = mann_whitney_p(competitor, mapal)
    if lo > 1 and p < ALPHA:
        v = "faster"
    elif hi < 1 and p < ALPHA:
        v = "slower"
    else:
        v = "no clear difference"
    return {"ratio": r, "lo": lo, "hi": hi, "p": p, "verdict": v, "n": [len(competitor), len(mapal)]}


def fmt(c):
    """`1.27× [1.18, 1.36] p=0.001 faster` (Mapal's verdict)."""
    p = "p<0.001" if c["p"] < 0.001 else f"p={c['p']:.3f}"
    return f"{c['ratio']:.2f}× [{c['lo']:.2f}, {c['hi']:.2f}] {p} {c['verdict']}"
