#!/usr/bin/env python3
"""site-geometry A4: the test targets for the files you touched — iterate on these; the whole workspace suite runs
once at integrate and once at judge.

usage: tests/targets.py [--base REF] [<file>..]   (default: what differs from REF (HEAD), untracked included)
prints one command per line, closest first:
  - a test file `crates/<p>/tests/<t>.rs` -> `cargo test -p <p> --test <t> [<test fn>]` (the #[test] functions
    whose body changed, when the change is inside them);
  - a shared test module (`tests/emu/`, `mapal-rewrite/tests/testgen/`) -> every test target that includes it;
  - a source file of package P -> P's targets named after the file's module (`splitk.rs` -> tests/splitk.rs), then
    P's unit tests (`--lib`), then P's other targets;
  - benches/gpu tooling -> its unittest file.
CARGO_FLAGS (default `--release`, the differential suite's profile) is put on every cargo line.
"""
import argparse
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLAGS = os.environ.get('CARGO_FLAGS', '--release')


def git(*a):
    return subprocess.run(['git', '-C', str(ROOT), *a], capture_output=True, text=True, check=True).stdout


def package(f):
    """(package name, package dir) of the nearest Cargo.toml with a [package]."""
    for d in (ROOT / f).parents:
        toml = d / 'Cargo.toml'
        if toml.exists():
            m = re.search(r'^\[package\][^\[]*?^name\s*=\s*"([^"]+)"', toml.read_text(), re.M | re.S)
            if m:
                return m[1], d
        if d == ROOT:
            break
    return None, None


def targets(pkg_dir):
    return sorted(p.stem for p in (pkg_dir / 'tests').glob('*.rs'))


def changed_tests(f, base):
    """The #[test] functions of test file `f` whose bodies hold a changed line (empty: the change is elsewhere)."""
    try:
        diff = git('diff', '-U0', base, '--', f)
    except subprocess.CalledProcessError:
        return set()
    lines = (ROOT / f).read_text().splitlines()
    starts = [(i, m[1]) for i, l in enumerate(lines) if (m := re.match(r'\s*(?:pub )?fn (\w+)', l))
              and any(lines[j].strip() == '#[test]' for j in range(max(0, i - 3), i))]
    fns, helper = set(), False
    for m in re.finditer(r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@', diff, re.M):
        lo = int(m[1]) - 1
        hi = lo + max(int(m[2] or 1), 1)
        for ln in range(lo, hi):
            owner = [name for i, name in starts if i <= ln]
            end = next((i for i, _ in starts if i > ln), len(lines))
            # inside a test fn when the nearest test start above has no top-level `fn` between it and the line
            if owner and not any(re.match(r'(?:pub )?fn ', lines[k]) for k in range(
                    max(i for i, _ in starts if i <= ln) + 1, min(ln + 1, end))):
                fns.add(owner[-1])
            else:
                helper = True
    return set() if helper else fns


def plan(files, base):
    cmds, seen = [], set()

    def add(c):
        if c not in seen:
            seen.add(c)
            cmds.append(c)
    all_tests = list(ROOT.glob('crates/**/tests/*.rs'))
    for f in files:
        p = Path(f)
        if f.startswith('benches/gpu/'):
            add('python3 -m unittest benches/gpu/test_harness.py')
            continue
        name, d = package(f)
        if not name:
            continue
        rel = p.relative_to(d.relative_to(ROOT)) if d != ROOT else p
        if rel.parts[0] == 'tests' and len(rel.parts) == 2 and p.suffix == '.rs':
            fns = changed_tests(f, base)
            for fn in sorted(fns) or ['']:
                add(f'cargo test {FLAGS} -p {name} --test {p.stem} {fn}'.rstrip())
        elif rel.parts[0] == 'tests':  # a shared module: every target that includes its directory
            mod = rel.parts[1]
            for t in sorted(all_tests):
                if re.search(rf'\bmod {mod}\b|/{mod}/', t.read_text()):
                    tn, _ = package(str(t.relative_to(ROOT)))
                    add(f'cargo test {FLAGS} -p {tn} --test {t.stem}')
        else:
            stem = p.parent.name if p.stem in ('mod', 'lib', 'main') else p.stem
            ts = targets(d)
            for t in ts:
                if stem in t:
                    add(f'cargo test {FLAGS} -p {name} --test {t}')
            add(f'cargo test {FLAGS} -p {name} --lib')
            for t in ts:
                add(f'cargo test {FLAGS} -p {name} --test {t}')
    return cmds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base', default='HEAD')
    ap.add_argument('files', nargs='*')
    a = ap.parse_args()
    files = a.files or (git('diff', '--name-only', a.base).split()
                        + git('ls-files', '--others', '--exclude-standard').split())
    print('\n'.join(plan(files, a.base)))


if __name__ == '__main__':
    main()
