#!/usr/bin/env python3
"""The packages package-vsix.py builds.

- The platform package (--target) carries the CLI at extension/server/<target>/mapal as an
  executable entry (mode 0755, Unix attributes), byte for byte, and names its target in the
  manifest; unpacked the way VS Code unpacks (the entry's mode on the file) it formats.
- The platform-neutral package carries no server and names no target.
- A binary for another machine, or one that does not answer `lsp`, is refused.

    (cd editors/vscode && npm install && npm run build)
    python3 editors/vscode/test/package_test.py      (MAPAL_BIN: the CLI, default target/debug/mapal)
"""

import importlib.util
import os
import pathlib
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "editors/vscode/package-vsix.py"
BINARY = (ROOT / os.environ.get("MAPAL_BIN", "target/debug/mapal")).resolve()

spec = importlib.util.spec_from_file_location("package_vsix", SCRIPT)
packager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packager)

fails = 0


def check(ok, what):
    global fails
    fails += not ok
    print(f"{'ok  ' if ok else 'FAIL'}  {what}")


def package(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)


def why(run):
    return f" ({run.stderr.strip()})" if run.returncode else ""


def main():
    target = packager.host_target()
    if target is None:
        print(f"skip: this platform is none of {sorted(packager.TARGETS)}")
        return 0
    if not BINARY.is_file():
        print(f"error: no CLI at {BINARY} (cargo build -p mapal-cli, or set MAPAL_BIN)", file=sys.stderr)
        return 1
    work = pathlib.Path(tempfile.mkdtemp(prefix="mapal-vscode-package-"))
    try:
        platform_vsix = work / "platform.vsix"
        run = package("--target", target, "--binary", BINARY, "--out", platform_vsix)
        check(run.returncode == 0, f"--target {target} packages {BINARY.name}{why(run)}")
        if run.returncode:
            return 1
        entry = packager.server_entry(target)
        with zipfile.ZipFile(platform_vsix) as z:
            info = z.getinfo(entry) if entry in z.namelist() else None
            check(info is not None, f"the server is at {entry}")
            if info is None:
                return 1
            mode = info.external_attr >> 16
            check(stat.S_ISREG(mode) and stat.S_IMODE(mode) == 0o755, f"its entry is mode 0755 ({oct(mode)})")
            check(info.create_system == 3, "with Unix attributes (made on Unix), as unzip reads them")
            check(z.read(entry) == BINARY.read_bytes(), "byte for byte the binary packaged")
            manifest = z.read("extension.vsixmanifest").decode()
            check(f'TargetPlatform="{target}"' in manifest, f"the manifest names {target}")
            others = [n for n in z.namelist() if n.startswith("extension/server/") and n != entry]
            check(not others, "no other server")
            # VS Code's unzip (yauzl) sets the entry's mode on the extracted file.
            unpacked = work / "unpacked"
            z.extractall(unpacked)
            server = unpacked / entry
            server.chmod(stat.S_IMODE(mode))
        source = "fn main(){1->println;}\n"
        fmt = subprocess.run([str(server), "fmt", "-"], input=source, capture_output=True, text=True)
        check(fmt.returncode == 0 and fmt.stdout == "fn main() {\n    1 -> println;\n}\n",
              "the unpacked server runs (fmt -)")

        neutral_vsix = work / "neutral.vsix"
        run = package("--out", neutral_vsix)
        check(run.returncode == 0, f"the platform-neutral package{why(run)}")
        with zipfile.ZipFile(neutral_vsix) as z:
            check(not [n for n in z.namelist() if n.startswith("extension/server/")], "carries no server")
            check("TargetPlatform" not in z.read("extension.vsixmanifest").decode(), "and names no target")

        foreign = next(t for t in packager.TARGETS if packager.TARGETS[t][1] != packager.TARGETS[target][1])
        run = package("--target", foreign, "--binary", BINARY, "--out", work / "foreign.vsix")
        check(run.returncode != 0 and f"not a {foreign} executable" in run.stderr,
              f"a {target} binary is refused for {foreign}")
        true = shutil.which("true")
        if true:
            run = package("--target", target, "--binary", true, "--out", work / "true.vsix")
            check(run.returncode != 0 and "did not answer initialize" in run.stderr,
                  "a binary without `lsp` is refused")
        check(not (work / "foreign.vsix").exists() and not (work / "true.vsix").exists(),
              "a refused binary writes no package")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"{fails} failure(s)" if fails else "all package assertions passed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
