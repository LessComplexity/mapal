#!/usr/bin/env python3
"""Package this extension as a .vsix, without vsce.

Why this exists: dropping a folder into ~/.vscode/extensions does NOT work on current
VS Code or Cursor. They keep `extensions.json` in that directory as the authoritative
registry and do not scan for unregistered folders, so a hand-placed directory (or a
symlink to one) is ignored in silence — indistinguishable from a broken extension.

The supported route is `code --install-extension <file>.vsix`, which writes the
registry entry. `vsce` normally builds the .vsix, but it needs npm; a .vsix is just a
ZIP with three things in it, so this builds one directly:

    extension.vsixmanifest   the XML manifest VS Code reads to register it
    [Content_Types].xml      OPC content-type map (required, or install fails)
    extension/...            the extension itself

Two kinds of package:

    package-vsix.py                       platform-neutral: mapal-lang-<version>.vsix, no
                                          server; the extension runs `mapal` from PATH
    package-vsix.py --target linux-x64    platform package: mapal-lang-<version>-linux-x64.vsix
        [--binary target/release/mapal]   with the CLI at extension/server/linux-x64/mapal
                                          (mode 0755): one install, extension + server

`--binary` alone packages for the platform this runs on. The binary is checked against the
target (ELF / Mach-O / PE machine) and, on the platform it is for, answers an LSP
`initialize` before it is packaged. The bundled server is the binary as it was packaged:
rebuild the .vsix after changing the language.

    (cd editors/vscode && npm install && npm run build)   # dist/extension.js: the
    python3 editors/vscode/package-vsix.py --target linux-x64   # client, bundled with esbuild
    code --install-extension editors/vscode/mapal-lang-<version>-linux-x64.vsix --force

The language client (vscode-languageclient) is bundled into dist/extension.js, so the
.vsix carries no node_modules.
"""

import argparse
import json
import pathlib
import platform
import subprocess
import sys
import time
import zipfile

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent

# Everything the extension needs at runtime. Deliberately explicit rather than a glob:
# a .vsix should not carry the packaging script or scratch files.
PAYLOAD = [
    "package.json",
    "dist/extension.js",
    "language-configuration.json",
    "syntaxes/mapal.tmLanguage.json",
    "icons/mapal.svg",
    "README.md",
]

# VS Code target platform -> (the executable's name, its binary format and machine).
# The server directory is the target's name, which is `${process.platform}-${process.arch}`
# for each of these: what extension.js looks for at run time.
TARGETS = {
    "linux-x64": ("mapal", ("elf", 0x3E)),
    "linux-arm64": ("mapal", ("elf", 0xB7)),
    "darwin-x64": ("mapal", ("macho", 0x01000007)),
    "darwin-arm64": ("mapal", ("macho", 0x0100000C)),
    "win32-x64": ("mapal.exe", ("pe", 0x8664)),
    "win32-arm64": ("mapal.exe", ("pe", 0xAA64)),
}

EXECUTABLE = 0o100755  # a regular file, rwxr-xr-x: unzip and VS Code restore it from the entry

VSIXMANIFEST = """<?xml version="1.0" encoding="utf-8"?>
<PackageManifest Version="2.0.0" xmlns="http://schemas.microsoft.com/developer/vsx-schema/2011" xmlns:d="http://schemas.microsoft.com/developer/vsx-schema-design/2011">
  <Metadata>
    <Identity Language="en-US" Id="{name}" Version="{version}" Publisher="{publisher}"{target} />
    <DisplayName>{display}</DisplayName>
    <Description xml:space="preserve">{description}</Description>
    <Tags>mapal,dataflow,compiler,lsp</Tags>
    <Categories>Programming Languages</Categories>
    <GalleryFlags>Public</GalleryFlags>
    <Properties>
      <Property Id="Microsoft.VisualStudio.Code.Engine" Value="{engine}" />
      <Property Id="Microsoft.VisualStudio.Code.ExtensionDependencies" Value="" />
      <Property Id="Microsoft.VisualStudio.Code.ExtensionPack" Value="" />
      <Property Id="Microsoft.VisualStudio.Code.ExtensionKind" Value="ui,workspace" />
    </Properties>
  </Metadata>
  <Installation>
    <InstallationTarget Id="Microsoft.VisualStudio.Code" />
  </Installation>
  <Dependencies/>
  <Assets>
    <Asset Type="Microsoft.VisualStudio.Code.Manifest" Path="extension/package.json" Addressable="true" />
    <Asset Type="Microsoft.VisualStudio.Services.Content.Details" Path="extension/README.md" Addressable="true" />
  </Assets>
</PackageManifest>
"""

CONTENT_TYPES = """<?xml version="1.0" encoding="utf-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension=".json" ContentType="application/json" />
  <Default Extension=".js" ContentType="application/javascript" />
  <Default Extension=".vsixmanifest" ContentType="text/xml" />
  <Default Extension=".xml" ContentType="text/xml" />
  <Default Extension=".svg" ContentType="image/svg+xml" />
  <Default Extension=".md" ContentType="text/markdown" />
  <Default Extension=".exe" ContentType="application/octet-stream" />
</Types>
"""


def host_target():
    """The VS Code target this machine is, or None."""
    system = {"Linux": "linux", "Darwin": "darwin", "Windows": "win32"}.get(platform.system())
    arch = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}.get(platform.machine().lower())
    target = f"{system}-{arch}"
    return target if target in TARGETS else None


def server_entry(target):
    """Where the bundled server sits inside the .vsix."""
    return f"extension/server/{target}/{TARGETS[target][0]}"


def machine(head):
    """(format, machine) of an executable's first bytes; a universal Mach-O fits any darwin."""
    if head[:4] == b"\x7fELF":
        return "elf", int.from_bytes(head[18:20], "little" if head[5] == 1 else "big")
    if head[:4] == b"\xcf\xfa\xed\xfe":
        return "macho", int.from_bytes(head[4:8], "little")
    if head[:4] == b"\xca\xfe\xba\xbe":
        return "macho", "universal"
    if head[:2] == b"MZ":
        pe = int.from_bytes(head[0x3C:0x40], "little")
        if head[pe:pe + 4] == b"PE\0\0":
            return "pe", int.from_bytes(head[pe + 4:pe + 6], "little")
    return None


def frame(message):
    body = json.dumps(message).encode()
    return b"Content-Length: %d\r\n\r\n" % len(body) + body


def answers_lsp(binary):
    """The binary runs `lsp`: initialize answered with capabilities, then a clean exit."""
    session = (frame({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                      "params": {"processId": None, "rootUri": None, "capabilities": {}}})
               + frame({"jsonrpc": "2.0", "method": "initialized", "params": {}})
               + frame({"jsonrpc": "2.0", "id": 2, "method": "shutdown"})
               + frame({"jsonrpc": "2.0", "method": "exit"}))
    try:
        run = subprocess.run([str(binary), "lsp"], input=session, capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"`{binary} lsp` did not run: {error}"
    if run.returncode != 0 or b'"capabilities"' not in run.stdout:
        detail = run.stderr.decode(errors="replace").strip()[:300]
        return f"`{binary} lsp` did not answer initialize (exit {run.returncode}) {detail}".rstrip()
    return None


def check_binary(binary, target):
    if not binary.is_file():
        hint = " (cargo build --release -p mapal-cli)" if binary == default_binary(target) else ""
        return f"no executable at {binary}{hint}"
    with binary.open("rb") as f:
        head = f.read(4096)
    got, want = machine(head), TARGETS[target][1]
    if got != want and not (got == ("macho", "universal") and want[0] == "macho"):
        return f"{binary} is not a {target} executable"
    if target == host_target():
        return answers_lsp(binary)
    return None


def default_binary(target):
    return ROOT / "target" / "release" / TARGETS[target][0]


def parse(argv):
    parser = argparse.ArgumentParser(description="Package the Mapal VS Code extension as a .vsix.")
    parser.add_argument("--target", choices=sorted(TARGETS),
                        help="platform package with the server bundled (default: none, platform-neutral)")
    parser.add_argument("--binary", type=pathlib.Path,
                        help="the mapal CLI to bundle (default: target/release/mapal); "
                             "alone, it packages for this platform")
    parser.add_argument("--out", type=pathlib.Path, help="the .vsix to write (default: next to this script)")
    args = parser.parse_args(argv)
    if args.binary and not args.target:
        args.target = host_target()
        if not args.target:
            parser.error(f"--binary: this platform is none of {sorted(TARGETS)}; pass --target")
    if args.target and not args.binary:
        args.binary = default_binary(args.target)
    return args


def main(argv=None):
    args = parse(argv)
    manifest = json.loads((HERE / "package.json").read_text())
    name, version = manifest["name"], manifest["version"]
    suffix = f"-{args.target}" if args.target else ""
    out = args.out or HERE / f"{name}-{version}{suffix}.vsix"

    missing = [p for p in PAYLOAD if not (HERE / p).exists()]
    if missing:
        hint = " (run `npm install && npm run build` in editors/vscode)" if "dist/extension.js" in missing else ""
        print(f"error: payload files missing: {missing}{hint}", file=sys.stderr)
        return 1
    stale = [p for p in ("extension.js", "package.json")
             if (HERE / p).stat().st_mtime > (HERE / "dist/extension.js").stat().st_mtime]
    if stale:
        print(f"error: dist/extension.js is older than {stale}; run `npm run build`", file=sys.stderr)
        return 1
    if args.target:
        problem = check_binary(args.binary, args.target)
        if problem:
            print(f"error: {problem}", file=sys.stderr)
            return 1

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(
            "extension.vsixmanifest",
            VSIXMANIFEST.format(
                name=name,
                version=version,
                publisher=manifest["publisher"],
                target=f' TargetPlatform="{args.target}"' if args.target else "",
                display=manifest["displayName"],
                description=manifest["description"],
                engine=manifest["engines"]["vscode"],
            ),
        )
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        for rel in PAYLOAD:
            z.write(HERE / rel, f"extension/{rel}")
        if args.target:
            # The mode goes in the entry's Unix attributes (made on Unix), whatever the
            # file's mode on disk: VS Code's unzip sets it on the extracted file.
            entry = zipfile.ZipInfo(server_entry(args.target), time.localtime(args.binary.stat().st_mtime)[:6])
            entry.create_system = 3
            entry.external_attr = EXECUTABLE << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(entry, args.binary.read_bytes())

    # Assert the archive is what an installer expects, rather than trusting it.
    with zipfile.ZipFile(out) as z:
        names = set(z.namelist())
        entrypoint = "extension/" + manifest["main"].removeprefix("./")
        for required in ("extension.vsixmanifest", "[Content_Types].xml", "extension/package.json", entrypoint):
            assert required in names, f"{required} missing from the vsix"
        servers = sorted(n for n in names if n.startswith("extension/server/"))
        if args.target:
            assert servers == [server_entry(args.target)], f"server entries {servers}"
            assert z.getinfo(servers[0]).external_attr >> 16 == EXECUTABLE, "server not executable"
        else:
            assert not servers, f"a platform-neutral package carries {servers}"
        bad = z.testzip()
        assert bad is None, f"corrupt entry: {bad}"
        entries = len(names)

    what = f"{args.target}, server {args.binary}" if args.target else "platform-neutral, `mapal` from PATH"
    print(f"wrote {out} ({out.stat().st_size} bytes, {entries} entries; {what})")
    print("install with:")
    print(f"  code   --install-extension {out} --force")
    print(f"  cursor --install-extension {out} --force")
    return 0


if __name__ == "__main__":
    sys.exit(main())
