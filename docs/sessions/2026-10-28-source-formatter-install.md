# 2026-10-28 — local formatter installation

## Continuation

Follow-up to `2026-10-28-source-formatter.md`: the user requested local installation.
`mapal` is now installed at `/Users/lesscomplex/.cargo/bin/mapal` and resolves on
PATH. No implementation work remains. Inspect with `command -v mapal` and
`mapal fmt --help`.

## Decision and verification

Used `CARGO_BUILD_JOBS=4 cargo install --path crates/mapal-cli --debug --locked --offline`
to reuse the tested local build without downloads or a release rebuild. Installation
succeeded. Piping `fn main(){1->println;}` into `mapal fmt -` produced the expected
multiline, four-space layout. No repository source files were formatted.

## Live state and open items

The local executable is installed; no task process remains. The editor plugins can
use their default `mapal` executable setting when Cargo's bin directory is on the
editor PATH. The VS Code extension still needs the previously built VSIX installed
if the user has the older highlighting-only extension. No code, architecture or
OpenSpec changes in this follow-up; prior drift result was 0 dead / 701 references.

## Docs reconciled

This immutable follow-up supersedes only the previous log's local CLI installation
state. Its implementation/test evidence and other live-state entries remain valid.
