# Mapal — VS Code support

The Mapal language server (go to definition, references, hover, rename, completion, diagnostics,
semantic highlighting, formatting), syntax highlighting, the **actual Mapal logo** on `.mapal`
files, and comment and bracket behavior. On linux-x64 it is one install: the platform package
bundles the server.

```
package.json                 language, grammar, semantic token scopes, settings, file icon
extension.js                 finds `mapal`, starts `mapal lsp` (vscode-languageclient); CLI formatter fallback
dist/extension.js            the bundle the .vsix ships (npm run build)
package-vsix.py              builds the .vsix: platform-neutral, or --target with the server bundled
server/<platform>-<arch>/    only inside a platform package: the bundled `mapal`
syntaxes/mapal.tmLanguage.json  highlighting without the server
language-configuration.json  // comments, bracket matching, auto-close
icons/mapal.svg              the mark, square variant
test/                        grammar (Python model + vscode-textmate), server, formatter, packages
```

## Language server

The extension starts `<mapal> lsp` — the compiler's own front end behind the Language Server
Protocol (crate [`mapal-lsp`](../../crates/mapal-lsp/)) — for every Mapal document. Which `mapal`,
in this order:

1. the `mapal.executablePath` setting, if you set it;
2. the server bundled with the extension for this platform (`server/linux-x64/mapal` in the
   linux-x64 package);
3. `mapal` on `PATH`.

The **Mapal** output channel says which one started (`Starting the language server: … lsp
(bundled with the extension (linux-x64))`), and carries the server's own log. Formatting without
the server uses the same `mapal`. Changing the setting restarts the server.

| Feature                          | What it does in Mapal                                                        |
| -------------------------------- | ---------------------------------------------------------------------------- |
| Go to Definition (F12)           | functions, types, fields, top-level values, locals, parameters, size params  |
| Go to Type Definition            | the `type` of a value                                                        |
| Find All References (Shift+F12)  | every use, the definition included; highlights the others under the cursor  |
| Hover                            | signature / type, the doc comment above it, builtin and primitive docs       |
| Rename Symbol (F2)               | all occurrences; refuses keywords and builtin names                          |
| Outline / Go to Symbol           | functions, types (with fields), top-level values                             |
| Completion                       | names in scope, builtins, keywords; `value.` lists the fields                |
| Problems                         | the compiler's parse, lower and check diagnostics, positioned                |
| Format Document                  | the same formatter as `mapal fmt`                                            |
| Semantic highlighting            | calls vs bindings, parameters, fields, top-level values, sizes               |

It keeps working on a file mid-edit. **Mapal: Restart Language Server** restarts it, resolving
`mapal` again (after rebuilding the CLI, say). With `"mapal.server.enabled": false`, or if
`mapal lsp` does not start (an older CLI), the extension falls back to TextMate highlighting and
formatting through `mapal fmt -`.

### One set of colours, two editors

`package.json`'s `semanticTokenScopes` maps each semantic token to the TextMate scope of its class,
and the Neovim plugin links the same tokens to the same classes' syntax groups, so a token is the
same class in VS Code and Neovim, with or without the server. The classes:
[`editors/test/classes.tsv`](../test/classes.tsv); every token of the shared fixture:
[`editors/test/fixture.tokens.tsv`](../test/fixture.tokens.tsv). A theme with its own semantic
token colours still wins, as VS Code intends.

## Why this and not an icon theme

VS Code has no "add one icon" API for file icon themes — a file icon theme is
**all-or-nothing**, so shipping a Mapal-only one would blank the icons for every other file
type in your project. That is why this uses `contributes.languages[].icon` instead (VS Code
≥ 1.66): the icon shows for `.mapal` files, and your existing icon theme keeps handling
everything else. If your active icon theme *does* define an icon for `.mapal`, the theme wins —
that is by design.

## Install

Not published to the Marketplace, so build a `.vsix` and install it.

**linux-x64: one install, extension and server.** The platform package carries the `mapal`
binary (`extension/server/linux-x64/mapal`, mode 0755), and the extension starts it:

```sh
cargo build --release -p mapal-cli
(cd editors/vscode && npm install && npm run build)       # bundles the language client
python3 editors/vscode/package-vsix.py --target linux-x64  # bundles target/release/mapal
code   --install-extension editors/vscode/mapal-lang-0.4.0-linux-x64.vsix --force   # or:
cursor --install-extension editors/vscode/mapal-lang-0.4.0-linux-x64.vsix --force
```

`--binary <path>` bundles another build (`--binary` alone: for the platform it runs on). The
packager refuses a binary for another machine (ELF / Mach-O / PE header) and one that does not
answer an LSP `initialize`. The bundled server is the binary as it was packaged: **rebuild the
`.vsix` after a language change** and reinstall it, or point `mapal.executablePath` at your build
meanwhile. It is dynamically linked against glibc, so it runs where that glibc is at least the
build machine's.

**Other platforms: the platform-neutral package and the CLI.**

```sh
cargo install --path crates/mapal-cli --locked             # `mapal` on PATH
(cd editors/vscode && npm install && npm run build)
python3 editors/vscode/package-vsix.py                     # mapal-lang-0.4.0.vsix, no server
code --install-extension editors/vscode/mapal-lang-0.4.0.vsix --force
```

`--target` also takes `linux-arm64`, `darwin-x64`, `darwin-arm64`, `win32-x64` and `win32-arm64`
with a `--binary` built for it, which makes that platform's package the same way.

Then restart the editor. Verify it actually registered — the message alone is not proof:

```sh
code --list-extensions | grep mapal-lang
```

If the `code` command is not on your PATH, use the binary inside the app:
`"/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code"`.

### Do not just copy the folder in

Dropping this directory (or a symlink to it) into `~/.vscode/extensions` **does not work** on
current VS Code or Cursor. They keep `extensions.json` in that directory as the authoritative
registry and do not scan for unregistered folders, so a hand-placed extension is ignored
**in silence** — no error, no icon, no highlighting, indistinguishable from a broken
extension. Only the CLI install writes the registry entry.

That is also why `package-vsix.py` exists: `vsce` is the normal way to build a `.vsix`, whereas a
`.vsix` is just a ZIP holding `extension.vsixmanifest`, `[Content_Types].xml` and an `extension/`
directory. The script builds one directly and asserts the archive contains all three before
claiming success; a platform package also names its `TargetPlatform` in the manifest and stores
the server's mode in its entry, which VS Code's unzip restores (the extension sets the bit back
if another unpacker drops it). The language client is bundled into `dist/extension.js` by
esbuild, so the `.vsix` carries no `node_modules`; the script refuses a bundle older than
`extension.js`.

### Iterating on the extension

A CLI install copies the files, so editing the repo afterwards changes nothing in the editor.
Either rebuild and reinstall:

```sh
(cd editors/vscode && npm run build) && python3 editors/vscode/package-vsix.py --target linux-x64 \
  && code --install-extension editors/vscode/mapal-lang-0.4.0-linux-x64.vsix --force
```

or launch a throwaway window that loads the source directly, which is better for grammar work:

```sh
code --extensionDevelopmentPath="$(pwd)/editors/vscode" .
```

### Checking it works

1. Open a `.mapal` file — the status bar should read **Mapal**, not *Plain Text*. If it says Plain
   Text the extension is not loaded and nothing else matters.
2. Command palette → `Developer: Inspect Editor Tokens and Scopes`, cursor on a token — should
   show `source.mapal` and a scope such as `support.function.builtin.mapal`.

## Formatting

Use **Format Document** in a Mapal document. The language server formats with the
same `mapal-syntax` formatter as `mapal fmt`, so the layout is the CLI's and Neovim's; a
file that does not parse is left as it is. Without the server the extension formats
unsaved buffer contents through `mapal fmt -` instead: syntax errors and missing
executables produce a notification without editing the document, and cancelled
requests and results for a changed document are discarded.

To select Mapal as the formatter and enable formatting on save, add to settings:

```json
{
  "[mapal]": {
    "editor.defaultFormatter": "mapal-lang.mapal-lang",
    "editor.formatOnSave": true
  },
  "mapal.executablePath": "/path/to/mapal/target/debug/mapal"
}
```

`mapal.executablePath` is empty by default: the bundled server, else `mapal` on PATH (see
[Language server](#language-server)); set, it wins over both. The server is `<path> lsp`, the
fallback formatter `<path> fmt -`. It accepts one executable path, including paths with spaces,
without extra arguments. In a remote VS Code window the CLI must be available where the extension
host runs (the platform package bundles it for that host's platform). The CLI fallback uses
four-space indentation, a ten-second process timeout and a 16 MiB output limit.

## Tests

```sh
cargo build -p mapal-cli # or set MAPAL_BIN to an already-built CLI
(cd editors/vscode && npm install)
editors/test.sh          # both editors
(cd editors/vscode && npm test)   # VS Code: builds the bundle, adds package_test.py
```

- `test/scope_test.py` — the grammar, by a Python model of TextMate's earliest-match rule, gives
  every token of the shared fixture its `static` class's scope; the arrow rules exclude every
  keyword and builtin; `semanticTokenScopes` matches `classes.tsv`.
- `test/grammar_test.js` — the same table in the real engine (vscode-textmate + Oniguruma).
- `test/lsp_test.js` — which `mapal` the extension runs (setting > bundled > PATH, only this
  platform's server); `mapal lsp`, started as bundled in an install, through vscode-jsonrpc (the
  client's protocol stack): the definitions, and every fixture token *painted* (semantic token
  mapped by `semanticTokenScopes` over the TextMate scope) as its class.
- `test/format_test.js` — the CLI formatter fallback: unsaved documents, paths with spaces,
  diagnostics, cancellation, stale document versions, and the same resolution order (a bundled
  server without its executable bit gets it back).
- `test/package_test.py` — `package-vsix.py`: the platform package holds the CLI byte for byte as
  an executable entry (mode 0755) under `extension/server/<target>/`, names its target, and the
  unpacked server runs; the neutral package holds no server; a foreign or non-`lsp` binary is
  refused.

## Syntax highlighting

`syntaxes/mapal.tmLanguage.json` highlights what one line can tell: comments, strings, guard
arrows (chrome as an arrow, discriminant by what it is; the name after one stays plain), `fn` and
`type` declarations with their size lists, a generic type's sizes, keywords, builtins, primitive
types, bindings, call position, arrows, numbers, types, operators, `_`, and the
reserved-and-rejected `category`. Its builtin, primitive and keyword lists are the language's own
(checked by `crates/mapal-lsp/tests/highlight.rs`).

**Two things to know if you edit it.**

The rule that decides outcomes is **earliest match wins**, ties broken by listed order — not
listed order alone. A rule like `(?<=->)\s*(ident)` begins its match at the *whitespace* after
the arrow, one column before the keyword rule that matches the identifier itself, so it wins on
position however the patterns are ordered. That is why `#binding`, `#call-position` and the guard
rules carry a negative lookahead listing every keyword and builtin; `scope_test.py` asserts it.

It has the **opposite precedence** to the Neovim file: TextMate takes the *earliest / first*
matching pattern, Vim the *last*. So this file is ordered most-specific first and
[`../nvim/syntax/mapal.vim`](../nvim/syntax/mapal.vim) least-specific first. Both are held to the
same fixture, so a rule added to one and not the other fails a test.
