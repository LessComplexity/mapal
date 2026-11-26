# Mapal — Neovim support

The Mapal language server, syntax highlighting, formatting, filetype detection and the file icon for
`*.mapal`.

```
ftdetect/mapal.vim        *.mapal -> filetype=mapal
ftplugin/mapal.lua        four-space indentation, :MapalFormat and gq
plugin/mapal_lsp.lua      starts the language server for mapal buffers
lua/mapal/lsp.lua         server registration + semantic-token colours
lua/mapal/format.lua      buffer formatting through mapal fmt -
syntax/mapal.vim          highlighting without the server
plugin/mapal_icon.lua     registers the file icon automatically, adds :MapalIcon
lua/mapal/icon.lua        icon module (glyph, color, diagnostics)
test/run.sh               highlighting, the language server and formatting, headless
```

Groups link to standard groups via `hi default link`, so your colorscheme drives the actual
colors and can override any of them.

## Neovim and VS Code side by side

Both editors get the language server (`mapal lsp`: go to definition, references, hover, rename,
completion, diagnostics, semantic highlighting), highlighting, formatting through `mapal fmt`, and a file
icon. Neither is published to a registry, so both load from disk.

|                 | Neovim                                   | VS Code                                          |
| --------------- | ---------------------------------------- | ------------------------------------------------ |
| Language server | started for `*.mapal` (`vim.lsp.enable`) | bundled (linux-x64) and started by the extension |
| Highlighting    | Vimscript syntax + semantic tokens       | TextMate grammar + semantic tokens               |
| File icon       | font glyph                               | the real SVG logo                                |
| Formatting      | `:MapalFormat`, `gggqG` or the server    | **Format Document**                              |
| Details         | this page                                | [`editors/vscode/`](../vscode/README.md)         |

Both highlight with the same classes ([`editors/test/classes.tsv`](../test/classes.tsv)), and the same
fixture tests the Vim syntax, the TextMate grammar and the server's semantic tokens
([below](#one-set-of-colours-two-editors)).

## Language server

`mapal lsp` is the compiler's own front end behind the Language Server Protocol (crate
[`mapal-lsp`](../../crates/mapal-lsp/)): it parses with `mapal-syntax`, reports `mapal-lower` /
`mapal-check` diagnostics, and resolves names by lowering's scoping rules — so `gd` on a name lands
where the compiler says it is defined. It keeps working on a file mid-edit.

| Key (LazyVim)        | Request                    | What it does in Mapal                                                       |
| -------------------- | -------------------------- | --------------------------------------------------------------------------- |
| `gd`                 | definition                 | functions, types, fields, top-level values, locals, parameters, size params |
| `gr`                 | references                 | every use, the definition included                                          |
| `gy`                 | type definition            | the `type` of a value                                                       |
| `K`                  | hover                      | signature / type, the doc comment above it, builtin and type docs           |
| `<leader>cr`         | rename                     | all occurrences; refuses keywords and builtin names                         |
| `<leader>ss`         | document symbols           | functions, types (with fields), top-level values                            |
| (insert)             | completion                 | names in scope, builtins, keywords; `value.` lists the fields               |
| (always)             | diagnostics                | parse, lower and check errors, positioned                                   |
| `<leader>cf`         | formatting                 | the same formatter as `mapal fmt`                                           |
| (always)             | semantic tokens            | calls vs bindings, parameters, fields, top-level values, sizes              |

The plugin registers the server with `vim.lsp.config` / `vim.lsp.enable` on Neovim ≥ 0.11 (any
config that maps LSP keys on `LspAttach` — LazyVim does — picks it up), and with `vim.lsp.start` on
`FileType` before that. It needs the CLI on `PATH` (or `vim.g.mapal_executable`, or the whole
command in `vim.g.mapal_lsp_cmd = { "/path/to/mapal", "lsp" }`); without it the rest of the plugin
works as before. `vim.g.mapal_lsp = false` turns the server off. Check it with `:checkhealth vim.lsp`
or `:lua =vim.lsp.get_clients({ name = "mapal" })`.

LazyVim's `gd` / `gy` / `gD` / `gI` are [Snacks](https://github.com/folke/snacks.nvim) pickers,
which drop every result on the cursor's line (`include_current = false`). In Mapal a definition
often shares its use's line — `acc, k -> acc + a[k]`, `fn f<rows>(x: [f32; rows])` — so the plugin
sets `include_current = true` for those pickers when they open from a mapal buffer
(`require("mapal.lsp").snacks()`; other filetypes keep Snacks' setting). fzf-lua's LazyVim mapping
passes `ignore_current_line=true` itself; with fzf-lua, map `gd` to `vim.lsp.buf.definition` for
mapal buffers.

### One set of colours, two editors

The syntax file highlights what one line can tell; the server knows the rest (whether `-> name;`
calls a function or binds a value, which names are parameters, fields or top-level values). Each
semantic token is linked back to the syntax group of its class (`lua/mapal/lsp.lua`), and VS Code
maps the same tokens to the same classes' TextMate scopes, so a token is the same class in Neovim
and VS Code, with or without the server. The classes and their group / scope / token:
[`editors/test/classes.tsv`](../test/classes.tsv). The class of every token of a shared fixture:
[`editors/test/fixture.tokens.tsv`](../test/fixture.tokens.tsv) — the Vim syntax, the TextMate
grammar and the server are all tested against it.

## Install

**lazy.nvim**

```lua
{ dir = "/path/to/mapal/editors/nvim", lazy = false } -- icon at startup; starts `mapal lsp` for .mapal
```

or, spelled out:

```lua
{
  dir = "/path/to/mapal/editors/nvim",
  lazy = false, -- NOT ft = "mapal": the icon must register at startup, see below
  -- optional, for the real logo instead of a stock glyph:
  -- config = function() local i = require("mapal.icon"); i.setup({ glyph = i.logo }) end,
}
```

**packer**

```lua
use { "/path/to/mapal/editors/nvim" }
```

**No plugin manager** — add the directory to your runtimepath:

```lua
vim.opt.runtimepath:append("/path/to/mapal/editors/nvim")
require("mapal.icon").setup()          -- optional: nvim-web-devicons / mini.icons
```

## Formatting

For the language server and formatting, install the CLI from the repository root with
`cargo install --path crates/mapal-cli --locked` (or copy `target/release/mapal` onto your `PATH`), or point the
plugin at your built `target/release/mapal` with `vim.g.mapal_executable` (in VS Code: `mapal.executablePath`),
then reopen a `.mapal` buffer:

```sh
cargo install --path crates/mapal-cli --locked
```

Use `:MapalFormat` or `gggqG` to format the whole buffer. The plugin sends your
unsaved text to `mapal fmt -`; it does not save the file. Indentation uses four
spaces, matching the CLI. Invalid source or an unavailable CLI leaves the buffer
unchanged and reports an error. `gq` on a partial selection is intentionally a no-op;
format the whole document instead.

If the CLI is not on Neovim's PATH, set its executable path (without arguments):

```lua
vim.g.mapal_executable = "/path/to/mapal/target/debug/mapal"
```

With no plugin manager, also enable `filetype plugin on`. If you use conform.nvim
for your normal formatting key or format-on-save, add this formatter to its setup:

```lua
formatters_by_ft = { mapal = { "mapal" } },
formatters = {
  mapal = {
    command = "mapal", -- or an absolute executable path
    args = { "fmt", "-" },
    stdin = true,
  },
},
```

The built-in adapter waits at most ten seconds and accepts up to 16 MiB of output.
It discards results if the buffer changes during formatting.

## File icon

**Registers itself** — no `setup()` call needed. Works with `mini.icons` (the LazyVim default)
and `nvim-web-devicons`; whichever is present.

**`lazy = false` is required**, not optional. A file tree showing a `.mapal` file has no `.mapal`
buffer open, so an `ft = "mapal"` spec never loads the plugin and the icon never appears.

```vim
:MapalIcon
```

reports what actually registered, which glyph, whether the font is installed, and — if nothing
registered — the likely reason.

### Getting the real logo, not an approximation

The Rust and C++ marks in a file tree are **font glyphs**: Nerd Fonts ships those brand logos as
characters. Providers take a glyph, not an image, so the only way to get Mapal's own mark is to
put it in a font — which [`../../assets/font/`](../../assets/font/) does, one glyph at U+F8F0. So Mapal's mark
ships as a single-glyph font, [`assets/font/MapalIcons.ttf`](../../assets/font/): install it and add it as a
terminal _fallback_ font, rather than us patching and redistributing someone else's Nerd Font. Then:

```lua
local icon = require("mapal.icon")
icon.setup({ glyph = icon.logo })     -- the real mark at U+F8F0; needs MapalIcons.ttf + a terminal fallback mapping
```

Install steps and per-terminal fallback config are in
[`assets/font/README.md`](../../assets/font/README.md). Without it the default is the closest
glyph your Nerd Font already has, so nothing breaks — you just do not get the real mark.

Needs a [Nerd Font](https://www.nerdfonts.com/) either way.

## What gets highlighted

| Class                                     | Group              | Links to              | Server token                 |
| ----------------------------------------- | ------------------ | --------------------- | ---------------------------- |
| `fn type mut loop seq map fold`           | `flowKeyword`      | `Keyword`             |                              |
| `ret`                                     | `flowRet`          | `Special`             |                              |
| builtins (`iota`, `sqrt`, `read`, …)       | `flowBuiltin`      | `Function`            | `function.defaultLibrary`    |
| `i32 f32 bf16 f8e4m3 bool tf32 …`         | `flowPrimType`     | `Type`                | `type.defaultLibrary`        |
| name after `fn`                           | `flowFnName`       | `Function`            | `function.declaration`       |
| a call (`-> f ->`, and `-> f;` with LSP)  | `flowFunction`     | `Function`            | `function`                   |
| `PascalCase` types                        | `flowTypeName`     | `Type`                | `type`                       |
| size parameters `<rows, columns>`         | `flowSizeParam`    | `Constant`            | `typeParameter`              |
| top-level values (with LSP)               | `flowConstant`     | `Constant`            | `variable.readonly`          |
| `-> name;` `mut name` `name <-` bindings  | `flowBinding`      | `Identifier`          | `variable.declaration` / `.modification` |
| parameters (with LSP)                     | `flowParameter`    | `@variable.parameter` | `parameter`                  |
| fields (with LSP)                         | `flowField`        | `@property`           | `property`                   |
| `->` `<-`, guard chrome `-true->`         | `flowArrow` / `flowGuardArrow` | `Statement` — composition *is* the language | |
| size-list brackets `<` `>`                | `flowSizeBracket`  | `Delimiter`           |                              |
| `_` (a type's default interpretation)     | `flowDefaultField` | `Special`             |                              |
| `category`                                | `flowReserved`     | `Error` (ADR-0006)    |                              |

Guard arrows are split-colored on purpose: the **chrome** (leading `-`, trailing `->`) reads as
flow plumbing, while the **discriminant** inside gets the color of what it is — `true`/`false` as
`Boolean`, `42` as `Number`, `_` as `Special`. The name right after a guard arrow heads the arm's
chain, so it stays plain (`-false-> b;` reads `b`).

Without the server, `x -> name;` is lexically identical whether `name` is a new binding or a call
to a function that returns nothing; the syntax file calls it a binding and the server corrects it.

## Tests

```sh
cargo build -p mapal-cli     # or set MAPAL_BIN to an already-built CLI
editors/test.sh              # both editors
editors/nvim/test/run.sh     # this one only
```

- `syntax_test.lua` — every token of the shared fixture gets its class's group (621 tokens), a
  builtin or primitive name where a definition goes keeps its class (as in the TextMate grammar),
  and nothing in `examples/` lights up as an error.
- `lsp_test.lua` — the plugin starts `mapal lsp`; `gd` (the definition request and
  `vim.lsp.buf.definition()`'s jump) lands on a function, a type, a binding, a parameter, a field,
  a top-level value, a size parameter, a struct-literal field and a loop's exit binding; `gr`, `K`,
  rename, symbols and diagnostics answer; every fixture token is *painted* (semantic token over
  syntax) as its class; with snacks.nvim installed (`$SNACKS_NVIM`, else lazy.nvim's checkout),
  `gd` through `Snacks.picker.lsp_definitions()` jumps to a definition on the use's own line.
- `format_test.lua` — the CLI on unsaved buffer text, `gq`, paths with spaces, invalid input,
  missing executables and concurrent edits.
