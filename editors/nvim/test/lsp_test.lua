-- The language server in a headless Neovim, the way an LSP-keyed config uses
-- it: the plugin starts `mapal lsp` for *.mapal buffers, then `gd`'s request
-- (textDocument/definition, and vim.lsp.buf.definition's jump) lands on the
-- definition of a function call, a type use, a binding use, a parameter, a
-- field, a top-level value and a size parameter; references, hover, rename,
-- diagnostics and semantic tokens answer too.
--
-- Run:  editors/nvim/test/run.sh   (MAPAL_BIN: the CLI, default target/debug/mapal)

vim.cmd("set noswapfile shadafile=NONE")
vim.opt.runtimepath:append("editors/nvim")
vim.cmd("filetype plugin on")
vim.cmd("syntax on")
vim.g.mapal_lsp_cmd = { vim.fn.fnamemodify(vim.env.MAPAL_BIN or "target/debug/mapal", ":p"), "lsp" }
vim.cmd("runtime! plugin/mapal_lsp.lua")

local failures = 0
local function say(s) io.stdout:write(s .. "\n") end
local function check(ok, what)
  if ok then
    say("ok    " .. what)
  else
    failures = failures + 1
    say("FAIL  " .. what)
  end
end

local function client_for(buf)
  vim.wait(10000, function()
    return #vim.lsp.get_clients({ bufnr = buf, name = "mapal" }) > 0
  end, 20)
  return vim.lsp.get_clients({ bufnr = buf, name = "mapal" })[1]
end

-- The 1-based line and 0-based byte column of the `nth` `needle`, plus `into`.
local function find(buf, needle, nth, into)
  local seen = 0
  for l, text in ipairs(vim.api.nvim_buf_get_lines(buf, 0, -1, false)) do
    local from = 1
    while true do
      local s = text:find(needle, from, true)
      if not s then break end
      seen = seen + 1
      if seen == (nth or 1) then
        return l, s - 1 + (into or 0)
      end
      from = s + 1
    end
  end
  error("not found: " .. needle)
end

local function params_at(buf, line, col)
  local text = vim.api.nvim_buf_get_lines(buf, line - 1, line, false)[1]
  return {
    textDocument = { uri = vim.uri_from_bufnr(buf) },
    position = { line = line - 1, character = vim.str_utfindex(text, "utf-16", col) },
  }
end

local function request(client, buf, method, params)
  local r = client:request_sync(method, params, 10000, buf)
  return r and r.result
end

-- `gd`: the definition request at `needle`, and the jump the default handler makes.
local function definition(client, buf, what, needle, into, target, target_nth)
  local l, c = find(buf, needle, 1, into)
  local want = find(buf, target, target_nth)
  local r = request(client, buf, "textDocument/definition", params_at(buf, l, c))
  local loc = r and (r.range and r or r[1])
  local got = loc and (loc.range or loc.targetSelectionRange).start.line + 1
  check(got == want, string.format("gd on %s (%s) -> line %s, want %d", what, needle, tostring(got), want))
  vim.api.nvim_win_set_cursor(0, { l, c })
  vim.lsp.buf.definition()
  vim.wait(5000, function()
    return vim.api.nvim_win_get_cursor(0)[1] == want
  end, 10)
  check(vim.api.nvim_win_get_cursor(0)[1] == want, "  vim.lsp.buf.definition() jumps there")
end

-- llama.mapal -----------------------------------------------------------------
vim.cmd("edit examples/llama.mapal")
local buf = vim.api.nvim_get_current_buf()
check(vim.bo[buf].filetype == "mapal", "filetype mapal")
local client = client_for(buf)
check(client ~= nil, "mapal lsp attached")
if not client then
  vim.cmd("cq!")
end

definition(client, buf, "a function call", "-> decoder_layer -> result", 3, "fn decoder_layer")
definition(client, buf, "a type use", "cache: KeyValueCache<element>,", 7, "type KeyValueCache")
definition(client, buf, "a binding use", "(weights.query, normed)", 16, "-> rms_norm -> normed;")
-- `positions` of decoder_layer (the second function with that parameter).
definition(client, buf, "a parameter", "(query, positions) -> rotate", 8, "    positions: [i32; tokens],", 2)
definition(client, buf, "a field", "cache.keys[key_row", 6, "    keys: [element; cache_size],")
definition(client, buf, "a top-level value", "hidden_size -> iota -> features", 0, "2048 -> hidden_size;")
definition(client, buf, "a rebind of a mut binding", "result.0 -> hidden;", 12, "mut hidden: [f32")

local l, c = find(buf, "fn rms_norm", 1, 3)
local p = params_at(buf, l, c)
p.context = { includeDeclaration = true }
local refs = request(client, buf, "textDocument/references", p)
check(refs and #refs == 4, "gr on rms_norm: its declaration and three calls (" .. (refs and #refs or 0) .. ")")

l, c = find(buf, "-> project -> gate", 1, 3)
local hover = request(client, buf, "textDocument/hover", params_at(buf, l, c))
local text = hover and hover.contents and hover.contents.value or ""
check(text:find("fn project<tokens, rows, columns>", 1, true) ~= nil, "K on project shows its signature")

l, c = find(buf, "mut layer: i32", 1, 4)
p = params_at(buf, l, c)
p.newName = "depth"
local edit = request(client, buf, "textDocument/rename", p)
local changes = edit and edit.changes and edit.changes[vim.uri_from_bufnr(buf)]
check(changes and #changes == 6, "rename of `mut layer`: 6 edits (" .. (changes and #changes or 0) .. ")")

local symbols = request(client, buf, "textDocument/documentSymbol", { textDocument = { uri = vim.uri_from_bufnr(buf) } })
check(symbols and #symbols > 30, "document symbols")

-- Semantic tokens: a terminal call (static syntax: a binding) is a function.
-- The highlighter paints the visible lines on redraw (a decoration provider):
-- bring the line into view and redraw while waiting.
do
  local lv, cv = find(buf, "-> most_likely_token -> first_token", 1, 3)
  vim.api.nvim_win_set_cursor(0, { lv, cv })
  vim.cmd("normal! zz")
end
vim.wait(5000, function()
  vim.cmd("redraw!")
  local lt, ct = find(buf, "-> most_likely_token -> first_token", 1, 3)
  for _, t in ipairs(vim.inspect_pos(buf, lt - 1, ct).semantic_tokens) do
    if t.opts.hl_group == "@lsp.type.function.mapal" then return true end
  end
  return false
end, 20)
local lt, ct = find(buf, "-> most_likely_token -> first_token", 1, 3)
local groups = {}
for _, t in ipairs(vim.inspect_pos(buf, lt - 1, ct).semantic_tokens) do
  groups[#groups + 1] = t.opts.hl_group
end
check(vim.tbl_contains(groups, "@lsp.type.function.mapal"),
  "semantic token on a call: " .. table.concat(groups, ","))
check(vim.fn.hlID("@lsp.type.function.mapal") ~= 0
  and vim.api.nvim_get_hl(0, { name = "@lsp.type.function.mapal" }).link == "flowFunction",
  "semantic function tokens link to flowFunction")

-- The links are classes.tsv's.
local want = {}
for line in io.lines("editors/test/classes.tsv") do
  local f = vim.split(line, "\t", { plain = true })
  if line:sub(1, 1) ~= "#" and f[1] ~= "class" and f[4] ~= "-" then
    local ty, mod = f[4]:match("^([^.]+)%.?(.*)$")
    local group = mod == "" and ("@lsp.type." .. ty .. ".mapal") or ("@lsp.typemod." .. ty .. "." .. mod .. ".mapal")
    want[group] = f[2]
  end
end
check(vim.deep_equal(want, require("mapal.lsp").links), "semantic links match editors/test/classes.tsv")

-- The shared fixture: size-generic functions, types, top-level values ------------
vim.cmd("edit editors/test/fixture.mapal")
buf = vim.api.nvim_get_current_buf()
client = client_for(buf)
definition(client, buf, "a generic type use", "(m: Matrix<rows", 4, "type Matrix")
definition(client, buf, "a size parameter", "-> [f32; rows]", 9, "fn scale<rows")
definition(client, buf, "a top-level value", "t -> t * width", 9, "64 -> width;")
definition(client, buf, "a field", "lit.r -> waves", 4, "    r: f32,")
definition(client, buf, "a struct literal field", "r: p.r + half", 0, "    r: f32,")
definition(client, buf, "a loop exit binding", "finished.0 -> seq", 0, "-> (total, counts) -> finished;")

-- What Neovim paints on every fixture token: the semantic token's group (the
-- highest-priority `@lsp…mapal` group the plugin links), else the syntax group.
-- It must be the token's class's group (classes.tsv) — the same class VS Code
-- paints (editors/vscode/test/lsp_test.js checks that side).
do
  vim.o.lines = 200
  vim.api.nvim_win_set_cursor(0, { 1, 0 })
  local links = require("mapal.lsp").links
  local function semantic_group(row, col)
    local best, prio = nil, -1
    for _, t in ipairs(vim.inspect_pos(buf, row - 1, col - 1).semantic_tokens) do
      local g = links[t.opts.hl_group]
      if g and t.opts.priority > prio then best, prio = g, t.opts.priority end
    end
    return best
  end
  vim.wait(5000, function()
    vim.cmd("redraw!")
    return semantic_group(5, 7) ~= nil
  end, 20)
  local vim_group = {}
  for line in io.lines("editors/test/classes.tsv") do
    local f = vim.split(line, "\t", { plain = true })
    if line:sub(1, 1) ~= "#" and f[1] ~= "class" then vim_group[f[1]] = f[2] end
  end
  local bad, n = 0, 0
  for line in io.lines("editors/test/fixture.tokens.tsv") do
    if line:sub(1, 1) ~= "#" then
      local f = vim.split(line, "\t", { plain = true })
      local row, col, class = tonumber(f[1]), tonumber(f[2]), f[4]
      local got = semantic_group(row, col) or vim.fn.synIDattr(vim.fn.synID(row, col, 1), "name")
      local want = vim_group[class]
      n = n + 1
      if not ((want == "-" and (got == "" or got == "flowGuardValue")) or got == want) then
        bad = bad + 1
        say(string.format("FAIL  %d:%d %s (%s): painted %s, want %s", row, col, f[3], class, got, want))
      end
    end
  end
  check(bad == 0, "all " .. n .. " fixture tokens painted as their class")
end

-- `gd` through Snacks' picker, as LazyVim maps it. The picker drops results on
-- the cursor's line unless the plugin keeps them for mapal buffers
-- (require("mapal.lsp").snacks); here a definition shares its use's line.
-- Needs snacks.nvim: $SNACKS_NVIM, else lazy.nvim's checkout; skipped without it.
do
  local dir = vim.env.SNACKS_NVIM or (vim.fn.stdpath("data") .. "/lazy/snacks.nvim")
  if vim.fn.isdirectory(dir .. "/lua/snacks") == 0 then
    say("skip  gd through Snacks' picker (no snacks.nvim at " .. dir .. ")")
  else
    vim.opt.runtimepath:append(dir)
    require("snacks").setup({ picker = { enabled = true } })
    local same_line = vim.fn.tempname() .. ".mapal"
    vim.fn.writefile({
      "fn total<rows>(x: [f32; rows]) -> f32 {",
      "    (0.0, x) -> fold {",
      "        acc, v -> acc + v",
      "    } -> ret;",
      "}",
      "",
      "fn main() {",
      "    4 -> iota -> map {",
      "        i -> i * 2",
      "    } -> twice;",
      "    twice[1] -> println;",
      "}",
    }, same_line)
    vim.cmd("edit " .. vim.fn.fnameescape(same_line))
    buf = vim.api.nvim_get_current_buf()
    client_for(buf)
    vim.keymap.set("n", "gd", function() Snacks.picker.lsp_definitions() end, { buffer = buf })
    local function snacks_gd(what, needle, into, target, target_into)
      local l, c = find(buf, needle, 1, into)
      local wl, wc = find(buf, target, 1, target_into)
      vim.api.nvim_win_set_cursor(0, { l, c })
      vim.api.nvim_feedkeys("gd", "mx", false)
      vim.wait(5000, function()
        local p = vim.api.nvim_win_get_cursor(0)
        return p[1] == wl and p[2] == wc
      end, 10)
      local p = vim.api.nvim_win_get_cursor(0)
      check(vim.api.nvim_get_current_buf() == buf and p[1] == wl and p[2] == wc,
        string.format("Snacks gd on %s -> %d:%d, want %d:%d", what, p[1], p[2], wl, wc))
      vim.api.nvim_set_current_buf(buf)
    end
    snacks_gd("a fold parameter on its line", "acc + v", 0, "acc, v", 0)
    snacks_gd("a size parameter in its signature", "; rows]", 2, "<rows>", 1)
    snacks_gd("a map parameter on its line", "i * 2", 0, "i -> i", 0)
    snacks_gd("a binding on another line", "twice[1]", 0, "-> twice;", 3)
  end
end

-- Diagnostics: the compiler's, positioned.
vim.cmd("enew")
local scratch = vim.fn.tempname() .. ".mapal"
vim.fn.writefile({ "fn main() {", "    1 -> x;", "    y -> println;", "}" }, scratch)
vim.cmd("edit " .. vim.fn.fnameescape(scratch))
buf = vim.api.nvim_get_current_buf()
client_for(buf)
vim.wait(10000, function() return #vim.diagnostic.get(buf) > 0 end, 20)
local ds = vim.diagnostic.get(buf)
check(#ds == 1 and ds[1].lnum == 2 and ds[1].col == 4, "diagnostic on the unknown name `y` (" .. #ds .. ")")

say(failures == 0 and "all lsp assertions passed" or (failures .. " lsp assertion(s) failed"))
vim.cmd(failures == 0 and "qa!" or "cq!")
