vim.cmd("set noswapfile shadafile=NONE")
vim.opt.runtimepath:append("editors/nvim")
vim.cmd("filetype plugin on")

local directory = vim.fn.tempname() .. "-mapal-nvim-format"
vim.fn.mkdir(directory, "p")
local binary = vim.fn.fnamemodify(vim.env.MAPAL_BIN or "target/debug/mapal", ":p")
local notices = {}
vim.notify = function(message) notices[#notices + 1] = message end

local function equal(actual, expected, label)
  assert(vim.deep_equal(actual, expected), label .. ": " .. vim.inspect(actual))
end

local function source(text)
  vim.api.nvim_buf_set_lines(0, 0, -1, false, vim.split(text, "\n", { plain = true }))
  vim.bo.endofline = true
end

local function text()
  return table.concat(vim.api.nvim_buf_get_lines(0, 0, -1, false), "\n")
end

local function fixture(name, script)
  local file = directory .. "/" .. name
  vim.fn.writefile({ "#!/bin/sh", '[ "$1" = fmt ] && [ "$2" = - ] || exit 9', script }, file)
  vim.fn.setfperm(file, "rwx------")
  return file
end

local function main()
  local file = directory .. "/unsaved.mapal"
  vim.fn.writefile({ "fn main(){}" }, file)
  vim.cmd("edit " .. vim.fn.fnameescape(file))
  assert(vim.bo.filetype == "mapal", "filetype detection")
  assert(vim.fn.exists(":MapalFormat") == 2, "buffer command registered")
  assert(vim.bo.expandtab and vim.bo.shiftwidth == 4 and vim.bo.tabstop == 4, "four-space settings")
  local format = require("mapal.format")
  local input = "fn main(){1->println;}"
  local expected = "fn main() {\n    1 -> println;\n}"

  local spaced = directory .. "/mapal executable"
  assert((vim.uv or vim.loop).fs_symlink(binary, spaced))
  vim.g.mapal_executable = spaced
  source(input)
  vim.cmd("MapalFormat")
  equal(text(), expected, "unsaved buffer formatting")
  equal(vim.fn.readfile(file), { "fn main(){}" }, "disk remains unchanged")
  local tick = vim.api.nvim_buf_get_changedtick(0)
  assert(format.format(), "idempotent format succeeds")
  equal(vim.api.nvim_buf_get_changedtick(0), tick, "unchanged text creates no edit")

  source(input)
  vim.cmd("normal! gggqG")
  equal(text(), expected, "whole-document gq")

  source("fn main( {")
  assert(not format.format(), "invalid source rejected")
  equal(text(), "fn main( {", "parse failure preserves buffer")
  assert(notices[#notices]:match("Mapal formatting:"), "diagnostics reported")

  vim.g.mapal_executable = directory .. "/missing"
  source(input)
  assert(not format.format(), "missing executable rejected")
  equal(text(), input, "spawn failure preserves buffer")

  vim.g.mapal_executable = fixture("failed formatter", "cat >/dev/null; printf 'partial'; printf 'syntax error' >&2; exit 1")
  assert(not format.format(), "nonzero exit rejected")
  equal(text(), input, "partial stdout discarded")
  assert(notices[#notices]:match("syntax error"), "stderr surfaced")

  vim.g.mapal_executable = fixture("delayed formatter", "sleep 0.1; cat")
  -- jobwait defers scheduled timers; inject the competing edit exactly when
  -- the completed job hands control back, before the adapter applies output.
  local jobwait = vim.fn.jobwait
  vim.fn.jobwait = function(...)
    local result = jobwait(...)
    source("fn edited(){}")
    return result
  end
  local applied = format.format()
  vim.fn.jobwait = jobwait
  assert(not applied, "stale response rejected")
  equal(text(), "fn edited(){}", "concurrent edit retained")

  vim.bo.modifiable = false
  assert(not format.format(), "unmodifiable buffer rejected")
  vim.bo.modifiable = true
  print("Neovim formatting: CLI, unsaved buffer, gq, paths, errors and stale buffer passed")
end

local ok, err = pcall(main)
vim.fn.delete(directory, "rf")
if not ok then
  print(err)
  vim.cmd("cq!")
else
  vim.cmd("qa!")
end
