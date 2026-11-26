-- The Vim syntax against the shared fixture: every token of
-- editors/test/fixture.mapal must get the group its `static` class names in
-- editors/test/classes.tsv (editors/test/fixture.tokens.tsv lists them). The
-- TextMate grammar and the language server are held to the same file
-- (editors/vscode/test/scope_test.py, crates/mapal-lsp/tests/highlight.rs).
--
-- Run:  editors/nvim/test/run.sh   (run.sh cd's to the repo root)

vim.cmd("set noswapfile")
local function say(s) io.stdout:write(s .. "\n") end
vim.cmd("set rtp+=editors/nvim")

local function tsv(path)
  local rows = {}
  for line in io.lines(path) do
    if line ~= "" and line:sub(1, 1) ~= "#" then
      rows[#rows + 1] = vim.split(line, "\t", { plain = true })
    end
  end
  return rows
end

local group = {}
for _, r in ipairs(tsv("editors/test/classes.tsv")) do
  if r[1] ~= "class" then group[r[1]] = r[2] end
end

-- Groups that deliberately highlight nothing: the name after a guard arrow.
local plain = { [""] = true, flowGuardValue = true }

vim.cmd("edit! editors/test/fixture.mapal")
vim.cmd("syntax on")

local failures, checked = 0, 0
for _, r in ipairs(tsv("editors/test/fixture.tokens.tsv")) do
  local lnum, col, text, class, static = tonumber(r[1]), tonumber(r[2]), r[3], r[4], r[5]
  local want_class = static == "=" and class or static
  local want = group[want_class]
  assert(want, "class " .. want_class .. " missing from classes.tsv")
  local got = vim.fn.synIDattr(vim.fn.synID(lnum, col, 1), "name")
  local ok = (want == "-" and plain[got]) or got == want
  checked = checked + 1
  if not ok then
    failures = failures + 1
    say(string.format("FAIL  %d:%d %-12s %-14s want %-16s got %s", lnum, col, text, want_class, want,
      got == "" and "«none»" or got))
  end
end
say(string.format("%d tokens checked against fixture.tokens.tsv", checked))

-- The reserved spelling is flagged; no valid example lights up as an error.
vim.cmd("enew")
vim.bo.filetype = "mapal"
vim.api.nvim_buf_set_lines(0, 0, -1, false, { "category Pixel { r: f32 }" })
vim.cmd("syntax on")
if vim.fn.synIDattr(vim.fn.synID(1, 1, 1), "name") ~= "flowReserved" then
  failures = failures + 1
  say("FAIL  `category` is not flagged as reserved")
end
for _, f in ipairs(vim.fn.glob(vim.fn.getcwd() .. "/examples/*.mapal", false, true)) do
  vim.cmd("edit! " .. vim.fn.fnameescape(f))
  vim.cmd("syntax on")
  for l = 1, vim.fn.line("$") do
    for c = 1, #vim.fn.getline(l) do
      if vim.fn.synIDattr(vim.fn.synID(l, c, 1), "name") == "flowReserved" then
        failures = failures + 1
        say(string.format("FAIL  %s:%d:%d flagged as reserved", vim.fn.fnamemodify(f, ":t"), l, c))
      end
    end
  end
end

say(failures == 0 and "all highlighting assertions passed"
  or (failures .. " highlighting assertion(s) failed"))
vim.cmd(failures == 0 and "qa!" or "cq!")
