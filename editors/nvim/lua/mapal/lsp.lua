-- The Mapal language server (`mapal lsp`) for *.mapal buffers: go to definition
-- (gd), references (gr), hover (K), rename, document symbols, completion,
-- diagnostics, formatting and semantic tokens.
--
-- Neovim >= 0.11 registers it with vim.lsp.config / vim.lsp.enable, so any
-- setup that maps LSP keys on LspAttach (LazyVim does) picks it up; older
-- versions start it with vim.lsp.start on FileType.
--
-- Semantic tokens are linked back to the syntax file's groups, class by class
-- (editors/test/classes.tsv), so a name has the same colour with the server as
-- without it, and the same class as in VS Code.
--
-- Snacks' LSP pickers (LazyVim's `gd`, `gy`, `gD`, `gI`) drop every result on
-- the cursor's line (`include_current = false`). In Mapal a definition is often
-- on the line of its use — a map / fold parameter (`acc, k -> acc + a[k]`), a
-- size parameter in a signature (`fn f<rows>(x: [f32; rows])`) — so in mapal
-- buffers those pickers keep them (M.snacks), and `gd` jumps instead of
-- reporting "No results found".
local M = {}

-- `@lsp.<type|typemod>.<token>.mapal` → the class's group in syntax/mapal.vim.
-- editors/nvim/test/lsp_test.lua checks this against classes.tsv.
M.links = {
  ["@lsp.typemod.function.declaration.mapal"] = "flowFnName",
  ["@lsp.type.function.mapal"] = "flowFunction",
  ["@lsp.typemod.function.defaultLibrary.mapal"] = "flowBuiltin",
  ["@lsp.type.type.mapal"] = "flowTypeName",
  ["@lsp.typemod.type.declaration.mapal"] = "flowTypeName",
  ["@lsp.typemod.type.defaultLibrary.mapal"] = "flowPrimType",
  ["@lsp.type.typeParameter.mapal"] = "flowSizeParam",
  ["@lsp.typemod.typeParameter.declaration.mapal"] = "flowSizeParam",
  ["@lsp.typemod.variable.readonly.mapal"] = "flowConstant",
  ["@lsp.typemod.variable.declaration.mapal"] = "flowBinding",
  ["@lsp.typemod.variable.modification.mapal"] = "flowBinding",
  ["@lsp.type.parameter.mapal"] = "flowParameter",
  ["@lsp.typemod.parameter.declaration.mapal"] = "flowParameter",
  ["@lsp.type.property.mapal"] = "flowField",
  ["@lsp.typemod.property.declaration.mapal"] = "flowField",
}

-- The server-only classes, also defined here so the links resolve before the
-- syntax file has loaded (`default`: a colorscheme's own choice wins).
local server_only = {
  flowConstant = "Constant",
  flowParameter = "@variable.parameter",
  flowField = "@property",
  flowFunction = "Function",
}

local function link()
  for group, target in pairs(server_only) do
    vim.api.nvim_set_hl(0, group, { link = target, default = true })
  end
  for group, target in pairs(M.links) do
    vim.api.nvim_set_hl(0, group, { link = target, default = true })
  end
end

-- The Snacks picker sources that go to one place: a result on the cursor's line
-- is the answer, not noise. (References keep Snacks' own setting.)
M.snacks_sources = { "lsp_definitions", "lsp_declarations", "lsp_type_definitions", "lsp_implementations" }

local wrapped = setmetatable({}, { __mode = "k" })

--- Keep same-line results in Snacks' go-to pickers for mapal buffers: each
--- source's `config` hook (chained after any the user set) sets
--- `include_current = true` when the picker opens from a mapal buffer. Returns
--- whether Snacks was there to configure. Idempotent.
function M.snacks()
  local Snacks = rawget(_G, "Snacks")
  if type(Snacks) ~= "table" then
    return false
  end
  local ok, picker = pcall(function()
    return Snacks.config.picker
  end)
  if not ok or type(picker) ~= "table" then
    return false
  end
  picker.sources = picker.sources or {}
  for _, name in ipairs(M.snacks_sources) do
    local source = picker.sources[name] or {}
    picker.sources[name] = source
    if not wrapped[source] then
      wrapped[source] = true
      local user = source.config
      source.config = function(opts)
        opts = (user and user(opts)) or opts
        if vim.bo[vim.api.nvim_get_current_buf()].filetype == "mapal" then
          opts.include_current = true
        end
        return opts
      end
    end
  end
  return true
end

--- The command that starts the server: opts.cmd, g:mapal_lsp_cmd, else
--- `{g:mapal_executable or "mapal", "lsp"}`.
function M.cmd(opts)
  local cmd = (opts and opts.cmd) or vim.g.mapal_lsp_cmd
  if type(cmd) == "string" then
    cmd = { cmd }
  end
  return cmd or { vim.g.mapal_executable or "mapal", "lsp" }
end

--- Register and enable the server for filetype `mapal`. Idempotent.
--- opts.cmd: the command (a list); default `mapal lsp`.
function M.setup(opts)
  local cmd = M.cmd(opts)
  link()
  vim.api.nvim_create_autocmd("ColorScheme", {
    group = vim.api.nvim_create_augroup("MapalSemanticLinks", { clear = true }),
    callback = link,
  })
  -- Snacks may load after this plugin: configure it when the server attaches.
  M.snacks()
  vim.api.nvim_create_autocmd("LspAttach", {
    group = vim.api.nvim_create_augroup("MapalSnacksPickers", { clear = true }),
    callback = function(ev)
      local client = vim.lsp.get_client_by_id(ev.data.client_id)
      if client and client.name == "mapal" then
        M.snacks()
      end
    end,
  })
  if vim.fn.executable(cmd[1]) ~= 1 then
    -- No CLI on PATH: highlighting and the rest of the plugin still work.
    return false
  end
  if vim.lsp.config and vim.lsp.enable then
    vim.lsp.config("mapal", {
      cmd = cmd,
      filetypes = { "mapal" },
      root_markers = { ".git" },
    })
    vim.lsp.enable("mapal")
  else
    vim.api.nvim_create_autocmd("FileType", {
      group = vim.api.nvim_create_augroup("MapalLsp", { clear = true }),
      pattern = "mapal",
      callback = function(ev)
        local root = vim.fs.root and vim.fs.root(ev.buf, { ".git" }) or vim.fn.getcwd()
        vim.lsp.start({ name = "mapal", cmd = cmd, root_dir = root }, { bufnr = ev.buf })
      end,
    })
  end
  return true
end

return M
