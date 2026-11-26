-- Start the Mapal language server for *.mapal buffers (lua/mapal/lsp.lua).
-- Opt out with `vim.g.mapal_lsp = false` before the plugin loads; point it at
-- a CLI off PATH with `vim.g.mapal_executable` (or the whole command with
-- `vim.g.mapal_lsp_cmd = { "/path/to/mapal", "lsp" }`).
if vim.g.loaded_mapal_lsp then
  return
end
vim.g.loaded_mapal_lsp = 1
if vim.g.mapal_lsp == false then
  return
end
require("mapal.lsp").setup()
