if vim.b.did_mapal_ftplugin then
  return
end
vim.b.did_mapal_ftplugin = true

vim.bo.expandtab = true
vim.bo.shiftwidth = 4
vim.bo.softtabstop = 4
vim.bo.tabstop = 4
vim.bo.formatexpr = "v:lua.require'mapal.format'.formatexpr()"

vim.api.nvim_buf_create_user_command(0, "MapalFormat", function()
  require("mapal.format").format()
end, { desc = "Format this Mapal buffer with mapal fmt" })

local undo = "setlocal expandtab< shiftwidth< softtabstop< tabstop< formatexpr<"
  .. " | silent! delcommand -buffer MapalFormat | unlet! b:did_mapal_ftplugin"
vim.b.undo_ftplugin = vim.b.undo_ftplugin and (vim.b.undo_ftplugin .. " | " .. undo) or undo
