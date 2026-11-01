-- One formatting policy: send the unsaved buffer to `mapal fmt -`.
local M = {}

local function fail(message)
  vim.notify("Mapal formatting: " .. message, vim.log.levels.ERROR)
  return false
end

local function run(source)
  local executable = vim.g.mapal_executable or "mapal"
  if type(executable) ~= "string" or executable == "" then
    return nil, "g:mapal_executable must be an executable name or path"
  end

  local stdout, stderr = {}, {}
  local bytes = 0
  local too_large = false
  local function collect(chunks)
    return function(job, data)
      local chunk = table.concat(data, "\n")
      bytes = bytes + #chunk
      if bytes > 16 * 1024 * 1024 then
        too_large = true
        vim.fn.jobstop(job)
      elseif chunk ~= "" then
        chunks[#chunks + 1] = chunk
      end
    end
  end

  local ok, job = pcall(vim.fn.jobstart, { executable, "fmt", "-" }, {
    on_stdout = collect(stdout),
    on_stderr = collect(stderr),
  })
  if not ok or job <= 0 then
    return nil, "cannot start " .. executable .. "; install the CLI or set g:mapal_executable"
  end

  local sent, message = pcall(function()
    vim.fn.chansend(job, source)
    vim.fn.chanclose(job, "stdin")
  end)
  if not sent then
    vim.fn.jobstop(job)
    return nil, "cannot send buffer to formatter: " .. tostring(message)
  end

  local status = vim.fn.jobwait({ job }, 10000)[1]
  if status < 0 then
    vim.fn.jobstop(job)
    return nil, status == -1 and "formatter timed out after 10 seconds" or "formatting interrupted"
  end
  if too_large then
    return nil, "formatter output exceeded 16 MiB"
  end
  if status ~= 0 then
    local diagnostic = vim.trim(table.concat(stderr))
    return nil, diagnostic ~= "" and diagnostic or ("formatter exited with status " .. status)
  end
  return table.concat(stdout)
end

function M.format()
  local buffer = vim.api.nvim_get_current_buf()
  if not vim.bo[buffer].modifiable then
    return fail("buffer is not modifiable")
  end
  local tick = vim.api.nvim_buf_get_changedtick(buffer)
  local source = table.concat(vim.api.nvim_buf_get_lines(buffer, 0, -1, false), "\n")
  if vim.bo[buffer].endofline then
    source = source .. "\n"
  end
  local view = vim.fn.winsaveview()
  local formatted, err = run(source)
  if not formatted then
    return fail(err)
  end
  if not vim.api.nvim_buf_is_loaded(buffer)
      or vim.api.nvim_get_current_buf() ~= buffer
      or vim.api.nvim_buf_get_changedtick(buffer) ~= tick then
    return fail("buffer changed while formatting; run :MapalFormat again")
  end
  if not vim.bo[buffer].modifiable then
    return fail("buffer is no longer modifiable")
  end
  if source == formatted then
    return true
  end

  local has_eol = formatted:sub(-1) == "\n"
  local body = has_eol and formatted:sub(1, -2) or formatted
  local lines = vim.split(body, "\n", { plain = true })
  vim.api.nvim_buf_set_lines(buffer, 0, -1, false, lines)
  vim.bo[buffer].endofline = has_eol
  vim.fn.winrestview(view)
  return true
end

function M.formatexpr()
  -- Never invoke a whole-document formatter from insert-mode text wrapping.
  if vim.v.char ~= "" or vim.fn.mode():match("^[iR]") then
    return 0
  end
  if vim.v.lnum ~= 1 or vim.v.count ~= vim.api.nvim_buf_line_count(0) then
    vim.notify("Mapal formats whole documents; use :MapalFormat or gggqG", vim.log.levels.INFO)
    return 0
  end
  M.format()
  -- A failure must not fall back to Vim's prose wrapping and change the buffer.
  return 0
end

return M
