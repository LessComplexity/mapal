"use strict";

const { execFile } = require("child_process");
const fs = require("fs");
const path = require("path");

// Which `mapal` runs, in this order: the mapal.executablePath setting when the user set it;
// the server a platform package bundles (server/<platform>-<arch>/mapal); `mapal` on PATH.
// The language server is `<executable> lsp`, the formatter fallback `<executable> fmt -`.
function bundledPath(extensionPath, platform = process.platform, arch = process.arch) {
  return path.join(extensionPath, "server", `${platform}-${arch}`, platform === "win32" ? "mapal.exe" : "mapal");
}

// A bundled binary is used when it runs; one whose executable bit an unpacker dropped gets it back.
function runnable(file) {
  try {
    if (!fs.statSync(file).isFile()) return false;
    fs.accessSync(file, fs.constants.X_OK);
    return true;
  } catch (error) {
    if (error.code !== "EACCES") return false;
  }
  try {
    fs.chmodSync(file, 0o755);
    fs.accessSync(file, fs.constants.X_OK);
    return true;
  } catch {
    return false;
  }
}

function resolveExecutable(config, extensionPath, host = {}) {
  const { platform = process.platform, arch = process.arch, canRun = runnable } = host;
  const setting = config.get("executablePath", "");
  if (typeof setting === "string" && setting.trim() !== "") {
    return { command: setting, source: "setting", detail: "the mapal.executablePath setting" };
  }
  if (extensionPath) {
    const bundled = bundledPath(extensionPath, platform, arch);
    if (canRun(bundled)) {
      return { command: bundled, source: "bundled", detail: `bundled with the extension (${platform}-${arch})` };
    }
  }
  return { command: "mapal", source: "path", detail: "`mapal` on PATH" };
}

// The CLI owns layout. These bounds also apply when a custom executable is used.
function formatText(executable, source, token) {
  if (token.isCancellationRequested) return Promise.resolve(undefined);
  return new Promise((resolve, reject) => {
    let finished = false;
    let cancellation;
    const finish = (error, output) => {
      if (finished) return;
      finished = true;
      if (cancellation) cancellation.dispose();
      if (error) reject(error);
      else resolve(output);
    };
    let child;
    try {
      child = execFile(executable, ["fmt", "-"], {
        encoding: "utf8",
        timeout: 10000,
        maxBuffer: 16 * 1024 * 1024,
        killSignal: "SIGKILL",
        windowsHide: true,
      }, (error, stdout, stderr) => {
        if (token.isCancellationRequested) return finish(null, undefined);
        if (error) {
          const message = stderr.trim() || error.message;
          return finish(new Error(message.slice(0, 2000)));
        }
        finish(null, stdout);
      });
    } catch (error) {
      finish(error);
      return;
    }

    const cancel = () => {
      child.kill("SIGKILL");
      finish(null, undefined);
    };
    cancellation = token.onCancellationRequested(cancel);
    if (finished) cancellation.dispose();
    // Cancellation can arrive between the initial check and listener registration.
    if (token.isCancellationRequested) cancel();
    child.stdin.on("error", (error) => {
      // EPIPE accompanies an early CLI diagnostic; the exit callback reports it.
      if (error.code !== "EPIPE") {
        child.kill("SIGKILL");
        finish(error);
      }
    });
    child.stdin.end(source);
  });
}

function createProvider(vscode, extensionPath) {
  return {
    async provideDocumentFormattingEdits(document, _options, token) {
      const source = document.getText();
      const version = document.version;
      const executable = resolveExecutable(
        vscode.workspace.getConfiguration("mapal", document.uri), extensionPath).command;
      try {
        const formatted = await formatText(executable, source, token);
        if (formatted === undefined || token.isCancellationRequested
            || document.isClosed || document.version !== version || formatted === source) {
          return [];
        }
        const range = new vscode.Range(document.positionAt(0), document.positionAt(source.length));
        return [vscode.TextEdit.replace(range, formatted)];
      } catch (error) {
        if (!token.isCancellationRequested) {
          vscode.window.showErrorMessage(`Mapal formatting: ${error.message}`);
        }
        return [];
      }
    },
  };
}

// The language server: `<executable> lsp` over stdio.
function serverOptions(executable) {
  return { command: executable, args: ["lsp"], transport: 0 /* TransportKind.stdio */ };
}

function clientOptions(outputChannel) {
  const options = {
    documentSelector: [
      { scheme: "file", language: "mapal" },
      { scheme: "untitled", language: "mapal" },
    ],
  };
  if (outputChannel) options.outputChannel = outputChannel;
  return options;
}

let client;
let cliFormatter;
let output;

function registerCliFormatter(vscode, context) {
  const cli = resolveExecutable(vscode.workspace.getConfiguration("mapal"), context.extensionPath);
  output.appendLine(`Formatting through \`${cli.command} fmt -\` (${cli.detail}).`);
  cliFormatter = vscode.languages.registerDocumentFormattingEditProvider(
    { language: "mapal" }, createProvider(vscode, context.extensionPath),
  );
  context.subscriptions.push(cliFormatter);
}

async function startServer(vscode, context) {
  const server = resolveExecutable(vscode.workspace.getConfiguration("mapal"), context.extensionPath);
  output.appendLine(`Starting the language server: ${server.command} lsp (${server.detail}).`);
  // Required here, not at the top: the formatter tests load this file without VS Code.
  const { LanguageClient } = require("vscode-languageclient/node");
  const next = new LanguageClient("mapal", "Mapal", serverOptions(server.command), clientOptions(output));
  try {
    await next.start();
  } catch (error) {
    output.appendLine(`The language server did not start: ${error.message}`);
    vscode.window.showWarningMessage(
      `Mapal: the language server did not start (${error.message}). Highlighting and formatting `
      + `still work; check that \`${server.command} lsp\` runs (${server.detail}; setting mapal.executablePath).`,
    );
    return false;
  }
  output.appendLine(`Language server running: ${server.command} lsp (${server.detail}).`);
  client = next;
  // One formatting provider at a time: the server's replaces the CLI fallback.
  if (cliFormatter) {
    cliFormatter.dispose();
    cliFormatter = undefined;
  }
  return true;
}

// Stop the running server and start one from the executable the settings resolve to now;
// one restart at a time, so a command and a settings change never start two servers.
let restarting = Promise.resolve();

function restartServer(vscode, context) {
  restarting = restarting.catch(() => false).then(async () => {
    if (client) {
      const running = client;
      client = undefined;
      try {
        await running.stop();
      } catch (error) {
        output.appendLine(`Stopping the language server: ${error.message}`);
      }
    }
    const started = await startServer(vscode, context);
    if (!started && !cliFormatter) registerCliFormatter(vscode, context);
    return started;
  });
  return restarting;
}

async function activate(context) {
  const vscode = require("vscode");
  output = vscode.window.createOutputChannel("Mapal");
  context.subscriptions.push(output, { dispose: () => client && client.stop() });
  context.subscriptions.push(vscode.commands.registerCommand("mapal.restartServer", async () => {
    if (await restartServer(vscode, context)) vscode.window.showInformationMessage("Mapal: language server started");
  }));
  // A changed executable takes effect at once for a running server; the CLI formatter
  // resolves it on every request.
  context.subscriptions.push(vscode.workspace.onDidChangeConfiguration((event) => {
    if (client && event.affectsConfiguration("mapal.executablePath")) restartServer(vscode, context);
  }));
  const enabled = vscode.workspace.getConfiguration("mapal").get("server.enabled", true);
  // The server formats too (the same mapal-syntax formatter as `mapal fmt`); the CLI
  // formatter is the fallback, so there is one formatting provider either way.
  if (!enabled || !(await startServer(vscode, context))) registerCliFormatter(vscode, context);
}

function deactivate() {
  return client ? client.stop() : undefined;
}

module.exports = {
  activate, deactivate, createProvider, formatText, serverOptions, clientOptions, resolveExecutable, bundledPath,
};
