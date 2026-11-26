"use strict";

const { execFile } = require("child_process");

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

function createProvider(vscode) {
  return {
    async provideDocumentFormattingEdits(document, _options, token) {
      const source = document.getText();
      const version = document.version;
      const executable = vscode.workspace.getConfiguration("mapal", document.uri)
        .get("executablePath", "mapal");
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

// The language server: `<mapal.executablePath> lsp` over stdio.
function serverOptions(executable) {
  return { command: executable, args: ["lsp"], transport: 0 /* TransportKind.stdio */ };
}

function clientOptions() {
  return {
    documentSelector: [
      { scheme: "file", language: "mapal" },
      { scheme: "untitled", language: "mapal" },
    ],
  };
}

let client;
let cliFormatter;

function registerCliFormatter(vscode, context) {
  cliFormatter = vscode.languages.registerDocumentFormattingEditProvider(
    { language: "mapal" }, createProvider(vscode),
  );
  context.subscriptions.push(cliFormatter);
}

async function startServer(vscode, context) {
  const config = vscode.workspace.getConfiguration("mapal");
  const executable = config.get("executablePath", "mapal");
  // Required here, not at the top: the formatter tests load this file without VS Code.
  const { LanguageClient } = require("vscode-languageclient/node");
  const next = new LanguageClient("mapal", "Mapal", serverOptions(executable), clientOptions());
  try {
    await next.start();
  } catch (error) {
    vscode.window.showWarningMessage(
      `Mapal: the language server did not start (${error.message}). Highlighting and formatting `
      + `still work; check that \`${executable} lsp\` runs (setting mapal.executablePath).`,
    );
    return false;
  }
  client = next;
  context.subscriptions.push({ dispose: () => client && client.stop() });
  // One formatting provider at a time: the server's replaces the CLI fallback.
  if (cliFormatter) {
    cliFormatter.dispose();
    cliFormatter = undefined;
  }
  return true;
}

async function activate(context) {
  const vscode = require("vscode");
  context.subscriptions.push(vscode.commands.registerCommand("mapal.restartServer", async () => {
    if (client) await client.restart();
    else if (await startServer(vscode, context)) vscode.window.showInformationMessage("Mapal: language server started");
  }));
  const enabled = vscode.workspace.getConfiguration("mapal").get("server.enabled", true);
  // The server formats too (the same mapal-syntax formatter as `mapal fmt`); the CLI
  // formatter is the fallback, so there is one formatting provider either way.
  if (!enabled || !(await startServer(vscode, context))) registerCliFormatter(vscode, context);
}

function deactivate() {
  return client ? client.stop() : undefined;
}

module.exports = { activate, deactivate, createProvider, formatText, serverOptions, clientOptions };
