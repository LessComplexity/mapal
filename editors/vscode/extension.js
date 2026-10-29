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

function activate(context) {
  const vscode = require("vscode");
  context.subscriptions.push(vscode.languages.registerDocumentFormattingEditProvider(
    { language: "mapal" }, createProvider(vscode),
  ));
}

module.exports = { activate, createProvider, formatText };
