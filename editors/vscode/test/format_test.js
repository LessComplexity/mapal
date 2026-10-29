"use strict";

const assert = require("assert").strict;
const fs = require("fs");
const os = require("os");
const path = require("path");
const { createProvider, formatText } = require("../extension");

const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mapal-vscode-format-"));
const binary = path.resolve(process.env.MAPAL_BIN || "target/debug/mapal");
const input = "fn main(){1->println;}\n";
const expected = "fn main() {\n    1 -> println;\n}\n";

function cancellationToken() {
  const callbacks = new Set();
  return {
    isCancellationRequested: false,
    onCancellationRequested(callback) {
      callbacks.add(callback);
      return { dispose: () => callbacks.delete(callback) };
    },
    cancel() {
      this.isCancellationRequested = true;
      for (const callback of [...callbacks]) callback();
    },
    listenerCount: () => callbacks.size,
  };
}

function fixture(name, body) {
  const file = path.join(directory, name);
  fs.writeFileSync(file, `#!${process.execPath}\n`
    + 'require("assert").deepStrictEqual(process.argv.slice(2), ["fmt", "-"]);\n'
    + 'let source = ""; process.stdin.setEncoding("utf8");\n'
    + 'process.stdin.on("data", chunk => source += chunk);\n'
    + `process.stdin.on("end", () => { ${body} });\n`, { mode: 0o755 });
  return file;
}

function document(source) {
  return {
    version: 1,
    isClosed: false,
    uri: { scheme: "untitled" },
    getText: () => source,
    positionAt: offset => ({ offset }),
  };
}

function editor(executable) {
  const errors = [];
  const api = {
    workspace: {
      getConfiguration(section, uri) {
        assert.equal(section, "mapal");
        assert.equal(uri.scheme, "untitled");
        return { get: key => {
          assert.equal(key, "executablePath");
          return executable;
        } };
      },
    },
    window: { showErrorMessage: message => errors.push(message) },
    Range: class { constructor(start, end) { this.start = start; this.end = end; } },
    TextEdit: { replace: (range, newText) => ({ range, newText }) },
  };
  return { provider: createProvider(api), errors };
}

async function main() {
  // Real CLI, unsaved text, and an executable path containing spaces.
  const spaced = path.join(directory, "mapal executable");
  fs.symlinkSync(binary, spaced);
  const success = editor(spaced);
  const edits = await success.provider.provideDocumentFormattingEdits(document(input), {}, cancellationToken());
  assert.equal(edits.length, 1);
  assert.equal(edits[0].newText, expected);
  assert.equal(edits[0].range.end.offset, input.length);
  assert.deepEqual(success.errors, []);
  assert.deepEqual(await success.provider.provideDocumentFormattingEdits(document(expected), {}, cancellationToken()), []);

  const badSource = editor(binary);
  assert.deepEqual(await badSource.provider.provideDocumentFormattingEdits(document("fn main( {"), {}, cancellationToken()), []);
  assert.equal(badSource.errors.length, 1);

  const missing = editor(path.join(directory, "missing"));
  assert.deepEqual(await missing.provider.provideDocumentFormattingEdits(document(input), {}, cancellationToken()), []);
  assert.equal(missing.errors.length, 1);

  const failing = editor(fixture("failed formatter", 'process.stdout.write("partial output"); process.stderr.write("syntax error"); process.exitCode = 1;'));
  assert.deepEqual(await failing.provider.provideDocumentFormattingEdits(document(input), {}, cancellationToken()), []);
  assert.match(failing.errors[0], /syntax error/);

  const delayed = fixture("delayed formatter", `setTimeout(() => process.stdout.write(${JSON.stringify(expected)}), 100);`);
  const stale = editor(delayed);
  const changedDocument = document(input);
  const pending = stale.provider.provideDocumentFormattingEdits(changedDocument, {}, cancellationToken());
  changedDocument.version += 1;
  assert.deepEqual(await pending, []);
  assert.deepEqual(stale.errors, []);

  const closed = document(input);
  const closing = stale.provider.provideDocumentFormattingEdits(closed, {}, cancellationToken());
  closed.isClosed = true;
  assert.deepEqual(await closing, []);

  const token = cancellationToken();
  const cancelled = formatText(delayed, input, token);
  token.cancel();
  assert.equal(await cancelled, undefined);
  assert.equal(token.listenerCount(), 0);
  const alreadyCancelled = cancellationToken();
  alreadyCancelled.cancel();
  assert.equal(await formatText("missing", input, alreadyCancelled), undefined);

  // A path is an executable argument, never shell input.
  const literalPath = fixture("formatter $(touch SHOULD_NOT_EXIST)", "process.stdout.write(source);");
  assert.equal(await formatText(literalPath, input, cancellationToken()), input);
  assert.equal(fs.existsSync(path.join(directory, "SHOULD_NOT_EXIST")), false);
  console.log("VS Code formatting: CLI, unsaved buffer, paths, errors, stale/closed documents and cancellation passed");
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
}).finally(() => fs.rmSync(directory, { recursive: true, force: true }));
