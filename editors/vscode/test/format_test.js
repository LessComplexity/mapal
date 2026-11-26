"use strict";

const assert = require("assert").strict;
const fs = require("fs");
const os = require("os");
const path = require("path");
const { bundledPath, createProvider, formatText } = require("../extension");

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

// `executable`: the mapal.executablePath setting (undefined: not set, the default "");
// `extensionPath`: where the extension is installed (a platform package bundles the server).
function editor(executable, extensionPath) {
  const errors = [];
  const api = {
    workspace: {
      getConfiguration(section, uri) {
        assert.equal(section, "mapal");
        assert.equal(uri.scheme, "untitled");
        return { get: (key, fallback) => {
          assert.equal(key, "executablePath");
          return executable === undefined ? fallback : executable;
        } };
      },
    },
    window: { showErrorMessage: message => errors.push(message) },
    Range: class { constructor(start, end) { this.start = start; this.end = end; } },
    TextEdit: { replace: (range, newText) => ({ range, newText }) },
  };
  return { provider: createProvider(api, extensionPath), errors };
}

// A fake CLI that marks its output with where it was found.
function marker(name, mark, mode = 0o755) {
  const file = fixture(name, `process.stdout.write(source + ${JSON.stringify(`// ${mark}\n`)});`);
  fs.chmodSync(file, mode);
  return file;
}

async function formatsWith(executable, extensionPath) {
  const { provider, errors } = editor(executable, extensionPath);
  const edits = await provider.provideDocumentFormattingEdits(document(input), {}, cancellationToken());
  assert.deepEqual(errors, []);
  assert.equal(edits.length, 1);
  return edits[0].newText.slice(input.length);
}

// Formatting runs the executable the server would: the setting, else the bundled
// server/<platform>-<arch>/mapal, else `mapal` on PATH.
async function resolutionOrder() {
  const extension = path.join(directory, "extension");
  const bundled = bundledPath(extension);
  fs.mkdirSync(path.dirname(bundled), { recursive: true });
  fs.renameSync(marker("bundled", "bundled"), bundled);
  const onPath = path.join(directory, "bin");
  fs.mkdirSync(onPath);
  fs.renameSync(marker("on path", "path"), path.join(onPath, "mapal"));
  const setting = marker("setting", "setting");
  const savedPath = process.env.PATH;
  process.env.PATH = onPath + path.delimiter + savedPath;
  try {
    assert.equal(await formatsWith(setting, extension), "// setting\n", "the setting wins over the bundled server");
    assert.equal(await formatsWith(undefined, extension), "// bundled\n", "unset: the bundled server");
    assert.equal(await formatsWith("", extension), "// bundled\n", "empty is unset");
    assert.equal(await formatsWith("  ", extension), "// bundled\n", "blank is unset");
    assert.equal(await formatsWith("mapal", extension), "// path\n", "set to `mapal`: PATH, as asked");
    assert.equal(await formatsWith(undefined, path.join(directory, "no-server")), "// path\n",
      "no bundled server: `mapal` on PATH");
    assert.equal(await formatsWith(undefined, undefined), "// path\n", "no extension path: PATH");
    // An unpacker that drops the executable bit: the extension restores it.
    fs.chmodSync(bundled, 0o644);
    assert.equal(await formatsWith(undefined, extension), "// bundled\n", "a bundled server without +x");
    assert.equal(fs.statSync(bundled).mode & 0o777, 0o755);
  } finally {
    process.env.PATH = savedPath;
  }
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

  await resolutionOrder();
  console.log("VS Code formatting: CLI, unsaved buffer, paths, errors, stale/closed documents, cancellation and "
    + "the executable's resolution (setting > bundled > PATH) passed");
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
}).finally(() => fs.rmSync(directory, { recursive: true, force: true }));
