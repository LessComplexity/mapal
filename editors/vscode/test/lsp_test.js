"use strict";
// The language server through the protocol stack VS Code's client uses
// (vscode-jsonrpc, as vscode-languageclient drives it): `mapal lsp` starts,
// answers definition requests, and — the check that matters for "the same
// highlights as Neovim" — every token of the shared fixture ends up with its
// class's scope once the semantic tokens are mapped through package.json's
// semanticTokenScopes over the TextMate scopes (what VS Code paints). The server is
// started the way the extension finds it: the setting, else the server a platform
// package bundles (here: MAPAL_BIN placed as one), else `mapal` on PATH.
//
//   cargo build -p mapal-cli && npm install && node test/lsp_test.js   (MAPAL_BIN: the CLI)

const assert = require("assert").strict;
const cp = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");
const rpc = require("vscode-jsonrpc/node");
const { grammar, scopes, tsv, root } = require("./grammar_test");
const { serverOptions, clientOptions, resolveExecutable, bundledPath } = require("../extension");

const binary = path.resolve(root, process.env.MAPAL_BIN || "target/debug/mapal");

// The mapal section of the settings with mapal.executablePath at `value` (undefined: unset).
const settings = (value) => ({
  get: (key, fallback) => {
    assert.equal(key, "executablePath");
    return value === undefined ? fallback : value;
  },
});

// Which executable the extension runs: the setting > the bundled server > `mapal` on PATH.
function resolutionOrder() {
  const ext = path.join(path.sep, "ext");
  const linux = { platform: "linux", arch: "x64" };
  const bundled = path.join(ext, "server", "linux-x64", "mapal");
  assert.equal(bundledPath(ext, "linux", "x64"), bundled);
  assert.equal(bundledPath(ext, "darwin", "arm64"), path.join(ext, "server", "darwin-arm64", "mapal"));
  assert.equal(bundledPath(ext, "win32", "x64"), path.join(ext, "server", "win32-x64", "mapal.exe"));
  const asked = [];
  const present = { ...linux, canRun: (file) => (asked.push(file), true) };
  const absent = { ...linux, canRun: () => false };
  const pick = (value, host, at = ext) => {
    const { command, source } = resolveExecutable(settings(value), at, host); // at null: no install path
    return [command, source];
  };
  assert.deepEqual(pick("/opt/mapal", present), ["/opt/mapal", "setting"], "the setting wins");
  assert.deepEqual(asked, [], "a set path is not second-guessed by the bundled server");
  assert.deepEqual(pick("mapal", present), ["mapal", "setting"], "set to `mapal`: PATH, as asked");
  assert.deepEqual(pick(undefined, present), [bundled, "bundled"], "unset: the bundled server");
  assert.deepEqual(asked, [bundled], "only this platform's server is looked for");
  assert.deepEqual(pick("", present), [bundled, "bundled"], "empty (the default) is unset");
  assert.deepEqual(pick(" ", present), [bundled, "bundled"], "blank is unset");
  assert.deepEqual(pick(undefined, absent), ["mapal", "path"], "no bundled server: PATH");
  assert.deepEqual(pick(undefined, present, null), ["mapal", "path"], "no extension path: PATH");
  const other = { platform: "linux", arch: "arm64", canRun: (file) => file === bundled };
  assert.deepEqual(pick(undefined, other), ["mapal", "path"], "another platform's server is not used");
}

async function main() {
  // The extension starts exactly `<executable> lsp` for mapal documents.
  assert.deepEqual(serverOptions("mapal"), { command: "mapal", args: ["lsp"], transport: 0 });
  assert.ok(clientOptions().documentSelector.every((s) => s.language === "mapal"));
  const channel = { name: "Mapal" };
  assert.equal(clientOptions(channel).outputChannel, channel, "server logs go to the Mapal output channel");
  resolutionOrder();

  // The server the extension resolves from an install that bundles one, the setting unset.
  const install = fs.mkdtempSync(path.join(os.tmpdir(), "mapal-vscode-lsp-"));
  fs.mkdirSync(path.dirname(bundledPath(install)), { recursive: true });
  fs.symlinkSync(binary, bundledPath(install));
  const server = resolveExecutable(settings(undefined), install);
  assert.deepEqual([server.command, server.source], [bundledPath(install), "bundled"]);

  const proc = cp.spawn(server.command, serverOptions(server.command).args, { stdio: ["pipe", "pipe", "inherit"] });
  const conn = rpc.createMessageConnection(
    new rpc.StreamMessageReader(proc.stdout), new rpc.StreamMessageWriter(proc.stdin));
  const diagnostics = new Map();
  conn.onNotification("textDocument/publishDiagnostics", (p) => diagnostics.set(p.uri, p.diagnostics));
  conn.listen();

  const init = await conn.sendRequest("initialize", { processId: process.pid, rootUri: null, capabilities: {} });
  conn.sendNotification("initialized", {});
  const legend = init.capabilities.semanticTokensProvider.legend;

  const open = (file) => {
    const uri = "file://" + path.join(root, file);
    const text = fs.readFileSync(path.join(root, file), "utf8");
    conn.sendNotification("textDocument/didOpen", { textDocument: { uri, languageId: "mapal", version: 1, text } });
    return { uri, text, lines: text.split("\n") };
  };
  const at = (doc, needle, into, nth = 1) => {
    let seen = 0;
    for (let l = 0; l < doc.lines.length; l++) {
      let from = 0;
      for (;;) {
        const i = doc.lines[l].indexOf(needle, from);
        if (i < 0) break;
        if (++seen === nth) return { line: l, character: i + into };
        from = i + 1;
      }
    }
    throw new Error(`not found: ${needle}`);
  };
  let fails = 0;
  const check = (ok, what) => { if (!ok) fails++; console.log(`${ok ? "ok  " : "FAIL"}  ${what}`); };

  // Go to definition (F12 / gd).
  const llama = open("examples/llama.mapal");
  const fixture = open("editors/test/fixture.mapal");
  const cases = [
    [llama, "a function call", "-> decoder_layer -> result", 3, "fn decoder_layer"],
    [llama, "a type use", "cache: KeyValueCache)", 7, "type KeyValueCache"],
    [llama, "a field", "cache.keys[key_row", 6, "    keys: [f32; cache_size],"],
    [fixture, "a size parameter", "-> [f32; rows]", 9, "fn scale<rows"],
    [fixture, "a top-level value", "t -> t * width", 9, "64 -> width;"],
  ];
  for (const [doc, what, needle, into, target] of cases) {
    const r = await conn.sendRequest("textDocument/definition", {
      textDocument: { uri: doc.uri }, position: at(doc, needle, into) });
    check(r && r.range.start.line === at(doc, target, 0).line, `definition of ${what}`);
  }

  // What VS Code paints: semantic token (mapped by semanticTokenScopes) over TextMate.
  const pkg = JSON.parse(fs.readFileSync(path.join(root, "editors/vscode/package.json"), "utf8"));
  const map = pkg.contributes.semanticTokenScopes.find((s) => s.language === "mapal").scopes;
  const sem = await conn.sendRequest("textDocument/semanticTokens/full", { textDocument: { uri: fixture.uri } });
  const painted = new Map();
  let line = 0;
  let col = 0;
  for (let i = 0; i < sem.data.length; i += 5) {
    const [dl, dc, , type, mods] = sem.data.slice(i, i + 5);
    col = dl ? dc : col + dc;
    line += dl;
    const mod = legend.tokenModifiers.filter((_, k) => mods & (1 << k));
    const key = [legend.tokenTypes[type], ...mod].join(".");
    painted.set(`${line + 1}:${col + 1}`, map[key] ? map[key][0] : `«unmapped ${key}»`);
  }
  const tm = scopes(await grammar(), fixture.lines);
  const classes = Object.fromEntries(tsv("editors/test/classes.tsv").filter((r) => r[0] !== "class")
    .map((r) => [r[0], r[2]]));
  let bad = 0;
  const rows = tsv("editors/test/fixture.tokens.tsv");
  for (const [l, c, text, cls] of rows) {
    const got = painted.get(`${l}:${c}`) || tm[Number(l) - 1][Number(c) - 1] || "-";
    if (got !== classes[cls]) {
      bad++;
      console.log(`FAIL  ${l}:${c} ${text} (${cls}): painted ${got}, want ${classes[cls]}`);
    }
  }
  check(bad === 0, `all ${rows.length} fixture tokens painted as their class`);

  // Diagnostics arrive for an open document; llama.mapal has none.
  await new Promise((r) => setTimeout(r, 500));
  check(diagnostics.has(llama.uri) && diagnostics.get(llama.uri).length === 0, "diagnostics published (none for llama)");

  await conn.sendRequest("shutdown");
  conn.sendNotification("exit");
  await new Promise((r) => proc.on("exit", r));
  conn.dispose();
  fs.rmSync(install, { recursive: true, force: true });
  console.log(fails ? `${fails} failure(s)` : "all lsp assertions passed");
  process.exitCode = fails ? 1 : 0;
}

main().catch((e) => { console.error(e); process.exitCode = 1; });
