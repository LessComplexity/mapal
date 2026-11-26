"use strict";
// The TextMate grammar in the real engine VS Code uses (vscode-textmate +
// vscode-oniguruma), against the shared fixture: every token gets the scope of
// its `static` class (editors/test/fixture.tokens.tsv, classes.tsv).
// scope_test.py checks the same table with a Python model of the engine; this
// one settles any doubt about the model (Oniguruma's regex dialect included).
//
//   npm install && node test/grammar_test.js

const fs = require("fs");
const path = require("path");
const vsctm = require("vscode-textmate");
const onig = require("vscode-oniguruma");

const root = path.resolve(__dirname, "../../..");
const grammarPath = path.join(root, "editors/vscode/syntaxes/mapal.tmLanguage.json");

function tsv(file) {
  return fs.readFileSync(path.join(root, file), "utf8").split("\n")
    .filter((l) => l && !l.startsWith("#")).map((l) => l.split("\t"));
}

async function grammar() {
  const wasm = fs.readFileSync(require.resolve("vscode-oniguruma/release/onig.wasm")).buffer;
  await onig.loadWASM(wasm);
  const registry = new vsctm.Registry({
    onigLib: Promise.resolve({
      createOnigScanner: (patterns) => new onig.OnigScanner(patterns),
      createOnigString: (s) => new onig.OnigString(s),
    }),
    loadGrammar: async (scope) => scope === "source.mapal"
      ? vsctm.parseRawGrammar(fs.readFileSync(grammarPath, "utf8"), grammarPath) : null,
  });
  return registry.loadGrammar("source.mapal");
}

/** The innermost scope at each column of each line, `-` for plain source. */
function scopes(g, lines) {
  let stack = vsctm.INITIAL;
  return lines.map((line) => {
    const r = g.tokenizeLine(line, stack);
    stack = r.ruleStack;
    const at = [];
    for (const t of r.tokens) {
      const s = t.scopes.length > 1 ? t.scopes[t.scopes.length - 1] : "-";
      for (let c = t.startIndex; c < t.endIndex; c++) at[c] = s;
    }
    return at;
  });
}

async function main() {
  const g = await grammar();
  const classes = Object.fromEntries(tsv("editors/test/classes.tsv").filter((r) => r[0] !== "class")
    .map((r) => [r[0], r[2]]));
  const lines = fs.readFileSync(path.join(root, "editors/test/fixture.mapal"), "utf8").split("\n");
  const at = scopes(g, lines);
  let fails = 0;
  let checked = 0;
  for (const [line, col, text, cls, stat] of tsv("editors/test/fixture.tokens.tsv")) {
    const want = classes[stat === "=" ? cls : stat];
    const got = at[Number(line) - 1][Number(col) - 1] || "-";
    checked++;
    if (got !== want) {
      fails++;
      console.log(`FAIL ${line}:${col} ${text.padEnd(12)} want ${want.padEnd(34)} got ${got}`);
    }
  }
  // Lines the fixture cannot hold (it compiles): a builtin or primitive name where a
  // definition goes keeps its class, as the Vim keywords do (scope_test.py STATIC_CASES).
  const cases = [
    ["fn zip(a: i32) -> i32 {", 3, "builtin"],
    ["type u8 {", 5, "primitive"],
    ["    mut f32: i32 <- 0;", 8, "primitive"],
    ["    } -> u8;", 9, "primitive"],
    ["    x -> bf16 -> y;", 9, "primitive"],
    ["        -true-> tf32;", 16, "primitive"],
    ["fn printer() {", 3, "function-declaration"],
  ];
  for (const [line, col, cls] of cases) {
    const got = scopes(g, [line])[0][col] || "-";
    checked++;
    if (got !== classes[cls]) {
      fails++;
      console.log(`FAIL \`${line}\` col ${col + 1}: want ${classes[cls]}, got ${got}`);
    }
  }
  // Every example tokenizes, and nothing valid is flagged as reserved.
  for (const f of fs.readdirSync(path.join(root, "examples")).filter((f) => f.endsWith(".mapal"))) {
    const ls = fs.readFileSync(path.join(root, "examples", f), "utf8").split("\n");
    scopes(g, ls).forEach((row, i) => {
      if (row.includes("invalid.illegal.reserved.mapal")) {
        fails++;
        console.log(`FAIL ${f}:${i + 1} flagged as reserved`);
      }
    });
  }
  console.log(`${checked} tokens checked in vscode-textmate`);
  console.log(fails ? `${fails} failure(s)` : "all grammar assertions passed");
  process.exitCode = fails ? 1 : 0;
}

module.exports = { grammar, scopes, tsv, root };

if (require.main === module) main().catch((e) => { console.error(e); process.exitCode = 1; });
