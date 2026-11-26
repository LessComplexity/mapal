#!/usr/bin/env node
// Render Mermaid diagrams to finished SVG with the exact engine mermaid.live runs (mermaid 12.3.0, default
// config, dagre layout, theme `redux-color`; the dark counterpart is `redux-dark-color`, which is how the
// editor derives it) in headless Chromium. Reads a JSON list [{id, text}] on stdin, writes
// {id: {dark, light, ...}} on stdout. The editor's font (Recursive Variable, from @fontsource-variable/recursive)
// is loaded first so label widths, and so every position, are the editor's.
//   env MERMAID_CHROME  path of chrome-headless-shell (default: Playwright's cache)
import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

export const MERMAID = "12.3.0";
export const FONT = "@fontsource-variable/recursive@5.2.5";
const THEMES = { dark: "redux-dark-color", light: "redux-color" };

function chromePath() {
  if (process.env.MERMAID_CHROME) return process.env.MERMAID_CHROME;
  const base = path.join(os.homedir(), ".cache/ms-playwright");
  const dirs = fs.existsSync(base) ? fs.readdirSync(base).filter((d) => d.startsWith("chromium_headless_shell-")).sort().reverse() : [];
  for (const d of dirs) {
    const p = path.join(base, d, "chrome-headless-shell-linux64/chrome-headless-shell");
    if (fs.existsSync(p)) return p;
  }
  throw new Error("no chrome-headless-shell found (set MERMAID_CHROME)");
}

const input = JSON.parse(fs.readFileSync(0, "utf8"));
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "mmd-"));
const page = path.join(tmp, "render.html");
fs.writeFileSync(page, `<!doctype html><meta charset="utf-8">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/${FONT}/crsv.css">
<body><script>
// the redux look jitters some shapes with Math.random: a seeded generator, reset before each diagram, keeps the SVG reproducible
window.seed = (a) => { Math.random = () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; };
</script>
<script type="module">
import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@${MERMAID}/dist/mermaid.esm.min.mjs";
window.renderAll = async (items, themes) => {
  await document.fonts.load('14px "Recursive Variable"');
  await document.fonts.ready;
  const out = {};
  for (const [k, theme] of Object.entries(themes)) {
    mermaid.initialize({ theme, startOnLoad: false });
    for (const it of items) {
      window.seed(12345);
      const r = await mermaid.render("mmd-" + it.id, it.text);
      (out[it.id] ||= {})[k] = r.svg;
    }
  }
  return out;
};
window.ready = true;
</script>`);

const port = 9700 + Math.floor(Math.random() * 200);
const ch = spawn(chromePath(), [`--remote-debugging-port=${port}`, "--no-sandbox", "about:blank"], { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let list;
for (let i = 0; i < 60; i++) { try { list = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); if (list.length) break; } catch {} await sleep(200); }
const ws = new WebSocket(list.find((t) => t.type === "page").webSocketDebuggerUrl);
await new Promise((r) => (ws.onopen = r));
let id = 0; const pend = new Map();
ws.onmessage = (m) => { const d = JSON.parse(m.data); if (d.id && pend.has(d.id)) { pend.get(d.id)(d); pend.delete(d.id); } };
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pend.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const ev = async (expression) => { const r = await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true }); if (r.result?.exceptionDetails) throw new Error(JSON.stringify(r.result.exceptionDetails).slice(0, 600)); return r.result.result.value; };
let ok = false;
try {
  await send("Page.enable");
  await send("Emulation.setDeviceMetricsOverride", { width: 1600, height: 1000, deviceScaleFactor: 1, mobile: false });
  await send("Page.navigate", { url: "file://" + page });
  for (let i = 0; i < 150 && !ok; i++) { await sleep(200); ok = (await ev("!!window.ready").catch(() => false)) === true; }
  if (!ok) throw new Error("mermaid did not load (network?)");
  const out = await ev(`window.renderAll(${JSON.stringify(input)}, ${JSON.stringify(THEMES)})`);
  process.stdout.write(JSON.stringify(out));
} finally {
  ws.close(); ch.kill(); fs.rmSync(tmp, { recursive: true, force: true });
}
