//! The server end to end over an in-memory connection: the requests an
//! editor sends (`gd`, `gr`, `K`, rename, completion, formatting, semantic
//! tokens) and the diagnostics it publishes.

use std::path::Path;
use std::thread::JoinHandle;
use std::time::Duration;

use lsp_server::{Connection, Message, Notification, Request, RequestId};
use mapal_lsp::text::LineMap;
use serde_json::{Value, json};

struct Client {
    conn: Connection,
    next: i32,
    server: Option<JoinHandle<()>>,
    notes: Vec<Notification>,
}

impl Client {
    fn start() -> Client {
        let (server, client) = Connection::memory();
        let handle = std::thread::spawn(move || mapal_lsp::serve(server).expect("server ran"));
        let mut c = Client {
            conn: client,
            next: 1,
            server: Some(handle),
            notes: Vec::new(),
        };
        let caps = c.request("initialize", json!({ "capabilities": {} }));
        assert!(
            caps["capabilities"]["definitionProvider"]
                .as_bool()
                .unwrap_or(false)
        );
        c.notify("initialized", json!({}));
        c
    }

    fn notify(&self, method: &str, params: Value) {
        self.conn
            .sender
            .send(Message::Notification(Notification::new(
                method.into(),
                params,
            )))
            .unwrap();
    }

    fn request(&mut self, method: &str, params: Value) -> Value {
        let id = RequestId::from(self.next);
        self.next += 1;
        self.conn
            .sender
            .send(Message::Request(Request::new(
                id.clone(),
                method.into(),
                params,
            )))
            .unwrap();
        loop {
            match self
                .conn
                .receiver
                .recv_timeout(Duration::from_secs(30))
                .expect("a reply")
            {
                Message::Response(r) if r.id == id => {
                    if let Some(e) = r.error {
                        return json!({ "error": e.message });
                    }
                    return r.result.unwrap_or(Value::Null);
                }
                Message::Notification(n) => self.notes.push(n),
                _ => {}
            }
        }
    }

    /// The next diagnostics published for `uri`.
    fn diagnostics(&mut self, uri: &str) -> Vec<Value> {
        loop {
            if let Some(i) = self.notes.iter().position(|n| {
                n.method == "textDocument/publishDiagnostics" && n.params["uri"] == uri
            }) {
                let n = self.notes.remove(i);
                return n.params["diagnostics"]
                    .as_array()
                    .cloned()
                    .unwrap_or_default();
            }
            if let Message::Notification(n) = self
                .conn
                .receiver
                .recv_timeout(Duration::from_secs(30))
                .expect("diagnostics")
            {
                self.notes.push(n);
            }
        }
    }

    fn open(&self, uri: &str, text: &str) {
        self.notify(
            "textDocument/didOpen",
            json!({ "textDocument": { "uri": uri, "languageId": "mapal", "version": 1, "text": text } }),
        );
    }

    fn stop(mut self) {
        self.request("shutdown", Value::Null);
        self.notify("exit", Value::Null);
        self.server.take().unwrap().join().unwrap();
    }
}

fn llama() -> String {
    std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/llama.mapal"),
    )
    .unwrap()
}

/// The LSP position of the `nth` (0-based) `needle` in `text`, `into` bytes in.
fn pos(text: &str, needle: &str, nth: usize, into: usize) -> Value {
    let at = text
        .match_indices(needle)
        .nth(nth)
        .unwrap_or_else(|| panic!("`{needle}` #{nth}"))
        .0
        + into;
    let p = LineMap::new(text).position(text, at as u32);
    json!({ "line": p.line, "character": p.character })
}

/// The 1-based line of the `nth` occurrence of `needle`.
fn line_of(text: &str, needle: &str, nth: usize) -> u64 {
    let at = text.match_indices(needle).nth(nth).unwrap().0;
    text[..at].matches('\n').count() as u64 + 1
}

const URI: &str = "file:///tmp/llama.mapal";

fn at(text: &str, needle: &str, nth: usize, into: usize) -> Value {
    json!({ "textDocument": { "uri": URI }, "position": pos(text, needle, nth, into) })
}

#[test]
fn definition_lands_on_each_kind_of_definition() {
    let text = llama();
    let mut c = Client::start();
    c.open(URI, &text);
    // (where the cursor is, the definition's line, what it is)
    let cases = [
        (
            ("-> decoder_layer -> result", 0, 3),
            line_of(&text, "fn decoder_layer", 0),
            "a function call",
        ),
        (
            ("cache: KeyValueCache<element>,", 0, 7),
            line_of(&text, "type KeyValueCache", 0),
            "a type use",
        ),
        (
            ("keys: [element; cache_size]", 0, 7),
            line_of(&text, "type KeyValueCache<element: float>", 0),
            "a format parameter in a type position (ADR-0051 D1)",
        ),
        (
            ("-> to<element>;", 0, 6),
            line_of(&text, "fn write_rows<tokens, element: float>", 0),
            "a format parameter as a stage's explicit argument (ADR-0051 D2)",
        ),
        (
            ("(weights.query, normed)", 0, 16),
            line_of(&text, "-> rms_norm -> normed;", 0),
            "a binding use",
        ),
        (
            ("(query, positions) -> rotate", 0, 8),
            line_of(&text, "    positions: [i32; tokens],\n) -> ([f32", 0),
            "a parameter",
        ),
        (
            ("cache.keys[key_row", 0, 6),
            line_of(&text, "    keys: [element; cache_size],", 0),
            "a field",
        ),
        (
            ("hidden_size -> iota -> features", 0, 0),
            line_of(&text, "2048 -> hidden_size;", 0),
            "a top-level value",
        ),
        (
            ("tokens * rows -> iota", 0, 0),
            line_of(&text, "fn project<tokens", 0),
            "a size parameter",
        ),
        (
            ("result.0 -> hidden;", 0, 12),
            line_of(&text, "mut hidden: [f32", 0),
            "a rebind of a mut binding",
        ),
        (
            ("model.layers[layer]", 0, 6),
            line_of(&text, "    layers: [DecoderLayerWeights", 0),
            "a field through a parameter",
        ),
        (
            ("prefilled.1;", 0, 0),
            line_of(&text, "-> forward -> prefilled;", 0),
            "a binding read in a statement",
        ),
    ];
    for ((needle, nth, into), want, what) in cases {
        let r = c.request("textDocument/definition", at(&text, needle, nth, into));
        let got = r["range"]["start"]["line"].as_u64().map(|l| l + 1);
        assert_eq!(got, Some(want), "{what}: `{needle}` went to {r}");
    }
    // A builtin has no definition in the file.
    let r = c.request(
        "textDocument/definition",
        at(&text, "-> iota -> features", 0, 3),
    );
    assert!(r.is_null(), "iota: {r}");
    c.stop();
}

#[test]
fn references_highlights_and_type_definition() {
    let text = llama();
    let mut c = Client::start();
    c.open(URI, &text);
    let mut p = at(&text, "fn rms_norm", 0, 3);
    p["context"] = json!({ "includeDeclaration": true });
    let refs = c.request("textDocument/references", p.clone());
    assert_eq!(
        refs.as_array().unwrap().len(),
        1 + text.matches("-> rms_norm ->").count()
    );
    p["context"] = json!({ "includeDeclaration": false });
    let refs = c.request("textDocument/references", p);
    assert_eq!(
        refs.as_array().unwrap().len(),
        text.matches("-> rms_norm ->").count()
    );

    let hl = c.request(
        "textDocument/documentHighlight",
        at(&text, "mut layer: i32", 0, 4),
    );
    let kinds: Vec<u64> = hl
        .as_array()
        .unwrap()
        .iter()
        .map(|h| h["kind"].as_u64().unwrap())
        .collect();
    assert!(
        kinds.contains(&3) && kinds.contains(&2),
        "write and read highlights: {kinds:?}"
    );

    // `gy` on a value of a declared type.
    let r = c.request(
        "textDocument/typeDefinition",
        at(&text, "(hidden, weights.attention_norm)", 0, 10),
    );
    assert_eq!(
        r["range"]["start"]["line"].as_u64().unwrap() + 1,
        line_of(&text, "type DecoderLayerWeights", 0)
    );
    c.stop();
}

#[test]
fn hover_shows_signatures_types_and_docs() {
    let text = llama();
    let mut c = Client::start();
    c.open(URI, &text);
    let h = |c: &mut Client, needle: &str, into: usize| {
        c.request("textDocument/hover", at(&text, needle, 0, into))["contents"]["value"]
            .as_str()
            .unwrap_or("")
            .to_string()
    };
    let project = h(&mut c, "-> project -> gate", 3);
    assert!(
        project.contains("fn project<tokens, rows, columns>(weight: Matrix<rows, columns>, x: [f32; tokens * columns]) -> [f32; tokens * rows]"),
        "{project}"
    );
    assert!(project.contains("A linear layer without bias"), "{project}");
    let result = h(&mut c, "result.0", 0);
    assert!(
        result.contains("result: ([f32; tokens * hidden_size], KeyValueCache<element>)"),
        "{result}"
    );
    let field = h(&mut c, "cache.keys[key_row", 6);
    assert!(
        field.contains("keys: [element; cache_size]") && field.contains("field of `KeyValueCache`"),
        "{field}"
    );
    let global = h(&mut c, "log_rope_theta;", 0);
    assert!(global.contains("ln(500000)"), "trailing comment: {global}");
    let sqrt = h(&mut c, "-> sqrt -> root", 3);
    assert!(sqrt.contains("builtin `sqrt`"), "{sqrt}");
    // ADR-0051: the generic conversions and a format parameter.
    let widen = h(&mut c, "-> widen<f32> -> key;", 3);
    assert!(
        widen.contains("builtin `widen`") && widen.contains("exactly"),
        "{widen}"
    );
    let to = h(&mut c, "-> to<element>;", 3);
    assert!(
        to.contains("builtin `to`") && to.contains("rounded"),
        "{to}"
    );
    let element = h(&mut c, "type KeyValueCache<element: float>", 19);
    assert!(
        element.contains("element: float")
            && element.contains("format parameter of `type KeyValueCache`, kind `float`"),
        "{element}"
    );
    let bf16 = h(&mut c, "[bf16; hidden_size]", 1);
    assert!(bf16.contains("primitive type `bf16`"), "{bf16}");
    c.stop();
}

/// Through the server: a value made by a function declared further down is
/// typed (hover, type definition, `value.` completion), and a `seq`'s binding
/// is defined after the block (definition, hover).
#[test]
fn later_declarations_and_seq_bindings_through_the_server() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/programs");
    let later = std::fs::read_to_string(dir.join("declared_later.mapal")).unwrap();
    let seq = std::fs::read_to_string(dir.join("seq_scope.mapal")).unwrap();
    let mut c = Client::start();
    c.open(URI, &later);
    let hover = c.request("textDocument/hover", at(&later, "made.bias", 0, 0));
    let hover = hover["contents"]["value"].as_str().unwrap_or("");
    assert!(hover.contains("made: Config"), "{hover}");
    let ty = c.request("textDocument/typeDefinition", at(&later, "made.bias", 0, 0));
    assert_eq!(
        ty["range"]["start"]["line"].as_u64().map(|l| l + 1),
        Some(line_of(&later, "type Config", 0)),
        "{ty}"
    );
    let gd = c.request("textDocument/definition", at(&later, "made.bias", 0, 5));
    assert_eq!(
        gd["range"]["start"]["line"].as_u64().map(|l| l + 1),
        Some(line_of(&later, "    bias: f32,", 0)),
        "{gd}"
    );
    let items = c.request("textDocument/completion", at(&later, "made.bias", 0, 5));
    let labels: Vec<&str> = items
        .as_array()
        .unwrap()
        .iter()
        .map(|i| i["label"].as_str().unwrap())
        .collect();
    assert_eq!(labels, ["scale", "bias"]);

    c.notify(
        "textDocument/didChange",
        json!({ "textDocument": { "uri": URI, "version": 2 }, "contentChanges": [{ "text": seq }] }),
    );
    let gd = c.request("textDocument/definition", at(&seq, "escaped + next", 0, 0));
    assert_eq!(
        gd["range"]["start"]["line"].as_u64().map(|l| l + 1),
        Some(line_of(&seq, "n * 10 -> escaped;", 0)),
        "{gd}"
    );
    let refs = c.request(
        "textDocument/references",
        json!({ "textDocument": { "uri": URI }, "position": pos(&seq, "escaped + next", 0, 0), "context": { "includeDeclaration": true } }),
    );
    assert_eq!(refs.as_array().map(Vec::len), Some(2), "{refs}");
    c.stop();
}

#[test]
fn rename_symbols_completion_formatting() {
    let text = llama();
    let mut c = Client::start();
    c.open(URI, &text);

    let mut p = at(&text, "mut hidden: [f32", 0, 4);
    let prep = c.request("textDocument/prepareRename", p.clone());
    assert!(prep["start"].is_object(), "{prep}");
    p["newName"] = json!("state");
    let edit = c.request("textDocument/rename", p.clone());
    let edits = edit["changes"][URI].as_array().unwrap();
    // `mut hidden` in forward: its declaration, two reads, one rebind.
    assert_eq!(edits.len(), 4, "{edit}");
    p["newName"] = json!("fold");
    assert!(
        c.request("textDocument/rename", p.clone())["error"].is_string(),
        "a keyword is refused"
    );
    p["newName"] = json!("iota");
    assert!(
        c.request("textDocument/rename", p)["error"].is_string(),
        "a builtin is refused"
    );

    let syms = c.request(
        "textDocument/documentSymbol",
        json!({ "textDocument": { "uri": URI } }),
    );
    let names: Vec<&str> = syms
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["name"].as_str().unwrap())
        .collect();
    for want in [
        "hidden_size",
        "Matrix",
        "KeyValueCache",
        "rotate",
        "forward",
        "main",
    ] {
        assert!(names.contains(&want), "{want} missing from {names:?}");
    }
    let kv = syms
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["name"] == "KeyValueCache")
        .unwrap();
    assert_eq!(kv["children"].as_array().unwrap().len(), 2);

    // Completion in a body: locals, parameters, globals, functions, builtins.
    let items = c.request(
        "textDocument/completion",
        at(&text, "(rotated_query, positions, updated_cache", 0, 0),
    );
    let labels: Vec<&str> = items
        .as_array()
        .unwrap()
        .iter()
        .map(|i| i["label"].as_str().unwrap())
        .collect();
    for want in [
        "rotated_query",
        "weights",
        "tokens",
        "hidden_size",
        "attend",
        "KeyValueCache",
        "sqrt",
        "map",
    ] {
        assert!(labels.contains(&want), "{want} missing from completion");
    }
    assert!(
        !labels.contains(&"attended"),
        "a later binding is not yet in scope"
    );

    // Member completion on a line the parser cannot finish.
    let edited = text.replacen(
        "    (rotated_query, positions,",
        "    updated_cache.\n    (rotated_query, positions,",
        1,
    );
    c.notify(
        "textDocument/didChange",
        json!({ "textDocument": { "uri": URI, "version": 2 }, "contentChanges": [{ "text": edited }] }),
    );
    let p = json!({ "textDocument": { "uri": URI }, "position": pos(&edited, "updated_cache.\n", 0, 14) });
    let items = c.request("textDocument/completion", p);
    let labels: Vec<&str> = items
        .as_array()
        .unwrap()
        .iter()
        .map(|i| i["label"].as_str().unwrap())
        .collect();
    assert_eq!(labels, ["keys", "values"]);
    // ... and go-to-definition still works elsewhere in the broken file.
    let r = c.request(
        "textDocument/definition",
        json!({ "textDocument": { "uri": URI }, "position": pos(&edited, "-> decoder_layer -> result", 0, 3) }),
    );
    assert_eq!(
        r["range"]["start"]["line"].as_u64().unwrap() + 1,
        line_of(&edited, "fn decoder_layer", 0)
    );

    // The committed example is formatted: no edits. An unformatted one: one.
    c.notify(
        "textDocument/didChange",
        json!({ "textDocument": { "uri": URI, "version": 3 }, "contentChanges": [{ "text": text }] }),
    );
    let f = c.request("textDocument/formatting", json!({ "textDocument": { "uri": URI }, "options": { "tabSize": 4, "insertSpaces": true } }));
    assert_eq!(f, json!([]));
    let messy = "fn main(){ 1->x; x->println; }\n";
    c.notify(
        "textDocument/didChange",
        json!({ "textDocument": { "uri": URI, "version": 4 }, "contentChanges": [{ "text": messy }] }),
    );
    let f = c.request("textDocument/formatting", json!({ "textDocument": { "uri": URI }, "options": { "tabSize": 4, "insertSpaces": true } }));
    assert_eq!(f[0]["newText"], mapal_syntax::format_source(messy).unwrap());
    c.stop();
}

#[test]
fn diagnostics_are_the_compilers() {
    let mut c = Client::start();
    c.open(URI, &llama());
    assert_eq!(
        c.diagnostics(URI),
        Vec::<Value>::new(),
        "llama.mapal compiles"
    );
    let bad = "fn main() {\n    1 -> x;\n    y -> println;\n}\n";
    c.open("file:///tmp/bad.mapal", bad);
    let ds = c.diagnostics("file:///tmp/bad.mapal");
    assert_eq!(ds.len(), 1, "{ds:?}");
    assert_eq!(
        ds[0]["range"]["start"],
        json!({ "line": 2, "character": 4 })
    );
    assert!(ds[0]["message"].as_str().unwrap().contains('y'));
    let parse = "fn main( {\n";
    c.open("file:///tmp/parse.mapal", parse);
    let ds = c.diagnostics("file:///tmp/parse.mapal");
    assert!(
        !ds.is_empty() && ds[0]["code"].as_str().unwrap().starts_with('P'),
        "{ds:?}"
    );
    c.notify(
        "textDocument/didClose",
        json!({ "textDocument": { "uri": "file:///tmp/parse.mapal" } }),
    );
    assert_eq!(
        c.diagnostics("file:///tmp/parse.mapal"),
        Vec::<Value>::new()
    );
    c.stop();
}

#[test]
fn semantic_tokens_decode_to_the_analysis() {
    let text = llama();
    let mut c = Client::start();
    c.open(URI, &text);
    let r = c.request(
        "textDocument/semanticTokens/full",
        json!({ "textDocument": { "uri": URI } }),
    );
    let data: Vec<u64> = r["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    assert_eq!(data.len() % 5, 0);
    let (mut line, mut col) = (0u64, 0u64);
    let mut decoded = Vec::new();
    for t in data.chunks(5) {
        if t[0] > 0 {
            col = 0;
        }
        line += t[0];
        col += t[1];
        decoded.push((line, col, t[2], t[3], t[4]));
    }
    let lines: Vec<&str> = text.lines().collect();
    let tok = |l: u64, c: u64, len: u64| &lines[l as usize][c as usize..(c + len) as usize];
    let types = mapal_lsp::server::TOKEN_TYPES;
    let mods = mapal_lsp::server::TOKEN_MODIFIERS;
    let find = |word: &str, ty: &str, m: Option<&str>| {
        decoded.iter().any(|&(l, c, len, t, bits)| {
            tok(l, c, len) == word
                && types[t as usize] == ty
                && m.map_or(bits == 0, |m| {
                    bits == 1 << mods.iter().position(|x| *x == m).unwrap()
                })
        })
    };
    assert!(find("decoder_layer", "function", Some("declaration")));
    assert!(find("decoder_layer", "function", None));
    assert!(find("iota", "function", Some("defaultLibrary")));
    assert!(find("KeyValueCache", "type", None));
    assert!(find("bf16", "type", Some("defaultLibrary")));
    assert!(find("hidden_size", "variable", Some("readonly")));
    assert!(find("hidden", "variable", Some("modification")));
    assert!(find("keys", "property", None));
    assert!(find("rows", "typeParameter", Some("declaration")));
    assert!(find("cell", "parameter", Some("declaration")));
    c.stop();
}
