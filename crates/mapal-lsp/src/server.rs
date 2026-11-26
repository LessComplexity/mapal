//! The stdio protocol loop (`lsp-server`) and the request handlers over
//! [`Analysis`]. Documents are synchronized whole (`TextDocumentSyncKind::FULL`):
//! a Mapal program is one file, and ADR-0008 (d) settles on a full reparse per
//! edit. Compile diagnostics (parse → lower → check) run on a worker thread so
//! a slow lowering never holds up a definition request.

use std::collections::HashMap;
use std::error::Error;
use std::sync::mpsc;

use lsp_server::{Connection, ErrorCode, Message, Notification, Request, Response};
use lsp_types::{
    CompletionItem, CompletionItemKind, CompletionOptions, CompletionParams, CompletionResponse,
    DiagnosticSeverity, DocumentFormattingParams, DocumentHighlight, DocumentHighlightKind,
    DocumentHighlightParams, DocumentSymbol, DocumentSymbolParams, DocumentSymbolResponse,
    GotoDefinitionParams, GotoDefinitionResponse, Hover, HoverContents, HoverParams,
    HoverProviderCapability, InitializeParams, Location, MarkupContent, MarkupKind, NumberOrString,
    OneOf, Position, PrepareRenameResponse, PublishDiagnosticsParams, Range, ReferenceParams,
    RenameOptions, RenameParams, SemanticToken, SemanticTokenModifier, SemanticTokenType,
    SemanticTokens, SemanticTokensFullOptions, SemanticTokensLegend, SemanticTokensOptions,
    SemanticTokensParams, SemanticTokensResult, SemanticTokensServerCapabilities,
    ServerCapabilities, ServerInfo, SymbolKind, TextDocumentIdentifier, TextDocumentPositionParams,
    TextDocumentSyncCapability, TextDocumentSyncKind, TextEdit, TypeDefinitionProviderCapability,
    Uri, WorkDoneProgressOptions, WorkspaceEdit,
    notification::{
        DidChangeTextDocument, DidCloseTextDocument, DidOpenTextDocument, DidSaveTextDocument,
        Notification as _, PublishDiagnostics,
    },
};
use mapal_syntax::{Diagnostic, Severity, SourceLoc};

use crate::analysis::{Analysis, Class, DefId, DefKind, Role, Ty};
use crate::docs;
use crate::text::LineMap;

type Res<T> = Result<T, Box<dyn Error + Send + Sync>>;

/// The semantic token legend. Standard LSP types and modifiers only, so
/// themes apply; each [`Class`] is at most one type and one modifier
/// (`editors/test/classes.tsv` lists the pairs).
pub const TOKEN_TYPES: [&str; 6] = [
    "function",
    "type",
    "typeParameter",
    "parameter",
    "variable",
    "property",
];
pub const TOKEN_MODIFIERS: [&str; 4] =
    ["declaration", "defaultLibrary", "readonly", "modification"];

/// A class's semantic token: `(type, modifier)` by name, or `None` when the
/// static grammars' class stands (a plain local read).
pub fn token_of(class: Class) -> Option<(&'static str, Option<&'static str>)> {
    Some(match class {
        Class::FunctionDeclaration => ("function", Some("declaration")),
        Class::Function => ("function", None),
        Class::Builtin => ("function", Some("defaultLibrary")),
        Class::TypeDeclaration => ("type", Some("declaration")),
        Class::Type => ("type", None),
        Class::Primitive => ("type", Some("defaultLibrary")),
        Class::SizeDeclaration => ("typeParameter", Some("declaration")),
        Class::Size => ("typeParameter", None),
        Class::Global => ("variable", Some("readonly")),
        Class::Binding => ("variable", Some("declaration")),
        Class::Rebind => ("variable", Some("modification")),
        Class::ParameterDeclaration => ("parameter", Some("declaration")),
        Class::Parameter => ("parameter", None),
        Class::FieldDeclaration => ("property", Some("declaration")),
        Class::Field => ("property", None),
        Class::Variable => return None,
    })
}

pub fn capabilities() -> ServerCapabilities {
    ServerCapabilities {
        text_document_sync: Some(TextDocumentSyncCapability::Kind(TextDocumentSyncKind::FULL)),
        definition_provider: Some(OneOf::Left(true)),
        declaration_provider: Some(lsp_types::DeclarationCapability::Simple(true)),
        type_definition_provider: Some(TypeDefinitionProviderCapability::Simple(true)),
        references_provider: Some(OneOf::Left(true)),
        document_highlight_provider: Some(OneOf::Left(true)),
        hover_provider: Some(HoverProviderCapability::Simple(true)),
        document_symbol_provider: Some(OneOf::Left(true)),
        rename_provider: Some(OneOf::Right(RenameOptions {
            prepare_provider: Some(true),
            work_done_progress_options: WorkDoneProgressOptions::default(),
        })),
        completion_provider: Some(CompletionOptions {
            trigger_characters: Some(vec![".".into()]),
            ..CompletionOptions::default()
        }),
        document_formatting_provider: Some(OneOf::Left(true)),
        semantic_tokens_provider: Some(SemanticTokensServerCapabilities::SemanticTokensOptions(
            SemanticTokensOptions {
                legend: SemanticTokensLegend {
                    token_types: TOKEN_TYPES
                        .iter()
                        .map(|t| SemanticTokenType::new(t))
                        .collect(),
                    token_modifiers: TOKEN_MODIFIERS
                        .iter()
                        .map(|m| SemanticTokenModifier::new(m))
                        .collect(),
                },
                full: Some(SemanticTokensFullOptions::Bool(true)),
                range: None,
                work_done_progress_options: WorkDoneProgressOptions::default(),
            },
        )),
        ..ServerCapabilities::default()
    }
}

/// `mapal lsp`: serve over stdin/stdout until the client exits.
pub fn run_stdio() -> Res<()> {
    let (connection, io_threads) = Connection::stdio();
    serve(connection)?;
    io_threads.join()?;
    Ok(())
}

/// Serve one client on `connection` (stdio, or an in-memory pair in tests).
pub fn serve(connection: Connection) -> Res<()> {
    let caps = serde_json::to_value(capabilities())?;
    let init = serde_json::json!({
        "capabilities": caps,
        "serverInfo": ServerInfo { name: "mapal-lsp".into(), version: Some(env!("CARGO_PKG_VERSION").into()) },
    });
    let (id, params) = connection.initialize_start()?;
    let _params: InitializeParams = serde_json::from_value(params).unwrap_or_default();
    connection.initialize_finish(id, init)?;

    let diag_tx = spawn_diagnostics(connection.sender.clone());
    let mut server = Server {
        docs: HashMap::new(),
        diag_tx,
    };
    for msg in &connection.receiver {
        match msg {
            Message::Request(req) => {
                if connection.handle_shutdown(&req)? {
                    return Ok(());
                }
                let resp = server.request(req);
                connection.sender.send(Message::Response(resp))?;
            }
            Message::Notification(n) => server.notification(n),
            Message::Response(_) => {}
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Diagnostics: parse → lower → check, the compiler's own front end.
// ---------------------------------------------------------------------------

enum DiagJob {
    Check {
        uri: Uri,
        version: i32,
        text: String,
    },
    Clear {
        uri: Uri,
    },
}

/// The diagnostics worker's stack. The front end recurses over the program's
/// nesting; a spawned thread's default (2 MiB) is a quarter of the main
/// thread's that `mapal build` runs it on, so a file the CLI reports on would
/// abort the server. Reserved, not committed: untouched pages cost nothing.
pub const DIAGNOSTICS_STACK: usize = 64 << 20;

fn spawn_diagnostics(sender: crossbeam_channel::Sender<Message>) -> mpsc::Sender<DiagJob> {
    let (tx, rx) = mpsc::channel::<DiagJob>();
    let worker = std::thread::Builder::new()
        .name("mapal-lsp diagnostics".into())
        .stack_size(DIAGNOSTICS_STACK);
    let spawned = worker.spawn(move || {
        while let Ok(first) = rx.recv() {
            // Only the latest text of each document is worth checking.
            let mut jobs = vec![first];
            while let Ok(more) = rx.try_recv() {
                let key = job_uri(&more).as_str().to_string();
                jobs.retain(|j| job_uri(j).as_str() != key);
                jobs.push(more);
            }
            for job in jobs {
                let (uri, version, diagnostics) = match job {
                    DiagJob::Check { uri, version, text } => {
                        let lines = LineMap::new(&text);
                        let ds = compile_diagnostics(&text)
                            .into_iter()
                            .map(|d| to_lsp_diagnostic(&text, &lines, &d))
                            .collect();
                        (uri, Some(version), ds)
                    }
                    DiagJob::Clear { uri } => (uri, None, Vec::new()),
                };
                let params = PublishDiagnosticsParams {
                    uri,
                    diagnostics,
                    version,
                };
                let note = Notification::new(PublishDiagnostics::METHOD.into(), params);
                if sender.send(Message::Notification(note)).is_err() {
                    return;
                }
            }
        }
    });
    if let Err(e) = spawned {
        // No worker: requests are still served, only diagnostics are missing.
        eprintln!("mapal lsp: no diagnostics worker: {e}");
    }
    tx
}

fn job_uri(j: &DiagJob) -> &Uri {
    match j {
        DiagJob::Check { uri, .. } | DiagJob::Clear { uri } => uri,
    }
}

/// The diagnostics `mapal build` would report: the parser's (lexer's
/// included) and, on a clean parse, lowering's and the checker's.
pub fn compile_diagnostics(text: &str) -> Vec<Diagnostic> {
    let po = mapal_syntax::parse(text);
    if po.diagnostics.iter().any(|d| d.severity == Severity::Error) {
        return po.diagnostics;
    }
    let mut out = po.diagnostics.clone();
    let program = po.program;
    let lowered =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            match mapal_lower::lower(text, &program) {
                Ok(ir) => mapal_check::check(text, &program, &ir),
                Err(ds) => ds,
            }
        }));
    match lowered {
        Ok(ds) => out.extend(ds),
        Err(_) => out.push(Diagnostic {
            code: mapal_syntax::DiagCode("X0001"),
            severity: Severity::Warning,
            span: SourceLoc::new(0, 0),
            message: "the compiler's front end stopped on this file (internal error); \
                      names still resolve"
                .into(),
            fix: None,
        }),
    }
    out
}

fn to_lsp_diagnostic(text: &str, lines: &LineMap, d: &Diagnostic) -> lsp_types::Diagnostic {
    lsp_types::Diagnostic {
        range: lines.range(text, d.span),
        severity: Some(match d.severity {
            Severity::Error => DiagnosticSeverity::ERROR,
            Severity::Warning => DiagnosticSeverity::WARNING,
        }),
        code: Some(NumberOrString::String(d.code.0.into())),
        source: Some("mapal".into()),
        message: d.message.clone(),
        ..lsp_types::Diagnostic::default()
    }
}

// ---------------------------------------------------------------------------
// Documents and requests.
// ---------------------------------------------------------------------------

struct Doc {
    uri: Uri,
    analysis: Analysis,
    lines: LineMap,
}

impl Doc {
    fn new(uri: Uri, text: String) -> Doc {
        let lines = LineMap::new(&text);
        Doc {
            uri,
            analysis: Analysis::new(&text),
            lines,
        }
    }

    fn text(&self) -> &str {
        &self.analysis.text
    }

    fn offset(&self, p: Position) -> u32 {
        self.lines.offset(self.text(), p)
    }

    fn range(&self, s: SourceLoc) -> Range {
        self.lines.range(self.text(), s)
    }

    fn location(&self, s: SourceLoc) -> Location {
        Location::new(self.uri.clone(), self.range(s))
    }
}

struct Server {
    docs: HashMap<String, Doc>,
    diag_tx: mpsc::Sender<DiagJob>,
}

fn key(uri: &Uri) -> String {
    uri.as_str().to_string()
}

impl Server {
    fn notification(&mut self, n: Notification) {
        match n.method.as_str() {
            DidOpenTextDocument::METHOD => {
                if let Ok(p) =
                    serde_json::from_value::<lsp_types::DidOpenTextDocumentParams>(n.params)
                {
                    let d = p.text_document;
                    self.update(d.uri, d.version, d.text);
                }
            }
            DidChangeTextDocument::METHOD => {
                if let Ok(p) =
                    serde_json::from_value::<lsp_types::DidChangeTextDocumentParams>(n.params)
                    && let Some(change) = p.content_changes.into_iter().last()
                {
                    self.update(p.text_document.uri, p.text_document.version, change.text);
                }
            }
            DidSaveTextDocument::METHOD => {}
            DidCloseTextDocument::METHOD => {
                if let Ok(p) =
                    serde_json::from_value::<lsp_types::DidCloseTextDocumentParams>(n.params)
                {
                    self.docs.remove(&key(&p.text_document.uri));
                    let _ = self.diag_tx.send(DiagJob::Clear {
                        uri: p.text_document.uri,
                    });
                }
            }
            _ => {}
        }
    }

    fn update(&mut self, uri: Uri, version: i32, text: String) {
        let _ = self.diag_tx.send(DiagJob::Check {
            uri: uri.clone(),
            version,
            text: text.clone(),
        });
        self.docs.insert(key(&uri), Doc::new(uri, text));
    }

    fn doc(&self, id: &TextDocumentIdentifier) -> Option<&Doc> {
        self.docs.get(&key(&id.uri))
    }

    fn at(&self, p: &TextDocumentPositionParams) -> Option<(&Doc, u32)> {
        let d = self.doc(&p.text_document)?;
        Some((d, d.offset(p.position)))
    }

    fn request(&mut self, req: Request) -> Response {
        let id = req.id.clone();
        match self.dispatch(req) {
            Ok(v) => Response::new_ok(id, v),
            Err(Failure::Unknown(m)) => Response::new_err(
                id,
                ErrorCode::MethodNotFound as i32,
                format!("unhandled method {m}"),
            ),
            Err(Failure::Params(e)) => Response::new_err(id, ErrorCode::InvalidParams as i32, e),
            Err(Failure::Refused(e)) => Response::new_err(id, ErrorCode::RequestFailed as i32, e),
        }
    }

    fn dispatch(&mut self, req: Request) -> Result<serde_json::Value, Failure> {
        fn params<P: serde::de::DeserializeOwned>(req: &Request) -> Result<P, Failure> {
            serde_json::from_value(req.params.clone()).map_err(|e| Failure::Params(e.to_string()))
        }
        let v = |x: Result<serde_json::Value, serde_json::Error>| {
            x.map_err(|e| Failure::Params(e.to_string()))
        };
        match req.method.as_str() {
            "textDocument/definition" | "textDocument/declaration" => {
                let p: GotoDefinitionParams = params(&req)?;
                v(serde_json::to_value(
                    self.definition(&p.text_document_position_params),
                ))
            }
            "textDocument/typeDefinition" => {
                let p: GotoDefinitionParams = params(&req)?;
                v(serde_json::to_value(
                    self.type_definition(&p.text_document_position_params),
                ))
            }
            "textDocument/references" => {
                let p: ReferenceParams = params(&req)?;
                v(serde_json::to_value(self.references(&p)))
            }
            "textDocument/documentHighlight" => {
                let p: DocumentHighlightParams = params(&req)?;
                v(serde_json::to_value(
                    self.highlights(&p.text_document_position_params),
                ))
            }
            "textDocument/hover" => {
                let p: HoverParams = params(&req)?;
                v(serde_json::to_value(
                    self.hover(&p.text_document_position_params),
                ))
            }
            "textDocument/documentSymbol" => {
                let p: DocumentSymbolParams = params(&req)?;
                v(serde_json::to_value(self.symbols(&p.text_document)))
            }
            "textDocument/prepareRename" => {
                let p: TextDocumentPositionParams = params(&req)?;
                v(serde_json::to_value(self.prepare_rename(&p)))
            }
            "textDocument/rename" => {
                let p: RenameParams = params(&req)?;
                let edit = self.rename(&p).map_err(Failure::Refused)?;
                v(serde_json::to_value(edit))
            }
            "textDocument/completion" => {
                let p: CompletionParams = params(&req)?;
                v(serde_json::to_value(
                    self.completion(&p.text_document_position),
                ))
            }
            "textDocument/formatting" => {
                let p: DocumentFormattingParams = params(&req)?;
                v(serde_json::to_value(self.format(&p.text_document)))
            }
            "textDocument/semanticTokens/full" => {
                let p: SemanticTokensParams = params(&req)?;
                v(serde_json::to_value(self.semantic_tokens(&p.text_document)))
            }
            m => Err(Failure::Unknown(m.to_string())),
        }
    }

    // --- navigation ---------------------------------------------------------

    fn definition(&self, p: &TextDocumentPositionParams) -> Option<GotoDefinitionResponse> {
        let (d, off) = self.at(p)?;
        let def = d.analysis.def_at(off)?;
        Some(GotoDefinitionResponse::Scalar(
            d.location(d.analysis.defs[def].name_span),
        ))
    }

    fn type_definition(&self, p: &TextDocumentPositionParams) -> Option<GotoDefinitionResponse> {
        let (d, off) = self.at(p)?;
        let a = &d.analysis;
        let def = &a.defs[a.def_at(off)?];
        let target = match def.kind {
            DefKind::Type => a.type_named(&def.name),
            DefKind::Function => a.type_def_of(&def.ty),
            _ => a.type_def_of(&def.ty),
        }?;
        Some(GotoDefinitionResponse::Scalar(
            d.location(a.defs[target].name_span),
        ))
    }

    fn references(&self, p: &ReferenceParams) -> Option<Vec<Location>> {
        let (d, off) = self.at(&p.text_document_position)?;
        let def = d.analysis.def_at(off)?;
        let with_decl = p.context.include_declaration;
        Some(
            d.analysis
                .occurrences(def)
                .filter(|o| with_decl || o.role != Role::Decl)
                .map(|o| d.location(o.span))
                .collect(),
        )
    }

    fn highlights(&self, p: &TextDocumentPositionParams) -> Option<Vec<DocumentHighlight>> {
        let (d, off) = self.at(p)?;
        let def = d.analysis.def_at(off)?;
        Some(
            d.analysis
                .occurrences(def)
                .map(|o| DocumentHighlight {
                    range: d.range(o.span),
                    kind: Some(match o.role {
                        Role::Decl | Role::Write => DocumentHighlightKind::WRITE,
                        _ => DocumentHighlightKind::READ,
                    }),
                })
                .collect(),
        )
    }

    // --- hover --------------------------------------------------------------

    fn hover(&self, p: &TextDocumentPositionParams) -> Option<Hover> {
        let (d, off) = self.at(p)?;
        let a = &d.analysis;
        let (span, markdown) = if let Some(o) = a.occ_at(off) {
            (o.span, hover_def(a, o.def))
        } else {
            let (span, class) = a.lexical_at(off)?;
            let name = a.slice(span);
            let what = match class {
                Class::Builtin => {
                    format!("builtin `{name}`\n\n{}", docs::builtin(name).unwrap_or(""))
                }
                _ => format!(
                    "primitive type `{name}`\n\n{}",
                    docs::primitive(name).unwrap_or("")
                ),
            };
            (span, what)
        };
        Some(Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: markdown,
            }),
            range: Some(d.range(span)),
        })
    }

    // --- symbols ------------------------------------------------------------

    #[allow(deprecated)] // `DocumentSymbol::deprecated` must still be written.
    fn symbols(&self, id: &TextDocumentIdentifier) -> Option<DocumentSymbolResponse> {
        let d = self.doc(id)?;
        let a = &d.analysis;
        let sym = |def: DefId, kind: SymbolKind, children: Option<Vec<DocumentSymbol>>| {
            let df = &a.defs[def];
            let mut range = d.range(df.span);
            let selection = d.range(df.name_span);
            if range.start > selection.start || range.end < selection.end {
                range = selection;
            }
            DocumentSymbol {
                name: df.name.clone(),
                detail: Some(detail(a, def)),
                kind,
                tags: None,
                deprecated: None,
                range,
                selection_range: selection,
                children,
            }
        };
        let mut out = Vec::new();
        for (i, df) in a.defs.iter().enumerate() {
            match df.kind {
                DefKind::Function => out.push(sym(i, SymbolKind::FUNCTION, None)),
                DefKind::Type => {
                    let fields = a
                        .fields_of(i)
                        .into_iter()
                        .map(|f| sym(f, SymbolKind::FIELD, None))
                        .collect();
                    out.push(sym(i, SymbolKind::STRUCT, Some(fields)));
                }
                DefKind::Global => out.push(sym(i, SymbolKind::CONSTANT, None)),
                _ => {}
            }
        }
        out.sort_by_key(|s| (s.range.start.line, s.range.start.character));
        Some(DocumentSymbolResponse::Nested(out))
    }

    // --- rename -------------------------------------------------------------

    fn prepare_rename(&self, p: &TextDocumentPositionParams) -> Option<PrepareRenameResponse> {
        let (d, off) = self.at(p)?;
        let o = d.analysis.occ_at(off)?;
        if d.analysis.defs[o.def].name == "_" {
            return None;
        }
        Some(PrepareRenameResponse::Range(d.range(o.span)))
    }

    fn rename(&self, p: &RenameParams) -> Result<Option<WorkspaceEdit>, String> {
        let Some((d, off)) = self.at(&p.text_document_position) else {
            return Ok(None);
        };
        let Some(def) = d.analysis.def_at(off) else {
            return Ok(None);
        };
        let new = p.new_name.as_str();
        valid_name(new)?;
        let edits = rename_edits(&d.analysis, def, new)
            .into_iter()
            .map(|(s, t)| TextEdit::new(d.range(s), t))
            .collect();
        Ok(Some(WorkspaceEdit {
            changes: Some(HashMap::from([(d.uri.clone(), edits)])),
            ..WorkspaceEdit::default()
        }))
    }

    // --- completion ---------------------------------------------------------

    fn completion(&self, p: &TextDocumentPositionParams) -> Option<CompletionResponse> {
        let (d, off) = self.at(p)?;
        Some(CompletionResponse::Array(completions(&d.analysis, off)))
    }

    // --- formatting ---------------------------------------------------------

    fn format(&self, id: &TextDocumentIdentifier) -> Option<Vec<TextEdit>> {
        let d = self.doc(id)?;
        let formatted = mapal_syntax::format_source(d.text()).ok()?;
        if formatted == d.text() {
            return Some(Vec::new());
        }
        let whole = SourceLoc::new(0, d.text().len() as u32);
        Some(vec![TextEdit::new(d.range(whole), formatted)])
    }

    // --- semantic tokens ----------------------------------------------------

    fn semantic_tokens(&self, id: &TextDocumentIdentifier) -> Option<SemanticTokensResult> {
        let d = self.doc(id)?;
        Some(SemanticTokensResult::Tokens(SemanticTokens {
            result_id: None,
            data: encode_tokens(&d.analysis, &d.lines),
        }))
    }
}

enum Failure {
    Unknown(String),
    Params(String),
    Refused(String),
}

/// The delta-encoded semantic tokens of a file (LSP §semanticTokens).
pub fn encode_tokens(a: &Analysis, lines: &LineMap) -> Vec<SemanticToken> {
    let mut out = Vec::new();
    let (mut prev_line, mut prev_start) = (0u32, 0u32);
    for (span, class) in a.classified() {
        let Some((ty, modifier)) = token_of(class) else {
            continue;
        };
        let start = lines.position(&a.text, span.start);
        let end = lines.position(&a.text, span.end);
        if end.line != start.line {
            continue;
        }
        let token_type = TOKEN_TYPES.iter().position(|t| *t == ty).unwrap_or(0) as u32;
        let token_modifiers_bitset = modifier
            .and_then(|m| TOKEN_MODIFIERS.iter().position(|x| *x == m))
            .map_or(0, |i| 1u32 << i);
        let delta_line = start.line - prev_line;
        let delta_start = if delta_line == 0 {
            start.character - prev_start
        } else {
            start.character
        };
        out.push(SemanticToken {
            delta_line,
            delta_start,
            length: end.character - start.character,
            token_type,
            token_modifiers_bitset,
        });
        prev_line = start.line;
        prev_start = start.character;
    }
    out
}

/// A name a rename may introduce: an identifier the lexer keeps an
/// identifier and lowering does not reserve.
pub fn valid_name(new: &str) -> Result<(), String> {
    let mut chars = new.chars();
    let ident = chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_');
    if !ident || new == "_" {
        return Err(format!("`{new}` is not an identifier"));
    }
    if mapal_syntax::KEYWORDS.contains(&new) {
        return Err(format!("`{new}` is a keyword"));
    }
    if mapal_lower::is_builtin(new) || mapal_lower::SCALAR_TYPES.contains(&new) {
        return Err(format!("`{new}` is a reserved name"));
    }
    Ok(())
}

/// The edits renaming `def` to `new`: every occurrence; a struct-literal pun
/// `T { name }` is spelled out (`name: new` for a renamed value, `new: name`
/// for a renamed field) so it keeps meaning both.
pub fn rename_edits(a: &Analysis, def: DefId, new: &str) -> Vec<(SourceLoc, String)> {
    a.occurrences(def)
        .map(|o| {
            let text = if !a.puns.contains(&o.span) {
                new.to_string()
            } else if a.defs[def].kind == DefKind::Field {
                format!("{new}: {}", a.slice(o.span))
            } else {
                format!("{}: {new}", a.slice(o.span))
            };
            (o.span, text)
        })
        .collect()
}

/// The hover text of a definition (Markdown).
pub fn hover_def(a: &Analysis, def: DefId) -> String {
    let d = &a.defs[def];
    let code = |s: &str| format!("```mapal\n{s}\n```");
    let mut out = match d.kind {
        DefKind::Function | DefKind::Type => code(&detail(a, def)),
        _ => code(&format!(
            "{}{}: {}",
            if d.mutable { "mut " } else { "" },
            d.name,
            d.ty.render()
        )),
    };
    let what = match d.kind {
        DefKind::Field => match d.container {
            Some(t) => format!("field of `{}`", a.defs[t].name),
            None => "field".into(),
        },
        DefKind::SizeParam => match d.container {
            Some(o) if a.defs[o].kind == DefKind::Type => {
                format!("size parameter of `type {}`", a.defs[o].name)
            }
            Some(o) => format!("size parameter of `fn {}`", a.defs[o].name),
            None => "size parameter".into(),
        },
        DefKind::Param => match d.container {
            Some(f) => format!("parameter of `fn {}`", a.defs[f].name),
            None => "parameter".into(),
        },
        k => k.describe().to_string(),
    };
    out.push_str(&format!("\n\n{what}"));
    if matches!(d.kind, DefKind::Global | DefKind::Local) {
        let stmt = one_line(a.slice(d.span));
        if !stmt.is_empty() {
            out.push_str(&format!(" — `{}`", truncate(&stmt, 120)));
        }
    }
    let comments = doc_comment(&a.text, d.span);
    if !comments.is_empty()
        && matches!(
            d.kind,
            DefKind::Function | DefKind::Type | DefKind::Global | DefKind::Field
        )
    {
        out.push_str("\n\n");
        out.push_str(&comments);
    }
    out
}

/// A one-line description: a function's signature, a type's header, a
/// value's type.
pub fn detail(a: &Analysis, def: DefId) -> String {
    let d = &a.defs[def];
    match d.kind {
        DefKind::Function | DefKind::Type => {
            let src = a.slice(d.span);
            let head = src.split('{').next().unwrap_or(src);
            let mut s = one_line(head);
            s = s.replace("( ", "(").replace(", )", ")");
            if d.kind == DefKind::Type {
                let fields: Vec<String> = a
                    .fields_of(def)
                    .into_iter()
                    .map(|f| format!("{}: {}", a.defs[f].name, a.defs[f].ty.render()))
                    .collect();
                s = format!("{s} {{ {} }}", fields.join(", "));
            }
            s
        }
        _ => d.ty.render(),
    }
}

fn one_line(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        return s.to_string();
    }
    let t: String = s.chars().take(n).collect();
    format!("{t}…")
}

/// The `//` comment lines directly above the line holding `span`, and a
/// trailing comment on its own line; ruler lines (`// ----`) dropped.
fn doc_comment(text: &str, span: SourceLoc) -> String {
    let start = span.start as usize;
    let line_start = text[..start.min(text.len())]
        .rfind('\n')
        .map_or(0, |i| i + 1);
    let mut lines: Vec<&str> = Vec::new();
    let mut cursor = line_start;
    while cursor > 0 {
        let prev_end = cursor - 1;
        let prev_start = text[..prev_end].rfind('\n').map_or(0, |i| i + 1);
        let line = text[prev_start..prev_end].trim();
        let Some(body) = line.strip_prefix("//") else {
            break;
        };
        lines.push(body.trim());
        cursor = prev_start;
    }
    lines.reverse();
    let line_end = text[start.min(text.len())..]
        .find('\n')
        .map_or(text.len(), |i| start + i);
    if let Some(i) = text[line_start..line_end].find("//")
        && line_start + i >= span.end as usize
    {
        lines.push(text[line_start + i + 2..line_end].trim());
    }
    lines
        .into_iter()
        .filter(|l| !l.chars().all(|c| c == '-' || c == '='))
        .collect::<Vec<_>>()
        .join("\n")
}

// ---------------------------------------------------------------------------
// Completion.
// ---------------------------------------------------------------------------

pub fn completions(a: &Analysis, off: u32) -> Vec<CompletionItem> {
    let text = &a.text;
    let off = (off as usize).min(text.len());
    let bytes = text.as_bytes();
    let mut start = off;
    while start > 0 && (bytes[start - 1].is_ascii_alphanumeric() || bytes[start - 1] == b'_') {
        start -= 1;
    }
    if start > 0 && bytes[start - 1] == b'.' {
        return member_completions(a, start - 1);
    }
    let mut items = Vec::new();
    for id in a.visible_at(off as u32) {
        let d = &a.defs[id];
        let kind = match d.kind {
            DefKind::Function => CompletionItemKind::FUNCTION,
            DefKind::Type => CompletionItemKind::STRUCT,
            DefKind::Global => CompletionItemKind::CONSTANT,
            DefKind::SizeParam => CompletionItemKind::TYPE_PARAMETER,
            DefKind::Field => CompletionItemKind::FIELD,
            _ => CompletionItemKind::VARIABLE,
        };
        items.push(CompletionItem {
            label: d.name.clone(),
            kind: Some(kind),
            detail: Some(detail(a, id)),
            ..CompletionItem::default()
        });
    }
    for b in mapal_lower::builtin_names() {
        items.push(CompletionItem {
            label: b.into(),
            kind: Some(CompletionItemKind::FUNCTION),
            detail: docs::builtin(b).map(str::to_string),
            ..CompletionItem::default()
        });
    }
    let precisions = mapal_lower::PRECISION_MEMBERS
        .iter()
        .filter(|p| !mapal_lower::SCALAR_TYPES.contains(p));
    for t in mapal_lower::SCALAR_TYPES.iter().chain(precisions) {
        items.push(CompletionItem {
            label: (*t).into(),
            kind: Some(CompletionItemKind::KEYWORD),
            detail: docs::primitive(t).map(str::to_string),
            ..CompletionItem::default()
        });
    }
    for k in [
        "fn", "type", "mut", "loop", "ret", "seq", "map", "fold", "true", "false",
    ] {
        items.push(CompletionItem {
            label: k.into(),
            kind: Some(CompletionItemKind::KEYWORD),
            ..CompletionItem::default()
        });
    }
    items
}

/// `base.` — the fields of the base's type. The base is a dotted name path
/// (`model.layers.`, `cache.`) resolved at the dot, so it works on a line the
/// parser cannot finish yet.
fn member_completions(a: &Analysis, dot: usize) -> Vec<CompletionItem> {
    let bytes = a.text.as_bytes();
    let mut s = dot;
    while s > 0
        && (bytes[s - 1].is_ascii_alphanumeric() || bytes[s - 1] == b'_' || bytes[s - 1] == b'.')
    {
        s -= 1;
    }
    let path: Vec<&str> = a.text[s..dot]
        .split('.')
        .filter(|p| !p.is_empty())
        .collect();
    let Some((first, rest)) = path.split_first() else {
        return Vec::new();
    };
    let Some(root) = a
        .visible_at(dot as u32)
        .into_iter()
        .find(|&id| a.defs[id].name == *first)
    else {
        return Vec::new();
    };
    let mut ty = a.defs[root].ty.clone();
    for seg in rest {
        ty = match seg.parse::<usize>() {
            Ok(i) => match &ty {
                Ty::Tuple(ts) => ts.get(i).cloned().unwrap_or(Ty::Unknown),
                _ => Ty::Unknown,
            },
            Err(_) => field_ty(a, &ty, seg),
        };
    }
    let mut items = Vec::new();
    if let Ty::Tuple(ts) = &ty {
        for (i, t) in ts.iter().enumerate() {
            items.push(CompletionItem {
                label: i.to_string(),
                kind: Some(CompletionItemKind::FIELD),
                detail: Some(t.render()),
                ..CompletionItem::default()
            });
        }
    }
    if let Some(t) = a
        .type_def_of(&ty)
        .filter(|_| !matches!(ty, Ty::Array { .. }))
    {
        for f in a.fields_of(t) {
            let fd = &a.defs[f];
            if fd.name == "_" {
                continue;
            }
            items.push(CompletionItem {
                label: fd.name.clone(),
                kind: Some(CompletionItemKind::FIELD),
                detail: Some(fd.ty.render()),
                ..CompletionItem::default()
            });
        }
    }
    items
}

fn field_ty(a: &Analysis, ty: &Ty, name: &str) -> Ty {
    let Ty::Named { name: tname, .. } = ty else {
        return Ty::Unknown;
    };
    let Some(t) = a.type_named(tname) else {
        return Ty::Unknown;
    };
    a.fields_of(t)
        .into_iter()
        .find(|&f| a.defs[f].name == name)
        .map(|f| a.defs[f].ty.clone())
        .unwrap_or(Ty::Unknown)
}
