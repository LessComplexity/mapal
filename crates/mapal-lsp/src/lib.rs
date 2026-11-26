//! `mapal-lsp`: the Mapal language server (ADR-0008 Part 2), served by
//! `mapal lsp` over stdio.
//!
//! A thin adapter over the language's own crates: `mapal-syntax` lexes,
//! parses and formats, `mapal-lower` + `mapal-check` report the compiler's
//! diagnostics, and [`analysis`] resolves names by lowering's scoping rules.
//! Built on `lsp-server` + `lsp-types` (the synchronous scaffold
//! rust-analyzer uses): the work per request is a reparse of one file, so an
//! async runtime would buy nothing and cost a dependency tree.

pub mod analysis;
pub mod docs;
pub mod server;
pub mod text;

pub use analysis::{Analysis, Class};
pub use server::{run_stdio, serve};
