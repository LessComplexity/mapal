//! The drawing both categorical Mermaid views build: nodes, nested subgraphs
//! and labelled arrows, kept as data so tests can check the structure of a
//! diagram (arrows and their ends, boxes, wires) before it is rendered to text.

use crate::graph::{MorphismId, ObjectId};
use crate::mermaid::escape;

/// What a node stands for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeKind {
    /// An object (category view) — `name: type`, or `value: type`.
    Object,
    /// The function's context `Γ` (category view).
    Context,
    /// A product object, made by one pairing from `Γ` (category view).
    Product,
    /// A morphism drawn as a box (string view).
    Box,
    /// An open wire end: an input, a constant or an output (string view).
    Terminal,
    /// A copy `Δ`: one value read by several steps (string view).
    Split,
    /// A junction where several steps write one loop state (string view).
    Merge,
}

/// How an arrow is to be read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeKind {
    /// A morphism between two objects (category view).
    Morphism,
    /// The pairing `⟨a, b⟩` from `Γ` into a product (category view).
    Pairing,
    /// A projection `π` out of a product (category view).
    Projection,
    /// The one wire of an object (string view).
    Wire,
    /// A wire a copy or a merge adds: copy to reader, writer to merge.
    Branch,
}

#[derive(Clone, Debug)]
pub struct DNode {
    pub id: String,
    pub label: String,
    pub kind: NodeKind,
    /// The subgraph holding it (`0` is the top level).
    pub sub: usize,
    pub obj: Option<ObjectId>,
    pub morph: Option<MorphismId>,
}

#[derive(Clone, Debug)]
pub struct DSub {
    pub id: String,
    pub label: String,
    pub parent: usize,
    /// A box that holds a body (string view).
    pub morph: Option<MorphismId>,
}

#[derive(Clone, Debug)]
pub struct DEdge {
    /// A node id, or (string view) the id of a box that is a subgraph.
    pub from: String,
    pub to: String,
    pub label: String,
    pub kind: EdgeKind,
    pub obj: Option<ObjectId>,
    pub morph: Option<MorphismId>,
}

/// A drawing: `subs[0]` is the top level (not drawn).
#[derive(Clone, Debug)]
pub struct Diagram {
    pub subs: Vec<DSub>,
    pub nodes: Vec<DNode>,
    pub edges: Vec<DEdge>,
}

impl Default for Diagram {
    fn default() -> Self {
        Diagram {
            subs: vec![DSub { id: String::new(), label: String::new(), parent: 0, morph: None }],
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }
}

impl Diagram {
    pub fn sub(&mut self, id: String, label: String, parent: usize, morph: Option<MorphismId>) -> usize {
        self.subs.push(DSub { id, label, parent, morph });
        self.subs.len() - 1
    }

    pub fn node(&mut self, id: String, label: String, kind: NodeKind, sub: usize) -> usize {
        self.nodes.push(DNode { id, label, kind, sub, obj: None, morph: None });
        self.nodes.len() - 1
    }

    pub fn edge(&mut self, from: &str, to: &str, label: String, kind: EdgeKind) -> usize {
        let label = if label.trim().is_empty() { "·".to_string() } else { label };
        self.edges.push(DEdge { from: from.into(), to: to.into(), label, kind, obj: None, morph: None });
        self.edges.len() - 1
    }

    /// The Mermaid text: nodes by subgraph, then every arrow.
    pub fn render(&self) -> String {
        let mut out = String::from("flowchart LR\n");
        self.draw_sub(0, 0, &mut out);
        for e in &self.edges {
            out.push_str(&format!("    {} -- \"{}\" --> {}\n", e.from, escape(&e.label), e.to));
        }
        out
    }

    fn draw_sub(&self, s: usize, depth: usize, out: &mut String) {
        let pad = "    ".repeat(depth + 1);
        if s != 0 {
            out.push_str(&format!("{pad}subgraph {}[\"{}\"]\n", self.subs[s].id, escape(&self.subs[s].label)));
        }
        let inner = if s == 0 { depth } else { depth + 1 };
        let ipad = "    ".repeat(inner + 1);
        for n in self.nodes.iter().filter(|n| n.sub == s) {
            let l = escape(&n.label);
            let line = match n.kind {
                NodeKind::Box => format!("{}[\"{l}\"]", n.id),
                NodeKind::Product => format!("{}([\"{l}\"])", n.id),
                _ => format!("{}((\"{l}\"))", n.id),
            };
            out.push_str(&format!("{ipad}{line}\n"));
        }
        for c in (1..self.subs.len()).filter(|&c| self.subs[c].parent == s) {
            self.draw_sub(c, inner, out);
        }
        if s != 0 {
            out.push_str(&format!("{pad}end\n"));
        }
    }
}
