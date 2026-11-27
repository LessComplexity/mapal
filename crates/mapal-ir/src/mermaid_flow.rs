//! The condensed flow view of the graph ([`CategoryIr::to_mermaid_flow`]).
//!
//! One node per object the source NAMED (bindings, parameters, loop states),
//! per `map` / `fold` / call result, per array read from a file, per printed
//! output, and per body result. Every other object (pairs, projections,
//! indexes, literals, the io token) is plumbing: an edge connects two drawn
//! nodes when the second reads the first through such plumbing alone. A body
//! reads its enclosing site's operands through its input slots, so a
//! dependence on slot `i` continues at the site's `Pair i` operand. A
//! `map` / `fold` whose body holds drawn nodes is a `subgraph` named by the
//! binding it produces; node ids are the full dump's (`f{i}o{j}`), so the two
//! views cross-reference.

use std::collections::{HashMap, HashSet};

use slotmap::SecondaryMap;

use crate::graph::{CategoryIr, FuncId, FuncKind, MorphismId, ObjectId, ObjectKind, Operation};
use crate::mermaid::escape;
use crate::ty::{Ty, Value, ty_contains_token, ty_label};

/// What a flow node stands for in the full graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FlowAnchor {
    /// An object of the graph.
    Object(ObjectId),
    /// A `Print` morphism (its output is the io token, which is not drawn).
    Print(MorphismId),
}

/// The flow view as data: nodes in drawing order and edges between them.
#[derive(Clone, Debug, Default)]
pub struct FlowGraph {
    pub nodes: Vec<FlowAnchor>,
    /// `(from, to, back)`: indexes into `nodes`; `back` when the dependence
    /// crosses a `LoopBack`.
    pub edges: Vec<(usize, usize, bool)>,
    /// Parallel to `edges`: the morphisms the edge stands for, nearest the
    /// target first (the steps that make the target from the source).
    pub paths: Vec<Vec<MorphismId>>,
}

/// Plumbing morphisms carry no step of their own.
pub(crate) fn is_plumbing(op: &Operation) -> bool {
    matches!(
        op,
        Operation::Pair { .. }
            | Operation::Proj { .. }
            | Operation::Output
            | Operation::LoopEnter
            | Operation::LoopBack
            | Operation::LoopExit
    )
}

/// The plain word for a step.
pub(crate) fn step_word(op: &Operation) -> String {
    match op {
        Operation::Add => "+".into(),
        Operation::Sub => "−".into(),
        Operation::Mul => "×".into(),
        Operation::Div => "÷".into(),
        Operation::Mod => "mod".into(),
        Operation::Neg => "neg".into(),
        Operation::Eq => "==".into(),
        Operation::Neq => "!=".into(),
        Operation::Lt => "<".into(),
        Operation::Gt => ">".into(),
        Operation::Le => "<=".into(),
        Operation::Ge => ">=".into(),
        Operation::And => "and".into(),
        Operation::Or => "or".into(),
        Operation::Not => "not".into(),
        Operation::Phi => "guard".into(),
        Operation::Call(_) => "call".into(),
        Operation::Map { .. } => "map".into(),
        Operation::Fold { .. } => "fold".into(),
        Operation::Index => "index".into(),
        Operation::Zip => "zip".into(),
        Operation::Enumerate => "enumerate".into(),
        Operation::Iota => "iota".into(),
        Operation::Fill => "fill".into(),
        Operation::Update => "update".into(),
        Operation::Print { .. } => "print".into(),
        Operation::TimeMs => "clock".into(),
        Operation::Read => "read".into(),
        Operation::Widen => "widen".into(),
        Operation::Narrow => "narrow".into(),
        Operation::Math(f) => f.label().into(),
        _ => "value".into(),
    }
}

impl FlowGraph {
    /// The label of edge `i`: the source text of the expression the edge stands
    /// for when `src` is given and a step on its path carries a span (trimmed
    /// to 24 characters), else the composed step words (at most 3, in dataflow
    /// order, joined with ` · `). Never empty.
    pub fn edge_label(&self, ir: &CategoryIr, i: usize, src: Option<&str>) -> String {
        let path = &self.paths[i];
        if let Some(src) = src {
            // A span covers an expression only on an `Index` (`a[row * n + k]`); the
            // other steps' spans are their operator token. An edge into a site reads
            // its operand inside the body: the body's index of that slot is its text.
            let mut cands: Vec<MorphismId> = path.clone();
            let slot = path.iter().find_map(|&m| match ir.morphism(m)?.op {
                Operation::Pair { slot, .. } => Some(slot),
                _ => None,
            });
            let body = path.iter().find_map(|&m| match ir.morphism(m)?.op {
                Operation::Map { body, .. } | Operation::Fold { body, .. } => Some(body),
                _ => None,
            });
            if let (Some(slot), Some(body)) = (slot, body) {
                cands.extend(ir.body_index_reads(body, slot));
            }
            for m in cands {
                let Some(mo) = ir.morphism(m) else { continue };
                if mo.op != Operation::Index || mo.loc.is_empty() {
                    continue;
                }
                let Some(t) = src.get(mo.loc.start as usize..mo.loc.end as usize) else {
                    continue;
                };
                let t = t.split_whitespace().collect::<Vec<_>>().join(" ");
                if t.is_empty() {
                    continue;
                }
                return if t.chars().count() > 24 {
                    let mut c: String = t.chars().take(23).collect();
                    c.push('…');
                    c
                } else {
                    t
                };
            }
        }
        let mut words: Vec<String> = Vec::new();
        for &m in path.iter().rev() {
            let Some(mo) = ir.morphism(m) else { continue };
            if is_plumbing(&mo.op) {
                continue;
            }
            let w = step_word(&mo.op);
            if words.last() != Some(&w) {
                words.push(w);
            }
        }
        if words.is_empty() {
            return "operand".into();
        }
        let from = words.len().saturating_sub(3);
        words[from..].join(" · ")
    }
}

pub(crate) struct Node {
    pub(crate) anchor: FlowAnchor,
    pub(crate) func: FuncId,
    pub(crate) id: String,
    pub(crate) label: String,
    /// The body drawn inside this node (a subgraph) when it is a site result.
    pub(crate) body: Option<FuncId>,
}

pub(crate) struct Ctx<'a> {
    ir: &'a CategoryIr,
    pub(crate) func_ix: HashMap<FuncId, usize>,
    /// Object ordinal within its function (the full dump's `f{i}o{j}`).
    pub(crate) ord: SecondaryMap<ObjectId, usize>,
    pub(crate) incoming: SecondaryMap<ObjectId, Vec<MorphismId>>,
    /// Body function -> the `Map` / `Fold` morphisms that use it.
    pub(crate) sites: HashMap<FuncId, Vec<MorphismId>>,
}

impl Ctx<'_> {
    pub(crate) fn is_body(&self, f: FuncId) -> bool {
        self.ir.func(f).is_some_and(|d| d.kind != FuncKind::Named)
    }

    fn site_body(op: &Operation) -> Option<FuncId> {
        match op {
            Operation::Map { body, .. } | Operation::Fold { body, .. } => Some(*body),
            _ => None,
        }
    }

    /// The incoming `Read` of a file-read array: `Proj 1` of a `Read` result.
    pub(crate) fn read_of(&self, o: ObjectId) -> Option<MorphismId> {
        for &m in self.incoming.get(o)? {
            let mo = self.ir.morphism(m)?;
            if mo.op == (Operation::Proj { index: 1 }) {
                for &r in self.incoming.get(mo.source)? {
                    if self.ir.morphism(r)?.op == Operation::Read {
                        return Some(r);
                    }
                }
            }
        }
        None
    }

    /// The operands a site reads at input slot `slot` (all when `None`).
    fn site_operands(&self, site: MorphismId, slot: Option<u32>) -> Vec<ObjectId> {
        let m = self.ir.morphism(site).unwrap();
        if let Some(i) = slot {
            let hits: Vec<ObjectId> = self.incoming[m.source]
                .iter()
                .filter_map(|&p| {
                    let pm = self.ir.morphism(p)?;
                    matches!(pm.op, Operation::Pair { slot, .. } if slot == i).then_some(pm.source)
                })
                .collect();
            if !hits.is_empty() {
                return hits;
            }
        }
        vec![m.source]
    }
}

impl CategoryIr {
    /// The flow view as data (see the module docs).
    pub fn flow_graph(&self) -> FlowGraph {
        let (_, g) = self.build_flow();
        g
    }

    /// A condensed, readable Mermaid view of the graph: the values the source
    /// named and what each reads, `map` / `fold` bodies as subgraphs named by
    /// the binding they produce. Deterministic; passes [`crate::lint_mermaid`].
    pub fn to_mermaid_flow(&self) -> String {
        self.flow_text(None)
    }

    /// [`Self::to_mermaid_flow`] with arrows labelled by the source text of the
    /// expression they stand for (`src` is the program the graph was lowered from).
    pub fn to_mermaid_flow_src(&self, src: &str) -> String {
        self.flow_text(Some(src))
    }

    fn flow_text(&self, src: Option<&str>) -> String {
        let (nodes, g) = self.build_flow();
        let mut out = String::from("flowchart LR\n");
        let ctx = self.flow_ctx();
        let named: Vec<FuncId> = self
            .funcs()
            .filter(|(_, d)| d.kind == FuncKind::Named)
            .map(|(f, _)| f)
            .collect();
        let wrap = named.len() > 1;
        let mut declared: HashSet<usize> = HashSet::new();
        for f in named {
            let ind = if wrap { 2 } else { 1 };
            let any = nodes.iter().any(|n| n.func == f);
            if wrap && any {
                out.push_str(&format!(
                    "    subgraph f{}[\"fn {}\"]\n",
                    ctx.func_ix[&f],
                    escape(&self.func(f).unwrap().name)
                ));
            }
            draw(&nodes, f, ind, &mut out, &mut declared);
            if wrap && any {
                out.push_str("    end\n");
            }
        }
        for (k, &(a, b, back)) in g.edges.iter().enumerate() {
            let label = g.edge_label(self, k, src);
            let label = if back { format!("{label} ↩") } else { label };
            let (a, b) = (&nodes_ref(&nodes, &g, a), &nodes_ref(&nodes, &g, b));
            out.push_str(&format!(
                "    {} -- \"{}\" --> {}\n",
                edge_id(a),
                escape(&label),
                edge_id(b)
            ));
        }
        out
    }

    pub(crate) fn flow_ctx(&self) -> Ctx<'_> {
        let func_ix: HashMap<FuncId, usize> =
            self.funcs().enumerate().map(|(i, (f, _))| (f, i)).collect();
        let mut ord = SecondaryMap::new();
        let mut counts: HashMap<FuncId, usize> = HashMap::new();
        for (o, _) in self.objects() {
            if let Some(f) = self.try_owner(o) {
                let c = counts.entry(f).or_insert(0);
                ord.insert(o, *c);
                *c += 1;
            }
        }
        let mut incoming: SecondaryMap<ObjectId, Vec<MorphismId>> = SecondaryMap::new();
        for (o, _) in self.objects() {
            incoming.insert(o, Vec::new());
        }
        let mut sites: HashMap<FuncId, Vec<MorphismId>> = HashMap::new();
        for (m, mo) in self.morphisms() {
            incoming[mo.target].push(m);
            if let Some(b) = Ctx::site_body(&mo.op) {
                sites.entry(b).or_default().push(m);
            }
        }
        Ctx {
            ir: self,
            func_ix,
            ord,
            incoming,
            sites,
        }
    }

    pub(crate) fn build_flow(&self) -> (Vec<Node>, FlowGraph) {
        let ctx = self.flow_ctx();
        // Pass 1: which objects are plain drawn values (before site results).
        let mut nodes: Vec<Node> = Vec::new();
        let mut node_of: SecondaryMap<ObjectId, usize> = SecondaryMap::new();

        // Body drawn? A map body always; a fold body when it holds a drawn node
        // other than its result. Decided over the functions' own values first.
        let mut own_nodes: HashMap<FuncId, usize> = HashMap::new();
        for (o, obj) in self.objects() {
            let Some(f) = self.try_owner(o) else { continue };
            if self.plain_value(&ctx, o, obj) {
                *own_nodes.entry(f).or_insert(0) += 1;
            }
        }
        for (_, mo) in self.morphisms() {
            if matches!(mo.op, Operation::Print { .. }) {
                if let Some(f) = self.try_owner(mo.source) {
                    *own_nodes.entry(f).or_insert(0) += 1;
                }
            }
        }
        // A body counts a nested site result as a node (always drawn).
        let mut drawn_body: HashMap<FuncId, bool> = HashMap::new();
        fn drawn(
            ctx: &Ctx,
            f: FuncId,
            own: &HashMap<FuncId, usize>,
            memo: &mut HashMap<FuncId, bool>,
        ) -> bool {
            if let Some(&d) = memo.get(&f) {
                return d;
            }
            memo.insert(f, false);
            let is_map = ctx.sites.get(&f).is_some_and(|ss| {
                ss.iter()
                    .any(|&m| matches!(ctx.ir.morphism(m).unwrap().op, Operation::Map { .. }))
            });
            let d = is_map || own.get(&f).copied().unwrap_or(0) > 0;
            memo.insert(f, d);
            d
        }

        for (fi, (f, def)) in self.funcs().enumerate() {
            let _ = fi;
            let mut print_k = 0usize;
            for &o in self.objects_of(f).iter() {
                let obj = self.object(o).unwrap();
                let oi = ctx.ord[o];
                let inc = &ctx.incoming[o];
                let site = inc.iter().copied().find(|&m| {
                    matches!(
                        self.morphism(m).unwrap().op,
                        Operation::Map { .. } | Operation::Fold { .. } | Operation::Call(_)
                    )
                });
                let fidx = ctx.func_ix[&f];
                let name = obj.name.clone();
                let ty = ty_label(&obj.ty);
                if ty_contains_token(&obj.ty) || obj.kind == ObjectKind::Constant {
                    continue;
                }
                let (label, body) = if let Some(s) = site {
                    let op = self.morphism(s).unwrap().op;
                    match op {
                        Operation::Map { body, .. } | Operation::Fold { body, .. } => {
                            let is_map = matches!(op, Operation::Map { .. });
                            let inner = drawn(&ctx, body, &own_nodes, &mut drawn_body)
                                && ctx.sites[&body][0] == s;
                            let word = if is_map { "map over cells" } else { "fold" };
                            match (&name, inner) {
                                (Some(n), true) => (format!("{n} = {word}"), Some(body)),
                                (None, true) => (word.to_string(), Some(body)),
                                (Some(n), false) => (format!("{n}: {ty} · {word}"), None),
                                (None, false) => (format!("{word}: {ty}"), None),
                            }
                        }
                        Operation::Call(c) => {
                            let callee = self.func(c).map_or("?".into(), |d| d.name.clone());
                            match &name {
                                Some(n) => (format!("{n}: {ty} · call {callee}"), None),
                                None => (format!("call {callee}: {ty}"), None),
                            }
                        }
                        _ => unreachable!(),
                    }
                } else if obj.kind == ObjectKind::Return {
                    if def.kind == FuncKind::Named {
                        (format!("ret: {ty}"), None)
                    } else {
                        // A body's result is drawn only inside a drawn body.
                        if !drawn(&ctx, f, &own_nodes, &mut drawn_body) {
                            continue;
                        }
                        (format!("result: {ty}"), None)
                    }
                } else if let Some(r) = ctx.read_of(o) {
                    let file = self.read_file(&ctx, r);
                    let n = name.clone().unwrap_or_else(|| "read".into());
                    (
                        match file {
                            Some(p) => format!("{n}: {ty} ← read {p}"),
                            None => format!("{n}: {ty} ← read"),
                        },
                        None,
                    )
                } else if obj.kind == ObjectKind::LoopMerge {
                    (
                        format!("⟲ {}: {ty}", name.as_deref().unwrap_or("state")),
                        None,
                    )
                } else if let Some(n) = name {
                    (format!("{n}: {ty}"), None)
                } else {
                    continue;
                };
                let is_sub = body.is_some();
                let id = if is_sub {
                    format!("s{fidx}o{oi}")
                } else {
                    format!("f{fidx}o{oi}")
                };
                node_of.insert(o, nodes.len());
                nodes.push(Node {
                    anchor: FlowAnchor::Object(o),
                    func: f,
                    id,
                    label,
                    body,
                });
            }
            for (m, mo) in self.morphisms() {
                if let Operation::Print { newline } = mo.op {
                    if self.try_owner(mo.source) == Some(f) {
                        nodes.push(Node {
                            anchor: FlowAnchor::Print(m),
                            func: f,
                            id: format!("p{}m{print_k}", ctx.func_ix[&f]),
                            label: if newline {
                                "println".into()
                            } else {
                                "print".into()
                            },
                            body: None,
                        });
                        print_k += 1;
                    }
                }
            }
        }

        // A body's nodes are drawn only inside the site result that hosts it;
        // drop those of a body no drawn site hosts (a second site of a shared body).
        loop {
            let hosted: HashSet<FuncId> = nodes.iter().filter_map(|n| n.body).collect();
            let keep = |n: &Node| !ctx.is_body(n.func) || hosted.contains(&n.func);
            if nodes.iter().all(keep) {
                break;
            }
            nodes.retain(keep);
        }
        node_of = SecondaryMap::new();
        for (i, n) in nodes.iter().enumerate() {
            if let FlowAnchor::Object(o) = n.anchor {
                node_of.insert(o, i);
            }
        }

        // Pass 2: edges, walking back through plumbing from each node's reads.
        let mut g = FlowGraph::default();
        g.nodes = nodes.iter().map(|n| n.anchor).collect();
        let mut seen: std::collections::BTreeMap<(usize, usize, bool), Vec<MorphismId>> =
            std::collections::BTreeMap::new();
        for (i, n) in nodes.iter().enumerate() {
            type P = Vec<MorphismId>;
            let starts: Vec<(ObjectId, bool, P)> = match n.anchor {
                FlowAnchor::Print(m) => vec![(self.morphism(m).unwrap().source, false, vec![m])],
                FlowAnchor::Object(o) => {
                    if n.body.is_some() {
                        continue; // its operands are read inside the body
                    }
                    ctx.incoming[o]
                        .iter()
                        .map(|&m| {
                            let mo = self.morphism(m).unwrap();
                            (mo.source, mo.op == Operation::LoopBack, vec![m])
                        })
                        .collect()
                }
            };
            let mut visited: HashSet<(ObjectId, bool)> = HashSet::new();
            let mut stack: Vec<(ObjectId, bool, P)> = starts;
            stack.reverse();
            while let Some((y, back, path)) = stack.pop() {
                if !visited.insert((y, back)) {
                    continue;
                }
                let yo = self.object(y).unwrap();
                // The io token orders effects; it is not a value that is read.
                if yo.ty == Ty::IoToken || yo.kind == ObjectKind::Constant {
                    continue;
                }
                if let Some(&d) = node_of.get(y) {
                    seen.entry((d, i, back)).or_insert(path);
                    continue;
                }
                // An undrawn site result: its operands, not the body.
                let f = self.owner(y);
                if y == self.func(f).unwrap().input && ctx.is_body(f) {
                    for &site in ctx.sites.get(&f).into_iter().flatten() {
                        stack.push((self.morphism(site).unwrap().source, back, path.clone()));
                    }
                    continue;
                }
                for &m in ctx.incoming[y].iter().rev() {
                    let mo = self.morphism(m).unwrap();
                    let b = back || mo.op == Operation::LoopBack;
                    let mut np = path.clone();
                    np.push(m);
                    if mo.source == self.func(f).unwrap().input && ctx.is_body(f) {
                        let slot = match mo.op {
                            Operation::Proj { index } => Some(index),
                            _ => None,
                        };
                        for &site in ctx.sites.get(&f).into_iter().flatten() {
                            for s in ctx.site_operands(site, slot) {
                                stack.push((s, b, np.clone()));
                            }
                        }
                    } else {
                        stack.push((mo.source, b, np));
                    }
                }
                // y may itself be a body input read directly (no morphism).
            }
        }
        for (k, p) in seen {
            g.edges.push(k);
            g.paths.push(p);
        }
        (nodes, g)
    }

    /// The `Index` morphisms of `body` whose array is the body's input slot `slot`
    /// (read straight off the input projection).
    fn body_index_reads(&self, body: FuncId, slot: u32) -> Vec<MorphismId> {
        let Some(def) = self.func(body) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for (m, mo) in self.morphisms() {
            if mo.op != Operation::Index || self.try_owner(mo.source) != Some(body) {
                continue;
            }
            let arr = self.morphisms().find_map(|(_, p)| {
                (p.target == mo.source && matches!(p.op, Operation::Pair { slot: 0, .. }))
                    .then_some(p.source)
            });
            let reads = arr.is_some_and(|x| {
                self.morphisms().any(|(_, q)| {
                    q.target == x
                        && q.source == def.input
                        && q.op == (Operation::Proj { index: slot })
                })
            });
            if reads {
                out.push(m);
            }
        }
        out
    }

    pub(crate) fn objects_of(&self, f: FuncId) -> Vec<ObjectId> {
        self.objects()
            .filter(|&(o, _)| self.try_owner(o) == Some(f))
            .map(|(o, _)| o)
            .collect()
    }

    /// Whether `o` is a drawn value that is not a site result or a body result.
    fn plain_value(&self, ctx: &Ctx, o: ObjectId, obj: &crate::graph::Object) -> bool {
        if ty_contains_token(&obj.ty)
            || obj.kind == ObjectKind::Constant
            || obj.kind == ObjectKind::Return
        {
            return false;
        }
        let is_site = ctx.incoming[o].iter().any(|&m| {
            matches!(
                self.morphism(m).unwrap().op,
                Operation::Map { .. } | Operation::Fold { .. } | Operation::Call(_)
            )
        });
        is_site
            || ctx.read_of(o).is_some()
            || obj.kind == ObjectKind::LoopMerge
            || obj.name.is_some()
    }

    /// The file a `Read` names, when its path operand is a literal.
    pub(crate) fn read_file(&self, ctx: &Ctx, read: MorphismId) -> Option<String> {
        let src = self.morphism(read)?.source;
        for &p in &ctx.incoming[src] {
            let pm = self.morphism(p)?;
            if matches!(pm.op, Operation::Pair { slot: 1, .. }) {
                if let Some(Value::Str(s)) = &self.object(pm.source)?.value {
                    return Some(format!("\"{s}\""));
                }
            }
        }
        None
    }
}

fn nodes_ref<'a>(nodes: &'a [Node], _g: &FlowGraph, i: usize) -> &'a Node {
    &nodes[i]
}

fn edge_id(n: &Node) -> &str {
    &n.id
}

/// Draw the nodes of function `f` (a function or a body) at indent `ind`;
/// a site result with a drawn body becomes a subgraph holding the body's nodes.
fn draw(nodes: &[Node], f: FuncId, ind: usize, out: &mut String, declared: &mut HashSet<usize>) {
    let pad = "    ".repeat(ind);
    for (i, n) in nodes.iter().enumerate() {
        if n.func != f || declared.contains(&i) {
            continue;
        }
        declared.insert(i);
        match (&n.body, n.anchor) {
            (Some(b), _) => {
                out.push_str(&format!(
                    "{pad}subgraph {}[\"{}\"]\n",
                    n.id,
                    escape(&n.label)
                ));
                draw(nodes, *b, ind + 1, out, declared);
                out.push_str(&format!("{pad}end\n"));
            }
            (None, FlowAnchor::Print(_)) => {
                out.push_str(&format!("{pad}{}([\"{}\"])\n", n.id, escape(&n.label)));
            }
            (None, _) => {
                out.push_str(&format!("{pad}{}[\"{}\"]\n", n.id, escape(&n.label)));
            }
        }
    }
}
