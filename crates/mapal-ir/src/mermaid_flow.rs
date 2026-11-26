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

use std::collections::{BTreeSet, HashMap, HashSet};

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
}

struct Node {
    anchor: FlowAnchor,
    func: FuncId,
    id: String,
    label: String,
    /// The body drawn inside this node (a subgraph) when it is a site result.
    body: Option<FuncId>,
}

struct Ctx<'a> {
    ir: &'a CategoryIr,
    func_ix: HashMap<FuncId, usize>,
    /// Object ordinal within its function (the full dump's `f{i}o{j}`).
    ord: SecondaryMap<ObjectId, usize>,
    incoming: SecondaryMap<ObjectId, Vec<MorphismId>>,
    /// Body function -> the `Map` / `Fold` morphisms that use it.
    sites: HashMap<FuncId, Vec<MorphismId>>,
}

impl Ctx<'_> {
    fn is_body(&self, f: FuncId) -> bool {
        self.ir
            .func(f)
            .is_some_and(|d| d.kind != FuncKind::Named)
    }

    fn site_body(op: &Operation) -> Option<FuncId> {
        match op {
            Operation::Map { body, .. } | Operation::Fold { body, .. } => Some(*body),
            _ => None,
        }
    }

    /// The incoming `Read` of a file-read array: `Proj 1` of a `Read` result.
    fn read_of(&self, o: ObjectId) -> Option<MorphismId> {
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
        for &(a, b, back) in &g.edges {
            let (a, b) = (&nodes_ref(&nodes, &g, a), &nodes_ref(&nodes, &g, b));
            if back {
                out.push_str(&format!("    {} -- \"loop ↩\" --> {}\n", edge_id(a), edge_id(b)));
            } else {
                out.push_str(&format!("    {} --> {}\n", edge_id(a), edge_id(b)));
            }
        }
        out
    }

    fn flow_ctx(&self) -> Ctx<'_> {
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

    fn build_flow(&self) -> (Vec<Node>, FlowGraph) {
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
                    (format!("⟲ {}: {ty}", name.as_deref().unwrap_or("state")), None)
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
                            label: if newline { "println".into() } else { "print".into() },
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
        let mut seen: BTreeSet<(usize, usize, bool)> = BTreeSet::new();
        for (i, n) in nodes.iter().enumerate() {
            let starts: Vec<(ObjectId, bool)> = match n.anchor {
                FlowAnchor::Print(m) => vec![(self.morphism(m).unwrap().source, false)],
                FlowAnchor::Object(o) => {
                    if n.body.is_some() {
                        continue; // its operands are read inside the body
                    }
                    ctx.incoming[o]
                        .iter()
                        .map(|&m| {
                            let mo = self.morphism(m).unwrap();
                            (mo.source, mo.op == Operation::LoopBack)
                        })
                        .collect()
                }
            };
            let mut visited: HashSet<(ObjectId, bool)> = HashSet::new();
            let mut stack: Vec<(ObjectId, bool)> = starts;
            stack.reverse();
            while let Some((y, back)) = stack.pop() {
                if !visited.insert((y, back)) {
                    continue;
                }
                let yo = self.object(y).unwrap();
                // The io token orders effects; it is not a value that is read.
                if yo.ty == Ty::IoToken || yo.kind == ObjectKind::Constant {
                    continue;
                }
                if let Some(&d) = node_of.get(y) {
                    seen.insert((d, i, back));
                    continue;
                }
                // An undrawn site result: its operands, not the body.
                let f = self.owner(y);
                if y == self.func(f).unwrap().input && ctx.is_body(f) {
                    for &site in ctx.sites.get(&f).into_iter().flatten() {
                        stack.push((self.morphism(site).unwrap().source, back));
                    }
                    continue;
                }
                for &m in ctx.incoming[y].iter().rev() {
                    let mo = self.morphism(m).unwrap();
                    let b = back || mo.op == Operation::LoopBack;
                    if mo.source == self.func(f).unwrap().input && ctx.is_body(f) {
                        let slot = match mo.op {
                            Operation::Proj { index } => Some(index),
                            _ => None,
                        };
                        for &site in ctx.sites.get(&f).into_iter().flatten() {
                            for s in ctx.site_operands(site, slot) {
                                stack.push((s, b));
                            }
                        }
                    } else {
                        stack.push((mo.source, b));
                    }
                }
                // y may itself be a body input read directly (no morphism).
            }
        }
        g.edges = seen.into_iter().collect();
        (nodes, g)
    }

    fn objects_of(&self, f: FuncId) -> Vec<ObjectId> {
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
    fn read_file(&self, ctx: &Ctx, read: MorphismId) -> Option<String> {
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
                out.push_str(&format!("{pad}subgraph {}[\"{}\"]\n", n.id, escape(&n.label)));
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
