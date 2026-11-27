//! The category view of the graph, drawn as the spec draws it (category-ir §4.1):
//! objects are nodes, morphisms are arrows labelled with their operation. A
//! product is made by ONE pairing arrow `⟨a, b⟩` from the function's context
//! `Γ`; the multi-input operation then leaves the product object as one arrow,
//! and a projection out of a product is an arrow `π₁` … No arrow ends in a
//! product except a pairing from `Γ` (the stored graph's `Pair i/n` in-edges
//! are how the pairing is kept, not morphisms of a cartesian category).
//!
//! [`CategoryIr::to_mermaid_category`] draws every object;
//! [`CategoryIr::to_mermaid_category_condensed`] only the objects the source
//! named, with each collapsed chain one arrow labelled with its steps.

use std::collections::{HashMap, HashSet};

use crate::graph::{
    CategoryIr, FuncId, FuncKind, MorphismId, Object, ObjectId, ObjectKind, Operation,
};
use crate::mermaid::{edge_label, node_label, value_label};
use crate::mermaid_diagram::{Diagram, EdgeKind, NodeKind};
use crate::mermaid_flow::{FlowAnchor, Node, is_plumbing, step_word};
use crate::ty::ty_label;

/// How an object is named inside a pairing label: the constant's value, the
/// name, else the type; the function's input is `Γ`.
pub(crate) fn component_name(ir: &CategoryIr, o: ObjectId) -> String {
    let obj = ir.object(o).unwrap();
    if ir
        .try_owner(o)
        .and_then(|f| ir.func(f))
        .is_some_and(|d| d.input == o)
    {
        return "Γ".into();
    }
    object_short(obj)
}

pub(crate) fn object_short(obj: &Object) -> String {
    match (&obj.value, obj.kind, &obj.name) {
        (Some(v), ObjectKind::Constant, _) => value_label(v),
        (_, _, Some(n)) => n.clone(),
        _ => ty_label(&obj.ty),
    }
}

/// The steps that make `o` when it stands alone (no drawn source): the step
/// words of the chain behind it, in dataflow order, at most 3.
pub(crate) fn chain_words(ir: &CategoryIr, ctx: &crate::mermaid_flow::Ctx, o: ObjectId) -> String {
    let mut words: Vec<String> = Vec::new();
    let mut stack = vec![o];
    let mut seen = HashSet::new();
    while let Some(y) = stack.pop() {
        if !seen.insert(y) {
            continue;
        }
        for &m in &ctx.incoming[y] {
            let mo = ir.morphism(m).unwrap();
            if !is_plumbing(&mo.op) {
                let w = step_word(&mo.op);
                if words.last() != Some(&w) {
                    words.push(w);
                }
            }
            let s = ir.object(mo.source).unwrap();
            if s.kind != ObjectKind::Constant && (s.name.is_none() || y == o) {
                stack.push(mo.source);
            }
        }
    }
    words.reverse();
    let from = words.len().saturating_sub(3);
    if words.is_empty() {
        "operand".into()
    } else {
        words[from..].join(" · ")
    }
}

fn site_word(op: &Operation) -> &'static str {
    if matches!(op, Operation::Map { .. }) {
        "map"
    } else {
        "fold"
    }
}

impl CategoryIr {
    /// Every object as a node and every morphism as an arrow, commutative-diagram
    /// style (see the module docs). Deterministic; passes [`crate::lint_mermaid`].
    pub fn to_mermaid_category(&self) -> String {
        self.category_diagram().render()
    }

    /// The named objects only; a collapsed chain is one arrow labelled with its steps.
    pub fn to_mermaid_category_condensed(&self) -> String {
        self.category_condensed_diagram(None).render()
    }

    /// [`Self::to_mermaid_category_condensed`] with source text labelling the arrows.
    pub fn to_mermaid_category_condensed_src(&self, src: &str) -> String {
        self.category_condensed_diagram(Some(src)).render()
    }

    /// The full category view as data.
    pub fn category_diagram(&self) -> Diagram {
        let ctx = self.flow_ctx();
        let mut d = Diagram::default();
        let mut sub_of: HashMap<FuncId, usize> = HashMap::new();
        for (f, def) in self.funcs() {
            if def.kind == FuncKind::Named {
                let s = d.sub(
                    format!("f{}", ctx.func_ix[&f]),
                    format!("fn {}", def.name),
                    0,
                    None,
                );
                sub_of.insert(f, s);
            }
        }
        // A body is drawn inside the function that holds its (first) site.
        loop {
            let mut progress = false;
            let mut pending = false;
            for (f, def) in self.funcs() {
                if sub_of.contains_key(&f) {
                    continue;
                }
                let site = ctx.sites.get(&f).and_then(|v| v.first()).copied();
                let (parent, label) = match site {
                    Some(m) => {
                        let mo = self.morphism(m).unwrap();
                        let Some(&p) = self.try_owner(mo.source).and_then(|g| sub_of.get(&g))
                        else {
                            pending = true;
                            continue;
                        };
                        let t = self.object(mo.target).unwrap();
                        let l = match &t.name {
                            Some(n) => format!("{} body of {n}", site_word(&mo.op)),
                            None => format!("{} body", site_word(&mo.op)),
                        };
                        (p, l)
                    }
                    None => (0, format!("fn {}", def.name)),
                };
                let s = d.sub(format!("f{}", ctx.func_ix[&f]), label, parent, None);
                sub_of.insert(f, s);
                progress = true;
            }
            if !pending || !progress {
                break;
            }
        }

        let id = |o: ObjectId| format!("f{}o{}", ctx.func_ix[&self.owner(o)], ctx.ord[o]);
        let mut pairs: HashMap<ObjectId, Vec<(u32, ObjectId)>> = HashMap::new();
        for (_, m) in self.morphisms() {
            if let Operation::Pair { slot, .. } = m.op {
                pairs.entry(m.target).or_default().push((slot, m.source));
            }
        }
        for v in pairs.values_mut() {
            v.sort();
        }
        for (o, obj) in self.objects() {
            let Some(f) = self.try_owner(o) else { continue };
            let sub = sub_of[&f];
            let (label, kind) = if self.func(f).unwrap().input == o {
                (format!("Γ = {}", ty_label(&obj.ty)), NodeKind::Context)
            } else if let Some(p) = pairs.get(&o) {
                let l: Vec<String> = p
                    .iter()
                    .map(|&(_, s)| ty_label(&self.object(s).unwrap().ty))
                    .collect();
                (l.join(" × "), NodeKind::Product)
            } else {
                (node_label(obj), NodeKind::Object)
            };
            let n = d.node(id(o), label, kind, sub);
            d.nodes[n].obj = Some(o);
        }
        let mut paired: HashSet<ObjectId> = HashSet::new();
        for (mid, m) in self.morphisms() {
            if self.try_owner(m.source).is_none() {
                continue;
            }
            match m.op {
                Operation::Pair { .. } => {
                    if paired.insert(m.target) {
                        let f = self.owner(m.target);
                        let names: Vec<String> = pairs[&m.target]
                            .iter()
                            .map(|&(_, s)| component_name(self, s))
                            .collect();
                        let e = d.edge(
                            &id(self.func(f).unwrap().input),
                            &id(m.target),
                            format!("⟨{}⟩", names.join(", ")),
                            EdgeKind::Pairing,
                        );
                        d.edges[e].obj = Some(m.target);
                    }
                }
                Operation::Proj { index } => {
                    let e = d.edge(
                        &id(m.source),
                        &id(m.target),
                        format!("π{}", sub_digits(index + 1)),
                        EdgeKind::Projection,
                    );
                    d.edges[e].morph = Some(mid);
                }
                op => {
                    let e = d.edge(
                        &id(m.source),
                        &id(m.target),
                        edge_label(&op),
                        EdgeKind::Morphism,
                    );
                    d.edges[e].morph = Some(mid);
                }
            }
        }
        d
    }

    /// The condensed category view as data.
    pub fn category_condensed_diagram(&self, src: Option<&str>) -> Diagram {
        let (nodes, g) = self.build_flow();
        let ctx = self.flow_ctx();
        let mut d = Diagram::default();
        let named: Vec<FuncId> = self
            .funcs()
            .filter(|(_, x)| x.kind == FuncKind::Named)
            .map(|(f, _)| f)
            .collect();
        let wrap = named.len() > 1;
        let mut sub_of: HashMap<FuncId, usize> = HashMap::new();
        for &f in &named {
            if wrap && nodes.iter().any(|n| n.func == f) {
                let s = d.sub(
                    format!("f{}", ctx.func_ix[&f]),
                    format!("fn {}", self.func(f).unwrap().name),
                    0,
                    None,
                );
                sub_of.insert(f, s);
            } else {
                sub_of.insert(f, 0);
            }
        }
        // The site node of each drawn body (its binding names the subgraph).
        let host: HashMap<FuncId, usize> = nodes
            .iter()
            .enumerate()
            .filter_map(|(i, n)| n.body.map(|b| (b, i)))
            .collect();
        let site_op = |i: usize| -> Option<(MorphismId, Operation)> {
            let FlowAnchor::Object(o) = nodes[i].anchor else {
                return None;
            };
            ctx.incoming[o].iter().find_map(|&m| {
                let op = self.morphism(m).unwrap().op;
                matches!(op, Operation::Map { .. } | Operation::Fold { .. }).then_some((m, op))
            })
        };
        loop {
            let mut progress = false;
            let mut hosts: Vec<(FuncId, usize)> = host.iter().map(|(&b, &i)| (b, i)).collect();
            hosts.sort_by_key(|&(_, i)| i);
            for (b, i) in hosts {
                if sub_of.contains_key(&b) {
                    continue;
                }
                let Some(&p) = sub_of.get(&nodes[i].func) else {
                    continue;
                };
                let word = site_op(i).map_or("map", |(_, op)| site_word(&op));
                let FlowAnchor::Object(o) = nodes[i].anchor else {
                    continue;
                };
                let label = match &self.object(o).unwrap().name {
                    Some(n) => format!("{word} body of {n}"),
                    None => format!("{word} body"),
                };
                let s = d.sub(format!("f{}", ctx.func_ix[&b]), label, p, None);
                sub_of.insert(b, s);
                progress = true;
            }
            if !progress {
                break;
            }
        }
        let nid = |n: &Node| match n.anchor {
            FlowAnchor::Object(o) => format!("f{}o{}", ctx.func_ix[&n.func], ctx.ord[o]),
            FlowAnchor::Print(_) => n.id.clone(),
        };
        let nname = |n: &Node| match n.anchor {
            FlowAnchor::Object(o) => object_short(self.object(o).unwrap()),
            FlowAnchor::Print(_) => n.label.clone(),
        };
        let mut gammas: HashMap<FuncId, String> = HashMap::new();
        let mut gamma = |d: &mut Diagram, f: FuncId| -> String {
            gammas
                .entry(f)
                .or_insert_with(|| {
                    let def = self.func(f).unwrap();
                    let gid = format!("g{}", ctx.func_ix[&f]);
                    let l = format!("Γ = {}", ty_label(&self.object(def.input).unwrap().ty));
                    let s = sub_of.get(&f).copied().unwrap_or(0);
                    let n = d.node(gid.clone(), l, NodeKind::Context, s);
                    d.nodes[n].obj = Some(def.input);
                    gid
                })
                .clone()
        };

        for n in &nodes {
            let sub = sub_of.get(&n.func).copied().unwrap_or(0);
            let (label, obj) = match n.anchor {
                FlowAnchor::Object(o) => (node_label(self.object(o).unwrap()), Some(o)),
                FlowAnchor::Print(_) => (n.label.clone(), None),
            };
            let k = d.node(nid(n), label, NodeKind::Object, sub);
            d.nodes[k].obj = obj;
        }

        let mut incoming: Vec<Vec<(usize, bool, String)>> = vec![Vec::new(); nodes.len()];
        for (k, &(a, b, back)) in g.edges.iter().enumerate() {
            incoming[b].push((a, back, g.edge_label(self, k, src)));
        }
        for (i, n) in nodes.iter().enumerate() {
            let target = nid(n);
            let sub_f = n.func;
            if let Some(b) = n.body {
                // The site: its operands (what the body reads from outside) → the result.
                let mut ops: Vec<usize> = Vec::new();
                for &(a, t, _) in &g.edges {
                    if nodes[t].func == b && nodes[a].func != b && !ops.contains(&a) {
                        ops.push(a);
                    }
                }
                let label = site_op(i).map_or("map".to_string(), |(_, op)| edge_label(&op));
                match ops.len() {
                    0 => {
                        let gid = gamma(&mut d, sub_f);
                        d.edge(&gid, &target, label, EdgeKind::Morphism);
                    }
                    1 => {
                        d.edge(&nid(&nodes[ops[0]]), &target, label, EdgeKind::Morphism);
                    }
                    _ => {
                        let gid = gamma(&mut d, sub_f);
                        let pid = format!("pr{i}");
                        let names: Vec<String> = ops.iter().map(|&a| nname(&nodes[a])).collect();
                        let tys: Vec<String> = ops
                            .iter()
                            .map(|&a| match nodes[a].anchor {
                                FlowAnchor::Object(o) => ty_label(&self.object(o).unwrap().ty),
                                _ => "io".into(),
                            })
                            .collect();
                        let sub = sub_of.get(&sub_f).copied().unwrap_or(0);
                        let p = d.node(pid.clone(), tys.join(" × "), NodeKind::Product, sub);
                        d.nodes[p].obj = match n.anchor {
                            FlowAnchor::Object(o) => Some(o),
                            _ => None,
                        };
                        d.edge(
                            &gid,
                            &pid,
                            format!("⟨{}⟩", names.join(", ")),
                            EdgeKind::Pairing,
                        );
                        d.edge(&pid, &target, label, EdgeKind::Morphism);
                    }
                }
                continue;
            }
            let inc = &incoming[i];
            if inc.is_empty() {
                let FlowAnchor::Object(o) = n.anchor else {
                    continue;
                };
                if ctx.incoming[o].is_empty() {
                    continue;
                }
                let label = match ctx.read_of(o) {
                    Some(r) => match self.read_file(&ctx, r) {
                        Some(p) => format!("read {p}"),
                        None => "read".into(),
                    },
                    None => chain_words(self, &ctx, o),
                };
                let gid = gamma(&mut d, sub_f);
                d.edge(&gid, &target, label, EdgeKind::Morphism);
                continue;
            }
            // Group the sources of this node, a body reading the outside through its Γ.
            let mut srcs: Vec<(String, String, String, String)> = Vec::new(); // id, name, ty, label
            for (a, back, l) in inc {
                let an = &nodes[*a];
                let (sid, sname, sty) = if an.func != n.func && ctx.is_body(n.func) {
                    let gid = gamma(&mut d, n.func);
                    (gid, "Γ".to_string(), "Γ".to_string())
                } else {
                    let ty = match an.anchor {
                        FlowAnchor::Object(o) => ty_label(&self.object(o).unwrap().ty),
                        _ => "io".into(),
                    };
                    (nid(an), nname(an), ty)
                };
                let l = if *back { format!("{l} ↩") } else { l.clone() };
                match srcs.iter_mut().find(|s| s.0 == sid) {
                    Some(s) => {
                        if !s.3.split(" ; ").any(|x| x == l) {
                            s.3 = format!("{} ; {l}", s.3);
                        }
                    }
                    None => srcs.push((sid, sname, sty, l)),
                }
            }
            if srcs.len() == 1 {
                d.edge(&srcs[0].0, &target, srcs[0].3.clone(), EdgeKind::Morphism);
            } else {
                let gid = gamma(&mut d, sub_f);
                let pid = format!("pr{i}");
                let sub = sub_of.get(&sub_f).copied().unwrap_or(0);
                let tys: Vec<&str> = srcs.iter().map(|s| s.2.as_str()).collect();
                let p = d.node(pid.clone(), tys.join(" × "), NodeKind::Product, sub);
                d.nodes[p].obj = match n.anchor {
                    FlowAnchor::Object(o) => Some(o),
                    _ => None,
                };
                let names: Vec<&str> = srcs.iter().map(|s| s.1.as_str()).collect();
                d.edge(
                    &gid,
                    &pid,
                    format!("⟨{}⟩", names.join(", ")),
                    EdgeKind::Pairing,
                );
                let labels: Vec<&str> = srcs.iter().map(|s| s.3.as_str()).collect();
                let mut uniq: Vec<&str> = Vec::new();
                for l in labels.iter().flat_map(|l| l.split(" ; ")) {
                    if !uniq.contains(&l) {
                        uniq.push(l);
                    }
                }
                d.edge(&pid, &target, uniq.join(" ; "), EdgeKind::Morphism);
            }
        }
        d
    }
}

/// Subscript digits for `π₁`.
pub(crate) fn sub_digits(n: u32) -> String {
    n.to_string()
        .chars()
        .map(|c| char::from_u32('₀' as u32 + c.to_digit(10).unwrap()).unwrap())
        .collect()
}
