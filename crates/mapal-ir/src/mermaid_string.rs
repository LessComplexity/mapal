//! The string-diagram view of the graph: the standard notation for categories
//! with products. Boxes are morphisms (the steps), wires are objects. A product
//! is only several wires entering one box, their slot order marked `①②`; there
//! are no pair nodes or arrows. A projection `π` takes one wire out of a bundle.
//! One value read by several steps is copied (the diagonal): a small `Δ` node
//! labelled with the value, from which a wire goes to each reader. Program
//! inputs, constants and outputs are open wire ends (round terminals); a `map` /
//! `fold` box is a subgraph holding its body's diagram.
//!
//! [`CategoryIr::to_mermaid_string`] draws every morphism; the condensed form
//! draws the named values as wires and each collapsed chain as one box labelled
//! with its steps.

use std::collections::HashMap;

use crate::graph::{CategoryIr, FuncId, FuncKind, MorphismId, ObjectId, ObjectKind, Operation};
use crate::mermaid::{node_label, value_label};
use crate::mermaid_cat::{chain_words, object_short, sub_digits};
use crate::mermaid_diagram::{Diagram, EdgeKind, NodeKind};
use crate::mermaid_flow::{FlowAnchor, step_word};

/// `①` … `⑳`, then `(n)`; a nested slot path concatenates.
fn marker(path: &[u32]) -> String {
    path.iter()
        .map(|&s| match s + 1 {
            n @ 1..=20 => char::from_u32('①' as u32 + n - 1).unwrap().to_string(),
            n => format!("({n})"),
        })
        .collect()
}

fn with_marker(path: &[u32], l: &str) -> String {
    if path.is_empty() { l.to_string() } else { format!("{} {l}", marker(path)) }
}

impl CategoryIr {
    /// Every morphism a box and every object a wire (see the module docs).
    /// Deterministic; passes [`crate::lint_mermaid`].
    pub fn to_mermaid_string(&self) -> String {
        self.string_diagram().render()
    }

    /// The named values as wires; a collapsed chain is one box labelled with its steps.
    pub fn to_mermaid_string_condensed(&self) -> String {
        self.string_condensed_diagram(None).render()
    }

    /// [`Self::to_mermaid_string_condensed`] with source text labelling the boxes.
    pub fn to_mermaid_string_condensed_src(&self, src: &str) -> String {
        self.string_condensed_diagram(Some(src)).render()
    }

    /// The label of a step's box.
    pub(crate) fn box_label(&self, op: &Operation) -> String {
        match op {
            Operation::Proj { index } => format!("π{}", sub_digits(index + 1)),
            Operation::Print { newline: true } => "println".into(),
            Operation::Print { newline: false } => "print".into(),
            Operation::LoopEnter => "⟲ enter".into(),
            Operation::LoopBack => "↩ back".into(),
            Operation::LoopExit => "exit".into(),
            Operation::Call(c) => format!("call {}", self.func(*c).map_or("?".into(), |d| d.name.clone())),
            Operation::Map { captures: 0, .. } => "map over cells".into(),
            Operation::Map { captures: k, .. } => format!("map over cells (+{k} captured)"),
            Operation::Fold { captures: k, .. } if *k > 0 => format!("fold (+{k} captured)"),
            op => step_word(op),
        }
    }

    /// The label of the wire a body step reads from the body's own input: the binding's
    /// name (else `cell`, the element; `captured n`) and its type, found by walking back
    /// from `o` to a projection of the body's input.
    pub(crate) fn body_input_wire(&self, ctx: &crate::mermaid_flow::Ctx, body: FuncId, o: ObjectId) -> Option<String> {
        let input = self.func(body)?.input;
        let mut stack = vec![o];
        let mut seen = std::collections::HashSet::new();
        while let Some(y) = stack.pop() {
            if !seen.insert(y) {
                continue;
            }
            for &m in &ctx.incoming[y] {
                let mo = self.morphism(m)?;
                if let Operation::Proj { index } = mo.op {
                    if mo.source == input {
                        let x = self.object(y)?;
                        let arity = match &self.object(input)?.ty {
                            crate::ty::Ty::Tuple(v) => v.len() as u32,
                            _ => 1,
                        };
                        let name = x.name.clone().unwrap_or_else(|| {
                            if index + 1 == arity { "cell".into() } else { format!("captured {}", index + 1) }
                        });
                        return Some(format!("{name}: {}", crate::ty::ty_label(&x.ty)));
                    }
                }
                if self.object(mo.source)?.kind != ObjectKind::Constant {
                    stack.push(mo.source);
                }
            }
        }
        None
    }

    /// The full string diagram as data.
    pub fn string_diagram(&self) -> Diagram {
        let ctx = self.flow_ctx();
        let mut d = Diagram::default();
        let ord_m: HashMap<MorphismId, usize> =
            self.morphisms().enumerate().map(|(k, (m, _))| (m, k)).collect();
        let mut claimed: HashMap<FuncId, MorphismId> = HashMap::new();
        for (m, mo) in self.morphisms() {
            if let Operation::Map { body, .. } | Operation::Fold { body, .. } = mo.op {
                claimed.entry(body).or_insert(m);
            }
        }
        let mut sub_of: HashMap<FuncId, usize> = HashMap::new();
        for (f, def) in self.funcs() {
            if def.kind == FuncKind::Named || !claimed.contains_key(&f) {
                let s = d.sub(format!("f{}", ctx.func_ix[&f]), format!("fn {}", def.name), 0, None);
                sub_of.insert(f, s);
            }
        }
        loop {
            let mut progress = false;
            let mut pending = false;
            for (f, _) in self.funcs() {
                if sub_of.contains_key(&f) {
                    continue;
                }
                let m = claimed[&f];
                let mo = self.morphism(m).unwrap();
                let Some(&p) = self.try_owner(mo.source).and_then(|g| sub_of.get(&g)) else {
                    pending = true;
                    continue;
                };
                let s = d.sub(format!("m{}", ord_m[&m]), self.box_label(&mo.op), p, Some(m));
                sub_of.insert(f, s);
                progress = true;
            }
            if !pending || !progress {
                break;
            }
        }
        let is_site_sub = |m: MorphismId| -> bool {
            matches!(self.morphism(m).unwrap().op, Operation::Map { body, .. } | Operation::Fold { body, .. } if claimed[&body] == m)
        };
        let box_id = |m: MorphismId| format!("m{}", ord_m[&m]);

        // Boxes: every morphism but the pairing (wire grouping) and the identity.
        for (m, mo) in self.morphisms() {
            if matches!(mo.op, Operation::Pair { .. } | Operation::Output) || is_site_sub(m) {
                continue;
            }
            let Some(f) = self.try_owner(mo.source) else { continue };
            let n = d.node(box_id(m), self.box_label(&mo.op), NodeKind::Box, sub_of[&f]);
            d.nodes[n].morph = Some(m);
        }
        // Terminals of each function: its input end and its output end.
        for (f, def) in self.funcs() {
            let fi = ctx.func_ix[&f];
            let n = d.node(format!("ti{fi}"), "in".into(), NodeKind::Terminal, sub_of[&f]);
            d.nodes[n].obj = Some(def.input);
            let n = d.node(format!("to{fi}"), "ret".into(), NodeKind::Terminal, sub_of[&f]);
            d.nodes[n].obj = Some(def.output);
        }

        // Readers of an object: the boxes (or the output end) its wire reaches,
        // through the pairings it is a component of, with the slot path.
        fn readers(
            ir: &CategoryIr,
            ctx: &crate::mermaid_flow::Ctx,
            o: ObjectId,
            box_id: &dyn Fn(MorphismId) -> String,
            out: &mut Vec<(String, Vec<u32>)>,
        ) {
            for &m in ir.out_edges(o) {
                let mo = ir.morphism(m).unwrap();
                match mo.op {
                    Operation::Pair { slot, .. } => {
                        let mut inner = Vec::new();
                        readers(ir, ctx, mo.target, box_id, &mut inner);
                        for (r, mut p) in inner {
                            p.insert(0, slot);
                            out.push((r, p));
                        }
                    }
                    Operation::Output => {
                        let f = ir.owner(mo.target);
                        out.push((format!("to{}", ctx.func_ix[&f]), Vec::new()));
                    }
                    _ => out.push((box_id(m), Vec::new())),
                }
            }
        }

        for (o, obj) in self.objects() {
            let Some(f) = self.try_owner(o) else { continue };
            let def = self.func(f).unwrap();
            let fi = ctx.func_ix[&f];
            let oi = ctx.ord[o];
            let sub = sub_of[&f];
            let inc = &ctx.incoming[o];
            if inc.iter().any(|&m| matches!(self.morphism(m).unwrap().op, Operation::Pair { .. })) {
                continue; // a product: its components' wires enter the reading box
            }
            if inc.iter().any(|&m| self.morphism(m).unwrap().op == Operation::Output) {
                continue; // the identity: the output end continues the source's wire
            }
            let producers: Vec<MorphismId> = inc.to_vec();
            let wl = if obj.kind == ObjectKind::Constant {
                obj.value.as_ref().map_or("·".into(), value_label)
            } else {
                node_label(obj)
            };
            let short = object_short(obj);
            let src = if o == def.input {
                format!("ti{fi}")
            } else if obj.kind == ObjectKind::Constant || producers.is_empty() {
                let id = format!("t{fi}o{oi}");
                let n = d.node(id.clone(), "•".into(), NodeKind::Terminal, sub);
                d.nodes[n].obj = Some(o);
                id
            } else if producers.len() == 1 {
                box_id(producers[0])
            } else {
                let id = format!("j{fi}o{oi}");
                let n = d.node(id.clone(), format!("⟲ {short}"), NodeKind::Merge, sub);
                d.nodes[n].obj = Some(o);
                for &m in &producers {
                    let e = d.edge(&box_id(m), &id, wl.clone(), EdgeKind::Branch);
                    d.edges[e].obj = Some(o);
                }
                id
            };
            let mut rs: Vec<(String, Vec<u32>)> = Vec::new();
            readers(self, &ctx, o, &box_id, &mut rs);
            if o == def.output {
                rs.push((format!("to{fi}"), Vec::new()));
            }
            if rs.is_empty() {
                let id = format!("x{fi}o{oi}");
                d.node(id.clone(), "∅".into(), NodeKind::Terminal, sub);
                rs.push((id, Vec::new()));
            }
            if rs.len() == 1 {
                let e = d.edge(&src, &rs[0].0, with_marker(&rs[0].1, &wl), EdgeKind::Wire);
                d.edges[e].obj = Some(o);
            } else {
                let id = format!("d{fi}o{oi}");
                let n = d.node(id.clone(), format!("Δ {short}"), NodeKind::Split, sub);
                d.nodes[n].obj = Some(o);
                let e = d.edge(&src, &id, wl.clone(), EdgeKind::Wire);
                d.edges[e].obj = Some(o);
                for (r, p) in &rs {
                    let e = d.edge(&id, r, with_marker(p, &short), EdgeKind::Branch);
                    d.edges[e].obj = Some(o);
                }
            }
        }
        d
    }

    /// The condensed string diagram as data.
    pub fn string_condensed_diagram(&self, src: Option<&str>) -> Diagram {
        let (nodes, g) = self.build_flow();
        let ctx = self.flow_ctx();
        let mut d = Diagram::default();
        let named: Vec<FuncId> =
            self.funcs().filter(|(_, x)| x.kind == FuncKind::Named).map(|(f, _)| f).collect();
        let wrap = named.len() > 1;
        let mut sub_of: HashMap<FuncId, usize> = HashMap::new();
        for &f in &named {
            if wrap && nodes.iter().any(|n| n.func == f) {
                let s = d.sub(format!("f{}", ctx.func_ix[&f]), format!("fn {}", self.func(f).unwrap().name), 0, None);
                sub_of.insert(f, s);
            } else {
                sub_of.insert(f, 0);
            }
        }
        let host: HashMap<FuncId, usize> =
            nodes.iter().enumerate().filter_map(|(i, n)| n.body.map(|b| (b, i))).collect();
        let nid = |i: usize| match nodes[i].anchor {
            FlowAnchor::Object(o) => format!("f{}o{}", ctx.func_ix[&nodes[i].func], ctx.ord[o]),
            FlowAnchor::Print(_) => nodes[i].id.clone(),
        };
        // `map over cells (uses a, b, along)`: the named values the body reads from outside.
        let site_label = |i: usize| -> String {
            let FlowAnchor::Object(o) = nodes[i].anchor else { return "map over cells".into() };
            let word = ctx.incoming[o]
                .iter()
                .find_map(|&m| match self.morphism(m).unwrap().op {
                    Operation::Map { .. } => Some("map over cells"),
                    Operation::Fold { .. } => Some("fold"),
                    _ => None,
                })
                .unwrap_or("map over cells");
            let Some(b) = nodes[i].body else { return word.into() };
            let mut uses: Vec<String> = Vec::new();
            for &(a, t, _) in &g.edges {
                if nodes[t].func == b && nodes[a].func != b {
                    if let FlowAnchor::Object(ao) = nodes[a].anchor {
                        let n = object_short(self.object(ao).unwrap());
                        if !uses.contains(&n) {
                            uses.push(n);
                        }
                    }
                }
            }
            if uses.is_empty() { word.into() } else { format!("{word} (uses {})", uses.join(", ")) }
        };
        loop {
            let mut progress = false;
            let mut hosts: Vec<(FuncId, usize)> = host.iter().map(|(&b, &i)| (b, i)).collect();
            hosts.sort_by_key(|&(_, i)| i);
            for (b, i) in hosts {
                if sub_of.contains_key(&b) {
                    continue;
                }
                let Some(&p) = sub_of.get(&nodes[i].func) else { continue };
                let s = d.sub(nid(i), site_label(i), p, None);
                sub_of.insert(b, s);
                progress = true;
            }
            if !progress {
                break;
            }
        }
        let is_ret_end = |i: usize| match nodes[i].anchor {
            FlowAnchor::Object(o) => {
                let obj = self.object(o).unwrap();
                obj.kind == ObjectKind::Return && !ctx.is_body(nodes[i].func)
            }
            _ => false,
        };

        // Sources of each node: distinct, in node order, labels merged.
        let mut srcs: Vec<Vec<(usize, bool, String)>> = vec![Vec::new(); nodes.len()];
        for (k, &(a, b, back)) in g.edges.iter().enumerate() {
            let l = g.edge_label(self, k, src);
            match srcs[b].iter_mut().find(|s| s.0 == a && s.1 == back) {
                Some(s) => {
                    if !s.2.split(" ; ").any(|x| x == l) {
                        s.2 = format!("{} ; {l}", s.2);
                    }
                }
                None => srcs[b].push((a, back, l)),
            }
        }
        // Each node's box (or terminal), and where its wire starts.
        let mut wire_from: Vec<Option<String>> = vec![None; nodes.len()];
        let mut enters: Vec<Option<String>> = vec![None; nodes.len()];
        for (i, n) in nodes.iter().enumerate() {
            let sub = sub_of.get(&n.func).copied().unwrap_or(0);
            match n.anchor {
                FlowAnchor::Print(_) => {
                    let ls: Vec<&str> = {
                        let mut v: Vec<&str> = Vec::new();
                        for s in &srcs[i] {
                            if !v.contains(&s.2.as_str()) {
                                v.push(&s.2);
                            }
                        }
                        v
                    };
                    let label = if ls.is_empty() { n.label.clone() } else { format!("{} · {}", ls.join(" ; "), n.label) };
                    let k = d.node(nid(i), label, NodeKind::Box, sub);
                    if let FlowAnchor::Print(m) = n.anchor {
                        d.nodes[k].morph = Some(m);
                    }
                    wire_from[i] = Some(nid(i));
                    enters[i] = Some(nid(i));
                }
                FlowAnchor::Object(o) => {
                    if n.body.is_some() {
                        wire_from[i] = Some(nid(i));
                        // the operands enter the body's own boxes
                        continue;
                    }
                    if is_ret_end(i) {
                        let k = d.node(format!("r{i}"), "ret".into(), NodeKind::Terminal, sub);
                        d.nodes[k].obj = Some(o);
                        enters[i] = Some(format!("r{i}"));
                        continue;
                    }
                    let label = if !srcs[i].is_empty() {
                        let mut v: Vec<&str> = Vec::new();
                        for s in &srcs[i] {
                            if !v.contains(&s.2.as_str()) {
                                v.push(&s.2);
                            }
                        }
                        Some(v.join(" ; "))
                    } else if ctx.incoming[o].is_empty() {
                        None
                    } else {
                        Some(match ctx.read_of(o) {
                            Some(r) => match self.read_file(&ctx, r) {
                                Some(p) => format!("read {p}"),
                                None => "read".into(),
                            },
                            None => chain_words(self, &ctx, o),
                        })
                    };
                    match label {
                        Some(l) => {
                            let k = d.node(format!("b{i}"), l, NodeKind::Box, sub);
                            d.nodes[k].obj = Some(o);
                            wire_from[i] = Some(format!("b{i}"));
                            enters[i] = Some(format!("b{i}"));
                            // a step that reads only the body's own input (`c ->` of a map body)
                            if srcs[i].is_empty() && ctx.is_body(n.func) && ctx.read_of(o).is_none() {
                                if let Some(w) = self.body_input_wire(&ctx, n.func, o) {
                                    let t = d.node(format!("ti{i}"), "in".into(), NodeKind::Terminal, sub);
                                    d.nodes[t].obj = Some(o);
                                    d.edge(&format!("ti{i}"), &format!("b{i}"), w, EdgeKind::Wire);
                                }
                            }
                        }
                        None => {
                            let k = d.node(format!("t{i}"), "in".into(), NodeKind::Terminal, sub);
                            d.nodes[k].obj = Some(o);
                            wire_from[i] = Some(format!("t{i}"));
                        }
                    }
                }
            }
        }
        // Readers of each node's wire.
        let mut readers: Vec<Vec<(String, Vec<u32>, bool)>> = vec![Vec::new(); nodes.len()];
        for (t, ss) in srcs.iter().enumerate() {
            let Some(to) = enters[t].clone() else { continue };
            for (k, (a, back, _)) in ss.iter().enumerate() {
                let path = if ss.len() > 1 { vec![k as u32] } else { Vec::new() };
                readers[*a].push((to.clone(), path, *back));
            }
        }
        for (i, n) in nodes.iter().enumerate() {
            let Some(from) = wire_from[i].clone() else { continue };
            let (wl, short, obj) = match n.anchor {
                FlowAnchor::Object(o) => {
                    let ob = self.object(o).unwrap();
                    (node_label(ob), object_short(ob), Some(o))
                }
                FlowAnchor::Print(_) => ("io".to_string(), "io".to_string(), None),
            };
            let mut rs = readers[i].clone();
            if rs.is_empty() {
                if matches!(n.anchor, FlowAnchor::Print(_)) {
                    continue;
                }
                let sub = sub_of.get(&n.func).copied().unwrap_or(0);
                let id = format!("x{i}");
                let ret = obj.is_some_and(|o| self.object(o).unwrap().kind == ObjectKind::Return);
                d.node(id.clone(), if ret { "ret" } else { "∅" }.into(), NodeKind::Terminal, sub);
                rs.push((id, Vec::new(), false));
            }
            let lab = |p: &[u32], back: bool, l: &str| {
                let l = if back { format!("↩ {l}") } else { l.to_string() };
                with_marker(p, &l)
            };
            let sub = sub_of.get(&n.func).copied().unwrap_or(0);
            if rs.len() == 1 {
                let e = d.edge(&from, &rs[0].0, lab(&rs[0].1, rs[0].2, &wl), EdgeKind::Wire);
                d.edges[e].obj = obj;
            } else {
                let id = format!("d{i}");
                let k = d.node(id.clone(), format!("Δ {short}"), NodeKind::Split, sub);
                d.nodes[k].obj = obj;
                let e = d.edge(&from, &id, wl.clone(), EdgeKind::Wire);
                d.edges[e].obj = obj;
                for (r, p, back) in &rs {
                    let e = d.edge(&id, r, lab(p, *back, &short), EdgeKind::Branch);
                    d.edges[e].obj = obj;
                }
            }
        }
        d
    }
}
