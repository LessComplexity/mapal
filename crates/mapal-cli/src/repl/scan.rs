//! The input scanner: brackets, comments and strings, item names, and what kind of input a line is.

// --- scanning -------------------------------------------------------------

pub(super) struct Scan {
    /// A bracket or string is still open.
    pub(super) open: bool,
    /// The text with comments blanked (same length).
    pub(super) code: String,
    /// Byte offsets of `->` at bracket depth 0.
    pub(super) arrows: Vec<usize>,
    /// How many `;` sit at bracket depth 0.
    pub(super) semis: usize,
}

pub(super) fn scan(s: &str) -> Scan {
    let b = s.as_bytes();
    let mut code = b.to_vec();
    let (mut depth, mut i) = (0i32, 0usize);
    let mut arrows = Vec::new();
    let mut semis = 0usize;
    let mut in_str = false;
    while i < b.len() {
        let c = b[i];
        if in_str {
            if c == b'\\' {
                i += 1;
            } else if c == b'"' {
                in_str = false;
            }
        } else if c == b'"' {
            in_str = true;
        } else if c == b'/' && b.get(i + 1) == Some(&b'/') {
            while i < b.len() && b[i] != b'\n' {
                code[i] = b' ';
                i += 1;
            }
            continue;
        } else if c == b'/' && b.get(i + 1) == Some(&b'*') {
            let mut j = i + 2;
            while j < b.len() && !(b[j] == b'*' && b.get(j + 1) == Some(&b'/')) {
                j += 1;
            }
            let end = (j + 2).min(b.len());
            for (k, x) in code.iter_mut().enumerate().take(end).skip(i) {
                if *x != b'\n' {
                    *x = b' ';
                    let _ = k;
                }
            }
            if j >= b.len() {
                depth += 1; // an unterminated block comment keeps the input open
            }
            i = end;
            continue;
        } else if matches!(c, b'(' | b'[' | b'{') {
            depth += 1;
        } else if matches!(c, b')' | b']' | b'}') {
            depth -= 1;
        } else if c == b'-' && b.get(i + 1) == Some(&b'>') && depth == 0 {
            arrows.push(i);
        } else if c == b';' && depth == 0 {
            semis += 1;
        }
        i += 1;
    }
    Scan {
        open: depth > 0 || in_str,
        code: String::from_utf8_lossy(&code).into_owned(),
        arrows,
        semis,
    }
}

pub(super) fn is_ident(s: &str) -> bool {
    let mut c = s.chars();
    c.next().is_some_and(|f| f.is_alphabetic() || f == '_')
        && c.all(|x| x.is_alphanumeric() || x == '_')
}

/// The first word of `code`, when it is exactly an item keyword.
pub(super) fn item_name(code: &str) -> Option<String> {
    let t = code.trim_start();
    let rest = t
        .strip_prefix("fn")
        .or_else(|| t.strip_prefix("type"))
        .filter(|r| r.starts_with(char::is_whitespace))?;
    let name: String = rest
        .trim_start()
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    Some(name)
}

pub(super) enum Kind {
    Item(String),
    /// A statement kept in the body; `Some(name)` when it ends by binding one.
    Stmt {
        bound: Option<String>,
    },
    Expr,
}
