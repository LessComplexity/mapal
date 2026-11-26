//! Byte offsets ↔ LSP positions. LSP counts columns in UTF-16 code units
//! (the default `positionEncoding`); the compiler counts bytes.

use lsp_types::{Position, Range};
use mapal_syntax::SourceLoc;

pub struct LineMap {
    starts: Vec<u32>,
}

impl LineMap {
    pub fn new(text: &str) -> LineMap {
        let mut starts = vec![0u32];
        starts.extend(
            text.bytes()
                .enumerate()
                .filter(|(_, b)| *b == b'\n')
                .map(|(i, _)| i as u32 + 1),
        );
        LineMap { starts }
    }

    pub fn position(&self, text: &str, offset: u32) -> Position {
        let offset = offset.min(text.len() as u32);
        let line = match self.starts.binary_search(&offset) {
            Ok(i) => i,
            Err(i) => i - 1,
        };
        let start = self.starts[line] as usize;
        let mut end = offset as usize;
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        let character = text[start..end].encode_utf16().count() as u32;
        Position::new(line as u32, character)
    }

    pub fn range(&self, text: &str, span: SourceLoc) -> Range {
        Range::new(
            self.position(text, span.start),
            self.position(text, span.end),
        )
    }

    /// The byte offset of `pos`, clamped to its line (and to the text).
    pub fn offset(&self, text: &str, pos: Position) -> u32 {
        let Some(&start) = self.starts.get(pos.line as usize) else {
            return text.len() as u32;
        };
        let line_end = self
            .starts
            .get(pos.line as usize + 1)
            .map(|&s| s as usize - 1)
            .unwrap_or(text.len());
        let mut units = 0u32;
        for (i, ch) in text[start as usize..line_end].char_indices() {
            if units >= pos.character {
                return start + i as u32;
            }
            units += ch.len_utf16() as u32;
        }
        line_end as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf16_columns_round_trip() {
        // `é` is 2 bytes / 1 unit; `𝑥` is 4 bytes / 2 units.
        let text = "// é𝑥\nfn f() {}\n";
        let m = LineMap::new(text);
        let f = text.find("fn").unwrap() as u32;
        assert_eq!(m.position(text, f), Position::new(1, 0));
        let after = text.find('\n').unwrap() as u32;
        assert_eq!(m.position(text, after), Position::new(0, 6));
        assert_eq!(m.offset(text, Position::new(0, 6)), after);
        assert_eq!(m.offset(text, Position::new(1, 3)), f + 3);
        assert_eq!(m.offset(text, Position::new(9, 0)), text.len() as u32);
    }
}
