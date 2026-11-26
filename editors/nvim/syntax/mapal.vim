" Vim syntax file
" Language:    Mapal (dataflow language)
" Maintainer:  Mapal project (editors/nvim)
"
" What a highlighter that sees one line at a time can know. The language
" server (`mapal lsp`, started by lua/mapal/lsp.lua) knows the rest — which
" name is a call, a parameter, a field, a top-level value — and its semantic
" tokens are linked back to these same groups, so a name has one colour with
" or without the server, in Neovim and in VS Code alike.
"
" The classes, and the group / TextMate scope / semantic token of each, are
" editors/test/classes.tsv; the expected class of every token of the shared
" fixture is editors/test/fixture.tokens.tsv (column `static` is what this
" file must say). Test: editors/nvim/test/run.sh. The builtin, primitive and
" keyword lists are the language's own (mapal-lower `builtin_names`,
" `SCALAR_TYPES`, mapal-syntax `KEYWORDS`; test `grammar_name_lists_are_the_languages`).
"
" ----------------------------------------------------------------------------
" VIM MATCH PRECEDENCE (load-bearing — read before reordering anything):
"   `syn keyword` items beat every `syn match`; between matches starting at
"   the same column the one DEFINED LAST wins. So groups go from least to most
"   specific: operators, then the name rules (call, binding), then arrows (win
"   over `-` and `>`), then size lists (`<` `>` as brackets), then guard
"   arrows (win over arrows and `-`), then the name after a guard arrow (a
"   read, not a binding), then comments and strings.
" The TextMate grammar takes the EARLIEST match instead (first listed on a
" tie), so it is ordered the other way round; editors/vscode/test/scope_test.py
" holds it to the same fixture.
" ----------------------------------------------------------------------------

if exists("b:current_syntax")
  finish
endif

syn case match

" Keywords (mapal-syntax KEYWORDS, less `ret`, the booleans and `category`).
syn keyword flowKeyword fn type mut loop seq map fold executor pub use void
syn keyword flowRet ret
syn keyword flowBoolean true false
" ADR-0006: `category` is the v0.1 spelling of `type`, reserved and rejected.
syn keyword flowReserved category

" The stage builtins (mapal-lower builtin_names): effects, collection ops and
" sources, the math builtins (ADR-0042 / ADR-0046), widening and narrowing.
syn keyword flowBuiltin print println time read zip enumerate iota fill exp sqrt sin cos
syn keyword flowBuiltin widen_i64 widen_f32 widen_f64 to_bf16 to_f16 to_f8e4m3 to_f8e5m2

" The scalar types (mapal-lower SCALAR_TYPES) and the `: compute` member tf32.
syn keyword flowPrimType i32 i64 u8 f32 f64 bf16 f16 f8e4m3 f8e5m2 bool tf32

" ADR-0048 D5: `_`, a type's default interpretation (`type Hidden { _: … }`).
syn match flowDefaultField "\<_\>"

" ---------------------------------------------------------------------------
" Operators (first: arrows and guards, defined later, win over `-` `<` `>`).
" `|` separates `: compute` members (`fold: bf16 | f32`).
" ---------------------------------------------------------------------------
syn match flowOperator "==\|!=\|<=\|>=\|&&\|||\|[+\-*/%=!<>|]"

" ---------------------------------------------------------------------------
" Names, as far as one line tells.
" ---------------------------------------------------------------------------
" Between two arrows a name is applied: `data -> f -> g -> ret;`.
syn match flowFunction "\%(->\s*\)\@<=\<[a-z_]\w*\>\%(\s*->\)\@="

" PascalCase names are types (defined after flowFunction: a type stays a type).
syn match flowTypeName "\<[A-Z]\w*\>"

" Bindings: `-> name;` and `-> name: T` bind (or rebind a `mut`); so do
" `mut name` and `name <- value`. A terminal call to a function that returns
" nothing (`-> shout;`) looks the same; the server tells them apart.
syn match flowBinding "\%(->\s*\%(mut\s\+\)\=\)\@<=\<[a-z_]\w*\>\ze\s*\%(;\|:\%(:\)\@!\)"
syn match flowBinding "\%(\<mut\s\+\)\@<=\<[a-z_]\w*\>"
syn match flowBinding "\<[a-z_]\w*\>\ze\s*<-"

" Function names after `fn`.
syn match flowFnName "\%(\<fn\s\+\)\@<=\h\w*"

" ---------------------------------------------------------------------------
" Numbers: integers and decimal floats (the lexer's two forms). Float last.
" ---------------------------------------------------------------------------
syn match flowNumber "\<\d\+\>"
syn match flowFloat  "\<\d\+\.\d\+\>"

" ---------------------------------------------------------------------------
" Arrows (after operators: `->` / `<-` win over `-`, `<`, `>`).
" ---------------------------------------------------------------------------
syn match flowArrow "->\|<-"

" ---------------------------------------------------------------------------
" Size lists (ADR-0047 / ADR-0048 D4), after operators and arrows so `<` `>`
" are brackets: `fn f<rows, columns>`, `type Matrix<rows, columns>`, and a
" generic type at its sizes `Matrix<hidden_size, 64>`. The names inside are
" sizes; commas stay plain (the region is transparent).
" ---------------------------------------------------------------------------
syn region flowSizeArgs transparent oneline matchgroup=flowSizeBracket
      \ start="\%(\<\u\w*\)\@<=<" end=">" contains=flowSizeParam,flowNumber,flowOperator
syn region flowSizeList transparent oneline matchgroup=flowSizeBracket
      \ start="\%(\<\%(fn\|type\)\s\+\h\w*\s*\)\@<=<" end=">" contains=flowSizeParam,flowNumber,flowOperator
syn match flowSizeParam contained "\<[a-z_]\w*\>"

" ---------------------------------------------------------------------------
" Guard arrows `-true->` `-false->` `-_->` `-42->` (single lexemes,
" ADR-0010), after arrows and operators so they win over both. The chrome
" reads as an arrow; the discriminant as what it is.
" ---------------------------------------------------------------------------
syn match flowGuardArrow "-\%(true\|false\|_\|\d\+\)->" contains=flowGuardBool,flowGuardInt,flowGuardWild
syn match flowGuardBool contained "\<\%(true\|false\)\>"
syn match flowGuardInt  contained "\<\d\+\>"
syn match flowGuardWild contained "_"

" The name right after a guard arrow heads the arm's chain: a read, not a
" binding or a call (`-false-> b;`, `-_-> n -> double;`). No highlight.
syn match flowGuardValue "\%(-\%(true\|false\|_\|\d\+\)->\s*\)\@<=\<[a-z_]\w*\>"

" ---------------------------------------------------------------------------
" Strings and comments (last: they win over everything inside them).
" ---------------------------------------------------------------------------
syn match  flowEscape contained "\\[nt\"\\]"
syn region flowString start=+"+ skip=+\\\\\|\\"+ end=+"+ oneline contains=flowEscape

syn keyword flowTodo contained TODO FIXME XXX NOTE
syn match   flowComment "//.*$" contains=flowTodo,@Spell

" ---------------------------------------------------------------------------
" Highlight links (`hi default link`: colorschemes can override).
" ---------------------------------------------------------------------------
hi default link flowComment      Comment
hi default link flowTodo         Todo
hi default link flowString       String
hi default link flowEscape       SpecialChar
hi default link flowNumber       Number
hi default link flowFloat        Float
hi default link flowBoolean      Boolean
hi default link flowKeyword      Keyword
hi default link flowRet          Special
hi default link flowReserved     Error
hi default link flowOperator     Operator
hi default link flowArrow        Statement
hi default link flowGuardArrow   Statement
hi default link flowGuardBool    Boolean
hi default link flowGuardInt     Number
hi default link flowGuardWild    Special
hi default link flowSizeBracket  Delimiter
hi default link flowDefaultField Special
hi default link flowBuiltin      Function
hi default link flowPrimType     Type
hi default link flowFnName       Function
hi default link flowFunction     Function
hi default link flowTypeName     Type
hi default link flowSizeParam    Constant
" A binding is a variable definition, not a call.
hi default link flowBinding      Identifier
" Only the language server tells these apart (classes.tsv); lua/mapal/lsp.lua
" links its semantic tokens to them.
hi default link flowConstant     Constant
hi default link flowParameter    @variable.parameter
hi default link flowField        @property

let b:current_syntax = "mapal"
