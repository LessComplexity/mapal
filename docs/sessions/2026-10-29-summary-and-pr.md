# 2026-10-29 — summary and pull request to main

Immutable. Follows [2026-10-26-site-geometry.md](2026-10-26-site-geometry.md); no new compiler work in this session.

## What happened

| item | state |
| --- | --- |
| `summary.md` (repo root) | plain-words summary of the cpu-gpu-boundary, recurrence-precision and site-geometry sprints: what changed, before → now ratios per workload and GPU with the fastest competitor named, what is still behind, next steps |
| push + PR | Sapir asked to push (CLAUDE.md "never push" overridden for this request): branch `site-geometry-latest` pushed to `origin`, PR opened against `main` — `gh pr view site-geometry-latest` |
| pre-push checks | no private keys, tokens or the vast API key in the 319-commit range (text, history and archives scanned); largest file 25.5 MB (`prof-*.tgz`) |
| clarified with Sapir | "prefill" = the attention prefill of a 1-layer, 1-head, width-256 model (KV cache built for a 512-token prompt, causal attention), then decode one token per step; "gen" = the same model as one natural loop, prefill/decode found by the compiler; competitors are cuBLAS / cuBLASLt / CUTLASS / cuDNN via PyTorch / hand-written CUDA / PyTorch / `torch.compile`, not FlashAttention or vLLM |

## Next session starts here

1. Read `docs/next-session.md` (candidate next sprint: one sizing model from facts — work per block after a split,
   last-round balance over the compute units, cost of combining split sums, cache residency) and `summary.md`.
2. The sprint is not approved yet: propose it, get approval, keep the workflow lean (CLAUDE.md "measure only what
   changed", "a mixed result is a missing fact").
3. If the PR has review comments: `gh pr view site-geometry-latest --comments`.

## Live state

| item | state |
| --- | --- |
| branch | `site-geometry-latest` (main checkout), pushed; PR open to `main` |
| vast rentals | none of ours |
| local jobs | none |
| untracked, not committed | `oainotes.md` (Sapir's notes), `.claude/` (agent worktrees, sprint scope files) |

## Resume

```sh
cd /Volumes/LessComplex/Personal/mapal
git status --short && git log --oneline -3
gh pr view site-geometry-latest
cat docs/next-session.md summary.md
```
