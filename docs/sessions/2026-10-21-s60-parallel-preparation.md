# S60 — parallel preparation, awaiting explicit GPU budget extension

Immutable checkpoint, 2026-10-21 local / 2026-10-22 UTC. Branch `s54-gpu-fusion`,
starting HEAD `bc7af76`. The user said to proceed after the three-workstream proposal.
Three GPT-6.1-sol workers completed isolated preparation; no new worktree was created.
No production source changed, no rental/upload/create occurred, and no GPU result is
claimed. Preserve unrelated `oainotes.md` and the frozen release emitter.

## Live state and authorization

An asynchronous question is pending: approve up to $1 additional for one reviewed
26.5 MB S60 batch, raising cumulative cap from $1 to $2; RTX 5090 offer 56809128
(machine 17419/host 43746) or equivalent available host, rate at most $0.80/hour,
55-minute creation-through-cleanup limit, aggregate at most $8/hour. **No answer has
arrived at this checkpoint. Do not infer approval or launch.** The prior cap has
$0.09634004682925368 remaining. The 03:13 UTC quote reserves $0.6601851852 including
$0.10 transfer allowance; this quote will need refreshing. No approval-review
rejection occurred during S60; the existing spending cap is why approval is needed.

Never touch unrelated instances 57065574, 57100404, 57126023 or 57133432. They were
observed read-only at 03:13 UTC. Every prior task rental was already closed in S59.
No active S60 process/cell or rental remains at this checkpoint. The three worker
agents are idle and reusable. Provider diagnostics may expose credentials; all raw
CLI/API failures must stay in private files, never tool output. Earlier Vast-key
rotation remains outstanding; do not rotate it automatically.

## Completed local work

See [the S60 evidence report](../performance/s60-parallel-preparation.md) for the
scope table, numerical contract and kernel findings. In particular:

- `gpu_tile` participates in preliminary fill-key selection before `rungs_at`;
  changing legality can change the eventual arithmetic route. A refutes the simple
  one-wave filter as a preservation of S58's wide wins. B is a different legal
  smaller-M geometry, not a measured optimization. Each passes 918,000 geometry
  comparisons, with 57 emissions preserved. The three native detection failures
  in the broader backend test also occur with an unchanged profile.
- The composed v3 packet covers M128/d256/h512 and M128/d1024/h2048, five
  implementations, matched synchronized resident-work clocks, separate correctness
  and final-only graphs, and raw PyTorch checker values. Larger coverage is four
  complete oracle rows, 256 sampled cells/stage; no all-output claim.
- The allocation gate remains the reviewed S58 v2: 640 ordinary processes then
  28 separate counter processes, forced streams and no timing admission.
- Root found and workers repaired retrieval outside the fetched tree, unmatched
  closing-event ordering, omitted raw checker export, and nested cancellation.
  Real local spawn/assignment and descendant tests pass; these are not GPU tests.
- Independent readers reparse raw values/events/images and reject corrupt synthetic
  artifacts. No result reader currently has GPU data.

Supercharge preflight passed. Semble MCP retrieval did not complete within the bounded
attempt; workers also attempted a bounded CLI fallback, then used already documented
symbols/paths. Architecture was reconciled implementation → architecture → status.
Strict OpenSpec validation passes 7/7; drift reports 0 dead / 572 references before
this immutable checkpoint. No active change is archived or falsely marked complete.

## Exact artifacts and owners

| Owner | Root directory | Responsibility |
| --- | --- | --- |
| `/root/comparison_analysis` | `/private/tmp/mapal-s60-composed` | Composed v3 and independent reader |
| `/root/comparison_run` | `/private/tmp/mapal-s60-gpu` | Frozen batch, fresh quote/binding, one-use launcher/cleanup |
| `/root/gap_actions` | `/private/tmp/mapal-s60-kernel` | A/B proof, optional B/capture packets, independent readers |
| Root | `/private/tmp/mapal-s60-root` | Review receipts and integration/admission |

Final packet SHA256 `b37f14727ac878394046d520e83ef60c1679f1efe758e9ad37cf4dac20a12432`;
26,444,972 upload bytes. Launcher SHA256
`90c514ad3b97b3c7f30909383703107f465ff3ccd3788b4366baa72e33bdc211`;
wrapper `ceedc88b50263e4c76b765ae784216aa7f374df65790e032e85dbc22f09b948e`.
`review-receipt.json` and `packet.json` bind all component hashes; root independently
verified every staged byte and safe archive inventory. No `root-review.json` authorizing
a launch exists. The one-use `launch-claim.json` does not exist.

Public receipts: `benches/results-s54-gpu/s60-parallel-composed/`, with a hash manifest.
Ignored durable archives under
`.claude/worktrees/_archive-s54-20261021/s60-parallel-composed/`:

- `preparation.tgz`: 885 files, 26,638,806 compressed bytes, SHA256
  `8f52deba863af578bdcabe9f9447f5abbf3a6f00d0ce61538a454aa3f0b34903`.
- `independent-readers.tgz`: 15 files, 17,940 bytes, SHA256
  `d9cd982eb5ffae80de65a75e1cb8d57811aafc8d94d1723792a49d3f7de0b44b`.

These are mode0600, parent0700; every archived member was reread and hash-verified.
Public receipts contain recovery paths/member hashes. Raw provider diagnostics are
excluded. Generated payload JSON/LLVM/PTX/objects and isolated patches remain archived;
the reusable harness is not yet integrated as a validated production benchmark.

## Resume commands and decision sequence

First read any reply to the pending budget question. Continue authorized local work
without a new permission question; new spending still requires that explicit reply.
After approval, reuse the operator agent to refresh the eligible quote, preserve the
unchanged payload hashes, and reseal the price-bound packet. Read-only refresh commands:

```sh
python3 /private/tmp/mapal-s60-gpu/query_offers.py > /private/tmp/mapal-s60-gpu/fresh-query-public.json
python3 /private/tmp/mapal-s60-gpu/inspect_inventory.py
```

Keep all raw API errors private. Root must independently verify fresh price/storage,
protected inventory, byte seals and the remaining cap; then write `root-review.json`
binding `approved: true`, actual packet/launcher hashes, `budget_extension_approved:
true`, approved cumulative cap 2 and `single_attempt: true`, with the actual approval
reference. Do not write those affirmative fields before the reply. The launcher rejects
a snapshot older than ten minutes or an existing launch claim. The authorized launch is:

```sh
python3 /private/tmp/mapal-s60-gpu/launch_once.py
```

Record instance/deadlines in mutable `docs/next-session.md` immediately. Required phase
order: allocation → pinned Torch bootstrap → both composed sizes; optional B then
instrumented capture. Absolute limit3300s, wrapper cutoff3120s, cleanup reserve180s.
Preserve failed/partial results. Any operational failure stops later GPU work; only the
reviewed allocation numeric classifier can release independent baseline composed work.
No retries beyond the explicitly approved attempt.

Local preflight checks (no GPU/network):

```sh
python3 /private/tmp/mapal-s60-composed/packet-v3/run_all.py --check-only
python3 /private/tmp/mapal-s60-kernel/bcomparison/run_b.py --out /private/tmp/mapal-s60-root/check-b --check-only
python3 /private/tmp/mapal-s60-kernel/diagnostics/run_diagnostics.py --out /private/tmp/mapal-s60-root/check-images --check-only
```

After retrieval, substitute the actual retrieved trace path in the readers below.
They exclusively create output directories; use fresh names when rerunning. The composed
reader uses sealed packet-v3 by default. Do not pass an unrelated outer hash manifest.

```sh
python3 /private/tmp/mapal-s58-allocation-gate/packet-v2/verify_gate.py /private/tmp/mapal-s58-allocation-gate/packet-v2 RETRIEVED_TRACE/allocation-gate-v2-results --packet-sha 32d618492a233f3c32708cc8c028a752dd4997413529e487ac4d0f07c8d18a87
python3 -B /private/tmp/mapal-s60-composed/analysis/read_results.py --results RETRIEVED_TRACE/composed-results --out /private/tmp/mapal-s60-root/composed-results
python3 /private/tmp/mapal-s60-kernel/analysis/read_b.py --packet /private/tmp/mapal-s60-kernel/bcomparison --results RETRIEVED_TRACE/kernel-B --out /private/tmp/mapal-s60-root/B-results
python3 /private/tmp/mapal-s60-kernel/analysis/read_images.py --packet /private/tmp/mapal-s60-kernel/diagnostics --results RETRIEVED_TRACE/kernel-diagnostics --out /private/tmp/mapal-s60-root/image-results
```

Root independently confirms rental absence and raw inventories before admitting results.
Publish the combined-program table, allocation counters and kernel comparison separately.
Integrate the allocation patch only if its own correctness gate passes; this still proves
no default5090 allocation speedup. Neither A nor B is promoted merely from local legality
or four timing cells. Keep independent branches, recurrent retention, graph-enabled
references and larger/control obligations open. Reconcile docs, strict checks, graph,
memory and an immutable result handoff; local commits only, no push.
