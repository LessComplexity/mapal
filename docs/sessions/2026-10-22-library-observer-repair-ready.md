# 2026-10-22 — Reviewed library observer ready for renewed approval

## Continuation brief

Compiler correction is committed at `50a57e5` and its scoped OpenSpec change is archived.
The repaired standalone library observer is sealed, independently reviewed and ready.
Automatic approval review rejected the new upload before execution because the prior
explicit approval named the original bundle. A precise revised-bundle/equivalent-host
question is pending. No task machine is live and no new upload occurred.

Resume with `cat docs/next-session.md` and
`cat /private/tmp/mapal-s61-gpu/attempt-12/approval-review-block.json`.
After explicit approval, refresh quote/inventory, bind the exact packet to the new reply
and use the reviewed launcher through normal approval. Do not bypass the rejection.

## Completed work and decisions

The first actual trace remains immutable and rejected. Four instrumentation defects are
repaired: callback invocation identity distinct from activity correlation, bounded ELF
helper subranges, provisional UID0 context metadata before authoritative creation, and
explicit activity-domain binding when driver/runtime correlations share the same number.
Only the observer's `capture.cpp` and `admit.py` change in production tooling; the remote
runner, request builder, benchmark binaries, numeric contract, CLI and caps are unchanged.

| Decision | Evidence and result |
| --- | --- |
| Use SDK per-invocation correlationData | Pinned header376–383 and actual-source tests; missing/reused/corrupt identities reject |
| Keep activity correlation separate | Unique exact callback/domain/CBID/status binding; wrong/contradictory/both-domain matches reject |
| Inventory helper ranges without exporting them as kernels | Bounded unique section owner; helper names cannot enter launch/decoder selection |
| Admit only narrow provisional context prefix | SetCurrent/PrimaryCtxRetain UID0 before observed creation; missing/late/positive-UID/unknown uses reject |
| Replay the complete recorded sequence locally | Caught the fourth domain-selection defect before another upload; fresh CPU-owned storage is synthetic proof only |
| Preserve original v1 verdict | Zero image/SASS admissions; no trace edits, deduplication or retrospective attribution |

53 source callback/parser cases,12 identity mutations and10 integration cases pass.
Independent checks add9 callback/context,8 ELF-range and8 domain cases. Full actual-source
synthetic replay preserves3,156 API rows,1,799 activity records,5 byte-identical images and
13 kernel records. Removing only new token fields/schema reproduces all original event
fields. Original invocation storage is not recovered; fresh live validation remains needed.

Independent final delta review SHA256
`f2517304605f3f0c7302df4094283c06fa6fffd74712c4ce419143fe7ae30ff2`.
Root checks all198 payload and199 archive files, safe archive members, reviewed source
hashes and unchanged runtime/request/protocol files. The remote payload excludes failed
captured images and generated replay traces/binaries; compact proof receipts and test
programs are included.

## Exact ready packet

- Observer archive: `/private/tmp/mapal-library-image-capture/library-image-capture-v2.tar.gz`,
  932,273bytes; SHA256 `48f288f43a9d2133a50e3572de8aa5604209fc254e5697e0c8027f15d5aaacaf`.
- Manifest: `21d58721e96250dcf30d8dd23799cfe7e337604fbb4993c81ad5f15cbbc85eac`.
- Operator packet: `/private/tmp/mapal-s61-gpu/attempt-12/packet.json`, SHA256
  `6138fcc648e7009c435e9744eb37b178237da9d4aa99ecc21b3e865d4ed2b637`.
- Root binding: `root-review.json` in that folder, SHA256
  `ae680703560f300b549e5d0806a2a3c4bc2460646f4aa2c4201df0a312b6811d`.
- Total payload9,157,695bytes,27,686 more than the previously approved packet. Original
  source archive713433bf… and337-file ordinary replay3d30714b… remain byte-identical.
- Last quote56803570, physical135433/346497, RTX5090, about$1.037037/hour including40GiB;
  refresh before claiming. Capture480s, outer510s, instance3300s, startup1200s and cleanup
  reserve180s. All normalized launcher/wrapper/guard logic matches the reviewed prior run.
- Independent reader: `/private/tmp/mapal-library-image-review/repair-review/live-admission-v2/read_live.py`,
  SHA256 `116d07c61ec1bf6ae60c8ead7d851aef24c42301d821b7c00ecc8be29b6ff820`.
  It checks numeric/provenance independently and explicitly replays the reviewed sealed
  attribution parser; it is not a second independent join implementation.

## Live state and authorization

All agents have completed their local turns. Operator `/root/comparison_run` is idle;
observer owner `/root/gap_analysis_s61` and reviewer `/root/composed_analysis_s61` are
available for continuation. No local test process or task rental is running.

All11 actual task rentals are closed; latest57265999 independently absent20:05:14.535Z.
Attempt12 is unclaimed: automatic review blocked before execution, create and upload.
Exact reason is saved in `approval-review-block.json`: the repaired observer delta was
not explicitly authorized for egress in the prior exact-bundle approval.

User question pending: approve the reviewed9.16MB repaired bundle on physical135433/346497
or an equivalent RTX5090, preserving eight-minute capture/55-minute instance limits and
existing-machine protections. Monetary caps remain removed. Protect unrelated IDs57065574,
57100404,57126023,57133432 and every fresh prelaunch ID. Only the exact new create-response
ID matching physical host and absent from prelaunch inventory permits access/cleanup.
Never print provider/account/API/create/wrapper logs; prior credential rotation remains open.

## Records, checks and next actions

Public proof is under `benches/results/gpu/disassembly/library-kernels/counter-free-capture/observation-schema-repair/`.
Persistent private source archive: `.benchmark-artifacts/gpu/library-images/preparation/library-image-capture-v2.tar.gz`.
The original failed trace/source remain separately retained. No new performance or library
SASS result exists; earlier competitive and fused comparisons are unaffected.

Compiler report now clarifies that affected controls preserve raw LLVM arithmetic counts
and PTX FMA counts, not every optimized PTX instruction count. Exact PTX evidence records
strided loop unrolling4→8 and virtual-register declaration22→35, plus occupancy hints in
three fallbacks. These are static compiler effects, not physical registers, occupancy or
speed measurements. Original review is immutable; a separate clarification is saved.

Strict OpenSpec7/0; drift0dead/581refs;460report links resolve; all2,802 relocated historical
raw hashes stay unchanged. Source/docs diff check passes; raw logs/patches keep original
whitespace. Code graph remains current from the compiler checkpoint10,177nodes/20,038edges;
no production source changed afterward. Semantic doc extraction lacks an LLM key; earlier
gbrain indexing encountered an existing PGLite lock, left untouched.

Open plans: capture-executed-cuda-images8/11, benchmark-large-composed-gpu-workloads6/15,
optimize-ordered-smallm-gpu4/7, eliminate-unmaterialized-gpu-allocations11/15. Only the scoped
arithmetic correction is archived. Next completed outcome requires fresh live observation
and complete admission, followed by actual SASS comparison against existing IR facts.

Living docs updated: next-session, current GPU results, root/component status and the
observer OpenSpec design/tasks. Related source/audit reports and organized evidence remain
linked. Preserve unrelated `oainotes.md`; local commits only, no push.
