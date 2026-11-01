# Stored epilogue member ordering investigation

2026-10-21. This is an investigation with the already built, frozen tools, not a compiler/runtime change. Both Exact and Contract fixtures emitted successfully, and the interpreter returned the expected output. GPU execution, disassembly and a wrong-value/race reproduction have **not** been performed. The proposed allocation elimination was **not** applied, including to these emitted files.

A stronger current-codegen variant retaining raw outputs also emitted successfully: [materialized-raw/README.md](materialized-raw/README.md). It has no runtime calls between its last chain launch and first stored-member legacy readback. That variant demonstrates a current emitted missing-producer-wait path without applying allocation elimination; GPU wrong output remains untested.

## What the emitted artifacts establish

Fixture: [stored-member-ordering.mapal](stored-member-ordering.mapal), SHA256 `5fc10fd7cb13630078e0afd4c639111227a9fd3d0ffcb6b544516dcef8aae1a0`. It has two independent 64×64 matrix products, each followed by a same-cell doubling map. It reads the second stored member's corners first, then the first member's corners, then all input corners. There are no time reads. The interpreter exits0 and prints `5520, 5236, 2028, 2016, 1, 2, 1, 1`, one per line; [oracle.stdout](oracle.stdout) preserves exact formatting.

Both [exact.host.ll](exact.host.ll) and [contract.host.ll](contract.host.ll) have the same relevant line layout:

| Evidence | Lines |
|---|---:|
| Raw target slots `%d22` and `%d24` start at zero | 67–70 |
| Both raw target buffers are allocated by the prologue | 111–112 |
| First group's raw output pointer is literal zero; stored member pointer is `%t27` | 119–121 |
| First grouped launch uses stream1 | 122 |
| Second stored member allocation occurs before its launch | 129 |
| Second group's raw output pointer is literal zero; stored member pointer is `%t38` | 135–137 |
| Second grouped launch uses stream2 | 138 |
| The **only** calls between that launch and first member readback free raw slots `%d22`/`%d24` | 139–140 |
| First read of the second group's stored output is legacy `mapal_gpu_d2h`, with no stream argument | 151–153 |
| First stored-output free occurs after its second corner read | 168–171 |
| Input slots remain live until their later host reads | first input free218 |

Neither raw slot is loaded or passed to a kernel. Both device tile kernels name `%o` only in their signatures, while their stored `%y0` member pointers are used in their bodies. [proof.json](proof.json) records the exact calls, raw-slot occurrences, kernel names and artifact hashes. These facts prove grouping/chain eligibility and the emitted call order; they do not prove an observed GPU race.

The **current** fixture retains an explicit context barrier: `mapal_gpu_free` checks a nonzero raw slot, sees `chains_busy` and calls `cuCtxSynchronize` (`crates/mapal-rt/gpu/mapal_gpu.c:553–563`). Consequently this emitted baseline is not a demonstrated wrong program. If a later change omits both raw allocations, their frees would return on zero at554–555 before this barrier. Whether that counterfactual produces observable wrong values has not been tested. No generic elimination should be claimed proved safe from these baseline files alone.

## Existing producer facts and all stream lookups

`crates/backends/llvm/src/func/resident.rs:293`, `ResidentPlan::stream_of`, examines only the object's own IR in-edges for membership in `plan.chain`. A stored member's own producer is its skipped Map; the base tile owns its actual launch. Every literal consumer in `crates/` is:

| Consumer | Location | Role |
|---|---|---|
| `emit_site_resident` | resident.rs:1985 | Select the site's launch stream |
| `emit_site_resident` | resident.rs:2016 | Select cross-chain waits for site/capture operands |
| `resident_elem` | resident.rs:2160 | Select stream for a kept array's host element copy |

Existing real-writer facts already exist in the same module: `epilogues` initializes `made[target(site)] = site` at710–712, then records `made[member.out] = site` at825; `Epi` is keyed by the base morphism and stores each `Member.out`; `arena`'s local `site_of` at1469–1474 resolves a member to the base site to compute its true birth. `ResidentPlan::alias` at302 maps carry views/exits to the merge slot, and `dslot` at1603 applies that identity. This fixture has no carry/loop aliases. Any future ordering correction must keep actual-writer identity and device-slot identity coherent; this is a requirement to investigate, not a new implementation.

`chains`' unordered-consumer veto constructs its `chained` set from base targets only at1337–1340 and checks non-chained sites/cells against that set at1347–1353. Stored-member cross-chain and legacy-consumer cases therefore need their own emitted fixtures; the current two-group host-readback fixture does not prove those additional cases.

The runtime creates chain streams with `CU_STREAM_NON_BLOCKING` (`mapal_gpu.c:159`). Its automatic `event(0,k)` at321–323 orders legacy producers **before** a chain launch; it does not order a later legacy readback after a chain. Legacy `mapal_gpu_d2h` at274 calls `cuMemcpyDtoH` with no explicit context/producer wait. `mapal_gpu_d2h_on` at388 queues the copy on the named producer stream and synchronizes that stream. CUDA documents that nonblocking streams do not implicitly synchronize with stream0: [driver stream creation](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__STREAM.html), [default-stream behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html).

## Reproducibility and freeze receipt

[commands.json](commands.json) contains the exact argv and exits. The emission command for Exact was:

```sh
/Volumes/LessComplex/Personal/mapal/target/release/examples/emit \
  /private/tmp/mapal-s56-gpu/ordering-audit/stored-member-ordering.mapal - \
  --rewrite --target=rtx-5090 --gpu-on=streams,epilogue --gpu-off=stream-gate \
  --device-out=/private/tmp/mapal-s56-gpu/ordering-audit/exact.device.ll \
  > /private/tmp/mapal-s56-gpu/ordering-audit/exact.host.ll
```

Contract adds `--contract` and uses separate `contract.*` paths. The interpreter command was:

```sh
/Volumes/LessComplex/Personal/mapal/target/release/examples/run \
  /private/tmp/mapal-s56-gpu/ordering-audit/stored-member-ordering.mapal \
  --budget 200000000
```

[before.json](before.json) and [after.json](after.json) are byte-equivalent snapshots of HEAD `b4152c5f4d9ad802d0388c1c64b628091d494e6a`, the frozen source-manifest hash, all328 source hashes and these three tool/library hashes:

| Frozen artifact | SHA256 |
|---|---|
| `target/release/examples/emit` | `f7f47973b607686b52cb4ac37c2876b384a0cc32dd82191a11341eb54ba1a5aa` |
| `target/release/examples/run` | `ad2c59a934134392b09e802d57d1020e7cb3a8fa8d1efef35da2a5bc736b9a63` |
| `target/release/libmapal_rt.a` | `70a897fc3905c6fa982aa010352eec2a57025cf7838baaf30b49cec5399a43be` |

No cargo, source edit, final bundle/oracle mutation or GPU operation occurred.

## Remaining decisive proof

After the frozen validation phase, first preserve this baseline and explicitly authorize a temp-only experiment removing exactly the two unused raw allocations at111–112, leaving zero initialization, all real output allocations, all launches and all frees untouched. Inspect/disassemble that variant to establish that no context wait remains before line153. It is not enough to reason from a proposed source edit.

Then compare baseline and that exact variant against the interpreter on an authorized GPU, both faces, ordinary asynchronous mode (`MAPAL_GPU_TRACE` unset), no added `time` reads or clock fences. A larger extent/repetition can widen the race window, but requires a separately recorded emitted artifact and oracle. Preserve stdout/exit status and a timeline showing the readback's ordering relative to the actual writer if available. A wrong value or readback before writer completion is decisive; repeated correct runs alone do not prove the absent dependency safe. Tracing with `MAPAL_GPU_TRACE=1` synchronizes every launch (`mapal_gpu.c:352–353`) and cannot establish ordinary asynchronous safety.

No runtime failure is claimed, and no fix is selected by this audit.
