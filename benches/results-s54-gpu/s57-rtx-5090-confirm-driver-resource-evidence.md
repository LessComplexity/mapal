# RTX5090 confirmation: saved driver-resource evidence

The accepted primary (`rtx-5090-primary-retry4`, machine44652/host309734) and confirmation (`rtx-5090-confirm-retry3`, machine145957/host617499) have the same frozen compiler/runtime bundle and **identical reported resources for all18 shipped kernel cells** (six sizes×Exact/Contract/TF32). TF32 4096 has the same actual Mapal entry, library kernel names and measured resource tuples in both Nsight activity databases. These equalities do not explain or invalidate the observed cross-host slowdown. No causal performance conclusion is established.

Both bind manifest `26f9dda81ed78ccfa0d92791d9ffe55d4320039fb8bb4b1351352c70647a9afa`, HEAD `b4152c5`. Rehashing all18 saved shipped PTX files matches the manifest; the recorded launch names match their entries/launch descriptors. TF32 4096 PTX is `a12fe0c79e32456641e8327e072e6bbe5269844678c1578a99a433ff4ddddea2`; its tile contains `mma.sync.aligned.m16n8k8...tf32.tf32` and async-copy/wait instructions. This is shared emitted-input identity, not a dump proving identical driver-generated machine code.

| Saved observation | Primary | Confirmation |
|---|---:|---:|
| Driver version (startup header) |615.78.08|590.48.01|
| Power limit W (startup snapshot) |450|546|
| TF32 4096 Mapal CUDA-event median ms |1.5460|4.4472|
| TF32 4096 cuBLAS CUDA-event median ms |1.3906|1.8883|
| TF32 4096 CUTLASS CUDA-event median ms |1.3701|1.8883|
| Decode Contract ordinary process wall median ms |450|5046|
| Decode Contract separately traced summed kernel median ms |72.112|9098.911|

Raw TF32 timing/resource lines are4502–4513 in each linked run text; startup telemetry is lines2–3. The ordinary30-sample event ranges are disjoint across hosts for every TF32 4096 leg. Startup current-SM clocks180/1995MHz and reported device clock-rate attributes2437/2407MHz are snapshots/attributes, not clocks sampled during each kernel. Both identify RTX5090/GB202,170SMs and96MiB L2. Different drivers/power settings or these resource facts alone do not identify a cause.

Actual TF32 4096 Nsight records (eight launches per named kernel per host) have matching tuples below. Static shared and local-memory-per-thread fields are0; the database's separate `localMemoryTotal` field is150405120 for each row and is preserved without interpreting it as spill traffic.

| Actual named route (both hosts) | Registers/thread | Threads/block | Grid | Dynamic shared bytes |
|---|---:|---:|---|---:|
| Mapal `mapal_tile_18a86cb342332edb` |122|256|32×32|37888|
| cuBLAS `cutlass_80_tensorop_s1688gemm_128x128_32x3_nn_align4` |234|128|256×4|98304|
| Pinned CUTLASS `MmaMultistage...128×128×16` |230|128|32×32|65536|

The receipt retains full kernel names, exact SQL, individual intervals and member hashes. Ordinary driver attributes agree with the Nsight Mapal row (122registers/0local bytes). The matching names and resource tuples do not prove identical library/driver SASS. Confirmation profiler intervals also include launch-to-launch variation: Mapal2.056036–4.516137ms, cuBLAS1.884707–4.320937ms and CUTLASS1.885380–4.314473ms across all eight launches, including warmup/validation; these are a separate profiled invocation from ordinary30-sample timing.

Offline `mm_4096_tf.cubin` and its SASS are byte-identical across hosts (cubin `564e49d6…`, SASS `b3b49990…`); offline ptxas reports124tile registers/no spills on both. `box_profile.sh:18–22` independently assembles/dumps these files. The timed harness instead loads PTX at `tile_bench.cu:345` and reads driver attributes at350–351, then launches that function at380. Thus the offline124register image is not evidence of the exact122register JIT image executed. The saved CUTLASS-containing harness SASS dumps also match; actual cuBLAS JIT/library machine code was not dumped.

Decode Contract uses the same frozen PTX `1b80bf6665de5f07d038a578c00628bfc381f6fadf3dd5a1e87dbcce7b455f70`. Both runs record the same three tile names with2048/1024/1024launches,8allocations,2048H2D,1024D2H,3144704inputbytes and1048576outputbytes. Ordinary wall ranges440–460ms versus4912–5104ms and separately traced kernel ranges72.027–72.280ms versus8880.935–9213.313ms establish a broader timing-scale difference; they cannot be combined as one clock decomposition. Decode has no saved Nsight timeline or per-function driver-resource table here, so its per-kernel JIT resources, chronology and cause remain unproved.

Both profiler archives contain Contract library timelines at1024/4096/8192, TF32 at4096, and seven Contract whole-program timelines (two e2e sizes plus epilogue/MLP/MLP2/independent/twin4096). Coverage does not include decode, Exact library timelines, TF32 other sizes, or all workload/control variants. Both `ncu-rtx-5090.csv` files are empty; the examined TF32 SQLite databases have no GPU-metric/hardware-counter tables. No achieved occupancy, tensor/SM utilization, stall, cache-hit, memory-traffic or contemporaneous clock/power measurement resolves the cause. These are explicit evidence gaps, not proof of a compiler defect or a tuning prescription.

[Primary raw records](s57-integrated-rtx-5090-primary-retry4.txt), [confirmation raw records](s57-integrated-rtx-5090-confirm-retry3.txt), [primary provenance](s57-integrated-rtx-5090-primary-retry4.json), [confirmation provenance](s57-integrated-rtx-5090-confirm-retry3.json), and the [reproducible receipt](s57-rtx-5090-confirm-driver-resource-evidence.json) bind the claims. Read-only inspection of saved artifacts only; no builds, new measurements, GPU actions or compiler/policy edits.

Independent presentation QA also checked the new confirmation narrative against the independently checked program CSV (report SHA `78ba609501c367a00be9dd39792432a2a5d3353388dfd5d95ff397d34ff046f7`, lines348–372). All eight stream rows match their labels, six iteration values, ON/OFF ratio and separate traced allocation counts at displayed precision: six disjoint ranges favor OFF, while `pt_indep_4096` Exact and `pt_mlp2_4096` Contract overlap and contain faster ON observations. All eight medians favor OFF, without establishing a uniform win. The decode paragraph's5046/14480ms,2.870×control/shipped ratio, ranges,8/12288allocations, four transfer-byte counts and primary450/1292ms match. Its separate serialized9098.911ms kernel total is correctly kept distinct from ordinary wall. No transcription/ratio/face/timing-boundary mismatch was found in this narrow QA; no root report was edited. The receipt preserves exact report hash/line and CSV bindings.
