# Standard Nsight Compute diagnostic packet

This CPU-tested runner consumes an admitted, uninstrumented competitive ordinary run and its exact existing candidate binary. It never builds, installs, rents, uploads, or changes GPU access controls. Root-approved initial scope is `rtx-5090__timing__deepseek_expert_m16_n2048_k7168__contract` and its ordinary-selected explicit Mapal variant/CUTLASS candidate.

After ordinary completion:

```
python3 /w/ncu-extension/build_request.py --packet /w/competitive-reference --ordinary /w/trace/competitive-ordinary --case rtx-5090__timing__deepseek_expert_m16_n2048_k7168__contract --out /w/trace/ncu-request.json
python3 /w/ncu-extension/run_ncu.py --request /w/trace/ncu-request.json --out /w/trace/ncu-launch --total-timeout 240 --sections LaunchStats
```

`run_ncu.py --check-only` validates hashes and original argument identity without invoking tools/GPU. Builder admits only successful ordinary summary, complete sealed payload/results, selected build receipt and existing binary, bounded full and actual sampled gates, independent all-three numeric passes, and held-out 30-observation confirmation. No timing selection is recomputed.

After `ncu-launch/summary.json.pass` is true only:

```
python3 /w/ncu-extension/run_ncu.py --request /w/trace/ncu-request.json --out /w/trace/ncu-dynamic --total-timeout 240 --sections LaunchStats,Occupancy,MemoryWorkloadAnalysis,SchedulerStats,InstructionStats
```

Total diagnostic budget is at most 480 seconds. Each phase first proves actual 2025.1.1 binary version/hash/options/sections and idle GPU. Discovery checks PATH, `/opt/nvidia/nsight-compute/2025.1.1/ncu`, then `/usr/local/cuda-12.8/nsight-compute-2025.1.1/ncu`; optional `--ncu PATH` selects an explicit reviewed executable. Missing sections/version/access fail separately, preserving ordinary results. Mapal profiling and complete source/numeric/import admission gates the two library legs. Counter permission is only marked demonstrated if recognized numeric hardware counters were actually exported; LaunchStats alone does not establish it.

Profiling uses `--config-file off --profile-from-start off --clock-control none --target-processes application-only` and explicit sections, selecting the harness's single `cudaProfilerStart/Stop` region. Target argv is the exact ordinary confirmation argv plus `--profile-leg NAME`; `--values-only` and `--tuning` are forbidden by the actual harness and are never added. It never passes launch-count1; every internal library kernel is retained. Separate diagnostic processes and `timing_admissible:false` forbid ordinary timing from instrumented/replayed calls. Both pre and post raw values are independently rechecked with the sealed checker/oracle; a harness completion boolean cannot override them.

Each leg preserves raw.ncu-rep, session/raw CSV, per-launch filtered raw and source-SASS pages from the same unchanged report, all command stdout/stderr/status, protocol and exact ordinary binary/input digests. Launch ID, process/context/stream/full kernel name/grid/block/metrics must match before source admission. Dynamic shared comparison accepts explicit base byte units only; unsupported units reject. This is authoritative NCU runtime/report attribution, not callback ELF bytes or a physical cubin SHA claim. Static SASS rows are not executed instruction counts. All dynamic counter values retain reported units and are diagnostic observations.

CPU fixtures are explicitly synthetic and do not prove a live NCU CSV/SASS format. The first actual Mapal smoke must pass these conservative parsers before later legs. Unsupported actual identity/format/source/counter cases remain failures, not fabricated disassembly. Existing Mapal CUPTI callback proof is separate evidence. No generic library cubin is substituted.

Outputs: fresh exclusive directory, schema `s61-ncu-results-v1`, pass, live_smoke_pass, counter_permission_demonstrated, per-leg admitted/failed/skipped_after_failure, errors, all command receipts/file seals. Exit0 only all3 legs admitted; exit1 diagnostic failure. Input preflight/CLI errors can fail before output creation; operator captures outer stdout/stderr/status. These optional failures never invalidate successful ordinary evidence.

Bounds: internal deadline <=240 seconds per approved phase, each child <=120 seconds (imports normally30), per-file128MiB and aggregate512MiB after commands, max256 launches/100000CSV rows. Own process groups are killed on error/timeout/cancellation/leader exit. Linux children use parent-death SIGKILL plus parent-race check. This has CPU subprocess coverage; GPU driver/profiler cancellation behavior remains a live proof obligation. No claim that filesystem quotas constrain total intermediate kernel-driver state.

Primary CLI reference: NVIDIA Nsight Compute 2025.1 CLI guide https://docs.nvidia.com/nsight-compute/2025.1/NsightComputeCli/index.html . No later --import-sass option is assumed. Captured Mapal callback images separately identify ptxas13.3/sm_120a while decoder is12.8; that decoder compatibility limitation must accompany static conclusions, independently of NCU's actual version/export evidence.
