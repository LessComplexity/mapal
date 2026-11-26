# Mapal — compiler optimizations and the first full LLM run (data summary for a presentation)

Period: 2026-11-19 → 2026-11-26 (Singapore time). Branch `local-rtx-5080` at `c808622c`, 285 commits since 2026-11-19.
Hardware: local AMD Threadripper 9970X + NVIDIA RTX 5080 (16 GB); rented on vast.ai: RTX 3090, A100 SXM4 40 GB, RTX 4090,
H100 SXM. Status: **final numbers.** The bf16 sprint is integrated and reviewed (`b8429388`, review fixes `9be9886d`);
the five-GPU round ran on that build (every competitor at the program boundary, the full model on every GPU in both
classes); the README rewrite with these numbers is merged (`aa372fce`, not pushed). The round's one bound miss turned out to be
the gate's bound, not the compiled program (section 6). The H100 16-bit wgmma sprint and a regression study run next.

---

## 0. Follow-up to the 2026-10-28 "TND + Mapal" update (what the previous deck said → where it stands)

The previous deck (12 slides, `TND + Mapal Update.pdf`, status update 2026-10-28) had three Mapal slides: **8 · DONE ·
"Mapal now beats libraries on attention"** (RTX 4090), **9 · NEXT · "Auto-fusion: Mapal compiles the model as one dataflow
graph"**, **12 · NEXT · "One sizing model for every site"**. This summary is the follow-up for those threads.
**TND slides (2–7, 10–11: DeepSeek V4 on B300, CUDA graph planner, radix cache) are not covered here** — that data lives
in the TND project, not in this repo; bring it from there for the TND part of the follow-up deck.

### 0.1 Slide 8 then → now (same seven rows; Mapal time ÷ best competitor; < 1 = Mapal faster)

Then: RTX 4090 only, competitors cuBLAS / CUTLASS / PyTorch / torch.compile. Now: five GPUs, and every leg in section 3
(adds torch.compile + CUDA graphs, reduce-overhead, SDPA cuDNN, FlashAttention-2/3, FlashInfer, Liger, hand-written
Triton and CUDA chains) — a much stronger bar — all timed at the program boundary (section 3.3). Rented columns: the
five-GPU round on `b8429388` (every box at its full clock and power); RTX 5080: after the gap fixes (`d945c641`).

| row (program) | Oct 28, RTX 4090 | now RTX 4090 | RTX 5080 | RTX 3090 | A100 | H100 | leader now |
|---|---:|---:|---:|---:|---:|---:|---|
| Prefill + decode loop (gen_1024_512_256) | 0.17 | 0.35 | 0.30 | **0.16** | 0.28 | 0.28 | torch.compile + CUDA graphs (A100: torch.compile) |
| Causal attention (causal_1024_256) | 0.18 | 0.35 | 0.30 | **0.15** | 0.32 | 0.23 | torch.compile + CUDA graphs (H100: reduce-overhead) |
| Attention prefill (prefill_1024_512_256) | 0.18 | 0.36 | 0.30 | **0.15** | 0.28 | 0.27 | torch.compile + CUDA graphs |
| Elementwise chain (ew_4096) | 0.46 | **0.33** | 0.35 | 0.58 | 1.07 | 1.35 | hand-written fused CUDA |
| 2-layer MLP (rt_mlp2_2048) | 0.75 | 0.92 | 0.93 | 0.96 | **0.87** | 1.09 | torch.compile + CUDA graphs (A100: torch.compile) |
| Decode (decode_1024_256) | 1.01 | 0.88 | **0.53** | 0.67 | 0.84 | 1.08 | hand-written KV-cache CUDA chain |
| Multi-channel conv (convmc_8_64_56_3) | 1.38 | 1.51 | **0.67** | 1.45 | 2.17 | 1.37 | PyTorch eager (cuDNN) |

- Attention-shaped programs: still 2.8–6.8× faster than the best competitor (0.15–0.36) on **all five** GPUs, now
  including torch.compile with CUDA graphs and the attention libraries.
- Decode: ahead on four of five GPUs (0.53–0.88) against a hand-written KV-cache CUDA decode; the H100 is 1.08
  (1.45 at f1d3a04f).
- Elementwise chain: 0.33–0.58 on the consumer GPUs; behind on the A100 / H100 (1.07 / 1.35) against hand-fused CUDA.
- **The RTX 4090 column moved with the box**: f1d3a04f's 4090 ran clock-locked at 1.2 GHz, this round's at its full
  2.45 GHz. At full clock the attention rows read 0.35–0.36 (0.14 on the locked box) and decode 0.88 (0.41); Mapal is
  still ahead on both, the per-row cause is not analysed (open). Multi-channel conv improved there (2.05 → 1.51) but
  stays an open item on four GPUs (conv window not contiguous in k; cuDNN leads).
- "Cells at or below the best competitor": Oct **26 → 54** (of 92, then 120; 4 GPUs) → 247 of 599 at the previous
  summary → now **279 of 599** program × GPU cells across 5 GPUs (RTX 5080 69/120, RTX 4090 63/120, A100 62/119,
  RTX 3090 51/120, H100 34/120) against the larger competitor set. By geomean: 5080 0.876, 4090 0.911, 3090 0.920
  (ahead), A100 1.006 (even), H100 1.101 (behind).

### 0.2 Slide 12 "NEXT · One sizing model for every site" — status of each item

| Oct 28 item (ratio then) | what was done | now |
|---|---|---|
| Sizing computed from the dataflow graph + each GPU's numbers (split work per block, last-wave fill, split-sum combine cost, cache residency); per-GPU search only settles near-ties | **Done**: hardware-as-properties rules (no per-GPU tables, no per-architecture switches); properties measured per GPU (matrix rates, barrier cycles, latency, L2 keep, broadcast loads, split cell cost); stream-K / split pricing; residence_plan; every measurement replayed against the rules | rules generalize to unmeasured GPUs; 279/599 cells at or below the best |
| Decode on H100 (1.51) — pick how split sums combine | Split launches for the decode's GEMV epilogue; combine priced in ns (18/20 measured arms right) | H100 B1 decode step 0.326 → 0.110 ms; this round the H100's B1 decode rows 0.41–0.75 and its LLM family 0.975 |
| Small sizes, 512 (1.5–2.2) — per-kernel overhead, block size | Host path out of the clock (site picks at module load, zero kernel instead of memset API, first-launch attributes at load), the fair timing boundary | RTX 5080 0.85–0.97 (ahead); re-timed on all GPUs at the program boundary: the boundary moves each GPU 5–11 % in Mapal's favour (section 3.3) |
| Matmul on 5090/H100 (1.1–1.9) — sizing model instead of a 4090-tuned table | Property-based tile/stream-K/raster rules; Hopper warp-specialized wgmma + TMA (per-call overhead 120–270 µs → 2–15 µs); tf32x3 on A100 | GEMM family: 5080 0.93, 4090 0.94, 3090 1.05, A100 1.16, H100 1.37 — the weak family; H100 still open (cuBLAS's persistent 128×256 kernel leads); no 5090 run since |
| Narrow matrix × vector (0.42–1.86) — last-wave fill, cache | GEMV in-flight sizing, evict-first loads for read-once weights, the zero pass off the host path | m1 projections: 5080 0.31–1.28, 4090 0.26–1.21, 3090 0.88–1.09, A100 0.78–1.23, H100 0.92–1.44; m16: 5080 0.55–1.02, H100 0.88–1.86 |
| Attention, other sizes (1.1–2.1) — more threads per row | Streamed (flash) attention with split-KV, row trees with shuffle reductions | attention family geomean 0.52 (3090), 0.63 (A100), 0.64 (5080), 0.65 (4090), 0.79 (H100) — ahead on every GPU; attn_* rows 0.61–1.88 (behind on the A100 / H100 / 3090 at one size each) |
| TF32 vs PyTorch TF32 (1.06–2.62) — TF32 everywhere allowed | tf32 tiles + split forms priced at their issue rate | still open: tf32 rows geomean 1.15 (5080, 3090), 1.21 (4090), 1.65 (H100), 1.72 (A100) |
| "A fair check against FlashAttention / vLLM kernels on the same prefill, decode and causal shapes" | **Done**: FlashAttention-2/3, FlashInfer, Liger, SDPA (efficient / cuDNN) in every round; and a full model against vLLM, SGLang, TensorRT-LLM, llama.cpp, gpt-fast | attention still ahead on every GPU; full model: section 5 |

### 0.3 Slide 9 "NEXT · Auto-fusion: Mapal compiles the model as one dataflow graph" — status

- **Mapal side: done to the point of a whole model.** A full LLM (Llama-3.2-1B, real weights) is one readable Mapal
  program compiled as one dataflow graph: kernels merged at compile time (vertical through calls, horizontal across
  data), sized per GPU from properties, the KV cache placed and updated in place on the GPU automatically. Section 5.
- **Not started: the TND hand-off** (TND's DeepSeek V4/V4.1 forward → Mapal graph → TND's graph planner) and its
  targets from slide 9 — V4 layer-mixing prompt step (60.4 ms vs DeepGEMM 18.4), MoE experts at 256 requests (495 vs
  718 tok/s), decode at 14k context (11.2 vs 7.9 ms), Qwen2.5 MLP + norm decode on A100 (1.4–1.9× behind SGLang). Mapal
  has no MoE / sparse-attention / selector programs yet; these need the PyTorch importer (next steps) or hand-written
  Mapal programs, and batch > 1 / serving features.

### 0.35 Between the two decks

- 2026-10-28 → 2026-11-19 (previous machine, 43 commits): `mapal fmt` source formatter with editor support (Oct 29);
  raw run output moved to GitHub releases with a manifest (Oct 29); site-geometry merge (Oct 31); cost-model plans
  v2 with calibration and a blind validation set, tile-efficiency kernel evidence on 4090/5090 (Nov 9); per-site measured
  choice of kernel candidates (Nov 10).
- 2026-11-19 → 2026-11-26 (this machine + rented GPUs, 285 commits): everything in sections 3–7.

### 0.4 Then → now in one line each

- Oct 28: "beats libraries on attention" on one GPU vs cuBLAS/CUTLASS/PyTorch/torch.compile → now: ahead on attention on
  five GPUs vs every attention library and CUDA-graph torch.compile, and a full LLM written in Mapal that beats every
  f32 engine (vLLM, gpt-fast, llama.cpp, Hugging Face) at the same precision on decode on all five GPUs; in bf16 it is
  within 5–15 % of the best engine on the RTX 5080 and 1.16–2.37× behind on the rented GPUs.
- Oct 28: "one sizing model" was a plan "not yet approved" → now implemented as property-based rules on five GPUs.
- Oct 28: "auto-fusion of the model" was next → whole-model compilation works for Llama; the DeepSeek / TND hand-off is
  still ahead.

---

## 1. What Mapal is (one slide)

- A dataflow language and compiler. **The language is the flow:** a program states data and how it flows
  (`data -> transform -> result;`), not control. Data is category-theory based (objects, morphisms; Mapal-Cat).
- **Thesis:** expressing data and its flow beats expressing control — the compiler derives what runs in parallel, what
  merges into one GPU kernel, what stays in registers / shared memory / cache, and what moves between devices.
- **Hardware is physical properties, never names:** decisions read measured/queried properties (SM count, bandwidths,
  latencies, matrix-instruction shapes and rates, cache sizes); a new chip is a new property set.
- **No runtime graph capture:** GPU work is merged at compile time — vertically (through calls) and horizontally (across
  data).
- **Correct numerics — the flow and the bound, not bit-identity (ADR-0049):** every value within an error bound stated
  before the program runs; the compiler orders and combines operations freely for speed and placement, so runs may
  differ in the last bits; bit-reproducible runs are a development mode (testing, debugging), not a language promise.
- **Readable source:** if natural code compiles worse, the compiler is generalized — never the program.
- **A language AI agents can write and verify:** deterministic semantics, precise errors, a formatter, and the
  interpreter as an exact reference to check output against. (Most of this week's code was written by AI agents.)

---

## 2. Headline results

1. **Full LLM (Llama-3.2-1B-Instruct, real weights, readable Mapal source), f32 class, all five GPUs:** Mapal is faster
   than every f32-compute engine on decode on every GPU (**0.55–0.95×** the best engine's time) and on time to first
   token from a 512-token prompt up (**0.43–0.95×**); at a 128-token prompt it is 4 % behind on the RTX 3090 and H100.
   RTX 5080: TTFT **1.60 / 1.62 / 1.87× faster** (128 / 512 / 2048), decode **1.71–1.80× faster**, at about half the
   memory. Tokens 128/128 identical to Hugging Face f32 on all five GPUs.
2. **Same model, bf16 class (what engines deploy by default):** behind the fastest bf16 engine on every GPU. RTX 5080:
   **1.05–1.15× on TTFT, 1.06–1.10× on decode** (1.31–1.63× / ~1.19× before the bf16 sprint). Rented: RTX 3090
   1.17–1.55× / 1.23–1.25×, RTX 4090 1.16–1.44× / 1.18–1.19×, A100 1.44–2.12× / 1.55–1.57×, H100 1.30–2.37× /
   1.49–1.56× (TTFT / decode; best: gpt-fast, SGLang, TensorRT-LLM).
3. **Kernel/workload suite (119 programs × 5 GPUs vs the fastest competitor at the same precision, both sides at the
   program boundary):** geomean **0.876** RTX 5080 · **0.920** RTX 3090 · 1.006 A100 · **0.911** RTX 4090 · 1.101 H100
   (Mapal time / best competitor time; < 1 = Mapal faster). Win / even / loss: 46/44/30 · 39/30/51 · 57/11/51 ·
   49/25/46 · 30/8/82. Ahead on the three consumer GPUs, even on the A100, behind on the H100; GEMM is the weak family.
4. **Correctness:** every timed program passes its value gate on every GPU. The one apparent bound miss
   (`px_llm_blkdec_b2_l16_bf16`, 2.30·10⁻² vs a gate of 1.62·10⁻²) was the gate's fault: the compiled result is bit
   for bit the program with the bf16 stores its `: bf16 | f32` permits; the gate's one-product bound was unsound for a
   chain of bf16 products and is replaced by a bound composed through every operation (ADR-0041 amended; `mapal run
   --interp --bound`). Numerics per ADR-0049: correct = within the stated bound, order free; bit-reproducible runs are a
   development mode.
5. **Cost:** this round's rentals $30.09 (four GPUs, one night); all rented rounds together about $101.

---

## 3. Kernel/workload suite — Mapal vs the fastest competitor, same precision

**Competitors (every leg, the fastest per row counts):** cuBLAS, cuBLASLt, CUTLASS, hand-written CUDA, PyTorch eager,
torch.compile, torch.compile + CUDA graphs (max-autotune), torch.compile reduce-overhead, SDPA (math / efficient / cuDNN),
hand-written Triton, Liger, FlashAttention-2 / 3, FlashInfer. TensorRT-LLM did not install on the kernel-suite stack
(torch 2.11).

**Metric:** geometric mean over the 119 programs timed on every GPU of (Mapal ms / fastest competitor ms). **< 1 = Mapal
faster.** Families: GEMM cells and projections, MLP chains, attention / softmax / long rows, LLM decode / prefill /
blocks (f32, tf32, bf16 storage), conv / FIR / elementwise, fan-in/out + loops + CPU+GPU. Precisions: f32, tf32, bf16
weights, bf16 KV cache, bf16 storage, fp8 weights (no competitor leg yet).

### 3.1 Progress per round (all rows)

| round (commit) | RTX 5080 | RTX 3090 | A100 | RTX 4090 | H100 |
|---|---:|---:|---:|---:|---:|
| 3a8e654f (2026-11-23) | 1.076 | 1.127 | 1.501 | 1.106 | 1.566 |
| 5fcb401b (2026-11-24) | 0.979 | 1.009 | 1.242 | 0.948 | 1.318 |
| f1d3a04f (2026-11-24/25) | 0.978 | 0.997 | 1.185 | 0.954 | 1.204 |
| gap-fixes (2026-11-25, 5080 only, 120 rows) | 0.878 | – | – | – | – |
| **five-GPU round, b8429388 / 9be9886d (2026-11-26), program boundary** | **0.876** | **0.920** | **1.006** | **0.911** | **1.101** |
| the same round on the old event bracket | – | 1.011 | 1.134 | 0.964 | 1.233 |

The 5080 column of the five-GPU round is the gap-fix integration (`d945c641`, 119 common rows); the rented columns are
`b8429388` (phase C on `9be9886d`: emission identical, null by construction). Win / even / loss now: 5080 46/44/30 ·
3090 39/30/51 · A100 57/11/51 · 4090 49/25/46 · H100 30/8/82. Box conditions: no box throttled; every part ran at its
full clock and power (f1d3a04f's 4090 was locked at 1.2 GHz, its H100 at 500 W). Host noise: the A100's and H100's
host-bound legs moved ±30–70 % between phases on the same box (the RTX parts within ±5 %).

### 3.2 Per family / precision (five-GPU round)

| geomean Mapal/best | RTX 5080 | RTX 3090 | A100 | RTX 4090 | H100 |
|---|---:|---:|---:|---:|---:|
| GEMM cells & projections | 0.926 | 1.048 | 1.161 | 0.935 | 1.373 |
| MLP chains | 0.974 | 1.094 | 1.075 | 0.945 | 1.285 |
| Attention / softmax / long rows | **0.639** | **0.517** | **0.628** | **0.647** | **0.790** |
| LLM decode / prefill / block | 0.987 | 0.973 | 1.013 | 1.053 | 0.975 |
| Conv / FIR / elementwise | **0.493** | **0.695** | 0.944 | **0.646** | **0.805** |
| Fan-in/out, loops, CPU+GPU | 0.979 | 1.030 | 0.977 | 0.957 | 1.108 |
| f32 rows (66) | 0.772 | 0.801 | 0.819 | 0.807 | 0.902 |
| tf32 rows (13) | 1.154 | 1.149 | 1.716 | 1.208 | 1.646 |
| bf16-weight rows (9) | 1.559 | 1.463 | 1.575 | 1.634 | 1.581 |
| bf16 KV-cache rows (7) | 0.971 | 0.935 | 0.919 | 1.097 | 0.896 |
| bf16 storage rows (25) | 0.846 | 1.005 | 1.135 | 0.835 | 1.409 |

Against f1d3a04f (event bracket then, so part of each move is the boundary, 3.3): the A100 and H100 improved in every
family and precision (largest: A100 GEMM 1.389 → 1.161, bf16 storage 1.423 → 1.135, tf32 2.087 → 1.716; H100 bf16
storage 1.620 → 1.409). The 3090 improved in most; its LLM (0.962 → 0.973), fan (1.026 → 1.030), tf32 (1.139 → 1.149)
and bf16-weight rows (1.419 → 1.463) moved the other way by 1–3 %. **The 4090 read worse on three groups** on this
round's full-clock box: LLM 0.903 → 1.053, bf16 weights 1.377 → 1.634, bf16 KV cache 0.863 → 1.097 — not root-caused
(open, section 10).

### 3.3 The fair timing boundary (gap-fix sprint) and the code's share

Competitors used to be timed with GPU events that excluded their host-side launch/Python dispatch, while Mapal always
timed itself the way a user waits (host clock around call + synchronize). Now every leg is timed the same way.

| RTX 5080, 120 rows | before | after |
|---|---:|---:|
| Mapal code only (new Mapal time / old) | — | 0.949 (≈5% faster) |
| Mapal vs competitors, old timing | 0.978 | 0.928 |
| Mapal vs competitors, same timing for both | — | **0.878** |

| rented GPUs, all rows | RTX 3090 | A100 | RTX 4090 | H100 |
|---|---:|---:|---:|---:|
| f1d3a04f, event bracket | 0.997 | 1.185 | 0.954 (1.2 GHz box) | 1.204 (500 W box) |
| b8429388, event bracket — **the code's share** | 1.011 | 1.134 | 0.964 | 1.233 |
| b8429388, program boundary — **plus the boundary's share** | **0.920** | **1.006** | **0.911** | **1.101** |

Like for like (same bracket, same harness): the A100 improved to 0.954× of its ratio on the code; the 3090, 4090 and
H100 are even (1.01×, 1.01×, 1.02×; the 4090 and H100 boxes also changed). The program boundary then moves every GPU
5–11 % in Mapal's favour (launch-bound torch legs).

---

## 4. Compiler optimizations delivered (changes → measured effect)

| area | what changed | measured effect |
|---|---|---|
| Compile-time kernel merging | Vertical (persistent kernels through calls) and horizontal (more rows per kernel, independent sites in one launch); facts: access_plan, residence_plan, unit_plan, write_delta | decode block 218 → 0.67 ms; decode 0.66×, causal/generation 0.77× of before |
| Hardware as properties | Removed per-architecture on/off switches; rules read measured properties (matrix rates, barrier cycles, latency, L2 keep, broadcast-load rate); calibration per GPU; every measurement replayed against the rules | rules now generalize to unmeasured parts; fp8 on RTX 4090: 16/55 → 55/55 modules assemble (PTX ISA as a capability) |
| Tensor cores | bf16/f16 mma tiles; tf32 split forms (tf32x3) priced at their issue rate; bf16 weights × f32 activations on tensor cores (bf16x2n/x3n splits); Hopper warp-specialized wgmma + TMA | A100 tf32x3 1.44–1.70× over the FFMA tile (20 A100 losses closed); bf16-weight prefills A100 1.95–2.42×, H100 1.20–1.29×; H100 per-call overhead 120–270 µs → 2–15 µs (9 losses → 4 wins / 8 even) |
| Flash attention | Streamed single-pass attention (stream_plan), split-KV, online-softmax rescale law (exp builtin) | 8192-token prefill 1065 → 47.9 ms (vs Liger 53.2), peak memory 0.41× torch; A100 / H100 long rows vs Liger 3.15 → 1.59 / 3.35 → 1.28 |
| Automatic caching | unchanged_plan / call_keep: values equal across calls/trips retained; automatic KV cache at model scale | KV-cache decode rows at or below the hand-written cache |
| Gap analysis by disassembly | 51 losing rows on the 5080 compared kernel-by-kernel with the leader's actual SASS (CUPTI timelines) → 14 causes | 3 parallel fix branches: host path out of the clock (picks at load, zero kernel instead of memset), split-combine pricing, deeper mma ring, raster by b-panel reuse, closed-form index values, shuffle-tree reductions, evict-first loads, bank-conflict-free swizzle → 5080 0.978 → 0.878 |
| Full-model compiler work | Arrays inside structs by address; nested loops (through calls); placement by bytes moved; partial array writes in place on the GPU (straight-line and loops); one device copy of every value; flash mask through position arrays; split-tile cost vs the FFMA tile's achieved rate | Llama: whole run 18.6 s → 1.3–1.5 s, now 441 ms end to end (f32, 512-token prompt, 128 tokens); decode 679.5 → 3.29 ms/token; TTFT@512 85 → 28.9 → **25.4 ms** (f32) / 18.1 → **14.1 ms** (bf16) on the readable program; peak memory 5.1 → 2.7–3.3 GB |
| **bf16 sprint: tensor-tile walk** (`mma-walk`) | A 16-bit tensor tile whose grid rounds unevenly over the SMs is walked by stream-K over every resident slot (whole tiles stored, shared tiles' partials added with `red`); priced from properties only (SMs, matrix-pipe rate, atomic rate, zero pass, DRAM rate) | isolated projections at **99–104 % of the calibrated 105.4 Tflop/s pipe** (down 512×2048×8192 216 → 164 µs, 2048×2048×8192 754 → 627 µs); in the model 90 % at 512, 94–106 % at 2048; bf16 TTFT@512 −1.0 ms; SASS checked by `evidence.py` (232 checks, 0 mismatch) |
| **bf16 sprint: causal skip** | Attention skips fully masked blocks when the cache's length is only known at run time (a run-time-finite cache) | attention at 2048 **16.3 → 7.9 ms** (bf16), 34.1 → 16.3 ms (f32); f32 prompt kernels at 2048 122.6 → 96.1 ms |
| **bf16 sprint: bf16 KV cache, stored rounding** (`narrow-window`, `round-post`, `round-store` on epilogues) | The bf16 program's cache stored bf16 (rounded once on write, widened on read); arrays read only as bf16 multiplicands stored rounded by their producer (now the attention output) | peak at 2048 3 498 → 2 954 MB; the decode reads half the cache bytes; query + output 8.16 → 7.02 ms at 2048 |
| **bf16 sprint: device gather + placement** (`device-trap`, `cand-made`) | The embedding gather runs on the device behind a device-side index check that keeps the oracle's trap (a copy-out is a checkpoint, from the review); a map whose operand a device candidate makes is a candidate itself | the prompt's host chain gone: host gaps 1.45 → 0.06 ms (f32, 512), 2.5 → 0.09 ms (bf16, 512) |
| **Decode residency** (calibrated placement, `carry-induct`, `fly-unroll`) | The KV cache's device residence through the decode loop's back edge; calibrated launch prices | the decode step is kernel-bound now: 3.21 ms of kernels, 324 launches, 0.03 ms of host gaps (2.47 GB of weights at 890 GB/s = 2.78 ms) |

---

## 5. The full LLM run

**Model:** Llama-3.2-1B-Instruct (hidden 2048, 16 layers, 32 query / 8 KV heads, head 64, SwiGLU 8192, RoPE θ=500000 with
llama3 scaling, vocab 128256, tied embeddings), real weights, bf16 checkpoint. Fits all five GPUs.

**Program:** `examples/llama.mapal` — 404 lines, readable (descriptive names, small functions, comments). Uses:
top-level size bindings (`2048 -> hidden_size;`), `Matrix<rows, columns>` types, per-layer weight structs and a layer
loop, one size-generic `forward<tokens>` for both prompt and generation, `-> sqrt / sin / cos / exp`, and
`"model.bin" -> read -> model: Model;` for the weights. The same text runs on the interpreter, CPU and GPU.

**Setup:** five GPUs (RTX 5080 local; RTX 3090, A100, RTX 4090, H100 rented), batch 1, prompts of 128 / 512 / 2048
tokens, 128 generated tokens (greedy). Time to first token (TTFT) = submit → first token on the host; decode = (t₁₂₈ −
t₂) / 126 ms per token; warmup 2 + 5 repetitions; medians. Mapal build: `b8429388` (5080: `3c629486`, the same
emission), re-timed on the final runtime `9be9886d` (within 0.2 % on the 5080; final / integrated 0.96–1.01 on the
rented GPUs, the A100's host noise aside).
**Competitors:** gpt-fast (max-autotune + CUDA graphs), vLLM 0.33, SGLang 0.5.23, TensorRT-LLM 1.3, llama.cpp (CUDA),
Hugging Face transformers 5.22 (eager, static cache + compile).

**Rule: compare type to type.** bf16 class only against bf16 engines; f32 class only against f32-compute engines.
llama.cpp in f16 is another 16-bit format: recorded, compared with neither class.

### 5.1 f32 class, RTX 5080 (f32 math; competitors use the checkpoint converted to f32, TF32 off)

| engine | TTFT ms (128 / 512 / 2048) | decode ms/token (128 / 512 / 2048) | peak MB (128 / 512 / 2048) | tokens vs HF f32 |
|---|---|---|---|---|
| **Mapal** | **8.54 / 25.4 / 95.5** | **3.24 / 3.29 / 3.45** | **2 722 / 2 730 / 3 018** | 128/128 |
| gpt-fast | 13.7 / 47.7 / 213.7 | 5.85 / 5.72 / 5.90 | 6 296 / 6 626 / 7 688 | 128/128 |
| vLLM | 14.6 / 41.0 / 178.8 | 6.23 / 5.94 / 6.22 | 9 188 / 9 188 / 9 236 (reserved) | 128/128 |
| llama.cpp | 14.2 / 41.3 / 203.9 | 6.03 / 6.31 / 7.75 | 5 148 / 5 200 / 5 396 | 128/128 |
| Hugging Face eager | 14.5 / 41.7 / 263.5 | 7.73 / 7.43 / 8.84 | 5 104 / 5 204 / 6 494 | reference |
| Hugging Face static + compile | 16.9 / 44.2 / 244.8 | 7.39 / 7.50 / 10.42 | 5 202 / 5 306 / 6 782 | 128/128 |
| SGLang | no f32 support (kernels are bf16/fp16 only) | — | — | — |
| TensorRT-LLM | runs, but f32 output is wrong (0/128 tokens) — excluded | — | — | — |

Mapal ÷ best engine: TTFT **0.62 / 0.62 / 0.53** (1.60 / 1.62 / 1.87× faster), decode **0.55 / 0.58 / 0.58** (1.80 /
1.74 / 1.71× faster), end to end 1.75× faster at 512, at about half the lowest engine's memory. Previous integration →
now: TTFT 10.4 / 28.9 / 125.3 → 8.54 / 25.4 / 95.5 ms, decode 3.76 / 3.59 / 4.00 → 3.24 / 3.29 / 3.45 ms/token.
**Why decode leads:** decode is memory-bandwidth bound; the f32 engines stream a 4.9 GB converted copy of the weights
per token, Mapal streams the 2.47 GB of bf16 bytes and widens them exactly in registers (the dataflow facts prove the
widening exact) — same values, half the bytes.

### 5.2 bf16 class, RTX 5080 (bf16 operands, f32 accumulation on the sums of products; bf16 KV cache)

| engine | TTFT ms (128 / 512 / 2048) | decode ms/token (128 / 512 / 2048) | peak MB (128 / 512 / 2048) |
|---|---|---|---|
| **Mapal** | 5.58 / 14.1 / 52.1 | 3.26 / 3.29 / 3.40 | **2 652** / 2 870 / 3 228 |
| gpt-fast | **4.94** / 13.7 / 56.1 | **2.97 / 2.98** / 3.33 | 3 410 / 3 532 / 4 074 |
| SGLang | 6.83 / 13.8 / 49.3 | 3.34 / 3.08 / 3.51 | 8 902 / 8 838 / 8 906 (reserved) |
| TensorRT-LLM | 9.49 / 16.0 / 45.7 | 3.31 / 3.49 / **3.19** | 8 810 / 8 790 / 8 784 (reserved) |
| vLLM | 6.79 / 14.9 / **45.3** | 3.49 / 3.51 / 3.24 | 9 236 (reserved) |
| llama.cpp (bf16) | 5.61 / **13.5** / 49.2 | 3.16 / 3.17 / 3.23 | 2 804 / 2 846 / **2 896** |
| Hugging Face eager | 6.63 / 15.5 / 56.0 | 4.67 / 4.74 / 5.03 | 2 726 / **2 780** / 2 962 |
| Hugging Face static + compile | 8.81 / 17.6 / 56.1 | 4.04 / 4.56 / 6.03 | 2 822 / 2 872 / 3 090 |

Mapal ÷ best: TTFT **1.13 / 1.05 / 1.15** (was 1.42 / 1.31 / 1.63 before the bf16 sprint), decode **1.10 / 1.10 / 1.06**
(was ~1.19). At 128 Mapal ties llama.cpp bf16 and is ahead of SGLang, vLLM, TensorRT-LLM and both HF; at 512 within
2–5 % of gpt-fast, SGLang and llama.cpp; at 2048 ahead of gpt-fast and both HF. Previous integration → now: TTFT 6.95 /
18.1 / 74.1 → 5.58 / 14.1 / 52.1 ms, decode 3.53 / 3.67 / 4.01 → 3.26 / 3.29 / 3.40 ms/token. Peak memory the lowest
of every engine at 128. **Tokens** (vs HF bf16 eager): 128 / 32 / 61 — inside the bf16 engines' own spread (7–89 at
512 among themselves; HF's own f32 and bf16 runs share 128 / 32 / 61). At 512 Mapal's tokens vary run to run (4
sequences in 5): the tensor-tile walk adds a tile's 3+ partials in arrival order (ADR-0049; deterministic with
`--gpu-off=mma-walk`).

### 5.3 Mapal's progress on this model, 2026-11-25 → 26 (RTX 5080)

| stage | whole run | TTFT @512 | decode ms/token | notes |
|---|---|---|---|---|
| first readable compile (scout's program, synthetic weights) | 18.6 s | 587 ms | 679.5 | small steps on the CPU, 570 MB copied per token |
| model-ready sprint (scout's program → the readable `examples/llama.mapal`, real weights) | 1.3–1.5 s | 187–204 → 85 ms | 4.9 → 3.34–3.55 | placement by bytes moved, cache stays on the GPU, real weights checked 128/128 vs HF f32 |
| prefill fixes (f32 class) | — | 28.9 ms | ~3.6–4.0 | in-place cache writes, tensor-core projections, flash mask, one weight copy |
| prefill fixes (bf16 class) | — | 18.1 ms | 3.5–4.0 | bf16 tensor tiles, bf16 attention |
| **bf16 sprint + integration, final** (`b8429388` / `9be9886d`) | 441 ms (f32) / 429 ms (bf16) | **25.4 ms** (f32) / **14.1 ms** (bf16) | **3.29** (both) | tensor-tile walk, causal skip, bf16 KV cache, device gather, calibrated decode placement |

### 5.4 The bf16 sprint: what it closed, what remains (RTX 5080, CUPTI)

| cause (before the sprint) | done | now |
|---|---|---|
| Tensor tiles at 77–81 % of peak (128/256 tiles round unevenly over 84 SMs) | stream-K tensor-tile walk (`mma-walk`) | projections 90 % (512) to 94–106 % (2048) of the pipe; isolated 99–104 % |
| Attention output and KV cache f32 while the engines keep bf16 | stored rounding after attention; bf16 KV cache | query + output 8.16 → 7.02 ms at 2048; peak 3 498 → 2 954 MB; decode cache bytes halved |
| Embedding lookup on the host (2.5 of 18.1 ms) | device-side gather with an index check; `cand-made` | host gaps 2.5 → 0.09 ms (bf16, 512) |
| Attention at 2048 computes fully masked blocks | exact causal skip on a run-time-finite cache | attention 16.3 → 7.9 ms |
| Decode launch-bound (288 launches a step) | calibrated placement, residency | kernel-bound: 3.21 ms of kernels, 0.03 ms of gaps; compile-time launch merging of q/k/v still open |

**What remains on the 5080** (`full-model.md`): (1) TTFT at 2048 (52.1 vs vLLM 45.3): attention 7.9 ms where
FlashAttention-class kernels do the causal half in ~3 ms (the stream's geometry at 2048 is 32-row blocks × 16-key
windows, 64 threads), and 4.9 ms of f32 elementwise cells (norms, RoPE, SiLU·up, residuals) the engines keep in bf16;
(2) TTFT at 128 (5.58 vs 4.94): the projections are 4.35 of 5.42 ms, 128-row tiles fill the 84 SMs in partial rounds
and nothing walks; (3) decode (3.26–3.40 vs 2.97): ~240 small kernels besides the GEMVs; (4) the walk's side effects:
run-dependent tokens at 512 and +170 MB peak (each walked site's workspace its own buffer).

### 5.5 The full model on five GPUs — Mapal vs the fastest engine of its class (absolute times)

f32 class (every f32-compute engine; SGLang and TensorRT-LLM not in the class):

| GPU | TTFT ms, Mapal (128 / 512 / 2048) | fastest engine | Mapal ÷ fastest | decode ms/token, Mapal | fastest engine | Mapal ÷ fastest |
|---|---|---|---|---|---|---|
| RTX 5080 | 8.54 / 25.4 / 95.5 | 13.7 gpt-fast / 41.0 vLLM / 179 vLLM | **0.62 / 0.62 / 0.53** | 3.24 / 3.29 / 3.45 | 5.85 / 5.72 / 5.90 gpt-fast | **0.55 / 0.58 / 0.58** |
| RTX 3090 | 19.4 / 56.0 / 226 | 18.7 / 58.8 / 275 llama.cpp | 1.04 / 0.95 / **0.82** | 3.97 / 4.04 / 4.22 | 6.07 / 6.15 / 6.30 gpt-fast | **0.65 / 0.66 / 0.67** |
| A100 | 16.2 / 40.0 / 137 | 22.1 llama.cpp / 76.8 vLLM / 316 vLLM | **0.73 / 0.52 / 0.43** | 3.56 / 3.82 / 3.89 | 4.14 / 4.21 / 4.37 gpt-fast | **0.86 / 0.91 / 0.89** |
| RTX 4090 | 7.42 / 20.1 / 75.7 | 10.3 gpt-fast / 31.6 vLLM / 143 vLLM | **0.72 / 0.64 / 0.53** | 3.42 / 3.47 / 3.61 | 5.53 / 5.54 / 5.71 gpt-fast | **0.62 / 0.63 / 0.63** |
| H100 | 10.8 / 21.8 / 60.9 | 10.4 gpt-fast / 29.6 HF eager / 122 llama.cpp | 1.04 / **0.74 / 0.50** | 2.05 / 2.06 / 2.14 | 2.22 / 2.16 / 2.26 gpt-fast | **0.93** / 0.95 / 0.95 |

bf16 class (every bf16 engine; llama.cpp f16 not in the class):

| GPU | TTFT ms, Mapal (128 / 512 / 2048) | fastest engine | Mapal ÷ fastest | decode ms/token, Mapal | fastest engine | Mapal ÷ fastest |
|---|---|---|---|---|---|---|
| RTX 5080 | 5.58 / 14.1 / 52.1 | 4.94 gpt-fast / 13.5 llama.cpp bf16 / 45.3 vLLM | 1.13 / 1.05 / 1.15 | 3.26 / 3.29 / 3.40 | 2.97 gpt-fast / 2.98 gpt-fast / 3.19 TensorRT-LLM | 1.10 / 1.10 / 1.06 |
| RTX 3090 | 10.6 / 23.8 / 91.6 | 6.86 gpt-fast / 20.4 SGLang / 70.3 TensorRT-LLM | 1.55 / 1.17 / 1.30 | 3.96 / 4.01 / 4.10 | 3.23 / 3.27 / 3.30 gpt-fast | 1.23 / 1.23 / 1.25 |
| A100 | 8.57 / 16.1 / 59.8 | 4.58 gpt-fast / 11.1 SGLang / 28.2 TensorRT-LLM | 1.87 / 1.44 / 2.12 | 3.66 / 3.69 / 3.77 | 2.35 / 2.38 / 2.39 gpt-fast | 1.56 / 1.55 / 1.57 |
| RTX 4090 | 6.55 / 11.9 / 41.6 | 4.54 / 10.3 gpt-fast / 33.7 TensorRT-LLM | 1.44 / 1.16 / 1.24 | 3.43 / 3.47 / 3.56 | 2.92 / 2.93 / 2.99 gpt-fast | 1.18 / 1.19 / 1.19 |
| H100 | 4.13 / 7.60 / 25.3 | 3.18 / 4.04 / 10.7 gpt-fast | 1.30 / 1.88 / 2.37 | 2.04 / 2.06 / 2.17 | 1.37 / 1.32 / 1.44 gpt-fast | 1.49 / 1.56 / 1.51 |

Tokens vs the same-class HF reference on the same box: f32 128 / 128 / 128 on every GPU; bf16 49 / 128 / 66 (3090),
49 / 32 / 57 (A100, varying run to run at 512 and 2048: the walk's arrival order), 49 / 7 / 61 (4090), 43 / 7 / 66
(H100), 128 / 32 / 61 (5080). Peak memory: f32 class 2.7–3.5 GB against 5.1 GB and more for every f32 engine; bf16
class 2.7–3.3 GB, level with llama.cpp and HF eager, against 8.8–47 GB for the pooled servers (3090: not recorded).
**Reading:** the f32 class leads where the bytes decide (decode, long prompts); the bf16 class trails most on the
datacenter parts, where gpt-fast's single-graph decode step (1.32–2.39 ms) and the TensorRT-LLM / SGLang prefill are
strongest — Mapal's decode step is 2.0–4.2 ms on every part, the launch-bound share largest on the fastest GPUs.

---

## 6. Correctness and engineering discipline

- **What correct means (ADR-0049, 2026-11-26, the user's decision):** a program is correct when its flow of operations
  is the program's dataflow and every value lies within the stated a-priori bound. The compiler orders, splits and
  combines operations freely (atomics, stream-K, split-K), so results may differ in the last bits between runs and
  targets; bit-reproducible runs (the exact / contract / fixed modes) are development modes for testing and debugging,
  not a language promise. Prompted by the tensor-tile walk, whose arrival-order sums make the bf16 class's greedy
  tokens vary run to run at near ties (RTX 5080 at 512, A100 at 512 / 2048) — every run within the bound.
- **Value gates on every run** (the default face within the stated bound; in the development modes, bit for bit with
  the interpreter, the oracle). Five-GPU round: probe gates 137 / 137 programs on every rented GPU, 35 / 35 on the 5080;
  device-exact oracles 136 / 136; split programs 8 / 8; differential 86–124 programs a part.
- **The apparent bound miss, resolved:** `px_llm_blkdec_b2_l16_bf16` read 2.30·10⁻² against a gate of 1.62·10⁻² on the four
  rented GPUs (identical value everywhere). Root cause: round-store keeps five arrays (`xn`, `q`, `att`, `hn`, `hh`) in
  bf16, every one permitted by the program's `: bf16 | f32`; the interpreter on that stored graph prints the device's
  bits. The gate allowed one product's two roundings for the whole program — unsound for a chain of bf16 products over
  three decode steps with a small value from cancellation. Fix (`c8e1966f`): a composed first-order bound per printed
  value (`mapal_interp::run_bound`); the program is inside it; 32 generated decode blocks test it (5 of 32 fail the old
  bound). Open: the composed bound is ~50× the actual error here (sound but loose); tightening it (running error
  analysis with actual magnitudes) is open work, and whether f32 gates switch to it (≈20× looser) is undecided.
- **The integration's review found three real issues, all fixed with tests that fail without the fix:** (1) the device
  index check could hang where the oracle traps — a copy-out to the host now raises the recorded trap (`index_oob`,
  rc 101, as the oracle); (2) the evidence checker misread a confirm-revert line and the runtime's head-to-head vetoes
  (103 → 84 mismatches; the remaining 202 rows of the full-suite run are pre-existing, untriaged tile-kernel classes);
  (3) the full-suite safety net on both runtimes: 34 / 35 programs null, the 35th bimodal on both arms.
- **Miscompiles found and fixed this week** (each with a test that fails without the fix): a hoisted memory clear
  racing a merged launch (zero-hoist × horizontal × arena); bf16 storage leaking through a shared function body;
  two GPU variants of one body under one name (wrong kernel called → zeros); flash attention's row numbering on the
  token-major layout; tensor-core attention reading the wrong rows; ±inf in a bf16 operand giving NaN in split tiles;
  plus earlier: masked row-slices NaN, a carried array returning its initial value, a callee's device loop returning
  zeros, an epilogue's escaping output never stored.
- **Tests:** full workspace on the integrated tree: 165 test binaries, **1 716 passed, 0 failed**, 10 ignored;
  generated-program property tests (random programs never seen during design) for every new rule (`mma_walk`,
  `flash_attention`, `half_tile`, `onto_device`, `placement`); four emulator modes; every rule replayed against every
  measurement.
- **Evidence pass:** every keep/remove decision cites the PTX/SASS difference that explains the timing; independent
  reviews corrected overstated claims (e.g. a 5080 figure revised 0.965 → 0.978; an "11/12" corrected from "12/12";
  HF f32 vs HF bf16 token counts re-checked against the raw tokens).

---

## 7. Language additions this week (ADRs)

| ADR | feature | syntax |
|---|---|---|
| 0042 | `exp` builtin (exact reference algorithm, bounded hardware form, online-softmax law) | `x -> exp` |
| 0043 | bf16 / f16 / fp8 storage types | `[bf16; n]`, `x -> to_bf16` |
| 0045 | load data from files | `"model.bin" -> read -> model: Model;` |
| 0046 | `sqrt`, `sin`, `cos` + one builtin-realization framework: contract + portable reference definition + per-target overrides admitted by the contract (ready for ASIC targets) | `x -> sqrt`, `angle -> sin` |
| 0047 | size-generic functions, sizes inferred at each call | `fn project<rows, columns>(w: [bf16; rows * columns], x: [f32; columns]) -> [f32; rows]` |
| 0048 | top-level bindings (no `const`: everything is immutable unless `mut`); compile-time evaluation of any value the dataflow proves computable; `_` as a type's default interpretation (aliases); size-generic types | `2048 -> hidden_size;` · `type Hidden { _: [f32; hidden_size] }` · `Matrix<rows, columns>` |
| 0049 | correctness is the flow and the bound: every value within the stated a-priori bound, operations ordered freely by the compiler; bit-identity is a development mode (testing, debugging), never a language promise | — (semantics; deterministic OFF arms such as `--gpu-off=mma-walk` for debugging) |

---

## 8. Method and fairness

- Same precision only (type to type); the fastest competitor leg per row; every competitor timed at the same boundary
  as Mapal (host clock around call + synchronize; the old event bracket kept as `dev_med_ms` for comparison).
- Same-box A/B (ABBA) for every optimization; machine load recorded; raw outputs archived outside git.
- Rented GPUs: this round $30.09 (RTX 3090 $2.20, A100 $3.38 + $1.62 for a box the watchdog destroyed mid-phase,
  RTX 4090 $3.04, H100 $19.72, failed 3090 starts $0.13; budget $40); all rented rounds together about $101 (8 rounds
  on 4 GPU types). Every box checked before use (a 30 s FFMA burn: none throttled), every instance destroyed after.
- Rules are physical-property based; no workload- or model-named code; every rule validated on randomly generated
  programs.

---

## 9. Next steps (in order, Singapore time)

Done since the last version of this summary: the bf16 sprint (integrated `b8429388`, reviewed `9be9886d`), the five-GPU
round (kernel suite at the program boundary + the full model in both classes on every GPU), ADR-0049, and the README
rewrite with these numbers (branch `docs/readme-rewrite`).

| # | step | contents | expected |
|---|---|---|---|
| 1 | **Bound miss — done** (the gate's bound was unsound; composed bound, `c8e1966f`) | the bf16 class's a-priori bound counts the bf16 KV cache's rounding (`precision_plan.bound` with the cache's `NarrowForm`); re-run `px_llm_blkdec_b2_l16_bf16` and the generated bf16 tests; re-check on the rented parts with the next round | Thu |
| 2 | **README — merged** (`aa372fce`) | review `docs/readme-rewrite` (charts per GPU and class, the headline tables, the correctness line) | after 1 |
| 3 | **bf16 class on the datacenter GPUs** | decode: merge the ~240 small kernels of a step at compile time (q/k/v with their rotations needs the inlined calls' walk order) — the launch-bound share is largest on the H100 / A100; prefill: the stream's block geometry from the matrix pipe's rate (as the tensor tile's), elementwise activations stored in the format their readers take, 128-row projections priced by the ordered tile's achieved pipe share | next sprint |
| 4 | **GEMM family** | H100 / A100 GEMM cells and projections (1.37 / 1.16): 16-bit wgmma tiles on the H100, TF32 rows (1.15–1.72) and bf16-weight rows (1.46–1.63) against cuBLAS / `torch.compile` | next sprint |
| 5 | **AMD backend** | a new property set, not a code path: MFMA shapes, LDS, 64-wide wavefronts, calibration; MI300-class and Radeon | after 3–4 |
| 6 | **PyTorch importer** | `torch.export` / FX graphs → Mapal IR, also emitted as readable Mapal source, so existing models run without rewriting | after 5 |
| 7 | **Ecosystem** | development (LSP, debugger, profiler, AI-agent tooling) → deployment (serving runtime, batching, paged KV cache, multi-GPU from the dataflow, quantization) → verification (bounds, equivalence certificates) → synthesis (FPGA / HLS first, then ASIC); a lab around the language | later |

---

## 10. Open work (known, measured, not yet closed)

**Correctness**
- **Bound quality:** the composed bound (sound) is ~50× the actual error on `px_llm_blkdec_b2_l16_bf16`; tighten it with a running error analysis over actual magnitudes; decide whether f32 gates move to it.
- The walk's run-to-run token variation is allowed (ADR-0049); a deterministic, workspace-free combine (each tile's
  partials summed in piece order at the ticket) would also drop the zero pass and the +170 MB peak.

**Full model (Llama-3.2-1B)**
- **bf16 class behind on every GPU**: RTX 5080 TTFT 1.05–1.15×, decode 1.06–1.10×; rented TTFT **3090 1.17–1.55,
  A100 1.44–2.12, 4090 1.16–1.44, H100 1.30–2.37**, decode **1.18–1.57** (gpt-fast's CUDA-graph decode leads on every
  rented GPU; TensorRT-LLM / SGLang / gpt-fast the prefill).
- 5080 specifics (section 5.4): attention at 2048 (7.9 ms vs ~3 ms in FlashAttention-class kernels); 4.9 ms of f32
  elementwise cells at 2048; 128-row projections in partial SM rounds; decode's ~240 small kernels.
- f32 class at a 128-token prompt: 4 % behind on the RTX 3090 (llama.cpp f32) and H100 (gpt-fast f32).
- Peak memory at 512 / 2048 above HF eager and llama.cpp in the bf16 class (the walk's workspaces, not colored by
  death across the 16 layers).
- GPU calibration: supplied automatically (environment → user cache → the repo's lines, matched by device facts);
  still open: a queued-launch price (~0.26 µs vs the isolated 11.1 µs) and a `mapal calibrate` that writes a full line.
- A loop whose exit reads the KV cache through a struct still writes the cache on the host (the model avoids it).
- Earlier finding not root-caused: before an exit-arm change, the readable form gave wrong GPU results with the arena on.
- Batch > 1 not measured for Mapal yet; no serving features yet (continuous batching, paged KV cache, multi-GPU, int4 /
  fp8 quantization of the model).
- Source-level nested loops (L1504) are expressed through calls; direct lowering is not done.

**Kernel suite / other GPUs**
- **GEMM is the weak family**: 0.93 (5080), 0.94 (4090), 1.05 (3090), 1.16 (A100), 1.37 (H100); TF32 rows behind on
  every GPU (1.15–1.72) and bf16-weight rows behind everywhere (1.46–1.63).
- H100 behind on average (1.101), A100 even (1.006); on the H100 cuBLAS's persistent 128×256 two-warpgroup kernel still
  leads; 16-bit tiles still on mma.sync (no 16-bit wgmma tile); datacenter Blackwell (tcgen05 / TMEM) not targeted yet.
- **RTX 4090 regressions to explain**: on this round's full-clock box (vs f1d3a04f's 1.2 GHz-locked one) the LLM
  family 0.903 → 1.053, bf16-weight rows 1.377 → 1.634, bf16 KV-cache rows 0.863 → 1.097, decode_1024_256 0.41 → 0.88,
  the attention rows 0.14 → 0.35 (Mapal still ahead on the last two). Not root-caused; the boxes differ (this round's
  sustained 2.45 GHz against a 3.12 GHz boost table, f1d3a04f's locked at 1.2 GHz), so the first check is a same-box
  A/B against f1d3a04f's emission.
- Multi-channel conv behind on four GPUs (1.37–2.17; conv windows not contiguous in k, cuDNN leads).
- The kernel suite's runtime safety net (the full suite on `9be9886d`'s runtime) ran on the 5080 only (34 / 35 null),
  not on the rented parts (budget); the A100 / H100 host-bound rows carry ±30 % host noise.
- 202 evidence mismatches in the full-suite run, the same rows on both runtimes, in pre-existing untriaged tile-kernel
  classes (`kmajor-a` 82, `full-tile` 32, `grid-share` 18, …); 6 `test_harness` failures that pre-date the sprint.
- Gap-fix leftovers: a kernel's first-launch host cost is not modelled; zero hoisting inside loops; stream-K splits
  that don't divide k, an ordered k-cut fix-up; `split_cell_ps` measured on the 5080 only; merged epilogues dropping
  the k split (C9).
- L2-keep sizes for A100 / H100 are placeholders (their sweeps show no plateau).
- fp8-weight rows have no competitor leg to compare against.

**Language**
- Out of scope so far: explicit instantiation of generic functions, generic element types, recursion; `/` and `%` in
  size expressions (not invertible for inference); an infinity literal (masks use −1e9).

**Competitors' own limits found (for the record)**
- SGLang has no f32 path (bf16/fp16 kernels only); TensorRT-LLM's f32 output is wrong (0/128 tokens); TensorRT-LLM did
  not install on the kernel-suite stack (torch 2.11). On the 3090 container NVML hides process memory (competitor
  peaks there are the device's growth).

---

## 11. Suggested slides and charts

0. Continuity with the 2026-10-28 deck: same layout and labels (dark title slide; "DONE · MAPAL" / "NEXT · MAPAL"
   kickers; Mapal in orange, TND in blue); open with a "since Oct 28" slide (section 0.4), then slide-8 then → now
   (0.1, dumbbell chart like the October one, log scale, 1 = parity, now with five GPUs), then the status of the
   October "next" items (0.2, 0.3).
1. Title + one-line thesis ("the language is the flow").
2. Philosophy (section 1 list).
3. Headline results (section 2).
4. Chart — full LLM, f32 class, five GPUs: Mapal ÷ the fastest f32 engine at TTFT@512 and decode, leader named
   (5.5; the README's `benches/results/charts/llm-f32-*.svg`).
5. Chart — full LLM, bf16 class, five GPUs: same layout (5.5; `llm-bf16-*.svg`), plus the 5080's sprint
   before → after (5.2, 5.4).
6. Chart — Mapal's progress on the model (5.3; log scale for decode).
7. Chart — kernel suite geomean per GPU across rounds (3.1; line or grouped bars, 1.0 reference line; the README's
   `suite-gpus-*.svg` for the latest round).
8. Table/heatmap — per family × GPU (3.2), blue < 1, red > 1 (the README's `suite-groups-*.svg`).
9. Compiler optimizations → effects (section 4).
10. Correctness: miscompiles caught and tests (section 6).
11. Language additions with syntax (section 7).
12. Method and fairness (section 8).
13. Next steps (section 9) — timeline.
14. Open work (section 10) — grouped: full model, other GPUs, language.
15. Roadmap / vision: AMD → PyTorch importer → ecosystem → verification → synthesis / ASIC lab.

Sources in the repo: `benches/results/evidence/matrix.md` (full matrix), `benches/results/evidence/local-rtx-5080/full-model.md`,
`…/bf16-tensor-tile.md`, `…/bf16-bf16-class.md`, `…/prefill-residency.md`, `…/prefill-precision.md`, `…/integration.md`,
`…/gaps-disasm.md`, `benches/results/evidence/cross-arch/README.md` (section b8429388 / 9be9886d),
`benches/results/evidence/full-model.json` (on `docs/readme-rewrite`), `docs/decisions/ADR-0042…0049`,
`docs/sessions/2026-11-24-local-rtx-5080-compiler-sprints.md`.
