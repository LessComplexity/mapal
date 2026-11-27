// ADR-0050 C3: the calibration probes the machine model's evaluator reads (crates/mapal-place, the GPU translation
// crates/backends/llvm/src/place.rs). Each prints `<probe> k=v ...` lines; place_calib.py reads them into facts.
//   start   the start timing of a grid's blocks past its first wave: a uniform-work kernel at 1 / 2 blocks an SM over
//           4 waves, each block's %globaltimer start / end and %smid; per wave past the first, the spread of its starts
//           over a block's run (lockstep: the wave's slots start together; stagger: they start as the ones before end,
//           spread over a run). `start lockstep|stagger spread_pm=` (median over waves and reps). Three loads: `l2`
//           (L2-resident, compute-bound: no variance to stagger), `dram` (every block streams its own 1/blocks of a
//           buffer 16x the L2: DRAM-contended), `gemm` (an ordered grid's operand reads: block (r, c) of a 16-wide grid
//           streams row panel r and column panel c over 1024 k steps of 8 KiB each (8 MiB a panel, ~4.5x the L2 in
//           all at 4 waves): the reuse and the DRAM contention a GEMM's waves see).
//   keep    the L2's hit fraction against reuse distance for three patterns, each a working set `ws` re-read back to
//           back (one kernel a pass): `cyclic` (one stream through the set), `rotated` (every block its own stream,
//           block b starting at b/blocks of the set: a walk's readers — a slice re-read by another SM after the rest of
//           the set), `strided` (a column panel of 256 B of every 1 KiB row: a GEMM's b panel); ws / L2 from
//           0.25 to 1.5. `keep pattern= ws= gbs= hit_pm=` (hit from the time a byte: between the pattern's own
//           plateau and the DRAM stream), then `keep pattern= keep_bytes=` (the largest ws at no less than the
//           plateau/DRAM midpoint: profile.rs l2_keep_of_sweep's rule).
//   eff     each operation's achieved fraction of its rate against independent accumulator chains a lane and resident
//           warps a scheduler: FFMA (`fma.rn.f32` chains), the warp TF32 product (`mma.sync.m16n8k8` tf32) and the
//           16-bit one (`mma.sync.m16n8k16` bf16), register operands, clock64 spans, every SM; `eff op= chains=
//           wps= flops_clk_sm= eff_pm=` against the profile's row rate (256 / 256 / 512 flops an SM a clock on the
//           RTX 5080), and `lat op= clk=` (one chain, one warp: the dependent issue latency).
//   clock   the sustained SM clock under continuous FFMA load: 4 s back to back, clock64 over %globaltimer per 100 ms
//           window; `clock sustained_khz= settle_ms=`.
//   peak    the FFMA peak by wall time (cudaEvent) at 8 chains, 16 warps a scheduler on every SM, 2 s, with the
//           clock64 / %globaltimer clock of the same launches: `peak tflops= khz= eff_pm=` (the flops a clock an SM over
//           the row rate) — the check that the sustained clock times the efficiency is the throughput.
//   prim    the cost model's primitives (ADR-0050, cost-model.md), each as (throughput saturated, latency of one use,
//           concurrency to saturate) over a sweep of what a unit keeps in flight: compute — FFMA, mma.sync TF32 /
//           bf16 / f16 and, where the part's JIT accepts the module, fp8 (e4m3) and wgmma; movers — global loads at
//           4 / 8 / 16 B and cp.async at 4 / 8 / 16 B, each from an L2-resident and a DRAM set, a GEMM's panel stream
//           by contiguous run and rows a unit streams at once (DRAM pages), TMA boxes, ld.shared 4 / 16 B and
//           ldmatrix.x4; result movement — st.global per width (DRAM; its latency a store, its fence and a load),
//           red.global.add.f32 a lane its own address / a warp one address / the grid 32 addresses, the epilogue's
//           shared-staged tile store; synchronization — bar.sync by block size, mbarrier phases, a grid barrier;
//           dispatch — the host's issue of a launch and the device's gap between dependent launches. `prim name=`
//           lines, then `calib place.prim=` (the calibration file's value) — every value measured on this part.
// build and run: benches/local/place_calib.sh <out-dir> [start|keep|eff|clock|peak|prim] (no probe: all; the
// optional modules: place_calib_opt.cu, loaded through the driver's JIT — a part whose JIT refuses one lacks it).
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

#define CK(x)                                                                              \
  do {                                                                                     \
    cudaError_t e = (x);                                                                   \
    if (e != cudaSuccess) {                                                                \
      fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e));            \
      exit(1);                                                                             \
    }                                                                                      \
  } while (0)
// The driver API's calls (the JIT'd modules, tensor maps).
#define CU(x)                                                                              \
  do {                                                                                     \
    CUresult e = (x);                                                                      \
    if (e != CUDA_SUCCESS) {                                                               \
      fprintf(stderr, "%s:%d driver error %d\n", __FILE__, __LINE__, (int)e);              \
      exit(1);                                                                             \
    }                                                                                      \
  } while (0)

static int sms, l2;

__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ unsigned smid() {
  unsigned s;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
  return s;
}

// ---- start timing -------------------------------------------------------------------------------------------------
// Uniform work: `iters` trips of 8 FFMA chains plus one 16-byte load a thread. `mode` 0: L2-resident reads
// scattered over `n`; 1: the block's own contiguous 1/gridDim of `n` (DRAM); 2: a GEMM's panels — block b of a
// 16-wide grid reads row panel b/16 and column panel b%16 (`n` holds both families), a k step a trip.
__global__ void start_k(unsigned long long *t, unsigned *sm, const float4 *buf, size_t n, int iters, float *sink,
                        int mode) {
  unsigned long long t0 = gtime();
  float a[8];
  for (int i = 0; i < 8; i++) a[i] = threadIdx.x * 1e-3f + i;
  size_t own = n / gridDim.x;
  size_t at = mode == 1 ? blockIdx.x * own + threadIdx.x : (blockIdx.x * blockDim.x + threadIdx.x) % n;
  // mode 2: a k step is 8 KiB of a row panel and 8 KiB of a column panel (512 float4 each).
  size_t half = n / 2, step = 512, row = blockIdx.x / 16, col = blockIdx.x % 16;
  for (int it = 0; it < iters; it++) {
    float4 v;
    if (mode == 2) {
      size_t k = it % 1024;
      float4 x = __ldcg(buf + ((row * 1024 + k) * step + threadIdx.x % step) % half);
      float4 y = __ldcg(buf + half + ((col * 1024 + k) * step + threadIdx.x % step) % half);
      v = make_float4(x.x + y.x, x.y + y.y, 0, 0);
    } else {
      v = __ldcg(buf + at);
    }
    at = mode == 1 ? blockIdx.x * own + (at - blockIdx.x * own + blockDim.x) % own : (at + 4099) % n;
#pragma unroll
    for (int i = 0; i < 8; i++) a[i] = fmaf(a[i], v.x, v.y);
  }
  float s = 0;
  for (int i = 0; i < 8; i++) s += a[i];
  if (s == 12345.f) *sink = s;
  __syncthreads();
  if (threadIdx.x == 0) {
    t[2 * blockIdx.x] = t0;
    t[2 * blockIdx.x + 1] = gtime();
    sm[blockIdx.x] = smid();
  }
}

static void start_run() {
  size_t big = (size_t)l2 * 16 / 16;  // float4s: 16x the L2
  float4 *buf;
  float *sink;
  CK(cudaMalloc(&buf, big * 16));
  CK(cudaMemset(buf, 0, big * 16));
  CK(cudaMalloc(&sink, 4));
  const char *names[] = {"l2", "dram", "gemm"};
  for (int mode = 0; mode < 3; mode++)
  for (int per : {1, 2}) {
    size_t n = mode == 0 ? (size_t)1 << 20 : big;  // l2: 16 MiB, L2-resident
    int threads = 256, waves = 4, blocks = sms * per * waves;
    unsigned long long *t;
    unsigned *sm;
    CK(cudaMalloc(&t, blocks * 16));
    CK(cudaMalloc(&sm, blocks * 4));
    std::vector<unsigned long long> h(2 * blocks);
    std::vector<double> spreads;
    for (int rep = 0; rep < 7; rep++) {
      start_k<<<blocks, threads>>>(t, sm, buf, n, mode == 0 ? 4000 : 1024, sink, mode);
      CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(h.data(), t, blocks * 16, cudaMemcpyDeviceToHost));
      if (rep == 0) continue;  // warm-up
      // Blocks by start time; a wave = the next sms*per to start.
      std::vector<std::pair<unsigned long long, unsigned long long>> se(blocks);
      for (int b = 0; b < blocks; b++) se[b] = {h[2 * b], h[2 * b + 1]};
      std::sort(se.begin(), se.end());
      std::vector<double> runs;
      for (auto &x : se) runs.push_back(double(x.second - x.first));
      std::sort(runs.begin(), runs.end());
      double run = runs[runs.size() / 2];
      int w = sms * per;
      for (int wv = 1; wv < waves; wv++) {
        // The middle 90 % of the wave's starts over one run.
        int lo = wv * w + w / 20, hi = wv * w + w - 1 - w / 20;
        spreads.push_back(double(se[hi].first - se[lo].first) / run);
      }
      if (rep == 1)
        printf("start load=%s per_sm=%d blocks=%d run_ns=%.0f first_wave_spread_pm=%.0f\n", names[mode], per,
               blocks, run, 1000.0 * double(se[w - 1 - w / 20].first - se[w / 20].first) / run);
    }
    std::sort(spreads.begin(), spreads.end());
    double med = spreads[spreads.size() / 2];
    // Lockstep where a wave's slots start within a quarter of a run of each other.
    printf("start load=%s per_sm=%d spread_pm=%.0f min_pm=%.0f max_pm=%.0f verdict=%s\n", names[mode], per,
           1000 * med, 1000 * spreads.front(), 1000 * spreads.back(), med < 0.25 ? "lockstep" : "stagger");
    CK(cudaFree(t));
    CK(cudaFree(sm));
  }
  CK(cudaFree(buf));
}

// ---- keep: hit against reuse distance --------------------------------------------------------------------------------
__global__ void cyclic_k(const float4 *p, size_t n, float *out) {
  float4 s = make_float4(0, 0, 0, 0);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    float4 v = __ldcg(p + i);
    s.x += v.x;
    s.y += v.y;
  }
  if (s.x + s.y == 12345.0f) *out = s.x;
}
// Every block its own chunk of the set, the chunks rotating a block a pass: a slice re-read by another SM after the
// rest of the set — a walk's readers (a slice's next reader is another block, at another offset).
__global__ void rotated_k(const float4 *p, size_t n, int pass, float *out) {
  float4 s = make_float4(0, 0, 0, 0);
  size_t chunk = n / gridDim.x, start = chunk * ((blockIdx.x + pass) % gridDim.x);
  for (size_t i = threadIdx.x; i < chunk; i += blockDim.x) {
    float4 v = __ldcg(p + start + i);
    s.x += v.x;
    s.y += v.y;
  }
  if (s.x + s.y == 12345.0f) *out = s.x;
}
// A 256-byte column panel of `rows` rows, `row4` float4s apart.
__global__ void strided_k(const float4 *p, size_t rows, size_t row4, float *out) {
  float4 s = make_float4(0, 0, 0, 0);
  size_t n = rows * 16;  // 16 float4 a row's panel
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    float4 v = __ldcg(p + (i / 16) * row4 + i % 16);
    s.x += v.x;
    s.y += v.y;
  }
  if (s.x + s.y == 12345.0f) *out = s.x;
}

static void keep_run() {
  size_t cap = (size_t)l2 * 16;
  char *buf;
  float *out;
  CK(cudaMalloc(&buf, cap));
  CK(cudaMemset(buf, 0, cap));
  CK(cudaMalloc(&out, 4));
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  double r[] = {0.25, 0.5, 0.625, 0.75, 0.8125, 0.875, 0.9375, 1.0, 1.025, 1.05, 1.1, 1.2, 1.35, 1.5, 3.0};
  int nr = sizeof(r) / sizeof(r[0]);
  const char *pats[] = {"cyclic", "rotated", "strided"};
  for (int pi = 0; pi < 3; pi++) {
    std::vector<double> gbs(nr);
    for (int ri = 0; ri < nr; ri++) {
      size_t ws = (size_t)(r[ri] * l2) / 4096 * 4096, n = ws / 16;
      size_t row4 = 1024 / 16, rows = ws / 256;  // strided: 256 B of every 1 KiB row
      if (pi == 2 && rows * 1024 > cap) {
        gbs[ri] = 0;
        continue;
      }
      int blocks = sms * 8;
      auto pass = [&](int i) {
        if (pi == 0) cyclic_k<<<blocks, 512>>>((const float4 *)buf, n, out);
        if (pi == 1) rotated_k<<<blocks, 512>>>((const float4 *)buf, n, i, out);
        if (pi == 2) strided_k<<<blocks, 512>>>((const float4 *)buf, rows, row4, out);
      };
      for (int i = 0; i < 5; i++) pass(i);
      cudaEventRecord(a);
      int reps = 20;
      for (int i = 0; i < reps; i++) pass(i);
      cudaEventRecord(b);
      CK(cudaEventSynchronize(b));
      float ms;
      cudaEventElapsedTime(&ms, a, b);
      gbs[ri] = reps * double(ws) / (ms * 1e6);
    }
    // The pattern's own plateau (its best rate below the L2) and its DRAM stream (the 3 x L2 point).
    double plateau = 0, dram = gbs[nr - 1];
    for (int ri = 0; ri < nr; ri++)
      if (r[ri] <= 0.75) plateau = std::max(plateau, gbs[ri]);
    double mid = 0.5 * (plateau + dram);
    size_t keep = 0;
    for (int ri = 0; ri < nr - 1; ri++) {
      if (gbs[ri] == 0) continue;
      double t = 1.0 / gbs[ri], tl = 1.0 / plateau, td = 1.0 / dram;
      double hit = std::clamp((td - t) / (td - tl), 0.0, 1.0);
      printf("keep pattern=%s ws_l2_pm=%.0f ws=%zu gbs=%.0f hit_pm=%.0f\n", pats[pi], 1000 * r[ri],
             (size_t)(r[ri] * l2) / 4096 * 4096, gbs[ri], 1000 * hit);
      if (gbs[ri] >= mid) keep = std::max(keep, (size_t)(r[ri] * l2) / 4096 * 4096);
    }
    // The sweep's resolution: the keep lies between its point and the next one the pattern measured.
    size_t next = keep;
    for (int ri = 0; ri < nr; ri++)
      if (gbs[ri] != 0 && (size_t)(r[ri] * l2) / 4096 * 4096 > keep) {
        next = (size_t)(r[ri] * l2) / 4096 * 4096;
        break;
      }
    printf("keep pattern=%s keep_bytes=%zu keep_hi_bytes=%zu plateau_gbs=%.0f dram_gbs=%.0f\n", pats[pi], keep,
           next, plateau, dram);
  }
  CK(cudaFree(buf));
}

// ---- eff: an operation's fraction of its rate against chains and warps ------------------------------------------------
template <int C>
__global__ void ffma_k(float *out, int iters, long long *cyc) {
  float a[C];
#pragma unroll
  for (int i = 0; i < C; i++) a[i] = threadIdx.x * 1e-3f + i;
  float x = out[0] + 1.0001f, y = 0.9999f;
  __syncthreads();
  long long t0 = clock64();
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int i = 0; i < C; i++) a[i] = fmaf(a[i], x, y);
  }
  long long t1 = clock64();
  float s = 0;
#pragma unroll
  for (int i = 0; i < C; i++) s += a[i];
  if (s == 12345.f) out[1] = s;
  if (threadIdx.x == 0) atomicMax((unsigned long long *)cyc + blockIdx.x % 1024, (unsigned long long)(t1 - t0));
}

// `C` independent m16n8k8 TF32 accumulators a warp (4 registers each), operands constant in registers.
template <int C>
__global__ void mma_tf32_k(float *out, int iters, long long *cyc) {
  float d[C][4];
#pragma unroll
  for (int i = 0; i < C; i++)
    for (int j = 0; j < 4; j++) d[i][j] = 0;
  unsigned a0 = __float_as_uint(out[0] + 1.f), a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0;
  __syncthreads();
  long long t0 = clock64();
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int i = 0; i < C; i++)
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
          "{%0,%1,%2,%3};"
          : "+f"(d[i][0]), "+f"(d[i][1]), "+f"(d[i][2]), "+f"(d[i][3])
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  }
  long long t1 = clock64();
  float s = 0;
#pragma unroll
  for (int i = 0; i < C; i++) s += d[i][0];
  if (s == 12345.f) out[1] = s;
  if (threadIdx.x == 0) atomicMax((unsigned long long *)cyc + blockIdx.x % 1024, (unsigned long long)(t1 - t0));
}

// `C` independent m16n8k16 16-bit accumulators a warp: `F` 0 bf16, 1 f16.
template <int C, int F>
__global__ void mma_h_k(float *out, int iters, long long *cyc) {
  float d[C][4];
#pragma unroll
  for (int i = 0; i < C; i++)
    for (int j = 0; j < 4; j++) d[i][j] = 0;
  unsigned a0 = (F ? 0x3c003c00u : 0x3f803f80u) + (unsigned)(out[0] != 0), a1 = a0, a2 = a0, a3 = a0, b0 = a0,
           b1 = a0;
  __syncthreads();
  long long t0 = clock64();
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int i = 0; i < C; i++) {
      if (F == 0)
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
            "{%0,%1,%2,%3};"
            : "+f"(d[i][0]), "+f"(d[i][1]), "+f"(d[i][2]), "+f"(d[i][3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      else
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
            "{%0,%1,%2,%3};"
            : "+f"(d[i][0]), "+f"(d[i][1]), "+f"(d[i][2]), "+f"(d[i][3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
  }
  long long t1 = clock64();
  float s = 0;
#pragma unroll
  for (int i = 0; i < C; i++) s += d[i][0];
  if (s == 12345.f) out[1] = s;
  if (threadIdx.x == 0) atomicMax((unsigned long long *)cyc + blockIdx.x % 1024, (unsigned long long)(t1 - t0));
}

typedef void (*kern)(float *, int, long long *);

// The compute sweeps' points for `prim`: per operation, (instructions in flight an SM = chains x warps, flops a
// clock an SM), and its one-chain one-warp latency (clocks).
struct CPt {
  double fly, per_clk, wps;
};
static std::map<std::string, std::vector<CPt>> cpts;
static std::map<std::string, std::vector<double>> clat;

// A compute sweep: warps a scheduler 0.25 .. 8 (one block an SM of 4·wps warps; a quarter: one warp an SM), each
// point's clock64 span (the block's longest), three runs (the last kept); `launch` runs `threads` lanes a block.
// A driver-API point the block cannot hold (its launch bounds, registers): refused, no point (the sweep goes on).
static bool eff_launch_failed = false;
static void eff_sweep(const char *op, int chains, std::function<void(int, float *, int, long long *)> launch,
                      double flops_per_trip_warp, int lanes_per_group = 32, std::vector<double> wpss = {}) {
  float *out;
  long long *cyc;
  CK(cudaMalloc(&out, 8));
  CK(cudaMemset(out, 0, 8));
  CK(cudaMalloc(&cyc, 1024 * 8));
  if (wpss.empty()) wpss = {0.25, 0.5, 1, 2, 4, 8};
  for (double wps : wpss) {
    int warps = std::max(1, (int)(4 * wps));
    int groups = std::max(1, warps * 32 / lanes_per_group);
    int threads = groups * lanes_per_group;
    int iters = 4096;
    std::vector<double> lat;
    bool fits = true;
    for (int rep = 0; rep < 4 && fits; rep++) {
      CK(cudaMemset(cyc, 0, 1024 * 8));
      launch(threads, out, iters, cyc);
      // A footprint the SM cannot hold (registers): no point.
      fits = cudaGetLastError() == cudaSuccess && !std::exchange(eff_launch_failed, false);
      CK(cudaDeviceSynchronize());
      if (chains == 1 && groups == 1 && rep > 0) {
        long long h0;
        CK(cudaMemcpy(&h0, cyc, 8, cudaMemcpyDeviceToHost));
        lat.push_back(double(h0) / iters);
      }
    }
    if (!fits) continue;
    std::vector<long long> h(1024);
    CK(cudaMemcpy(h.data(), cyc, 1024 * 8, cudaMemcpyDeviceToHost));
    long long span = *std::max_element(h.begin(), h.begin() + std::min(sms, 1024));
    double flops = flops_per_trip_warp * groups * iters * chains;  // an SM's
    double per_clk = flops / double(span);
    cpts[op].push_back({double(chains) * groups, per_clk, wps});
    printf("effpt op=%s chains=%d wps=%.2f flops_clk_sm=%.1f\n", op, chains, wps, per_clk);
    if (!lat.empty()) {
      // One chain, one group: a trip's clocks are the dependent issue latency.
      std::sort(lat.begin(), lat.end());
      clat[op] = lat;
      printf("lat op=%s clk=%.2f\n", op, lat[lat.size() / 2]);
    }
  }
  CK(cudaFree(out));
  CK(cudaFree(cyc));
}

// The eff lines: each point's fraction of the operation's saturated rate (the sweep's best: no part's row rate).
static void eff_print(const char *op, int lanes_per_group = 32) {
  auto &v = cpts[op];
  double best = 0;
  for (auto &p : v) best = std::max(best, p.per_clk);
  for (auto &p : v) {
    int groups = std::max(1, std::max(1, (int)(4 * p.wps)) * 32 / lanes_per_group);
    printf("eff op=%s chains=%.0f wps=%.2f flops_clk_sm=%.1f eff_pm=%.0f\n", op, p.fly / groups, p.wps, p.per_clk,
           1000.0 * p.per_clk / best);
  }
}

template <void (*K)(float *, int, long long *)>
static std::function<void(int, float *, int, long long *)> rt() {
  return [](int threads, float *out, int iters, long long *cyc) { K<<<sms, threads>>>(out, iters, cyc); };
}

static void eff_run() {
  // flops a warp a trip a chain: FFMA 32 lanes x 2; m16n8k8 2·16·8·8; m16n8k16 2·16·8·16.
  eff_sweep("ffma", 1, rt<ffma_k<1>>(), 64);
  eff_sweep("ffma", 2, rt<ffma_k<2>>(), 64);
  eff_sweep("ffma", 4, rt<ffma_k<4>>(), 64);
  eff_sweep("ffma", 8, rt<ffma_k<8>>(), 64);
  eff_sweep("ffma", 16, rt<ffma_k<16>>(), 64);
  eff_sweep("ffma", 32, rt<ffma_k<32>>(), 64);
  eff_sweep("ffma", 64, rt<ffma_k<64>>(), 64);
  eff_sweep("mma.m16n8k8.tf32", 1, rt<mma_tf32_k<1>>(), 2048);
  eff_sweep("mma.m16n8k8.tf32", 2, rt<mma_tf32_k<2>>(), 2048);
  eff_sweep("mma.m16n8k8.tf32", 4, rt<mma_tf32_k<4>>(), 2048);
  eff_sweep("mma.m16n8k8.tf32", 8, rt<mma_tf32_k<8>>(), 2048);
  eff_sweep("mma.m16n8k8.tf32", 16, rt<mma_tf32_k<16>>(), 2048);
  eff_sweep("mma.m16n8k16.bf16", 1, rt<mma_h_k<1, 0>>(), 4096);
  eff_sweep("mma.m16n8k16.bf16", 2, rt<mma_h_k<2, 0>>(), 4096);
  eff_sweep("mma.m16n8k16.bf16", 4, rt<mma_h_k<4, 0>>(), 4096);
  eff_sweep("mma.m16n8k16.bf16", 8, rt<mma_h_k<8, 0>>(), 4096);
  eff_sweep("mma.m16n8k16.bf16", 16, rt<mma_h_k<16, 0>>(), 4096);
  eff_sweep("mma.m16n8k16.f16", 1, rt<mma_h_k<1, 1>>(), 4096);
  eff_sweep("mma.m16n8k16.f16", 4, rt<mma_h_k<4, 1>>(), 4096);
  eff_sweep("mma.m16n8k16.f16", 16, rt<mma_h_k<16, 1>>(), 4096);
  for (const char *op : {"ffma", "mma.m16n8k8.tf32", "mma.m16n8k16.bf16", "mma.m16n8k16.f16"}) eff_print(op);
}

// ---- clock: the sustained clock under FFMA load ---------------------------------------------------------------------------
__global__ void clock_k(float *out, int iters, unsigned long long *span) {
  float a[8];
  for (int i = 0; i < 8; i++) a[i] = threadIdx.x * 1e-3f + i;
  unsigned long long g0 = gtime();
  long long c0 = clock64();
  for (int it = 0; it < iters; it++)
#pragma unroll
    for (int i = 0; i < 8; i++) a[i] = fmaf(a[i], 1.0001f, 0.9999f);
  long long c1 = clock64();
  unsigned long long g1 = gtime();
  float s = 0;
  for (int i = 0; i < 8; i++) s += a[i];
  if (s == 12345.f) out[0] = s;
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    span[0] = c1 - c0;
    span[1] = g1 - g0;
  }
}

static void clock_run() {
  float *out;
  unsigned long long *span;
  CK(cudaMalloc(&out, 4));
  CK(cudaMalloc(&span, 16));
  std::vector<double> khz;
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  double elapsed = 0;
  while (elapsed < 4000) {
    cudaEventRecord(a);
    clock_k<<<sms * 8, 256>>>(out, 200000, span);
    cudaEventRecord(b);
    CK(cudaEventSynchronize(b));
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    elapsed += ms;
    unsigned long long h[2];
    CK(cudaMemcpy(h, span, 16, cudaMemcpyDeviceToHost));
    khz.push_back(1e6 * double(h[0]) / double(h[1]));
  }
  // The last second's windows.
  size_t n = khz.size(), last = std::max<size_t>(1, n / 4);
  std::vector<double> tail(khz.end() - last, khz.end());
  std::sort(tail.begin(), tail.end());
  double sus = tail[tail.size() / 2];
  size_t settle = 0;
  for (size_t i = 0; i < n; i++)
    if (std::abs(khz[i] - sus) > 0.02 * sus) settle = i + 1;
  printf("clock sustained_khz=%.0f settle_windows=%zu windows=%zu first_khz=%.0f\n", sus, settle, n, khz[0]);
}

// ---- peak: the FFMA throughput by wall time, and the clock of the same launches --------------------------------------
static void peak_run() {
  float *out;
  unsigned long long *span;
  CK(cudaMalloc(&out, 4));
  CK(cudaMalloc(&span, 16));
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  // Warm the clock to its sustained state first (1 s), then 2 s timed.
  double elapsed = 0, flops = 0, ms_all = 0;
  std::vector<double> khz;
  int blocks = sms * 8, threads = 256, iters = 200000;
  while (elapsed < 3000) {
    cudaEventRecord(a);
    clock_k<<<blocks, threads>>>(out, iters, span);
    cudaEventRecord(b);
    CK(cudaEventSynchronize(b));
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    elapsed += ms;
    if (elapsed < 1000) continue;
    unsigned long long h[2];
    CK(cudaMemcpy(h, span, 16, cudaMemcpyDeviceToHost));
    khz.push_back(1e6 * double(h[0]) / double(h[1]));
    flops += double(blocks) * threads * iters * 8 * 2;
    ms_all += ms;
  }
  std::sort(khz.begin(), khz.end());
  double k = khz[khz.size() / 2];
  double tflops = flops / (ms_all * 1e9);
  // flops a clock an SM by wall time, over the FFMA row's 256.
  double eff = tflops * 1e12 / (double(sms) * k * 1e3) / 256.0;
  printf("peak tflops=%.2f khz=%.0f eff_pm=%.0f windows=%zu\n", tflops, k, 1000 * eff, khz.size());
}

// ---- prim: every primitive's (throughput saturated, latency of one use, concurrency to saturate) --------------------
// Each primitive is swept over what it keeps in flight a unit (an SM): warps x independent uses a lane, or boxes a
// block x blocks. Per point the rate a unit (median of 3 timed runs; the best point's min / max its interval);
// `rate` the sweep's best, `conc` the least in flight reaching 90 % of it (interval: the largest point below the
// knee .. the knee), `lat` one use's latency (one lane, a dependent chain: median and min / max of 3). Units: compute
// flop / clock / SM, clocks, instructions / SM; movers bytes / ns / SM, ns, bytes / SM; reductions ops / ns / SM;
// barriers and launches their cost (ns or clocks) and rate 1 / cost. One `prim` line each, then the calibration
// file's `place.prim=` value (milli-units: name:rate:lo:hi:lat:lo:hi:conc:lo:hi).
struct Pt {
  double fly, rate, lo, hi;
};
struct PrimRow {
  std::string name;
  double rate, rlo, rhi, lat, llo, lhi, conc, clo, chi;
  const char *ru, *lu, *cu;
};
static std::vector<PrimRow> prows;
static double prim_ghz = 0;  // the clock prim converts clocks with (measured under load at its start)
static int smem_sm = 0, smem_optin = 0, max_thr_sm = 0;

static double med(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

static void put(const std::string &name, const std::vector<Pt> &pts, std::vector<double> lat, const char *ru,
                const char *lu, const char *cu) {
  PrimRow r{name, 0, 0, 0, 0, 0, 0, 0, 0, 0, ru, lu, cu};
  for (auto &p : pts)
    if (p.rate > r.rate) r.rate = p.rate, r.rlo = p.lo, r.rhi = p.hi;
  r.conc = 1e300;
  for (auto &p : pts)
    if (p.rate >= 0.9 * r.rate) r.conc = std::min(r.conc, p.fly);
  r.clo = 0;
  for (auto &p : pts)
    if (p.rate < 0.9 * r.rate && p.fly < r.conc) r.clo = std::max(r.clo, p.fly);
  if (r.clo == 0) r.clo = r.conc / 2;
  r.chi = r.conc;
  if (!lat.empty()) {
    std::sort(lat.begin(), lat.end());
    r.lat = lat[lat.size() / 2], r.llo = lat.front(), r.lhi = lat.back();
  }
  for (auto &p : pts) printf("primpt name=%s fly=%.0f rate=%.4g\n", name.c_str(), p.fly, p.rate);
  printf("prim name=%s rate=%.4g rate_lo=%.4g rate_hi=%.4g unit=%s lat=%.4g lat_lo=%.4g lat_hi=%.4g lat_unit=%s "
         "conc=%.4g conc_lo=%.4g conc_hi=%.4g conc_unit=%s little=%.4g\n",
         name.c_str(), r.rate, r.rlo, r.rhi, ru, r.lat, r.llo, r.lhi, lu, r.conc, r.clo, r.chi, cu, r.lat * r.rate);
  fflush(stdout);
  prows.push_back(r);
}

// Times `f` (one launch or a sequence) by events: three timed runs after one warm-up, ms each.
static std::vector<double> timed(const std::function<void()> &f, int reps = 3) {
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a));
  CK(cudaEventCreate(&b));
  f();
  CK(cudaDeviceSynchronize());
  std::vector<double> ms;
  for (int r = 0; r < reps; r++) {
    CK(cudaEventRecord(a));
    f();
    CK(cudaEventRecord(b));
    CK(cudaEventSynchronize(b));
    float t;
    CK(cudaEventElapsedTime(&t, a, b));
    ms.push_back(t);
  }
  CK(cudaGetLastError());
  CK(cudaEventDestroy(a));
  CK(cudaEventDestroy(b));
  return ms;
}

// A rate point from timed runs: `units` (bytes or ops) a run over the SMs, per ns per SM.
static Pt rate_pt(double fly, double units, const std::vector<double> &ms) {
  std::vector<double> r;
  for (double m : ms) r.push_back(units / (m * 1e6) / sms);
  std::sort(r.begin(), r.end());
  return {fly, r[r.size() / 2], r.front(), r.back()};
}

__device__ __forceinline__ unsigned fold(unsigned v) { return v; }
__device__ __forceinline__ unsigned fold(uint2 v) { return v.x ^ v.y; }
__device__ __forceinline__ unsigned fold(uint4 v) { return v.x ^ v.y ^ v.z ^ v.w; }
template <typename T>
__device__ __forceinline__ T mk(unsigned x) {
  T t;
  memset(&t, 0, sizeof(T));
  *(unsigned *)&t = x;
  return t;
}

// Global loads of `T` (4 / 8 / 16 B) through the L2 (ld.global.cg): `U` independent loads a lane in flight, a
// warp's 32 lanes contiguous; `passes` over `n` elements.
template <typename T, int U>
__global__ void ldg_k(const T *p, size_t n, int passes, unsigned *out) {
  unsigned acc = 0;
  size_t stride = (size_t)gridDim.x * blockDim.x, i0 = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (int ps = 0; ps < passes; ps++)
    for (size_t i = i0; i + (U - 1) * stride < n; i += U * stride) {
      T v[U];
#pragma unroll
      for (int u = 0; u < U; u++) v[u] = __ldcg(p + i + u * stride);
#pragma unroll
      for (int u = 0; u < U; u++) acc ^= fold(v[u]);
    }
  if (acc == 0x9e3779b9u) *out = acc;
}
// One lane's dependent chain: each element's first word the next index.
template <typename T>
__global__ void chase_k(const T *p, int steps, unsigned long long *out) {
  unsigned i = (unsigned)out[1];
  unsigned long long t0 = gtime();
  for (int s = 0; s < steps; s++) i = fold(__ldcg(p + i));
  out[0] = gtime() - t0;
  out[1] = i;
}

// Global stores of `T`: `U` a lane a trip.
template <typename T, int U>
__global__ void stg_k(T *p, size_t n, int passes) {
  size_t stride = (size_t)gridDim.x * blockDim.x, i0 = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (int ps = 0; ps < passes; ps++)
    for (size_t i = i0; i + (U - 1) * stride < n; i += U * stride) {
#pragma unroll
      for (int u = 0; u < U; u++) __stcg(p + i + u * stride, mk<T>((unsigned)(i + u + ps)));
    }
}
// A store and the fence that makes it visible to the device: one lane, a chain.
__global__ void stfence_k(unsigned *p, int steps, unsigned long long *out) {
  unsigned long long t0 = gtime();
  unsigned v = 0;
  for (int s = 0; s < steps; s++) {
    __stcg(p + (v & 1023), v + s);
    __threadfence();
    v = __ldcg(p + ((v + 1) & 1023));
  }
  out[0] = gtime() - t0;
  out[1] = v;
}

__device__ __forceinline__ unsigned sm32(const void *p) { return (unsigned)__cvta_generic_to_shared(p); }
// cp.async of `W` bytes (4 / 8: .ca, 16: .cg) global -> shared: `U` a lane, committed, waited for all.
template <int W, int U>
__global__ void cpa_k(const char *p, size_t n, int passes, unsigned *out) {
  extern __shared__ __align__(16) char sm[];
  size_t stride = (size_t)gridDim.x * blockDim.x * W, i0 = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * W;
  for (int ps = 0; ps < passes; ps++)
    for (size_t i = i0; i + (U - 1) * stride < n; i += U * stride) {
#pragma unroll
      for (int u = 0; u < U; u++) {
        unsigned dst = sm32(sm + ((size_t)u * blockDim.x + threadIdx.x) * W);
        if (W == 16)
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(p + i + u * stride) : "memory");
        else
          asm volatile("cp.async.ca.shared.global [%0], [%1], %2;" ::"r"(dst), "l"(p + i + u * stride), "n"(W)
                       : "memory");
      }
      asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
    }
  if (sm[threadIdx.x * W] == 123 && sm[threadIdx.x * W + 1] == 45) *out = 1;
}
// One copy's latency: a lane copies 16 B, waits, reads the landed word as the next index (a chain).
__global__ void cpa_chase_k(const uint4 *p, int steps, unsigned long long *out) {
  __shared__ __align__(16) uint4 s;
  unsigned i = (unsigned)out[1];
  unsigned long long t0 = gtime();
  for (int k = 0; k < steps; k++) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\ncp.async.commit_group;\ncp.async.wait_group 0;" ::"r"(
                     sm32(&s)),
                 "l"(p + i)
                 : "memory");
    i = *(volatile unsigned *)&s.x;
  }
  out[0] = gtime() - t0;
  out[1] = i;
}

// A panel stream (a GEMM's operand panels): block b's `rows` rows of `row_bytes` each, a trip `run` contiguous
// bytes of every row at the trip's offset (16-byte cp.async.cg, `U` trips in flight a block), the next trip the
// next `run` of the same rows; the rows of block b are b's own (no reuse), the buffer `rows x gridDim` rows.
template <int U>
__global__ void panel_k(const char *p, int rows, int row_bytes, int run, int trips, unsigned *out) {
  extern __shared__ __align__(16) char sm[];
  int per_trip = rows * run / 16;  // 16-byte copies a trip
  const char *base = p + (size_t)blockIdx.x * rows * row_bytes;
  int tr = row_bytes / run;
  for (int t = 0; t < trips; t += U) {
#pragma unroll
    for (int u = 0; u < U; u++) {
      int at = (t + u) % tr;
      for (int c = threadIdx.x; c < per_trip; c += blockDim.x) {
        int r = c / (run / 16), o = c % (run / 16);
        const char *g = base + (size_t)r * row_bytes + (size_t)at * run + o * 16;
        unsigned dst = sm32(sm + ((size_t)u * per_trip + c) * 16);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(g) : "memory");
      }
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
    __syncthreads();
  }
  if (sm[threadIdx.x] == 123 && sm[threadIdx.x + 1] == 45) *out = 1;
}

// Shared loads of `W` bytes (4: ld.shared.u32, 16: ld.shared.v4): `U` independent a lane a trip, conflict-free.
template <int W, int U>
__global__ void lds_k(int iters, int zero, long long *cyc, unsigned *out) {
  __shared__ __align__(16) unsigned sm[8192];
  for (int i = threadIdx.x; i < 8192; i += blockDim.x) sm[i] = i * zero;
  __syncthreads();
  unsigned acc = 0, off = threadIdx.x * (W / 4);
  long long t0 = clock64();
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int u = 0; u < U; u++) {
      unsigned a = sm32(sm + ((off + (u * 32 + it) * (W / 4) * 32) & 8191 & ~(W / 4 - 1)));
      if (W == 16) {
        unsigned x, y, z, w;
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "r"(a));
        acc ^= x ^ y ^ z ^ w;
      } else {
        unsigned x;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x) : "r"(a));
        acc ^= x;
      }
    }
  }
  long long t1 = clock64();
  if (acc == 0x9e3779b9u) *out = acc;
  if (threadIdx.x == 0) atomicMax((unsigned long long *)cyc + blockIdx.x % 1024, (unsigned long long)(t1 - t0));
}
// ldmatrix.x4 (four 8x8 b16 tiles a warp: 512 B): `U` independent a warp a trip.
template <int U>
__global__ void ldsm_k(int iters, int zero, long long *cyc, unsigned *out) {
  __shared__ __align__(16) unsigned sm[8192];
  for (int i = threadIdx.x; i < 8192; i += blockDim.x) sm[i] = i * zero;
  __syncthreads();
  unsigned acc = 0, lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  long long t0 = clock64();
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int u = 0; u < U; u++) {
      // 32 rows of 16 B, contiguous (conflict-free), a different 512 B block each use.
      unsigned a = sm32(sm + (((warp * U + u + it) * 128 + lane * 4) & 8191));
      unsigned x, y, z, w;
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                   : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                   : "r"(a));
      acc ^= x ^ y ^ z ^ w;
    }
  }
  long long t1 = clock64();
  if (acc == 0x9e3779b9u) *out = acc;
  if (threadIdx.x == 0) atomicMax((unsigned long long *)cyc + blockIdx.x % 1024, (unsigned long long)(t1 - t0));
}
// One use's latency in shared memory: a pointer chase — each lane's next address is the word it loads (the shared
// words hold their own lanes' addresses), nothing else on the chain (no branch, no address arithmetic: test
// `place_calib_chase_is_pure` reads the SASS) — of `KIND` 0 ld.shared.u32 (lane l at word l: conflict-free), 1
// ld.shared.v4.u32 (lane l at bytes 16 l: the 512-byte access's own wavefronts), 2 ldmatrix.x4 (lane l supplies
// row l % 8 of matrix l / 8 at bytes 16 l and receives matrix 0's row l / 4, word l % 4 — word l, which holds
// bytes 16 l), clocks a use.
template <int KIND>
__global__ void lds_chase_k(int steps, long long *out) {
  __shared__ __align__(16) unsigned sm[8192];
  unsigned lane = threadIdx.x % 32, base = sm32(sm);
  unsigned a = base + lane * (KIND == 0 ? 4 : 16);
  sm[KIND == 1 ? lane * 4 : lane] = a;
  __syncwarp();
  // The other words a wide load returns, folded off the chain (else ptxas narrows the load to the one word read).
  unsigned sink = 0;
  long long t0 = clock64();
#pragma unroll 8
  for (int s = 0; s < steps; s++) {
    if (KIND == 0) {
      asm volatile("ld.shared.u32 %0, [%0];" : "+r"(a));
    } else if (KIND == 1) {
      unsigned y, z, w;
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%0];" : "+r"(a), "=r"(y), "=r"(z), "=r"(w));
      sink ^= y ^ z ^ w;
    } else {
      unsigned y, z, w;
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%0];"
                   : "+r"(a), "=r"(y), "=r"(z), "=r"(w));
      sink ^= y ^ z ^ w;
    }
  }
  long long t1 = clock64();
  if (threadIdx.x == 0) out[0] = t1 - t0, out[1] = a ^ sink;
}

// red.global.add.f32: `mode` 0 a lane its own address, a warp's 32 lanes contiguous (4 sectors an instruction; an
// L2-resident window), 1 a warp's 32 lanes one address (contended within the warp), 2 every lane of the grid one
// of 32 addresses, 3 a warp matrix accumulator's fragment (lane l: row l / 4, columns 2 (l % 4) and + 1 — 8 rows of
// 32 bytes, 8 sectors an instruction), 4 a lane its own 32-byte sector (32 sectors an instruction); `U` a lane a
// trip.
template <int U>
__global__ void red_k(float *p, size_t n, int iters, int mode) {
  size_t i0 = (size_t)blockIdx.x * blockDim.x + threadIdx.x, stride = (size_t)gridDim.x * blockDim.x;
  size_t lane = threadIdx.x % 32, warp = i0 / 32, warps = stride / 32;
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int u = 0; u < U; u++) {
      size_t k = (size_t)(it * U + u), at;
      if (mode == 0) at = (i0 + k * stride) % n;
      else if (mode == 1) at = ((i0 / 32) * 32 + k * stride) % n;
      else if (mode == 2) at = threadIdx.x % 32;
      else if (mode == 3) at = ((warp + k * warps) * 8 * 4096 + (lane / 4) * 4096 + (lane % 4) * 2 + (u & 1)) % n;
      else at = ((warp + k * warps) * 256 + lane * 8) % n;
      atomicAdd(p + at, 1.0f);  // the return unused: red.global.add.f32
    }
  }
}
// One atomic's latency: atom.global.add.f32 whose result moves the next address (times a run-time zero).
__global__ void atom_chase_k(float *p, int steps, int zero, unsigned long long *out) {
  float v = 0;
  unsigned long long t0 = gtime();
  for (int s = 0; s < steps; s++) v = atomicAdd(p + ((int)v * zero & 1023), 1.0f);
  out[0] = gtime() - t0;
  out[1] = (unsigned long long)v;
}

// The epilogue's store through shared memory: a block of 256 lanes stages a 64 x 128 f32 tile (32 KiB) in shared
// memory (each lane its 32 values, st.shared.v4), a barrier, then stores it row-major and coalesced
// (ld.shared.v4 + st.global.v4) to its tile of a DRAM-sized output; `tiles` tiles a block.
__global__ void epi_k(float4 *outp, size_t tiles_all, int tiles) {
  extern __shared__ __align__(16) float4 st[];
  for (int t = 0; t < tiles; t++) {
    size_t tile = ((size_t)blockIdx.x * tiles + t) % tiles_all;
#pragma unroll
    for (int j = 0; j < 8; j++) st[threadIdx.x * 8 + ((j + threadIdx.x) & 7)] = make_float4(t, j, 0, 1);
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 8; j++) {
      int e = j * 256 + threadIdx.x;
      __stcg(outp + tile * 2048 + e, st[e]);
    }
    __syncthreads();
  }
}

// bar.sync: a block's barrier back to back (clocks a barrier, the block's longest span).
__global__ void bar_k(int iters, long long *cyc) {
  __syncthreads();
  long long t0 = clock64();
  for (int i = 0; i < iters; i++) __syncthreads();
  long long t1 = clock64();
  if (threadIdx.x == 0) cyc[blockIdx.x % 1024] = t1 - t0;
}
// mbarrier: every lane arrives, then waits for the phase (mbarrier.arrive + test_wait.parity): clocks a phase.
__global__ void mbar_k(int iters, long long *cyc) {
  __shared__ __align__(8) unsigned long long bar;
  if (threadIdx.x == 0)
    asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(sm32(&bar)), "r"(blockDim.x) : "memory");
  __syncthreads();
  long long t0 = clock64();
  for (int i = 0; i < iters; i++) {
    unsigned long long st;
    asm volatile("mbarrier.arrive.shared.b64 %0, [%1];" : "=l"(st) : "r"(sm32(&bar)) : "memory");
    unsigned ok = 0;
    while (!ok)
      asm volatile("{ .reg .pred p; mbarrier.test_wait.parity.shared.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                   : "=r"(ok)
                   : "r"(sm32(&bar)), "r"(i & 1)
                   : "memory");
  }
  long long t1 = clock64();
  if (threadIdx.x == 0) cyc[blockIdx.x % 1024] = t1 - t0;
}
// A grid barrier over co-resident blocks (a cooperative launch): an arrival counter and a generation word.
__global__ void gsync_k(unsigned *cnt, volatile unsigned *gen, int iters, unsigned long long *out) {
  unsigned long long t0 = gtime();
  for (int i = 0; i < iters; i++) {
    __syncthreads();
    if (threadIdx.x == 0) {
      unsigned g = *gen;
      __threadfence();
      if (atomicAdd(cnt, 1) == gridDim.x - 1) {
        *cnt = 0;
        __threadfence();
        *gen = g + 1;
      } else
        while (*gen == g) {
        }
    }
    __syncthreads();
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) out[0] = gtime() - t0;
}
// Launch and the in-stream gap: each launch's first block records its start and end on the device timer.
// Launch and the in-stream gap: each launch's first block records its start, spins `spin` ns (longer than the
// host's issue, so the queue runs ahead and the gap is the device's), and records its end.
__global__ void empty_k(unsigned long long *t, int i, unsigned spin) {
  unsigned long long t0 = gtime();
  if (spin)
    while (gtime() - t0 < spin) {
    }
  if (threadIdx.x == 0 && blockIdx.x == 0) t[2 * i] = t0, t[2 * i + 1] = gtime();
}

// The driver's JIT of a primitive module (the part executes it when the JIT accepts it).
static CUmodule jit(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) return nullptr;
  std::string ptx;
  char b[4096];
  size_t n;
  while ((n = fread(b, 1, sizeof b, f)) > 0) ptx.append(b, n);
  fclose(f);
  char log[4096] = {0};
  CUjit_option o[] = {CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
  void *v[] = {log, (void *)(size_t)sizeof log};
  CUmodule m = nullptr;
  if (cuModuleLoadDataEx(&m, ptx.c_str(), 2, o, v) != CUDA_SUCCESS) {
    std::string l(log);
    for (auto &c : l)
      if (c == '\n') c = ' ';
    printf("prim module=%s avail=0 jit=\"%.200s\"\n", path, l.c_str());
    return nullptr;
  }
  printf("prim module=%s avail=1\n", path);
  return m;
}

// The memory sources a mover streams from: an L2-resident set (a quarter of the L2) and a DRAM one (16 x the L2,
// within [256 MiB, 1 GiB]).
struct Src2 {
  const char *name;
  size_t bytes;
};

template <typename T, int U>
static Pt ldg_pt(const char *buf, size_t bytes, int warps) {
  int bps = std::max(1, warps / 16), threads = 32 * warps / bps, blocks = sms * bps;
  size_t n = bytes / sizeof(T);
  int passes = std::max(1, (int)((size_t)1 << 28) / (int)std::max<size_t>(bytes, 1));
  passes = std::min(passes, 64);
  unsigned *out;
  CK(cudaMalloc(&out, 4));
  auto ms = timed([&] { ldg_k<T, U><<<blocks, threads>>>((const T *)buf, n, passes, out); });
  CK(cudaFree(out));
  size_t per = (size_t)blocks * threads * U;  // elements a sweep step
  double moved = double(n / per * per) * sizeof(T) * passes;
  return rate_pt(double(warps) * 32 * U * sizeof(T), moved, ms);
}

template <typename T>
static void ldg_prim(const char *name, const char *buf, size_t bytes, std::vector<double> lat) {
  std::vector<Pt> pts;
  for (int w : {1, 2, 4, 8, 16, 32}) {
    if (w * 32 > max_thr_sm) continue;
    pts.push_back(ldg_pt<T, 1>(buf, bytes, w));
    pts.push_back(ldg_pt<T, 2>(buf, bytes, w));
    pts.push_back(ldg_pt<T, 4>(buf, bytes, w));
    pts.push_back(ldg_pt<T, 8>(buf, bytes, w));
  }
  put(name, pts, lat, "B/ns/sm", "ns", "B/sm");
}

// The chase's buffer: `slots` slots `stride` bytes apart in one random cycle (Sattolo), each slot's first word the
// next slot's element index (T elements).
static void fill_chase(char *dev, size_t bytes, size_t stride, size_t esize) {
  size_t slots = bytes / stride;
  std::vector<unsigned> next(slots);
  std::vector<size_t> perm(slots);
  for (size_t i = 0; i < slots; i++) perm[i] = i;
  unsigned long long x = 88172645463325252ull;
  for (size_t i = slots - 1; i > 0; i--) {
    x ^= x << 13, x ^= x >> 7, x ^= x << 17;
    std::swap(perm[i], perm[x % i]);
  }
  std::vector<unsigned> host(bytes / 4, 0);
  for (size_t i = 0; i < slots; i++) {
    size_t a = perm[i], b = perm[(i + 1) % slots];
    host[a * stride / 4] = (unsigned)(b * stride / esize);
  }
  CK(cudaMemcpy(dev, host.data(), bytes, cudaMemcpyHostToDevice));
}

// One lane's chase: from the cycle's start for an L2 set (`carry` false: every rep re-reads the slots, after a
// warm rep), else carried on (a DRAM set: no slot read twice).
template <typename T>
static std::vector<double> chase(const char *buf, int steps, bool carry) {
  unsigned long long *o, h[2];
  CK(cudaMalloc(&o, 16));
  CK(cudaMemset(o, 0, 16));
  std::vector<double> ns;
  for (int r = 0; r < 4; r++) {
    if (!carry) CK(cudaMemset(o, 0, 16));
    chase_k<T><<<1, 1>>>((const T *)buf, steps, o);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h, o, 16, cudaMemcpyDeviceToHost));
    if (r > 0) ns.push_back(double(h[0]) / steps);
  }
  CK(cudaFree(o));
  return ns;
}

template <int W, int U>
static Pt cpa_pt(const char *buf, size_t bytes, int warps) {
  int bps = std::max(1, warps / 16), threads = 32 * warps / bps, blocks = sms * bps;
  size_t smem = (size_t)threads * U * W;
  if (smem > (size_t)smem_optin || smem * bps > (size_t)smem_sm) return {0, 0, 0, 0};
  CK(cudaFuncSetAttribute(cpa_k<W, U>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  int passes = std::min(64, std::max(1, (int)(((size_t)1 << 28) / bytes)));
  unsigned *out;
  CK(cudaMalloc(&out, 4));
  auto ms = timed([&] { cpa_k<W, U><<<blocks, threads, smem>>>(buf, bytes, passes, out); });
  CK(cudaFree(out));
  size_t per = (size_t)blocks * threads * U * W;
  return rate_pt(double(warps) * 32 * U * W, double(bytes / per * per) * passes, ms);
}

template <int W>
static void cpa_prim(const char *name, const char *buf, size_t bytes, std::vector<double> lat) {
  std::vector<Pt> pts;
  for (int w : {1, 2, 4, 8, 16, 32}) {
    if (w * 32 > max_thr_sm) continue;
    for (Pt p : {cpa_pt<W, 1>(buf, bytes, w), cpa_pt<W, 2>(buf, bytes, w), cpa_pt<W, 4>(buf, bytes, w),
                 cpa_pt<W, 8>(buf, bytes, w)})
      if (p.fly > 0) pts.push_back(p);
  }
  put(name, pts, lat, "B/ns/sm", "ns", "B/sm");
}

template <int U>
static Pt panel_pt(const char *buf, size_t bytes, int run, int bps) {
  int rows = 128, row_bytes = 8192, threads = 256, blocks = sms * bps;
  size_t smem = (size_t)U * rows * run;
  if (smem > (size_t)smem_optin || smem * bps > (size_t)smem_sm) return {0, 0, 0, 0};
  if ((size_t)blocks * rows * row_bytes > bytes) return {0, 0, 0, 0};
  CK(cudaFuncSetAttribute(panel_k<U>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  int trips = row_bytes / run;  // every block reads its rows whole once
  unsigned *out;
  CK(cudaMalloc(&out, 4));
  auto ms = timed([&] { panel_k<U><<<blocks, threads, smem>>>(buf, rows, row_bytes, run, trips, out); });
  CK(cudaFree(out));
  return rate_pt(double(bps) * U * rows * run, double(blocks) * rows * row_bytes, ms);
}

static void prim_compute() {
  if (!cpts.count("ffma")) eff_run();
  auto row = [&](const std::string &op) {
    std::vector<Pt> pts;
    for (auto &p : cpts[op]) pts.push_back({p.fly, p.per_clk, p.per_clk, p.per_clk});
    put(op, pts, clat[op], "flop/clk/sm", "clk", "instr/sm");
  };
  for (auto &kv : cpts) row(kv.first);
}

static void prim_run() {
  CK(cudaDeviceGetAttribute(&smem_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0));
  CK(cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
  CK(cudaDeviceGetAttribute(&max_thr_sm, cudaDevAttrMaxThreadsPerMultiProcessor, 0));
  CK(cudaFree(0));
  CU(cuInit(0));
  // The clock prim converts with: clock64 over %globaltimer under FFMA load, the median of 1 s of windows.
  {
    float *o;
    unsigned long long *sp, h[2];
    CK(cudaMalloc(&o, 4));
    CK(cudaMalloc(&sp, 16));
    std::vector<double> k;
    for (int i = 0; i < 40; i++) {
      clock_k<<<sms * 8, 256>>>(o, 50000, sp);
      CK(cudaMemcpy(h, sp, 16, cudaMemcpyDeviceToHost));
      if (i >= 10) k.push_back(double(h[0]) / double(h[1]));
    }
    prim_ghz = med(k);
    printf("prim clock_ghz=%.4f\n", prim_ghz);
    CK(cudaFree(o));
    CK(cudaFree(sp));
  }
  // ---- compute: the operations' sweeps (eff) and the modules the JIT may refuse ----
  {
    float *out;
    long long *cyc;
    CK(cudaMalloc(&out, 8));
    CK(cudaMemset(out, 0, 8));
    CK(cudaMalloc(&cyc, 1024 * 8));
    if (CUmodule m = jit("place_calib_fp8.ptx")) {
      for (int c : {1, 2, 4, 8}) {
        CUfunction f;
        std::string nm = "fp8_c" + std::to_string(c);
        if (cuModuleGetFunction(&f, m, nm.c_str()) != CUDA_SUCCESS) continue;
        eff_sweep("mma.m16n8k32.e4m3", c,
                  [f](int threads, float *o, int iters, long long *cy) {
                    void *a[] = {&o, &iters, &cy};
                    eff_launch_failed |= cuLaunchKernel(f, sms, 1, 1, threads, 1, 1, 0, 0, a, 0) != CUDA_SUCCESS;
                  },
                  2.0 * 16 * 8 * 32);
      }
      eff_print("mma.m16n8k32.e4m3");
    }
    if (CUmodule m = jit("place_calib_wgmma.ptx")) {
      for (int c : {1, 2}) {
        CUfunction f;
        std::string nm = "wgmma_c" + std::to_string(c);
        if (cuModuleGetFunction(&f, m, nm.c_str()) != CUDA_SUCCESS) continue;
        // Warpgroups (128 lanes) an SM: 1 .. 4 (as many warps a scheduler).
        eff_sweep("wg.m64n128k16.bf16", c,
                  [f](int threads, float *o, int iters, long long *cy) {
                    void *a[] = {&o, &iters, &cy};
                    eff_launch_failed |= cuLaunchKernel(f, sms, 1, 1, threads, 1, 1, 0, 0, a, 0) != CUDA_SUCCESS;
                  },
                  2.0 * 64 * 128 * 16, 128, {1, 2, 3, 4});
      }
      eff_print("wg.m64n128k16.bf16", 128);
    }
    CK(cudaFree(out));
    CK(cudaFree(cyc));
  }
  prim_compute();
  // ---- movers: global loads by width and source; cp.async by width and source; the panel stream by run ----
  size_t l2b = (size_t)l2, dram = std::min<size_t>(std::max<size_t>(16 * l2b, (size_t)256 << 20), (size_t)1 << 30);
  char *buf;
  CK(cudaMalloc(&buf, dram));
  CK(cudaMemset(buf, 0, dram));
  Src2 srcs[] = {{"l2", l2b / 4 / 4096 * 4096}, {"dram", dram}};
  for (auto s : srcs) {
    // One use's latency: a chase over the set's 256-byte slots in one random cycle (L2: re-read from the start
    // after a warm rep; DRAM: 1 GiB of slots, carried on, so no slot is read twice).
    bool carry = std::string(s.name) != "l2";
    fill_chase(buf, s.bytes, 256, 16);
    auto l16 = chase<uint4>(buf, 20000, carry);
    fill_chase(buf, s.bytes, 256, 8);
    auto l8 = chase<uint2>(buf, 20000, carry);
    fill_chase(buf, s.bytes, 256, 4);
    auto l4 = chase<unsigned>(buf, 20000, carry);
    std::vector<double> lc;
    {
      fill_chase(buf, s.bytes, 256, 16);
      unsigned long long *o, h[2];
      CK(cudaMalloc(&o, 16));
      CK(cudaMemset(o, 0, 16));
      for (int r = 0; r < 4; r++) {
        if (!carry) CK(cudaMemset(o, 0, 16));
        cpa_chase_k<<<1, 1>>>((const uint4 *)buf, 20000, o);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(h, o, 16, cudaMemcpyDeviceToHost));
        if (r > 0) lc.push_back(double(h[0]) / 20000);
      }
      CK(cudaFree(o));
    }
    std::string n(s.name);
    ldg_prim<unsigned>(("ld.global.4." + n).c_str(), buf, s.bytes, l4);
    ldg_prim<uint2>(("ld.global.8." + n).c_str(), buf, s.bytes, l8);
    ldg_prim<uint4>(("ld.global.16." + n).c_str(), buf, s.bytes, l16);
    cpa_prim<4>(("cp.async.4." + n).c_str(), buf, s.bytes, lc);
    cpa_prim<8>(("cp.async.8." + n).c_str(), buf, s.bytes, lc);
    cpa_prim<16>(("cp.async.16." + n).c_str(), buf, s.bytes, lc);
    // The panel stream (a tile's operand panel, `run` contiguous bytes of each of 128 rows a trip): DRAM's and
    // the L2's rate by run (a set of the source's size; the L2 one re-read by every pass).
    // DRAM pages and the cache's misses in flight: the rate at a short run falls as more rows stream at once and
    // as more of a row's line is in flight (its sectors' misses wait on one fill), so one row per (run, rows a
    // unit streams at once: 128 x blocks an SM, bytes in flight a unit) — each point its own row.
    for (int run : {16, 32, 64, 128, 256})
      for (int bps : {1, 2, 4})
        for (Pt p : {panel_pt<1>(buf, s.bytes, run, bps), panel_pt<2>(buf, s.bytes, run, bps),
                     panel_pt<4>(buf, s.bytes, run, bps), panel_pt<8>(buf, s.bytes, run, bps)})
          if (p.fly > 0)
            put("cp.async.16." + n + ".run" + std::to_string(run) + ".rows" + std::to_string(128 * bps) + ".fly" +
                    std::to_string((long long)p.fly),
                {p}, lc, "B/ns/sm", "ns", "B/sm");
  }
  // TMA boxes (the JIT's), L2-resident (a 2048 x 2048 f32 map) and DRAM (2048 x rows of 16 x the L2).
  if (CUmodule m = jit("place_calib_tma.ptx")) {
    CUfunction f;
    CU(cuModuleGetFunction(&f, m, "tma_k"));
    CU(cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_optin - 1024));
    int *sink;
    CK(cudaMalloc(&sink, 4096 * 4));
    for (auto s : srcs) {
      int nx = 2048, ny = (int)std::min<size_t>(s.bytes / (nx * 4), 1 << 20);
      if (std::string(s.name) == "l2") ny = std::min(ny, 2048);
      for (int h : {32, 128}) {
        int bytes = 32 * 4 * h, buf1 = (bytes + 1023) / 1024 * 1024;
        CUtensorMap tm{};
        cuuint64_t dims[2] = {(cuuint64_t)nx, (cuuint64_t)ny}, strides[1] = {(cuuint64_t)nx * 4};
        cuuint32_t box[2] = {32, (cuuint32_t)h}, es[2] = {1, 1};
        CU(cuTensorMapEncodeTiled(&tm, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, buf, dims, strides, box, es,
                                  CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                  CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        std::vector<Pt> pts;
        std::vector<double> lat;
        int copies = 2048;
        for (int bps : {1, 2})
          for (int depth : {1, 2, 4, 8}) {
            if ((size_t)depth * buf1 > (size_t)smem_optin - 1024 || (size_t)depth * buf1 * bps > (size_t)smem_sm)
              continue;
            int blocks = sms * bps, sm_ = depth * buf1;
            auto ms = timed([&] {
              void *a[] = {&tm, &bytes, &h, &nx, &ny, &depth, &copies, &sink};
              CU(cuLaunchKernel(f, blocks, 1, 1, 32, 1, 1, sm_, 0, a, 0));
            });
            pts.push_back(rate_pt(double(depth) * bytes * bps, double(bytes) * copies * blocks, ms));
            if (depth == 1 && bps == 1) {
              int one = 1;
              auto ms1 = timed([&] {
                void *a[] = {&tm, &bytes, &h, &nx, &ny, &depth, &copies, &sink};
                CU(cuLaunchKernel(f, one, 1, 1, 32, 1, 1, sm_, 0, a, 0));
              });
              for (double x : ms1) lat.push_back(x * 1e6 / copies);
            }
          }
        put("tma.box" + std::to_string(bytes) + "." + s.name, pts, lat, "B/ns/sm", "ns", "B/sm");
      }
    }
    CK(cudaFree(sink));
  }
  // ---- shared memory: ld.shared 4 / 16 B and ldmatrix.x4 (clocks: bytes / clock / SM) ----
  {
    long long *cyc, *o;
    unsigned *out;
    CK(cudaMalloc(&cyc, 1024 * 8));
    CK(cudaMalloc(&o, 16));
    CK(cudaMalloc(&out, 4));
    auto span = [&](const std::function<void(int)> &f, int threads) {
      std::vector<double> v;
      for (int r = 0; r < 3; r++) {
        CK(cudaMemset(cyc, 0, 1024 * 8));
        f(threads);
        CK(cudaDeviceSynchronize());
        std::vector<long long> h(1024);
        CK(cudaMemcpy(h.data(), cyc, 1024 * 8, cudaMemcpyDeviceToHost));
        v.push_back(double(*std::max_element(h.begin(), h.begin() + std::min(sms, 1024))));
      }
      return v;
    };
    auto lat = [&](void (*chase)(int, long long *)) {
      std::vector<double> v;
      long long h[2];
      for (int r = 0; r < 4; r++) {
        chase<<<1, 32>>>(4096, o);
        CK(cudaMemcpy(h, o, 16, cudaMemcpyDeviceToHost));
        if (r > 0) v.push_back(double(h[0]) / 4096);
      }
      return v;
    };
    int iters = 2048, zero = 0;
    auto shpt = [&](double fly, double bytes_sm, std::vector<double> spans) {
      std::vector<double> r;
      for (double s : spans) r.push_back(bytes_sm / s);
      std::sort(r.begin(), r.end());
      return Pt{fly, r[1], r[0], r[2]};
    };
#define LDS_PTS(W, vec)                                                                                       \
  for (int w : {1, 2, 4, 8, 16, 32}) {                                                                        \
    double b = double(w) * 32 * (W) * iters;                                                                  \
    vec.push_back(shpt(w * 32.0 * (W) * 1, b * 1, span([&](int t) { lds_k<W, 1><<<sms, t>>>(iters, zero, cyc, out); }, w * 32))); \
    vec.push_back(shpt(w * 32.0 * (W) * 4, b * 4, span([&](int t) { lds_k<W, 4><<<sms, t>>>(iters, zero, cyc, out); }, w * 32))); \
  }
    std::vector<Pt> p4, p16, pm;
    LDS_PTS(4, p4)
    LDS_PTS(16, p16)
    for (int w : {1, 2, 4, 8, 16, 32}) {
      double b = double(w) * 512 * iters;
      pm.push_back(shpt(w * 512.0, b, span([&](int t) { ldsm_k<1><<<sms, t>>>(iters, zero, cyc, out); }, w * 32)));
      pm.push_back(shpt(w * 2048.0, 4 * b, span([&](int t) { ldsm_k<4><<<sms, t>>>(iters, zero, cyc, out); }, w * 32)));
    }
    put("ld.shared.4", p4, lat(lds_chase_k<0>), "B/clk/sm", "clk", "B/sm");
    put("ld.shared.16", p16, lat(lds_chase_k<1>), "B/clk/sm", "clk", "B/sm");
    put("ldmatrix.x4", pm, lat(lds_chase_k<2>), "B/clk/sm", "clk", "B/sm");
    CK(cudaFree(cyc));
    CK(cudaFree(o));
    CK(cudaFree(out));
  }
  // ---- result movement: stores by width (DRAM), red.global.add.f32 three ways, the epilogue through shared ----
  {
    unsigned long long *o, h[2];
    CK(cudaMalloc(&o, 16));
    std::vector<double> sl;
    for (int r = 0; r < 4; r++) {
      stfence_k<<<1, 1>>>((unsigned *)buf, 4096, o);
      CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(h, o, 16, cudaMemcpyDeviceToHost));
      if (r > 0) sl.push_back(double(h[0]) / 4096);
    }
    // Stores of each width to a DRAM-sized buffer and to an L2-resident one (a quarter of the L2, `passes` times).
    auto st = [&](auto tag, const char *name, size_t bytes) {
      using T = decltype(tag);
      std::vector<Pt> pts;
      size_t n = bytes / sizeof(T);
      int passes = std::min(64, std::max(1, (int)(((size_t)1 << 28) / bytes)));
      for (int w : {1, 2, 4, 8, 16, 32}) {
        if (w * 32 > max_thr_sm) continue;
        int bps = std::max(1, w / 16), threads = 32 * w / bps, blocks = sms * bps;
        auto ms1 = timed([&] { stg_k<T, 1><<<blocks, threads>>>((T *)buf, n, passes); });
        auto ms4 = timed([&] { stg_k<T, 4><<<blocks, threads>>>((T *)buf, n, passes); });
        size_t per1 = (size_t)blocks * threads, per4 = per1 * 4;
        pts.push_back(rate_pt(w * 32.0 * sizeof(T), double(n / per1 * per1) * sizeof(T) * passes, ms1));
        pts.push_back(rate_pt(w * 128.0 * sizeof(T), double(n / per4 * per4) * sizeof(T) * passes, ms4));
      }
      put(name, pts, sl, "B/ns/sm", "ns", "B/sm");
    };
    st((unsigned)0, "st.global.4.dram", dram);
    st(uint2{}, "st.global.8.dram", dram);
    st(uint4{}, "st.global.16.dram", dram);
    st((unsigned)0, "st.global.4.l2", l2b / 4 / 4096 * 4096);
    st(uint4{}, "st.global.16.l2", l2b / 4 / 4096 * 4096);
    // Reductions: ops / ns / SM.
    float *rp = (float *)buf;
    size_t rn = l2b / 8 / 4;
    const char *rn_[] = {"red.add.f32", "red.add.f32.warp", "red.add.f32.one", "red.add.f32.frag",
                         "red.add.f32.sector"};
    std::vector<double> al;
    for (int r = 0; r < 4; r++) {
      atom_chase_k<<<1, 1>>>(rp, 4096, 0, o);
      CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(h, o, 16, cudaMemcpyDeviceToHost));
      if (r > 0) al.push_back(double(h[0]) / 4096);
    }
    for (int mode = 0; mode < 5; mode++) {
      std::vector<Pt> pts;
      for (int w : {1, 2, 4, 8, 16, 32}) {
        if (w * 32 > max_thr_sm) continue;
        int bps = std::max(1, w / 16), threads = 32 * w / bps, blocks = sms * bps;
        int iters = mode == 2 ? 16 : 256;
        auto ms1 = timed([&] { red_k<1><<<blocks, threads>>>(rp, rn, iters, mode); });
        auto ms4 = timed([&] { red_k<4><<<blocks, threads>>>(rp, rn, iters, mode); });
        pts.push_back(rate_pt(w * 32.0, double(blocks) * threads * iters, ms1));
        pts.push_back(rate_pt(w * 128.0, double(blocks) * threads * iters * 4, ms4));
      }
      put(rn_[mode], pts, al, "op/ns/sm", "ns", "op/sm");
    }
    // The epilogue: 32 KiB tiles staged in shared memory, 1 .. 3 blocks an SM.
    size_t tiles_all = dram / 32768;
    CK(cudaFuncSetAttribute(epi_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 32768));
    std::vector<Pt> pts;
    std::vector<double> el;
    for (int bps : {1, 2, 3}) {
      if ((size_t)bps * 33792 > (size_t)smem_sm) continue;
      int blocks = sms * bps, tiles = (int)(tiles_all / blocks);
      auto ms = timed([&] { epi_k<<<blocks, 256, 32768>>>((float4 *)buf, tiles_all, tiles); });
      pts.push_back(rate_pt(bps * 32768.0, double(blocks) * tiles * 32768, ms));
      if (bps == 1) {
        auto ms1 = timed([&] { epi_k<<<1, 256, 32768>>>((float4 *)buf, tiles_all, 64); });
        for (double x : ms1) el.push_back(x * 1e6 / 64);
      }
    }
    put("epilogue.smem.st.global.16", pts, el, "B/ns/sm", "ns", "B/sm");
    CK(cudaFree(o));
  }
  // ---- synchronization: bar.sync by block size, mbarrier phases, a grid barrier ----
  {
    long long *cyc;
    CK(cudaMalloc(&cyc, 1024 * 8));
    int iters = 4096;
    for (int threads : {128, 256, 512, 1024}) {
      std::vector<double> v;
      for (int r = 0; r < 4; r++) {
        bar_k<<<sms, threads>>>(iters, cyc);
        CK(cudaDeviceSynchronize());
        std::vector<long long> h(1024);
        CK(cudaMemcpy(h.data(), cyc, 1024 * 8, cudaMemcpyDeviceToHost));
        if (r > 0) v.push_back(double(*std::max_element(h.begin(), h.begin() + std::min(sms, 1024))) / iters);
      }
      double c = med(v);
      put("bar.sync." + std::to_string(threads), {{1, 1 / c, 1 / v.back(), 1 / v.front()}}, v, "bar/clk/sm", "clk",
          "bar/sm");
    }
    {
      std::vector<double> v;
      for (int r = 0; r < 4; r++) {
        mbar_k<<<sms, 256>>>(iters, cyc);
        CK(cudaDeviceSynchronize());
        std::vector<long long> h(1024);
        CK(cudaMemcpy(h.data(), cyc, 1024 * 8, cudaMemcpyDeviceToHost));
        if (r > 0) v.push_back(double(*std::max_element(h.begin(), h.begin() + std::min(sms, 1024))) / iters);
      }
      std::sort(v.begin(), v.end());
      put("mbarrier.256", {{1, 1 / v[1], 1 / v[2], 1 / v[0]}}, v, "phase/clk/sm", "clk", "phase/sm");
    }
    {
      unsigned *cnt;
      unsigned long long *o, h0;
      CK(cudaMalloc(&cnt, 8));
      CK(cudaMalloc(&o, 8));
      std::vector<double> v;
      int gi = 2000;
      for (int r = 0; r < 4; r++) {
        CK(cudaMemset(cnt, 0, 8));
        unsigned *gen = cnt + 1;
        void *a[] = {&cnt, &gen, &gi, &o};
        CK(cudaLaunchCooperativeKernel((void *)gsync_k, sms, 256, a, 0, 0));
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(&h0, o, 8, cudaMemcpyDeviceToHost));
        if (r > 0) v.push_back(double(h0) / gi);
      }
      std::sort(v.begin(), v.end());
      put("grid.sync", {{1, 1 / v[1], 1 / v[2], 1 / v[0]}}, v, "sync/ns", "ns", "sync");
      CK(cudaFree(cnt));
      CK(cudaFree(o));
    }
    CK(cudaFree(cyc));
  }
  // ---- dispatch: the host's issue of a launch (wall time, the queue not waited on), the device's gap between
  // dependent launches (each spins 20 us: the queue runs ahead of the device) ----
  {
    int n = 1000;
    unsigned long long *t;
    CK(cudaMalloc(&t, 16 * n));
    for (int blocks : {1, sms}) {
      std::vector<double> issue, gaps;
      for (int r = 0; r < 4; r++) {
        CK(cudaDeviceSynchronize());
        auto w0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 200; i++) empty_k<<<blocks, 32>>>(t, i, 0);
        auto w1 = std::chrono::steady_clock::now();
        CK(cudaDeviceSynchronize());
        for (int i = 0; i < n; i++) empty_k<<<blocks, 32>>>(t, i, 20000);
        CK(cudaDeviceSynchronize());
        std::vector<unsigned long long> h(2 * n);
        CK(cudaMemcpy(h.data(), t, 16 * n, cudaMemcpyDeviceToHost));
        // The mean gap over the second half (the timer's ticks are coarse: a mean of many differences, not one).
        double g = 0;
        for (int i = n / 2; i < n; i++) g += double(h[2 * i]) - double(h[2 * i - 1]);
        if (r > 0) {
          issue.push_back(std::chrono::duration<double, std::nano>(w1 - w0).count() / 200);
          gaps.push_back(g / (n - n / 2));
        }
      }
      std::sort(issue.begin(), issue.end());
      std::sort(gaps.begin(), gaps.end());
      std::string b = blocks == 1 ? "1" : "sms";
      put("launch.issue." + b, {{1, 1 / issue[1], 1 / issue.back(), 1 / issue.front()}}, issue, "launch/ns", "ns",
          "launch");
      put("launch.gap." + b, {{1, 1 / gaps[1], 1 / gaps.back(), 1 / gaps.front()}}, gaps, "launch/ns", "ns",
          "launch");
    }
    CK(cudaFree(t));
  }
  CK(cudaFree(buf));
  // The calibration file's value: milli-units, name:rate:lo:hi:lat:lo:hi:conc:lo:hi.
  printf("calib place.prim=");
  auto mu = [](double x) { return (unsigned long long)std::llround(std::max(0.0, x) * 1000.0); };
  for (size_t i = 0; i < prows.size(); i++) {
    auto &r = prows[i];
    printf("%s%s:%llu:%llu:%llu:%llu:%llu:%llu:%llu:%llu:%llu", i ? "," : "", r.name.c_str(), mu(r.rate),
           mu(r.rlo), mu(r.rhi), mu(r.lat), mu(r.llo), mu(r.lhi), mu(r.conc), mu(r.clo), mu(r.chi));
  }
  printf(" place.prim_ghz=%llu\n", (unsigned long long)std::llround(prim_ghz * 1e6));
}

int main(int argc, char **argv) {
  CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  CK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0));
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  printf("device name=\"%s\" sms=%d l2_bytes=%d cc=%d%d\n", p.name, sms, l2, p.major, p.minor);
  bool all = argc < 2;
  auto want = [&](const char *k) {
    if (all) return true;
    for (int i = 1; i < argc; i++)
      if (!strcmp(argv[i], k)) return true;
    return false;
  };
  if (want("clock")) clock_run();
  if (want("start")) start_run();
  if (want("keep")) keep_run();
  if (want("eff")) eff_run();
  if (want("peak")) peak_run();
  if (want("prim")) prim_run();
  return 0;
}
