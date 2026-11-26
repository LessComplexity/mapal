// ADR-0050 C3: the calibration probes the machine model's evaluator reads (crates/mapal-place, the GPU translation
// crates/backends/llvm/src/place.rs). Each prints `<probe> k=v ...` lines; place_calib.py reads them into facts.
//   start   the start timing of a grid's blocks past its first wave: a uniform-work kernel at 1 / 2 blocks an SM over
//           4 waves, each block's %globaltimer start / end and %smid; per wave past the first, the spread of its starts
//           over a block's run (lockstep: the wave's slots start together; stagger: they start as the ones before end,
//           spread over a run). `start lockstep|stagger spread_pm=` (median over waves and reps).
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
// build: nvcc -O3 -arch=sm_120 -std=c++17 place_calib.cu -o place_calib; run: ./place_calib [start|keep|eff|clock]
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

#define CK(x)                                                                              \
  do {                                                                                     \
    cudaError_t e = (x);                                                                   \
    if (e != cudaSuccess) {                                                                \
      fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e));            \
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
// Uniform work: `iters` trips of 8 FFMA chains plus one L2-resident 16-byte load a thread.
__global__ void start_k(unsigned long long *t, unsigned *sm, const float4 *buf, size_t n, int iters, float *sink) {
  unsigned long long t0 = gtime();
  float a[8];
  for (int i = 0; i < 8; i++) a[i] = threadIdx.x * 1e-3f + i;
  size_t at = (blockIdx.x * blockDim.x + threadIdx.x) % n;
  for (int it = 0; it < iters; it++) {
    float4 v = __ldcg(buf + at);
    at = (at + 4099) % n;
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
  size_t n = 1 << 20;  // 16 MiB: L2-resident
  float4 *buf;
  float *sink;
  CK(cudaMalloc(&buf, n * 16));
  CK(cudaMemset(buf, 0, n * 16));
  CK(cudaMalloc(&sink, 4));
  for (int per : {1, 2}) {
    int threads = 256, waves = 4, blocks = sms * per * waves;
    unsigned long long *t;
    unsigned *sm;
    CK(cudaMalloc(&t, blocks * 16));
    CK(cudaMalloc(&sm, blocks * 4));
    std::vector<unsigned long long> h(2 * blocks);
    std::vector<double> spreads;
    for (int rep = 0; rep < 7; rep++) {
      start_k<<<blocks, threads>>>(t, sm, buf, n, 4000, sink);
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
        printf("start per_sm=%d blocks=%d run_ns=%.0f first_wave_spread_pm=%.0f\n", per, blocks, run,
               1000.0 * double(se[w - 1 - w / 20].first - se[w / 20].first) / run);
    }
    std::sort(spreads.begin(), spreads.end());
    double med = spreads[spreads.size() / 2];
    // Lockstep where a wave's slots start within a quarter of a run of each other.
    printf("start per_sm=%d spread_pm=%.0f min_pm=%.0f max_pm=%.0f verdict=%s\n", per, 1000 * med,
           1000 * spreads.front(), 1000 * spreads.back(), med < 0.25 ? "lockstep" : "stagger");
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
    printf("keep pattern=%s keep_bytes=%zu plateau_gbs=%.0f dram_gbs=%.0f\n", pats[pi], keep, plateau, dram);
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

template <int C>
__global__ void mma_bf16_k(float *out, int iters, long long *cyc) {
  float d[C][4];
#pragma unroll
  for (int i = 0; i < C; i++)
    for (int j = 0; j < 4; j++) d[i][j] = 0;
  unsigned a0 = 0x3f803f80u + (unsigned)(out[0] != 0), a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0;
  __syncthreads();
  long long t0 = clock64();
  for (int it = 0; it < iters; it++) {
#pragma unroll
    for (int i = 0; i < C; i++)
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
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

typedef void (*kern)(float *, int, long long *);

static void eff_one(const char *op, int chains, kern k, double flops_per_trip_warp, double rate) {
  float *out;
  long long *cyc;
  CK(cudaMalloc(&out, 8));
  CK(cudaMemset(out, 0, 8));
  CK(cudaMalloc(&cyc, 1024 * 8));
  // Warps a scheduler: 0.25 .. 8 (one block an SM of 4·wps warps; a quarter: one warp an SM).
  double wpss[] = {0.25, 0.5, 1, 2, 4, 8};
  for (double wps : wpss) {
    int warps = std::max(1, (int)(4 * wps));
    int threads = 32 * warps;
    int iters = 4096;
    for (int rep = 0; rep < 3; rep++) {
      CK(cudaMemset(cyc, 0, 1024 * 8));
      k<<<sms, threads>>>(out, iters, cyc);
      CK(cudaDeviceSynchronize());
    }
    std::vector<long long> h(1024);
    CK(cudaMemcpy(h.data(), cyc, 1024 * 8, cudaMemcpyDeviceToHost));
    long long span = *std::max_element(h.begin(), h.begin() + std::min(sms, 1024));
    double flops = flops_per_trip_warp * warps * iters * chains;  // an SM's
    double per_clk = flops / double(span);
    printf("eff op=%s chains=%d wps=%.2f flops_clk_sm=%.1f eff_pm=%.0f\n", op, chains, wps, per_clk,
           1000.0 * per_clk / rate);
    if (chains == 1 && wps == 0.25) {
      // One chain, one warp: a trip's clocks are the dependent issue latency.
      printf("lat op=%s clk=%.2f\n", op, double(span) / iters);
    }
  }
  CK(cudaFree(out));
  CK(cudaFree(cyc));
}

static void eff_run() {
  // flops a warp a trip a chain: FFMA 32 lanes x 2; m16n8k8 2·16·8·8; m16n8k16 2·16·8·16.
  eff_one("ffma", 1, ffma_k<1>, 64, 256);
  eff_one("ffma", 2, ffma_k<2>, 64, 256);
  eff_one("ffma", 4, ffma_k<4>, 64, 256);
  eff_one("ffma", 8, ffma_k<8>, 64, 256);
  eff_one("ffma", 16, ffma_k<16>, 64, 256);
  eff_one("ffma", 32, ffma_k<32>, 64, 256);
  eff_one("ffma", 64, ffma_k<64>, 64, 256);
  eff_one("mma.m16n8k8.tf32", 1, mma_tf32_k<1>, 2048, 256);
  eff_one("mma.m16n8k8.tf32", 2, mma_tf32_k<2>, 2048, 256);
  eff_one("mma.m16n8k8.tf32", 4, mma_tf32_k<4>, 2048, 256);
  eff_one("mma.m16n8k8.tf32", 8, mma_tf32_k<8>, 2048, 256);
  eff_one("mma.m16n8k8.tf32", 16, mma_tf32_k<16>, 2048, 256);
  eff_one("mma.m16n8k16.bf16", 1, mma_bf16_k<1>, 4096, 512);
  eff_one("mma.m16n8k16.bf16", 2, mma_bf16_k<2>, 4096, 512);
  eff_one("mma.m16n8k16.bf16", 4, mma_bf16_k<4>, 4096, 512);
  eff_one("mma.m16n8k16.bf16", 8, mma_bf16_k<8>, 4096, 512);
  eff_one("mma.m16n8k16.bf16", 16, mma_bf16_k<16>, 4096, 512);
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
  return 0;
}
