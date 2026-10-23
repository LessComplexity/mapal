// Mapal's emitted tile kernel vs cuBLAS vs CUTLASS, one f32 C = A·B site,
// row-major, compute-only, on whatever GPU this runs on (S52).
//
// The Mapal leg is the PTX the compiler emitted from a `.mapal` source
// (`emit --target=<gpu> --device-out`, then `llc`), loaded with the driver API
// and launched with the grid the emitter chose — this harness adds nothing to
// it. cuBLAS runs `cublasSgemm` with the default math mode (pure FP32: TF32 is
// opt-in for sgemm, and the error column below would show it). CUTLASS runs
// its stock SIMT f32 GEMM (`device::Gemm<..., OpClassSimt, CUTLASS_ARCH>`).
//
// Values: every leg is checked on sampled cells. Mapal must equal a CPU
// reference that folds k IN ORDER with the same rounding (separate mul+add,
// or fmaf with --contract) BIT FOR BIT; cuBLAS/CUTLASS are compared to an f64
// reference and their max relative error is printed.
//
// Timing: legs interleaved round-robin, each launch bracketed by cudaEvents,
// L2 flushed (256 MB memset) before every timed launch, warmup 3 per leg.
// --warm (S62, after the face flag): no flush — each leg re-reads operands its
// previous launch left in L2, the state of a producer feeding its consumer.
//
// --tf32: the TF32 face. cuBLAS runs CUBLAS_TF32_TENSOR_OP_MATH, CUTLASS its
// OpClassTensorOp GEMM; values are bounded, not bit-exact: Mapal's max error vs
// f64 must stay within 4x cuBLAS-TF32's own.
//
// usage: tile_bench M N K <mapal.ptx> <kernel> gx gy block smem reps [--contract|--tf32]
//        tile_bench --program n gemms ops     (cuBLAS whole-program floor, S53)
//        tile_bench --h2d mb reps             (pageable vs pinned copy rate, S53)
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include "cutlass/gemm/device/gemm.h"

#ifndef CUTLASS_ARCH
#define CUTLASS_ARCH cutlass::arch::Sm80
#endif

#define CUDA(x)                                                                \
  do {                                                                         \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess) {                                                   \
      fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_));                 \
      exit(102);                                                               \
    }                                                                          \
  } while (0)
#define CU(x)                                                                  \
  do {                                                                         \
    CUresult r_ = (x);                                                         \
    if (r_ != CUDA_SUCCESS) {                                                  \
      const char *s_ = 0;                                                      \
      cuGetErrorString(r_, &s_);                                               \
      fprintf(stderr, "%s: %s\n", #x, s_ ? s_ : "?");                          \
      exit(102);                                                               \
    }                                                                          \
  } while (0)

using CutlassGemm = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor, float, cutlass::layout::RowMajor, float,
    cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt, CUTLASS_ARCH>;
// --tf32: CUTLASS's tensor-core GEMM for f32 operands (TF32 mma, sm_80+), in
// the configuration of CUTLASS's own Ampere TF32 example (examples/14): the
// library default (128x256x64) does not instantiate for these layouts.
using CutlassTf32 = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor, float, cutlass::layout::RowMajor, float,
    cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>, cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>, cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

static void device_facts() {
  int dev = 0, v = 0;
  cudaDeviceProp p;
  CUDA(cudaGetDeviceProperties(&p, dev));
  int drv = 0;
  cudaDriverGetVersion(&drv);
  printf("device name=\"%s\" cc=%d.%d driver=%d\n", p.name, p.major, p.minor, drv);
  struct {
    const char *k;
    cudaDeviceAttr a;
  } attrs[] = {
      {"sms", cudaDevAttrMultiProcessorCount},
      {"regs_per_sm", cudaDevAttrMaxRegistersPerMultiprocessor},
      {"regs_per_block", cudaDevAttrMaxRegistersPerBlock},
      {"smem_per_sm", cudaDevAttrMaxSharedMemoryPerMultiprocessor},
      {"smem_per_block", cudaDevAttrMaxSharedMemoryPerBlockOptin},
      {"smem_reserved_per_block", cudaDevAttrReservedSharedMemoryPerBlock},
      {"max_threads_per_sm", cudaDevAttrMaxThreadsPerMultiProcessor},
      {"max_blocks_per_sm", cudaDevAttrMaxBlocksPerMultiprocessor},
      {"max_threads_per_block", cudaDevAttrMaxThreadsPerBlock},
      {"warp", cudaDevAttrWarpSize},
      {"l2_bytes", cudaDevAttrL2CacheSize},
      {"sm_clock_khz", cudaDevAttrClockRate},
      {"mem_clock_khz", cudaDevAttrMemoryClockRate},
      // S54 `l2-window`: the persisting set-aside and the largest window.
      {"l2_persist_bytes", cudaDevAttrMaxPersistingL2CacheSize},
      {"l2_window_max", cudaDevAttrMaxAccessPolicyWindowSize},
  };
  printf("facts");
  for (auto &a : attrs) {
    CUDA(cudaDeviceGetAttribute(&v, a.a, dev));
    printf(" %s=%d", a.k, v);
  }
  printf("\n");
}

static float frand(uint64_t &s) {
  s = s * 6364136223846793005ull + 1442695040888963407ull;
  return (float)((s >> 40) & 0xFFFFFF) / (float)0x800000 - 1.0f;  // [-1, 1)
}

static double wall_ms() {
  timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

// The bench programs' operand formula, `(t·mul + add) % mod − off`, built on
// the device — what a hand-written CUDA program does instead of copying in.
__global__ void gen_operand(float *o, long long n, int mul, int add, int mod, int off) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) o[i] = (float)(((int)i * mul + add) % mod - off);
}

// --program n gemms ops: the cuBLAS whole-program floor for a Mapal e2e
// program (S53 `resident`): context, handle, alloc, `ops` operands generated
// on the device, `gemms` chained sgemms (each multiplies the last result by
// the next operand, the loop's `x ← x·w` when ops < gemms + 1), one 4-byte
// copy-out. One process = one program run; every phase synchronized.
static int program_floor(int n, int gemms, int ops) {
  double t0 = wall_ms();
  CUDA(cudaFree(0));
  double t1 = wall_ms();
  cublasHandle_t h;
  cublasCreate(&h);
  double t2 = wall_ms();
  size_t bytes = (size_t)n * n * 4;
  std::vector<float *> m(ops + 2);
  for (auto &p : m) CUDA(cudaMalloc(&p, bytes));
  CUDA(cudaDeviceSynchronize());
  double t3 = wall_ms();
  const int mul[3] = {7, 11, 5}, add[3] = {13, 57, 3}, mod[3] = {101, 103, 97}, off[3] = {50, 51, 48};
  long long cells = (long long)n * n;
  for (int k = 0; k < ops; ++k)
    gen_operand<<<(unsigned)((cells + 255) / 256), 256>>>(m[k], cells, mul[k % 3], add[k % 3],
                                                           mod[k % 3], off[k % 3]);
  CUDA(cudaDeviceSynchronize());
  double t4 = wall_ms();
  const float one = 1.0f, zero = 0.0f;
  float *x = m[0], *y = m[ops], *spare = m[ops + 1];
  for (int g = 0; g < gemms; ++g) {
    float *w = m[1 + (g % (ops - 1))];
    // Row-major y = x·w is column-major yᵀ = wᵀ·xᵀ.
    cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, w, n, x, n, &zero, y, n);
    x = y;
    std::swap(y, spare);
  }
  CUDA(cudaDeviceSynchronize());
  double t5 = wall_ms();
  float v;
  CUDA(cudaMemcpy(&v, x, 4, cudaMemcpyDeviceToHost));
  double t6 = wall_ms();
  printf("program n=%d gemms=%d ops=%d ctx_ms=%.3f handle_ms=%.3f alloc_ms=%.3f gen_ms=%.3f "
         "gemm_ms=%.3f d2h_ms=%.3f after_ctx_ms=%.3f total_ms=%.3f value=%g\n",
         n, gemms, ops, t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t6 - t1, t6 - t0, v);
  return 0;
}

// --h2d mb reps: the copy-in rate from pageable (malloc) vs pinned
// (cudaMallocHost) memory, and the copy-out rate, interleaved; medians.
// Measurement only (plan-s53 row 11): whether pinned memory is worth a rung.
static int h2d_probe(int mb, int reps) {
  CUDA(cudaFree(0));
  size_t bytes = (size_t)mb << 20;
  float *pageable = (float *)malloc(bytes), *pinned, *d;
  memset(pageable, 1, bytes);
  CUDA(cudaMallocHost(&pinned, bytes));
  memset(pinned, 1, bytes);
  CUDA(cudaMalloc(&d, bytes));
  std::vector<double> ms[4];
  for (int r = 0; r < reps + 1; ++r) {
    float *src[2] = {pageable, pinned};
    for (int k = 0; k < 2; ++k) {
      double a = wall_ms();
      CUDA(cudaMemcpy(d, src[k], bytes, cudaMemcpyHostToDevice));
      double b = wall_ms();
      CUDA(cudaMemcpy(src[k], d, bytes, cudaMemcpyDeviceToHost));
      double c = wall_ms();
      if (r) {  // first round is warmup
        ms[k].push_back(b - a);
        ms[2 + k].push_back(c - b);
      }
    }
  }
  const char *names[4] = {"h2d_pageable", "h2d_pinned", "d2h_pageable", "d2h_pinned"};
  printf("copy_rate mb=%d reps=%d", mb, reps);
  for (int k = 0; k < 4; ++k) {
    std::sort(ms[k].begin(), ms[k].end());
    double med = ms[k][ms[k].size() / 2];
    printf(" %s_ms=%.3f %s_gbps=%.2f", names[k], med, names[k], bytes / med / 1e6);
  }
  printf("\n");
  return 0;
}

// --- S53 `--decode T d reps`: the library floor for the decode program
// (gen_tiles.py `decode`), written the way a user would by hand: x, K, V, o
// device-resident, preallocated; per step one tiny kernel writes token row t
// of x from the previous output, then four cublasSgemv: K[t] = x[t]·Wk,
// V[t] = x[t]·Wv (the KV cache: one row each), s = K·q_t, o = sᵀ·V over all T
// rows (the program's extent; rows past t are zero). One 4-byte-per-value d2h
// at the end. Same formulas as the program; values are close, not bit-exact
// (cuBLAS's own summation order).
__global__ void decode_row(float *xrow, const float *o, int t, int d) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < d) xrow[j] = (float)(((t * d + j) * 7 + 13) % 101 - 50) / 64.0f + o[j];
}
__global__ void decode_q(float *q, int t, int d, float div) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < d) q[k] = (float)(((t * d + k) * 5 + 3) % 97 - 48) / div;
}

static int decode_main(int T, int d, int reps) {
  CUDA(cudaFree(0));
  device_facts();
  const float div = 16.0f * d;
  std::vector<float> wk((size_t)d * d), wv((size_t)d * d);
  for (int i = 0; i < d * d; ++i) {
    wk[i] = (float)((i * 7 + 13) % 101 - 50) / div;
    wv[i] = (float)((i * 11 + 57) % 103 - 51) / div;
  }
  float *dWk, *dWv, *dx, *dK, *dV, *dq, *ds, *dout;
  CUDA(cudaMalloc(&dWk, wk.size() * 4));
  CUDA(cudaMalloc(&dWv, wv.size() * 4));
  CUDA(cudaMalloc(&dx, (size_t)T * d * 4));
  CUDA(cudaMalloc(&dK, (size_t)T * d * 4));
  CUDA(cudaMalloc(&dV, (size_t)T * d * 4));
  CUDA(cudaMalloc(&dq, d * 4));
  CUDA(cudaMalloc(&ds, T * 4));
  CUDA(cudaMalloc(&dout, d * 4));
  CUDA(cudaMemcpy(dWk, wk.data(), wk.size() * 4, cudaMemcpyHostToDevice));
  CUDA(cudaMemcpy(dWv, wv.data(), wv.size() * 4, cudaMemcpyHostToDevice));
  cublasHandle_t h;
  cublasCreate(&h);
  cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
  const float one = 1.0f, zero = 0.0f;
  const int tb = 256;
  auto run = [&] {
    CUDA(cudaMemset(dx, 0, (size_t)T * d * 4));
    CUDA(cudaMemset(dK, 0, (size_t)T * d * 4));
    CUDA(cudaMemset(dV, 0, (size_t)T * d * 4));
    CUDA(cudaMemset(dout, 0, d * 4));
    for (int t = 0; t < T; ++t) {
      float *xr = dx + (size_t)t * d;
      decode_row<<<(d + tb - 1) / tb, tb>>>(xr, dout, t, d);
      // Row-major W (d×d) is column-major Wᵀ: y = Wᵀ·x[t] is K[t] = x[t]·W.
      cublasSgemv(h, CUBLAS_OP_N, d, d, &one, dWk, d, xr, 1, &zero, dK + (size_t)t * d, 1);
      cublasSgemv(h, CUBLAS_OP_N, d, d, &one, dWv, d, xr, 1, &zero, dV + (size_t)t * d, 1);
      decode_q<<<(d + tb - 1) / tb, tb>>>(dq, t, d, div);
      // K row-major T×d is column-major d×T: s = Kᵀ_cm·q.
      cublasSgemv(h, CUBLAS_OP_T, d, T, &one, dK, d, dq, 1, &zero, ds, 1);
      cublasSgemv(h, CUBLAS_OP_N, d, T, &one, dV, d, ds, 1, &zero, dout, 1);
    }
  };
  run();  // warmup (and the values)
  CUDA(cudaDeviceSynchronize());
  std::vector<float> o(d);
  CUDA(cudaMemcpy(o.data(), dout, d * 4, cudaMemcpyDeviceToHost));
  double sum = 0;
  for (float v : o) sum += v;
  printf("decode_floor values o0=%.9g olast=%.9g sum=%.9g\n", o[0], o[d - 1], sum);
  cudaEvent_t e0, e1;
  CUDA(cudaEventCreate(&e0));
  CUDA(cudaEventCreate(&e1));
  std::vector<float> ms;
  for (int r = 0; r < reps; ++r) {
    CUDA(cudaEventRecord(e0));
    run();
    CUDA(cudaEventRecord(e1));
    CUDA(cudaEventSynchronize(e1));
    float x;
    CUDA(cudaEventElapsedTime(&x, e0, e1));
    ms.push_back(x);
  }
  std::sort(ms.begin(), ms.end());
  printf("decode_floor T=%d d=%d reps=%d min_ms=%.3f med_ms=%.3f max_ms=%.3f per_step_us=%.2f\n", T, d,
         reps, ms.front(), ms[ms.size() / 2], ms.back(), 1e3 * ms[ms.size() / 2] / T);
  return 0;
}

// --- S62 `--splitk M N K reps dir stem...` (ADR-0040): every Mapal arm of one
// cell (`<dir>/<stem>.ptx` + `.launch`: `kernel gx gy block smem [S rkernel
// rgx rgy rblock rsmem]`), cuBLAS and stock CUTLASS in ONE process, timed
// round-robin, the order reversed on odd reps (ABBA). The face is the stem's
// suffix before any `.`: `_rc` reassoc+contract, `_r` reassoc, `_c` contract,
// else exact. A split arm's two launches (partials into a preallocated
// workspace, then the reduce) are one timed leg.
// Gates, before any timing, per Mapal arm: bit-exact on 256 sampled cells
// against the CPU fold in the arm's order (ADR-0040 §2: S contiguous partial
// folds, added in partition order; S = 1 is the ordered fold); on a reassoc
// arm also |got - ordered| <= (g_K + g_{K/S+S}) * sum|a_k b_k| (ADR-0040 §3,
// f64). After timing: the whole output re-run and compared byte for byte, and
// its FNV-1a printed (box_run.sh runs two processes: the hashes must match).
// A failed gate drops that arm from timing and exits 1 at the end.
struct Arm {
  std::string stem;
  bool contract = false, reassoc = false;
  CUfunction fn = 0, red = 0;
  unsigned gx = 0, gy = 0, block = 0, smem = 0, s = 1, rgx = 0, rgy = 0, rblock = 0, rsmem = 0;
  float *out = 0, *ws = 0;
  bool ok = true;
};

static uint64_t fnv(const std::vector<float> &v) {
  uint64_t h = 0xcbf29ce484222325ull;
  const unsigned char *p = (const unsigned char *)v.data();
  for (size_t i = 0; i < v.size() * 4; ++i) h = (h ^ p[i]) * 0x100000001b3ull;
  return h;
}

static int splitk_main(int argc, char **argv) {
  const int M = atoi(argv[2]), N = atoi(argv[3]), K = atoi(argv[4]), reps = atoi(argv[5]);
  const std::string dir = argv[6];
  CUDA(cudaFree(0));
  device_facts();
  std::vector<float> A((size_t)M * K), B((size_t)K * N);
  uint64_t seed = 0x5eed;
  for (auto &x : A) x = frand(seed);
  for (auto &x : B) x = frand(seed);
  float *dA, *dB, *dLib[2];
  CUDA(cudaMalloc(&dA, A.size() * 4));
  CUDA(cudaMalloc(&dB, B.size() * 4));
  for (auto &c : dLib) CUDA(cudaMalloc(&c, (size_t)M * N * 4));
  CUDA(cudaMemcpy(dA, A.data(), A.size() * 4, cudaMemcpyHostToDevice));
  CUDA(cudaMemcpy(dB, B.data(), B.size() * 4, cudaMemcpyHostToDevice));
  const size_t flush_bytes = 256u << 20;
  void *flush;
  CUDA(cudaMalloc(&flush, flush_bytes));

  std::vector<Arm> arms;
  for (int i = 7; i < argc; ++i) {
    Arm a;
    a.stem = argv[i];
    std::string face = a.stem.substr(0, a.stem.find('.'));
    auto ends = [&](const char *x) {
      size_t n = strlen(x);
      return face.size() >= n && face.compare(face.size() - n, n, x) == 0;
    };
    a.reassoc = ends("_r") || ends("_rc");
    a.contract = ends("_c") || ends("_rc");
    std::string path = dir + "/" + a.stem;
    FILE *f = fopen((path + ".ptx").c_str(), "rb");
    FILE *l = fopen((path + ".launch").c_str(), "r");
    if (!f || !l) { fprintf(stderr, "missing %s.ptx/.launch\n", path.c_str()); return 102; }
    std::string ptx;
    char buf[1 << 16];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) ptx.append(buf, n);
    fclose(f);
    char kname[64] = {0}, rname[64] = {0};
    int got = fscanf(l, "%63s %u %u %u %u %u %63s %u %u %u %u", kname, &a.gx, &a.gy, &a.block,
                     &a.smem, &a.s, rname, &a.rgx, &a.rgy, &a.rblock, &a.rsmem);
    fclose(l);
    if (got < 5) { fprintf(stderr, "bad launch line %s\n", path.c_str()); return 102; }
    if (got < 11) a.s = 1;
    CUmodule mod;
    CU(cuModuleLoadData(&mod, ptx.c_str()));
    CU(cuModuleGetFunction(&a.fn, mod, kname));
    if (a.smem > 48 * 1024)
      CU(cuFuncSetAttribute(a.fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, a.smem));
    int regs = 0, spill = 0, rregs = 0;
    cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, a.fn);
    cuFuncGetAttribute(&spill, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, a.fn);
    if (a.s > 1) {
      CU(cuModuleGetFunction(&a.red, mod, rname));
      cuFuncGetAttribute(&rregs, CU_FUNC_ATTRIBUTE_NUM_REGS, a.red);
      CUDA(cudaMalloc(&a.ws, (size_t)a.s * M * N * 4));
    }
    CUDA(cudaMalloc(&a.out, (size_t)M * N * 4));
    printf("arm stem=%s face=%s%s kernel=%s grid=%ux%u block=%u smem=%u regs=%d local_bytes=%d S=%u", a.stem.c_str(),
           a.reassoc ? "reassoc+" : "", a.contract ? "contract" : "exact", kname, a.gx, a.gy, a.block, a.smem,
           regs, spill, a.s);
    if (a.s > 1) printf(" reduce=%s rgrid=%ux%u rblock=%u rregs=%d", rname, a.rgx, a.rgy, a.rblock, rregs);
    printf("\n");
    arms.push_back(a);
  }

  cublasHandle_t h;
  cublasCreate(&h);
  cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
  CutlassGemm cutlass_gemm;
  CutlassGemm::Arguments cargs({M, N, K}, {dA, K}, {dB, N}, {dLib[1], N}, {dLib[1], N}, {1.0f, 0.0f});
  if (cutlass_gemm.can_implement(cargs) != cutlass::Status::kSuccess ||
      cutlass_gemm.initialize(cargs) != cutlass::Status::kSuccess) {
    fprintf(stderr, "cutlass cannot run %dx%dx%d\n", M, N, K);
    return 102;
  }
  const float one = 1.0f, zero = 0.0f;
  const size_t na = arms.size();
  auto run = [&](size_t leg) {
    if (leg < na) {
      Arm &a = arms[leg];
      float *o = a.s > 1 ? a.ws : a.out;
      void *args[] = {&dA, &dB, &o};
      CU(cuLaunchKernel(a.fn, a.gx, a.gy, 1, a.block, 1, 1, a.smem, 0, args, 0));
      if (a.s > 1) {
        CUdeviceptr none = 0;
        void *rargs[] = {&a.ws, &none, &a.out};
        CU(cuLaunchKernel(a.red, a.rgx, a.rgy, 1, a.rblock, 1, 1, a.rsmem, 0, rargs, 0));
      }
    } else if (leg == na) {
      cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, dB, N, dA, K, &zero, dLib[0], N);
    } else {
      cutlass_gemm();
    }
  };
  auto name = [&](size_t leg) { return leg < na ? arms[leg].stem : std::string(leg == na ? "cublas" : "cutlass"); };
  for (size_t leg = 0; leg < na + 2; ++leg)
    for (int w = 0; w < 3; ++w) run(leg);
  CUDA(cudaDeviceSynchronize());

  // --- value gates (sampled cells; the 256 of the single-arm harness)
  std::vector<std::pair<int, int>> cells = {{0, 0}, {M - 1, N - 1}, {0, N - 1}, {M - 1, 0}};
  uint64_t cs = 0xce11;
  for (int i = 0; i < 252; ++i) {
    cs = cs * 6364136223846793005ull + 1;
    cells.push_back({(int)((cs >> 33) % M), (int)((cs >> 13) % N)});
  }
  const double u = std::ldexp(1.0, -24);
  auto gamma = [&](double n) { return n * u / (1 - n * u); };
  // The CPU fold of one cell over k in [k0, k1) from `acc`, with the face's step.
  auto fold = [&](int i, int j, int k0, int k1, float acc, bool contract) {
    for (int k = k0; k < k1; ++k) {
      float a = A[(size_t)i * K + k], b = B[(size_t)k * N + j];
      acc = contract ? fmaf(a, b, acc) : acc + a * b;
    }
    return acc;
  };
  std::vector<std::vector<float>> before(na);
  bool all_ok = true;
  std::vector<float> C((size_t)M * N);
  for (size_t leg = 0; leg < na + 2; ++leg) {
    float *src = leg < na ? arms[leg].out : dLib[leg - na];
    CUDA(cudaMemcpy(C.data(), src, C.size() * 4, cudaMemcpyDeviceToHost));
    double maxrel = 0, maxabs_ord = 0;
    int mism = 0, over = 0;
    for (auto [i, j] : cells) {
      double ref = 0, mag = 0;
      for (int k = 0; k < K; ++k) {
        double p = (double)A[(size_t)i * K + k] * B[(size_t)k * N + j];
        ref += p;
        mag += std::fabs(p);
      }
      float got = C[(size_t)i * N + j];
      maxrel = std::max(maxrel, std::fabs(got - ref) / std::max(std::fabs(ref), 1e-3));
      if (leg >= na) continue;
      const Arm &a = arms[leg];
      float ordered = fold(i, j, 0, K, 0.0f, a.contract);
      float want = ordered;
      if (a.s > 1) {  // ADR-0040 §2: partitions from 0 (partition 0 from the seed, 0 here), added in order
        int kc = K / a.s;
        want = fold(i, j, 0, kc, 0.0f, a.contract);
        for (unsigned z = 1; z < a.s; ++z) want = want + fold(i, j, z * kc, (z + 1) * kc, 0.0f, a.contract);
      }
      if (memcmp(&got, &want, 4) != 0) ++mism;
      double d = std::fabs((double)got - ordered);
      maxabs_ord = std::max(maxabs_ord, d);
      if (a.reassoc && d > (gamma(K) + gamma(K / a.s + a.s)) * mag) ++over;
    }
    printf("values leg=%s cells=%zu maxrel_vs_f64=%.3e", name(leg).c_str(), cells.size(), maxrel);
    if (leg < na) {
      Arm &a = arms[leg];
      a.ok = !mism && !over;
      all_ok &= a.ok;
      before[leg] = C;
      printf(" S=%u bit_exact_vs_reference=%s maxabs_vs_ordered=%.3e", a.s, mism ? "NO" : "yes", maxabs_ord);
      if (a.reassoc) printf(" within_adr0040_bound=%s", over ? "NO" : "yes");
      printf(" fnv=%016llx%s", (unsigned long long)fnv(C), a.ok ? "" : " GATE-FAILED");
    }
    printf("\n");
  }

  // --- timing: legs round-robin, reversed on odd reps (ABBA)
  cudaEvent_t t0, t1;
  CUDA(cudaEventCreate(&t0));
  CUDA(cudaEventCreate(&t1));
  std::vector<std::vector<float>> ms(na + 2);
  for (int r = 0; r < reps; ++r)
    for (size_t x = 0; x < na + 2; ++x) {
      size_t leg = r & 1 ? na + 1 - x : x;
      if (leg < na && !arms[leg].ok) continue;
      CUDA(cudaMemsetAsync(flush, r & 0xff, flush_bytes));
      CUDA(cudaEventRecord(t0));
      run(leg);
      CUDA(cudaEventRecord(t1));
      CUDA(cudaEventSynchronize(t1));
      float e;
      CUDA(cudaEventElapsedTime(&e, t0, t1));
      ms[leg].push_back(e);
    }
  for (size_t leg = 0; leg < na + 2; ++leg) {
    if (ms[leg].empty()) continue;
    auto v = ms[leg];
    std::sort(v.begin(), v.end());
    printf("time leg=%s M=%d N=%d K=%d n=%zu min_ms=%.4f med_ms=%.4f max_ms=%.4f\n", name(leg).c_str(), M, N, K,
           v.size(), v.front(), v[v.size() / 2], v.back());
  }
  // --- determinism within the process: the whole output again, byte for byte
  for (size_t leg = 0; leg < na; ++leg) {
    if (!arms[leg].ok) continue;
    CUDA(cudaMemset(arms[leg].out, 0xA5, (size_t)M * N * 4));
    run(leg);
    CUDA(cudaDeviceSynchronize());
    CUDA(cudaMemcpy(C.data(), arms[leg].out, C.size() * 4, cudaMemcpyDeviceToHost));
    bool same = !memcmp(C.data(), before[leg].data(), C.size() * 4);
    all_ok &= same;
    printf("determinism leg=%s same_output=%s fnv=%016llx\n", arms[leg].stem.c_str(), same ? "yes" : "NO",
           (unsigned long long)fnv(C));
  }
  for (size_t leg = 0; leg < na + 2; ++leg) {
    printf("samples leg=%s ", name(leg).c_str());
    for (size_t i = 0; i < ms[leg].size(); ++i) printf("%s%.4f", i ? "," : "", ms[leg][i]);
    printf("\n");
  }
  return all_ok ? 0 : 1;
}

int main(int argc, char **argv) {
  if (argc >= 8 && !strcmp(argv[1], "--splitk")) return splitk_main(argc, argv);
  if (argc >= 5 && !strcmp(argv[1], "--program"))
    return program_floor(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]));
  if (argc >= 4 && !strcmp(argv[1], "--h2d"))
    return h2d_probe(atoi(argv[2]), atoi(argv[3]));
  if (argc >= 4 && !strcmp(argv[1], "--decode"))
    return decode_main(atoi(argv[2]), atoi(argv[3]), argc > 4 ? atoi(argv[4]) : 10);
  if (argc < 11) {
    fprintf(stderr, "usage: %s M N K ptx kernel gx gy block smem reps [--contract]\n", argv[0]);
    return 2;
  }
  const int M = atoi(argv[1]), N = atoi(argv[2]), K = atoi(argv[3]);
  const char *ptx_path = argv[4], *kname = argv[5];
  const unsigned gx = atoi(argv[6]), gy = atoi(argv[7]), block = atoi(argv[8]),
                 smem = atoi(argv[9]);
  const int reps = atoi(argv[10]);
  const bool tf32 = argc > 11 && !strcmp(argv[11], "--tf32");
  const bool contract = tf32 || (argc > 11 && !strcmp(argv[11], "--contract"));
  bool warm = false;
  for (int i = 11; i < argc; ++i) warm |= !strcmp(argv[i], "--warm");

  CUDA(cudaFree(0));  // primary context, shared by the runtime and driver legs
  device_facts();

  std::vector<float> A((size_t)M * K), B((size_t)K * N);
  uint64_t seed = 0x5eed;
  for (auto &x : A) x = frand(seed);
  for (auto &x : B) x = frand(seed);

  float *dA, *dB, *dC[3];
  CUDA(cudaMalloc(&dA, A.size() * 4));
  CUDA(cudaMalloc(&dB, B.size() * 4));
  for (auto &c : dC) CUDA(cudaMalloc(&c, (size_t)M * N * 4));
  CUDA(cudaMemcpy(dA, A.data(), A.size() * 4, cudaMemcpyHostToDevice));
  CUDA(cudaMemcpy(dB, B.data(), B.size() * 4, cudaMemcpyHostToDevice));
  const size_t flush_bytes = 256u << 20;
  void *flush;
  CUDA(cudaMalloc(&flush, flush_bytes));

  // --- Mapal: the emitted PTX, launched as emitted.
  FILE *f = fopen(ptx_path, "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", ptx_path); return 102; }
  std::string ptx;
  char buf[1 << 16];
  size_t n;
  while ((n = fread(buf, 1, sizeof buf, f)) > 0) ptx.append(buf, n);
  fclose(f);
  CUmodule mod;
  CUfunction fn;
  CU(cuModuleLoadData(&mod, ptx.c_str()));
  CU(cuModuleGetFunction(&fn, mod, kname));
  if (smem > 48 * 1024)
    CU(cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem));
  int regs = 0, spill = 0;
  cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, fn);
  cuFuncGetAttribute(&spill, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fn);
  printf("mapal kernel=%s grid=%ux%u block=%u smem=%u regs=%d local_bytes=%d\n",
         kname, gx, gy, block, smem, regs, spill);

  cublasHandle_t h;
  cublasCreate(&h);
  cublasSetMathMode(h, tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH);
  CutlassGemm cutlass_gemm;
  CutlassGemm::Arguments cargs({M, N, K}, {dA, K}, {dB, N}, {dC[2], N}, {dC[2], N},
                               {1.0f, 0.0f});
  CutlassTf32 cutlass_tf32;
  CutlassTf32::Arguments targs({M, N, K}, {dA, K}, {dB, N}, {dC[2], N}, {dC[2], N},
                               {1.0f, 0.0f});
  auto ok = tf32 ? cutlass_tf32.can_implement(targs) : cutlass_gemm.can_implement(cargs);
  if (ok != cutlass::Status::kSuccess) {
    fprintf(stderr, "cutlass cannot implement %dx%dx%d\n", M, N, K);
    return 102;
  }
  // The TF32 GEMM's default tile wants more than 48 KB of shared memory.
  auto init = tf32 ? cutlass_tf32.initialize(targs) : cutlass_gemm.initialize(cargs);
  if (init != cutlass::Status::kSuccess) {
    fprintf(stderr, "cutlass initialize failed\n");
    return 102;
  }

  const float one = 1.0f, zero = 0.0f;
  auto run = [&](int leg) {
    if (leg == 0) {
      void *args[] = {&dA, &dB, &dC[0]};
      CU(cuLaunchKernel(fn, gx, gy, 1, block, 1, 1, smem, 0, args, 0));
    } else if (leg == 1) {
      // Row-major C = A·B is column-major Cᵀ = Bᵀ·Aᵀ: the same flops.
      cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, dB, N, dA, K, &zero, dC[1], N);
    } else if (tf32) {
      cutlass_tf32();
    } else {
      cutlass_gemm();
    }
  };
  const char *names[3] = {"mapal", "cublas", "cutlass"};
  for (int leg = 0; leg < 3; ++leg)
    for (int w = 0; w < 3; ++w) run(leg);
  CUDA(cudaDeviceSynchronize());

  // --- values, on sampled cells
  std::vector<float> C((size_t)M * N);
  uint64_t cs = 0xce11;
  std::vector<std::pair<int, int>> cells = {{0, 0}, {M - 1, N - 1}, {0, N - 1}, {M - 1, 0}};
  for (int i = 0; i < 252; ++i) {
    cs = cs * 6364136223846793005ull + 1;
    cells.push_back({(int)((cs >> 33) % M), (int)((cs >> 13) % N)});
  }
  double leg_maxrel[3] = {0, 0, 0};
  for (int leg = 2; leg >= 0; --leg) {  // libraries first: their error bounds Mapal's on --tf32
    CUDA(cudaMemcpy(C.data(), dC[leg], C.size() * 4, cudaMemcpyDeviceToHost));
    double maxrel = 0;
    int mismatches = 0;
    for (auto [i, j] : cells) {
      float acc = 0.0f;
      double ref = 0.0;
      for (int k = 0; k < K; ++k) {
        float a = A[(size_t)i * K + k], b = B[(size_t)k * N + j];
        acc = contract ? fmaf(a, b, acc) : acc + a * b;
        ref += (double)a * b;
      }
      float got = C[(size_t)i * N + j];
      if (leg == 0 && !tf32 && memcmp(&got, &acc, 4) != 0) ++mismatches;
      maxrel = std::max(maxrel, std::fabs(got - ref) / std::max(std::fabs(ref), 1e-3));
    }
    leg_maxrel[leg] = maxrel;
    printf("values leg=%s cells=%zu maxrel_vs_f64=%.3e", names[leg], cells.size(), maxrel);
    if (leg == 0 && !tf32) printf(" bit_exact_vs_in_order_fold=%s", mismatches ? "NO" : "yes");
    if (leg == 0 && tf32) {
      bool within = maxrel <= 4 * leg_maxrel[1];
      printf(" within_4x_cublas_tf32=%s", within ? "yes" : "NO");
      if (!within) mismatches = 1;
    }
    printf("\n");
    if (leg == 0 && mismatches) {
      fprintf(stderr, "MAPAL VALUES WRONG: %d of %zu cells differ from the in-order fold\n",
              mismatches, cells.size());
      return 1;
    }
  }

  // --- timing
  cudaEvent_t t0, t1;
  CUDA(cudaEventCreate(&t0));
  CUDA(cudaEventCreate(&t1));
  std::vector<float> ms[3];
  for (int r = 0; r < reps; ++r)
    for (int leg = 0; leg < 3; ++leg) {
      if (!warm) CUDA(cudaMemsetAsync(flush, r & 0xff, flush_bytes));
      CUDA(cudaEventRecord(t0));
      run(leg);
      CUDA(cudaEventRecord(t1));
      CUDA(cudaEventSynchronize(t1));
      float x;
      CUDA(cudaEventElapsedTime(&x, t0, t1));
      ms[leg].push_back(x);
    }
  const double flops = 2.0 * M * N * K;
  double med[3];
  for (int leg = 0; leg < 3; ++leg) {
    auto v = ms[leg];
    std::sort(v.begin(), v.end());
    med[leg] = v[v.size() / 2];
    printf("time leg=%s M=%d N=%d K=%d n=%d min_ms=%.4f med_ms=%.4f max_ms=%.4f tflops_med=%.2f\n",
           names[leg], M, N, K, reps, v.front(), med[leg], v.back(), flops / med[leg] / 1e9);
  }
  printf("ratio mapal_vs_cublas=%.3f mapal_vs_cutlass=%.3f (>1 = mapal faster, medians)\n",
         med[1] / med[0], med[2] / med[0]);
  printf("samples");
  for (int leg = 0; leg < 3; ++leg) {
    printf(" %s=", names[leg]);
    for (size_t i = 0; i < ms[leg].size(); ++i) printf("%s%.4f", i ? "," : "", ms[leg][i]);
  }
  printf("\n");
  return 0;
}
