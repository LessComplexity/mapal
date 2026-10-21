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
//
// --tf32: the TF32 face. cuBLAS runs CUBLAS_TF32_TENSOR_OP_MATH, CUTLASS its
// OpClassTensorOp GEMM; values are bounded, not bit-exact: Mapal's max error vs
// f64 must stay within 4x cuBLAS-TF32's own.
//
// usage: tile_bench M N K <mapal.ptx> <kernel> gx gy block smem reps [--contract|--tf32]
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

int main(int argc, char **argv) {
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
      CUDA(cudaMemsetAsync(flush, r & 0xff, flush_bytes));
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
