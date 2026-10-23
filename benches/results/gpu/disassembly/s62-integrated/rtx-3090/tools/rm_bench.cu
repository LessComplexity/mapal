// Library legs of Mapal's residual MLP `pt_rm_m<M>_d<d>_h<h>` (gen_tiles.py):
//   hh = relu(x·w1 + bias),  y = hh·w2 + x
// with the program's operand laws, generated on the device before any clock.
// Legs (pure FP32: CUBLAS_COMPUTE_32F, default math, no TF32):
//   cublas        sgemm, bias+relu kernel, D2D copy of x into y, sgemm beta=1
//   lt            cuBLASLt RELU_BIAS epilogue, then C=x beta=1 (two calls)
//   cutlass       stock SIMT f32 GEMM (tile_bench's), LinearCombinationRelu with the
//                 bias as C (ldc 0), then LinearCombination C=x beta=1
//   cutlass_tuned the 16x64x16 / warp 8x32x16 / 2-stage SIMT configuration
//                 (docs/performance/gpu/kernels/competitive-cutlass.md), same epilogues
// Values: every leg's whole y against an fp64 host reference within
// gamma(d+h)-scaled componentwise bounds, and y[0], y[last] against Mapal's
// printed values (argv) within 1e-3 relative (a summation-order bound).
// Timing per leg, legs round-robin ABBA:
//   first_ms   host wall of the first call in this process (lazy kernel loading
//              and library first-use included), launch to synchronize
//   event_ms   CUDA events around one call, L2 flushed (256 MiB memset) first
//   wall_ms    host wall, launch to synchronize, same reps (Mapal's `iter ms` form)
// TB_PROFILE_LEG=<leg>: no timing; cudaProfilerStart, one call, synchronize,
// cudaProfilerStop (the counter-free observer's selected region).
// usage: rm_bench M d h reps [mapal_y0 mapal_ylast]
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/gemm/device/gemm.h"

#define CUDA(x)                                                                                 \
  do {                                                                                          \
    cudaError_t e_ = (x);                                                                       \
    if (e_ != cudaSuccess) {                                                                    \
      fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_));                                  \
      exit(102);                                                                                \
    }                                                                                           \
  } while (0)
#define BLAS(x)                                                                                 \
  do {                                                                                          \
    cublasStatus_t s_ = (x);                                                                    \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                                          \
      fprintf(stderr, "%s: cublas status %d\n", #x, (int)s_);                                   \
      exit(102);                                                                                \
    }                                                                                           \
  } while (0)

using RowMajor = cutlass::layout::RowMajor;
using Cfg = cutlass::gemm::device::DefaultGemmConfiguration<cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
                                                            float, float, float, float>;
template <class Epi>
using Stock = cutlass::gemm::device::Gemm<float, RowMajor, float, RowMajor, float, RowMajor, float,
                                          cutlass::arch::OpClassSimt, cutlass::arch::Sm80, Cfg::ThreadblockShape,
                                          Cfg::WarpShape, Cfg::InstructionShape, Epi,
                                          cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
                                          Cfg::kStages>;
template <class Epi>
using Tuned = cutlass::gemm::device::Gemm<float, RowMajor, float, RowMajor, float, RowMajor, float,
                                          cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
                                          cutlass::gemm::GemmShape<16, 64, 16>, cutlass::gemm::GemmShape<8, 32, 16>,
                                          cutlass::gemm::GemmShape<1, 1, 1>, Epi,
                                          cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, 2, 1, 1,
                                          false>;
using Lin = cutlass::epilogue::thread::LinearCombination<float, 1, float, float>;
using Relu = cutlass::epilogue::thread::LinearCombinationRelu<float, 1, float, float>;

__global__ void gen_op(float *o, long long cnt, int mul, int add, int mod, int off, float div) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < cnt) o[i] = (float)(((int)i * mul + add) % mod - off) / div;
}
__global__ void bias_relu(float *o, const float *x, const float *b, long long cnt, int cols) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cnt) return;
  float v = x[i] + b[i % cols];
  o[i] = v > 0.f ? v : 0.f;
}

static int M, D, H;
static float *X, *W1, *W2, *BIAS, *S1, *HH, *Y;
static cublasHandle_t hb;
static cublasLtHandle_t lt;
static void *ws;
static const size_t WS = 32u << 20;

template <class G>
static void cut(int m, int n, int k, const float *a, const float *b, const float *c, int ldc, float *o, float beta) {
  G g;
  typename G::Arguments args({m, n, k}, {a, k}, {b, n}, {c, ldc}, {o, n}, {1.f, beta});
  if (g(args, nullptr, 0) != cutlass::Status::kSuccess) {
    fprintf(stderr, "cutlass failed %dx%dx%d\n", m, n, k);
    exit(102);
  }
}

// Row-major O[m,n] = A[m,k]·B[k,n] is column-major Oᵀ = Bᵀ·Aᵀ.
struct LtGemm {
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t la, lb, lc;
  cublasLtMatmulAlgo_t algo;
  LtGemm(int m, int n, int k, cublasLtEpilogue_t epi, const float *bias) {
    BLAS(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    BLAS(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof epi));
    if (bias) BLAS(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof bias));
    BLAS(cublasLtMatrixLayoutCreate(&la, CUDA_R_32F, n, k, n));  // Bᵀ: n x k, ld n
    BLAS(cublasLtMatrixLayoutCreate(&lb, CUDA_R_32F, k, m, k));  // Aᵀ: k x m, ld k
    BLAS(cublasLtMatrixLayoutCreate(&lc, CUDA_R_32F, n, m, n));  // Oᵀ: n x m, ld n
    cublasLtMatmulPreference_t pref;
    BLAS(cublasLtMatmulPreferenceCreate(&pref));
    BLAS(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &WS, sizeof WS));
    cublasLtMatmulHeuristicResult_t res;
    int got = 0;
    BLAS(cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 1, &res, &got));
    if (!got) {
      fprintf(stderr, "no cublasLt algorithm\n");
      exit(102);
    }
    algo = res.algo;
    cublasLtMatmulPreferenceDestroy(pref);
  }
  void run(const float *a, const float *b, const float *c, float *o, float beta) {
    const float one = 1.f;
    BLAS(cublasLtMatmul(lt, op, &one, b, la, a, lb, &beta, c, lc, o, lc, &algo, ws, WS, 0));
  }
};

static double wall_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: rm_bench M d h reps [mapal_y0 mapal_ylast]\n");
    return 2;
  }
  M = atoi(argv[1]), D = atoi(argv[2]), H = atoi(argv[3]);
  const int reps = atoi(argv[4]);
  const char *only = getenv("TB_PROFILE_LEG");
  CUDA(cudaFree(0));
  cudaDeviceProp p;
  CUDA(cudaGetDeviceProperties(&p, 0));
  printf("device name=\"%s\" cc=%d.%d sms=%d\n", p.name, p.major, p.minor, p.multiProcessorCount);
  const long long nx = (long long)M * D, nw = (long long)D * H, nh = (long long)M * H;
  CUDA(cudaMalloc(&X, nx * 4));
  CUDA(cudaMalloc(&W1, nw * 4));
  CUDA(cudaMalloc(&W2, nw * 4));
  CUDA(cudaMalloc(&BIAS, H * 4));
  CUDA(cudaMalloc(&S1, nh * 4));
  CUDA(cudaMalloc(&HH, nh * 4));
  CUDA(cudaMalloc(&Y, nx * 4));
  gen_op<<<(unsigned)((nx + 255) / 256), 256>>>(X, nx, 7, 13, 101, 50, 7.f);
  gen_op<<<(unsigned)((nw + 255) / 256), 256>>>(W1, nw, 11, 57, 103, 51, 9.f);
  gen_op<<<(unsigned)((nw + 255) / 256), 256>>>(W2, nw, 5, 3, 97, 48, 11.f);
  gen_op<<<(H + 255) / 256, 256>>>(BIAS, H, 1, 0, 13, 6, 5.f);
  CUDA(cudaDeviceSynchronize());
  BLAS(cublasCreate(&hb));
  BLAS(cublasSetMathMode(hb, CUBLAS_DEFAULT_MATH));
  CUDA(cudaMalloc(&ws, WS));
  BLAS(cublasSetWorkspace(hb, ws, WS));
  BLAS(cublasLtCreate(&lt));
  LtGemm up(M, H, D, CUBLASLT_EPILOGUE_RELU_BIAS, BIAS), down(M, D, H, CUBLASLT_EPILOGUE_DEFAULT, nullptr);

  struct Leg {
    const char *name;
    std::function<void()> run;
    int launches;
  };
  const float one = 1.f, zero = 0.f;
  std::vector<Leg> legs = {
      {"cublas",
       [&] {
         BLAS(cublasSgemm(hb, CUBLAS_OP_N, CUBLAS_OP_N, H, M, D, &one, W1, H, X, D, &zero, S1, H));
         bias_relu<<<(unsigned)((nh + 255) / 256), 256>>>(HH, S1, BIAS, nh, H);
         CUDA(cudaMemcpyAsync(Y, X, nx * 4, cudaMemcpyDeviceToDevice, 0));
         BLAS(cublasSgemm(hb, CUBLAS_OP_N, CUBLAS_OP_N, D, M, H, &one, W2, D, HH, H, &one, Y, D));
       },
       3},
      {"lt",
       [&] {
         up.run(X, W1, HH, HH, 0.f);
         down.run(HH, W2, X, Y, 1.f);
       },
       2},
      {"cutlass",
       [&] {
         cut<Stock<Relu>>(M, H, D, X, W1, BIAS, 0, HH, 1.f);
         cut<Stock<Lin>>(M, D, H, HH, W2, X, D, Y, 1.f);
       },
       2},
      {"cutlass_tuned",
       [&] {
         cut<Tuned<Relu>>(M, H, D, X, W1, BIAS, 0, HH, 1.f);
         cut<Tuned<Lin>>(M, D, H, HH, W2, X, D, Y, 1.f);
       },
       2},
  };

  if (only) {
    for (auto &L : legs)
      if (!strcmp(L.name, only)) {
        CUDA(cudaProfilerStart());
        L.run();
        CUDA(cudaDeviceSynchronize());
        CUDA(cudaProfilerStop());
        float y0;
        CUDA(cudaMemcpy(&y0, Y, 4, cudaMemcpyDeviceToHost));
        printf("profiled leg=%s y0=%.9g\n", only, y0);
        return 0;
      }
    fprintf(stderr, "unknown leg %s\n", only);
    return 2;
  }

  // fp64 host reference of y, and the componentwise magnitude for the bound.
  std::vector<float> hx(nx), hw1(nw), hw2(nw), hbias(H);
  CUDA(cudaMemcpy(hx.data(), X, nx * 4, cudaMemcpyDeviceToHost));
  CUDA(cudaMemcpy(hw1.data(), W1, nw * 4, cudaMemcpyDeviceToHost));
  CUDA(cudaMemcpy(hw2.data(), W2, nw * 4, cudaMemcpyDeviceToHost));
  CUDA(cudaMemcpy(hbias.data(), BIAS, H * 4, cudaMemcpyDeviceToHost));
  std::vector<double> hh(nh), hmag(nh), ref(nx), mag(nx);
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < H; ++j) {
      double s = 0, m = 0;
      for (int k = 0; k < D; ++k) {
        double t = (double)hx[(size_t)i * D + k] * hw1[(size_t)k * H + j];
        s += t, m += std::fabs(t);
      }
      s += hbias[j], m += std::fabs(hbias[j]);
      hh[(size_t)i * H + j] = s > 0 ? s : 0, hmag[(size_t)i * H + j] = m;
    }
  const double u = std::ldexp(1.0, -24);
  const double g1 = (D + 2) * u / (1 - (D + 2) * u), g2 = (H + 2) * u / (1 - (H + 2) * u);
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < D; ++j) {
      double s = 0, m = 0;
      for (int k = 0; k < H; ++k) {
        s += hh[(size_t)i * H + k] * hw2[(size_t)k * D + j];
        // |hh| <= hmag and relu is 1-Lipschitz: (g1 + g2)·Σ hmag·|w2| covers the
        // first product's error carried through w2 plus this sum's own.
        m += hmag[(size_t)i * H + k] * std::fabs(hw2[(size_t)k * D + j]);
      }
      ref[(size_t)i * D + j] = s + hx[(size_t)i * D + j];
      mag[(size_t)i * D + j] = m + std::fabs(hx[(size_t)i * D + j]);
    }
  double want[2] = {0, 0};
  const bool mapal = argc >= 7;
  if (mapal) want[0] = strtod(argv[5], nullptr), want[1] = strtod(argv[6], nullptr);

  std::vector<float> hy(nx);
  int bad = 0;
  std::vector<double> first(legs.size());
  for (size_t L = 0; L < legs.size(); ++L) {
    CUDA(cudaMemset(Y, 0xff, nx * 4));
    CUDA(cudaMemset(HH, 0xff, nh * 4));
    CUDA(cudaDeviceSynchronize());
    double t0 = wall_ms();
    legs[L].run();
    CUDA(cudaDeviceSynchronize());
    first[L] = wall_ms() - t0;
    CUDA(cudaMemcpy(hy.data(), Y, nx * 4, cudaMemcpyDeviceToHost));
    double worst = 0;  // max |got - ref| / ((g1 + g2)·mag)
    int over = 0;
    for (long long i = 0; i < nx; ++i) {
      double b = (g1 + g2) * mag[i] + 1e-30, d = std::fabs((double)hy[i] - ref[i]);
      if (!(d <= b)) ++over;
      worst = std::max(worst, d / b);
    }
    double vm = 0;
    if (mapal)
      for (int c = 0; c < 2; ++c) {
        double g = hy[c ? nx - 1 : 0];
        vm = std::max(vm, std::fabs(g - want[c]) / std::max(std::fabs(want[c]), 1.0));
      }
    const bool ok = !over && (!mapal || vm <= 1e-3);
    bad += !ok;
    printf("rm_values M=%d d=%d h=%d leg=%s y0=%.9g ylast=%.9g over_bound=%d worst_bound_ratio=%.3e vs_mapal=%.3e "
           "gate=%s\n",
           M, D, H, legs[L].name, hy[0], hy[nx - 1], over, worst, vm, ok ? "pass" : "FAIL");
  }
  if (bad) return 1;

  void *flush;
  const size_t flush_bytes = 256u << 20;
  CUDA(cudaMalloc(&flush, flush_bytes));
  cudaEvent_t e0, e1;
  CUDA(cudaEventCreate(&e0));
  CUDA(cudaEventCreate(&e1));
  for (auto &L : legs)
    for (int w = 0; w < 3; ++w) L.run();
  CUDA(cudaDeviceSynchronize());
  std::vector<std::vector<float>> ev(legs.size()), wl(legs.size());
  for (int r = 0; r < reps; ++r)
    for (size_t x = 0; x < legs.size(); ++x) {
      size_t L = r & 1 ? legs.size() - 1 - x : x;
      CUDA(cudaMemsetAsync(flush, r & 0xff, flush_bytes));
      CUDA(cudaDeviceSynchronize());
      double t0 = wall_ms();
      CUDA(cudaEventRecord(e0));
      legs[L].run();
      CUDA(cudaEventRecord(e1));
      CUDA(cudaEventSynchronize(e1));
      double t1 = wall_ms();
      float ms;
      CUDA(cudaEventElapsedTime(&ms, e0, e1));
      ev[L].push_back(ms);
      wl[L].push_back((float)(t1 - t0));
    }
  for (size_t L = 0; L < legs.size(); ++L) {
    auto a = ev[L], b = wl[L];
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    printf("rm_time M=%d d=%d h=%d leg=%s launches=%d reps=%d first_ms=%.4f event_med_ms=%.4f event_min_ms=%.4f "
           "wall_med_ms=%.4f wall_min_ms=%.4f\n",
           M, D, H, legs[L].name, legs[L].launches, reps, first[L], a[a.size() / 2], a.front(), b[b.size() / 2],
           b.front());
    printf("rm_samples M=%d d=%d h=%d leg=%s event_ms=", M, D, H, legs[L].name);
    for (size_t i = 0; i < ev[L].size(); ++i) printf("%s%.4f", i ? "," : "", ev[L][i]);
    printf(" wall_ms=");
    for (size_t i = 0; i < wl[L].size(); ++i) printf("%s%.4f", i ? "," : "", wl[L][i]);
    printf("\n");
  }
  return 0;
}
