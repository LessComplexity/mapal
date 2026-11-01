// s62 disasm (rtx-4090): the library legs of the residual MLP whole program
// `pt_rm_m<M>_d<d>_h<h>` (gen_tiles.py): y = relu(x·W1 + bias)·W2 + x,
// x M×d, W1 d×h, W2 h×d, row-major, the program's operand laws.
//   eager          cublasSgemm, bias+relu kernel, cublasSgemm, residual kernel (4 launches)
//   lt             cuBLASLt RELU_BIAS epilogue, then cuBLASLt beta=1 with C = x (2 launches)
//   cutlass        stock SIMT CUTLASS (tile_bench's configuration), LinearCombinationRelu with
//                  the bias as C (ldc = 0), then LinearCombination beta=1 with C = x (2 launches)
//   cutlass_tuned  the same with competitive-cutlass.md's selected 16x64x16 / 8x32x16
// FP32 only (CUBLAS_COMPUTE_32F, default math, NVIDIA_TF32_OVERRIDE=0 from the caller).
// Gate before timing: every leg's whole y against a CPU f64 reference,
// |leg - ref| <= 1e-3 * max(|ref|, rms(ref)); the Mapal values (y[0], y[last]) likewise.
// Timing: first call of each leg in this process (host clock around call + sync: the
// cold single iteration Mapal's `iter ms` is), then legs round-robin ABBA with a 256 MiB
// flush before every rep, cudaEvent GPU time.
// PROFILE_LEG=<leg>: after the gates, one call in a cudaProfilerStart/Stop region; exit.
// usage: mlp_bench M d h reps [mapal_y0 mapal_ylast]
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

#define CUDA(x)                                                                         \
  do {                                                                                  \
    cudaError_t e_ = (x);                                                               \
    if (e_ != cudaSuccess) { fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); exit(102); } \
  } while (0)
#define BLAS(x)                                                                         \
  do {                                                                                  \
    cublasStatus_t s_ = (x);                                                            \
    if (s_ != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "%s: status %d\n", #x, (int)s_); exit(102); } \
  } while (0)

using RM = cutlass::layout::RowMajor;
using Lin = cutlass::epilogue::thread::LinearCombination<float, 1, float, float>;
using Relu = cutlass::epilogue::thread::LinearCombinationRelu<float, 1, float, float>;
using Cfg = cutlass::gemm::device::DefaultGemmConfiguration<cutlass::arch::OpClassSimt, cutlass::arch::Sm80, float,
                                                            float, float, float>;
template <class Epi>
using Stock = cutlass::gemm::device::Gemm<float, RM, float, RM, float, RM, float, cutlass::arch::OpClassSimt,
                                          cutlass::arch::Sm80, Cfg::ThreadblockShape, Cfg::WarpShape,
                                          Cfg::InstructionShape, Epi,
                                          cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Cfg::kStages>;
template <class Epi>
using Tuned = cutlass::gemm::device::Gemm<float, RM, float, RM, float, RM, float, cutlass::arch::OpClassSimt,
                                          cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 64, 16>,
                                          cutlass::gemm::GemmShape<8, 32, 16>, cutlass::gemm::GemmShape<1, 1, 1>, Epi,
                                          cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, 2, 1, 1, false>;

__global__ void bias_relu(float *o, const float *s, const float *b, int cnt, int cols) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < cnt) { float v = s[i] + b[i % cols]; o[i] = v > 0.f ? v : 0.f; }
}
__global__ void add(float *o, const float *s, const float *x, int cnt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < cnt) o[i] = s[i] + x[i];
}

static int M, D, H;
static float *x, *w1, *w2, *bias, *s1, *hh, *s2, *y;
static cublasHandle_t hb;
static cublasLtHandle_t lt;
static void *ws;
static const size_t WS = 32u << 20;

// Row-major O(m×n) = X(m×k)·Y(k×n) is column-major Oᵀ = Yᵀ·Xᵀ.
static void sgemm(float *o, const float *a, const float *b, int m, int n, int k) {
  const float one = 1, zero = 0;
  BLAS(cublasSgemm(hb, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &one, b, n, a, k, &zero, o, n));
}

struct LtPlan {
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t la, lb, lc;
  cublasLtMatmulAlgo_t algo;
  int m, n, k;
  LtPlan(int m_, int n_, int k_, cublasLtEpilogue_t epi, const float *b) : m(m_), n(n_), k(k_) {
    BLAS(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    BLAS(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof epi));
    if (b) BLAS(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &b, sizeof b));
    BLAS(cublasLtMatrixLayoutCreate(&la, CUDA_R_32F, n, k, n));  // Yᵀ
    BLAS(cublasLtMatrixLayoutCreate(&lb, CUDA_R_32F, k, m, k));  // Xᵀ
    BLAS(cublasLtMatrixLayoutCreate(&lc, CUDA_R_32F, n, m, n));  // Oᵀ (and C)
    cublasLtMatmulPreference_t pref;
    BLAS(cublasLtMatmulPreferenceCreate(&pref));
    BLAS(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &WS, sizeof WS));
    cublasLtMatmulHeuristicResult_t res;
    int got = 0;
    BLAS(cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 1, &res, &got));
    if (!got) { fprintf(stderr, "no cublasLt algorithm for epilogue %d\n", (int)epi); exit(102); }
    algo = res.algo;
    cublasLtMatmulPreferenceDestroy(pref);
  }
  void run(float *o, const float *a, const float *b, float beta, const float *c) {
    const float one = 1;
    BLAS(cublasLtMatmul(lt, op, &one, b, la, a, lb, &beta, c ? c : o, lc, o, lc, &algo, ws, WS, 0));
  }
};

template <class G>
static void cut(float *o, const float *a, const float *b, int m, int n, int k, const float *c, int ldc, float beta) {
  G g;
  typename G::Arguments args({m, n, k}, {a, k}, {b, n}, {c, ldc}, {o, n}, {1.f, beta});
  if (g(args) != cutlass::Status::kSuccess) { fprintf(stderr, "cutlass failed\n"); exit(102); }
}

static float law(long long t, int mul, int add, int mod, int off, float div) {
  return (float)((int)((t * mul + add) % mod) - off) / div;
}

int main(int argc, char **argv) {
  if (argc < 5) { fprintf(stderr, "usage: %s M d h reps [mapal_y0 mapal_ylast]\n", argv[0]); return 2; }
  M = atoi(argv[1]); D = atoi(argv[2]); H = atoi(argv[3]);
  const int reps = atoi(argv[4]);
  std::vector<float> hx((size_t)M * D), hw1((size_t)D * H), hw2((size_t)H * D), hb_((size_t)H);
  for (size_t t = 0; t < hx.size(); ++t) hx[t] = law(t, 7, 13, 101, 50, 7.f);
  for (size_t t = 0; t < hw1.size(); ++t) hw1[t] = law(t, 11, 57, 103, 51, 9.f);
  for (size_t t = 0; t < hw2.size(); ++t) hw2[t] = law(t, 5, 3, 97, 48, 11.f);
  for (int t = 0; t < H; ++t) hb_[t] = (float)(t % 13 - 6) / 5.f;
  cudaDeviceProp p;
  CUDA(cudaGetDeviceProperties(&p, 0));
  printf("device name=\"%s\" cc=%d.%d\n", p.name, p.major, p.minor);
  float **bufs[] = {&x, &w1, &w2, &bias, &s1, &hh, &s2, &y};
  size_t sizes[] = {hx.size(), hw1.size(), hw2.size(), hb_.size(), (size_t)M * H, (size_t)M * H, (size_t)M * D,
                    (size_t)M * D};
  for (int i = 0; i < 8; ++i) CUDA(cudaMalloc(bufs[i], sizes[i] * 4));
  CUDA(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
  CUDA(cudaMemcpy(w1, hw1.data(), hw1.size() * 4, cudaMemcpyHostToDevice));
  CUDA(cudaMemcpy(w2, hw2.data(), hw2.size() * 4, cudaMemcpyHostToDevice));
  CUDA(cudaMemcpy(bias, hb_.data(), hb_.size() * 4, cudaMemcpyHostToDevice));
  BLAS(cublasCreate(&hb));
  BLAS(cublasSetMathMode(hb, CUBLAS_DEFAULT_MATH));
  BLAS(cublasLtCreate(&lt));
  CUDA(cudaMalloc(&ws, WS));
  void *flush;
  const size_t FL = 256u << 20;
  CUDA(cudaMalloc(&flush, FL));
  LtPlan l1(M, H, D, CUBLASLT_EPILOGUE_RELU_BIAS, bias), l2(M, D, H, CUBLASLT_EPILOGUE_DEFAULT, nullptr);
  const int cnt1 = M * H, cnt2 = M * D;
  struct Leg { const char *name; int launches; std::function<void()> run; };
  std::vector<Leg> legs = {
      {"eager", 4, [&] {
         sgemm(s1, x, w1, M, H, D);
         bias_relu<<<(cnt1 + 255) / 256, 256>>>(hh, s1, bias, cnt1, H);
         sgemm(s2, hh, w2, M, D, H);
         add<<<(cnt2 + 255) / 256, 256>>>(y, s2, x, cnt2);
       }},
      {"lt", 2, [&] { l1.run(hh, x, w1, 0.f, nullptr); l2.run(y, hh, w2, 1.f, x); }},
      {"cutlass", 2, [&] {
         cut<Stock<Relu>>(hh, x, w1, M, H, D, bias, 0, 1.f);
         cut<Stock<Lin>>(y, hh, w2, M, D, H, x, D, 1.f);
       }},
#ifndef NO_TUNED  // box_disasm.sh rebuilds with -DNO_TUNED if this instantiation fails
      {"cutlass_tuned", 2, [&] {
         cut<Tuned<Relu>>(hh, x, w1, M, H, D, bias, 0, 1.f);
         cut<Tuned<Lin>>(y, hh, w2, M, D, H, x, D, 1.f);
       }},
#endif
  };
  // The cold single iteration: each leg's first call in this process (lazy module load included).
  for (auto &L : legs) {
    CUDA(cudaDeviceSynchronize());
    auto t0 = std::chrono::steady_clock::now();
    L.run();
    CUDA(cudaDeviceSynchronize());
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    printf("first_call leg=%s M=%d d=%d h=%d launches=%d wall_ms=%.4f\n", L.name, M, D, H, L.launches, ms);
  }
  // CPU f64 reference.
  std::vector<double> ref((size_t)M * D), hrow(H);
  double rms = 0;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < H; ++j) {
      double a = 0;
      for (int k = 0; k < D; ++k) a += (double)hx[(size_t)i * D + k] * hw1[(size_t)k * H + j];
      a += hb_[j];
      hrow[j] = a > 0 ? a : 0;
    }
    for (int j = 0; j < D; ++j) {
      double a = 0;
      for (int k = 0; k < H; ++k) a += hrow[k] * hw2[(size_t)k * D + j];
      ref[(size_t)i * D + j] = a + hx[(size_t)i * D + j];
      rms += ref[(size_t)i * D + j] * ref[(size_t)i * D + j];
    }
  }
  rms = std::sqrt(rms / ref.size());
  auto tol = [&](double r) { return 1e-3 * std::max(std::fabs(r), rms); };
  bool ok = true;
  if (argc >= 7) {
    double m0 = atof(argv[5]), m1 = atof(argv[6]);
    bool g = std::fabs(m0 - ref[0]) <= tol(ref[0]) && std::fabs(m1 - ref.back()) <= tol(ref.back());
    printf("values leg=mapal y0=%.7g ylast=%.7g ref0=%.7g reflast=%.7g gate=%s\n", m0, m1, ref[0], ref.back(),
           g ? "pass" : "FAIL");
    ok &= g;
  }
  std::vector<float> got(ref.size());
  for (auto &L : legs) {
    CUDA(cudaMemset(y, 0, got.size() * 4));
    L.run();
    CUDA(cudaDeviceSynchronize());
    CUDA(cudaMemcpy(got.data(), y, got.size() * 4, cudaMemcpyDeviceToHost));
    double worst = 0;
    size_t bad = 0;
    for (size_t i = 0; i < got.size(); ++i) {
      double d = std::fabs(got[i] - ref[i]);
      worst = std::max(worst, d / std::max(std::fabs(ref[i]), rms));
      bad += d > tol(ref[i]);
    }
    printf("values leg=%s y0=%.7g ylast=%.7g max_scaled_err=%.3e bad=%zu gate=%s\n", L.name, got[0], got.back(),
           worst, bad, bad ? "FAIL" : "pass");
    ok &= !bad;
  }
  if (const char *pl = getenv("PROFILE_LEG")) {
    for (auto &L : legs)
      if (!strcmp(L.name, pl)) {
        CUDA(cudaDeviceSynchronize());
        CUDA(cudaProfilerStart());
        L.run();
        CUDA(cudaDeviceSynchronize());
        CUDA(cudaProfilerStop());
        printf("profile_leg leg=%s calls=1 gates=%s\n", pl, ok ? "pass" : "FAIL");
        return ok ? 0 : 1;
      }
    return 2;
  }
  if (!ok) { printf("GATE-FAILED: no timing\n"); return 1; }
  for (auto &L : legs) for (int w = 0; w < 3; ++w) L.run();
  CUDA(cudaDeviceSynchronize());
  cudaEvent_t e0, e1;
  CUDA(cudaEventCreate(&e0));
  CUDA(cudaEventCreate(&e1));
  std::vector<std::vector<float>> ms(legs.size());
  for (int r = 0; r < reps; ++r)
    for (size_t q = 0; q < legs.size(); ++q) {
      size_t L = r & 1 ? legs.size() - 1 - q : q;
      CUDA(cudaMemsetAsync(flush, r & 0xff, FL));
      CUDA(cudaEventRecord(e0));
      legs[L].run();
      CUDA(cudaEventRecord(e1));
      CUDA(cudaEventSynchronize(e1));
      float t;
      CUDA(cudaEventElapsedTime(&t, e0, e1));
      ms[L].push_back(t);
    }
  for (size_t L = 0; L < legs.size(); ++L) {
    auto v = ms[L];
    std::sort(v.begin(), v.end());
    printf("time leg=%s M=%d d=%d h=%d launches=%d n=%zu min_ms=%.4f med_ms=%.4f max_ms=%.4f\n", legs[L].name, M, D,
           H, legs[L].launches, v.size(), v.front(), v[v.size() / 2], v.back());
    printf("samples leg=%s ", legs[L].name);
    for (size_t i = 0; i < ms[L].size(); ++i) printf("%s%.4f", i ? "," : "", ms[L][i]);
    printf("\n");
  }
  return 0;
}
