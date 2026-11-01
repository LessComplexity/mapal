// disasm-h100-pcie: the library legs of the S62 residual MLP chain `pt_rm_m<M>_d<d>_h<h>`
// (gen_tiles.py): y = relu(x·W1 + bias)·W2 + x; x M×d, W1 d×h, W2 h×d, bias h; row-major,
// pure FP32 (CUBLAS_DEFAULT_MATH, CUBLAS_COMPUTE_32F; run with NVIDIA_TF32_OVERRIDE=0).
// Operands are the program's laws, generated on the device before the clock (as Mapal's are).
//   cublas         sgemm, bias+relu kernel, sgemm, residual-add kernel (4 launches)
//   lt             cuBLASLt RELU_BIAS epilogue, then cuBLASLt with beta=1, C=x (fused epilogues)
//   cutlass        stock SIMT GEMM (tile_bench's) with LinearCombinationRelu (bias = C, ldc 0)
//                  then LinearCombination with beta=1, C=x
//   cutlass_tuned  the same epilogues on the 16x64x16 / 8x32x16 configuration
// Timing: legs round-robin, order reversed on odd reps (ABBA), no L2 flush (the program's
// operands are generated right before its region, so they sit in L2 there too). Per rep:
// CUDA-event ms and host wall ms (issue .. cudaStreamSynchronize), the latter is what the
// program's own `iter ms` brackets. `first_wall_ms`: each leg's first call in the process.
// Values: every leg's whole y against an f64 CPU reference, |got-ref| <= 1e-3·rms(ref), and
// Mapal's printed y[0], y[last] (argv) against the same reference.
// usage: rm_bench M d h reps [mapal_y0 mapal_ylast]     TB_REGION=1: each leg once inside
// cudaProfilerStart/Stop after timing (the counter-free observer's selection).
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

#define CUDA(x)                                                                \
  do {                                                                         \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess) {                                                   \
      fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_));                 \
      exit(102);                                                               \
    }                                                                          \
  } while (0)
#define BLAS(x)                                                                \
  do {                                                                         \
    cublasStatus_t s_ = (x);                                                   \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                         \
      fprintf(stderr, "%s: cublas status %d\n", #x, (int)s_);                  \
      exit(102);                                                               \
    }                                                                          \
  } while (0)

using Row = cutlass::layout::RowMajor;
using Lin = cutlass::epilogue::thread::LinearCombination<float, 1, float, float>;
using Relu = cutlass::epilogue::thread::LinearCombinationRelu<float, 1, float, float>;
using Cfg = cutlass::gemm::device::DefaultGemmConfiguration<cutlass::arch::OpClassSimt, cutlass::arch::Sm80, float,
                                                            float, float, float>;
template <class Epi>
using Stock = cutlass::gemm::device::Gemm<float, Row, float, Row, float, Row, float, cutlass::arch::OpClassSimt,
                                          cutlass::arch::Sm80, Cfg::ThreadblockShape, Cfg::WarpShape,
                                          Cfg::InstructionShape, Epi,
                                          cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Cfg::kStages>;
template <class Epi>
using Tuned = cutlass::gemm::device::Gemm<float, Row, float, Row, float, Row, float, cutlass::arch::OpClassSimt,
                                          cutlass::arch::Sm80, cutlass::gemm::GemmShape<16, 64, 16>,
                                          cutlass::gemm::GemmShape<8, 32, 16>, cutlass::gemm::GemmShape<1, 1, 1>, Epi,
                                          cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, 2, 1, 1, false>;

__global__ void gen_op(float *o, long long cnt, int mul, int add, int mod, int off, float div) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < cnt) o[i] = (float)(((int)i * mul + add) % mod - off) / div;
}
__global__ void bias_relu(float *o, const float *s, const float *b, long long cnt, int cols) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < cnt) { float v = s[i] + b[i % cols]; o[i] = v > 0.f ? v : 0.f; }
}
__global__ void res_add(float *o, const float *s, const float *x, long long cnt) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < cnt) o[i] = s[i] + x[i];
}

static cublasLtHandle_t lt;
static void *ws;
static const size_t WS = 32u << 20;

// Row-major O(M×N) = X(M×K)·Y(K×N) [+ beta·C] is column-major Oᵀ(N×M) = Yᵀ·Xᵀ: A = Y, B = X.
struct LtPlan {
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t la, lb, lc;
  cublasLtMatmulAlgo_t algo;
  LtPlan(int M, int N, int K, cublasLtEpilogue_t epi, const float *bias) {
    BLAS(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    BLAS(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof epi));
    if (bias) BLAS(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof bias));
    BLAS(cublasLtMatrixLayoutCreate(&la, CUDA_R_32F, N, K, N));
    BLAS(cublasLtMatrixLayoutCreate(&lb, CUDA_R_32F, K, M, K));
    BLAS(cublasLtMatrixLayoutCreate(&lc, CUDA_R_32F, N, M, N));
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
  void run(float *o, const float *x, const float *y, float beta, const float *c) {
    const float one = 1.f;
    BLAS(cublasLtMatmul(lt, op, &one, y, la, x, lb, &beta, c ? c : o, lc, o, lc, &algo, ws, WS, 0));
  }
};

template <class G>
static void cut(int M, int N, int K, float *o, const float *x, const float *y, float beta, const float *c, int ldc) {
  G g;
  typename G::Arguments args({M, N, K}, {x, K}, {y, N}, {c, ldc}, {o, N}, {1.f, beta});
  if (g(args, nullptr, 0) != cutlass::Status::kSuccess) { fprintf(stderr, "cutlass failed\n"); exit(102); }
}
template <class G>
static void facts(const char *leg, int M, int N, int K) {
  cudaFuncAttributes a{};
  CUDA(cudaFuncGetAttributes(&a, cutlass::Kernel<typename G::GemmKernel>));
  typename G::ThreadblockSwizzle sw;
  auto t = sw.get_tiled_shape({M, N, K}, {G::ThreadblockShape::kM, G::ThreadblockShape::kN, G::ThreadblockShape::kK}, 1);
  dim3 g = sw.get_grid_shape(t);
  printf("libarm leg=%s M=%d N=%d K=%d tb=%d,%d,%d threads=%d grid=%u,%u,%u dyn_smem=%zu regs=%d local_bytes=%zu static_smem=%zu\n",
         leg, M, N, K, G::ThreadblockShape::kM, G::ThreadblockShape::kN, G::ThreadblockShape::kK,
         G::GemmKernel::kThreadCount, g.x, g.y, g.z, sizeof(typename G::GemmKernel::SharedStorage), a.numRegs,
         a.localSizeBytes, a.sharedSizeBytes);
}

static double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
  if (argc < 5) { fprintf(stderr, "usage: %s M d h reps [mapal_y0 mapal_ylast]\n", argv[0]); return 2; }
  const int M = atoi(argv[1]), d = atoi(argv[2]), h = atoi(argv[3]), reps = atoi(argv[4]);
  CUDA(cudaFree(0));
  cudaDeviceProp prop;
  CUDA(cudaGetDeviceProperties(&prop, 0));
  printf("device name=\"%s\" cc=%d.%d sms=%d\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
  const long long nx = (long long)M * d, nw = (long long)d * h, nh = (long long)M * h;
  float *x, *w1, *w2, *bias, *s1, *hh, *s2, *y;
  CUDA(cudaMalloc(&x, nx * 4)); CUDA(cudaMalloc(&w1, nw * 4)); CUDA(cudaMalloc(&w2, nw * 4));
  CUDA(cudaMalloc(&bias, h * 4)); CUDA(cudaMalloc(&s1, nh * 4)); CUDA(cudaMalloc(&hh, nh * 4));
  CUDA(cudaMalloc(&s2, nx * 4)); CUDA(cudaMalloc(&y, nx * 4));
  gen_op<<<(unsigned)((nx + 255) / 256), 256>>>(x, nx, 7, 13, 101, 50, 7.f);
  gen_op<<<(unsigned)((nw + 255) / 256), 256>>>(w1, nw, 11, 57, 103, 51, 9.f);
  gen_op<<<(unsigned)((nw + 255) / 256), 256>>>(w2, nw, 5, 3, 97, 48, 11.f);
  gen_op<<<(h + 255) / 256, 256>>>(bias, h, 1, 0, 13, 6, 5.f);
  CUDA(cudaDeviceSynchronize());

  cublasHandle_t hb;
  BLAS(cublasCreate(&hb));
  BLAS(cublasSetMathMode(hb, CUBLAS_DEFAULT_MATH));
  BLAS(cublasLtCreate(&lt));
  CUDA(cudaMalloc(&ws, WS));
  BLAS(cublasSetWorkspace(hb, ws, WS));
  LtPlan up(M, h, d, CUBLASLT_EPILOGUE_RELU_BIAS, bias), down(M, d, h, CUBLASLT_EPILOGUE_DEFAULT, nullptr);
  facts<Stock<Relu>>("cutlass.up", M, h, d);
  facts<Stock<Lin>>("cutlass.down", M, d, h);
  facts<Tuned<Relu>>("cutlass_tuned.up", M, h, d);
  facts<Tuned<Lin>>("cutlass_tuned.down", M, d, h);
  const float one = 1.f, zero = 0.f;
  struct Leg { const char *name; int launches; std::function<void()> run; };
  std::vector<Leg> legs = {
      {"cublas", 4, [&] {
         BLAS(cublasSgemm(hb, CUBLAS_OP_N, CUBLAS_OP_N, h, M, d, &one, w1, h, x, d, &zero, s1, h));
         bias_relu<<<(unsigned)((nh + 255) / 256), 256>>>(hh, s1, bias, nh, h);
         BLAS(cublasSgemm(hb, CUBLAS_OP_N, CUBLAS_OP_N, d, M, h, &one, w2, d, hh, h, &zero, s2, d));
         res_add<<<(unsigned)((nx + 255) / 256), 256>>>(y, s2, x, nx);
       }},
      {"lt", 2, [&] { up.run(hh, x, w1, 0.f, nullptr); down.run(y, hh, w2, 1.f, x); }},
      {"cutlass", 2, [&] {
         cut<Stock<Relu>>(M, h, d, hh, x, w1, 1.f, bias, 0);
         cut<Stock<Lin>>(M, d, h, y, hh, w2, 1.f, x, d);
       }},
      {"cutlass_tuned", 2, [&] {
         cut<Tuned<Relu>>(M, h, d, hh, x, w1, 1.f, bias, 0);
         cut<Tuned<Lin>>(M, d, h, y, hh, w2, 1.f, x, d);
       }},
  };

  // f64 reference of the whole y from the device's own operands.
  std::vector<float> hx(nx), hw1(nw), hw2(nw), hbias(h);
  CUDA(cudaMemcpy(hx.data(), x, nx * 4, cudaMemcpyDeviceToHost));
  CUDA(cudaMemcpy(hw1.data(), w1, nw * 4, cudaMemcpyDeviceToHost));
  CUDA(cudaMemcpy(hw2.data(), w2, nw * 4, cudaMemcpyDeviceToHost));
  CUDA(cudaMemcpy(hbias.data(), bias, h * 4, cudaMemcpyDeviceToHost));
  std::vector<double> rh(nh), ry(nx);
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < h; ++j) {
      double s = 0;
      for (int k = 0; k < d; ++k) s += (double)hx[(size_t)i * d + k] * hw1[(size_t)k * h + j];
      s += hbias[j];
      rh[(size_t)i * h + j] = s > 0 ? s : 0;
    }
  double ss = 0;
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < d; ++j) {
      double s = 0;
      for (int k = 0; k < h; ++k) s += rh[(size_t)i * h + k] * hw2[(size_t)k * d + j];
      ry[(size_t)i * d + j] = s + hx[(size_t)i * d + j];
      ss += ry[(size_t)i * d + j] * ry[(size_t)i * d + j];
    }
  const double rms = std::sqrt(ss / nx);
  int bad = 0;
  if (argc >= 7) {
    double m0 = strtod(argv[5], nullptr), m1 = strtod(argv[6], nullptr);
    double dv = std::max(std::fabs(m0 - ry[0]), std::fabs(m1 - ry[nx - 1])) / rms;
    bad += !(dv <= 1e-3);
    printf("rm_values leg=mapal y0=%.9g ylast=%.9g ref=%.9g,%.9g vs_ref_rel_rms=%.3e gate=%s\n", m0, m1, ry[0],
           ry[nx - 1], dv, dv <= 1e-3 ? "pass" : "FAIL");
  }
  std::vector<double> first(legs.size());
  std::vector<float> got(nx);
  for (size_t L = 0; L < legs.size(); ++L) {
    CUDA(cudaMemset(y, 0, nx * 4));
    CUDA(cudaDeviceSynchronize());
    double t0 = now_ms();
    legs[L].run();
    CUDA(cudaStreamSynchronize(0));
    first[L] = now_ms() - t0;
    for (int w = 0; w < 5; ++w) legs[L].run();
    CUDA(cudaDeviceSynchronize());
    CUDA(cudaMemcpy(got.data(), y, nx * 4, cudaMemcpyDeviceToHost));
    double dv = 0;
    bool fin = true;
    for (long long i = 0; i < nx; ++i) {
      fin &= std::isfinite(got[i]);
      dv = std::max(dv, std::fabs(got[i] - ry[i]) / rms);
    }
    bool ok = fin && dv <= 1e-3;
    bad += !ok;
    printf("rm_values leg=%s y0=%.9g ylast=%.9g vs_ref_rel_rms=%.3e gate=%s\n", legs[L].name, got[0], got[nx - 1], dv,
           ok ? "pass" : "FAIL");
  }
  cudaEvent_t e0, e1;
  CUDA(cudaEventCreate(&e0));
  CUDA(cudaEventCreate(&e1));
  std::vector<std::vector<double>> ev(legs.size()), wall(legs.size());
  for (int r = 0; r < reps; ++r)
    for (size_t q = 0; q < legs.size(); ++q) {
      size_t L = r & 1 ? legs.size() - 1 - q : q;
      CUDA(cudaDeviceSynchronize());
      double t0 = now_ms();
      CUDA(cudaEventRecord(e0));
      legs[L].run();
      CUDA(cudaEventRecord(e1));
      CUDA(cudaStreamSynchronize(0));
      double t1 = now_ms();
      float ms;
      CUDA(cudaEventElapsedTime(&ms, e0, e1));
      ev[L].push_back(ms);
      wall[L].push_back(t1 - t0);
    }
  for (size_t L = 0; L < legs.size(); ++L) {
    auto a = ev[L], b = wall[L];
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    printf("rm leg=%s M=%d d=%d h=%d launches=%d reps=%d event_med_ms=%.4f event_min_ms=%.4f wall_med_ms=%.4f "
           "wall_min_ms=%.4f first_wall_ms=%.4f\n",
           legs[L].name, M, d, h, legs[L].launches, reps, a[a.size() / 2], a.front(), b[b.size() / 2], b.front(),
           first[L]);
    printf("rm_samples leg=%s event=", legs[L].name);
    for (size_t i = 0; i < ev[L].size(); ++i) printf("%s%.4f", i ? "," : "", ev[L][i]);
    printf(" wall=");
    for (size_t i = 0; i < wall[L].size(); ++i) printf("%s%.4f", i ? "," : "", wall[L][i]);
    printf("\n");
  }
  if (getenv("TB_REGION")) {
    CUDA(cudaDeviceSynchronize());
    CUDA(cudaProfilerStart());
    for (auto &l : legs) l.run();
    CUDA(cudaDeviceSynchronize());
    CUDA(cudaProfilerStop());
    printf("region legs=%zu\n", legs.size());
  }
  return bad ? 1 : 0;
}
