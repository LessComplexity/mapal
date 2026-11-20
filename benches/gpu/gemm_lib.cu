// model-scale: the library legs of the layer's projections (gen_tiles.py `pt_pj_m<M>_n<N>_k<K>`): C = A·B, A M×K and
// B K×N row-major, on the program's operands (OPS 'a', 'b': `(t·mul + add) % mod − off`, ÷ div), generated on the
// device before the clock as Mapal's are.
//
// Legs (pure FP32: CUBLAS_DEFAULT_MATH / CUBLAS_COMPUTE_32F; `--tf32`: their TF32 twins only,
// CUBLAS_TF32_TENSOR_OP_MATH / CUBLAS_COMPUTE_32F_FAST_TF32):
//   cublas   cublasSgemm (the column-major transpose trick: Cᵀ = Bᵀ·Aᵀ).
//   lt       cublasLtMatmul: up to 8 heuristic algorithms (32 MB workspace), each timed over 5 runs, the fastest
//            kept — the best cuBLASLt offers for the shape.
// Timing (chain_bench.cu's convention): 3 warm-ups per leg, then `reps` repetitions round-robin, each after a
// 256 MB L2 flush, bracketed by CUDA events; min / median / max ms.
// Values: c[0] and c[M·N − 1] against Mapal's printed values when given:
// |leg − mapal| ≤ tol · max(|mapal|, rms(leg output)), tol 1e-3 (f32) / 1e-2 (TF32).
//
// usage: gemm_lib <stem> <M> <N> <K> <reps> [--tf32] [mapal c[0] c[last]]
// prints `gemm prog=<stem> leg=<leg> algo=<i>/<n> min_ms= med_ms= max_ms= within=yes|no|no-mapal maxdev= values=`.
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x)                                                                          \
  do {                                                                                 \
    cudaError_t e_ = (x);                                                              \
    if (e_ != cudaSuccess) {                                                           \
      fprintf(stderr, "cuda %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
      exit(1);                                                                         \
    }                                                                                  \
  } while (0)
#define BLAS(x)                                                              \
  do {                                                                       \
    int s_ = (int)(x);                                                       \
    if (s_ != 0) {                                                           \
      fprintf(stderr, "blas status %d at %s:%d\n", s_, __FILE__, __LINE__); \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

__global__ void law(float *o, long long n, long long mul, long long add, long long mod, long long off, float div) {
  for (long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x; t < n; t += (long long)gridDim.x * blockDim.x)
    o[t] = (float)((t * mul + add) % mod - off) / div;
}

int main(int argc, char **argv) {
  if (argc < 6) {
    fprintf(stderr, "usage: gemm_lib <stem> M N K reps [--tf32] [mapal c0 clast]\n");
    return 2;
  }
  const std::string stem = argv[1];
  const int M = atoi(argv[2]), N = atoi(argv[3]), K = atoi(argv[4]), reps = atoi(argv[5]);
  int ai = 6;
  const bool tf32 = argc > ai && !strcmp(argv[ai], "--tf32");
  if (tf32) ++ai;
  std::vector<double> want;
  for (; ai < argc; ++ai) want.push_back(atof(argv[ai]));
  const double tol = tf32 ? 1e-2 : 1e-3;

  float *A, *B, *C, *flush;
  void *ws;
  const size_t WS = 32 << 20;
  CK(cudaMalloc(&A, sizeof(float) * M * K));
  CK(cudaMalloc(&B, sizeof(float) * K * (size_t)N));
  CK(cudaMalloc(&C, sizeof(float) * M * (size_t)N));
  CK(cudaMalloc(&flush, 256 << 20));
  CK(cudaMalloc(&ws, WS));
  law<<<1024, 256>>>(A, (long long)M * K, 7, 13, 101, 50, 7.0f);
  law<<<1024, 256>>>(B, (long long)K * N, 11, 57, 103, 51, 9.0f);
  CK(cudaDeviceSynchronize());

  cublasHandle_t h;
  cublasLtHandle_t lt;
  BLAS(cublasCreate(&h));
  BLAS(cublasLtCreate(&lt));
  BLAS(cublasSetMathMode(h, tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH));
  const float one = 1.0f, zero = 0.0f;

  // cuBLASLt: Dᵀ (N×M) = Bᵀ (N×K) · Aᵀ (K×M), column-major.
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t la, lb, lc;
  BLAS(cublasLtMatmulDescCreate(&op, tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F, CUDA_R_32F));
  BLAS(cublasLtMatrixLayoutCreate(&la, CUDA_R_32F, N, K, N));
  BLAS(cublasLtMatrixLayoutCreate(&lb, CUDA_R_32F, K, M, K));
  BLAS(cublasLtMatrixLayoutCreate(&lc, CUDA_R_32F, N, M, N));
  cublasLtMatmulPreference_t pref;
  BLAS(cublasLtMatmulPreferenceCreate(&pref));
  BLAS(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &WS, sizeof WS));
  cublasLtMatmulHeuristicResult_t res[8];
  int nres = 0;
  BLAS(cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 8, res, &nres));
  auto run_lt = [&](int i) {
    BLAS(cublasLtMatmul(lt, op, &one, B, la, A, lb, &zero, C, lc, C, lc, &res[i].algo, ws, WS, 0));
  };
  auto run_cublas = [&]() { BLAS(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, B, N, A, K, &zero, C, N)); };
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  auto time_once = [&](auto &&f) {
    CK(cudaMemsetAsync(flush, 0, 256 << 20));
    CK(cudaEventRecord(e0));
    f();
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    return ms;
  };
  // The fastest heuristic algorithm (median of 5, after 2 warm-ups).
  int best = 0;
  float best_ms = 1e30f;
  for (int i = 0; i < nres; ++i) {
    std::vector<float> t;
    for (int r = 0; r < 7; ++r) {
      float ms = time_once([&] { run_lt(i); });
      if (r >= 2) t.push_back(ms);
    }
    std::sort(t.begin(), t.end());
    if (t[2] < best_ms) best_ms = t[2], best = i;
  }

  struct Leg {
    const char *name;
    std::vector<float> ms;
    std::vector<float> vals;
    double dev = NAN, rms = 0;
  };
  std::vector<Leg> legs = {{tf32 ? "cublas-tf32" : "cublas"}, {tf32 ? "lt-tf32" : "lt"}};
  if (nres == 0) legs.pop_back();
  auto run_leg = [&](size_t l) {
    if (l == 0)
      run_cublas();
    else
      run_lt(best);
  };
  std::vector<float> host((size_t)M * N);
  for (size_t l = 0; l < legs.size(); ++l) {  // values first
    run_leg(l);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(host.data(), C, sizeof(float) * host.size(), cudaMemcpyDeviceToHost));
    double ss = 0;
    for (float v : host) ss += (double)v * v;
    legs[l].rms = std::sqrt(ss / host.size());
    legs[l].vals = {host[0], host.back()};
    if (want.size() >= 2) {
      legs[l].dev = 0;
      for (int i = 0; i < 2; ++i)
        legs[l].dev = std::max(legs[l].dev, std::fabs(legs[l].vals[i] - want[i]) /
                                                std::max({std::fabs(want[i]), legs[l].rms, 1e-30}));
    }
    for (int w = 0; w < 3; ++w) run_leg(l);
  }
  CK(cudaDeviceSynchronize());
  for (int r = 0; r < reps; ++r)
    for (size_t l = 0; l < legs.size(); ++l) legs[l].ms.push_back(time_once([&] { run_leg(l); }));
  for (auto &L : legs) {
    std::sort(L.ms.begin(), L.ms.end());
    const char *within = want.size() < 2 ? "no-mapal" : L.dev <= tol ? "yes" : "no";
    printf("gemm prog=%s leg=%s algo=%d/%d min_ms=%.4f med_ms=%.4f max_ms=%.4f within=%s maxdev=%.3e values=%.7g,%.7g\n",
           stem.c_str(), L.name, strstr(L.name, "lt") ? best : -1, nres, L.ms[0], L.ms[L.ms.size() / 2], L.ms.back(),
           within, L.dev, L.vals[0], L.vals[1]);
  }
  return 0;
}
