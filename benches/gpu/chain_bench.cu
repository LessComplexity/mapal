// The library way to run Mapal's multi-computation probes (S54): every timed
// probe of gen_tiles.py (`pt_<probe>_<n>`), rebuilt on the device with the same
// operand laws and run the ways a CUDA user would, next to Mapal's own numbers.
//
// Legs (only those a probe admits):
//   eager    cuBLAS sgemm per tile site + one simple CUDA kernel per per-cell step
//            or reduction, one stream — what an eager framework issues.
//   streams  eager with the independent parts on two streams (fan-out, diamond,
//            independent sites, two MLPs).
//   lt       the best fused library form: cuBLASLt with the epilogue the probe fits
//            (BIAS, RELU, RELU_BIAS; alpha for a power-of-two scale; beta·C for a
//            residual or a join of two products).
//   lt2      lt with the two independent MLPs on two streams.
//   cutlass  CUTLASS's stock SIMT f32 GEMM (tile_bench's configuration) with a fused
//            LinearCombination / LinearCombinationRelu epilogue (a bias is C with
//            ldc = 0, CUTLASS example 12's broadcast).
//   gemv     rowsum: sgemm + cublasSgemv against a ones vector.
//   fused    softmax_rows: sgemm + one warp-per-row softmax kernel (max, exp, sum,
//            quotient in one pass over the row: what a fusing compiler emits).
// rowsum and softmax_rows: every library leg reduces in its own order (a warp
// tree, gemv's blocking), Mapal in the oracle's (one in-order chain per row) —
// `chain_note ... library order != oracle order`.
//   fused    ew: the three maps as one kernel (what a fusing compiler emits).
// All math is pure FP32 (CUBLAS_COMPUTE_32F, default sgemm math: no TF32).
//
// Timing: legs interleaved round-robin, each repetition bracketed by cudaEvents on
// the legacy stream (both side streams are blocking, so the brackets cover them),
// 256 MB L2 flush before every timed repetition, 3 warmups per leg. GPU time
// only: operands are generated on the device before the clock, as Mapal's are.
//
// Values: Mapal's timed probe prints [0] and [last] of every output; pass them
// (in print order) after `reps` and every leg is gated:
// |leg − mapal| ≤ 1e-3 · max(|mapal|, rms(leg output)) — a summation-order bound,
// not bit equality (the libraries fold in their own order).
//
// usage: chain_bench <probe> <n> <reps> [mapal values...]
//   probe: epi bias ew pro indep mlp fanout diamond residual rowsum softmax_rows mlp2 loop
//          decode_<T> (n = d; the eager decode loop with and without a cache, S54)
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "cutlass/epilogue/thread/linear_combination_relu.h"
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
#define BLAS(x)                                                                \
  do {                                                                         \
    cublasStatus_t s_ = (x);                                                   \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                         \
      fprintf(stderr, "%s: cublas status %d\n", #x, (int)s_);                  \
      exit(102);                                                               \
    }                                                                          \
  } while (0)

using RowMajor = cutlass::layout::RowMajor;
using Cfg = cutlass::gemm::device::DefaultGemmConfiguration<cutlass::arch::OpClassSimt, CUTLASS_ARCH, float,
                                                            float, float, float>;
template <class Epi>
using SimtGemm = cutlass::gemm::device::Gemm<
    float, RowMajor, float, RowMajor, float, RowMajor, float, cutlass::arch::OpClassSimt, CUTLASS_ARCH,
    Cfg::ThreadblockShape, Cfg::WarpShape, Cfg::InstructionShape, Epi,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, Cfg::kStages>;
using CutLin = SimtGemm<cutlass::epilogue::thread::LinearCombination<float, 1, float, float>>;
using CutRelu = SimtGemm<cutlass::epilogue::thread::LinearCombinationRelu<float, 1, float, float>>;

// --- the probes' data: gen_tiles.py OPS / BIAS, mirrored.
struct Op {
  const char *name;
  int mul, add, mod, off;
  float div;
};
static const Op OPS[] = {{"a", 7, 13, 101, 50, 7.f},  {"b", 11, 57, 103, 51, 9.f},
                         {"e", 5, 3, 97, 48, 11.f},   {"f", 3, 1, 89, 44, 13.f},
                         {"g", 13, 7, 107, 53, 15.f}, {"p", 17, 29, 109, 54, 17.f}};

__global__ void gen_op(float *o, long long cnt, int mul, int add, int mod, int off, float div) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < cnt) o[i] = (float)(((int)i * mul + add) % mod - off) / div;
}

// One simple kernel per per-cell step (the eager form), selected by `op`.
enum { AFFINE, ADD, MUL, BIAS_ADD, BIAS_RELU, RELU, EW3 };
__global__ void cell(int op, float *o, const float *x, const float *y, long long cnt, int cols, float s,
                     float c) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cnt) return;
  float v = x[i];
  switch (op) {
    case AFFINE: o[i] = v * s + c; break;
    case ADD: o[i] = v + y[i]; break;
    case MUL: o[i] = v * y[i]; break;
    case BIAS_ADD: o[i] = v + y[i % cols]; break;
    case BIAS_RELU: v = v + y[i % cols]; o[i] = v > 0.f ? v : 0.f; break;
    case RELU: o[i] = v > 0.f ? v : 0.f; break;
    case EW3: o[i] = (v * 3.f + 2.f) * y[i]; break;
  }
}

// rowsum's eager reduction: one warp per row.
__global__ void row_sum(float *r, const float *c, int n) {
  int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
  if (row >= n) return;
  float s = 0.f;
  for (int j = lane; j < n; j += 32) s += c[(size_t)row * n + j];
  for (int d = 16; d; d /= 2) s += __shfl_down_sync(0xffffffffu, s, d);
  if (!lane) r[row] = s;
}

// softmax_rows' exp: gen_tiles.py `expo`, (1 + x/256)^256 with the base clamped at 0.
__device__ float expo(float x) {
  float v = fmaxf(x / 256.f + 1.f, 0.f);
  for (int i = 0; i < 8; ++i) v = v * v;
  return v;
}

// softmax_rows, eager: a row max and a row sum (one warp per row, as row_sum),
// exp and the quotient one thread per cell.
__global__ void row_max(float *r, const float *c, int n) {
  int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
  if (row >= n) return;
  float m = -1e9f;
  for (int j = lane; j < n; j += 32) m = fmaxf(m, c[(size_t)row * n + j]);
  for (int d = 16; d; d /= 2) m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, d));
  if (!lane) r[row] = m;
}
__global__ void sm_exp(float *o, const float *c, const float *mx, long long cnt, int n) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < cnt) o[i] = expo((c[i] - mx[i / n]) / (float)n);
}
__global__ void sm_div(float *o, const float *ex, const float *den, long long cnt, int n) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < cnt) o[i] = ex[i] / den[i / n];
}
// softmax_rows, fused: one warp per row — max, exp + sum, quotient.
__global__ void softmax_row(float *y, const float *c, int n) {
  int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
  if (row >= n) return;
  const float *x = c + (size_t)row * n;
  float *o = y + (size_t)row * n;
  float m = -1e9f, s = 0.f;
  for (int j = lane; j < n; j += 32) m = fmaxf(m, x[j]);
  for (int d = 16; d; d /= 2) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, d));
  for (int j = lane; j < n; j += 32) s += expo((x[j] - m) / (float)n);
  for (int d = 16; d; d /= 2) s += __shfl_xor_sync(0xffffffffu, s, d);
  for (int j = lane; j < n; j += 32) o[j] = expo((x[j] - m) / (float)n) / s;
}

static int n;
static long long nn;
static std::map<std::string, float *> buf;
static cudaStream_t st[2];
static cublasHandle_t hb[2];
static cublasLtHandle_t lt;
static void *ws[2];
static const size_t WS = 32u << 20;
static int launches;
static bool lt_unavailable;

static float *B(const std::string &k, long long cnt = 0) {
  auto it = buf.find(k);
  if (it != buf.end()) return it->second;
  float *p;
  CUDA(cudaMalloc(&p, (cnt ? cnt : nn) * 4));
  return buf[k] = p;
}

static void ew(int s, int op, const char *o, const char *x, const char *y, float sc = 0, float c = 0) {
  cell<<<(unsigned)((nn + 255) / 256), 256, 0, st[s]>>>(op, B(o), B(x), y ? B(y) : nullptr, nn, n, sc, c);
  ++launches;
}

// Row-major O = alpha·X·Y + beta·C is column-major Oᵀ = alpha·Yᵀ·Xᵀ + beta·Cᵀ.
static void sgemm(int s, const char *o, const char *x, const char *y, float alpha = 1, float beta = 0) {
  BLAS(cublasSgemm(hb[s], CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, B(y), n, B(x), n, &beta, B(o), n));
  ++launches;
}

// cuBLASLt with an epilogue: one plan (descriptors + heuristic algo) per use.
struct LtPlan {
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t l;
  cublasLtMatmulAlgo_t algo;
  LtPlan(cublasLtEpilogue_t epi, const float *bias) {
    BLAS(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    BLAS(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof epi));
    if (bias) BLAS(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof bias));
    BLAS(cublasLtMatrixLayoutCreate(&l, CUDA_R_32F, n, n, n));
    cublasLtMatmulPreference_t pref;
    BLAS(cublasLtMatmulPreferenceCreate(&pref));
    BLAS(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &WS, sizeof WS));
    cublasLtMatmulHeuristicResult_t res;
    int got = 0;
    // No algorithm for this epilogue on this part drops the lt legs, not the run.
    cublasStatus_t s = cublasLtMatmulAlgoGetHeuristic(lt, op, l, l, l, l, pref, 1, &res, &got);
    if (s != CUBLAS_STATUS_SUCCESS || !got) {
      printf("chain_note cublasLt has no algorithm for epilogue %d (status %d): lt legs dropped\n", (int)epi,
             (int)s);
      lt_unavailable = true;
    }
    algo = res.algo;
    cublasLtMatmulPreferenceDestroy(pref);
  }
  void run(int s, const char *o, const char *x, const char *y, float alpha = 1, float beta = 0,
           const char *c = nullptr) {
    const float *cp = c ? B(c) : B(o);
    BLAS(cublasLtMatmul(lt, op, &alpha, B(y), l, B(x), l, &beta, cp, l, B(o), l, &algo, ws[s], WS, st[s]));
    ++launches;
  }
};

template <class G>
static void cut(int s, const char *o, const char *x, const char *y, float alpha, float beta,
                const char *c = nullptr, int ldc = -1) {
  G g;
  const float *cp = c ? B(c) : B(o);
  typename G::Arguments args({n, n, n}, {B(x), n}, {B(y), n}, {cp, ldc < 0 ? n : ldc}, {B(o), n},
                             {alpha, beta});
  if (g(args, nullptr, st[s]) != cutlass::Status::kSuccess) {
    fprintf(stderr, "cutlass failed\n");
    exit(102);
  }
  ++launches;
}

// s waits for everything issued so far on `from`.
static void join(int s, int from) {
  cudaEvent_t e;
  CUDA(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
  CUDA(cudaEventRecord(e, st[from]));
  CUDA(cudaStreamWaitEvent(st[s], e, 0));
  CUDA(cudaEventDestroy(e));
}

struct Leg {
  const char *name;
  std::function<void()> run;
};

// --- S54 `decode_<T> <d>`: gen_tiles.py `decode` as an eager framework loop
// (one launch per op, one stream, nothing synchronized inside the loop), the
// way a user writes it, with the operand laws of tile_bench's `--decode`:
//   cache    the hand-written KV cache: per step the token row, K[t] and V[t]
//            (two sgemv into the cache rows), q, s = K·q, o = sᵀ·V (all T rows).
//   nocache  the same loop recomputing K = x·Wk and V = x·Wv whole every step
//            (two sgemm) — what the program says, without a hand-written cache.
// Host wall (steady_clock around the loop and its final synchronize) and event
// time, launches per step; gated on Mapal's out[0] and out[d-1].
__global__ void dec_row(float *xrow, const float *o, int t, int d) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < d) xrow[j] = (float)(((t * d + j) * 7 + 13) % 101 - 50) / 64.0f + o[j];
}
__global__ void dec_q(float *q, int t, int d, float div) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < d) q[k] = (float)(((t * d + k) * 5 + 3) % 97 - 48) / div;
}
static int decode_main(int T, int d, int reps, int argc, char **argv) {
  CUDA(cudaFree(0));
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
  BLAS(cublasCreate(&h));
  BLAS(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));
  const float one = 1.0f, zero = 0.0f;
  const int tb = 256;
  int nl = 0;
  auto loop = [&](bool cache) {
    nl = 0;
    CUDA(cudaMemset(dx, 0, (size_t)T * d * 4));
    CUDA(cudaMemset(dK, 0, (size_t)T * d * 4));
    CUDA(cudaMemset(dV, 0, (size_t)T * d * 4));
    CUDA(cudaMemset(dout, 0, d * 4));
    for (int t = 0; t < T; ++t) {
      float *xr = dx + (size_t)t * d;
      dec_row<<<(d + tb - 1) / tb, tb>>>(xr, dout, t, d);
      if (cache) {  // Row-major W (d×d) is column-major Wᵀ: K[t] = Wᵀ·x[t].
        BLAS(cublasSgemv(h, CUBLAS_OP_N, d, d, &one, dWk, d, xr, 1, &zero, dK + (size_t)t * d, 1));
        BLAS(cublasSgemv(h, CUBLAS_OP_N, d, d, &one, dWv, d, xr, 1, &zero, dV + (size_t)t * d, 1));
      } else {  // Row-major K = x·W is column-major Kᵀ = Wᵀ·xᵀ.
        BLAS(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, d, T, d, &one, dWk, d, dx, d, &zero, dK, d));
        BLAS(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, d, T, d, &one, dWv, d, dx, d, &zero, dV, d));
      }
      dec_q<<<(d + tb - 1) / tb, tb>>>(dq, t, d, div);
      BLAS(cublasSgemv(h, CUBLAS_OP_T, d, T, &one, dK, d, dq, 1, &zero, ds, 1));
      BLAS(cublasSgemv(h, CUBLAS_OP_N, d, T, &one, dV, d, ds, 1, &zero, dout, 1));
      nl += 6;
    }
  };
  std::vector<double> want;
  for (int i = 4; i < argc; ++i) want.push_back(strtod(argv[i], nullptr));
  int bad = 0;
  for (bool cache : {true, false}) {
    const char *leg = cache ? "cache" : "nocache";
    loop(cache);
    CUDA(cudaDeviceSynchronize());
    std::vector<float> o(d);
    CUDA(cudaMemcpy(o.data(), dout, d * 4, cudaMemcpyDeviceToHost));
    double ss = 0;
    for (float v : o) ss += (double)v * v;
    const double rms = std::sqrt(ss / d);
    double dev = 0;
    const float got[2] = {o[0], o[d - 1]};
    for (size_t c = 0; c < want.size() && c < 2; ++c)
      dev = std::max(dev, std::fabs(got[c] - want[c]) / std::max(std::fabs(want[c]), rms));
    const bool ok = std::isfinite(rms) && rms > 0 && (want.size() < 2 || dev <= 1e-3);
    bad += !ok;
    printf("chain_values probe=decode T=%d d=%d leg=%s values=%.9g,%.9g gate=%s vs_mapal=%.3e\n", T, d, leg, got[0],
           got[1], ok ? (want.size() < 2 ? "pass-no-mapal" : "pass") : "FAIL", dev);
  }
  cudaEvent_t e0, e1;
  CUDA(cudaEventCreate(&e0));
  CUDA(cudaEventCreate(&e1));
  std::vector<double> wall[2], ev[2];
  for (int r = 0; r < reps; ++r)
    for (int L = 0; L < 2; ++L) {
      auto t0 = std::chrono::steady_clock::now();
      CUDA(cudaEventRecord(e0));
      loop(L == 0);
      CUDA(cudaEventRecord(e1));
      CUDA(cudaEventSynchronize(e1));
      wall[L].push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
      float x;
      CUDA(cudaEventElapsedTime(&x, e0, e1));
      ev[L].push_back(x);
    }
  for (int L = 0; L < 2; ++L) {
    std::sort(wall[L].begin(), wall[L].end());
    std::sort(ev[L].begin(), ev[L].end());
    printf("chain probe=decode T=%d d=%d leg=%s launches=%d per_step=6 reps=%d wall_min_ms=%.3f wall_med_ms=%.3f "
           "wall_max_ms=%.3f event_med_ms=%.3f\n",
           T, d, L == 0 ? "cache" : "nocache", nl, reps, wall[L].front(), wall[L][reps / 2], wall[L].back(),
           ev[L][reps / 2]);
  }
  return bad ? 1 : 0;
}

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s <probe> <n> <reps> [mapal values...]\n", argv[0]);
    return 2;
  }
  if (!strncmp(argv[1], "decode_", 7)) return decode_main(atoi(argv[1] + 7), atoi(argv[2]), atoi(argv[3]), argc, argv);
  const std::string probe = argv[1];
  n = atoi(argv[2]);
  nn = (long long)n * n;
  const int reps = atoi(argv[3]);
  CUDA(cudaFree(0));
  cudaDeviceProp prop;
  CUDA(cudaGetDeviceProperties(&prop, 0));
  printf("device name=\"%s\" cc=%d.%d\n", prop.name, prop.major, prop.minor);
  for (int s = 0; s < 2; ++s) {
    CUDA(cudaStreamCreate(&st[s]));  // blocking: the legacy-stream events bracket them
    BLAS(cublasCreate(&hb[s]));
    BLAS(cublasSetStream(hb[s], st[s]));
    BLAS(cublasSetMathMode(hb[s], CUBLAS_DEFAULT_MATH));
    CUDA(cudaMalloc(&ws[s], WS));
    BLAS(cublasSetWorkspace(hb[s], ws[s], WS));
  }
  BLAS(cublasLtCreate(&lt));
  for (const Op &o : OPS)
    gen_op<<<(unsigned)((nn + 255) / 256), 256>>>(B(o.name), nn, o.mul, o.add, o.mod, o.off, o.div);
  // The loop's weight: ÷ n (gen_tiles.py `probe_ops`); the column biases.
  gen_op<<<(unsigned)((nn + 255) / 256), 256>>>(B("w"), nn, 11, 57, 103, 51, (float)n);
  gen_op<<<(n + 255) / 256, 256>>>(B("bias", n), n, 1, 0, 13, 6, 5.f);
  gen_op<<<(n + 255) / 256, 256>>>(B("bias2", n), n, 1, 0, 11, 5, 4.f);
  std::vector<float> ones(n, 1.f);
  CUDA(cudaMemcpy(B("ones", n), ones.data(), n * 4, cudaMemcpyHostToDevice));
  CUDA(cudaDeviceSynchronize());

  std::vector<std::pair<const char *, long long>> outs;
  std::vector<Leg> legs;
  const int LOOP_ITERS = 8;  // gen_tiles.py `loop_act(n, 8)`
  if (probe == "epi") {  // d = (a·b)·0.5 + 1
    outs = {{"d", nn}};
    auto *p = new LtPlan(CUBLASLT_EPILOGUE_BIAS, B("ones"));
    legs = {{"eager", [] { sgemm(0, "c", "a", "b"); ew(0, AFFINE, "d", "c", nullptr, 0.5f, 1.f); }},
            {"lt", [p] { p->run(0, "d", "a", "b", 0.5f); }},
            {"cutlass", [] { cut<CutLin>(0, "d", "a", "b", 0.5f, 1.f, "ones", 0); }}};
  } else if (probe == "bias") {  // d = a·b + bias[j]
    outs = {{"d", nn}};
    auto *p = new LtPlan(CUBLASLT_EPILOGUE_BIAS, B("bias"));
    legs = {{"eager", [] { sgemm(0, "c", "a", "b"); ew(0, BIAS_ADD, "d", "c", "bias"); }},
            {"lt", [p] { p->run(0, "d", "a", "b"); }},
            {"cutlass", [] { cut<CutLin>(0, "d", "a", "b", 1.f, 1.f, "bias", 0); }}};
  } else if (probe == "ew") {  // z = (a·3 + 2)·b
    outs = {{"z", nn}};
    legs = {{"eager",
             [] {
               ew(0, AFFINE, "x", "a", nullptr, 3.f, 0.f);
               ew(0, AFFINE, "y", "x", nullptr, 1.f, 2.f);
               ew(0, MUL, "z", "y", "b");
             }},
            {"fused", [] { ew(0, EW3, "z", "a", "b"); }}};
  } else if (probe == "pro") {  // c = (a·0.25)·b
    outs = {{"c", nn}};
    auto *p = new LtPlan(CUBLASLT_EPILOGUE_DEFAULT, nullptr);
    legs = {{"eager", [] { ew(0, AFFINE, "a2", "a", nullptr, 0.25f, 0.f); sgemm(0, "c", "a2", "b"); }},
            {"lt", [p] { p->run(0, "c", "a", "b", 0.25f); }},
            {"cutlass", [] { cut<CutLin>(0, "c", "a", "b", 0.25f, 0.f); }}};
  } else if (probe == "indep") {  // g = a·b + e·f
    outs = {{"g", nn}};
    auto *p = new LtPlan(CUBLASLT_EPILOGUE_DEFAULT, nullptr);
    legs = {{"eager", [] { sgemm(0, "c", "a", "b"); sgemm(0, "d", "e", "f"); ew(0, ADD, "g", "c", "d"); }},
            {"streams",
             [] {
               sgemm(0, "c", "a", "b");
               sgemm(1, "d", "e", "f");
               join(0, 1);
               ew(0, ADD, "g", "c", "d");
             }},
            {"lt", [p] { sgemm(0, "c", "a", "b"); p->run(0, "g", "e", "f", 1.f, 1.f, "c"); }},
            {"cutlass", [] { cut<CutLin>(0, "c", "a", "b", 1.f, 0.f); cut<CutLin>(0, "g", "e", "f", 1.f, 1.f, "c"); }}};
  } else if (probe == "mlp") {  // d = ((a·b)·0.5)·e
    outs = {{"d", nn}};
    auto *p = new LtPlan(CUBLASLT_EPILOGUE_DEFAULT, nullptr);
    legs = {{"eager", [] { sgemm(0, "c", "a", "b"); ew(0, AFFINE, "h", "c", nullptr, 0.5f, 0.f); sgemm(0, "d", "h", "e"); }},
            {"lt", [p] { p->run(0, "h", "a", "b", 0.5f); p->run(0, "d", "h", "e"); }},
            {"cutlass", [] { cut<CutLin>(0, "h", "a", "b", 0.5f, 0.f); cut<CutLin>(0, "d", "h", "e", 1.f, 0.f); }}};
  } else if (probe == "fanout") {  // c = a·b; h = c·0.5 + 1; d = c·e
    outs = {{"c", nn}, {"h", nn}, {"d", nn}};
    legs = {{"eager", [] { sgemm(0, "c", "a", "b"); ew(0, AFFINE, "h", "c", nullptr, 0.5f, 1.f); sgemm(0, "d", "c", "e"); }},
            {"streams",
             [] {
               sgemm(0, "c", "a", "b");
               join(1, 0);
               ew(1, AFFINE, "h", "c", nullptr, 0.5f, 1.f);
               sgemm(0, "d", "c", "e");
               join(0, 1);
             }}};
  } else if (probe == "diamond") {  // c = a·b; h = c·e + c·g
    outs = {{"h", nn}};
    auto *p = new LtPlan(CUBLASLT_EPILOGUE_DEFAULT, nullptr);
    legs = {{"eager",
             [] {
               sgemm(0, "c", "a", "b");
               sgemm(0, "d1", "c", "e");
               sgemm(0, "d2", "c", "g");
               ew(0, ADD, "h", "d1", "d2");
             }},
            {"streams",
             [] {
               sgemm(0, "c", "a", "b");
               join(1, 0);
               sgemm(0, "d1", "c", "e");
               sgemm(1, "d2", "c", "g");
               join(0, 1);
               ew(0, ADD, "h", "d1", "d2");
             }},
            {"lt", [p] { sgemm(0, "c", "a", "b"); sgemm(0, "d1", "c", "e"); p->run(0, "h", "c", "g", 1.f, 1.f, "d1"); }},
            {"cutlass",
             [] {
               cut<CutLin>(0, "c", "a", "b", 1.f, 0.f);
               cut<CutLin>(0, "d1", "c", "e", 1.f, 0.f);
               cut<CutLin>(0, "h", "c", "g", 1.f, 1.f, "d1");
             }}};
  } else if (probe == "residual") {  // d = a·b + a
    outs = {{"d", nn}};
    auto *p = new LtPlan(CUBLASLT_EPILOGUE_DEFAULT, nullptr);
    legs = {{"eager", [] { sgemm(0, "c", "a", "b"); ew(0, ADD, "d", "c", "a"); }},
            {"lt", [p] { p->run(0, "d", "a", "b", 1.f, 1.f, "a"); }},
            {"cutlass", [] { cut<CutLin>(0, "d", "a", "b", 1.f, 1.f, "a"); }}};
  } else if (probe == "rowsum") {  // r[i] = Σ_j (a·b)[i][j]
    outs = {{"r", n}};
    B("r", n);
    legs = {{"eager",
             [] {
               sgemm(0, "c", "a", "b");
               row_sum<<<(n + 7) / 8, 256, 0, st[0]>>>(B("r"), B("c"), n);
               ++launches;
             }},
            {"gemv", [] {
               sgemm(0, "c", "a", "b");
               const float one = 1.f, zero = 0.f;
               // Row-major c is column-major cᵀ: r = (cᵀ)ᵀ·1.
               BLAS(cublasSgemv(hb[0], CUBLAS_OP_T, n, n, &one, B("c"), n, B("ones"), 1, &zero, B("r"), 1));
               ++launches;
             }}};
  } else if (probe == "softmax_rows") {  // y = softmax over the rows of a·b, temperature n
    outs = {{"y", nn}};
    B("mx", n);
    B("den", n);
    legs = {{"eager",
             [] {
               sgemm(0, "c", "a", "b");
               row_max<<<(n + 7) / 8, 256, 0, st[0]>>>(B("mx"), B("c"), n);
               sm_exp<<<(unsigned)((nn + 255) / 256), 256, 0, st[0]>>>(B("ex"), B("c"), B("mx"), nn, n);
               row_sum<<<(n + 7) / 8, 256, 0, st[0]>>>(B("den"), B("ex"), n);
               sm_div<<<(unsigned)((nn + 255) / 256), 256, 0, st[0]>>>(B("y"), B("ex"), B("den"), nn, n);
               launches += 4;
             }},
            {"fused", [] {
               sgemm(0, "c", "a", "b");
               softmax_row<<<(n + 7) / 8, 256, 0, st[0]>>>(B("y"), B("c"), n);
               ++launches;
             }}};
  } else if (probe == "mlp2") {  // d1 = relu(a·b + bias)·e ; d2 = relu(f·g + bias2)·p
    outs = {{"d1", nn}, {"d2", nn}};
    auto *p1 = new LtPlan(CUBLASLT_EPILOGUE_RELU_BIAS, B("bias"));
    auto *p2 = new LtPlan(CUBLASLT_EPILOGUE_RELU_BIAS, B("bias2"));
    auto *p0 = new LtPlan(CUBLASLT_EPILOGUE_DEFAULT, nullptr);
    auto eager = [](int s1, int s2) {
      sgemm(s1, "c1", "a", "b");
      ew(s1, BIAS_RELU, "h1", "c1", "bias");
      sgemm(s1, "d1", "h1", "e");
      sgemm(s2, "c2", "f", "g");
      ew(s2, BIAS_RELU, "h2", "c2", "bias2");
      sgemm(s2, "d2", "h2", "p");
      if (s1 != s2) join(s1, s2);
    };
    auto fused = [p0, p1, p2](int s1, int s2) {
      p1->run(s1, "h1", "a", "b");
      p0->run(s1, "d1", "h1", "e");
      p2->run(s2, "h2", "f", "g");
      p0->run(s2, "d2", "h2", "p");
      if (s1 != s2) join(s1, s2);
    };
    legs = {{"eager", [eager] { eager(0, 0); }},
            {"streams", [eager] { eager(0, 1); }},
            {"lt", [fused] { fused(0, 0); }},
            {"lt2", [fused] { fused(0, 1); }},
            {"cutlass", [] {
               cut<CutRelu>(0, "h1", "a", "b", 1.f, 1.f, "bias", 0);
               cut<CutLin>(0, "d1", "h1", "e", 1.f, 0.f);
               cut<CutRelu>(0, "h2", "f", "g", 1.f, 1.f, "bias2", 0);
               cut<CutLin>(0, "d2", "h2", "p", 1.f, 0.f);
             }}};
  } else if (probe == "loop") {  // x ← relu(x·w), 8 steps from x = a
    outs = {{"out", nn}};
    auto *p = new LtPlan(CUBLASLT_EPILOGUE_RELU, nullptr);
    // Step i reads x_i and writes x_{i+1}; x_0 = a, x_8 = out.
    static const char *xs[] = {"a", "x1", "x2", "x1", "x2", "x1", "x2", "x1", "out"};
    legs = {{"eager",
             [] {
               for (int i = 0; i < LOOP_ITERS; ++i) {
                 sgemm(0, "y", xs[i], "w");
                 ew(0, RELU, xs[i + 1], "y", nullptr);
               }
             }},
            {"lt", [p] { for (int i = 0; i < LOOP_ITERS; ++i) p->run(0, xs[i + 1], xs[i], "w"); }},
            {"cutlass", [] {
               for (int i = 0; i < LOOP_ITERS; ++i) cut<CutRelu>(0, xs[i + 1], xs[i], "w", 1.f, 0.f);
             }}};
  } else {
    fprintf(stderr, "unknown probe %s\n", probe.c_str());
    return 2;
  }
  if (lt_unavailable)
    legs.erase(std::remove_if(legs.begin(), legs.end(), [](const Leg &l) { return !strncmp(l.name, "lt", 2); }),
               legs.end());
  for (auto &o : outs) B(o.first, o.second);
  if (probe == "rowsum" || probe == "softmax_rows")
    printf("chain_note probe=%s n=%d every library leg reduces in its own order (warp tree, gemv blocking): "
           "library order != oracle order\n",
           probe.c_str(), n);

  // Mapal's printed values, in print order: [0] and [last] of every output.
  std::vector<double> want;
  for (int i = 4; i < argc; ++i) want.push_back(strtod(argv[i], nullptr));
  const bool gate = want.size() == 2 * outs.size();
  if (argc > 4 && !gate)
    printf("chain_gate probe=%s n=%d SKIPPED: %zu values for %zu outputs\n", probe.c_str(), n, want.size(),
           outs.size());

  // Every leg: its printed cells against Mapal's, and its whole output against
  // the first leg's (so a leg cannot pass on two corners alone), both relative
  // to the output's rms. A NaN deviation fails.
  int bad = 0;
  std::vector<int> nlaunch(legs.size());
  std::vector<std::vector<float>> ref(outs.size());
  for (size_t L = 0; L < legs.size(); ++L) {
    for (int w = 0; w < 3; ++w) {
      launches = 0;
      legs[L].run();
    }
    nlaunch[L] = launches;
    CUDA(cudaDeviceSynchronize());
    std::string vals;
    double vs_mapal = 0, vs_first = 0;
    bool sane = true;  // every value finite, every output nonzero
    for (size_t k = 0; k < outs.size(); ++k) {
      std::vector<float> h(outs[k].second);
      CUDA(cudaMemcpy(h.data(), B(outs[k].first), h.size() * 4, cudaMemcpyDeviceToHost));
      double ss = 0;
      for (float v : h) ss += (double)v * v;
      const double rms = std::sqrt(ss / h.size());
      sane = sane && std::isfinite(rms) && rms > 0;
      if (L == 0) ref[k] = h;
      for (size_t i = 0; sane && i < h.size(); ++i)
        vs_first = std::max(vs_first, std::fabs((double)h[i] - ref[k][i]) / rms);
      const float got[2] = {h.front(), h.back()};
      for (int c = 0; c < 2; ++c) {
        char t[64];
        snprintf(t, sizeof t, "%s%.9g", vals.empty() ? "" : ",", got[c]);
        vals += t;
        if (gate && sane) {
          const double m = want[2 * k + c];
          sane = std::isfinite(m);
          vs_mapal = std::max(vs_mapal, std::fabs(got[c] - m) / std::max(std::fabs(m), rms));
        }
      }
    }
    const bool ok = sane && vs_first <= 1e-3 && (!gate || vs_mapal <= 1e-3);
    bad += !ok;
    printf("chain_values probe=%s n=%d leg=%s values=%s gate=%s vs_mapal=%.3e vs_%s=%.3e\n", probe.c_str(), n,
           legs[L].name, vals.c_str(), ok ? (gate ? "pass" : "pass-no-mapal") : "FAIL", vs_mapal, legs[0].name,
           vs_first);
  }

  void *flush;
  const size_t flush_bytes = 256u << 20;
  CUDA(cudaMalloc(&flush, flush_bytes));
  cudaEvent_t t0, t1;
  CUDA(cudaEventCreate(&t0));
  CUDA(cudaEventCreate(&t1));
  std::vector<std::vector<float>> ms(legs.size());
  for (int r = 0; r < reps; ++r)
    for (size_t L = 0; L < legs.size(); ++L) {
      CUDA(cudaMemsetAsync(flush, r & 0xff, flush_bytes));
      CUDA(cudaEventRecord(t0));
      legs[L].run();
      CUDA(cudaEventRecord(t1));
      CUDA(cudaEventSynchronize(t1));
      float x;
      CUDA(cudaEventElapsedTime(&x, t0, t1));
      ms[L].push_back(x);
    }
  for (size_t L = 0; L < legs.size(); ++L) {
    auto v = ms[L];
    std::sort(v.begin(), v.end());
    printf("chain probe=%s n=%d leg=%s launches=%d reps=%d min_ms=%.4f med_ms=%.4f max_ms=%.4f\n", probe.c_str(), n,
           legs[L].name, nlaunch[L], reps, v.front(), v[v.size() / 2], v.back());
    printf("chain_samples probe=%s n=%d leg=%s ms=", probe.c_str(), n, legs[L].name);
    for (size_t i = 0; i < ms[L].size(); ++i) printf("%s%.4f", i ? "," : "", ms[L][i]);
    printf("\n");
  }
  return bad ? 1 : 0;
}
