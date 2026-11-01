// S63 (from the S62 disassembly sessions, `benches/results/gpu/disassembly/s62-integrated/rtx-5090/tools/`): the residual MLP of `pt_rm_m<M>_d<d>_h<h>` (benches/gpu/gen_tiles.py)
//   s1 = x·w1 (M×h, K=d); hh = relu(s1 + bias[j]); s2 = hh·w2 (M×d, K=h); y = s2 + x
// on Mapal's input laws, every implementation in ONE process:
//   Mapal      the two kernels the emitter wrote for the program (`<dir>/<stem>.ptx`,
//              `<stem>.k`: per launch `kernel gx gy block smem` + its five argument roles),
//              launched the way mapal_gpu.c launches them (cuModuleGetFunction per launch)
//   cublas     2 cublasSgemm (default math, FP32) + bias/relu kernel + residual kernel
//   lt         cuBLASLt, fused: RELU_BIAS epilogue on the up projection, beta=1 residual
//              (C = x) on the down projection; COMPUTE_32F; heuristic #0, 32 MiB workspace
//   cutlass    stock SIMT config (128x128x8 / 32x64x8, 2 stages), fused: LinearCombinationRelu
//              with the bias broadcast as C (ldc 0, beta 1), residual as C (beta 1)
//   cutlass_tuned  the same epilogues on the 16x64x16 / 8x32x16 config (competitive-cutlass.md)
// Gates before timing: Mapal arms bit-exact vs the CPU ordered fold of their face on 256
// sampled outputs (_r/_rc: within 1e-3 relative of f64); every leg's max relative error vs f64.
// `cold` = the first execution of each leg in this process (lazy module/function loading and
// library first-call work included: the state of a one-shot program's iteration), legs in
// argument order (DIS_REVERSE=1: reversed). Then 5 warmups per leg and `reps` iterations
// round-robin, ABBA; each iteration: wall clock around (event, work, event, stream sync).
// No L2 flush: the program's iteration reads operands it just generated (L2-resident).
// DIS_PROFILE=1: one cudaProfilerStart/Stop region with every leg once, no timing.
// usage: mlp_bench M d h reps dir stem...
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <string>
#include <vector>

#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/gemm/device/gemm.h"

#define CUDA(x)                                                                         \
  do {                                                                                  \
    cudaError_t e_ = (x);                                                               \
    if (e_ != cudaSuccess) {                                                            \
      fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_));                          \
      exit(102);                                                                        \
    }                                                                                   \
  } while (0)
#define CU(x)                                                                           \
  do {                                                                                  \
    CUresult r_ = (x);                                                                  \
    if (r_ != CUDA_SUCCESS) {                                                           \
      const char *s_ = 0;                                                               \
      cuGetErrorString(r_, &s_);                                                        \
      fprintf(stderr, "%s: %s\n", #x, s_ ? s_ : "?");                                   \
      exit(102);                                                                        \
    }                                                                                   \
  } while (0)
#define BLAS(x)                                                                         \
  do {                                                                                  \
    cublasStatus_t s_ = (x);                                                            \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                                  \
      fprintf(stderr, "%s: status %d\n", #x, (int)s_);                                  \
      exit(102);                                                                        \
    }                                                                                   \
  } while (0)

template <int BM, int BN, int BK, int WM, int WN, int WK, class Epi>
using Gemm = cutlass::gemm::device::Gemm<float, cutlass::layout::RowMajor, float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<BM, BN, BK>, cutlass::gemm::GemmShape<WM, WN, WK>, cutlass::gemm::GemmShape<1, 1, 1>,
    Epi, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, 2, 1, 1, false>;
using Lin = cutlass::epilogue::thread::LinearCombination<float, 1, float, float>;
using Relu = cutlass::epilogue::thread::LinearCombinationRelu<float, 1, float, float>;

static double wall_ms() {
  timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

__global__ void bias_relu(const float *s1, const float *bias, float *hh, int n, int h) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) {
    float u = s1[t] + bias[t % h];
    hh[t] = u > 0.0f ? u : 0.0f;
  }
}
__global__ void residual(const float *s2, const float *x, float *y, int n) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) y[t] = s2[t] + x[t];
}

struct Bufs {
  float *x, *w1, *w2, *bias, *hh, *y, *s1, *s2;
};

struct Leg {
  std::string name;
  virtual void work(cudaStream_t s) = 0;
  virtual float *out() = 0;
  virtual ~Leg() {}
};

// Mapal: `<stem>.k` lines `kernel gx gy block smem r0 r1 r2 r3 r4` (roles x w1 w2 bias hh y 0).
struct MapalLeg : Leg {
  CUmodule mod;
  struct L { std::string k; unsigned gx, gy, block, smem; std::string role[5]; };
  std::vector<L> ls;
  std::map<std::string, CUdeviceptr> ptr;
  float *y;
  bool contract = false, reassoc = false;
  MapalLeg(const std::string &dir, const std::string &stem, const Bufs &b, float *own_y, float *own_hh) {
    name = stem;
    std::string face = stem.substr(0, stem.find('.'));
    auto ends = [&](const char *x) { size_t n = strlen(x); return face.size() >= n && face.compare(face.size() - n, n, x) == 0; };
    reassoc = ends("_r") || ends("_rc");
    contract = ends("_c") || ends("_rc");
    std::string path = dir + "/" + stem;
    FILE *f = fopen((path + ".ptx").c_str(), "rb");
    FILE *k = fopen((path + ".k").c_str(), "r");
    if (!f || !k) { fprintf(stderr, "missing %s.ptx/.k\n", path.c_str()); exit(102); }
    std::string ptx; char buf[1 << 16]; size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) ptx.append(buf, n);
    fclose(f);
    CU(cuModuleLoadData(&mod, ptx.c_str()));
    char kn[64], r[5][16];
    L l;
    while (fscanf(k, "%63s %u %u %u %u %15s %15s %15s %15s %15s", kn, &l.gx, &l.gy, &l.block, &l.smem, r[0], r[1], r[2], r[3], r[4]) == 10) {
      l.k = kn;
      for (int i = 0; i < 5; ++i) l.role[i] = r[i];
      ls.push_back(l);
    }
    fclose(k);
    y = own_y;
    ptr = {{"x", (CUdeviceptr)b.x}, {"w1", (CUdeviceptr)b.w1}, {"w2", (CUdeviceptr)b.w2}, {"bias", (CUdeviceptr)b.bias},
           {"hh", (CUdeviceptr)own_hh}, {"y", (CUdeviceptr)own_y}, {"0", 0}};
    for (auto &x : ls) {
      CUfunction fn;
      CU(cuModuleGetFunction(&fn, mod, x.k.c_str()));
      int regs = 0, local = 0;
      cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, fn);
      cuFuncGetAttribute(&local, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fn);
      printf("arm stem=%s kernel=%s grid=%ux%u block=%u smem=%u regs=%d local_bytes=%d args=%s,%s,%s,%s,%s\n", stem.c_str(),
             x.k.c_str(), x.gx, x.gy, x.block, x.smem, regs, local, x.role[0].c_str(), x.role[1].c_str(), x.role[2].c_str(),
             x.role[3].c_str(), x.role[4].c_str());
    }
  }
  void work(cudaStream_t s) override {
    for (auto &x : ls) {
      CUfunction fn;
      CU(cuModuleGetFunction(&fn, mod, x.k.c_str()));  // mapal_gpu.c: per launch
      if (x.smem > 48 * 1024) CU(cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)x.smem));
      CUdeviceptr d[5];
      void *args[5];
      for (int i = 0; i < 5; ++i) { d[i] = ptr.at(x.role[i]); args[i] = &d[i]; }
      CU(cuLaunchKernel(fn, x.gx, x.gy, 1, x.block, 1, 1, x.smem, (CUstream)s, args, 0));
    }
  }
  float *out() override { return y; }
};

struct CublasLeg : Leg {
  cublasHandle_t h; Bufs b; int M, d, hd;
  CublasLeg(const Bufs &b_, int M_, int d_, int h_) : b(b_), M(M_), d(d_), hd(h_) {
    name = "cublas";
    BLAS(cublasCreate(&h));
    BLAS(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));
  }
  void work(cudaStream_t s) override {
    const float one = 1, zero = 0;
    BLAS(cublasSetStream(h, s));
    BLAS(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, hd, M, d, &one, b.w1, hd, b.x, d, &zero, b.s1, hd));
    bias_relu<<<(M * hd + 255) / 256, 256, 0, s>>>(b.s1, b.bias, b.hh, M * hd, hd);
    BLAS(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, d, M, hd, &one, b.w2, d, b.hh, hd, &zero, b.s2, d));
    residual<<<(M * d + 255) / 256, 256, 0, s>>>(b.s2, b.x, b.y, M * d);
  }
  float *out() override { return b.y; }
};

// Row-major O(M×N) = L(M×K)·R(K×N) [+ C]: column-major O^T(N×M) = R^T(N×K)·L^T(K×M).
struct LtGemm {
  cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t a, b, c, d; cublasLtMatmulAlgo_t algo; size_t wsz;
  const float *L, *R, *C; float *D; float beta;
  LtGemm(cublasLtHandle_t lt, int M, int N, int K, const float *L_, const float *R_, const float *C_, float *D_,
         const float *bias, bool relu, size_t cap, const char *site) : L(L_), R(R_), C(C_ ? C_ : D_), D(D_), beta(C_ ? 1.f : 0.f) {
    BLAS(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasLtEpilogue_t epi = bias ? (relu ? CUBLASLT_EPILOGUE_RELU_BIAS : CUBLASLT_EPILOGUE_BIAS) : CUBLASLT_EPILOGUE_DEFAULT;
    BLAS(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof epi));
    if (bias) BLAS(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof bias));
    BLAS(cublasLtMatrixLayoutCreate(&a, CUDA_R_32F, N, K, N));
    BLAS(cublasLtMatrixLayoutCreate(&b, CUDA_R_32F, K, M, K));
    BLAS(cublasLtMatrixLayoutCreate(&c, CUDA_R_32F, N, M, N));
    BLAS(cublasLtMatrixLayoutCreate(&d, CUDA_R_32F, N, M, N));
    cublasLtMatmulPreference_t pref;
    BLAS(cublasLtMatmulPreferenceCreate(&pref));
    BLAS(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &cap, sizeof cap));
    cublasLtMatmulHeuristicResult_t res[8];
    int ret = 0;
    BLAS(cublasLtMatmulAlgoGetHeuristic(lt, op, a, b, c, d, pref, 8, res, &ret));
    cublasLtMatmulPreferenceDestroy(pref);
    if (!ret) { fprintf(stderr, "no Lt heuristic for %s\n", site); exit(102); }
    algo = res[0].algo;
    wsz = res[0].workspaceSize;
    uint32_t v[5] = {0}; size_t sz;
    cublasLtMatmulAlgoConfigAttributes_t at[5] = {CUBLASLT_ALGO_CONFIG_ID, CUBLASLT_ALGO_CONFIG_TILE_ID, CUBLASLT_ALGO_CONFIG_STAGES_ID,
                                                 CUBLASLT_ALGO_CONFIG_SPLITK_NUM, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME};
    for (int i = 0; i < 5; ++i) cublasLtMatmulAlgoConfigGetAttribute(&algo, at[i], &v[i], sizeof v[i], &sz);
    printf("lt site=%s api_mnk=%d,%d,%d epilogue=%d beta=%g heuristics=%d chosen=0 algo_id=%u tile=%u stages=%u splitk=%u reduction=%u workspace=%zu waves=%.3f\n",
           site, N, M, K, (int)epi, beta, ret, v[0], v[1], v[2], v[3], v[4], wsz, res[0].wavesCount);
  }
  void run(cublasLtHandle_t lt, void *ws, cudaStream_t s) {
    const float one = 1;
    BLAS(cublasLtMatmul(lt, op, &one, R, a, L, b, &beta, C, c, D, d, &algo, ws, wsz, s));
  }
};
struct LtLeg : Leg {
  cublasLtHandle_t lt; void *ws; LtGemm *g1, *g2; Bufs b;
  LtLeg(const Bufs &b_, int M, int d, int h) : b(b_) {
    name = "lt";
    const size_t cap = 32u << 20;
    BLAS(cublasLtCreate(&lt));
    CUDA(cudaMalloc(&ws, cap));
    g1 = new LtGemm(lt, M, h, d, b.x, b.w1, nullptr, b.hh, b.bias, true, cap, "up+bias+relu");
    g2 = new LtGemm(lt, M, d, h, b.hh, b.w2, b.x, b.y, nullptr, false, cap, "down+residual");
  }
  void work(cudaStream_t s) override { g1->run(lt, ws, s); g2->run(lt, ws, s); }
  float *out() override { return b.y; }
};

template <class G1, class G2>
struct CutlassLeg : Leg {
  G1 g1; G2 g2; Bufs b;
  CutlassLeg(const char *n, const Bufs &b_, int M, int d, int h) : b(b_) {
    name = n;
    typename G1::Arguments a1({M, h, d}, {b.x, d}, {b.w1, h}, {b.bias, 0}, {b.hh, h}, {1.0f, 1.0f});
    typename G2::Arguments a2({M, d, h}, {b.hh, h}, {b.w2, d}, {b.x, d}, {b.y, d}, {1.0f, 1.0f});
    if (g1.can_implement(a1) != cutlass::Status::kSuccess || g1.initialize(a1) != cutlass::Status::kSuccess ||
        g2.can_implement(a2) != cutlass::Status::kSuccess || g2.initialize(a2) != cutlass::Status::kSuccess) {
      fprintf(stderr, "%s cannot run\n", n);
      exit(102);
    }
    facts<G1>("up+bias+relu", M, h, d);
    facts<G2>("down+residual", M, d, h);
  }
  template <class G>
  void facts(const char *site, int M, int N, int K) {
    cudaFuncAttributes a{};
    CUDA(cudaFuncGetAttributes(&a, cutlass::Kernel<typename G::GemmKernel>));
    typename G::ThreadblockSwizzle sw;
    dim3 g = sw.get_grid_shape(sw.get_tiled_shape({M, N, K}, {G::ThreadblockShape::kM, G::ThreadblockShape::kN, G::ThreadblockShape::kK}, 1));
    printf("libfacts leg=%s site=%s tb=%dx%dx%d warp=%dx%dx%d threads=%d grid=%ux%ux%u dyn_smem=%zu regs=%d local_bytes=%zu static_smem=%zu\n",
           name.c_str(), site, G::ThreadblockShape::kM, G::ThreadblockShape::kN, G::ThreadblockShape::kK, G::WarpShape::kM,
           G::WarpShape::kN, G::WarpShape::kK, G::GemmKernel::kThreadCount, g.x, g.y, g.z,
           sizeof(typename G::GemmKernel::SharedStorage), a.numRegs, a.localSizeBytes, a.sharedSizeBytes);
  }
  void work(cudaStream_t s) override {
    if (g1(s) != cutlass::Status::kSuccess || g2(s) != cutlass::Status::kSuccess) { fprintf(stderr, "cutlass run failed\n"); exit(102); }
  }
  float *out() override { return b.y; }
};

int main(int argc, char **argv) {
  if (argc < 6) { fprintf(stderr, "usage: mlp_bench M d h reps dir stem...\n"); return 2; }
  const int M = atoi(argv[1]), d = atoi(argv[2]), h = atoi(argv[3]), reps = atoi(argv[4]);
  const std::string dir = argv[5];
  CUDA(cudaFree(0));
  cudaStream_t st;
  CUDA(cudaStreamCreate(&st));
  // Mapal's laws (gen_tiles.py OPS a/b/e, BIAS bias).
  auto law = [](int n, int mul, int add, int mod, int off, float div) {
    std::vector<float> v(n);
    for (int t = 0; t < n; ++t) v[t] = (float)((t * mul + add) % mod - off) / div;
    return v;
  };
  std::vector<float> X = law(M * d, 7, 13, 101, 50, 7.0f), W1 = law(d * h, 11, 57, 103, 51, 9.0f),
                     W2 = law(h * d, 5, 3, 97, 48, 11.0f), Bi(h);
  for (int t = 0; t < h; ++t) Bi[t] = (float)(t % 13 - 6) / 5.0f;
  auto up = [&](const std::vector<float> &v) { float *p; CUDA(cudaMalloc(&p, v.size() * 4)); CUDA(cudaMemcpy(p, v.data(), v.size() * 4, cudaMemcpyHostToDevice)); return p; };
  auto mk = [&](size_t n) { float *p; CUDA(cudaMalloc(&p, n * 4)); return p; };
  Bufs base{up(X), up(W1), up(W2), up(Bi), 0, 0, 0, 0};
  std::vector<Leg *> legs;
  for (int i = 6; i < argc; ++i) legs.push_back(new MapalLeg(dir, argv[i], base, mk((size_t)M * d), mk((size_t)M * h)));
  const size_t nm = legs.size();
  auto own = [&]() { Bufs b = base; b.hh = mk((size_t)M * h); b.y = mk((size_t)M * d); b.s1 = mk((size_t)M * h); b.s2 = mk((size_t)M * d); return b; };
  legs.push_back(new CublasLeg(own(), M, d, h));
  legs.push_back(new LtLeg(own(), M, d, h));
  legs.push_back(new CutlassLeg<Gemm<128, 128, 8, 32, 64, 8, Relu>, Gemm<128, 128, 8, 32, 64, 8, Lin>>("cutlass", own(), M, d, h));
  legs.push_back(new CutlassLeg<Gemm<16, 64, 16, 8, 32, 16, Relu>, Gemm<16, 64, 16, 8, 32, 16, Lin>>("cutlass_tuned", own(), M, d, h));
  const size_t nl = legs.size();
  CUDA(cudaDeviceSynchronize());

  // --- cold: each leg's first execution in this process
  bool rev = getenv("DIS_REVERSE") != nullptr;
  for (size_t x = 0; x < nl; ++x) {
    Leg *l = legs[rev ? nl - 1 - x : x];
    double t0 = wall_ms();
    l->work(st);
    CUDA(cudaStreamSynchronize(st));
    printf("cold leg=%s wall_ms=%.4f\n", l->name.c_str(), wall_ms() - t0);
  }

  // --- gates: CPU ordered folds (exact: mul then add; contract: fmaf) and f64, 256 sampled outputs
  std::vector<std::pair<int, int>> cells = {{0, 0}, {M - 1, d - 1}, {0, d - 1}, {M - 1, 0}};
  uint64_t cs = 0xce11;
  for (int i = 0; i < 252; ++i) {
    cs = cs * 6364136223846793005ull + 1;
    cells.push_back({(int)((cs >> 33) % M), (int)((cs >> 13) % d)});
  }
  std::map<int, std::vector<float>> hrow[2];  // [contract][row] = hh row in the face's ordered arithmetic
  std::map<int, std::vector<double>> hrow64;
  auto hh_row = [&](int i, int c) -> std::vector<float> & {
    auto &m = hrow[c];
    if (!m.count(i)) {
      std::vector<float> r(h);
      for (int j = 0; j < h; ++j) {
        float acc = 0.0f;
        for (int k = 0; k < d; ++k) {
          float a = X[(size_t)i * d + k], b = W1[(size_t)k * h + j];
          if (c) acc = fmaf(a, b, acc); else { volatile float p = a * b; acc = acc + p; }
        }
        float u = acc + Bi[j];
        r[j] = u > 0.0f ? u : 0.0f;
      }
      m[i] = r;
    }
    return m[i];
  };
  auto hh_row64 = [&](int i) -> std::vector<double> & {
    if (!hrow64.count(i)) {
      std::vector<double> r(h);
      for (int j = 0; j < h; ++j) {
        double acc = 0;
        for (int k = 0; k < d; ++k) acc += (double)X[(size_t)i * d + k] * W1[(size_t)k * h + j];
        double u = acc + Bi[j];
        r[j] = u > 0 ? u : 0;
      }
      hrow64[i] = r;
    }
    return hrow64[i];
  };
  bool all_ok = true;
  std::vector<float> Y((size_t)M * d);
  for (size_t li = 0; li < nl; ++li) {
    Leg *l = legs[li];
    CUDA(cudaMemset(l->out(), 0xA5, Y.size() * 4));
    l->work(st);
    CUDA(cudaStreamSynchronize(st));
    CUDA(cudaMemcpy(Y.data(), l->out(), Y.size() * 4, cudaMemcpyDeviceToHost));
    double maxrel = 0; int mism = 0;
    bool mapal = li < nm;
    int c = mapal ? ((MapalLeg *)l)->contract : 0;
    bool reassoc = mapal && ((MapalLeg *)l)->reassoc;
    for (auto [i, j] : cells) {
      auto &r64 = hh_row64(i);
      double ref = 0;
      for (int k = 0; k < h; ++k) ref += r64[k] * W2[(size_t)k * d + j];
      ref += X[(size_t)i * d + j];
      float got = Y[(size_t)i * d + j];
      maxrel = std::max(maxrel, std::fabs(got - ref) / std::max(std::fabs(ref), 1e-3));
      if (mapal) {
        auto &r = hh_row(i, c);
        float acc = 0.0f;
        for (int k = 0; k < h; ++k) {
          float a = r[k], b = W2[(size_t)k * d + j];
          if (c) acc = fmaf(a, b, acc); else { volatile float p = a * b; acc = acc + p; }
        }
        float want = acc + X[(size_t)i * d + j];
        if (memcmp(&got, &want, 4)) ++mism;
      }
    }
    bool ok = mapal && !reassoc ? mism == 0 : maxrel < 1e-3;
    all_ok &= ok;
    printf("values leg=%s cells=%zu maxrel_vs_f64=%.3e", l->name.c_str(), cells.size(), maxrel);
    if (mapal) printf(" bit_exact_vs_ordered=%s", mism ? "NO" : "yes");
    printf(" y0=%.9g ylast=%.9g%s\n", Y[0], Y[Y.size() - 1], ok ? "" : " GATE-FAILED");
  }
  if (getenv("DIS_PROFILE")) {
    CUDA(cudaProfilerStart());
    for (auto *l : legs) { l->work(st); CUDA(cudaDeviceSynchronize()); }
    CUDA(cudaProfilerStop());
    printf("profile_region legs=%zu all_ok=%d\n", nl, (int)all_ok);
    return all_ok ? 0 : 1;
  }
  if (!all_ok) { printf("GATE FAILED: no timing\n"); return 1; }

  // --- timing
  for (auto *l : legs) for (int w = 0; w < 5; ++w) l->work(st);
  CUDA(cudaStreamSynchronize(st));
  cudaEvent_t e0, e1;
  CUDA(cudaEventCreate(&e0));
  CUDA(cudaEventCreate(&e1));
  std::vector<std::vector<float>> ev(nl), wl(nl);
  for (int r = 0; r < reps; ++r)
    for (size_t x = 0; x < nl; ++x) {
      size_t li = (r & 1) ^ rev ? nl - 1 - x : x;
      double t0 = wall_ms();
      CUDA(cudaEventRecord(e0, st));
      legs[li]->work(st);
      CUDA(cudaEventRecord(e1, st));
      CUDA(cudaStreamSynchronize(st));
      double t1 = wall_ms();
      float e;
      CUDA(cudaEventElapsedTime(&e, e0, e1));
      ev[li].push_back(e);
      wl[li].push_back((float)(t1 - t0));
    }
  auto med = [](std::vector<float> v) { std::sort(v.begin(), v.end()); return v; };
  for (size_t li = 0; li < nl; ++li) {
    auto a = med(ev[li]), b = med(wl[li]);
    printf("time leg=%s M=%d d=%d h=%d n=%zu event_min_ms=%.4f event_med_ms=%.4f wall_min_ms=%.4f wall_med_ms=%.4f wall_max_ms=%.4f\n",
           legs[li]->name.c_str(), M, d, h, a.size(), a.front(), a[a.size() / 2], b.front(), b[b.size() / 2], b.back());
  }
  for (size_t li = 0; li < nl; ++li) {
    printf("samples leg=%s event=", legs[li]->name.c_str());
    for (size_t i = 0; i < ev[li].size(); ++i) printf("%s%.4f", i ? "," : "", ev[li][i]);
    printf(" wall=");
    for (size_t i = 0; i < wl[li].size(); ++i) printf("%s%.4f", i ? "," : "", wl[li][i]);
    printf("\n");
  }
  return 0;
}
