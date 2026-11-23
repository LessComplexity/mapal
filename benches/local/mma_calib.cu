// ADR-0043 D4: the measured accumulator of the warp tensor-core products
// (`Caps::mma_acc`). Products of narrow operands are exact in f32; what a part
// may round is the accumulation `d = c + sum_k a_k b_k` of one instruction.
// For each shape the part runs (`mma.sync.m16n8k16` bf16 / f16, `m16n8k32`
// e4m3 / e5m2, all `.f32` accumulate):
//   1. probes of the final rounding: c = 1 plus one exact product whose exact
//      sum lies between two f32 values (round to nearest vs toward zero);
//   2. probes of the alignment: addends below the largest one's last bit
//      (kept, as an exact sum would, or dropped before they are added);
//   3. 2^20 random instructions (c and the operands over 2^-8 .. 2^8, both
//      signs): the worst |d - exact| in units of 2^-24 * (|c| + sum |a b|).
//   4. (f32-tensor-cores) the alignment guard bits `g` (`MatrixOp::acc_g`):
//      c = 1 plus K addends of 2^-(23+j) each (j = 1, 2, ...): an addend is kept
//      iff it lies at or above 2^(e_max - 23 - g), so d - 1 = K 2^-(23+j) while
//      j <= g and 0 past it (K >= 2^j keeps the kept sum one ulp or more: TF32
//      probes g <= 3, the 16-bit shapes <= 4, fp8 <= 5); `g0_j` the same with
//      c = 0 and one product 1 (the products aligned among themselves), `gneg_j`
//      negative addends (truncation of magnitudes or toward -inf). `g` is the
//      deepest j kept by every variant.
//   5. subnormals: a product below the least normal f32 (`sub_prod`), a
//      subnormal operand (`sub_op`), a subnormal c (`sub_c`): kept or flushed.
//   6. split realizations of an f32 product (`f32-tensor-cores` design §2.3):
//      per instruction, the kept part products summed smallest class first from
//      a zero accumulator, then one round-to-nearest FADD onto c (`split`), and
//      the same products chained onto c itself (`chain`, ADR-0041 "Later") —
//      tf32x3 (hi / lo TF32 parts, 3 products) and bf16x6 (three bf16 parts,
//      i + j <= 2), f32 operands random over 2^-8..2^8 and with cancellation:
//      the worst |d - exact| in units of 2^-24 (|c| + sum |a b|) (exact of the
//      f32 operands) — the realized e + C_1 of the design's coefficient.
// Each line reads `mma-acc <shape> <name> <value>`; `units` (the worst
// relative error, rounded up, in units of 2^-24) is the property.
// Build: nvcc -O3 -arch=sm_120 benches/local/mma_calib.cu -o mma_calib
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

enum Fmt { BF16, F16, E4M3, E5M2, TF32 };

// One instruction per warp: A 16 x K row-major, B K x 8 (column n holds k),
// C / D 16 x 8, elements as raw bits (one byte or two each).
template <Fmt F>
__global__ void run(const uint8_t *A, const uint8_t *B, const float *C, float *D, int n) {
  const int E = (F == TF32) ? 1 : (F == BF16 || F == F16) ? 2 : 4;  // elements per 32-bit register
  const int K = 8 * E;
  const int bytes = 4 / E;
  int w = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (w >= n) return;
  int lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
  const uint8_t *a = A + (size_t)w * 16 * K * bytes;
  const uint8_t *b = B + (size_t)w * K * 8 * bytes;
  const float *c = C + (size_t)w * 128;
  float *d = D + (size_t)w * 128;
  uint32_t ar[4], br[2];
  for (int r = 0; r < 4; r++) {
    int row = g + 8 * (r & 1), k0 = E * t + (K / 2) * (r >> 1);
    uint32_t v = 0;
    memcpy(&v, a + (row * K + k0) * bytes, 4);
    ar[r] = v;
  }
  for (int r = 0; r < 2; r++) {
    uint32_t v = 0;
    for (int e = 0; e < E; e++) {
      int k = E * t + (K / 2) * r + e;
      uint32_t x = 0;
      memcpy(&x, b + (g * K + k) * bytes, bytes);  // B stored n-major: (n, k)
      v |= x << (8 * bytes * e);
    }
    br[r] = v;
  }
  float c0 = c[g * 8 + 2 * t], c1 = c[g * 8 + 2 * t + 1];
  float c2 = c[(g + 8) * 8 + 2 * t], c3 = c[(g + 8) * 8 + 2 * t + 1];
  float d0, d1, d2, d3;
  if (F == TF32)
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(ar[0]), "r"(ar[1]), "r"(ar[2]), "r"(ar[3]), "r"(br[0]), "r"(br[1]), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
  else if (F == BF16)
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(ar[0]), "r"(ar[1]), "r"(ar[2]), "r"(ar[3]), "r"(br[0]), "r"(br[1]), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
  else if (F == F16)
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(ar[0]), "r"(ar[1]), "r"(ar[2]), "r"(ar[3]), "r"(br[0]), "r"(br[1]), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
  else if (F == E4M3)
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(ar[0]), "r"(ar[1]), "r"(ar[2]), "r"(ar[3]), "r"(br[0]), "r"(br[1]), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
  else
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(ar[0]), "r"(ar[1]), "r"(ar[2]), "r"(ar[3]), "r"(br[0]), "r"(br[1]), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
  d[g * 8 + 2 * t] = d0;
  d[g * 8 + 2 * t + 1] = d1;
  d[(g + 8) * 8 + 2 * t] = d2;
  d[(g + 8) * 8 + 2 * t + 1] = d3;
}

// Host encode / decode of each format (exact for representable values).
static uint32_t enc(Fmt f, float x) {
  switch (f) {
    case TF32: {
      uint32_t b;
      memcpy(&b, &x, 4);
      return (b + 0x1000u) & 0xffffe000u;  // `cvt.rna.tf32.f32` of a finite value
    }
    case BF16: {
      uint32_t b;
      memcpy(&b, &x, 4);
      return b >> 16;  // callers pass bf16-representable values
    }
    case F16: {
      __half h = __float2half_rn(x);
      uint16_t b;
      memcpy(&b, &h, 2);
      return b;
    }
    case E4M3:
      return __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
    default:
      return __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E5M2);
  }
}
static float dec(Fmt f, uint32_t b) {
  switch (f) {
    case TF32: {
      float x;
      memcpy(&x, &b, 4);
      return x;
    }
    case BF16: {
      uint32_t w = b << 16;
      float x;
      memcpy(&x, &w, 4);
      return x;
    }
    case F16: {
      uint16_t h = (uint16_t)b;
      __half v;
      memcpy(&v, &h, 2);
      return __half2float(v);
    }
    default: {
      __half_raw r = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, f == E4M3 ? __NV_E4M3 : __NV_E5M2);
      __half v;
      memcpy(&v, &r, 2);
      return __half2float(v);
    }
  }
}

struct Case {
  std::vector<float> a, b, c;  // a: 16 x K, b: n-major 8 x K, c: 16 x 8
};

template <Fmt F>
static std::vector<float> launch(const std::vector<Case> &cs, int K) {
  int bytes = (F == TF32) ? 4 : (F == BF16 || F == F16) ? 2 : 1, n = (int)cs.size();
  std::vector<uint8_t> A((size_t)n * 16 * K * bytes), B((size_t)n * 8 * K * bytes);
  std::vector<float> C((size_t)n * 128), D((size_t)n * 128);
  for (int w = 0; w < n; w++) {
    for (int i = 0; i < 16 * K; i++) {
      uint32_t v = enc(F, cs[w].a[i]);
      memcpy(&A[((size_t)w * 16 * K + i) * bytes], &v, bytes);
    }
    for (int i = 0; i < 8 * K; i++) {
      uint32_t v = enc(F, cs[w].b[i]);
      memcpy(&B[((size_t)w * 8 * K + i) * bytes], &v, bytes);
    }
    memcpy(&C[(size_t)w * 128], cs[w].c.data(), 128 * 4);
  }
  uint8_t *dA, *dB;
  float *dC, *dD;
  cudaMalloc(&dA, A.size());
  cudaMalloc(&dB, B.size());
  cudaMalloc(&dC, C.size() * 4);
  cudaMalloc(&dD, D.size() * 4);
  cudaMemcpy(dA, A.data(), A.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B.data(), B.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dC, C.data(), C.size() * 4, cudaMemcpyHostToDevice);
  run<F><<<(n + 3) / 4, 128>>>(dA, dB, dC, dD, n);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    fprintf(stderr, "launch: %s\n", cudaGetErrorString(e));
    exit(1);
  }
  cudaMemcpy(D.data(), dD, D.size() * 4, cudaMemcpyDeviceToHost);
  cudaFree(dA), cudaFree(dB), cudaFree(dC), cudaFree(dD);
  return D;
}

static Case zero(int K) { return Case{std::vector<float>(16 * K), std::vector<float>(8 * K), std::vector<float>(128)}; }

// d[0][0] of a case whose row 0 of A and column 0 of B carry `prods` (each
// product p as a * b, both near sqrt(|p|) so both are normal in the format)
// and c[0][0] = c.
static Case probe(Fmt f, int K, float c, std::vector<float> prods) {
  (void)f;
  Case x = zero(K);
  x.c[0] = c;
  for (size_t i = 0; i < prods.size(); i++) {
    float s = ldexpf(1.0f, ilogbf(fabsf(prods[i])) / 2);
    x.a[i] = prods[i] / s;
    x.b[i] = s;
  }
  return x;
}

template <Fmt F>
static void shape(const char *name, int K) {
  Fmt f = F;
  // c = 1 (16-bit operands) or 2^12 (fp8: its products reach 2^-18 at the
  // least normal operands), u its unit roundoff in f32.
  float one = (f == E4M3 || f == E5M2) ? 4096.0f : 1.0f, u = ldexpf(one, -24);
  // 1. the final rounding: 1 + 1.5 u (RN -> 1 + 2u, RZ -> 1), 1 - u/2
  //    (exactly between 1 - u and 1: RN-even -> 1, RZ -> 1 - u).
  // 2. the alignment: 1 + three addends of u/2 (exact 1 + 1.5u: kept -> 1 + 2u
  //    under RN; dropped -> 1).
  std::vector<std::pair<const char *, Case>> probes = {
      {"rn_up", probe(f, K, one, {1.5f * u})},
      {"rn_down", probe(f, K, one, {-0.5f * u})},
      {"align", probe(f, K, one, {0.5f * u, 0.5f * u, 0.5f * u})},
      {"align_c0", probe(f, K, 0.0f, {one, 0.5f * u, 0.5f * u, 0.5f * u})},
  };
  std::vector<Case> cs;
  for (auto &p : probes) cs.push_back(p.second);
  auto d = launch<F>(cs, K);
  for (size_t i = 0; i < probes.size(); i++) {
    double exact = probes[i].second.c[0];
    for (int k = 0; k < K; k++) exact += (double)dec(f, enc(f, probes[i].second.a[k])) * dec(f, enc(f, probes[i].second.b[k]));
    printf("mma-acc %s %s got %.9g exact %.12g (c + %+.3f u)\n", name, probes[i].first, d[i * 128], exact,
           (d[i * 128] - one) / u);
  }
  // 3. random instructions.
  srand(12345);
  int n = 1 << 16;
  double worst_m = 0;
  for (int rep = 0; rep < 16; rep++) {
    std::vector<Case> rc(n, zero(K));
    for (auto &x : rc) {
      auto r = [&](float lo_exp, float hi_exp) {
        float m = 1.0f + (rand() & 0xffff) / 65536.0f;
        float e = lo_exp + (rand() % (int)(hi_exp - lo_exp + 1));
        float v = ldexpf(m, (int)e) * ((rand() & 1) ? -1.0f : 1.0f);
        return dec(f, enc(f, v));  // the format's value
      };
      float lo = (f == E4M3) ? -6 : (f == E5M2) ? -8 : -8, hi = (f == E4M3) ? 7 : 8;
      for (auto &v : x.a) v = r(lo, hi);
      for (auto &v : x.b) v = r(lo, hi);
      for (auto &v : x.c) v = ldexpf(1.0f + (rand() & 0xffff) / 65536.0f, rand() % 17 - 8) * ((rand() & 1) ? -1.f : 1.f);
    }
    auto dd = launch<F>(rc, K);
    for (int w = 0; w < n; w++)
      for (int i = 0; i < 16; i++)
        for (int j = 0; j < 8; j++) {
          double exact = rc[w].c[i * 8 + j], m = fabs(exact);
          for (int k = 0; k < K; k++) {
            double p = (double)rc[w].a[i * K + k] * rc[w].b[j * K + k];
            exact += p;
            m += fabs(p);
          }
          double got = dd[(size_t)w * 128 + i * 8 + j], err = fabs(got - exact);
          worst_m = fmax(worst_m, err / (m * ldexp(1.0, -24)));
        }
  }
  printf("mma-acc %s worst_rel_M %.4f units_of_2^-24\n", name, worst_m);
  printf("mma-acc %s units %d\n", name, (int)ceil(worst_m));
}

// Round an exact sum toward zero to f32 (the measured final rounding).
static float rz(double s) {
  float d = (float)s;
  if (std::isfinite(d) && fabs((double)d) > fabs(s)) d = nextafterf(d, 0.0f);
  return d;
}

// A case whose row 0 of A and column 0 of B carry the pairs (a_k, b_k) as given.
static Case probe2(int K, float c, std::vector<std::pair<float, float>> ab) {
  Case x = zero(K);
  x.c[0] = c;
  for (size_t i = 0; i < ab.size(); i++) x.a[i] = ab[i].first, x.b[i] = ab[i].second;
  return x;
}

template <Fmt F>
static double exact00(const Case &x, int K) {
  double e = x.c[0];
  for (int k = 0; k < K; k++) e += (double)dec(F, enc(F, x.a[k])) * dec(F, enc(F, x.b[k]));
  return e;
}

// 4. the alignment guard bits; 5. subnormals.
template <Fmt F>
static void guards(const char *name, int K) {
  Fmt f = F;
  float one = (f == E4M3 || f == E5M2) ? 4096.0f : 1.0f, u = ldexpf(one, -24);
  int jmax = 0;
  while ((2 << jmax) <= K) jmax++;  // K >= 2^j keeps the kept sum one ulp or more
  struct P {
    const char *tag;
    int j;
    Case c;
    float dropped;
  };
  std::vector<P> ps;
  for (int j = 1; j <= jmax + 1; j++) {
    float a = ldexpf(one, -(23 + j));
    ps.push_back({"g", j, probe(f, K, one, std::vector<float>(K, a)), one});
    std::vector<float> v(K, a);
    v[0] = one;
    ps.push_back({"g0", j, probe(f, K, 0.0f, v), one});
    ps.push_back({"gneg", j, probe(f, K, one, std::vector<float>(K, -a)), one});
  }
  std::vector<Case> cs;
  for (auto &p : ps) cs.push_back(p.c);
  auto d = launch<F>(cs, K);
  int g = 0;
  bool open = true;
  for (int j = 1; j <= jmax + 1; j++) {
    bool all = true, any = false;
    for (size_t i = 0; i < ps.size(); i++) {
      if (ps[i].j != j) continue;
      double ex = exact00<F>(ps[i].c, K);
      float kept = rz(ex), got = d[i * 128];
      const char *v = kept == ps[i].dropped ? "n/a" : got == kept ? "kept" : got == ps[i].dropped ? "dropped" : "other";
      printf("mma-acc %s %s_%d got %.9g exact %.12g (c + %+.3f u) %s\n", name, ps[i].tag, j, got, ex, (got - one) / u, v);
      if (kept != ps[i].dropped) {
        any = true;
        all = all && got == kept;
      }
    }
    if (open && any && all && j <= jmax) g = j;
    if (any && !all) open = false;
  }
  printf("mma-acc %s g %d%s\n", name, g, open ? " (every probe kept: g >= this, the probe's reach)" : "");
  // 5. subnormals.
  std::vector<std::pair<const char *, Case>> sp;
  if (f == TF32 || f == BF16) {
    sp.push_back({"sub_prod", probe2(K, 0.0f, {{ldexpf(1, -65), ldexpf(1, -65)}})});
    sp.push_back({"sub_op", probe2(K, 0.0f, {{ldexpf(1, -127), ldexpf(1, 10)}})});
  } else if (f == F16) {
    sp.push_back({"sub_op", probe2(K, 0.0f, {{ldexpf(1, -20), ldexpf(1, 4)}})});
  } else if (f == E4M3) {
    sp.push_back({"sub_op", probe2(K, 0.0f, {{ldexpf(1, -8), 1.0f}})});
  } else {
    sp.push_back({"sub_op", probe2(K, 0.0f, {{ldexpf(1, -15), 1.0f}})});
  }
  sp.push_back({"sub_c", probe2(K, ldexpf(1, -130), {})});
  cs.clear();
  for (auto &p : sp) cs.push_back(p.second);
  d = launch<F>(cs, K);
  for (size_t i = 0; i < sp.size(); i++) {
    double ex = exact00<F>(sp[i].second, K);
    float got = d[i * 128];
    printf("mma-acc %s %s got %.9g exact %.9g %s\n", name, sp[i].first, got, ex,
           got == (float)ex ? "kept" : got == 0.0f ? "flushed" : "other");
  }
}

// 6. split realizations of f32 products.
enum Form { X3_TF32, X6_BF16 };

__device__ __forceinline__ uint32_t tf32_rna(float x) {
  uint32_t r;
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ void mma_tf32(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// The three bf16 parts of x: rn(x), rn of the residual, rn of what is left (exact above 2^-110).
__device__ __forceinline__ void bf16_parts(float x, uint16_t (&p)[3]) {
  float r = x;
  for (int i = 0; i < 3; i++) {
    __nv_bfloat16 h = __float2bfloat16_rn(r);
    memcpy(&p[i], &h, 2);
    r = r - __bfloat162float(h);
  }
}

// One instruction tile per warp: A 16 x K, B n-major 8 x K, C 16 x 8, all f32.
// `chain`: the part products accumulate onto c itself; else from zero, then one RN add onto c.
template <Form S>
__global__ void split_k(const float *A, const float *B, const float *C, float *D, int n, int chain) {
  const int K = (S == X3_TF32) ? 8 : 16;
  int w = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (w >= n) return;
  int lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
  const float *a = A + (size_t)w * 16 * K, *b = B + (size_t)w * 8 * K, *c = C + (size_t)w * 128;
  float *d = D + (size_t)w * 128;
  float cv[4] = {c[g * 8 + 2 * t], c[g * 8 + 2 * t + 1], c[(g + 8) * 8 + 2 * t], c[(g + 8) * 8 + 2 * t + 1]};
  float acc[4];
  for (int q = 0; q < 4; q++) acc[q] = chain ? cv[q] : 0.0f;
  if (S == X3_TF32) {
    uint32_t ah[4], al[4], bh[2], bl[2];
    for (int r = 0; r < 4; r++) {
      float x = a[(g + 8 * (r & 1)) * K + t + 4 * (r >> 1)];
      ah[r] = tf32_rna(x);
      al[r] = tf32_rna(x - __uint_as_float(ah[r]));
    }
    for (int r = 0; r < 2; r++) {
      float x = b[g * K + t + 4 * r];
      bh[r] = tf32_rna(x);
      bl[r] = tf32_rna(x - __uint_as_float(bh[r]));
    }
    mma_tf32(acc, al, bh);
    mma_tf32(acc, ah, bl);
    mma_tf32(acc, ah, bh);
  } else {
    uint32_t ap[3][4], bp[3][2];
    for (int r = 0; r < 4; r++) {
      int row = g + 8 * (r & 1), k0 = 2 * t + 8 * (r >> 1);
      uint16_t lo[3], hi[3];
      bf16_parts(a[row * K + k0], lo);
      bf16_parts(a[row * K + k0 + 1], hi);
      for (int i = 0; i < 3; i++) ap[i][r] = (uint32_t)lo[i] | (uint32_t)hi[i] << 16;
    }
    for (int r = 0; r < 2; r++) {
      int k0 = 2 * t + 8 * r;
      uint16_t lo[3], hi[3];
      bf16_parts(b[g * K + k0], lo);
      bf16_parts(b[g * K + k0 + 1], hi);
      for (int i = 0; i < 3; i++) bp[i][r] = (uint32_t)lo[i] | (uint32_t)hi[i] << 16;
    }
    // i + j <= 2, smallest class first: (1,1), (0,2), (2,0), (0,1), (1,0), (0,0).
    const int order[6][2] = {{1, 1}, {0, 2}, {2, 0}, {0, 1}, {1, 0}, {0, 0}};
    for (int o = 0; o < 6; o++) mma_bf16(acc, ap[order[o][0]], bp[order[o][1]]);
  }
  for (int q = 0; q < 4; q++)
    if (!chain) acc[q] = __fadd_rn(cv[q], acc[q]);
  d[g * 8 + 2 * t] = acc[0];
  d[g * 8 + 2 * t + 1] = acc[1];
  d[(g + 8) * 8 + 2 * t] = acc[2];
  d[(g + 8) * 8 + 2 * t + 1] = acc[3];
}

static uint32_t lcg_state = 12345;
static uint32_t rnd() {
  lcg_state = lcg_state * 1664525u + 1013904223u;
  return lcg_state >> 8;
}
// A random f32 with a full 23-bit significand, |x| in [2^lo, 2^(hi+1)), either sign.
static float rf(int lo, int hi) {
  float m = 1.0f + (float)(rnd() & 0x7fffff) * ldexpf(1.0f, -23);
  return ldexpf(m, lo + (int)(rnd() % (unsigned)(hi - lo + 1))) * ((rnd() & 1) ? -1.0f : 1.0f);
}

template <Form S>
static void split(const char *name) {
  const int K = (S == X3_TF32) ? 8 : 16;
  const int n = 1 << 15;
  float *dA, *dB, *dC, *dD;
  cudaMalloc(&dA, (size_t)n * 16 * K * 4);
  cudaMalloc(&dB, (size_t)n * 8 * K * 4);
  cudaMalloc(&dC, (size_t)n * 128 * 4);
  cudaMalloc(&dD, (size_t)n * 128 * 4);
  std::vector<float> A((size_t)n * 16 * K), B((size_t)n * 8 * K), C((size_t)n * 128), D((size_t)n * 128);
  const char *dists[] = {"uniform", "wide", "cancel"};
  for (int dist = 0; dist < 3; dist++) {
    double worst[2] = {0, 0};
    for (int rep = 0; rep < 8; rep++) {
      for (int w = 0; w < n; w++) {
        float *a = &A[(size_t)w * 16 * K], *b = &B[(size_t)w * 8 * K], *c = &C[(size_t)w * 128];
        for (int i = 0; i < 16 * K; i++) a[i] = dist == 1 ? rf(-30, 30) : rf(-8, 8);
        for (int i = 0; i < 8 * K; i++) b[i] = dist == 1 ? rf(-30, 30) : rf(-8, 8);
        if (dist == 2)  // pairs that nearly cancel: a[2i+1] = -a[2i], b[2i+1] = b[2i] (1 + small)
          for (int i = 0; i < 16; i++)
            for (int k = 0; k + 1 < K; k += 2) a[i * K + k + 1] = -a[i * K + k];
        if (dist == 2)
          for (int j = 0; j < 8; j++)
            for (int k = 0; k + 1 < K; k += 2)
              b[j * K + k + 1] = b[j * K + k] * (1.0f + (float)(rnd() % 64) * ldexpf(1.0f, -20));
        for (int i = 0; i < 128; i++) c[i] = dist == 2 ? rf(-20, -12) : dist == 1 ? rf(-30, 30) : rf(-8, 8);
      }
      cudaMemcpy(dA, A.data(), A.size() * 4, cudaMemcpyHostToDevice);
      cudaMemcpy(dB, B.data(), B.size() * 4, cudaMemcpyHostToDevice);
      cudaMemcpy(dC, C.data(), C.size() * 4, cudaMemcpyHostToDevice);
      for (int chain = 0; chain < 2; chain++) {
        split_k<S><<<(n + 3) / 4, 128>>>(dA, dB, dC, dD, n, chain);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
          fprintf(stderr, "split launch: %s\n", cudaGetErrorString(e));
          exit(1);
        }
        cudaMemcpy(D.data(), dD, D.size() * 4, cudaMemcpyDeviceToHost);
        for (int w = 0; w < n; w++)
          for (int i = 0; i < 16; i++)
            for (int j = 0; j < 8; j++) {
              double ex = C[(size_t)w * 128 + i * 8 + j], m = fabs(ex);
              for (int k = 0; k < K; k++) {
                double p = (double)A[(size_t)w * 16 * K + i * K + k] * B[(size_t)w * 8 * K + j * K + k];
                ex += p;
                m += fabs(p);
              }
              double err = fabs((double)D[(size_t)w * 128 + i * 8 + j] - ex);
              worst[chain] = fmax(worst[chain], err / (m * ldexp(1.0, -24)));
            }
      }
    }
    printf("mma-acc %s %s split_worst_rel_M %.4f chain_worst_rel_M %.4f units_of_2^-24\n", name, dists[dist], worst[0],
           worst[1]);
  }
  cudaFree(dA), cudaFree(dB), cudaFree(dC), cudaFree(dD);
}

int main() {
  shape<BF16>("m16n8k16.bf16", 16);
  shape<F16>("m16n8k16.f16", 16);
  shape<E4M3>("m16n8k32.e4m3", 32);
  shape<E5M2>("m16n8k32.e5m2", 32);
  shape<TF32>("m16n8k8.tf32", 8);
  guards<TF32>("m16n8k8.tf32", 8);
  guards<BF16>("m16n8k16.bf16", 16);
  guards<F16>("m16n8k16.f16", 16);
  guards<E4M3>("m16n8k32.e4m3", 32);
  guards<E5M2>("m16n8k32.e5m2", 32);
  split<X3_TF32>("split.tf32x3");
  split<X6_BF16>("split.bf16x6");
  return 0;
}
