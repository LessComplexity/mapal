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

enum Fmt { BF16, F16, E4M3, E5M2 };

// One instruction per warp: A 16 x K row-major, B K x 8 (column n holds k),
// C / D 16 x 8, elements as raw bits (one byte or two each).
template <Fmt F>
__global__ void run(const uint8_t *A, const uint8_t *B, const float *C, float *D, int n) {
  const int E = (F == BF16 || F == F16) ? 2 : 4;  // elements per 32-bit register
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
  if (F == BF16)
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
  int bytes = (F == BF16 || F == F16) ? 2 : 1, n = (int)cs.size();
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

int main() {
  shape<BF16>("m16n8k16.bf16", 16);
  shape<F16>("m16n8k16.f16", 16);
  shape<E4M3>("m16n8k32.e4m3", 32);
  shape<E5M2>("m16n8k32.e5m2", 32);
  return 0;
}
