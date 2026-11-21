// ADR-0042 D5: the measured properties behind the device's hardware `exp`.
//   1. `ex2.approx.ftz.f32` over every f32 in [-1/2, 1/2] against the f64 exp2:
//      worst relative error -> `Caps::ex2` (units of 2^-24, rounded up).
//   2. The hardware form (Cody-Waite reduction, ex2.approx, the oracle's scaling
//      and specials) over every f32 in [lo, hi] against the f64 exp, beside CUDA's
//      expf and __expf: worst error in ulps of the exact result.
// Build: nvcc -O3 -arch=sm_120 benches/local/exp_calib.cu -o exp_calib
#include <cmath>
#include <cstdint>
#include <cstdio>

__device__ float ex2a(float a) {
  float r;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(a));
  return r;
}

__device__ float hw(float x) {
  const float HI = __int_as_float(0x42b17217), LO = __int_as_float(0xc2cff1b4);
  const float L2E = __int_as_float(0x3fb8aa3b), SH = 12582912.0f;
  const float H = __int_as_float(0x3f317200), L = __int_as_float(0x35bfbe8e);
  float xc = x > HI ? HI : x;
  xc = xc < LO ? LO : xc;
  xc = (xc == xc) ? xc : 0.0f;
  float t = __fmaf_rn(xc, L2E, SH);
  float kf = __fsub_rn(t, SH);
  int k = (int)kf;
  float r = __fmaf_rn(-kf, H, xc);
  r = __fmaf_rn(-kf, L, r);
  float p = ex2a(__fmul_rn(r, L2E));
  int k1 = k >> 1, k2 = k - k1;
  float y = __fmul_rn(__fmul_rn(p, __int_as_float((k1 + 127) << 23)), __int_as_float((k2 + 127) << 23));
  if (x > HI) y = INFINITY;
  if (x < LO) y = 0.0f;
  if (x != x) y = x;
  return y;
}

__device__ unsigned long long worst[5];

__global__ void sweep_ex2(uint32_t base) {
  float a = __uint_as_float(base + blockIdx.x * blockDim.x + threadIdx.x);
  if (!(a >= -0.5f && a <= 0.5f)) return;
  double r = exp2((double)a), rel = fabs((double)ex2a(a) - r) / r;
  atomicMax(&worst[0], (unsigned long long)(rel * 1e15));
}

__global__ void sweep_exp(uint32_t base) {
  float x = __uint_as_float(base + blockIdx.x * blockDim.x + threadIdx.x);
  const float HI = __int_as_float(0x42b17217), LO = __int_as_float(0xc2cff1b4);
  if (!(x >= LO && x <= HI)) return;
  double r = exp((double)x);
  double ulp = r < 1.1754943508222875e-38 ? ldexp(1.0, -149) : ldexp(1.0, ilogb(r) - 23);
  float g[3] = {hw(x), expf(x), __expf(x)};
  for (int v = 0; v < 3; v++) atomicMax(&worst[1 + v], (unsigned long long)(fabs((double)g[v] - r) / ulp * 1e6));
}

int main() {
  for (uint64_t b = 0; b < (1ull << 32); b += (1ull << 30)) {
    sweep_ex2<<<(1u << 30) / 256, 256>>>((uint32_t)b);
    sweep_exp<<<(1u << 30) / 256, 256>>>((uint32_t)b);
  }
  cudaDeviceSynchronize();
  unsigned long long h[5];
  cudaMemcpyFromSymbol(h, worst, sizeof h);
  double rel = h[0] / 1e15;
  printf("ex2.approx.ftz.f32 on [-1/2, 1/2]: worst relative 2^%.3f = %.3f units of 2^-24 -> Caps::ex2 = %d\n",
         log2(rel), rel * 16777216.0, (int)ceil(rel * 16777216.0));
  printf("exp over every f32 in [lo, hi], worst ulp: hardware form %.4f  expf %.4f  __expf %.4f\n", h[1] / 1e6,
         h[2] / 1e6, h[3] / 1e6);
  return 0;
}
