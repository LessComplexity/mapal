// integration: how much of a cyclic read stream larger than the L2 the L2 keeps — the bandwidth of re-reading a
// working set of `ws` bytes back to back (each pass one kernel, as a program's repeated calls), ws / L2 at the given
// ratios (the first is a warm-up). `l2_keep_bytes` (profile.rs `DeviceCost`, `l2_keep_of_sweep`) = the largest ws
// re-read at no less than the midpoint of the sweep's plateau (its highest rate: this stream's L2 rate, which on the
// RTX 3090 is 1.2 TB/s, not the derived 2.4) and the DRAM rate (RTX 5080: 1.025 x the 64 MiB L2; RTX 3090: 1.03 x).
// build: nvcc -O2 -arch=sm_120 l2keep.cu -o l2keep; run: ./l2keep
#include <cstdio>
#include <cuda_runtime.h>
__global__ void rd(const float4 *p, size_t n, float *out) {
  float4 s = make_float4(0, 0, 0, 0);
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    float4 v = __ldcg(p + i); s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
  }
  if (s.x + s.y + s.z + s.w == 12345.0f) *out = s.x;
}
int main() {
  int l2 = 0, sms = 0; cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0); cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  size_t cap = (size_t)l2 * 4; char *buf; float *out; cudaMalloc(&buf, cap); cudaMalloc(&out, 4); cudaMemset(buf, 0, cap);
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  // bugs-properties: from a quarter of the L2 — the 3090 / A100 / H100 re-read a 1.0 × L2 stream below the
  // midpoint (5fcb401b), so their keep lies below the old sweep's first point.
  double r[] = {0.25, 0.375, 0.5, 0.625, 0.75, 0.8125, 0.875, 0.9375, 1.0, 1.005, 1.01, 1.015, 1.02, 1.025, 1.03, 1.035, 1.04, 1.045, 1.05};
  printf("l2_bytes=%d sms=%d\n", l2, sms);
  for (double x : r) {
    size_t ws = (size_t)(x * l2) / 16 * 16, n = ws / 16;
    for (int i = 0; i < 5; i++) rd<<<sms * 8, 512>>>((const float4 *)buf, n, out);
    cudaEventRecord(a);
    for (int i = 0; i < 20; i++) rd<<<sms * 8, 512>>>((const float4 *)buf, n, out);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms, a, b);
    printf("ws/l2=%.3f ws_mb=%.1f gbs=%.0f\n", x, ws / 1048576.0, 20.0 * ws / (ms * 1e6));
  }
  return 0;
}
