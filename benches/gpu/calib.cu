// plan-cost calibration (terms.md open item 4): the device facts site_ns reads.
//   chase   dependent ld.global.cg latency (ns) vs footprint, 1 / 8 / 32 warps per SM (lane 0 of each warp chases)
//   bw      achieved read GB/s vs active SMs x warps/SM x 16 B loads in flight per thread, DRAM (1 GiB) and L2 (8 MiB, re-read)
//   loaded  dependent-load latency of one warp while the other SMs stream DRAM (nstream SMs; achieved GB/s printed)
//   lds     ld.shared.b32 / v2 / v4 conflict-free throughput, bytes per SM clock (clock64) and GB/s per SM
//   gsync   kernel-merging `calls`: one grid barrier of a co-resident grid (the emitted barrier: block barrier, one
//           arrival per block on a flip-bit counter, acquire spin; `fenced` = the fence + relaxed form) vs one dependent
//           launch of the same grid (an empty kernel back to back on one stream, device-bound behind a long kernel; `pdl`:
//           launched as a programmatic dependent launch, each kernel opening with `griddepcontrol.wait`, as the runtime
//           launches tile kernels on sm_90+) vs one device-side tail launch (CDP2 `cudaStreamTailLaunch`, a comparison
//           only), ns each, by grid and block
// usage: calib [chase|bw|loaded|lds|gsync ...]   (all when none).  Output: `<kind> k=v ...` lines, one per point.
// build: nvcc -O3 -arch=sm_<cc> -std=c++17 -rdc=true calib.cu -o calib -lcudadevrt   (rdc for the tail launch)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <algorithm>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA error %s at %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)
static int sms, clk_khz;

__device__ __forceinline__ unsigned ldcg(const unsigned* p) { unsigned v; asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p)); return v; }
__device__ __forceinline__ uint4 ldcg4(const uint4* p) { uint4 v; asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)); return v; }

// one dependent chain per warp (lane 0): next[] holds word indices, one element per 128 B line
__global__ void chase(const unsigned* next, unsigned lines, int iters, unsigned* sink, long long* cyc) {
    if (threadIdx.x % 32) return;
    unsigned p = ((blockIdx.x * 64u + threadIdx.x / 32 * 7919u) % lines) * 32u; // start line varies per warp
    long long t0 = clock64();
    for (int i = 0; i < iters; i++) p = ldcg(next + p);
    long long t1 = clock64();
    if (threadIdx.x == 0) { cyc[blockIdx.x] = t1 - t0; sink[blockIdx.x] = p; }
}
static void chase_run() {
    size_t maxb = (size_t)1 << 30; unsigned *d, *sink; long long* cyc;
    CK(cudaMalloc(&d, maxb)); CK(cudaMalloc(&sink, 4096 * 4)); CK(cudaMalloc(&cyc, 4096 * 8));
    std::mt19937 rng(7);
    for (size_t bytes : {2u << 10, 8u << 10, 32u << 10, 128u << 10, 512u << 10, 2u << 20, 8u << 20, 32u << 20, 64u << 20, 128u << 20, 256u << 20, 1u << 30}) {
        size_t lines = bytes / 128; std::vector<unsigned> perm(lines), h(bytes / 4);
        std::iota(perm.begin(), perm.end(), 0); std::shuffle(perm.begin() + 1, perm.end(), rng); // one cycle through every line
        for (size_t i = 0; i < lines; i++) h[perm[i] * 32] = perm[(i + 1) % lines] * 32;
        CK(cudaMemcpy(d, h.data(), bytes, cudaMemcpyHostToDevice));
        for (int wps : {1, 8, 32}) {
            int iters = bytes <= (8u << 20) ? 20000 : 4000;
            for (int rep = 0; rep < 2; rep++) { // rep 0 warms (fills L2), rep 1 reported
                chase<<<sms, wps * 32>>>(d, (unsigned)lines, iters, sink, cyc); CK(cudaDeviceSynchronize());
            }
            std::vector<long long> c(sms); CK(cudaMemcpy(c.data(), cyc, sms * 8, cudaMemcpyDeviceToHost));
            double cy = 0; for (auto x : c) cy += x; cy /= sms;
            printf("chase bytes=%zu wps=%d ns=%.1f cycles=%.1f\n", bytes, wps, cy / iters * 1e6 / clk_khz, cy / iters);
        }
    }
}

// each thread keeps L independent 16 B loads in flight; grid = active SMs blocks of wps warps; each block sweeps its own slice
__global__ void bw(const uint4* buf, size_t n16, int L, int rounds, unsigned* sink) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x, nt = (size_t)gridDim.x * blockDim.x;
    unsigned acc = 0; size_t base = tid;
    for (int r = 0; r < rounds; r++) {
        for (size_t i = base; i + (size_t)(L - 1) * nt < n16; i += (size_t)L * nt) {
            uint4 v[8];
#pragma unroll
            for (int j = 0; j < 8; j++) if (j < L) v[j] = ldcg4(buf + i + (size_t)j * nt);
#pragma unroll
            for (int j = 0; j < 8; j++) if (j < L) acc ^= v[j].x ^ v[j].y ^ v[j].z ^ v[j].w;
        }
    }
    if (acc == 0x12345678) sink[0] = acc;
}
static void bw_run() {
    size_t maxb = (size_t)1 << 30; uint4* d; unsigned* sink;
    CK(cudaMalloc(&d, maxb)); CK(cudaMemset(d, 1, maxb)); CK(cudaMalloc(&sink, 4));
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int lvl = 0; lvl < 2; lvl++) {
        size_t bytes = lvl ? (8u << 20) : maxb; const char* nm = lvl ? "l2" : "dram"; int rounds = lvl ? 200 : 2;
        for (int s : {1, std::max(1, sms / 16), std::max(1, sms / 8), std::max(1, sms / 4), std::max(1, sms / 2), sms})
            for (int wps : {1, 2, 4, 8, 16, 32})
                for (int L : {1, 2, 4, 8}) {
                    bw<<<s, wps * 32>>>(d, bytes / 16, L, 1, sink); CK(cudaDeviceSynchronize()); // warm
                    cudaEventRecord(a); bw<<<s, wps * 32>>>(d, bytes / 16, L, rounds, sink); cudaEventRecord(b); CK(cudaEventSynchronize(b));
                    float ms; cudaEventElapsedTime(&ms, a, b);
                    double bytes_read = (double)bytes * rounds; // a tail of < L*nt elements per round is skipped: scale
                    size_t nt = (size_t)s * wps * 32; size_t per = (size_t)L * nt, n16 = bytes / 16;
                    bytes_read = (double)(n16 / per * per) * 16 * rounds;
                    printf("bw level=%s sms=%d wps=%d L=%d inflight_kb=%.1f gbs=%.1f\n", nm, s, wps, L, (double)s * wps * 32 * L * 16 / 1024, bytes_read / ms / 1e6);
                }
    }
}

// one chaser warp (block 0) with nstream streaming blocks; streamers run until the chaser raises the flag
__global__ void loaded(const unsigned* next, const uint4* buf, size_t n16, int iters, volatile int* flag, unsigned* sink, long long* out, unsigned long long* streamed) {
    if (blockIdx.x == 0) {
        if (threadIdx.x) return;
        unsigned p = 0; long long t0 = clock64();
        for (int i = 0; i < iters; i++) p = ldcg(next + p);
        out[0] = clock64() - t0; sink[0] = p; *flag = 1; __threadfence_system();
        return;
    }
    size_t tid = (size_t)(blockIdx.x - 1) * blockDim.x + threadIdx.x, nt = (size_t)(gridDim.x - 1) * blockDim.x; unsigned acc = 0; unsigned long long cnt = 0;
    while (!*flag) {
        for (size_t i = tid; i + 7 * nt < n16 && !*flag; i += 8 * nt) {
            uint4 v[8];
#pragma unroll
            for (int j = 0; j < 8; j++) v[j] = ldcg4(buf + i + (size_t)j * nt);
#pragma unroll
            for (int j = 0; j < 8; j++) acc ^= v[j].x ^ v[j].w;
            cnt += 8;
        }
    }
    if (acc == 0x12345678) sink[1] = acc;
    atomicAdd(streamed, cnt * 16);
}
static void loaded_run() {
    size_t bytes = (size_t)1 << 30, lines = bytes / 128; unsigned *dn, *sink; uint4* buf; long long* out; unsigned long long* st; int* flag;
    CK(cudaMalloc(&dn, bytes)); CK(cudaMalloc(&buf, bytes)); CK(cudaMemset(buf, 1, bytes)); CK(cudaMalloc(&sink, 16)); CK(cudaMalloc(&out, 8));
    CK(cudaMalloc(&st, 8)); CK(cudaMalloc(&flag, 4));
    std::vector<unsigned> perm(lines), h(bytes / 4); std::mt19937 rng(9);
    std::iota(perm.begin(), perm.end(), 0); std::shuffle(perm.begin() + 1, perm.end(), rng);
    for (size_t i = 0; i < lines; i++) h[perm[i] * 32] = perm[(i + 1) % lines] * 32;
    CK(cudaMemcpy(dn, h.data(), bytes, cudaMemcpyHostToDevice));
    for (int ns : {0, std::max(1, sms / 16), std::max(1, sms / 8), std::max(1, sms / 4), std::max(1, sms / 2), sms * 3 / 4, sms - 1}) {
        int iters = 4000; CK(cudaMemset(st, 0, 8)); CK(cudaMemset(flag, 0, 4));
        loaded<<<1 + ns, 1024>>>(dn, buf, bytes / 16, iters, flag, sink, out, st); CK(cudaDeviceSynchronize());
        long long c; unsigned long long s; CK(cudaMemcpy(&c, out, 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&s, st, 8, cudaMemcpyDeviceToHost));
        double ns_t = (double)c / iters * 1e6 / clk_khz, secs = (double)c / (clk_khz * 1e3);
        printf("loaded nstream=%d ns=%.1f stream_gbs=%.1f\n", ns, ns_t, s / secs / 1e9);
    }
}

template <int W> __global__ void lds(unsigned* out, long long* cyc, int iters) {
    extern __shared__ unsigned sm[]; // 32 KB used
    for (int i = threadIdx.x; i < 8192; i += blockDim.x) sm[i] = i;
    __syncthreads();
    unsigned acc = 0; unsigned a = (threadIdx.x * W * 4) & 32767;
    long long t0 = clock64();
    for (int i = 0; i < iters; i++) {
#pragma unroll
        for (int j = 0; j < 8; j++) {
            unsigned ad = (a + j * 4096 * 0 + ((j * blockDim.x * W * 4) & 32767)) & 32767;
            unsigned x, y, z, w;
            if (W == 1) { asm volatile("ld.shared.b32 %0, [%1];" : "=r"(x) : "r"(ad)); acc ^= x; }
            if (W == 2) { asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];" : "=r"(x), "=r"(y) : "r"(ad)); acc ^= x ^ y; }
            if (W == 4) { asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(x), "=r"(y), "=r"(z), "=r"(w) : "r"(ad)); acc ^= x ^ y ^ z ^ w; }
        }
        a = (a + 36) & 32767 & ~(W * 4 - 1);
    }
    long long t1 = clock64();
    __syncthreads();
    if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
    if (acc == 0x12345678) out[0] = acc;
}
template <int W> static void lds_one() {
    unsigned* o; long long* c; CK(cudaMalloc(&o, 4)); CK(cudaMalloc(&c, 4096 * 8));
    for (int thr : {256, 1024}) {
        int iters = 20000;
        lds<W><<<sms, thr, 32768>>>(o, c, iters); CK(cudaDeviceSynchronize());
        lds<W><<<sms, thr, 32768>>>(o, c, iters); CK(cudaDeviceSynchronize());
        std::vector<long long> h(sms); CK(cudaMemcpy(h.data(), c, sms * 8, cudaMemcpyDeviceToHost));
        double cy = 0; for (auto x : h) cy += x; cy /= sms;
        double bytes = (double)thr * W * 4 * 8 * iters; // per block (one block per SM)
        printf("lds width=%d bits thr=%d bytes_per_clk=%.1f gbs_per_sm=%.1f\n", W * 32, thr, bytes / cy, bytes / (cy / (clk_khz * 1e3)) / 1e9);
    }
}

// kernel-merging `calls`: the grid barrier `func/calls.rs` emits (`@mapal_gsync`), in CUDA. Block 0 adds
// 0x80000000 - (nb - 1), every other block 1: the counter's top bit flips once all nb arrived. The low 31 bits start
// at zero (a zeroed counter) and every complete barrier restores them, so a later barrier (or launch) of any grid
// starts from a flip bit and zeros; a block reading the flip bit before the last arrival is impossible because the
// partial sums borrow from it.
template <int V> __device__ __forceinline__ void gsync(unsigned* ctr, unsigned nb) {
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned inc = blockIdx.x == 0 ? 0x80000000u - (nb - 1) : 1u, old, cur;
        if (V == 0) {
            __threadfence();
            asm volatile("atom.add.relaxed.gpu.global.u32 %0, [%1], %2;" : "=r"(old) : "l"(ctr), "r"(inc) : "memory");
            do { asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(cur) : "l"(ctr) : "memory"); } while (((old ^ cur) & 0x80000000u) == 0);
            __threadfence();
        } else {
            asm volatile("atom.add.release.gpu.global.u32 %0, [%1], %2;" : "=r"(old) : "l"(ctr), "r"(inc) : "memory");
            do { asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(cur) : "l"(ctr) : "memory"); } while (((old ^ cur) & 0x80000000u) == 0);
        }
    }
    __syncthreads();
}
template <int V> __global__ void gsync_k(unsigned* ctr, int n, unsigned long long* t) {
    unsigned long long t0, t1;
    if (blockIdx.x == 0 && threadIdx.x == 0) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    for (int i = 0; i < n; i++) gsync<V>(ctr, gridDim.x);
    if (blockIdx.x == 0 && threadIdx.x == 0) { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1)); t[0] = t1 - t0; }
}
__global__ void empty_k(unsigned long long* t, int i) {
    if (blockIdx.x == 0 && threadIdx.x == 0) { unsigned long long g; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g)); t[i] = g; }
}
// the same, as a programmatic dependent launch: it may start while the one before drains, and waits for it here
__global__ void pdl_k(unsigned long long* t, int i) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (blockIdx.x == 0 && threadIdx.x == 0) { unsigned long long g; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g)); t[i] = g; }
    asm volatile("griddepcontrol.launch_dependents;");
}
__global__ void spin_k(long long cycles) { long long s = clock64(); while (clock64() - s < cycles) {} }
// device-side launch, a comparison only: each kernel stamps its start and tail-launches the next of the same grid
__global__ void tail_k(unsigned long long* t, int i, int n) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long g; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g)); t[i] = g;
        if (i < n) tail_k<<<gridDim, blockDim, 0, cudaStreamTailLaunch>>>(t, i + 1, n);
    }
}
static void gsync_run() {
    unsigned* ctr; unsigned long long* t; CK(cudaMalloc(&ctr, 4)); CK(cudaMemset(ctr, 0, 4));
    const int nk = 512; CK(cudaMalloc(&t, (nk + 1) * 8));
    for (int block : {128, 256, 512}) {
        int per = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, gsync_k<1>, block, 0));
        for (int grid : {1, 2, 4, 8, 16, 32, 42, 84, 168, 252, 336, 504, 1008}) {
            if (grid > per * sms) continue;
            const int n = 4000; std::vector<double> v, f;
            for (int rep = 0; rep < 5; rep++) {
                int nn = n; void* a[] = {&ctr, &nn, &t};
                CK(cudaLaunchCooperativeKernel((void*)gsync_k<1>, grid, block, a, 0, 0)); CK(cudaDeviceSynchronize());
                unsigned long long h; CK(cudaMemcpy(&h, t, 8, cudaMemcpyDeviceToHost)); v.push_back((double)h / n);
                CK(cudaLaunchCooperativeKernel((void*)gsync_k<0>, grid, block, a, 0, 0)); CK(cudaDeviceSynchronize());
                CK(cudaMemcpy(&h, t, 8, cudaMemcpyDeviceToHost)); f.push_back((double)h / n);
            }
            std::sort(v.begin(), v.end()); std::sort(f.begin(), f.end());
            // the same grid as dependent launches: device-bound behind a 2 ms spin, start to start
            std::vector<double> w;
            for (int rep = 0; rep < 5; rep++) {
                spin_k<<<1, 32>>>((long long)clk_khz * 2);
                for (int i = 0; i <= nk; i++) empty_k<<<grid, block>>>(t, i);
                CK(cudaDeviceSynchronize());
                std::vector<unsigned long long> h(nk + 1); CK(cudaMemcpy(h.data(), t, (nk + 1) * 8, cudaMemcpyDeviceToHost));
                w.push_back((double)(h[nk] - h[0]) / nk);
            }
            std::sort(w.begin(), w.end());
            std::vector<double> q;
            for (int rep = 0; rep < 5; rep++) {
                spin_k<<<1, 32>>>((long long)clk_khz * 2);
                for (int i = 0; i <= nk; i++) {
                    cudaLaunchAttribute at[1];
                    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
                    at[0].val.programmaticStreamSerializationAllowed = 1;
                    cudaLaunchConfig_t cfg = {};
                    cfg.gridDim = grid; cfg.blockDim = block; cfg.attrs = at; cfg.numAttrs = 1;
                    CK(cudaLaunchKernelEx(&cfg, pdl_k, t, i));
                }
                CK(cudaDeviceSynchronize());
                std::vector<unsigned long long> h(nk + 1); CK(cudaMemcpy(h.data(), t, (nk + 1) * 8, cudaMemcpyDeviceToHost));
                q.push_back((double)(h[nk] - h[0]) / nk);
            }
            std::sort(q.begin(), q.end());
            std::vector<double> d;
            for (int rep = 0; rep < 5; rep++) {
                tail_k<<<grid, block>>>(t, 0, nk);
                CK(cudaDeviceSynchronize());
                std::vector<unsigned long long> h(nk + 1); CK(cudaMemcpy(h.data(), t, (nk + 1) * 8, cudaMemcpyDeviceToHost));
                d.push_back((double)(h[nk] - h[0]) / nk);
            }
            std::sort(d.begin(), d.end());
            printf("gsync block=%d grid=%d per_sm=%d barrier_ns=%.1f [%.1f-%.1f] fenced_ns=%.1f launch_ns=%.1f [%.1f-%.1f] "
                   "pdl_ns=%.1f [%.1f-%.1f] tail_ns=%.1f [%.1f-%.1f]\n", block, grid, per, v[2], v[0], v[4], f[2], w[2], w[0],
                   w[4], q[2], q[0], q[4], d[2], d[0], d[4]);
        }
    }
}
int main(int argc, char** argv) {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0)); sms = p.multiProcessorCount;
    CK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0));
    int mclk = 0; // CUDA 13 dropped cudaDeviceProp::memoryClockRate; the attribute stays
    CK(cudaDeviceGetAttribute(&mclk, cudaDevAttrMemoryClockRate, 0));
    printf("calib device=%s sms=%d clock_khz=%d l2_mb=%d mem_clock_khz=%d bus_bits=%d\n", p.name, sms, clk_khz, p.l2CacheSize >> 20, mclk, p.memoryBusWidth);
    std::string want; for (int i = 1; i < argc; i++) want += std::string(argv[i]) + " ";
    auto on = [&](const char* k) { return want.empty() || want.find(k) != std::string::npos; };
    if (on("chase")) chase_run();
    if (on("bw")) bw_run();
    if (on("loaded")) loaded_run();
    if (on("lds")) { lds_one<1>(); lds_one<2>(); lds_one<4>(); }
    if (on("gsync")) gsync_run();
    return 0;
}
