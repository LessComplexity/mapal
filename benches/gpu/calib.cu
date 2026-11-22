// plan-cost calibration (terms.md open item 4): the device facts site_ns reads.
//   chase   dependent ld.global.cg latency (ns) vs footprint, 1 / 8 / 32 warps per SM (lane 0 of each warp chases)
//   bw      achieved read GB/s vs active SMs x warps/SM x 16 B loads in flight per thread, DRAM (1 GiB) and L2 (8 MiB, re-read)
//   loaded  dependent-load latency of one warp while the other SMs stream DRAM (nstream SMs; achieved GB/s printed)
//   lds     ld.shared.b32 / v2 / v4 conflict-free throughput, bytes per SM clock (clock64) and GB/s per SM
//   gsync   kernel-merging `calls`: one grid barrier of a co-resident grid (the emitted barrier: block barrier, one
//           arrival per block on a flip-bit counter, acquire spin; `fenced` = the fence + relaxed form) vs one dependent
//           launch of the same grid (an empty kernel back to back on one stream, device-bound behind a long kernel; `pdl`:
//           launched as a programmatic dependent launch, each kernel opening with `griddepcontrol.wait`, as the runtime
//           launches tile kernels on sm_90+; `coop`: a cooperative launch, as a persistent kernel is launched) vs one
//           device-side tail launch (CDP2 `cudaStreamTailLaunch`, a comparison
//           only), ns each, by grid and block
//   copy    hardware-properties `pinned`: host->device copy rate from pageable vs page-locked host memory, by size
//           (`cudaMemcpy`, the runtime's synchronous copy, median of 9 after 2 warm), and their ratio per mille — the
//           profile's `pageable_pm` is the median ratio at >= 4 MiB (the sizes a loop's arrays cross)
//   ffma    the FFMA peak under load (flops per ns: 8 independent chains a thread, 8 blocks of 256 a SM, best of 5)
//           and the SM clock it implies at the compute capability's FP32 lanes per SM (`fp32_lanes_of_cc`: 64 on
//           sm_60 / sm_80, else 128); then **sustained**: the same kernel back to back for 4 s, its rate per 100 ms
//           window — `sustained_clock_khz` (median of the last second's windows: the clock a power or thermal limit
//           holds under continuous load, the profile's `clock_khz`) and `settle_ms` (the first window from which every
//           window is within 2 % of it: how long a measurement must load the part before its clock is the sustained one)
//   head    `pdl`: a short chain right after a host synchronize (a tiny kernel, then `n` = 1 / 2 / 4 kernels of
//           sms x 256 threads, host-timed from the first launch to the synchronize, median of 400, PDL and plain
//           interleaved) — `pdl_head_ns`: what a chain launched programmatically costs beyond a plain one at its head,
//           where the host issue, not the device, is exposed (the n = 1 difference; positive = PDL slower)
//   body    kernel-merging `calls`: what a chain's member bodies cost inside one persistent kernel against their own
//           launches — `coop_body_cycles`. The chain: level 0 two one-block row folds side by side (decode's rows: 256
//           threads, k = 256 over a 256 x 256 matrix, 8 loads in flight a thread), then levels of one wide fold each
//           (`nb` blocks, a thread one output over 8 loads in flight), every level reading the one before. Separate:
//           one launch a level, `__restrict__` parameters (Mapal's launch kernels: noalias, the read-only path).
//           Persistent (`calls.rs persistent_kernel`): one cooperative launch, the bodies inlined over their blocks —
//           the first block straight-line, the rest in a grid-stride loop — a grid barrier between levels, parameters
//           without `__restrict__`. Shapes: every level within one round of the persistent grid (`rounds=1`, 2 / 3 / 4
//           levels; the wide levels 1.5 x sms blocks, decode's 128 on 84) and 3 levels whose wide levels take 2 / 4
//           rounds of the grid (the grid-stride form: `nb` = rounds x the persistent wave). Each arm runs 129 chains
//           back to back, device-bound behind a 10 ms spin, with work and with the bodies skipped (a runtime flag: the
//           same code), ABBA-rotated, 15 reps: per chain, start to start and its span (the first block's start to the
//           last block's end — no launch of its own, so not the cooperative launch's dispatch, which starts on a
//           2.048 us grid on the RTX 5080). body_ns = (pers - pers_empty) - (sep - sep_empty) of the spans (the empty
//           arms are the boundaries the rule prices: the barriers against the launches between levels; body_ss_ns the
//           same of start to start, with the launches), median [quartiles]; body_cycles at the sustained clock (`ffma`,
//           run first when not selected); run_khz the SM clock during the chains (clock64 over globaltimer, block 0).
//           The `coop_body_cycles` line is the 4-round chain's: the grid-stride form, the bodies' largest change here.
// The `calib device=` line carries the device query's physical facts the rung rules read (`DeviceQuery`).
// usage: calib [chase|bw|loaded|lds|gsync|copy|ffma|head|body ...]   (all when none).  Output: `<kind> k=v ...` lines, one per point.
// build: nvcc -O3 -arch=sm_<cc> -std=c++17 -rdc=true calib.cu -o calib -lcudadevrt   (rdc for the tail launch)
#include <chrono>
#include <cmath>
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
static double sus_khz;  // `ffma`'s sustained clock, once measured (`body` converts ns to cycles at it)

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
    double one[2] = {0, 0}; // cross-arch rules: what one SM streams alone, best point, per level
    // gemv-stream `stream-fly`: per level, the in-flight bytes a grid on every SM keeps per SM at each point, and
    // its rate — the knee is the smallest point at or above 90 % of the level's best rate.
    std::vector<std::pair<double, double>> all[2];
    for (int lvl = 0; lvl < 2; lvl++) {
        size_t bytes = lvl ? (8u << 20) : maxb; const char* nm = lvl ? "l2" : "dram"; int rounds = lvl ? 200 : 2;
        for (int s : {1, std::max(1, sms / 16), std::max(1, sms / 8), std::max(1, sms / 4), std::max(1, sms / 2), sms})
            for (int wps : {1, 2, 4, 8, 16, 32})
                for (int L : {1, 2, 4, 8}) {
                    bw<<<s, wps * 32>>>(d, bytes / 16, L, 1, sink); // warm
                    // A block the registers cannot hold (1024 threads at L = 8 on sm_86 / sm_89) never runs: its
                    // events would time an empty stream (cross-arch: 2^20 GB/s points). Say so instead.
                    if (cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
                        printf("bw level=%s sms=%d wps=%d L=%d launch_failed=%s\n", nm, s, wps, L, cudaGetErrorName(e));
                        continue;
                    }
                    CK(cudaDeviceSynchronize());
                    cudaEventRecord(a); bw<<<s, wps * 32>>>(d, bytes / 16, L, rounds, sink); cudaEventRecord(b); CK(cudaEventSynchronize(b));
                    float ms; cudaEventElapsedTime(&ms, a, b);
                    double bytes_read = (double)bytes * rounds; // a tail of < L*nt elements per round is skipped: scale
                    size_t nt = (size_t)s * wps * 32; size_t per = (size_t)L * nt, n16 = bytes / 16;
                    bytes_read = (double)(n16 / per * per) * 16 * rounds;
                    printf("bw level=%s sms=%d wps=%d L=%d inflight_kb=%.1f gbs=%.1f\n", nm, s, wps, L, (double)s * wps * 32 * L * 16 / 1024, bytes_read / ms / 1e6);
                    if (s == 1) one[lvl] = std::max(one[lvl], bytes_read / ms / 1e6);
                    if (s == sms) all[lvl].push_back({(double)wps * 32 * L * 16, bytes_read / ms / 1e6});
                }
    }
    // `sm_bytes_per_ns` (the profile's): one SM streaming an L2-resident operand alone — a one-row pass's rate
    printf("bw one_sm dram_gbs=%.1f l2_gbs=%.1f sm_bytes_per_ns=%.0f\n", one[0], one[1], one[1]);
    // `fly_bytes_per_sm` (the profile's, DRAM): the smallest in-flight bytes an SM at which a grid on every SM
    // reads at 90 % of the best rate measured at that level.
    double knee[2] = {0, 0};
    for (int lvl = 0; lvl < 2; lvl++) {
        double best = 0;
        for (auto& p : all[lvl]) best = std::max(best, p.second);
        for (auto& p : all[lvl])
            if (p.second >= 0.9 * best && (knee[lvl] == 0 || p.first < knee[lvl])) knee[lvl] = p.first;
    }
    printf("bw fly dram_bytes_per_sm=%.0f l2_bytes_per_sm=%.0f fly_bytes_per_sm=%.0f\n", knee[0], knee[1], knee[0]);
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
// (sm_90+ only: below, `griddepcontrol` does not exist and the leg is an ordinary launch, printed pdl_ns=0)
__global__ void pdl_k(unsigned long long* t, int i) {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    if (blockIdx.x == 0 && threadIdx.x == 0) { unsigned long long g; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g)); t[i] = g; }
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;");
#endif
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
    int major = 0; CK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0));
    const int pdl_ok = major >= 9;  // programmatic dependent launch: sm_90+
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
            for (int rep = 0; rep < 5 * pdl_ok; rep++) {
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
            if (q.empty()) q.assign(5, 0.0);
            std::sort(q.begin(), q.end());
            std::vector<double> c;
            for (int rep = 0; rep < 5; rep++) {
                spin_k<<<1, 32>>>((long long)clk_khz * 2);
                for (int i = 0; i <= nk; i++) {
                    void* a[] = {&t, &i};
                    CK(cudaLaunchCooperativeKernel((void*)empty_k, grid, block, a, 0, 0));
                }
                CK(cudaDeviceSynchronize());
                std::vector<unsigned long long> h(nk + 1); CK(cudaMemcpy(h.data(), t, (nk + 1) * 8, cudaMemcpyDeviceToHost));
                c.push_back((double)(h[nk] - h[0]) / nk);
            }
            std::sort(c.begin(), c.end());
            std::vector<double> d;
            for (int rep = 0; rep < 5; rep++) {
                tail_k<<<grid, block>>>(t, 0, nk);
                CK(cudaDeviceSynchronize());
                std::vector<unsigned long long> h(nk + 1); CK(cudaMemcpy(h.data(), t, (nk + 1) * 8, cudaMemcpyDeviceToHost));
                d.push_back((double)(h[nk] - h[0]) / nk);
            }
            std::sort(d.begin(), d.end());
            printf("gsync block=%d grid=%d per_sm=%d barrier_ns=%.1f [%.1f-%.1f] fenced_ns=%.1f launch_ns=%.1f [%.1f-%.1f] "
                   "pdl_ns=%.1f [%.1f-%.1f] coop_ns=%.1f [%.1f-%.1f] tail_ns=%.1f [%.1f-%.1f]\n", block, grid, per, v[2], v[0],
                   v[4], f[2], w[2], w[0], w[4], q[2], q[0], q[4], c[2], c[0], c[4], d[2], d[0], d[4]);
        }
    }
}
// hardware-properties `pinned`: one H2D copy of `bytes` from `src`, median of 9 after 2 warm, in seconds.
static double h2d_s(void* dst, const void* src, size_t bytes) {
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    std::vector<float> t;
    for (int r = 0; r < 11; r++) {
        CK(cudaEventRecord(a)); CK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice)); CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b)); float ms; CK(cudaEventElapsedTime(&ms, a, b));
        if (r >= 2) t.push_back(ms);
    }
    std::sort(t.begin(), t.end());
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    return t[t.size() / 2] / 1e3;
}
static void copy_run() {
    std::vector<double> ratios;
    for (size_t mb : {1, 4, 16, 64, 256}) {
        size_t bytes = mb << 20;
        void* d; CK(cudaMalloc(&d, bytes));
        // Pageable: ordinary heap memory, touched first (no first-touch fault inside the copy).
        char* h = (char*)malloc(bytes); memset(h, 1, bytes);
        double pg = h2d_s(d, h, bytes);
        CK(cudaHostRegister(h, bytes, cudaHostRegisterDefault));
        double pn = h2d_s(d, h, bytes);
        CK(cudaHostUnregister(h)); free(h); CK(cudaFree(d));
        double r = pn / pg;  // pageable rate / pinned rate
        if (mb >= 4) ratios.push_back(r);
        printf("copy mb=%zu pageable_gbs=%.2f pinned_gbs=%.2f ratio_pm=%.0f\n", mb, bytes / pg / 1e9, bytes / pn / 1e9, r * 1e3);
    }
    std::sort(ratios.begin(), ratios.end());
    printf("copy pageable_pm=%.0f (median ratio at >= 4 MiB)\n", ratios[ratios.size() / 2] * 1e3);
}
// ffma: 8 independent FMA chains a thread; `a`, `b` and the store guard come from the host (nothing folds).
__global__ void ffma_k(float* out, int iters, float a, float b, int store) {
    float x[8];
    for (int j = 0; j < 8; j++) x[j] = threadIdx.x * 1e-3f + j;
    for (int i = 0; i < iters; i++) {
#pragma unroll
        for (int j = 0; j < 8; j++) x[j] = fmaf(x[j], a, b);
    }
    float acc = 0.f;
    for (int j = 0; j < 8; j++) acc += x[j];
    if (store || acc == 1234.5f) out[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
static void ffma_run(int cc) {
    const int block = 256, grid = sms * 8, iters = 1 << 16;
    float* out; CK(cudaMalloc(&out, (size_t)grid * block * sizeof(float)));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    ffma_k<<<grid, block>>>(out, iters / 16, 0.999f, 1e-4f, 0); CK(cudaDeviceSynchronize());
    double best = 0;
    for (int rep = 0; rep < 5; rep++) {
        CK(cudaEventRecord(e0)); ffma_k<<<grid, block>>>(out, iters, 0.999f, 1e-4f, 0); CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1)); float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
        best = std::max(best, 2.0 * 8 * iters * (double)grid * block / (ms * 1e6));
    }
    const int lanes = (cc == 60 || cc == 80) ? 64 : 128;
    printf("ffma flops_per_ns=%.0f lanes=%d implied_clock_khz=%.0f\n", best, lanes, best / (2.0 * sms * lanes) * 1e6);
    // sustained: back to back for 4 s; one rate per 100 ms window (whole launches, events around each window)
    const double flops = 2.0 * 8 * iters * (double)grid * block;
    std::vector<double> win;
    for (double total = 0; total < 4000;) {
        float ms = 0; int n = 0;
        CK(cudaEventRecord(e0));
        for (; ms < 100; n++) {
            ffma_k<<<grid, block>>>(out, iters, 0.999f, 1e-4f, 0);
            if (n % 4 == 3) { CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&ms, e0, e1)); }
        }
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&ms, e0, e1));
        win.push_back(flops * n / (ms * 1e6)); total += ms;
    }
    std::vector<double> last(win.end() - std::min<size_t>(10, win.size()), win.end());
    std::sort(last.begin(), last.end());
    const double sus = last[last.size() / 2];
    sus_khz = sus / (2.0 * sms * lanes) * 1e6;
    size_t settle = win.size();
    while (settle > 0 && std::fabs(win[settle - 1] / sus - 1) < 0.02) settle--;
    printf("ffma sustained_flops_per_ns=%.0f sustained_clock_khz=%.0f settle_ms=%zu windows=%zu first=%.0f\n", sus,
           sus / (2.0 * sms * lanes) * 1e6, settle * 100, win.size(), win[0]);
    CK(cudaFree(out));
}
// head: a tiny kernel, then n kernels of sms x 256 threads, right after a synchronize; host wall to the next synchronize
__global__ void tiny_k(float* o) { if (threadIdx.x == 0) o[blockIdx.x] += 1.f; }
__global__ void work_k(float* o, int pdl) {
#if __CUDA_ARCH__ >= 900
    if (pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
    o[blockIdx.x * blockDim.x + threadIdx.x] += 1.f;
}
static void head_run() {
    int major = 0; CK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0));
    if (major < 9) { printf("head pdl_head_ns=0 (no griddepcontrol)\n"); return; }
    float* o; CK(cudaMalloc(&o, (size_t)sms * 256 * 4)); CK(cudaMemset(o, 0, (size_t)sms * 256 * 4));
    for (int n : {1, 2, 4}) {
        std::vector<double> t[2];
        for (int rep = 0; rep < 420; rep++) {
            for (int pdl = 0; pdl < 2; pdl++) {
                CK(cudaDeviceSynchronize());
                auto a = std::chrono::steady_clock::now();
                tiny_k<<<1, 32>>>(o);
                for (int i = 0; i < n; i++) {
                    cudaLaunchAttribute at[1];
                    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
                    at[0].val.programmaticStreamSerializationAllowed = pdl;
                    cudaLaunchConfig_t cfg = {};
                    cfg.gridDim = sms; cfg.blockDim = 256; cfg.attrs = at; cfg.numAttrs = 1;
                    CK(cudaLaunchKernelEx(&cfg, work_k, o, pdl));
                }
                CK(cudaDeviceSynchronize());
                if (rep >= 20) t[pdl].push_back(std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - a).count());
            }
        }
        for (auto& v : t) std::sort(v.begin(), v.end());
        printf("head n=%d plain_ns=%.0f pdl_ns=%.0f diff_ns=%.0f\n", n, t[0][t[0].size() / 2], t[1][t[1].size() / 2],
               t[1][t[1].size() / 2] - t[0][t[0].size() / 2]);
    }
    CK(cudaFree(o));
}
// body: the chain's members (header), inlined into both forms; a block is 256 threads.
__device__ __forceinline__ unsigned long long gtime() { unsigned long long g; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g)); return g; }
// a row fold: one block, thread j sums x[k] * w[k * 256 + j] over k < 256 (decode's rows level)
__device__ __forceinline__ void row_fold(const float* x, const float* w, float* y) {
    const int j = threadIdx.x; float acc = 0.f;
#pragma unroll 8
    for (int k = 0; k < 256; k++) acc = fmaf(x[k], w[k * 256 + j], acc);
    y[j] = acc;
}
// a wide fold: block b, thread one output j = b * 256 + t over 8 loads of w (and 8 broadcast reads of the level before)
__device__ __forceinline__ void wide_fold(const float* x, int nx, const float* w, float* y, unsigned b, int n) {
    const int j = b * 256 + threadIdx.x; float acc = 0.f;
#pragma unroll
    for (int k = 0; k < 8; k++) acc = fmaf(x[(b * 8 + k) % nx], w[k * n + j], acc);
    y[j] = acc;
}
__global__ void __launch_bounds__(256) row_k(const float* __restrict__ x, const float* __restrict__ w0, const float* __restrict__ w1,
                                             float* __restrict__ r0, float* __restrict__ r1, int work, unsigned long long* t, int i) {
    if (blockIdx.x == 0 && threadIdx.x == 0) t[i] = gtime();
    if (!work) return;
    if (blockIdx.x == 0) row_fold(x, w0, r0); else row_fold(x, w1, r1);
}
// the chain's end: every block of its last level stamps its end (the latest wins)
__device__ __forceinline__ void stamp_end(unsigned long long* e, int i) { __syncthreads(); if (threadIdx.x == 0) atomicMax(e + i, gtime()); }
__global__ void __launch_bounds__(256) wide_k(const float* __restrict__ x, int nx, const float* __restrict__ w, float* __restrict__ y, int n, int work,
                                              unsigned long long* e, int i) {
    if (work) wide_fold(x, nx, w, y, blockIdx.x, n);
    if (e) stamp_end(e, i);
}
struct BodyP { const float *x, *w0, *w1, *ww[3]; float *r0, *r1, *p[3]; };
template <int L>
__global__ void __launch_bounds__(256) pers_k(BodyP a, int nb, int work, unsigned* ctr, unsigned long long* t, int i, unsigned long long* ck,
                                              unsigned long long* e) {
    unsigned long long g0 = 0, c0 = 0;
    if (blockIdx.x == 0 && threadIdx.x == 0) { g0 = gtime(); c0 = clock64(); t[i] = g0; }
    const unsigned g = gridDim.x;
    auto row = [&](unsigned b) { if (b == 0) row_fold(a.x, a.w0, a.r0); else row_fold(a.x, a.w1, a.r1); };
    if (work) {  // the first block straight-line, then the grid-stride rest (none at level 0: the grid holds both)
        if (blockIdx.x < 2) row(blockIdx.x);
        __syncthreads();
        for (unsigned b = blockIdx.x + g; b < 2; b += g) { row(b); __syncthreads(); }
    }
#pragma unroll
    for (int l = 1; l < L; l++) {
        gsync<1>(ctr, g);
        if (work) {
            const float* in = l == 1 ? a.r0 : a.p[l - 2]; const int nx = l == 1 ? 256 : nb * 256;
            if (blockIdx.x < (unsigned)nb) wide_fold(in, nx, a.ww[l - 1], a.p[l - 1], blockIdx.x, nb * 256);
            __syncthreads();
            for (unsigned b = blockIdx.x + g; b < (unsigned)nb; b += g) { wide_fold(in, nx, a.ww[l - 1], a.p[l - 1], b, nb * 256); __syncthreads(); }
        }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) { ck[2 * i] = clock64() - c0; ck[2 * i + 1] = gtime() - g0; }
    stamp_end(e, i);
}
__global__ void fill_k(float* v, size_t n) { for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) v[i] = (float)(i % 17) * 0.01f - 0.08f; }
static void body_run(int cc) {
    if (sus_khz == 0) ffma_run(cc);
    const int R = 128, reps = 15, one = sms * 3 / 2;
    int per[5] = {}, per_w = 0, regs[5] = {};
    void* fn[5] = {nullptr, nullptr, (void*)pers_k<2>, (void*)pers_k<3>, (void*)pers_k<4>};
    for (int L = 2; L <= 4; L++) {
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per[L], fn[L], 256, 0));
        cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, fn[L])); regs[L] = fa.numRegs;
    }
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_w, wide_k, 256, 0));
    cudaFuncAttributes fr, fw; CK(cudaFuncGetAttributes(&fr, row_k)); CK(cudaFuncGetAttributes(&fw, wide_k));
    const int nbmax = std::max(one, 4 * per[3] * sms);
    const size_t n = (size_t)nbmax * 256;
    BodyP P; float* buf;
    const size_t tot = 256 + 2 * 65536 + 3 * 8 * n + 2 * 256 + 3 * n;
    CK(cudaMalloc(&buf, tot * 4)); fill_k<<<sms * 8, 256>>>(buf, tot);
    float* q = buf; P.x = q; q += 256; P.w0 = q; q += 65536; P.w1 = q; q += 65536;
    for (int l = 0; l < 3; l++) { P.ww[l] = q; q += 8 * n; }
    P.r0 = q; q += 256; P.r1 = q; q += 256;
    for (int l = 0; l < 3; l++) { P.p[l] = q; q += n; }
    unsigned* ctr; unsigned long long *t, *ck, *ends;
    CK(cudaMalloc(&ctr, 4)); CK(cudaMemset(ctr, 0, 4)); CK(cudaMalloc(&t, (R + 1) * 8)); CK(cudaMalloc(&ck, 2 * (R + 1) * 8));
    CK(cudaMalloc(&ends, (R + 1) * 8));
    double run_clk = 0, run_ns = 0, line = 0;
    // one arm: 129 chains back to back behind the spin: start to start a chain, and a chain's own span (its first
    // block's start to its last block's end: no launch of the chain's own, so no cooperative-launch dispatch in it)
    auto arm = [&](int L, int nb, int grid, bool pers, int work, double* span) {
        CK(cudaMemset(ends, 0, (R + 1) * 8));
        spin_k<<<1, 32>>>((long long)clk_khz * 10);
        for (int i = 0; i <= R; i++) {
            if (pers) {
                void* a[] = {&P, &nb, &work, &ctr, &t, &i, &ck, &ends};
                CK(cudaLaunchCooperativeKernel(fn[L], grid, 256, a, 0, 0));
            } else {
                row_k<<<2, 256>>>(P.x, P.w0, P.w1, P.r0, P.r1, work, t, i);
                for (int l = 1; l < L; l++)
                    wide_k<<<nb, 256>>>(l == 1 ? P.r0 : P.p[l - 2], l == 1 ? 256 : nb * 256, P.ww[l - 1], P.p[l - 1], nb * 256, work,
                                        l == L - 1 ? ends : nullptr, i);
            }
        }
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        std::vector<unsigned long long> h(R + 1), z(R + 1);
        CK(cudaMemcpy(h.data(), t, (R + 1) * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(z.data(), ends, (R + 1) * 8, cudaMemcpyDeviceToHost));
        if (pers && work) {
            std::vector<unsigned long long> c(2 * (R + 1)); CK(cudaMemcpy(c.data(), ck, c.size() * 8, cudaMemcpyDeviceToHost));
            for (int i = 0; i <= R; i++) { run_clk += (double)c[2 * i]; run_ns += (double)c[2 * i + 1]; }
        }
        double sp = 0; for (int i = 0; i <= R; i++) sp += (double)(z[i] - h[i]);
        *span = sp / (R + 1);
        return (double)(h[R] - h[0]) / R;
    };
    struct Sh { int L, nb; };
    const int wave = per[3] * sms;
    for (Sh s : {Sh{2, one}, Sh{3, one}, Sh{4, one}, Sh{3, 2 * wave}, Sh{3, 4 * wave}}) {
        const int grid = std::min(per[s.L] * sms, std::max(2, s.nb)), rounds = (s.nb + grid - 1) / grid;
        // the two forms compute the same chain (the last level's output, bit for bit)
        const size_t on = (size_t)s.nb * 256; std::vector<float> o[2]; double sp;
        for (int f = 0; f < 2; f++) {
            CK(cudaMemset(P.p[s.L - 2], 0, on * 4)); arm(s.L, s.nb, grid, f == 1, 1, &sp);
            o[f].resize(on); CK(cudaMemcpy(o[f].data(), P.p[s.L - 2], on * 4, cudaMemcpyDeviceToHost));
        }
        const int match = memcmp(o[0].data(), o[1].data(), on * 4) == 0;
        run_clk = run_ns = 0;
        std::vector<double> v[4], w[4], body, bin;  // sep, pers, sep0, pers0: start to start (v) and span (w)
        for (int rep = 0; rep < reps; rep++) {
            double m[4], n[4];
            for (int k = 0; k < 4; k++) { const int a = (k + rep) % 4; m[a] = arm(s.L, s.nb, grid, a & 1, a < 2, &n[a]); }
            for (int a = 0; a < 4; a++) { v[a].push_back(m[a]); w[a].push_back(n[a]); }
            body.push_back((m[1] - m[3]) - (m[0] - m[2]));
            bin.push_back((n[1] - n[3]) - (n[0] - n[2]));
        }
        for (auto& x : v) std::sort(x.begin(), x.end());
        for (auto& x : w) std::sort(x.begin(), x.end());
        std::sort(body.begin(), body.end()); std::sort(bin.begin(), bin.end());
        const int md = reps / 2, lo = reps / 4, hi = reps - 1 - reps / 4;
        const double b = bin[md], rk = run_clk / run_ns * 1e6;
        printf("body levels=%d rounds=%d blocks=%d grid=%d per_sm=%d per_sm_alone=%d regs_sep=%d/%d regs_pers=%d match=%d "
               "sep_ns=%.0f sep0_ns=%.0f pers_ns=%.0f pers0_ns=%.0f body_ss_ns=%.0f [%.0f-%.0f] span_sep=%.0f span_sep0=%.0f "
               "span_pers=%.0f span_pers0=%.0f body_ns=%.0f [%.0f-%.0f] run_khz=%.0f body_cycles=%.0f\n",
               s.L, rounds, s.nb, grid, per[s.L], per_w, fr.numRegs, fw.numRegs, regs[s.L], match, v[0][md], v[2][md], v[1][md],
               v[3][md], body[md], body[lo], body[hi], w[0][md], w[2][md], w[1][md], w[3][md], b, bin[lo], bin[hi], rk,
               b * sus_khz / 1e6);
        if (s.L == 3 && rounds == 4) line = b;
    }
    printf("body coop_body_cycles=%.0f coop_body_ns=%.0f sustained_clock_khz=%.0f (levels=3 rounds=4)\n", line * sus_khz / 1e6, line,
           sus_khz);
    CK(cudaFree(buf)); CK(cudaFree(ctr)); CK(cudaFree(t)); CK(cudaFree(ck)); CK(cudaFree(ends));
}
int main(int argc, char** argv) {
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0)); sms = p.multiProcessorCount;
    CK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, 0));
    int mclk = 0; // CUDA 13 dropped cudaDeviceProp::memoryClockRate; the attribute stays
    CK(cudaDeviceGetAttribute(&mclk, cudaDevAttrMemoryClockRate, 0));
    printf("calib device=%s sms=%d clock_khz=%d l2_mb=%d mem_clock_khz=%d bus_bits=%d\n", p.name, sms, clk_khz, p.l2CacheSize >> 20, mclk, p.memoryBusWidth);
    int bps = 0; CK(cudaDeviceGetAttribute(&bps, cudaDevAttrMaxBlocksPerMultiprocessor, 0));
    printf("facts cc=%d sms=%d l2_bytes=%d smem_per_sm=%zu smem_per_block=%zu max_threads_per_sm=%d regs_per_sm=%d "
           "max_blocks_per_sm=%d clock_khz=%d mem_clock_khz=%d mem_bus_bits=%d dram_peak_bytes_per_ns=%.0f\n",
           p.major * 10 + p.minor, sms, p.l2CacheSize, p.sharedMemPerMultiprocessor, p.sharedMemPerBlockOptin,
           p.maxThreadsPerMultiProcessor, p.regsPerMultiprocessor, bps, clk_khz, mclk, p.memoryBusWidth,
           2.0 * mclk * p.memoryBusWidth / 8 / 1e6);
    std::string want; for (int i = 1; i < argc; i++) want += std::string(argv[i]) + " ";
    auto on = [&](const char* k) { return want.empty() || want.find(k) != std::string::npos; };
    if (on("chase")) chase_run();
    if (on("bw")) bw_run();
    if (on("loaded")) loaded_run();
    if (on("lds")) { lds_one<1>(); lds_one<2>(); lds_one<4>(); }
    if (on("gsync")) gsync_run();
    if (on("copy")) copy_run();
    if (on("ffma")) ffma_run(p.major * 10 + p.minor);
    if (on("head")) head_run();
    if (on("body")) body_run(p.major * 10 + p.minor);
    return 0;
}
