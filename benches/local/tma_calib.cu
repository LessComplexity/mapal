// hopper-wgmma-tma: the bulk-copy engine's cost per copy (`DeviceCost::tma_copy_ns`) — one elected thread per SM
// issuing back-to-back copies into a ring of `depth` shared buffers, each on its own transaction barrier, every SM
// at once, from an L2-resident 2048 x 2048 f32 tensor (16 MiB). Forms: a row copy (`cp.async.bulk`, 128 B: the
// rejected row staging's unit) and tensor boxes (`cp.async.bulk.tensor.2d`, 128-byte swizzle) of 32 x {8, 32, 128,
// 256} f32 (1 / 4 / 16 / 32 KiB). Depth 1 is the latency of one copy (issue to phase flip); the deepest ring its
// throughput: ns per copy per SM and GB/s over the part.
//
// usage: nvcc -O3 -std=c++17 -arch=sm_120 tma_calib.cu -lcuda -o tma_calib && ./tma_calib   (sm_90a on the H100)
// prints `tma form=<row|box> bytes=<B> depth=<d> ns_per_copy=<per SM> gbs=<part>`; `tma_copy_ns` is the box rows'
// ns_per_copy at the deepest ring (the engine's issue cost at a stage's box sizes).
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x)                                                                    \
    do {                                                                         \
        auto r_ = (x);                                                           \
        if (r_ != 0) {                                                           \
            fprintf(stderr, "%s: %d (line %d)\n", #x, (int)r_, __LINE__);        \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

constexpr int N = 2048, COPIES = 4096, MAXD = 8;

__device__ __forceinline__ unsigned smem32(const void *p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ bool try_wait(unsigned bar, unsigned parity) {
    unsigned ok;
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                 : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
    return ok;
}

// form 0: row copies of `bytes`; form 1: boxes of the map (rows `h` of 32 f32).
__global__ void __launch_bounds__(32) copies(const __grid_constant__ CUtensorMap tm, const float *src, int form,
                                             int bytes, int h, int depth, int *sink) {
    extern __shared__ __align__(1024) unsigned char sm[];
    __shared__ __align__(8) unsigned long long bars[MAXD];
    if (threadIdx.x != 0) return;
    for (int s = 0; s < depth; s++)
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem32(&bars[s])) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    int buf = (bytes + 1023) / 1024 * 1024;
    for (int i = 0; i < COPIES; i++) {
        int s = i % depth;
        unsigned bar = smem32(&bars[s]);
        if (i >= depth)
            while (!try_wait(bar, ((i / depth) - 1) & 1)) {
            }
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
        unsigned dst = smem32(sm + s * buf);
        int x = (i * 32 + blockIdx.x * 64) % N, y = (i * 7 + blockIdx.x * 13) % (N - h);
        if (form == 0) {
            const float *g = src + (size_t)y * N + x;
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
                         "l"(g), "r"(bytes), "r"(bar)
                         : "memory");
        } else {
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(dst),
                "l"(&tm), "r"(x), "r"(y), "r"(bar)
                : "memory");
        }
    }
    for (int i = COPIES; i < COPIES + depth; i++) {
        int s = i % depth;
        while (!try_wait(smem32(&bars[s]), ((i / depth) - 1) & 1)) {
        }
    }
    sink[blockIdx.x] = sm[5];
}

int main() {
    CK(cuInit(0));
    int dev = 0, sms = 0;
    CK(cudaGetDevice(&dev));
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    float *src;
    int *sink;
    CK(cudaMalloc(&src, (size_t)N * N * 4));
    CK(cudaMalloc(&sink, 4096 * 4));
    std::vector<float> h((size_t)N * N);
    for (size_t i = 0; i < h.size(); i++) h[i] = (float)(i % 977);
    CK(cudaMemcpy(src, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    int optin = 0;
    CK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    CK(cudaFuncSetAttribute(copies, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - 1024));
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    struct Form {
        int form, h;
    };
    std::vector<Form> forms = {{0, 0}, {1, 8}, {1, 32}, {1, 128}, {1, 256}};
    for (auto f : forms) {
        int bytes = f.form == 0 ? 128 : 32 * 4 * f.h;
        CUtensorMap tm{};
        if (f.form == 1) {
            cuuint64_t dims[2] = {N, N}, strides[1] = {N * 4};
            cuuint32_t box[2] = {32, (cuuint32_t)f.h}, es[2] = {1, 1};
            CK(cuTensorMapEncodeTiled(&tm, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, src, dims, strides, box, es,
                                      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
        }
        for (int depth : {1, 2, 4, 8}) {
            int buf = (bytes + 1023) / 1024 * 1024;
            if (depth * buf > optin - 1024) continue;
            std::vector<float> ms;
            for (int r = 0; r < 7; r++) {
                CK(cudaEventRecord(a));
                copies<<<sms, 32, depth * buf>>>(tm, src, f.form, bytes, f.form ? f.h : 1, depth, sink);
                CK(cudaEventRecord(b));
                CK(cudaEventSynchronize(b));
                float t;
                CK(cudaEventElapsedTime(&t, a, b));
                ms.push_back(t);
            }
            CK(cudaGetLastError());
            std::sort(ms.begin(), ms.end());
            double t = ms[3] * 1e6;  // ns, median
            printf("tma form=%s bytes=%d depth=%d ns_per_copy=%.1f gbs=%.0f\n", f.form ? "box" : "row", bytes, depth,
                   t / COPIES, (double)bytes * COPIES * sms / t);
        }
    }
    return 0;
}
