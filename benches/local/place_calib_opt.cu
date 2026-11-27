// ADR-0050 cost model: the primitives a part may lack, built to PTX for the least target that has each one and
// loaded by `place_calib prim` through the driver's JIT (cuModuleLoadDataEx). A part executes a primitive when its
// JIT accepts the module — the capability is the PTX ISA's, never a part's name. One source, one module a family:
//   -DPRIM_FP8    mma.sync.m16n8k32 e4m3 (compute_89)
//   -DPRIM_TMA    cp.async.bulk.tensor.2d box copies on transaction barriers (compute_90)
//   -DPRIM_WGMMA  wgmma.mma_async m64n128k16 bf16, both operands from shared-memory descriptors (compute_90a)
// build (place_calib.sh): nvcc -ptx -arch=compute_89 -DPRIM_FP8 place_calib_opt.cu -o place_calib_fp8.ptx, and the
// same for TMA (compute_90) and WGMMA (compute_90a).
#include <cuda.h>

#ifdef PRIM_FP8
// `C` independent m16n8k32 e4m3 accumulators a warp, operands constant in registers (the clock64 span of `iters`
// trips, the block's longest into cyc[block]).
#define FP8_K(C)                                                                                               \
    extern "C" __global__ void fp8_c##C(float *out, int iters, long long *cyc) {                               \
        float d[C][4];                                                                                         \
        _Pragma("unroll") for (int i = 0; i < C; i++) for (int j = 0; j < 4; j++) d[i][j] = 0;                 \
        unsigned a0 = 0x38383838u + (unsigned)(out[0] != 0), a1 = a0, a2 = a0, a3 = a0, b0 = a0, b1 = a0;      \
        __syncthreads();                                                                                       \
        long long t0 = clock64();                                                                              \
        for (int it = 0; it < iters; it++) {                                                                   \
            _Pragma("unroll") for (int i = 0; i < C; i++) asm volatile(                                        \
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "  \
                "{%0,%1,%2,%3};"                                                                               \
                : "+f"(d[i][0]), "+f"(d[i][1]), "+f"(d[i][2]), "+f"(d[i][3])                                   \
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));                                       \
        }                                                                                                      \
        long long t1 = clock64();                                                                              \
        float s = 0;                                                                                           \
        _Pragma("unroll") for (int i = 0; i < C; i++) s += d[i][0];                                            \
        if (s == 12345.f) out[1] = s;                                                                          \
        if (threadIdx.x == 0) atomicMax((unsigned long long *)cyc + blockIdx.x % 1024, (unsigned long long)(t1 - t0)); \
    }
FP8_K(1)
FP8_K(2)
FP8_K(4)
FP8_K(8)
#endif

#ifdef PRIM_TMA
__device__ __forceinline__ unsigned smem32(const void *p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ bool try_wait(unsigned bar, unsigned parity) {
    unsigned ok;
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                 : "=r"(ok)
                 : "r"(bar), "r"(parity)
                 : "memory");
    return ok;
}
// One elected thread a block issues `copies` boxes of `bytes` (`h` rows of the map's 32-element box) into a ring of
// `depth` buffers, each on its own transaction barrier: `depth` boxes in flight a block, over an `nx` x `ny` f32
// map (L2-resident or 16x the L2). Depth 1, one block: a copy's issue-to-phase-flip time.
extern "C" __global__ void __launch_bounds__(32) tma_k(const __grid_constant__ CUtensorMap tm, int bytes, int h,
                                                       int nx, int ny, int depth, int copies, int *sink) {
    extern __shared__ __align__(1024) unsigned char sm[];
    __shared__ __align__(8) unsigned long long bars[16];
    if (threadIdx.x != 0) return;
    for (int s = 0; s < depth; s++)
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem32(&bars[s])) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    int buf = (bytes + 1023) / 1024 * 1024;
    for (int i = 0; i < copies; i++) {
        int s = i % depth;
        unsigned bar = smem32(&bars[s]);
        if (i >= depth)
            while (!try_wait(bar, ((i / depth) - 1) & 1)) {
            }
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
        unsigned dst = smem32(sm + s * buf);
        int x = (i * 32 + blockIdx.x * 64) % nx, y = (i * 7 * h + blockIdx.x * 13 * h) % (ny - h);
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], "
            "[%4];" ::"r"(dst),
            "l"(&tm), "r"(x), "r"(y), "r"(bar)
            : "memory");
    }
    for (int i = copies; i < copies + depth; i++) {
        int s = i % depth;
        while (!try_wait(smem32(&bars[s]), ((i / depth) - 1) & 1)) {
        }
    }
    sink[blockIdx.x] = sm[5];
}
#endif

#ifdef PRIM_WGMMA
// `C` independent m64n128k16 bf16 accumulators a warpgroup (64 f32 registers a thread each), `a` and `b` from
// shared-memory descriptors (no swizzle; the bytes' values do not matter to the rate): a trip issues the C
// products, commits and waits for all.
__device__ __forceinline__ unsigned long long desc(const void *p) {
    unsigned a = (unsigned)__cvta_generic_to_shared(p);
    // start >> 4 (bits 0-13), leading byte offset 128 >> 4 (16-29), stride byte offset 256 >> 4 (32-45).
    return (unsigned long long)((a >> 4) & 0x3fff) | ((unsigned long long)(128 >> 4) << 16) |
           ((unsigned long long)(256 >> 4) << 32);
}
#define ACC64(i)                                                                                                       \
    "+f"(d[i][0]), "+f"(d[i][1]), "+f"(d[i][2]), "+f"(d[i][3]), "+f"(d[i][4]), "+f"(d[i][5]), "+f"(d[i][6]),          \
        "+f"(d[i][7]), "+f"(d[i][8]), "+f"(d[i][9]), "+f"(d[i][10]), "+f"(d[i][11]), "+f"(d[i][12]), "+f"(d[i][13]),  \
        "+f"(d[i][14]), "+f"(d[i][15]), "+f"(d[i][16]), "+f"(d[i][17]), "+f"(d[i][18]), "+f"(d[i][19]),               \
        "+f"(d[i][20]), "+f"(d[i][21]), "+f"(d[i][22]), "+f"(d[i][23]), "+f"(d[i][24]), "+f"(d[i][25]),               \
        "+f"(d[i][26]), "+f"(d[i][27]), "+f"(d[i][28]), "+f"(d[i][29]), "+f"(d[i][30]), "+f"(d[i][31])
#define ACC64B(i)                                                                                                      \
    "+f"(d[i][32]), "+f"(d[i][33]), "+f"(d[i][34]), "+f"(d[i][35]), "+f"(d[i][36]), "+f"(d[i][37]), "+f"(d[i][38]),   \
        "+f"(d[i][39]), "+f"(d[i][40]), "+f"(d[i][41]), "+f"(d[i][42]), "+f"(d[i][43]), "+f"(d[i][44]),               \
        "+f"(d[i][45]), "+f"(d[i][46]), "+f"(d[i][47]), "+f"(d[i][48]), "+f"(d[i][49]), "+f"(d[i][50]),               \
        "+f"(d[i][51]), "+f"(d[i][52]), "+f"(d[i][53]), "+f"(d[i][54]), "+f"(d[i][55]), "+f"(d[i][56]),               \
        "+f"(d[i][57]), "+f"(d[i][58]), "+f"(d[i][59]), "+f"(d[i][60]), "+f"(d[i][61]), "+f"(d[i][62]),               \
        "+f"(d[i][63])
#define WG_ONE(i)                                                                                                      \
    asm volatile(                                                                                                      \
        "{ .reg .pred p; setp.ne.b32 p, %66, 0;\n"                                                                     \
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,"  \
        "%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41," \
        "%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, p, 1, 1, " \
        "0, 0; }"                                                                                                      \
        : ACC64(i), ACC64B(i)                                                                                          \
        : "l"(da), "l"(db), "r"(1))
#define WG_K(C)                                                                                                        \
    extern "C" __global__ void __launch_bounds__(128) wgmma_c##C(float *out, int iters, long long *cyc) {             \
        __shared__ __align__(1024) unsigned char sm[16384];                                                            \
        for (int i = threadIdx.x; i < 16384; i += blockDim.x) sm[i] = (unsigned char)(i * 7 + (out[0] != 0));          \
        float d[C][64];                                                                                                \
        _Pragma("unroll") for (int i = 0; i < C; i++) _Pragma("unroll") for (int j = 0; j < 64; j++) d[i][j] = 0;     \
        unsigned long long da = desc(sm), db = desc(sm + 8192);                                                        \
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");                                                   \
        __syncthreads();                                                                                               \
        long long t0 = clock64();                                                                                      \
        for (int it = 0; it < iters; it++) {                                                                           \
            asm volatile("wgmma.fence.sync.aligned;" ::: "memory");                                                    \
            _Pragma("unroll") for (int i = 0; i < C; i++) WG_ONE(i);                                                   \
            asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");                                             \
            asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");                                             \
        }                                                                                                              \
        long long t1 = clock64();                                                                                      \
        float s = 0;                                                                                                   \
        _Pragma("unroll") for (int i = 0; i < C; i++) s += d[i][0];                                                    \
        if (s == 12345.f) out[1] = s;                                                                                  \
        if (threadIdx.x == 0) atomicMax((unsigned long long *)cyc + blockIdx.x % 1024, (unsigned long long)(t1 - t0)); \
    }
WG_K(1)
WG_K(2)
#endif
