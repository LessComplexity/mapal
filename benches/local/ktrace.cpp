// usage: g++ -O2 -shared -fPIC -I/opt/cuda/targets/x86_64-linux/include ktrace.cpp -o ktrace.so -L/opt/cuda/targets/x86_64-linux/lib -lcupti
//        CUDA_INJECTION64_PATH=$PWD/ktrace.so KTRACE_OUT=out.tsv [KTRACE_API=1] <program>
// CUPTI activity tracer, loaded with CUDA_INJECTION64_PATH=ktrace.so; writes one TSV line per
// kernel / memcpy / memset to $KTRACE_OUT (append). Columns:
// K name grid block regs smem_static smem_dyn start end stream  |  C bytes kind start end  |  S bytes start end
#include <cupti.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// The record versions this CUPTI writes (an older CUPTI: -DKTRACE_KERNEL=CUpti_ActivityKernel9
// -DKTRACE_MEMCPY=CUpti_ActivityMemcpy5 -DKTRACE_MEMSET=CUpti_ActivityMemset4, the CUDA 13.0 image).
#ifndef KTRACE_KERNEL
#define KTRACE_KERNEL CUpti_ActivityKernel13
#endif
#ifndef KTRACE_MEMCPY
#define KTRACE_MEMCPY CUpti_ActivityMemcpy7
#endif
#ifndef KTRACE_MEMSET
#define KTRACE_MEMSET CUpti_ActivityMemset5
#endif

static FILE *out;

static void CUPTIAPI buf_req(uint8_t **buf, size_t *size, size_t *max) {
    *size = 16 << 20;
    *buf = (uint8_t *)aligned_alloc(8, *size);
    *max = 0;
}

static void CUPTIAPI buf_done(CUcontext, uint32_t, uint8_t *buf, size_t, size_t valid) {
    CUpti_Activity *r = nullptr;
    while (cuptiActivityGetNextRecord(buf, valid, &r) == CUPTI_SUCCESS) {
        if (r->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL || r->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
            auto *k = (KTRACE_KERNEL *)r;
            fprintf(out, "K\t%s\t%d,%d,%d\t%d,%d,%d\t%u\t%d\t%u\t%llu\t%llu\t%u\n", k->name, k->gridX, k->gridY,
                    k->gridZ, k->blockX, k->blockY, k->blockZ, k->registersPerThread, k->staticSharedMemory,
                    k->dynamicSharedMemory, (unsigned long long)k->start, (unsigned long long)k->end, k->streamId);
        } else if (r->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
            auto *m = (KTRACE_MEMCPY *)r;
            fprintf(out, "C\t%llu\t%u\t%llu\t%llu\n", (unsigned long long)m->bytes, m->copyKind,
                    (unsigned long long)m->start, (unsigned long long)m->end);
        } else if (r->kind == CUPTI_ACTIVITY_KIND_DRIVER || r->kind == CUPTI_ACTIVITY_KIND_RUNTIME) {
            auto *a = (CUpti_ActivityAPI *)r;
            const char *nm = nullptr;
            cuptiGetCallbackName(r->kind == CUPTI_ACTIVITY_KIND_DRIVER ? CUPTI_CB_DOMAIN_DRIVER_API : CUPTI_CB_DOMAIN_RUNTIME_API, a->cbid, &nm);
            fprintf(out, "%c\t%s\t%llu\t%llu\n", r->kind == CUPTI_ACTIVITY_KIND_DRIVER ? 'D' : 'R', nm ? nm : "?",
                    (unsigned long long)a->start, (unsigned long long)a->end);
        } else if (r->kind == CUPTI_ACTIVITY_KIND_MEMSET) {
            auto *m = (KTRACE_MEMSET *)r;
            fprintf(out, "S\t%llu\t%llu\t%llu\n", (unsigned long long)m->bytes, (unsigned long long)m->start,
                    (unsigned long long)m->end);
        }
    }
    free(buf);
}

static void flush_all() {
    cuptiActivityFlushAll(1);
    if (out) fflush(out);
}

extern "C" int InitializeInjection(void) {
    const char *p = getenv("KTRACE_OUT");
    out = fopen(p ? p : "ktrace.tsv", "a");
    if (!out) return 0;
    cuptiActivityRegisterCallbacks(buf_req, buf_done);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET);
    if (getenv("KTRACE_API")) cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER);
    if (getenv("KTRACE_API")) cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME);
    atexit(flush_all);
    return 1;
}
