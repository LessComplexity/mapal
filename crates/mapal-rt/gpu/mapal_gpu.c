/* The GPU half of the runtime (plan-s41 step 5, S52): the `h2d` / `launch` /
 * `d2h` transmissions one offloaded tile site needs, over the CUDA driver API.
 *
 * C rather than Rust on purpose: it links against `libcuda`, which only exists
 * on a machine with an NVIDIA driver, so it is compiled there, next to the
 * emitted host object and `libmapal_rt.a`:
 *
 *   cc host.o crates/mapal-rt/gpu/mapal_gpu.c libmapal_rt.a \
 *      -I$CUDA/include -L$CUDA/lib64/stubs -lcuda -lpthread -ldl -lm
 *   MAPAL_PTX=prog.ptx ./a.out
 *
 * The kernel is found by name: the emitter passes the site's 64-bit kernel id
 * and the device unit names the kernel `mapal_tile_<id as 16 hex digits>`.
 * Any driver failure is an infrastructure failure, not a program result: it
 * exits 102, the CUDA backend's protocol, so a harness never mistakes it for a
 * trap (101) or a value. */
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* MAPAL_GPU_TRACE=1: one stderr line per site with the wall time of every
 * phase, each closed by a synchronize, so the host side's share of a site is
 * measured rather than assumed. Off by default; the phases are unchanged. */
static int trace = -1;
static double now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

#define CK(x)                                                                  \
  do {                                                                         \
    CUresult r_ = (x);                                                         \
    if (r_ != CUDA_SUCCESS) {                                                  \
      const char *s_ = 0;                                                      \
      cuGetErrorString(r_, &s_);                                               \
      fprintf(stderr, "mapal_gpu: %s: %s\n", #x, s_ ? s_ : "?");               \
      exit(102);                                                               \
    }                                                                          \
  } while (0)

static CUmodule module;

static void load_module(void) {
  if (module)
    return;
  const char *path = getenv("MAPAL_PTX");
  if (!path) {
    fprintf(stderr, "mapal_gpu: MAPAL_PTX is not set\n");
    exit(102);
  }
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "mapal_gpu: cannot open %s\n", path);
    exit(102);
  }
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *ptx = malloc((size_t)n + 1);
  if (!ptx || fread(ptx, 1, (size_t)n, f) != (size_t)n) {
    fprintf(stderr, "mapal_gpu: cannot read %s\n", path);
    exit(102);
  }
  ptx[n] = 0;
  fclose(f);

  CUdevice dev;
  CUcontext ctx;
  CK(cuInit(0));
  CK(cuDeviceGet(&dev, 0));
  CK(cuDevicePrimaryCtxRetain(&ctx, dev));
  CK(cuCtxSetCurrent(ctx));
  CK(cuModuleLoadData(&module, ptx));
  free(ptx);
}

void mapal_gpu_tile(uint64_t id, const void *a, uint64_t a_bytes,
                    const void *b, uint64_t b_bytes, void *out,
                    uint64_t out_bytes, uint32_t gx, uint32_t gy,
                    uint32_t block, uint32_t smem) {
  if (trace < 0)
    trace = getenv("MAPAL_GPU_TRACE") != 0;
  double t[8];
  t[0] = now_ms();
  load_module();
  t[1] = now_ms();
  char name[32];
  snprintf(name, sizeof name, "mapal_tile_%016llx", (unsigned long long)id);
  CUfunction fn;
  CK(cuModuleGetFunction(&fn, module, name));
  if (smem > 48 * 1024)
    CK(cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                          (int)smem));

  CUdeviceptr da, db, dout;
  t[2] = now_ms();
  CK(cuMemAlloc(&da, a_bytes));
  CK(cuMemAlloc(&db, b_bytes));
  CK(cuMemAlloc(&dout, out_bytes));
  t[3] = now_ms();
  CK(cuMemcpyHtoD(da, a, a_bytes));
  CK(cuMemcpyHtoD(db, b, b_bytes));
  t[4] = now_ms();
  void *args[] = {&da, &db, &dout};
  CK(cuLaunchKernel(fn, gx, gy, 1, block, 1, 1, smem, 0, args, 0));
  CK(cuCtxSynchronize());
  t[5] = now_ms();
  CK(cuMemcpyDtoH(out, dout, out_bytes));
  t[6] = now_ms();
  CK(cuMemFree(da));
  CK(cuMemFree(db));
  CK(cuMemFree(dout));
  t[7] = now_ms();
  if (trace)
    fprintf(stderr,
            "mapal_gpu_trace kernel=%s load_ms=%.3f lookup_ms=%.3f alloc_ms=%.3f "
            "h2d_ms=%.3f kernel_ms=%.3f d2h_ms=%.3f free_ms=%.3f bytes_in=%llu "
            "bytes_out=%llu\n",
            name, t[1] - t[0], t[2] - t[1], t[3] - t[2], t[4] - t[3],
            t[5] - t[4], t[6] - t[5], t[7] - t[6],
            (unsigned long long)(a_bytes + b_bytes),
            (unsigned long long)out_bytes);
}

/* The clock fence's device half (plan-s53): the emitter calls this before every
 * `time` read on a GPU profile, so the clock never closes over a kernel still
 * running. No module loaded means no launch yet: nothing to wait for. */
void mapal_gpu_sync(void) {
  if (module)
    CK(cuCtxSynchronize());
}
