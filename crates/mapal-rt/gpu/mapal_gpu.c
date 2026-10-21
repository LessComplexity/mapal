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
#include <pthread.h>
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

/* The host-runtime rungs (S53), passed by the emitter's first call in `main`
 * (`mapal_gpu_init`; the bits are `GPU_INIT_EAGER` / `GPU_INIT_POOL` in
 * crates/backends/llvm/src/func/gpu.rs). A program without that call runs the
 * S52 path: init at the first site, cuMemAlloc / cuMemFree. */
enum { EAGER = 1, POOL = 2 };
static unsigned flags;

static CUmodule module;
static CUcontext ctx;
static pthread_t init_thread;
static int init_state; /* 0 not started, 1 running on init_thread, 2 ready */
/* Trace: wall time of cuInit, of the primary context (+ pool setup), of the
 * PTX read + module load, and when mapal_gpu_init ran. */
static double init_ms, ctx_ms, module_ms, t_init;

/* cuInit, the primary context and the module: everything a site needs before
 * its first call, none of which depends on the program's values. */
static void init_all(void) {
  double t0 = now_ms();
  CUdevice dev;
  CK(cuInit(0));
  double t1 = now_ms();
  CK(cuDeviceGet(&dev, 0));
  CK(cuDevicePrimaryCtxRetain(&ctx, dev));
  CK(cuCtxSetCurrent(ctx));
  if (flags & POOL) {
    /* The program's device footprint is static, so memory a site frees is
     * kept for the next site: release threshold = everything. A device
     * without stream-ordered pools keeps cuMemAlloc. */
    int ok = 0;
    CK(cuDeviceGetAttribute(&ok, CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED,
                            dev));
    if (ok) {
      CUmemoryPool pool;
      cuuint64_t keep = UINT64_MAX;
      CK(cuDeviceGetDefaultMemPool(&pool, dev));
      CK(cuMemPoolSetAttribute(pool, CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, &keep));
    } else {
      flags &= ~POOL;
    }
  }
  double t2 = now_ms();

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
  CK(cuModuleLoadData(&module, ptx));
  free(ptx);
  init_ms = t1 - t0;
  ctx_ms = t2 - t1;
  module_ms = now_ms() - t2;
}

static void *init_main(void *unused) {
  (void)unused;
  init_all();
  return 0;
}

/* A program that exits (a trap, 101) before its first site joins the init
 * first, so the driver is never torn down under a running cuInit. */
static void join_init(void) {
  if (init_state == 1 && !pthread_equal(pthread_self(), init_thread))
    pthread_join(init_thread, 0);
}

/* The first call of `main` in a module with a GPU site. Rung `eager-init`:
 * start the init on a thread, so context creation overlaps the host work
 * before the first site. A failed pthread_create falls back to the lazy path. */
void mapal_gpu_init(uint32_t f) {
  flags = f;
  t_init = now_ms();
  if ((flags & EAGER) && pthread_create(&init_thread, 0, init_main, 0) == 0) {
    init_state = 1;
    atexit(join_init);
  }
}

/* Every site calls it; returns the ms this site waited for the init. */
static double load_module(void) {
  if (init_state == 2)
    return 0;
  double t0 = now_ms();
  if (init_state == 1) {
    pthread_join(init_thread, 0);
    CK(cuCtxSetCurrent(ctx)); /* the context is current on the init thread */
  } else {
    init_all();
  }
  init_state = 2;
  return now_ms() - t0;
}

/* The device allocator of every site. Rung `pool`: stream-ordered on the
 * default stream, from the device's default pool; else cuMemAlloc/cuMemFree. */
static CUdeviceptr dev_alloc(uint64_t bytes) {
  CUdeviceptr p;
  if (flags & POOL)
    CK(cuMemAllocAsync(&p, bytes, 0));
  else
    CK(cuMemAlloc(&p, bytes));
  return p;
}

static void dev_free(CUdeviceptr p) {
  if (flags & POOL)
    CK(cuMemFreeAsync(p, 0));
  else
    CK(cuMemFree(p));
}

void mapal_gpu_tile(uint64_t id, const void *a, uint64_t a_bytes,
                    const void *b, uint64_t b_bytes, void *out,
                    uint64_t out_bytes, uint32_t gx, uint32_t gy,
                    uint32_t block, uint32_t smem) {
  if (trace < 0)
    trace = getenv("MAPAL_GPU_TRACE") != 0;
  double t[8];
  t[0] = now_ms();
  int first = init_state != 2;
  double wait_ms = load_module();
  t[1] = now_ms();
  char name[32];
  snprintf(name, sizeof name, "mapal_tile_%016llx", (unsigned long long)id);
  CUfunction fn;
  CK(cuModuleGetFunction(&fn, module, name));
  if (smem > 48 * 1024)
    CK(cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                          (int)smem));

  t[2] = now_ms();
  CUdeviceptr da = dev_alloc(a_bytes);
  CUdeviceptr db = dev_alloc(b_bytes);
  CUdeviceptr dout = dev_alloc(out_bytes);
  if (trace) /* the pool's calls return before the work: close the phase */
    CK(cuCtxSynchronize());
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
  dev_free(da);
  dev_free(db);
  dev_free(dout);
  if (trace)
    CK(cuCtxSynchronize());
  t[7] = now_ms();
  /* The init phases and `lead_ms` (mapal_gpu_init to this site, the host work
   * the init can overlap) belong to the first site; `wait_ms` is what this
   * site paid for them: all of them on the lazy path. */
  if (trace)
    fprintf(stderr,
            "mapal_gpu_trace kernel=%s flags=%u lead_ms=%.3f init_ms=%.3f "
            "ctx_ms=%.3f module_ms=%.3f wait_ms=%.3f lookup_ms=%.3f "
            "alloc_ms=%.3f h2d_ms=%.3f kernel_ms=%.3f d2h_ms=%.3f "
            "free_ms=%.3f bytes_in=%llu bytes_out=%llu\n",
            name, flags, first && t_init ? t[0] - t_init : 0,
            first ? init_ms : 0, first ? ctx_ms : 0, first ? module_ms : 0,
            wait_ms, t[2] - t[1], t[3] - t[2], t[4] - t[3], t[5] - t[4],
            t[6] - t[5], t[7] - t[6], (unsigned long long)(a_bytes + b_bytes),
            (unsigned long long)out_bytes);
}
