/* The GPU half of the runtime (plan-s41 step 5, S52; split in S53): the
 * `alloc` / `h2d` / `launch` / `d2h` / `free` transmissions of offloaded tile
 * sites, over the CUDA driver API.
 *
 * C rather than Rust on purpose: it links against `libcuda`, which only exists
 * on a machine with an NVIDIA driver, so it is compiled there, next to the
 * emitted host object and `libmapal_rt.a`:
 *
 *   cc host.o crates/mapal-rt/gpu/mapal_gpu.c libmapal_rt.a \
 *      -I$CUDA/include -L$CUDA/lib64/stubs -lcuda -lpthread -ldl -lm
 *   MAPAL_PTX=prog.ptx ./a.out
 *
 * Two host ABIs over the same entries:
 *  - `mapal_gpu_tile`: one call per site that allocates, copies both operands
 *    in, launches, copies the whole output back and frees (S52; the OFF arm of
 *    the `resident` rung).
 *  - the split entries (S53 `resident`): the emitter keeps an object's device
 *    copy in a per-object slot (`uint64_t`, 0 = none), copies in at the
 *    producer, copies out only what the host reads, and frees at death.
 *    Launches are asynchronous on the one (legacy default) stream; `d2h` is
 *    synchronous, so a host read always sees every launch before it.
 *
 * Kernels are found by name: the emitter passes a 64-bit id and the device
 * unit names the kernel `mapal_tile_<id as 16 hex digits>` (a tile site) or
 * `mapal_gen_<id>` (a producer computed on the device, `devgen`).
 * Any driver failure is an infrastructure failure, not a program result: it
 * exits 102, the CUDA backend's protocol, so a harness never mistakes it for a
 * trap (101) or a value. */
#include <cuda.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* MAPAL_GPU_TRACE=1: one stderr line per operation with its wall time, and a
 * `mapal_gpu_trace_total` line at exit summing every phase, the bytes each way
 * and the operation counts, so two arms of one program compare by one line.
 * Under the trace a launch is followed by a synchronize, so its time is the
 * kernel's; off, launches stay asynchronous. `load_ms` is what the program
 * waited for the context and module (`wait`): all of `init + ctx + module` on
 * the lazy path, that minus `lead` (host work from `mapal_gpu_init` to the
 * first site) under `eager-init`. */
static int trace = -1;
static struct {
  double load, alloc, h2d, kernel, gen, d2h, free;
  double lead, init, ctx, module;
  unsigned long long in, out, n_h2d, n_d2h, n_launch, n_gen, n_alloc;
} tot;
static unsigned flags;
static double now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}
static void total_line(void) {
  fprintf(stderr,
          "mapal_gpu_trace_total flags=%u load_ms=%.3f lead_ms=%.3f "
          "init_ms=%.3f ctx_ms=%.3f module_ms=%.3f alloc_ms=%.3f h2d_ms=%.3f "
          "kernel_ms=%.3f gen_ms=%.3f d2h_ms=%.3f free_ms=%.3f bytes_in=%llu "
          "bytes_out=%llu h2d=%llu d2h=%llu launches=%llu gens=%llu allocs=%llu\n",
          flags, tot.load, tot.lead, tot.init, tot.ctx, tot.module, tot.alloc,
          tot.h2d, tot.kernel, tot.gen, tot.d2h, tot.free, tot.in, tot.out,
          tot.n_h2d, tot.n_d2h, tot.n_launch, tot.n_gen, tot.n_alloc);
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

/* The host-runtime rung (S53), passed by the emitter's first call in `main`
 * (`mapal_gpu_init`; the bit is `GPU_INIT_EAGER` in
 * crates/backends/llvm/src/func/gpu.rs). A program without that call runs the
 * S52 path: init at the first site. */
enum { EAGER = 1 };

static CUmodule module;
static CUcontext ctx;
static pthread_t init_thread;
static int init_state; /* 0 not started, 1 running on init_thread, 2 ready */
static double t_init;  /* when mapal_gpu_init ran */

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
  tot.init = t1 - t0;
  tot.ctx = t2 - t1;
  tot.module = now_ms() - t2;
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

/* Every entry that needs the device calls it first: joins the eager init, or
 * runs the lazy one. */
static void load_module(void) {
  if (init_state == 2)
    return;
  if (trace < 0) {
    trace = getenv("MAPAL_GPU_TRACE") != 0;
    if (trace)
      atexit(total_line);
  }
  double t0 = now_ms();
  if (init_state == 1) {
    pthread_join(init_thread, 0);
    CK(cuCtxSetCurrent(ctx)); /* the context is current on the init thread */
  } else {
    init_all();
  }
  init_state = 2;
  tot.load = now_ms() - t0;
  tot.lead = t_init ? t0 - t_init : 0;
  if (trace)
    fprintf(stderr,
            "mapal_gpu_trace op=load flags=%u wait_ms=%.3f lead_ms=%.3f "
            "init_ms=%.3f ctx_ms=%.3f module_ms=%.3f\n",
            flags, tot.load, tot.lead, tot.init, tot.ctx, tot.module);
}

static CUfunction kernel(const char *prefix, uint64_t id, uint32_t smem,
                         char *name, size_t cap) {
  snprintf(name, cap, "%s_%016llx", prefix, (unsigned long long)id);
  CUfunction fn;
  CK(cuModuleGetFunction(&fn, module, name));
  if (smem > 48 * 1024)
    CK(cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                          (int)smem));
  return fn;
}

/* The device buffer of `*slot`, allocated on first use: a slot is one object's
 * device copy, so a producer inside a loop reuses its buffer every iteration. */
uint64_t mapal_gpu_alloc(uint64_t *slot, uint64_t bytes) {
  load_module();
  if (*slot)
    return *slot;
  double t0 = now_ms();
  CUdeviceptr d;
  CK(cuMemAlloc(&d, bytes));
  *slot = (uint64_t)d;
  double ms = now_ms() - t0;
  tot.alloc += ms;
  tot.n_alloc++;
  if (trace)
    fprintf(stderr, "mapal_gpu_trace op=alloc ms=%.3f bytes=%llu\n", ms,
            (unsigned long long)bytes);
  return *slot;
}

void mapal_gpu_h2d(uint64_t d, const void *h, uint64_t bytes) {
  double t0 = now_ms();
  CK(cuMemcpyHtoD((CUdeviceptr)d, h, bytes));
  double ms = now_ms() - t0;
  tot.h2d += ms;
  tot.in += bytes;
  tot.n_h2d++;
  if (trace)
    fprintf(stderr, "mapal_gpu_trace op=h2d ms=%.3f bytes=%llu\n", ms,
            (unsigned long long)bytes);
}

/* `bytes` from byte `off` of the device buffer `d` into host `h`.
 * Synchronous: it waits for every launch before it on the stream. */
void mapal_gpu_d2h(void *h, uint64_t d, uint64_t off, uint64_t bytes) {
  double t0 = now_ms();
  CK(cuMemcpyDtoH(h, (CUdeviceptr)(d + off), bytes));
  double ms = now_ms() - t0;
  tot.d2h += ms;
  tot.out += bytes;
  tot.n_d2h++;
  if (trace)
    fprintf(stderr, "mapal_gpu_trace op=d2h ms=%.3f bytes=%llu\n", ms,
            (unsigned long long)bytes);
}

void mapal_gpu_launch(uint64_t id, uint64_t a, uint64_t b, uint64_t out,
                      uint32_t gx, uint32_t gy, uint32_t block, uint32_t smem) {
  load_module();
  char name[40];
  CUfunction fn = kernel("mapal_tile", id, smem, name, sizeof name);
  CUdeviceptr da = a, db = b, dout = out;
  void *args[] = {&da, &db, &dout};
  double t0 = now_ms();
  CK(cuLaunchKernel(fn, gx, gy, 1, block, 1, 1, smem, 0, args, 0));
  if (trace)
    CK(cuCtxSynchronize());
  double ms = now_ms() - t0;
  tot.kernel += ms;
  tot.n_launch++;
  if (trace)
    fprintf(stderr, "mapal_gpu_trace op=launch kernel=%s ms=%.3f\n", name, ms);
}

/* A kernel over `nptr` device pointers (S54): a site whose store also
 * evaluates per-cell consumers takes their operands and outputs after its
 * own `a, b, out`. `stream` is the stream index; only 0 (the legacy default
 * stream) exists until streams do. */
void mapal_gpu_launchv(uint64_t id, uint32_t stream, uint32_t nptr,
                       const uint64_t *ptrs, uint32_t gx, uint32_t gy,
                       uint32_t block, uint32_t smem) {
  load_module();
  char name[40];
  CUfunction fn = kernel("mapal_tile", id, smem, name, sizeof name);
  CUdeviceptr d[16];
  void *args[16];
  if (nptr > 16 || stream != 0) {
    fprintf(stderr, "mapal_gpu: launchv nptr=%u stream=%u\n", nptr, stream);
    exit(102);
  }
  for (uint32_t i = 0; i < nptr; i++) {
    d[i] = (CUdeviceptr)ptrs[i];
    args[i] = &d[i];
  }
  double t0 = now_ms();
  CK(cuLaunchKernel(fn, gx, gy, 1, block, 1, 1, smem, 0, args, 0));
  if (trace)
    CK(cuCtxSynchronize());
  double ms = now_ms() - t0;
  tot.kernel += ms;
  tot.n_launch++;
  if (trace)
    fprintf(stderr, "mapal_gpu_trace op=launch kernel=%s ms=%.3f\n", name, ms);
}

/* `devgen`: fill `n` elements of `out` from the producer's element law. The
 * kernel is one thread per element, `MAPAL_GEN_THREADS` per block — the
 * emitter's naive-kernel width, mirrored here. */
#define MAPAL_GEN_THREADS 256
void mapal_gpu_gen(uint64_t id, uint64_t out, uint64_t n) {
  load_module();
  char name[40];
  CUfunction fn = kernel("mapal_gen", id, 0, name, sizeof name);
  CUdeviceptr dout = out;
  uint64_t count = n;
  void *args[] = {&dout, &count};
  unsigned grid = (unsigned)((n + MAPAL_GEN_THREADS - 1) / MAPAL_GEN_THREADS);
  double t0 = now_ms();
  CK(cuLaunchKernel(fn, grid, 1, 1, MAPAL_GEN_THREADS, 1, 1, 0, 0, args, 0));
  if (trace)
    CK(cuCtxSynchronize());
  double ms = now_ms() - t0;
  tot.gen += ms;
  tot.n_gen++;
  if (trace)
    fprintf(stderr, "mapal_gpu_trace op=gen kernel=%s ms=%.3f n=%llu\n", name,
            ms, (unsigned long long)n);
}

/* Free `*slot`'s buffer if it has one and mark the slot empty — so freeing at
 * an object's death and again at function exit is one free. */
void mapal_gpu_free(uint64_t *slot) {
  if (!*slot)
    return;
  double t0 = now_ms();
  CK(cuMemFree((CUdeviceptr)*slot));
  *slot = 0;
  double ms = now_ms() - t0;
  tot.free += ms;
  if (trace)
    fprintf(stderr, "mapal_gpu_trace op=free ms=%.3f\n", ms);
}

/* The clock fence's device half (plan-s53): the emitter calls this before every
 * `time` read on a GPU profile, so the clock never closes over a kernel still
 * running. No module loaded means no launch yet: nothing to wait for. An eager
 * init still running is device work in flight too (S54: a program whose first
 * launch is after the read — every operand built in its kernel — would
 * otherwise time the context inside its clock): it is joined. */
void mapal_gpu_sync(void) {
  if (init_state == 1)
    load_module();
  if (init_state == 2)
    CK(cuCtxSynchronize());
}

/* One site, whole round trip (the S52 ABI): composed of the split entries. */
void mapal_gpu_tile(uint64_t id, const void *a, uint64_t a_bytes,
                    const void *b, uint64_t b_bytes, void *out,
                    uint64_t out_bytes, uint32_t gx, uint32_t gy,
                    uint32_t block, uint32_t smem) {
  uint64_t da = 0, db = 0, dout = 0;
  mapal_gpu_alloc(&da, a_bytes);
  mapal_gpu_alloc(&db, b_bytes);
  mapal_gpu_alloc(&dout, out_bytes);
  mapal_gpu_h2d(da, a, a_bytes);
  mapal_gpu_h2d(db, b, b_bytes);
  mapal_gpu_launch(id, da, db, dout, gx, gy, block, smem);
  mapal_gpu_d2h(out, dout, 0, out_bytes);
  mapal_gpu_free(&da);
  mapal_gpu_free(&db);
  mapal_gpu_free(&dout);
}
