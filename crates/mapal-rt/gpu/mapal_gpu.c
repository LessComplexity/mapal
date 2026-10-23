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
 *  - S54 `streams` (R7): sites on independent chains launch on their own
 *    non-blocking streams (`mapal_gpu_launchv`'s stream index; 0 is the legacy
 *    stream, i.e. program order). A chain stream joins program order at each
 *    launch: if any legacy-stream operation was enqueued since its last join,
 *    the launch first waits on an event recorded there. Cross-chain edges are
 *    the emitter's `mapal_gpu_event`; a chain output is copied out on its own
 *    stream (`mapal_gpu_d2h_on`). `pinned`: `mapal_gpu_pin` page-locks a
 *    crossing host array once; `mapal_gpu_unpin` releases every pin.
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
 * kernel's; off, launches stay asynchronous. MAPAL_GPU_TRACE=2 (S62): the same
 * lines without that synchronize, so a launch's time is its enqueue (CPU
 * submission cost, `now_ms` bookkeeping included). `load_ms` is what the program
 * waited for the context and module (`wait`): all of `init + ctx + module` on
 * the lazy path, that minus `lead` (host work from `mapal_gpu_init` to the
 * first site) under `eager-init`. */
static int trace = -1;
static struct {
  double load, alloc, h2d, kernel, gen, d2h, free;
  double lead, init, ctx, module;
  unsigned long long in, out, n_h2d, n_d2h, n_launch, n_gen, n_alloc;
  unsigned long long n_l2, l2_persist; /* S54 `l2-window` */
  double pin;
  unsigned long long n_pin, n_event, n_pdl;
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
          "bytes_out=%llu h2d=%llu d2h=%llu launches=%llu gens=%llu allocs=%llu "
          "l2_windows=%llu l2_persist=%llu pin_ms=%.3f pins=%llu events=%llu pdl=%llu\n",
          flags, tot.load, tot.lead, tot.init, tot.ctx, tot.module, tot.alloc,
          tot.h2d, tot.kernel, tot.gen, tot.d2h, tot.free, tot.in, tot.out,
          tot.n_h2d, tot.n_d2h, tot.n_launch, tot.n_gen, tot.n_alloc, tot.n_l2,
          tot.l2_persist, tot.pin, tot.n_pin, tot.n_event, tot.n_pdl);
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
enum { EAGER = 1, STREAMS = 2, PDL = 4 };

/* Chain streams (`STREAMS`): created with the context. Eight because that is
 * the driver's default count of hardware work queues
 * (CUDA_DEVICE_MAX_CONNECTIONS): a ninth stream would share a queue, so chain
 * `s` takes stream `1 + (s - 1) % 8` and two chains that meet in one only
 * serialize. Slot 0 is the legacy stream. */
#define MAPAL_GPU_STREAMS 8
static CUstream streams[MAPAL_GPU_STREAMS + 1];
/* Program order on the legacy stream: operations enqueued there, and the count
 * each chain stream has joined (waited behind). */
static unsigned long long legacy_ops, joined[MAPAL_GPU_STREAMS + 1];
/* Whether the last operation enqueued on a slot was a kernel (`pdl`). */
static int last_kernel[MAPAL_GPU_STREAMS + 1];
/* A chain stream may hold work the host has not waited for: the next free
 * waits for the device once, then frees run as on one stream. */
static int chains_busy;
/* One reusable event per recording slot (a wait captures the event's state
 * when it is enqueued, so re-recording it later is safe). */
static CUevent evs[MAPAL_GPU_STREAMS + 1];
static unsigned slot_of(uint32_t s) {
  return s ? 1 + (s - 1) % MAPAL_GPU_STREAMS : 0;
}

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
  if (flags & STREAMS)
    for (int i = 1; i <= MAPAL_GPU_STREAMS; i++)
      CK(cuStreamCreate(&streams[i], CU_STREAM_NON_BLOCKING));
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
    const char *t = getenv("MAPAL_GPU_TRACE");
    trace = t ? atoi(t) : 0;
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

static uint64_t dev_alloc(uint64_t *slot, uint64_t bytes, const char *op) {
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
    fprintf(stderr, "mapal_gpu_trace op=%s ms=%.3f bytes=%llu\n", op, ms,
            (unsigned long long)bytes);
  return *slot;
}

/* The device buffer of `*slot`, allocated on first use: a slot is one object's
 * device copy, so a producer inside a loop reuses its buffer every iteration. */
uint64_t mapal_gpu_alloc(uint64_t *slot, uint64_t bytes) {
  return dev_alloc(slot, bytes, "alloc");
}

/* S54 `arena`: the function's one allocation, laid out at compile time (each
 * straight-line object at its offset, from death positions). Made on the first
 * ask like any slot; released by `mapal_gpu_free` at function exit. */
uint64_t mapal_gpu_arena(uint64_t *slot, uint64_t bytes) {
  return dev_alloc(slot, bytes, "arena");
}

void mapal_gpu_h2d(uint64_t d, const void *h, uint64_t bytes) {
  double t0 = now_ms();
  CK(cuMemcpyHtoD((CUdeviceptr)d, h, bytes));
  legacy_ops++;
  last_kernel[0] = 0;
  double ms = now_ms() - t0;
  tot.h2d += ms;
  tot.in += bytes;
  tot.n_h2d++;
  if (trace)
    fprintf(stderr, "mapal_gpu_trace op=h2d ms=%.3f bytes=%llu\n", ms,
            (unsigned long long)bytes);
}

/* S63 `splitk-atomic` / `streamk` (ADR-0040 D4): zero `bytes` of the device
 * buffer `d` on the legacy stream, ahead of the kernel whose partials add into
 * it. Not a kernel: the launch behind it is not a dependent (PDL) launch. */
void mapal_gpu_zero(uint64_t d, uint64_t bytes) {
  load_module();
  CK(cuMemsetD8Async((CUdeviceptr)d, 0, bytes, streams[0]));
  legacy_ops++;
  last_kernel[0] = 0;
}

/* `bytes` from byte `off` of the device buffer `d` into host `h`.
 * Synchronous: it waits for every launch before it on the stream. */
void mapal_gpu_d2h(void *h, uint64_t d, uint64_t off, uint64_t bytes) {
  double t0 = now_ms();
  CK(cuMemcpyDtoH(h, (CUdeviceptr)(d + off), bytes));
  last_kernel[0] = 0;
  double ms = now_ms() - t0;
  tot.d2h += ms;
  tot.out += bytes;
  tot.n_d2h++;
  if (trace)
    fprintf(stderr, "mapal_gpu_trace op=d2h ms=%.3f bytes=%llu\n", ms,
            (unsigned long long)bytes);
}

/* `wait` waits for everything `record` has enqueued so far. */
static void event(unsigned record, unsigned wait, int automatic) {
  double t0 = now_ms();
  if (!evs[record])
    CK(cuEventCreate(&evs[record], CU_EVENT_DISABLE_TIMING));
  CK(cuEventRecord(evs[record], streams[record]));
  CK(cuStreamWaitEvent(streams[wait], evs[record], 0));
  last_kernel[wait] = 0;
  chains_busy = 1;
  tot.n_event++;
  if (trace)
    fprintf(stderr, "mapal_gpu_trace op=event record=%u wait=%u auto=%d ms=%.3f\n",
            record, wait, automatic, now_ms() - t0);
}

/* A cross-chain edge (the emitter's): stream `wait` waits for everything
 * stream `record` has enqueued so far. */
void mapal_gpu_event(uint32_t record, uint32_t wait) {
  load_module();
  event(slot_of(record), slot_of(wait), 0);
}

/* One tile kernel on stream slot `k`. A chain stream first joins program
 * order (see the header). Under `PDL` (sm_90+; every tile kernel then opens
 * with `griddepcontrol.wait`), a kernel right behind another kernel on its
 * stream is a programmatic dependent launch: it may start while the first
 * drains, and waits for it at its entry. */
static void launch_on(unsigned k, uint64_t id, void **args, uint32_t gx,
                      uint32_t gy, uint32_t block, uint32_t smem) {
  load_module();
  char name[40];
  CUfunction fn = kernel("mapal_tile", id, smem, name, sizeof name);
  if (k && joined[k] != legacy_ops) {
    event(0, k, 1);
    joined[k] = legacy_ops;
  }
  int pdl = (flags & PDL) && last_kernel[k];
  double t0 = now_ms();
  if (pdl) {
    CUlaunchAttribute at[1];
    at[0].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
    at[0].value.programmaticStreamSerializationAllowed = 1;
    CUlaunchConfig cfg = {0};
    cfg.gridDimX = gx;
    cfg.gridDimY = gy;
    cfg.gridDimZ = 1;
    cfg.blockDimX = block;
    cfg.blockDimY = 1;
    cfg.blockDimZ = 1;
    cfg.sharedMemBytes = smem;
    cfg.hStream = streams[k];
    cfg.attrs = at;
    cfg.numAttrs = 1;
    CK(cuLaunchKernelEx(&cfg, fn, args, 0));
    tot.n_pdl++;
  } else {
    CK(cuLaunchKernel(fn, gx, gy, 1, block, 1, 1, smem, streams[k], args, 0));
  }
  last_kernel[k] = 1;
  if (!k)
    legacy_ops++;
  else
    chains_busy = 1;
  if (trace == 1)
    CK(cuCtxSynchronize());
  double ms = now_ms() - t0;
  tot.kernel += ms;
  tot.n_launch++;
  if (trace)
    fprintf(stderr, "mapal_gpu_trace op=launch kernel=%s ms=%.3f stream=%u pdl=%d\n",
            name, ms, k, pdl);
}

void mapal_gpu_launch(uint64_t id, uint64_t a, uint64_t b, uint64_t out,
                      uint32_t gx, uint32_t gy, uint32_t block, uint32_t smem) {
  CUdeviceptr da = a, db = b, dout = out;
  void *args[] = {&da, &db, &dout};
  launch_on(0, id, args, gx, gy, block, smem);
}

/* The S54 launch (plan-s54 §4): `nptr` device pointers, on chain `stream`. */
void mapal_gpu_launchv(uint64_t id, uint32_t stream, uint32_t nptr,
                       const uint64_t *ptrs, uint32_t gx, uint32_t gy,
                       uint32_t block, uint32_t smem) {
  CUdeviceptr d[16];
  void *args[16];
  if (nptr > 16) {
    fprintf(stderr, "mapal_gpu: launchv with %u pointers\n", nptr);
    exit(102);
  }
  for (uint32_t i = 0; i < nptr; i++) {
    d[i] = ptrs[i];
    args[i] = &d[i];
  }
  launch_on(slot_of(stream), id, args, gx, gy, block, smem);
}

/* `d2h` on chain `stream` (its producer's): queued behind that chain's
 * kernels, and waited for, so the host sees the values on return. */
void mapal_gpu_d2h_on(uint32_t stream, void *h, uint64_t d, uint64_t off,
                      uint64_t bytes) {
  unsigned k = slot_of(stream);
  double t0 = now_ms();
  CK(cuMemcpyDtoHAsync(h, (CUdeviceptr)(d + off), bytes, streams[k]));
  CK(cuStreamSynchronize(streams[k]));
  last_kernel[k] = 0;
  double ms = now_ms() - t0;
  tot.d2h += ms;
  tot.out += bytes;
  tot.n_d2h++;
  if (trace)
    fprintf(stderr, "mapal_gpu_trace op=d2h ms=%.3f bytes=%llu stream=%u\n", ms,
            (unsigned long long)bytes, k);
}

/* `pinned`: page-lock a host array that crosses the bus, once. Best effort —
 * a range the driver refuses (already registered through a neighbour's page,
 * say) is copied pageable, which is slower and exactly as correct; the trace
 * says which. */
static struct pin { const void *h; } *pins;
static size_t n_pins, cap_pins;
void mapal_gpu_pin(const void *h, uint64_t bytes) {
  load_module();
  for (size_t i = 0; i < n_pins; i++)
    if (pins[i].h == h)
      return;
  double t0 = now_ms();
  CUresult r = cuMemHostRegister((void *)h, bytes, 0);
  double ms = now_ms() - t0;
  tot.pin += ms;
  if (r == CUDA_SUCCESS) {
    if (n_pins == cap_pins) {
      cap_pins = cap_pins ? 2 * cap_pins : 16;
      pins = realloc(pins, cap_pins * sizeof *pins);
      if (!pins)
        exit(102);
    }
    pins[n_pins++].h = h;
    tot.n_pin++;
  }
  if (trace)
    fprintf(stderr, "mapal_gpu_trace op=pin ms=%.3f bytes=%llu ok=%d\n", ms,
            (unsigned long long)bytes, r == CUDA_SUCCESS);
}

/* Release every pin: the emitter calls it before the function that pinned
 * returns its frame. */
void mapal_gpu_unpin(void) {
  for (size_t i = 0; i < n_pins; i++)
    CK(cuMemHostUnregister((void *)pins[i].h));
  n_pins = 0;
}

/* S54 `l2-window`: `mapal_gpu_launch` with an L2 access-policy window over
 * `win_bytes` of the buffer at `win` (the emitter picks the buffer the graph
 * proves read again: a chain's intermediate, else the operand the most blocks
 * read). Lines it touches are marked persisting, so later traffic evicts them
 * last. The persisting set-aside and the largest window are the device's own
 * attributes; the set-aside is reserved once, at the first windowed launch, so
 * a program without one runs exactly the plain path. A part that reports no
 * persisting L2 launches plainly. */
static size_t persist_bytes, window_max;
void mapal_gpu_launch_l2(uint64_t id, uint64_t a, uint64_t b, uint64_t out,
                         uint32_t gx, uint32_t gy, uint32_t block,
                         uint32_t smem, uint64_t win, uint64_t win_bytes) {
  load_module();
  static int ready;
  if (!ready) {
    CUdevice dev;
    int p = 0, w = 0;
    CK(cuCtxGetDevice(&dev));
    CK(cuDeviceGetAttribute(
        &p, CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE, dev));
    CK(cuDeviceGetAttribute(
        &w, CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE, dev));
    persist_bytes = (size_t)p;
    window_max = (size_t)w;
    /* A context that refuses the set-aside (a shared or partitioned part)
     * runs the plain path: the window is a policy, never a value. */
    if (persist_bytes &&
        cuCtxSetLimit(CU_LIMIT_PERSISTING_L2_CACHE_SIZE, persist_bytes) !=
            CUDA_SUCCESS)
      persist_bytes = 0;
    tot.l2_persist = persist_bytes;
    if (trace)
      fprintf(stderr, "mapal_gpu_trace op=l2 persist_bytes=%zu window_max=%zu\n",
              persist_bytes, window_max);
    ready = 1;
  }
  if (!persist_bytes || !window_max) {
    mapal_gpu_launch(id, a, b, out, gx, gy, block, smem);
    return;
  }
  char name[40];
  CUfunction fn = kernel("mapal_tile", id, smem, name, sizeof name);
  CUdeviceptr da = a, db = b, dout = out;
  void *args[] = {&da, &db, &dout};
  size_t bytes = win_bytes < window_max ? (size_t)win_bytes : window_max;
  CUlaunchAttribute attr = {0};
  attr.id = CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW;
  attr.value.accessPolicyWindow.base_ptr = (void *)(uintptr_t)win;
  attr.value.accessPolicyWindow.num_bytes = bytes;
  attr.value.accessPolicyWindow.hitRatio =
      bytes <= persist_bytes ? 1.0f : (float)persist_bytes / (float)bytes;
  attr.value.accessPolicyWindow.hitProp = CU_ACCESS_PROPERTY_PERSISTING;
  attr.value.accessPolicyWindow.missProp = CU_ACCESS_PROPERTY_STREAMING;
  CUlaunchConfig cfg = {0};
  cfg.gridDimX = gx;
  cfg.gridDimY = gy;
  cfg.gridDimZ = 1;
  cfg.blockDimX = block;
  cfg.blockDimY = 1;
  cfg.blockDimZ = 1;
  cfg.sharedMemBytes = smem;
  cfg.attrs = &attr;
  cfg.numAttrs = 1;
  double t0 = now_ms();
  if (cuLaunchKernelEx(&cfg, fn, args, 0) != CUDA_SUCCESS) {
    if (trace)
      fprintf(stderr, "mapal_gpu_trace op=l2 refused: plain launches\n");
    window_max = 0;
    mapal_gpu_launch(id, a, b, out, gx, gy, block, smem);
    return;
  }
  if (trace == 1)
    CK(cuCtxSynchronize());
  double ms = now_ms() - t0;
  tot.kernel += ms;
  tot.n_launch++;
  tot.n_l2++;
  if (trace)
    fprintf(stderr,
            "mapal_gpu_trace op=launch kernel=%s ms=%.3f l2_window=%zu\n",
            name, ms, bytes);
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
  legacy_ops++;
  last_kernel[0] = 1;
  if (trace == 1)
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
  /* A chain stream may still read the buffer: free only when every stream
   * is done (cuMemFree's own wait is not documented to cover non-blocking
   * streams). The emitter moves frees past the last chained launch. */
  if (chains_busy) {
    CK(cuCtxSynchronize());
    chains_busy = 0;
  }
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
  if (init_state == 2) {
    CK(cuCtxSynchronize());
    for (int i = 0; i <= MAPAL_GPU_STREAMS; i++)
      joined[i] = legacy_ops;
    chains_busy = 0;
  }
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
