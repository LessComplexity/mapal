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
 *  - cpu-gpu-boundary: `preload` resolves every kernel at module load;
 *    `entry-arena` makes the arena at function entry and places loop-cone
 *    buffers in it (`mapal_gpu_free` releases a member with its arena);
 *    `async-h2d` (`mapal_gpu_h2d_async`) enqueues an upload without waiting
 *    for the device, the source staged at the call (`stage`: a page-locked
 *    ring; else the driver's staging of a pageable source).
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
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "mapal_choice.h"

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
  unsigned long long n_preload, n_async, n_staged; /* cpu-gpu-boundary */
  unsigned long long n_tmap; /* hopper-wgmma-tma: tensor maps encoded */
  double warm;               /* sweep-gemm-2 `warm-zero`: the init thread's zero */
  /* hopper-launch: host time a launch spends on its tensor maps (recipe
   * lookup + encodes, outside the launch's own `ms`), the init thread's TMA
   * warm, and the recipes read from the module text at load. */
  double tmap, tma_warm;
  unsigned long long n_recipe;
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
          "l2_windows=%llu l2_persist=%llu pin_ms=%.3f pins=%llu events=%llu pdl=%llu "
          "preloaded=%llu h2d_async=%llu staged=%llu tmaps=%llu warm_ms=%.3f "
          "tmap_ms=%.3f tma_warm_ms=%.3f recipes=%llu\n",
          flags, tot.load, tot.lead, tot.init, tot.ctx, tot.module, tot.alloc,
          tot.h2d, tot.kernel, tot.gen, tot.d2h, tot.free, tot.in, tot.out,
          tot.n_h2d, tot.n_d2h, tot.n_launch, tot.n_gen, tot.n_alloc, tot.n_l2,
          tot.l2_persist, tot.pin, tot.n_pin, tot.n_event, tot.n_pdl,
          tot.n_preload, tot.n_async, tot.n_staged, tot.n_tmap, tot.warm,
          tot.tmap, tot.tma_warm, tot.n_recipe);
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
enum {
  EAGER = 1,
  STREAMS = 2,
  PDL = 4,
  PRELOAD = 8,
  STAGE = 16,
  WARM_ZERO = 32,
  TMA_COLD = 64,
  ZERO_KERNEL = 128 /* gaps-disasm C1 `zero-host`: the module zeroes, by kernel */
};

/* sweep-gemm-2 `warm-zero` (`WARM_ZERO`, set when the module's launches zero
 * an output): the driver initializes its memset path lazily — measured on
 * CUDA 13 / rtx-5080, a process's first `cuMemsetD8Async` of 8 KiB or more
 * costs 13–17 us on the host, every later one 2–4 us whatever its size (a
 * fill below 8 KiB pays its own first ~6 us), and a thread's first fill
 * still 7–8 us after another thread's. One fill of each kind into a kept
 * scratch buffer on the init thread (the process's part), then again on the
 * thread that loads the module (its own part, before the first site): the
 * program's zeros run at their steady cost. */
#define WARM_ZERO_BYTES (64 * 1024)
static CUdeviceptr warm_buf;

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

#include "mapal_gpu_fills.inc"
#include "mapal_gpu_modscan.inc"
#include "mapal_gpu_init.inc"
#include "mapal_gpu_memory.inc"
#include "mapal_gpu_launch.inc"
#include "mapal_gpu_transfer.inc"
#include "mapal_gpu_gen.inc"
#include "mapal_gpu_sites.inc"
