/* Device-computed producers, free, the trap word and sync.
 * Included by `mapal_gpu.c` at this place in its translation unit;
 * not compiled on its own. */

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
  if (arena_member(slot)) {
    *slot = 0;
    return;
  }
  double t0 = now_ms();
  /* A chain stream may still read the buffer: free only when every stream
   * is done (cuMemFree's own wait is not documented to cover non-blocking
   * streams). The emitter moves frees past the last chained launch. */
  if (chains_busy) {
    CK(cuCtxSynchronize());
    chains_busy = 0;
  }
  {
    CUdeviceptr base;
    size_t size = 0;
    if (cuMemGetAddressRange(&base, &size, (CUdeviceptr)*slot) == CUDA_SUCCESS &&
        size <= live_bytes)
      live_bytes -= size;
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

/* bf16-class `device-trap`: a cell that checks an unproven index records a
 * trap into the module's word `mapal_dev_trap` (i64::MAX: none; else
 * `topo << 8 | kind`). Its launch arms the next host checkpoint (a print, the
 * clock, a trap, the exit: `mapal_rt_checkpoint_hook`), which waits for the
 * device and reads the word — a recorded trap is raised there, the oracle's
 * outcome. A copy-out is a checkpoint too (`mapal_gpu_d2h`, `_d2h_on`): the
 * host computes on what it copies. Unarmed, a checkpoint costs nothing. */
extern void mapal_trap(uint32_t kind) __attribute__((noreturn));
extern void mapal_rt_checkpoint_hook(void (*f)(void));
static int trap_armed, trap_hooked;
static long long trap_word(void);
static void trap_poll(void) {
  if (!trap_armed)
    return;
  trap_armed = 0;
  long long v = trap_word();
  if (v != 0x7fffffffffffffffLL)
    mapal_trap((uint32_t)(v & 0xff));
}
static void trap_exit(void) {
  if (!trap_armed)
    return;
  trap_armed = 0;
  long long v = trap_word();
  if (v != 0x7fffffffffffffffLL) {
    fflush(stdout);
    fprintf(stderr, "mapal trap: %s\n", (v & 0xff) == 0 ? "div_zero" : "index_oob");
    _exit(101);
  }
}
void mapal_gpu_trap_arm(void) {
  trap_armed = 1;
  if (!trap_hooked) {
    trap_hooked = 1;
    mapal_rt_checkpoint_hook(trap_poll);
    atexit(trap_exit);
  }
}

static long long trap_word(void) {
  static CUdeviceptr word;
  size_t sz = 0;
  long long v = 0x7fffffffffffffffLL;
  if (!word && cuModuleGetGlobal(&word, &sz, module, "mapal_dev_trap") != CUDA_SUCCESS)
    return v;
  CK(cuCtxSynchronize());
  CK(cuMemcpyDtoH(&v, word, sizeof v));
  return v;
}
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

