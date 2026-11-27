/* Device memory: alloc, arena, the copies, zero, events.
 * Included by `mapal_gpu.c` at this place in its translation unit;
 * not compiled on its own. */

static uint64_t dev_alloc(uint64_t *slot, uint64_t bytes, const char *op) {
  load_module();
  if (*slot)
    return *slot;
  static int peak_env = -1;
  if (peak_env < 0) {
    const char *e = getenv("MAPAL_GPU_PEAK");
    peak_env = e && atoi(e) > 0;
    if (peak_env)
      atexit(peak_line);
  }
  double t0 = now_ms();
  CUdeviceptr d;
  CK(cuMemAlloc(&d, bytes));
  *slot = (uint64_t)d;
  live_bytes += bytes;
  n_peak_allocs++;
  if (live_bytes > peak_bytes)
    peak_bytes = live_bytes;
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
static struct arena {
  uint64_t *slot, base, bytes;
} *arenas;
static size_t n_arenas, cap_arenas;
uint64_t mapal_gpu_arena(uint64_t *slot, uint64_t bytes) {
  if (*slot)
    return *slot;
  dev_alloc(slot, bytes, "arena");
  if (n_arenas == cap_arenas) {
    cap_arenas = cap_arenas ? 2 * cap_arenas : 8;
    arenas = realloc(arenas, cap_arenas * sizeof *arenas);
    if (!arenas)
      exit(102);
  }
  arenas[n_arenas++] = (struct arena){slot, *slot, bytes};
  return *slot;
}

/* `entry-arena`: a slot holding an arena member's address (a loop-cone
 * buffer, possibly moved by a back edge) is released with its arena: emptied,
 * never freed on its own. The arena's own slot frees the allocation. */
static int arena_member(uint64_t *slot) {
  for (size_t i = 0; i < n_arenas; i++) {
    if (arenas[i].slot == slot) {
      arenas[i] = arenas[--n_arenas];
      return 0;
    }
    if (*slot >= arenas[i].base && *slot < arenas[i].base + arenas[i].bytes)
      return 1;
  }
  return 0;
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

/* caching-memory `retain-pieces`: `height` rows of `width` bytes from `src`
 * (rows `spitch` apart) to `dst` (rows `dpitch` apart), device to device, on
 * the legacy stream behind the kernel that wrote them. */
void mapal_gpu_copy2d(uint64_t dst, uint64_t dpitch, uint64_t src,
                      uint64_t spitch, uint64_t width, uint64_t height) {
  load_module();
  double t0 = now_ms();
  CUDA_MEMCPY2D c;
  memset(&c, 0, sizeof c);
  c.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  c.srcDevice = (CUdeviceptr)src;
  c.srcPitch = spitch;
  c.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  c.dstDevice = (CUdeviceptr)dst;
  c.dstPitch = dpitch;
  c.WidthInBytes = width;
  c.Height = height;
  CK(cuMemcpy2DAsync(&c, streams[0]));
  legacy_ops++;
  last_kernel[0] = 0;
  if (trace)
    fprintf(stderr, "mapal_gpu_trace op=copy2d ms=%.3f bytes=%llu\n",
            now_ms() - t0, (unsigned long long)(width * height));
}

/* S63 `splitk-atomic` / `streamk` (ADR-0040 D4): zero `bytes` of the device
 * buffer `d` on the legacy stream, ahead of the kernel whose partials add into
 * it — the runtime's zero kernel (`zero-host`; the launch behind it may then
 * be a dependent (PDL) launch: its `griddepcontrol.wait` waits for the whole
 * zero grid), or the memset API. */
void mapal_gpu_zero(uint64_t d, uint64_t bytes) {
  load_module();
  double t0 = now_ms();
  zero_on(0, (CUdeviceptr)d, bytes);
  legacy_ops++;
  if (trace)
    fprintf(stderr, "mapal_gpu_trace op=zero ms=%.3f bytes=%llu\n", now_ms() - t0,
            (unsigned long long)bytes);
}

/* gaps-disasm C1 `zero-hoist`: a site's zero issued ahead of it (after the
 * last access to its bytes), skipped where its table `c`'s pick is decided —
 * at load (`load-dispatch`) or by an earlier execution — and adds nothing
 * into the output (a workspace split, an ordered candidate). Undecided (a
 * site that will trial) or without a table: the zero. */
void mapal_gpu_zero_for(const struct mapal_cand *c, uint64_t d, uint64_t bytes) {
  if (c) {
    struct choice_memo *m = choice_memo_of(c);
    if (m->c == c && !c[m->pick].zero)
      return;
  }
  mapal_gpu_zero(d, bytes);
}

/* `device-trap`: an armed copy-out is a checkpoint (`trap_poll`, below). */
static void trap_poll(void);

/* `bytes` from byte `off` of the device buffer `d` into host `h`.
 * Synchronous: it waits for every launch before it on the stream. A copy-out
 * hands the host device values, so a recorded trap is raised here (armed
 * only: a checking cell ran since the last checkpoint) — the host never
 * computes on the element a trapped read stands in for (a loop whose exit
 * reads it may never end; the oracle stopped at the read). */
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
  trap_poll();
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

