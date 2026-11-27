/* Pinned sources, async and streamed uploads, and the L2-evicting launch.
 * Included by `mapal_gpu.c` at this place in its translation unit;
 * not compiled on its own. */

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
    pins[n_pins].h = h;
    pins[n_pins++].bytes = bytes;
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

static int pinned_src(const void *h) {
  const unsigned char *p = h;
  for (size_t i = 0; i < n_pins; i++) {
    const unsigned char *b = pins[i].h;
    if (p >= b && p < b + pins[i].bytes)
      return 1;
  }
  return 0;
}

/* `async-h2d`: `bytes` of host `h` into device `d`, enqueued on the legacy
 * stream (program order) without waiting for the device. The source is staged
 * before the call returns, so the host may rewrite it at once: into a slot of
 * the page-locked ring (`stage`), else by the driver (a pageable source: CUDA
 * copies it to a staging buffer immediately, with no device synchronization).
 * A page-locked source too large for a slot would be read by the DMA after
 * the return: it takes the synchronous copy. */
void mapal_gpu_h2d_async(uint64_t d, const void *h, uint64_t bytes) {
  load_module();
  if (!(stage_buf && bytes <= STAGE_SLOT) && pinned_src(h)) {
    mapal_gpu_h2d(d, h, bytes);
    return;
  }
  double t0 = now_ms();
  if (stage_buf && bytes <= STAGE_SLOT) {
    unsigned k = stage_next++ % STAGE_SLOTS;
    if (stage_used[k])
      CK(cuEventSynchronize(stage_ev[k]));
    unsigned char *s = stage_buf + (size_t)k * STAGE_SLOT;
    memcpy(s, h, bytes);
    CK(cuMemcpyHtoDAsync((CUdeviceptr)d, s, bytes, streams[0]));
    CK(cuEventRecord(stage_ev[k], streams[0]));
    stage_used[k] = 1;
    tot.n_staged++;
  } else {
    CK(cuMemcpyHtoDAsync((CUdeviceptr)d, h, bytes, streams[0]));
  }
  legacy_ops++;
  last_kernel[0] = 0;
  double ms = now_ms() - t0;
  tot.h2d += ms;
  tot.in += bytes;
  tot.n_h2d++;
  tot.n_async++;
  if (trace)
    fprintf(stderr, "mapal_gpu_trace op=h2d_async ms=%.3f bytes=%llu\n", ms,
            (unsigned long long)bytes);
}

/* `chain-h2d`: an upload every reader of which is a site of chain `stream`,
 * enqueued on that chain's stream — staged as `mapal_gpu_h2d_async` stages —
 * so another chain never waits for it. Its buffer is fresh (a function with
 * chains has no arena) and its readers are on the same stream, so nothing
 * else orders against it; a free waits for every stream (`chains_busy`). */
void mapal_gpu_h2d_on(uint32_t stream, uint64_t d, const void *h,
                      uint64_t bytes) {
  load_module();
  unsigned k = slot_of(stream);
  int staged = stage_buf && bytes <= STAGE_SLOT;
  if (!staged && pinned_src(h)) {
    mapal_gpu_h2d(d, h, bytes);
    return;
  }
  double t0 = now_ms();
  if (staged) {
    unsigned i = stage_next++ % STAGE_SLOTS;
    if (stage_used[i])
      CK(cuEventSynchronize(stage_ev[i]));
    unsigned char *s = stage_buf + (size_t)i * STAGE_SLOT;
    memcpy(s, h, bytes);
    CK(cuMemcpyHtoDAsync((CUdeviceptr)d, s, bytes, streams[k]));
    CK(cuEventRecord(stage_ev[i], streams[k]));
    stage_used[i] = 1;
    tot.n_staged++;
  } else {
    CK(cuMemcpyHtoDAsync((CUdeviceptr)d, h, bytes, streams[k]));
  }
  last_kernel[k] = 0;
  if (k)
    chains_busy = 1;
  else
    legacy_ops++;
  double ms = now_ms() - t0;
  tot.h2d += ms;
  tot.in += bytes;
  tot.n_h2d++;
  tot.n_async++;
  if (trace)
    fprintf(stderr, "mapal_gpu_trace op=h2d_on ms=%.3f bytes=%llu stream=%u\n",
            ms, (unsigned long long)bytes, k);
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
  /* A kernel reading tensor maps launches plainly (its maps encoded there). */
  if (tmap_of(fn, name)->n) {
    mapal_gpu_launch(id, a, b, out, gx, gy, block, smem);
    return;
  }
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

