/* Tensor maps and the launch entries (plain, cooperative, vector).
 * Included by `mapal_gpu.c` at this place in its translation unit;
 * not compiled on its own. */

/* hopper-wgmma-tma: the tensor maps a kernel's box copies read
 * (`cp.async.bulk.tensor`). The device unit carries each such kernel's recipe
 * as the global `<kernel>_tmap` — `[count, then TMAP_WORDS words a map: the
 * pointer argument, element bytes, rank, swizzle bytes, base offset bytes,
 * dims[5], byte strides of dims 1..4, box[5], 0]` — so the host ABI is
 * unchanged: the launch encodes each map (`cuTensorMapEncodeTiled`) from the
 * pointer it is handed and passes it by value after the pointers (a 128-byte
 * `__grid_constant__` parameter).
 *
 * hopper-launch: a map's encode depends only on the recipe and the buffer's
 * address, so each map keeps the last `TMAP_SLOTS` addresses it was encoded
 * for: a buffer whose address is fixed (an arena member, a weight, a cached
 * allocation) is encoded once for the program's life, and a loop that
 * alternates a map between buffers (a carried state's two halves) re-encodes
 * nothing (`tmaps=` on the trace's total line counts encodes; `tmap_ms` the
 * host time a launch spends here). The recipe itself is the module text's
 * (`tmap_scan`, at load). */
#define TMAP_SLOTS 4
struct tmap_e {
  CUfunction fn;
  uint32_t n;
  uint64_t rec[TMAP_MAX][TMAP_WORDS];
  CUdeviceptr ptr[TMAP_MAX][TMAP_SLOTS];
  CUtensorMap map[TMAP_MAX][TMAP_SLOTS];
  uint32_t next[TMAP_MAX]; /* the slot the next new address replaces */
};
/* The table: one entry a listed recipe (TMAP_KERNELS when the module lists
 * more names than `tmap_scan` holds), made and touched at load
 * (`tmap_table`). Made at the first launch, its zeroed pages (768 KiB at 256
 * entries) cost that launch 28-35 us on the RTX 5080, up to 260 us when the
 * init ran on the program's own thread (`hopper-launch.md`). */
static struct tmap_e *tmaps;
#define TMAP_KERNELS 256
static size_t n_tmaps, cap_tmaps;
static void tmap_table(void) {
  cap_tmaps = n_tmap_names > 0   ? (size_t)n_tmap_names
              : n_tmap_names < 0 ? TMAP_KERNELS
                                 : 0;
  if (!cap_tmaps)
    return;
  if (!(tmaps = malloc(cap_tmaps * sizeof *tmaps))) {
    fprintf(stderr, "mapal_gpu: tensor-map table: out of memory\n");
    exit(102);
  }
  memset(tmaps, 0, cap_tmaps * sizeof *tmaps);
}
static struct tmap_e tmap_none; /* a kernel the module declares no recipe for */
static struct tmap_e *tmap_of(CUfunction fn, const char *name) {
  int ix = tmap_index(name);
  if (ix == -2)
    return &tmap_none;
  for (size_t i = 0; i < n_tmaps; i++)
    if (tmaps[i].fn == fn)
      return &tmaps[i];
  if (n_tmaps == cap_tmaps) {
    fprintf(stderr, "mapal_gpu: tensor-map table full\n");
    exit(102);
  }
  struct tmap_e *e = &tmaps[n_tmaps++];
  memset(e, 0, sizeof *e);
  e->fn = fn;
  uint64_t buf[TMAP_REC_WORDS] = {0};
  char g[64];
  snprintf(g, sizeof g, "%s_tmap", name);
  if (ix >= 0 && tmap_ok[ix]) {
    memcpy(buf, tmap_rec[ix], sizeof buf);
  } else {
    /* A recipe the module text does not spell as `tmap_parse` reads. */
    CUdeviceptr d;
    size_t bytes;
    if (cuModuleGetGlobal(&d, &bytes, module, g) != CUDA_SUCCESS)
      return e;
    if (bytes > sizeof buf || bytes < 8) {
      fprintf(stderr, "mapal_gpu: %s: %zu bytes\n", g, bytes);
      exit(102);
    }
    CK(cuMemcpyDtoH(buf, d, bytes));
  }
  e->n = (uint32_t)buf[0];
  if (e->n > TMAP_MAX) {
    fprintf(stderr, "mapal_gpu: %s: %u maps\n", g, e->n);
    exit(102);
  }
  memcpy(e->rec, buf + 1, e->n * TMAP_WORDS * sizeof(uint64_t));
  for (uint32_t m = 0; m < e->n; m++)
    for (uint32_t s = 0; s < TMAP_SLOTS; s++)
      e->ptr[m][s] = ~(CUdeviceptr)0;
  return e;
}
/* One map of recipe `r` (TMAP_WORDS words) over the bytes at `p`. */
static void tmap_encode(CUtensorMap *out, const uint64_t *r, CUdeviceptr p,
                        uint32_t m) {
  cuuint64_t dims[5], strides[4];
  cuuint32_t box[5], es[5];
  uint32_t rank = (uint32_t)r[2];
  for (uint32_t i = 0; i < 5; i++) {
    dims[i] = r[5 + i];
    box[i] = (cuuint32_t)r[14 + i];
    es[i] = 1;
  }
  for (uint32_t i = 0; i < 4; i++)
    strides[i] = r[10 + i];
  CUtensorMapSwizzle sw = r[3] == 128 ? CU_TENSOR_MAP_SWIZZLE_128B
                          : r[3] == 64 ? CU_TENSOR_MAP_SWIZZLE_64B
                          : r[3] == 32 ? CU_TENSOR_MAP_SWIZZLE_32B
                                       : CU_TENSOR_MAP_SWIZZLE_NONE;
  CUtensorMapDataType ty = r[1] == 2 ? CU_TENSOR_MAP_DATA_TYPE_UINT16
                                     : CU_TENSOR_MAP_DATA_TYPE_UINT32;
  CUresult rc = cuTensorMapEncodeTiled(
      out, ty, rank, (void *)(uintptr_t)p, dims, strides, box, es,
      CU_TENSOR_MAP_INTERLEAVE_NONE, sw, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (rc != CUDA_SUCCESS) {
    const char *s_ = 0;
    cuGetErrorString(rc, &s_);
    fprintf(stderr,
            "mapal_gpu: cuTensorMapEncodeTiled (map %u, rank %u, ptr "
            "%llx): %s\n",
            m, rank, (unsigned long long)p, s_ ? s_ : "?");
    exit(102);
  }
  tot.n_tmap++;
}
/* `args` (the `nptr` pointer arguments) followed by `e`'s maps for this
 * launch's pointers, in `out`: each from its address's slot, encoded into
 * the oldest slot on a new address. */
static void tmap_args(struct tmap_e *e, void **args, uint32_t nptr,
                      void **out) {
  for (uint32_t i = 0; i < nptr; i++)
    out[i] = args[i];
  for (uint32_t m = 0; m < e->n; m++) {
    const uint64_t *r = e->rec[m];
    if (r[0] >= nptr) {
      fprintf(stderr, "mapal_gpu: tensor map %u names argument %llu of %u\n",
              m, (unsigned long long)r[0], nptr);
      exit(102);
    }
    CUdeviceptr p = *(CUdeviceptr *)args[r[0]] + r[4];
    uint32_t s = 0;
    while (s < TMAP_SLOTS && e->ptr[m][s] != p)
      s++;
    if (s == TMAP_SLOTS) {
      s = e->next[m];
      e->next[m] = (s + 1) % TMAP_SLOTS;
      tmap_encode(&e->map[m][s], r, p, m);
      e->ptr[m][s] = p;
    }
    out[nptr + m] = &e->map[m][s];
  }
}

/* hopper-launch `tma-warm` (unless `TMA_COLD`, the measurement arm; a module
 * whose text carries a tensor-map recipe, `tmap_scan`): the driver readies a context's tensor-memory-access
 * path at the first launch of a kernel holding a `cp.async.bulk.tensor` —
 * measured on CUDA 13 / RTX 5080 (driver 620.73), that launch's host call
 * takes 1.34–1.49 ms, every later one 3–4 us, a second such kernel's first
 * launch ~10 us more than a plain one, and one launch of any such kernel (its
 * own module, another thread) readies the path for all of them. On the H100
 * it is the 120–270 us a `wgmma-ws` program's call spent outside its kernel
 * (cross-arch 5fcb401b finding 3). So the init thread launches the runtime's
 * own one-box kernel once — a 32 × 8-word box, 128-byte swizzle, onto an
 * `mbarrier` — with one encode (the encoder's own first call is ~2 us more
 * than a later one): the program's first TMA launch runs at its steady cost. */
static const char tma_warm_ptx[] =
    ".version 8.0\n.target sm_90\n.address_size 64\n"
    ".extern .shared .align 1024 .b8 mapal_warm_smem[];\n"
    ".visible .entry mapal_tma_warm(.param .u64 o, .param .align 64 .b8 "
    "tm[128]) {\n"
    "  .reg .b64 %rd<4>; .reg .b32 %r<6>; .reg .f32 %f<2>; .reg .pred %p;\n"
    "  mov.u32 %r0, %tid.x; setp.ne.u32 %p, %r0, 0; @%p bra done;\n"
    "  mov.b64 %rd0, tm; cvta.param.u64 %rd1, %rd0;\n"
    "  mov.u32 %r1, mapal_warm_smem; add.u32 %r2, %r1, 1024;\n"
    "  mbarrier.init.shared::cta.b64 [%r2], 1;\n"
    "  fence.proxy.async.shared::cta;\n"
    "  mbarrier.arrive.expect_tx.shared::cta.b64 _, [%r2], 1024;\n"
    "  mov.u32 %r3, 0;\n"
    "  cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::"
    "complete_tx::bytes [%r1], [%rd1, {%r3, %r3}], [%r2];\n"
    "w: mbarrier.try_wait.parity.shared::cta.b64 %p, [%r2], 0; @!%p bra w;\n"
    "  ld.shared.f32 %f1, [%r1];\n"
    "  ld.param.u64 %rd2, [o]; st.global.f32 [%rd2], %f1;\n"
    "done: ret;\n}\n";
static void tma_warm(void) {
  double t0 = now_ms();
  CUmodule wm;
  CUfunction wk;
  if (cuModuleLoadData(&wm, tma_warm_ptx) != CUDA_SUCCESS ||
      cuModuleGetFunction(&wk, wm, "mapal_tma_warm") != CUDA_SUCCESS)
    return; /* a device without the copy engine: nothing to ready */
  CUdeviceptr buf;
  CK(cuMemAlloc(&buf, 4096));
  CK(cuMemsetD8Async(buf, 0, 4096, streams[0]));
  /* rank 2, u32, 128-byte swizzle, dims (32, 8), row stride 128 B, box (32, 8) */
  uint64_t r[TMAP_WORDS] = {0, 4, 2, 128, 0, 32, 8, 1, 1, 1, 128, 0, 0, 0,
                            32, 8, 1, 1, 1, 0};
  CUtensorMap tm;
  tmap_encode(&tm, r, buf, 0);
  void *args[] = {&buf, &tm};
  CK(cuLaunchKernel(wk, 1, 1, 1, 32, 1, 1, 2048, streams[0], args, 0));
  CK(cuStreamSynchronize(streams[0]));
  CK(cuMemFree(buf));
  tot.n_tmap--; /* the warm's encode is not the program's */
  tot.tma_warm = now_ms() - t0;
}

/* One tile kernel on stream slot `k`. A chain stream first joins program
 * order (see the header). Under `PDL` (sm_90+; every tile kernel then opens
 * with `griddepcontrol.wait`), a kernel right behind another kernel on its
 * stream is a programmatic dependent launch: it may start while the first
 * drains, and waits for it at its entry. */
static void launch_on(unsigned k, uint64_t id, void **args, uint32_t nptr,
                      uint32_t gx, uint32_t gy, uint32_t block, uint32_t smem) {
  load_module();
  char name[40];
  CUfunction fn = kernel("mapal_tile", id, smem, name, sizeof name);
  double tq = now_ms();
  struct tmap_e *tm = tmap_of(fn, name);
  void *targs[32 + TMAP_MAX];
  if (tm->n) {
    tmap_args(tm, args, nptr, targs);
    args = targs;
    tot.tmap += now_ms() - tq;
  }
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

/* kernel-merging `calls`: a persistent kernel (each body strides over its
 * blocks, a grid barrier between levels) needs every block resident. It
 * launches cooperatively — the driver refuses a grid it cannot co-schedule
 * rather than let the barrier deadlock — at most `gx` blocks and at most the
 * loaded kernel's own occupancy (its ptxas registers, its shared memory and
 * block) times the SMs. */
struct coop_occ_e {
  CUfunction fn;
  uint32_t block, smem;
  int per;
};
static struct coop_occ_e coop_occ[64];
static size_t n_coop_occ;
static int coop_sms;

/* The blocks of `fn` an SM holds at `block` threads and `smem` bytes (queried
 * once a recipe; `load-dispatch` asks at load for a registered table). */
static int coop_per(CUfunction fn, uint32_t block, uint32_t smem) {
  if (!coop_sms) {
    CUdevice dev;
    CK(cuCtxGetDevice(&dev));
    CK(cuDeviceGetAttribute(&coop_sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
  }
  for (size_t i = 0; i < n_coop_occ; i++)
    if (coop_occ[i].fn == fn && coop_occ[i].block == block && coop_occ[i].smem == smem)
      return coop_occ[i].per;
  int per = -1;
  CK(cuOccupancyMaxActiveBlocksPerMultiprocessor(&per, fn, (int)block, smem));
  if (n_coop_occ < sizeof coop_occ / sizeof *coop_occ)
    coop_occ[n_coop_occ++] = (struct coop_occ_e){fn, block, smem, per};
  return per;
}

static void launch_coop_on(unsigned k, uint64_t id, void **args, uint32_t gx,
                           uint32_t block, uint32_t smem) {
  load_module();
  char name[40];
  CUfunction fn = kernel("mapal_tile", id, smem, name, sizeof name);
  if (k && joined[k] != legacy_ops) {
    event(0, k, 1);
    joined[k] = legacy_ops;
  }
  int per = coop_per(fn, block, smem);
  if (per < 1) {
    fprintf(stderr, "mapal_gpu: %s: no block of %u threads, %u B fits an SM\n", name, block, smem);
    exit(102);
  }
  uint32_t g = gx < (uint32_t)(per * coop_sms) ? gx : (uint32_t)(per * coop_sms);
  if (!g)
    g = 1;
  double t0 = now_ms();
  CK(cuLaunchCooperativeKernel(fn, g, 1, 1, block, 1, 1, smem, streams[k], args));
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
    fprintf(stderr, "mapal_gpu_trace op=launch kernel=%s ms=%.3f stream=%u pdl=0 coop=%u\n",
            name, ms, k, g);
}

/* The cooperative launch over `nptr` device pointers (`launchv`'s ABI). */
void mapal_gpu_launch_coop(uint64_t id, uint32_t stream, uint32_t nptr,
                           const uint64_t *ptrs, uint32_t gx, uint32_t gy,
                           uint32_t block, uint32_t smem) {
  CUdeviceptr d[32];
  void *args[32];
  (void)gy;
  if (nptr > 32) {
    fprintf(stderr, "mapal_gpu: launch_coop with %u pointers\n", nptr);
    exit(102);
  }
  for (uint32_t i = 0; i < nptr; i++) {
    d[i] = ptrs[i];
    args[i] = &d[i];
  }
  launch_coop_on(slot_of(stream), id, args, gx, block, smem);
}

void mapal_gpu_launch(uint64_t id, uint64_t a, uint64_t b, uint64_t out,
                      uint32_t gx, uint32_t gy, uint32_t block, uint32_t smem) {
  CUdeviceptr da = a, db = b, dout = out;
  void *args[] = {&da, &db, &dout};
  launch_on(0, id, args, 3, gx, gy, block, smem);
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
  launch_on(slot_of(stream), id, args, nptr, gx, gy, block, smem);
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
  trap_poll();
}

/* `pinned`: page-lock a host array that crosses the bus, once. Best effort —
 * a range the driver refuses (already registered through a neighbour's page,
 * say) is copied pageable, which is slower and exactly as correct; the trace
 * says which. */
static struct pin { const void *h; uint64_t bytes; } *pins;
static size_t n_pins, cap_pins;

