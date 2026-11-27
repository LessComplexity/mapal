/* Initialization: the site registry, the driver, module load, the trace and the peak line.
 * Included by `mapal_gpu.c` at this place in its translation unit;
 * not compiled on its own. */

/* `stage`: a ring of page-locked slots an async upload is copied into and
 * DMA'd from. A slot is reused only once its event says the DMA from it ran,
 * so the host runs at most `STAGE_SLOTS` uploads ahead of the device. */
#define STAGE_SLOTS 64
#define STAGE_SLOT (64 * 1024)
static unsigned char *stage_buf;
static CUevent stage_ev[STAGE_SLOTS];
static int stage_used[STAGE_SLOTS];
static unsigned stage_next;
static pthread_t init_thread;
static int init_state; /* 0 not started, 1 running on init_thread, 2 ready */
static double t_init;  /* when mapal_gpu_init ran */

static void tma_warm(void);
static void tmap_table(void);
static void choice_ready_dev(void);
static CUfunction kernel(const char *prefix, uint64_t id, uint32_t smem,
                         char *name, size_t cap);
static int coop_per(CUfunction fn, uint32_t block, uint32_t smem);

/* gaps-disasm C1 `load-dispatch`: the module's site tables, registered by the
 * host before its init (`mapal_gpu_sites`), each resolved when the module
 * loads — its pick where one is decided without a run (`choice_preset`:
 * forced, stored), every candidate's kernels with their shared-memory limit
 * set, a cooperative table's occupancy — so a one-call program's first site
 * goes from its clock to its first driver call through a memo hit. */
static const struct mapal_site_reg *site_regs;
static uint32_t n_site_regs, n_site_preset;
void mapal_gpu_sites(const struct mapal_site_reg *regs, uint32_t n) {
  site_regs = regs;
  n_site_regs = n;
}
static void sites_resolve(void) {
  char name[40];
  choice_ready_dev();
  for (uint32_t r = 0; r < n_site_regs; r++) {
    const struct mapal_site_reg *g = &site_regs[r];
    n_site_preset += choice_preset(g->site, g->n, g->c) >= 0;
    for (uint32_t i = 0; i < g->n; i++) {
      const struct mapal_cand *c = &g->c[i];
      CUfunction fn = kernel("mapal_tile", c->id, c->smem, name, sizeof name);
      if (c->rid)
        kernel("mapal_tile", c->rid, 0, name, sizeof name);
      if (g->flags & MAPAL_SITE_COOP)
        coop_per(fn, c->block, c->smem);
    }
  }
}

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
  module_hash = choice_fnv(ptx, (size_t)n, CHOICE_FNV0);
  CK(cuModuleLoadData(&module, ptx));
  tmap_scan(ptx);
  tmap_table();
  if (flags & PRELOAD)
    preload(ptx);
  free(ptx);
  if (!(flags & TMA_COLD) && n_tmap_names)
    tma_warm();
  if (flags & STAGE) {
    void *b;
    CK(cuMemHostAlloc(&b, (size_t)STAGE_SLOTS * STAGE_SLOT, 0));
    stage_buf = b;
    for (int i = 0; i < STAGE_SLOTS; i++)
      CK(cuEventCreate(&stage_ev[i], CU_EVENT_DISABLE_TIMING));
  }
  if (flags & STREAMS)
    for (int i = 1; i <= MAPAL_GPU_STREAMS; i++)
      CK(cuStreamCreate(&streams[i], CU_STREAM_NON_BLOCKING));
  zero_load();
  if (flags & WARM_ZERO) {
    double w0 = now_ms();
    CK(cuMemAlloc(&warm_buf, WARM_ZERO_BYTES));
    warm_fills();
    CK(cuStreamSynchronize(streams[0]));
    tot.warm = now_ms() - w0;
  }
  if (n_site_regs)
    sites_resolve();
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
static void trace_env(void) {
  if (trace < 0) {
    const char *t = getenv("MAPAL_GPU_TRACE");
    trace = t ? atoi(t) : 0;
    if (trace)
      atexit(total_line);
  }
}

void mapal_gpu_init(uint32_t f) {
  flags = f;
  t_init = now_ms();
  /* Before the init thread: its site decisions (`load-dispatch`) trace. */
  trace_env();
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
  trace_env();
  double t0 = now_ms();
  if (init_state == 1) {
    pthread_join(init_thread, 0);
    CK(cuCtxSetCurrent(ctx)); /* the context is current on the init thread */
    if (flags & WARM_ZERO) {
      double w0 = now_ms();
      warm_fills(); /* this thread's part (`warm-zero`) */
      tot.warm += now_ms() - w0;
    }
  } else {
    init_all();
  }
  init_state = 2;
  /* The site choice (device tag + wisdom read) is part of loading: a program's first
   * `mapal_gpu_site` would otherwise pay it inside its own timed region (50–90 us on
   * rtx-5080, local-rtx-5080/gpu-analysis.md). */
  choice_ready_dev();
  tot.load = now_ms() - t0;
  tot.lead = t_init ? t0 - t_init : 0;
  if (trace)
    fprintf(stderr,
            "mapal_gpu_trace op=load flags=%u wait_ms=%.3f lead_ms=%.3f "
            "init_ms=%.3f ctx_ms=%.3f module_ms=%.3f sites=%u preset=%u zero=%s\n",
            flags, tot.load, tot.lead, tot.init, tot.ctx, tot.module, n_site_regs,
            n_site_preset, zero_fn ? "kernel" : "memset");
}

static void choice_ready_dev(void) {
  if (choice.ready)
    return;
  CUdevice dev;
  char name[64] = "gpu", tag[96];
  int major = 0, minor = 0;
  CK(cuCtxGetDevice(&dev));
  CK(cuDeviceGetName(name, sizeof name, dev));
  CK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
  CK(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
  snprintf(tag, sizeof tag, "%s-sm%d%d", name, major, minor);
  choice_init(tag, module_hash, 1, trace);
}

static CUfunction kernel(const char *prefix, uint64_t id, uint32_t smem,
                         char *name, size_t cap) {
  struct fn_entry *e = fn_find(prefix, id);
  if (e) {
    memcpy(name, e->name, cap < sizeof e->name ? cap : sizeof e->name);
    name[cap - 1] = 0;
    if (smem > e->smem) {
      CK(cuFuncSetAttribute(e->fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                            (int)smem));
      e->smem = smem;
    }
    return e->fn;
  }
  snprintf(name, cap, "%s_%016llx", prefix, (unsigned long long)id);
  for (size_t i = 0; i < n_fns; i++)
    if (!strcmp(fns[i].name, name)) {
      if (smem > fns[i].smem) {
        CK(cuFuncSetAttribute(fns[i].fn,
                              CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                              (int)smem));
        fns[i].smem = smem;
      }
      return fns[i].fn;
    }
  CUfunction fn;
  CK(cuModuleGetFunction(&fn, module, name));
  if (smem > 48 * 1024)
    CK(cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                          (int)smem));
  return fn;
}

/* caching-memory: the device bytes the program holds (every cuMemAlloc
 * less every cuMemFree) and their high-water mark — the program's peak
 * device memory, the counterpart of torch.cuda.max_memory_allocated.
 * MAPAL_GPU_PEAK=1 prints it at exit (`mapal_gpu_peak`). */
static uint64_t live_bytes, peak_bytes, n_peak_allocs;
static void peak_line(void) {
  fprintf(stderr, "mapal_gpu_peak bytes=%llu mb=%.1f allocs=%llu\n",
          (unsigned long long)peak_bytes, peak_bytes / 1048576.0,
          (unsigned long long)n_peak_allocs);
}

