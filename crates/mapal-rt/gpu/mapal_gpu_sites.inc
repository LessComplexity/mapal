/* Site trials (the tune context) and the one-call tile entry.
 * Included by `mapal_gpu.c` at this place in its translation unit;
 * not compiled on its own. */

/* per-site-choice B2/B3: a site with candidates. `cands` holds one launch
 * recipe per candidate (candidate 0 the rules' plan); `ptrs[0..nptr)` are the
 * kernel's pointer arguments, `ptrs[out]` the output. A recipe zeroes the
 * output first (`zero`), or runs into the workspace at `*ws_slot` (made at the
 * table's largest `ws` on first use) and reduces it into the output. The pick
 * is `mapal_choice.h`'s; every launch goes through `launch_on`. */
struct site_env {
  const struct mapal_cand *c;
  uint32_t stream, nptr, out;
  const uint64_t *ptrs;
  uint64_t *ws_slot, ws_max;
  int zeroed;  /* MAPAL_SITE_ZEROED: an earlier kernel zeroed the output */
  int trialed; /* a trial ran in this call: the output holds its sums */
  int coop;    /* MAPAL_SITE_COOP: persistent kernels, launched cooperatively */
  int cold;    /* MAPAL_SITE_COLD: an operand lives in DRAM in the program */
  int nozero;  /* this run's zero pass was made outside its timed region */
};

static void site_run(void *e, uint32_t i) {
  struct site_env *x = e;
  const struct mapal_cand *c = &x->c[i];
  unsigned k = slot_of(x->stream);
  CUdeviceptr d[32];
  void *args[32];
  for (uint32_t p = 0; p < x->nptr; p++) {
    d[p] = x->ptrs[p];
    args[p] = &d[p];
  }
  if (c->zero && (!x->zeroed || x->trialed) && !x->nozero)
    mapal_gpu_zero(x->ptrs[x->out], c->zero);
  if (c->ws && !c->rid) {
    /* sweep-runtime-2 `streamk-epi`: a workspace without a reduce kernel —
     * a group's stream-K kernel adds its sums and draws its tiles' tickets
     * there, zeroed first; it follows the table's pointers. */
    d[x->nptr] = mapal_gpu_alloc(x->ws_slot, x->ws_max);
    mapal_gpu_zero(d[x->nptr], c->ws);
    args[x->nptr] = &d[x->nptr];
    launch_on(k, c->id, args, x->nptr + 1, c->gx, c->gy, c->block, c->smem);
    return;
  }
  if (c->ws)
    d[x->out] = mapal_gpu_alloc(x->ws_slot, x->ws_max);
  if (x->coop) {
    launch_coop_on(k, c->id, args, c->gx, c->block, c->smem);
    return;
  }
  launch_on(k, c->id, args, x->nptr, c->gx, c->gy, c->block, c->smem);
  if (c->ws) {
    /* The reduce launch: `(ws, b, out, …)` — a group's reduce (`tile-post`)
     * evaluates its epilogue over the table's other pointers too. */
    CUdeviceptr r[32];
    void *ra[32];
    for (uint32_t p = 0; p < x->nptr; p++) {
      r[p] = x->ptrs[p];
      ra[p] = &r[p];
    }
    r[0] = d[x->out];
    r[x->out] = x->ptrs[x->out];
    if (x->nptr == 3)
      r[1] = 0;
    launch_on(k, c->rid, ra, x->nptr, c->rgx, 1, c->rblock, 0);
  }
}

static uint64_t now_ns(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

/* tune-context: how a trial times a candidate (`MAPAL_GPU_TRIAL`, a runtime
 * measurement flag). Unset: what the program runs, where it runs it — the
 * device time between two events on the site's stream around the recipe (no
 * host synchronization inside it: a two-launch recipe pays its second launch
 * as the program does, not a wall-clock sync), a ZEROED site's zero pass made
 * before the timed region (the program runs the candidate without it), and a
 * COLD site's L2 evicted before each run (an untimed streaming write of twice
 * the L2's bytes): its operand lives in DRAM in the program (`residence_plan`),
 * and a back-to-back trial would read it from L2 — the RTX 5080's f8e4m3 FFN
 * GEMVs trialled 21 µs warm against ~75 µs in the program, the 4090's block
 * decode picked a candidate 1.27x slower in context. `isolated`: the wall time
 * around `cuCtxSynchronize` (before); `nocold`: without the eviction. */
static int trial_mode = -1; /* 0 isolated, 1 events + eviction, 2 events */
static int trial_get(void) {
  if (trial_mode < 0) {
    const char *t = getenv("MAPAL_GPU_TRIAL");
    if (!t || !*t)
      trial_mode = 1;
    else if (!strcmp(t, "isolated"))
      trial_mode = 0;
    else if (!strcmp(t, "nocold"))
      trial_mode = 2;
    else {
      fprintf(stderr, "mapal: MAPAL_GPU_TRIAL=%s: expected isolated or nocold\n", t);
      exit(2);
    }
  }
  return trial_mode;
}
static CUdeviceptr evict_buf;
static size_t evict_bytes;
static CUevent trial_ev[2];

/* A trial run: the recipe to completion (the device was idle before it). */
static uint64_t site_timed(void *e, uint32_t i) {
  struct site_env *x = e;
  x->trialed = 1;
  int mode = trial_get();
  if (mode == 0) {
    CK(cuCtxSynchronize());
    uint64_t t0 = now_ns();
    site_run(e, i);
    CK(cuCtxSynchronize());
    return now_ns() - t0;
  }
  const struct mapal_cand *c = &x->c[i];
  unsigned k = slot_of(x->stream);
  CK(cuCtxSynchronize());
  if (c->zero && x->zeroed) {
    mapal_gpu_zero(x->ptrs[x->out], c->zero);
    x->nozero = 1;
  }
  if (x->cold && mode == 1) {
    if (!evict_buf) {
      CUdevice dev;
      int l2 = 0;
      CK(cuCtxGetDevice(&dev));
      CK(cuDeviceGetAttribute(&l2, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev));
      evict_bytes = 2 * (size_t)(l2 > 0 ? l2 : 1 << 20);
      CK(cuMemAlloc(&evict_buf, evict_bytes));
    }
    CK(cuMemsetD8Async(evict_buf, 0, evict_bytes, streams[0]));
  }
  if (!trial_ev[0]) {
    CK(cuEventCreate(&trial_ev[0], CU_EVENT_DEFAULT));
    CK(cuEventCreate(&trial_ev[1], CU_EVENT_DEFAULT));
  }
  CK(cuCtxSynchronize());
  CK(cuEventRecord(trial_ev[0], streams[k]));
  site_run(e, i);
  CK(cuEventRecord(trial_ev[1], streams[k]));
  CK(cuEventSynchronize(trial_ev[1]));
  x->nozero = 0;
  float ms = 0;
  CK(cuEventElapsedTime(&ms, trial_ev[0], trial_ev[1]));
  return (uint64_t)((double)ms * 1e6);
}

/* Before a trial: the workspace any candidate takes, so none is timed with
 * its first allocation. */
static void site_warm(void *e) {
  struct site_env *x = e;
  if (x->ws_max)
    mapal_gpu_alloc(x->ws_slot, x->ws_max);
}

void mapal_gpu_site(uint64_t site, uint32_t flags, uint32_t stream, uint32_t n,
                    const struct mapal_cand *cands, uint32_t nptr,
                    const uint64_t *ptrs, uint32_t out, uint64_t *ws_slot) {
  load_module();
  if (nptr > 31 || out >= nptr || !n) {
    fprintf(stderr, "mapal_gpu: site with %u pointers, out %u, %u candidates\n",
            nptr, out, n);
    exit(102);
  }
  choice_ready_dev();
  struct site_env x = {cands, stream, nptr, out, ptrs, ws_slot, 0, 0, 0,
                       (flags & MAPAL_SITE_COOP) != 0, (flags & MAPAL_SITE_COLD) != 0, 0};
  for (uint32_t i = 0; i < n; i++)
    if (cands[i].ws > x.ws_max)
      x.ws_max = cands[i].ws;
  /* The output passed as an input too: a re-run would read what it wrote. */
  int safe = 1;
  for (uint32_t p = 0; p < nptr; p++)
    safe &= p == out || !ptrs[out] || ptrs[p] != ptrs[out];
  struct choice_dev d = {site_run, site_timed, site_warm, &x};
  /* kernel-merging `zero-fold`: an earlier kernel zeroed the output; the
   * pick runs without its zero pass unless a trial ran before it. */
  x.zeroed = (flags & MAPAL_SITE_ZEROED) != 0;
  choice_site(site, flags, n, cands, safe, &d);
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

