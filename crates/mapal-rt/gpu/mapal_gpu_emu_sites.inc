/* The emulated site trials, registry, zero-for and device producers.
 * Included by `mapal_gpu_emu.c` at this place in its translation unit;
 * not compiled on its own. */

/* per-site-choice: `mapal_gpu.c`'s site entry over the emulated launches.
 * `MAPAL_GPU_PICK=<index>` (default 0; `rules`, `auto`, `tune` 0) runs that candidate, `all` every
 * candidate in order (the last one's output stays: the re-run check). With
 * `MAPAL_EMU_NS=<ns>,<ns>,..` (a fake clock: candidate i takes the i-th, the
 * last for the rest) the shipped decision code runs — wisdom, trial, margin —
 * against the device `emu`; without it nothing is read or written. */
struct site_env {
  const struct mapal_cand *c;
  uint32_t stream, nptr, out;
  const uint64_t *ptrs;
  uint64_t *ws_slot, ws_max;
  int zeroed;  /* MAPAL_SITE_ZEROED: an earlier kernel zeroed the output */
  int trialed; /* a trial ran in this call: the output holds its sums */
  int coop;    /* MAPAL_SITE_COOP: persistent kernels, launched cooperatively */
};
static void site_run(void *e, uint32_t i) {
  struct site_env *x = e;
  const struct mapal_cand *c = &x->c[i];
  uint64_t p[32];
  memcpy(p, x->ptrs, x->nptr * sizeof *p);
  if (c->zero && (!x->zeroed || x->trialed)) mapal_gpu_zero(x->ptrs[x->out], c->zero);
  if (c->ws && !c->rid) {
    /* `streamk-epi`: the zeroed workspace follows the table's pointers. */
    p[x->nptr] = mapal_gpu_alloc(x->ws_slot, x->ws_max);
    mapal_gpu_zero(p[x->nptr], c->ws);
    mapal_gpu_launchv(c->id, x->stream, x->nptr + 1, p, c->gx, c->gy, c->block, c->smem);
    return;
  }
  if (c->ws) p[x->out] = mapal_gpu_alloc(x->ws_slot, x->ws_max);
  if (x->coop) {
    mapal_gpu_launch_coop(c->id, x->stream, x->nptr, p, c->gx, c->gy, c->block, c->smem);
    return;
  }
  if (x->nptr == 3)
    mapal_gpu_launch(c->id, p[0], p[1], p[2], c->gx, c->gy, c->block, c->smem);
  else
    mapal_gpu_launchv(c->id, x->stream, x->nptr, p, c->gx, c->gy, c->block, c->smem);
  if (c->ws && x->nptr == 3) mapal_gpu_launch(c->rid, p[x->out], 0, x->ptrs[x->out], c->rgx, 1, c->rblock, 0);
  if (c->ws && x->nptr != 3) {
    /* A group's reduce (`tile-post`): `(ws, b, out, …)`. */
    uint64_t r[32];
    memcpy(r, x->ptrs, x->nptr * sizeof *r);
    r[0] = p[x->out];
    mapal_gpu_launchv(c->rid, x->stream, x->nptr, r, c->rgx, 1, c->rblock, 0);
  }
}
static uint64_t site_timed(void *e, uint32_t i) {
  ((struct site_env *)e)->trialed = 1;
  site_run(e, i);
  const char *ns = getenv("MAPAL_EMU_NS");
  uint64_t t = 0;
  for (uint32_t j = 0; ns && *ns; j++) {
    char *end;
    t = strtoull(ns, &end, 10);
    if (j == i || *end != ',') break;
    ns = end + 1;
  }
  return t;
}
static void site_warm(void *e) {
  struct site_env *x = e;
  if (x->ws_max) mapal_gpu_alloc(x->ws_slot, x->ws_max);
}
void mapal_gpu_site(uint64_t site, uint32_t flags, uint32_t stream, uint32_t n, const struct mapal_cand *cands,
                    uint32_t nptr, const uint64_t *ptrs, uint32_t out, uint64_t *ws_slot) {
  if (nptr > 31 || out >= nptr || !n) {
    fprintf(stderr, "emu: site with %u pointers, out %u, %u candidates\n", nptr, out, n);
    exit(102);
  }
  struct site_env x = {cands, stream, nptr, out, ptrs, ws_slot, 0, 0, 0, (flags & MAPAL_SITE_COOP) != 0};
  for (uint32_t i = 0; i < n; i++)
    if (cands[i].ws > x.ws_max) x.ws_max = cands[i].ws;
  const char *pick = getenv("MAPAL_GPU_PICK");
  if (pick && !strcmp(pick, "all")) {
    for (uint32_t i = 0; i < n; i++) site_run(&x, i);
    return;
  }
  if (!getenv("MAPAL_EMU_NS")) {
    int i = choice_pick_env();
    x.zeroed = (flags & MAPAL_SITE_ZEROED) != 0;
    site_run(&x, i < 0 ? 0 : (uint32_t)i < n ? (uint32_t)i : n - 1);
    return;
  }
  x.zeroed = (flags & MAPAL_SITE_ZEROED) != 0;
  const char *t = getenv("MAPAL_GPU_TRACE");
  choice_init("emu-sm0", 0, 1, t && atoi(t));
  int safe = 1;
  for (uint32_t p = 0; p < nptr; p++) safe &= p == out || !ptrs[out] || ptrs[p] != ptrs[out];
  struct choice_dev d = {site_run, site_timed, site_warm, &x};
  choice_site(site, flags, n, cands, safe, &d);
}
/* gaps-disasm C1 `load-dispatch`: under the fake clock, the decisions a
 * registered table takes without a run (forced, stored) are made here, before
 * `main`'s body — as the device runtime makes them when the module loads. */
static const struct mapal_site_reg *emu_regs;
static uint32_t emu_nregs;
void mapal_gpu_sites(const struct mapal_site_reg *regs, uint32_t n) {
  emu_regs = regs;
  emu_nregs = n;
  if (!getenv("MAPAL_EMU_NS")) return;
  const char *t = getenv("MAPAL_GPU_TRACE");
  choice_init("emu-sm0", 0, 1, t && atoi(t));
  for (uint32_t i = 0; i < n; i++) choice_preset(regs[i].site, regs[i].n, regs[i].c);
}
/* gaps-disasm C1 `zero-hoist`: a zero issued ahead of its site, skipped where
 * the table's pick — decided (the memo) or forced (MAPAL_GPU_PICK, the
 * registered length clamping it) — adds nothing into the output. */
void mapal_gpu_zero_for(const struct mapal_cand *c, uint64_t d, uint64_t bytes) {
  const char *all = getenv("MAPAL_GPU_PICK");
  if (c && !(all && !strcmp(all, "all"))) {
    struct choice_memo *m = choice_memo_of(c);
    int64_t pick = -1;
    uint32_t n = 0;
    for (uint32_t i = 0; i < emu_nregs; i++)
      if (emu_regs[i].c == c) n = emu_regs[i].n;
    if (m->c == c)
      pick = m->pick;
    else if (!getenv("MAPAL_EMU_NS") && n) {
      int i = choice_pick_env();
      pick = i < 0 ? 0 : (uint32_t)i < n ? i : n - 1;
    }
    if (pick >= 0 && !c[pick].zero) return;
  }
  mapal_gpu_zero(d, bytes);
}
void mapal_gpu_pin(const void *h, uint64_t bytes) { (void)h, (void)bytes; }
void mapal_gpu_unpin(void) {}
void mapal_gpu_gen(uint64_t id, uint64_t out, uint64_t n) {
  void (*k)(void *, uint64_t) = (void (*)(void *, uint64_t))sym("mapal_gen", id);
  for (ctaid_x = 0; (uint64_t)ctaid_x * 256 < n; ctaid_x++)
    for (tid_x = 0; tid_x < 256; tid_x++) k((void *)out, n);
}
void mapal_gpu_tile(uint64_t id, const void *a, uint64_t ab, const void *b, uint64_t bb, void *out, uint64_t ob,
                    uint32_t gx, uint32_t gy, uint32_t block, uint32_t smem) {
  uint64_t da = 0, db = 0, dout = 0;
  mapal_gpu_alloc(&da, ab);
  mapal_gpu_alloc(&db, bb);
  mapal_gpu_alloc(&dout, ob);
  mapal_gpu_h2d(da, a, ab);
  mapal_gpu_h2d(db, b, bb);
  mapal_gpu_launch(id, da, db, dout, gx, gy, block, smem);
  mapal_gpu_d2h(out, dout, 0, ob);
  mapal_gpu_free(&da);
  mapal_gpu_free(&db);
  mapal_gpu_free(&dout);
}

