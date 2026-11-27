/* The emulated launches: tensor maps, plain, vector and cooperative.
 * Included by `mapal_gpu_emu.c` at this place in its translation unit;
 * not compiled on its own. */

/* hopper-wgmma-tma: the kernel's tensor-map recipe (`<kernel>_tmap`, see
 * `mapal_gpu.c`), encoded into the emulator's own map per launch. */
#define EMU_TMAP_MAX 4
static uint32_t emu_tmaps(uint64_t id, const uint64_t *ptrs, uint32_t nptr, struct emu_tmap *out) {
  char name[64];
  snprintf(name, sizeof name, "mapal_tile_%016llx_tmap", (unsigned long long)id);
  const uint64_t *r = dlsym(RTLD_DEFAULT, name);
  if (!r) return 0;
  uint32_t n = (uint32_t)r[0];
  if (n > EMU_TMAP_MAX) exit(102);
  for (uint32_t m = 0; m < n; m++) {
    const uint64_t *q = r + 1 + 20 * m;
    struct emu_tmap *t = &out[m];
    memset(t, 0, sizeof *t);
    if (q[0] >= nptr) {
      fprintf(stderr, "emu: tensor map %u names argument %llu of %u\n", m, (unsigned long long)q[0], nptr);
      exit(102);
    }
    t->base = ptrs[q[0]] + q[4];
    t->elem = (uint32_t)q[1];
    t->rank = (uint32_t)q[2];
    t->swizzle = (uint32_t)q[3];
    if (t->base % 16) {
      fprintf(stderr, "emu: tensor map %u base not 16-byte aligned\n", m);
      exit(102);
    }
    for (uint32_t i = 0; i < 5; i++) {
      t->dims[i] = (uint32_t)q[5 + i];
      t->box[i] = (uint32_t)q[14 + i];
    }
    t->strides[0] = t->elem;
    for (uint32_t i = 1; i < 5; i++) {
      t->strides[i] = q[9 + i];
      if (i < t->rank && t->strides[i] % 16) {
        fprintf(stderr, "emu: tensor map %u stride %u not 16-byte\n", m, i);
        exit(102);
      }
    }
  }
  return n;
}
/* `MAPAL_GPU_TRACE=1`: one line per launch, `mapal_gpu.c`'s `op=launch` (the
 * kernel only: no clock here). */
static void emu_trace_launch(uint64_t id) {
  const char *t = getenv("MAPAL_GPU_TRACE");
  if (t && atoi(t)) fprintf(stderr, "mapal_gpu_trace op=launch kernel=mapal_tile_%016llx\n", (unsigned long long)id);
}
void mapal_gpu_launch(uint64_t id, uint64_t a, uint64_t b, uint64_t o, uint32_t gx, uint32_t gy, uint32_t block,
                      uint32_t smem) {
  {
    struct emu_tmap tm[EMU_TMAP_MAX];
    uint64_t ptrs[3] = {a, b, o};
    if (emu_tmaps(id, ptrs, 3, tm)) {
      mapal_gpu_launchv(id, 0, 3, ptrs, gx, gy, block, smem);
      return;
    }
  }
  emu_trace_launch(id);
  void (*k)(void *, void *, void *) = (void (*)(void *, void *, void *))sym("mapal_tile", id);
  /* A module with shuffles (`emu_threaded`: `fold-lanes`) runs threaded too. */
  if (smem || dlsym(RTLD_DEFAULT, "emu_threaded")) {
    if (smem > sizeof emu_smem) {
      fprintf(stderr, "emu: smem %u over the emulator's %zu\n", smem, sizeof emu_smem);
      exit(102);
    }
    pthread_t *th = malloc(block * sizeof *th);
    struct emu_thread *xs = malloc(block * sizeof *xs);
    b_n = block;
    for (uint32_t blk = 0; blk < gx * gy; blk++) {
        emu_block(blk, gx, gx * gy);
        if (serial()) s_start(block);
        for (uint32_t t = 0; t < block; t++) {
          xs[t] = (struct emu_thread){k, (void *)a, (void *)b, (void *)o, (int32_t)t};
          if (pthread_create(&th[t], 0, emu_run, &xs[t])) {
            fprintf(stderr, "emu: pthread_create\n");
            exit(102);
          }
        }
        for (uint32_t t = 0; t < block; t++) pthread_join(th[t], 0);
      }
    free(th);
    free(xs);
    n_launch++;
    return;
  }
  for (uint32_t blk = 0; blk < gx * gy; blk++) {
    emu_block(blk, gx, gx * gy);
    for (tid_x = 0; tid_x < (int32_t)block; tid_x++) k((void *)a, (void *)b, (void *)o);
  }
  n_launch++;
}
/* An epilogue or chained launch (S54): the kernel takes `nptr` pointers.
 * Run thread by thread (an epilogue kernel stages no shared memory here:
 * tests that reach a staged one turn `async` off, as for `launch`). */
typedef void (*kv_t)();
typedef void *P;
static void call_v(kv_t k, uint32_t n, void **p) {
  switch (n) {
  case 1: ((void (*)(P))k)(p[0]); break;
  case 2: ((void (*)(P, P))k)(p[0], p[1]); break;
  case 3: ((void (*)(P, P, P))k)(p[0], p[1], p[2]); break;
  case 4: ((void (*)(P, P, P, P))k)(p[0], p[1], p[2], p[3]); break;
  case 5: ((void (*)(P, P, P, P, P))k)(p[0], p[1], p[2], p[3], p[4]); break;
  case 6: ((void (*)(P, P, P, P, P, P))k)(p[0], p[1], p[2], p[3], p[4], p[5]); break;
  case 7: ((void (*)(P, P, P, P, P, P, P))k)(p[0], p[1], p[2], p[3], p[4], p[5], p[6]); break;
  case 8: ((void (*)(P, P, P, P, P, P, P, P))k)(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]); break;
  case 9: ((void (*)(P, P, P, P, P, P, P, P, P))k)(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8]); break;
  case 10:
    ((void (*)(P, P, P, P, P, P, P, P, P, P))k)(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9]);
    break;
  case 11:
    ((void (*)(P, P, P, P, P, P, P, P, P, P, P))k)(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
                                                    p[10]);
    break;
  case 12:
    ((void (*)(P, P, P, P, P, P, P, P, P, P, P, P))k)(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9],
                                                       p[10], p[11]);
    break;
  case 13:
    ((void (*)(P, P, P, P, P, P, P, P, P, P, P, P, P))k)(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8],
                                                          p[9], p[10], p[11], p[12]);
    break;
  case 14:
    ((void (*)(P, P, P, P, P, P, P, P, P, P, P, P, P, P))k)(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8],
                                                             p[9], p[10], p[11], p[12], p[13]);
    break;
  case 15:
    ((void (*)(P, P, P, P, P, P, P, P, P, P, P, P, P, P, P))k)(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7],
                                                                p[8], p[9], p[10], p[11], p[12], p[13], p[14]);
    break;
  case 16:
    ((void (*)(P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P))k)(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7],
                                                                   p[8], p[9], p[10], p[11], p[12], p[13], p[14],
                                                                   p[15]);
    break;
  default:
    if (n <= 32) {
      /* `calls`: up to 32 pointers; a callee takes its own leading ones. */
      void *q[32] = {0};
      memcpy(q, p, n * sizeof *q);
      ((void (*)(P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P))k)(
          q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16],
          q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25], q[26], q[27], q[28], q[29], q[30], q[31]);
      break;
    }
    fprintf(stderr, "emu: launchv with %u pointers\n", n);
    exit(102);
  }
}
struct emu_vthread {
  kv_t k;
  uint32_t n;
  void **p;
  int32_t t;
};
static void *emu_vrun(void *q) {
  struct emu_vthread *x = q;
  tid_x = x->t;
  if (serial()) {
    pthread_mutex_lock(&b_mu);
    s_wait();
    pthread_mutex_unlock(&b_mu);
  }
  call_v(x->k, x->n, x->p);
  if (serial()) s_end();
  return 0;
}
void mapal_gpu_launchv(uint64_t id, uint32_t stream, uint32_t nptr, const uint64_t *ptrs, uint32_t gx, uint32_t gy,
                       uint32_t block, uint32_t smem) {
  (void)stream;
  emu_trace_launch(id);
  kv_t k = (kv_t)sym("mapal_tile", id);
  void *p[32 + EMU_TMAP_MAX];
  for (uint32_t i = 0; i < nptr && i < 32; i++) p[i] = (void *)ptrs[i];
  /* hopper-wgmma-tma: the maps follow the pointers. */
  struct emu_tmap tm[EMU_TMAP_MAX];
  uint32_t nm = emu_tmaps(id, ptrs, nptr, tm);
  for (uint32_t m = 0; m < nm; m++) p[nptr + m] = &tm[m];
  nptr += nm;
  /* A module with shuffles (`emu_threaded`) runs its launches threaded. */
  if (smem || dlsym(RTLD_DEFAULT, "emu_threaded")) {
    pthread_t *th = malloc(block * sizeof *th);
    struct emu_vthread *xs = malloc(block * sizeof *xs);
    b_n = block;
    for (uint32_t blk = 0; blk < gx * gy; blk++) {
        emu_block(blk, gx, gx * gy);
        if (serial()) s_start(block);
        for (uint32_t t = 0; t < block; t++) {
          xs[t] = (struct emu_vthread){k, nptr, p, (int32_t)t};
          if (pthread_create(&th[t], 0, emu_vrun, &xs[t])) exit(102);
        }
        for (uint32_t t = 0; t < block; t++) pthread_join(th[t], 0);
      }
    free(th);
    free(xs);
  } else {
    for (uint32_t blk = 0; blk < gx * gy; blk++) {
      emu_block(blk, gx, gx * gy);
      for (tid_x = 0; tid_x < (int32_t)block; tid_x++) call_v(k, nptr, p);
    }
  }
  n_launch++;
}
/* kernel-merging `calls` (see the header): the cooperative launch. */
#define EMU_COOP_MAX 64
enum { C_NEW, C_RUN, C_PARK, C_DONE };
static uint32_t coop_nb, coop_gx, coop_block, coop_left;
static unsigned coop_arr[EMU_COOP_MAX], coop_go[EMU_COOP_MAX];
static unsigned char coop_st[EMU_COOP_MAX];
static _Thread_local uint32_t my_blk;
/* The blocks not holding the baton wait here, apart from the running
 * block's barriers (`b_cv`): a barrier wakes only the running block. */
static pthread_cond_t c_cv = PTHREAD_COND_INITIALIZER;
/* The baton to the next block that may run (lock held). */
static void coop_next(void) {
  unsigned lo = ~0u;
  for (uint32_t b = 0; b < coop_nb; b++)
    if (coop_arr[b] < lo) lo = coop_arr[b];
  for (uint32_t i = 0; i < coop_nb; i++) {
    uint32_t b = reverse_blocks() ? coop_nb - 1 - i : i;
    if (coop_st[b] == C_NEW || (coop_st[b] == C_PARK && coop_arr[b] <= lo)) {
      coop_st[b] = C_RUN;
      ctaid_x = (int32_t)(b % coop_gx);
      ctaid_y = (int32_t)(b / coop_gx);
      b_n = coop_block;
      b_count = 0;
      memset(k_count, 0, sizeof k_count);
      coop_left = coop_block;
      if (serial()) s_start(coop_block);
      coop_go[b]++;
      pthread_cond_broadcast(&c_cv);
      return;
    }
  }
  for (uint32_t b = 0; b < coop_nb; b++)
    if (coop_st[b] != C_DONE) {
      fprintf(stderr, "emu: a grid barrier not every block reaches\n");
      exit(105);
    }
}
/* A thread of the running block reached a grid barrier (`done` = 0) or its
 * end; the last one passes the baton (lock held). */
static void coop_arrive(int done) {
  int t = tid_x;
  if (serial()) s_st[t] = done ? S_DONE : S_PARK;
  if (--coop_left == 0) {
    coop_st[my_blk] = done ? C_DONE : C_PARK;
    coop_arr[my_blk] += !done;
    coop_next();
  } else if (serial()) {
    s_pass(t);
  }
}
/* `@mapal_gsync` (`func/calls.rs`): the block meets, parks, and resumes when
 * every block reached this barrier. */
void mapal_gsync(void *ctr) {
  (void)ctr;
  emu_barrier();
  pthread_mutex_lock(&b_mu);
  unsigned g = coop_go[my_blk];
  coop_arrive(0);
  while (coop_go[my_blk] == g) pthread_cond_wait(&c_cv, &b_mu);
  if (serial()) s_wait();
  pthread_mutex_unlock(&b_mu);
}
struct emu_cthread {
  kv_t k;
  uint32_t n, blk;
  void **p;
  int32_t t;
};
static void *emu_crun(void *q) {
  struct emu_cthread *x = q;
  tid_x = x->t;
  my_blk = x->blk;
  aq_len = aq_group = 0;
  pthread_mutex_lock(&b_mu);
  while (!coop_go[my_blk]) pthread_cond_wait(&c_cv, &b_mu);
  if (serial()) s_wait();
  pthread_mutex_unlock(&b_mu);
  call_v(x->k, x->n, x->p);
  pthread_mutex_lock(&b_mu);
  coop_arrive(1);
  pthread_mutex_unlock(&b_mu);
  return 0;
}
void mapal_gpu_launch_coop(uint64_t id, uint32_t stream, uint32_t nptr, const uint64_t *ptrs, uint32_t gx, uint32_t gy,
                           uint32_t block, uint32_t smem) {
  (void)stream, (void)gy;
  emu_trace_launch(id);
  kv_t k = (kv_t)sym("mapal_tile", id);
  if (smem > sizeof emu_smem || nptr > 32 || block > 1024) {
    fprintf(stderr, "emu: coop launch smem %u, %u pointers, block %u\n", smem, nptr, block);
    exit(102);
  }
  void *p[32];
  for (uint32_t i = 0; i < nptr; i++) p[i] = (void *)ptrs[i];
  const char *e = getenv("MAPAL_EMU_GRID");
  uint32_t cap = e ? (uint32_t)atoi(e) : 3, g = gx;
  if (cap && g > cap) g = cap;
  if (g > EMU_COOP_MAX) g = EMU_COOP_MAX;
  if (!g) g = 1;
  coop_nb = g;
  coop_gx = g;
  coop_block = block;
  nctaid_x = (int32_t)g;
  memset(coop_arr, 0, sizeof coop_arr);
  memset(coop_go, 0, sizeof coop_go);
  memset(coop_st, C_NEW, sizeof coop_st);
  uint32_t nt = g * block;
  pthread_t *th = malloc(nt * sizeof *th);
  struct emu_cthread *xs = malloc(nt * sizeof *xs);
  pthread_attr_t at;
  pthread_attr_init(&at);
  pthread_attr_setstacksize(&at, 1 << 20);
  for (uint32_t i = 0; i < nt; i++) {
    xs[i] = (struct emu_cthread){k, nptr, i / block, p, (int32_t)(i % block)};
    if (pthread_create(&th[i], &at, emu_crun, &xs[i])) {
      fprintf(stderr, "emu: pthread_create\n");
      exit(102);
    }
  }
  pthread_attr_destroy(&at);
  pthread_mutex_lock(&b_mu);
  coop_next();
  pthread_mutex_unlock(&b_mu);
  for (uint32_t i = 0; i < nt; i++) pthread_join(th[i], 0);
  free(th);
  free(xs);
  nctaid_x = 1;
  n_launch++;
}
/* S54 `l2-window`: the window is a cache policy, never a value. */
void mapal_gpu_launch_l2(uint64_t id, uint64_t a, uint64_t b, uint64_t o, uint32_t gx, uint32_t gy, uint32_t block,
                         uint32_t smem, uint64_t win, uint64_t win_bytes) {
  (void)win, (void)win_bytes;
  mapal_gpu_launch(id, a, b, o, gx, gy, block, smem);
}

