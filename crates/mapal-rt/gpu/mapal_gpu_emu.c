/* A host emulation of mapal_gpu.c's entries (S54), for checking the host half
 * of the resident path on a machine with no GPU: device memory is host memory
 * (filled with 0xA5, so a read of a cell nothing wrote shows), and a launch
 * runs the kernel once per thread, in order. The device unit is compiled for
 * the host with its special registers renamed to `emu_*` (the test that links
 * this does it). A staged kernel (smem > 0, S63) runs block by block, its
 * threads as pthreads meeting at `emu_barrier` over one shared `emu_smem`; the
 * one-thread-per-cell and devgen kernels run thread by thread.
 *
 * With MAPAL_EMU_SERIAL=1 (2) a threaded block runs its threads one at a
 * time, each from one barrier or warp meet to the next, in thread order
 * (reverse thread order): a write and a read of shared memory by two lanes
 * that no barrier orders then meet in the one order every run, so a missing
 * barrier is a wrong value, not a race that shows by chance.
 *
 * Used by crates/backends/llvm/tests/retain_delta.rs; never shipped. */
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mapal_choice.h"

static int32_t ctaid_x, ctaid_y;
/* MAPAL_EMU_REVERSE=1: a grid's blocks run last to first. A GPU runs a grid's
 * blocks in any order, so two blocks of one launch that race (a merged
 * launch's members sharing bytes: `horizontal`) show as a wrong value. */
static int reverse_blocks(void) {
  static int r = -1;
  if (r < 0) {
    const char *e = getenv("MAPAL_EMU_REVERSE");
    r = e && atoi(e);
  }
  return r;
}
/* Block `i` of `n` (x fastest) in the run's order. */
static void emu_block(uint32_t i, uint32_t gx, uint32_t n) {
  uint32_t b = reverse_blocks() ? n - 1 - i : i;
  ctaid_x = (int32_t)(b % gx);
  ctaid_y = (int32_t)(b / gx);
}
/* One block's shared memory (blocks run one at a time). */
unsigned char emu_smem[1 << 20];
/* `barrier0` for the `b_n` threads of the running block (macOS has no
 * pthread_barrier_t). */
static pthread_mutex_t b_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t b_cv = PTHREAD_COND_INITIALIZER;
static unsigned b_n, b_count, b_gen;
static _Thread_local int32_t tid_x;
/* MAPAL_EMU_SERIAL (see the header): the thread holding the turn runs; one
 * reaching a meet (or its end) hands the turn to the next runnable thread in
 * the order; a meet every thread (every lane of the warp) reached makes them
 * runnable again and gives the turn to the first in the order. */
enum { S_RUN, S_BAR, S_WARP, S_DONE };
static int s_mode = -1, s_turn;
static unsigned char s_st[1024];
static unsigned s_bar, s_warp[32];
static int serial(void) {
  if (s_mode < 0) {
    const char *e = getenv("MAPAL_EMU_SERIAL");
    s_mode = e ? atoi(e) : 0;
  }
  return s_mode;
}
static int s_first(void) {
  for (unsigned i = 0; i < b_n; i++) {
    unsigned t = s_mode == 2 ? b_n - 1 - i : i;
    if (s_st[t] == S_RUN) return (int)t;
  }
  return -1;
}
/* The turn after `t` (lock held). */
static void s_pass(int t) {
  int next = -1, any = 0;
  for (unsigned i = 1; i <= b_n; i++) {
    unsigned u = s_mode == 2 ? (t + b_n - i) % b_n : (t + i) % b_n;
    if (s_st[u] == S_RUN) {
      next = (int)u;
      break;
    }
    any |= s_st[u] != S_DONE;
  }
  if (next < 0 && any) {
    fprintf(stderr, "emu: serial block deadlocked (a barrier not every thread reaches)\n");
    exit(104);
  }
  s_turn = next;
  pthread_cond_broadcast(&b_cv);
}
static void s_wait(void) {
  while (s_turn != tid_x) pthread_cond_wait(&b_cv, &b_mu);
}
static void s_start(unsigned n) {
  b_n = n;
  s_bar = 0;
  memset(s_warp, 0, sizeof s_warp);
  memset(s_st, S_RUN, n);
  s_turn = s_first();
}
static void s_end(void) {
  pthread_mutex_lock(&b_mu);
  s_st[tid_x] = S_DONE;
  s_pass(tid_x);
  pthread_mutex_unlock(&b_mu);
}
/* A meet of warp `warp`'s lanes, or (`warp` < 0) of threads [0, `n`) of the
 * block — a barrier of the block's first `n` threads (`horizontal`: a merged
 * grid's member of fewer threads than the block, whose other threads exited). */
static void s_meet_n(int warp, unsigned n) {
  pthread_mutex_lock(&b_mu);
  int t = tid_x;
  unsigned lo = warp < 0 ? 0 : (unsigned)warp * 32, hi = warp < 0 ? n : lo + 32 > b_n ? b_n : lo + 32;
  unsigned *c = warp < 0 ? &s_bar : &s_warp[warp];
  s_st[t] = warp < 0 ? S_BAR : S_WARP;
  if (++*c == hi - lo) {
    *c = 0;
    for (unsigned u = lo; u < hi; u++) s_st[u] = S_RUN;
    s_turn = s_first();
    pthread_cond_broadcast(&b_cv);
  } else {
    s_pass(t);
  }
  s_wait();
  pthread_mutex_unlock(&b_mu);
}
static void s_meet(int warp) { s_meet_n(warp, b_n); }
/* `barrier.sync 0, n`: the block's first `n` threads meet. */
void emu_barrier_n(int32_t n) {
  if (serial()) {
    s_meet_n(-1, (unsigned)n);
    return;
  }
  pthread_mutex_lock(&b_mu);
  unsigned g = b_gen;
  if (++b_count == (unsigned)n) {
    b_count = 0;
    b_gen++;
    pthread_cond_broadcast(&b_cv);
  } else {
    while (g == b_gen) pthread_cond_wait(&b_cv, &b_mu);
  }
  pthread_mutex_unlock(&b_mu);
}
void emu_barrier(void) {
  if (serial()) {
    s_meet(-1);
    return;
  }
  pthread_mutex_lock(&b_mu);
  unsigned g = b_gen;
  if (++b_count == b_n) {
    b_count = 0;
    b_gen++;
    pthread_cond_broadcast(&b_cv);
  } else {
    while (g == b_gen) pthread_cond_wait(&b_cv, &b_mu);
  }
  pthread_mutex_unlock(&b_mu);
}
/* tile-efficiency T2: `cp.async` as a per-thread deferred copy queue —
 * committed in groups, applied at `wait_group n` with the newest `n` groups
 * left pending, so a missing or short wait reads stale shared memory. */
#define EMU_ASYNC_MAX 1024
struct emu_copy {
  void *d;
  const void *s;
  uint32_t n, g;
};
static _Thread_local struct emu_copy aq[EMU_ASYNC_MAX];
static _Thread_local uint32_t aq_len, aq_group;
void emu_cp_async(void *d, const void *s, uint32_t n) {
  if (aq_len == EMU_ASYNC_MAX) {
    fprintf(stderr, "emu: cp.async queue full\n");
    exit(103);
  }
  aq[aq_len++] = (struct emu_copy){d, s, n, aq_group};
}
void emu_cp_commit(void) { aq_group++; }
void emu_cp_wait(uint32_t n) {
  uint32_t keep = 0;
  for (uint32_t i = 0; i < aq_len; i++) {
    if (aq[i].g + n < aq_group)
      memcpy(aq[i].d, aq[i].s, aq[i].n);
    else
      aq[keep++] = aq[i];
  }
  aq_len = keep;
}
struct emu_thread {
  void (*k)(void *, void *, void *);
  void *a, *b, *o;
  int32_t t;
};
static void *emu_run(void *p) {
  struct emu_thread *x = p;
  tid_x = x->t;
  aq_len = aq_group = 0;
  if (serial()) {
    pthread_mutex_lock(&b_mu);
    s_wait();
    pthread_mutex_unlock(&b_mu);
  }
  x->k(x->a, x->b, x->o);
  if (serial()) s_end();
  return 0;
}
int32_t emu_tid_x(void) { return tid_x; }
/* `cell-lanes`: `shfl.sync.bfly` over a warp of a threaded launch — each lane
 * posts its value, the warp meets, each reads its partner's, the warp meets. */
int32_t emu_laneid(void) { return tid_x & 31; }
static float w_x[1024];
static unsigned w_count[32], w_gen[32];
static void warp_meet(int w) {
  if (serial()) {
    s_meet(w);
    return;
  }
  pthread_mutex_lock(&b_mu);
  unsigned g = w_gen[w];
  if (++w_count[w] == 32) {
    w_count[w] = 0;
    w_gen[w]++;
    pthread_cond_broadcast(&b_cv);
  } else {
    while (g == w_gen[w]) pthread_cond_wait(&b_cv, &b_mu);
  }
  pthread_mutex_unlock(&b_mu);
}
float emu_shfl_bfly(float v, int32_t off) {
  w_x[tid_x] = v;
  warp_meet(tid_x / 32);
  float r = w_x[tid_x ^ off];
  warp_meet(tid_x / 32);
  return r;
}
int32_t emu_ctaid_x(void) { return ctaid_x; }
int32_t emu_ctaid_y(void) { return ctaid_y; }
static unsigned long long n_h2d, n_d2h, n_launch, b_in, b_out, n_zero;
static void total(void) {
  fprintf(stderr, "emu_total h2d=%llu d2h=%llu launches=%llu bytes_in=%llu bytes_out=%llu zeros=%llu\n", n_h2d,
          n_d2h, n_launch, b_in, b_out, n_zero);
}
void mapal_gpu_init(uint32_t f) { (void)f; atexit(total); }
void mapal_gpu_sync(void) {}
uint64_t mapal_gpu_alloc(uint64_t *slot, uint64_t bytes) {
  if (!*slot) {
    /* garbage, not zeros: a read of a never-written cell must show */
    unsigned char *p = malloc(bytes);
    memset(p, 0xA5, bytes);
    *slot = (uint64_t)p;
  }
  return *slot;
}
/* `arena` (S54): one allocation per function, made on the first ask. Live
 * arenas are kept, as `mapal_gpu.c` keeps them (`entry-arena`): a slot holding
 * a member's address is emptied by `mapal_gpu_free`, never freed. */
static struct { uint64_t *slot, base, bytes; } arenas[256];
static size_t n_arenas;
uint64_t mapal_gpu_arena(uint64_t *slot, uint64_t bytes) {
  if (*slot) return *slot;
  mapal_gpu_alloc(slot, bytes);
  if (n_arenas == 256) exit(102);
  arenas[n_arenas].slot = slot;
  arenas[n_arenas].base = *slot;
  arenas[n_arenas++].bytes = bytes;
  return *slot;
}
static int arena_member(uint64_t *slot) {
  for (size_t i = 0; i < n_arenas; i++) {
    if (arenas[i].slot == slot) {
      arenas[i] = arenas[--n_arenas];
      return 0;
    }
    if (*slot >= arenas[i].base && *slot < arenas[i].base + arenas[i].bytes) return 1;
  }
  return 0;
}
void mapal_gpu_h2d(uint64_t d, const void *h, uint64_t bytes) {
  memcpy((void *)d, h, bytes);
  n_h2d++;
  b_in += bytes;
}
/* `async-h2d`: the source is staged at the call, so a copy now is exact. */
void mapal_gpu_h2d_async(uint64_t d, const void *h, uint64_t bytes) { mapal_gpu_h2d(d, h, bytes); }
void mapal_gpu_h2d_on(uint32_t stream, uint64_t d, const void *h, uint64_t bytes) {
  (void)stream;
  mapal_gpu_h2d(d, h, bytes);
}
/* `streams`: one thread runs every chain in program order, so an edge is a no-op. */
void mapal_gpu_event(uint32_t record, uint32_t wait) { (void)record, (void)wait; }
void mapal_gpu_d2h(void *h, uint64_t d, uint64_t off, uint64_t bytes);
void mapal_gpu_d2h_on(uint32_t stream, void *h, uint64_t d, uint64_t off, uint64_t bytes) {
  (void)stream;
  mapal_gpu_d2h(h, d, off, bytes);
}
void mapal_gpu_d2h(void *h, uint64_t d, uint64_t off, uint64_t bytes) {
  memcpy(h, (void *)(d + off), bytes);
  n_d2h++;
  b_out += bytes;
}
/* S63: the output an atomic split adds into. */
void mapal_gpu_zero(uint64_t d, uint64_t bytes) {
  memset((void *)d, 0, bytes);
  n_zero++;
}
void mapal_gpu_free(uint64_t *slot) {
  if (*slot && !arena_member(slot)) free((void *)*slot);
  *slot = 0;
}
static void *sym(const char *prefix, uint64_t id) {
  char name[64];
  snprintf(name, sizeof name, "%s_%016llx", prefix, (unsigned long long)id);
  void *f = dlsym(RTLD_DEFAULT, name);
  if (!f) {
    fprintf(stderr, "emu: no kernel %s\n", name);
    exit(102);
  }
  return f;
}
/* `MAPAL_GPU_TRACE=1`: one line per launch, `mapal_gpu.c`'s `op=launch` (the
 * kernel only: no clock here). */
static void emu_trace_launch(uint64_t id) {
  const char *t = getenv("MAPAL_GPU_TRACE");
  if (t && atoi(t)) fprintf(stderr, "mapal_gpu_trace op=launch kernel=mapal_tile_%016llx\n", (unsigned long long)id);
}
void mapal_gpu_launch(uint64_t id, uint64_t a, uint64_t b, uint64_t o, uint32_t gx, uint32_t gy, uint32_t block,
                      uint32_t smem) {
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
  default: fprintf(stderr, "emu: launchv with %u pointers\n", n); exit(102);
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
  void *p[16];
  for (uint32_t i = 0; i < nptr && i < 16; i++) p[i] = (void *)ptrs[i];
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
/* S54 `l2-window`: the window is a cache policy, never a value. */
void mapal_gpu_launch_l2(uint64_t id, uint64_t a, uint64_t b, uint64_t o, uint32_t gx, uint32_t gy, uint32_t block,
                         uint32_t smem, uint64_t win, uint64_t win_bytes) {
  (void)win, (void)win_bytes;
  mapal_gpu_launch(id, a, b, o, gx, gy, block, smem);
}
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
};
static void site_run(void *e, uint32_t i) {
  struct site_env *x = e;
  const struct mapal_cand *c = &x->c[i];
  uint64_t p[16];
  memcpy(p, x->ptrs, x->nptr * sizeof *p);
  if (c->zero && (!x->zeroed || x->trialed)) mapal_gpu_zero(x->ptrs[x->out], c->zero);
  if (c->ws) p[x->out] = mapal_gpu_alloc(x->ws_slot, x->ws_max);
  if (x->nptr == 3)
    mapal_gpu_launch(c->id, p[0], p[1], p[2], c->gx, c->gy, c->block, c->smem);
  else
    mapal_gpu_launchv(c->id, x->stream, x->nptr, p, c->gx, c->gy, c->block, c->smem);
  if (c->ws) mapal_gpu_launch(c->rid, p[x->out], 0, x->ptrs[x->out], c->rgx, 1, c->rblock, 0);
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
  if (nptr > 16 || out >= nptr || !n) {
    fprintf(stderr, "emu: site with %u pointers, out %u, %u candidates\n", nptr, out, n);
    exit(102);
  }
  struct site_env x = {cands, stream, nptr, out, ptrs, ws_slot, 0, 0, 0};
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
