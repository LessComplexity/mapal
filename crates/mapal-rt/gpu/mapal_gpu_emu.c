/* A host emulation of mapal_gpu.c's entries (S54), for checking the host half
 * of the resident path on a machine with no GPU: device memory is host memory
 * (filled with 0xA5, so a read of a cell nothing wrote shows), and a launch
 * runs the kernel once per thread, in order. The device unit is compiled for
 * the host with its special registers renamed to `emu_*` (the test that links
 * this does it). A staged kernel (smem > 0, S63) runs block by block, its
 * threads as pthreads meeting at `emu_barrier` over one shared `emu_smem`; the
 * one-thread-per-cell and devgen kernels run thread by thread.
 *
 * compile-time-graph B: a launch region records its ops as `mapal_gpu.c`
 * does (the emitter's tags give each op's edges) and replays them at its end
 * in a topological order — program order, or with MAPAL_EMU_SHUFFLE=<seed> a
 * seeded random one, so an edge the tags miss shows as a wrong value.
 *
 * Used by crates/backends/llvm/tests/retain_delta.rs; never shipped. */
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mapal_choice.h"

static _Thread_local int32_t tid_x;
static int32_t ctaid_x, ctaid_y;
/* One block's shared memory (blocks run one at a time). */
unsigned char emu_smem[1 << 20];
/* `barrier0` for the `b_n` threads of the running block (macOS has no
 * pthread_barrier_t). */
static pthread_mutex_t b_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t b_cv = PTHREAD_COND_INITIALIZER;
static unsigned b_n, b_count, b_gen;
void emu_barrier(void) {
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
  x->k(x->a, x->b, x->o);
  return 0;
}
int32_t emu_tid_x(void) { return tid_x; }
/* `cell-lanes`: `shfl.sync.bfly` over a warp of a threaded launch — each lane
 * posts its value, the warp meets, each reads its partner's, the warp meets. */
int32_t emu_laneid(void) { return tid_x & 31; }
static float w_x[1024];
static unsigned w_count[32], w_gen[32];
static void warp_meet(int w) {
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
static unsigned long long n_h2d, n_d2h, n_launch, b_in, b_out;
static unsigned long long n_graph, n_graph_nodes;
static void total(void) {
  fprintf(stderr, "emu_total h2d=%llu d2h=%llu launches=%llu bytes_in=%llu bytes_out=%llu graphs=%llu graph_nodes=%llu\n",
          n_h2d, n_d2h, n_launch, b_in, b_out, n_graph, n_graph_nodes);
}
void mapal_gpu_init(uint32_t f) { (void)f; atexit(total); }
/* compile-time-graph B (see the header). */
#define REG_NODES 128
#define REG_OPS 64
enum { RN_LAUNCH = 1, RN_LAUNCHV, RN_ZERO, RN_GEN };
struct rnode {
  int kind;
  uint64_t id, a, b, o, n;
  uint32_t nptr, gx, gy, block, smem;
  uint64_t ptrs[16];
  int ndeps;
  uint16_t deps[REG_NODES];
};
static struct {
  int on, n, op, cur, untagged;
  uint64_t deps, deps_out;
  int last[REG_OPS];
  struct rnode nodes[REG_NODES];
} rec;
static void rn_dep(struct rnode *x, int i) {
  if (i < 0) return;
  for (int j = 0; j < x->ndeps; j++)
    if (x->deps[j] == i) return;
  x->deps[x->ndeps++] = (uint16_t)i;
}
static struct rnode *rec_node(int kind) {
  if (!rec.on || rec.n == REG_NODES) return 0;
  struct rnode *x = &rec.nodes[rec.n];
  memset(x, 0, sizeof *x);
  x->kind = kind;
  if (rec.op < 0) {
    for (int i = 0; i < rec.n; i++) x->deps[x->ndeps++] = (uint16_t)i;
    rec.untagged = rec.n;
  } else {
    uint64_t mask = kind == RN_ZERO && rec.cur < 0 ? rec.deps_out : rec.deps;
    for (int o = 0; o < REG_OPS; o++)
      if ((mask >> o) & 1) rn_dep(x, rec.last[o]);
    rn_dep(x, rec.cur);
    rn_dep(x, rec.untagged);
    rec.cur = rec.n;
    rec.last[rec.op] = rec.n;
  }
  rec.n++;
  return x;
}
void mapal_gpu_launch(uint64_t id, uint64_t a, uint64_t b, uint64_t o, uint32_t gx, uint32_t gy, uint32_t block,
                      uint32_t smem);
void mapal_gpu_launchv(uint64_t id, uint32_t stream, uint32_t nptr, const uint64_t *ptrs, uint32_t gx, uint32_t gy,
                       uint32_t block, uint32_t smem);
void mapal_gpu_zero(uint64_t d, uint64_t bytes);
void mapal_gpu_gen(uint64_t id, uint64_t out, uint64_t n);
static void run_node(const struct rnode *x) {
  switch (x->kind) {
  case RN_LAUNCH: mapal_gpu_launch(x->id, x->a, x->b, x->o, x->gx, x->gy, x->block, x->smem); break;
  case RN_LAUNCHV: mapal_gpu_launchv(x->id, 0, x->nptr, x->ptrs, x->gx, x->gy, x->block, x->smem); break;
  case RN_ZERO: mapal_gpu_zero(x->a, x->n); break;
  case RN_GEN: mapal_gpu_gen(x->id, x->o, x->n); break;
  }
}
/* Replay the recording (recording off): in program order, or a seeded
 * random topological order. */
static void rec_run(int shuffle) {
  int n = rec.n, done[REG_NODES] = {0};
  rec.on = 0;
  rec.n = 0;
  const char *sh = getenv("MAPAL_EMU_SHUFFLE");
  unsigned long long seed = shuffle && sh ? strtoull(sh, 0, 10) : 0;
  static struct rnode nodes[REG_NODES];
  memcpy(nodes, rec.nodes, (size_t)n * sizeof nodes[0]);
  for (int k = 0; k < n; k++) {
    int ready[REG_NODES], nr = 0;
    for (int i = 0; i < n; i++) {
      if (done[i]) continue;
      int ok = 1;
      for (int j = 0; j < nodes[i].ndeps && ok; j++) ok = done[nodes[i].deps[j]];
      if (ok) ready[nr++] = i;
    }
    int pick = ready[0];
    if (seed) {
      seed = seed * 6364136223846793005ull + 1442695040888963407ull;
      pick = ready[(seed >> 33) % (unsigned long long)nr];
    }
    done[pick] = 1;
    run_node(&nodes[pick]);
  }
}
static void rec_flush(void) {
  if (rec.on) rec_run(0);
}
void mapal_gpu_region_begin(uint64_t rid) {
  (void)rid;
  const char *g = getenv("MAPAL_GPU_GRAPH");
  if (g && !atoi(g)) return;
  rec_flush();
  rec.on = 1;
  rec.n = 0;
  rec.op = rec.cur = rec.untagged = -1;
  for (int i = 0; i < REG_OPS; i++) rec.last[i] = -1;
}
void mapal_gpu_tag(int32_t op, uint64_t deps, uint64_t deps_out) {
  rec.op = op >= 0 && op < REG_OPS ? op : -1;
  rec.deps = deps;
  rec.deps_out = deps_out;
  rec.cur = -1;
}
void mapal_gpu_region_end(uint64_t rid) {
  (void)rid;
  if (!rec.on) return;
  rec.op = -1;
  if (rec.n) {
    n_graph++;
    n_graph_nodes += (unsigned long long)rec.n;
  }
  rec_run(1);
}
void mapal_gpu_sync(void) { rec_flush(); }
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
  rec_flush();
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
  rec_flush();
  memcpy(h, (void *)(d + off), bytes);
  n_d2h++;
  b_out += bytes;
}
/* S63: the output an atomic split adds into. */
void mapal_gpu_zero(uint64_t d, uint64_t bytes) {
  struct rnode *x = rec_node(RN_ZERO);
  if (x) {
    x->a = d;
    x->n = bytes;
    return;
  }
  memset((void *)d, 0, bytes);
}
void mapal_gpu_free(uint64_t *slot) {
  rec_flush();
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
  struct rnode *x = rec_node(RN_LAUNCH);
  if (x) {
    x->id = id;
    x->a = a;
    x->b = b;
    x->o = o;
    x->gx = gx;
    x->gy = gy;
    x->block = block;
    x->smem = smem;
    return;
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
    for (ctaid_y = 0; ctaid_y < (int32_t)gy; ctaid_y++)
      for (ctaid_x = 0; ctaid_x < (int32_t)gx; ctaid_x++) {
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
  for (ctaid_y = 0; ctaid_y < (int32_t)gy; ctaid_y++)
    for (ctaid_x = 0; ctaid_x < (int32_t)gx; ctaid_x++)
      for (tid_x = 0; tid_x < (int32_t)block; tid_x++) k((void *)a, (void *)b, (void *)o);
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
  call_v(x->k, x->n, x->p);
  return 0;
}
void mapal_gpu_launchv(uint64_t id, uint32_t stream, uint32_t nptr, const uint64_t *ptrs, uint32_t gx, uint32_t gy,
                       uint32_t block, uint32_t smem) {
  (void)stream;
  struct rnode *x = rec_node(RN_LAUNCHV);
  if (x) {
    x->id = id;
    x->nptr = nptr;
    x->gx = gx;
    x->gy = gy;
    x->block = block;
    x->smem = smem;
    for (uint32_t i = 0; i < nptr && i < 16; i++) x->ptrs[i] = ptrs[i];
    return;
  }
  emu_trace_launch(id);
  kv_t k = (kv_t)sym("mapal_tile", id);
  void *p[16];
  for (uint32_t i = 0; i < nptr && i < 16; i++) p[i] = (void *)ptrs[i];
  /* A module with shuffles (`emu_threaded`) runs its launches threaded. */
  if (smem || dlsym(RTLD_DEFAULT, "emu_threaded")) {
    pthread_t *th = malloc(block * sizeof *th);
    struct emu_vthread *xs = malloc(block * sizeof *xs);
    b_n = block;
    for (ctaid_y = 0; ctaid_y < (int32_t)gy; ctaid_y++)
      for (ctaid_x = 0; ctaid_x < (int32_t)gx; ctaid_x++) {
        for (uint32_t t = 0; t < block; t++) {
          xs[t] = (struct emu_vthread){k, nptr, p, (int32_t)t};
          if (pthread_create(&th[t], 0, emu_vrun, &xs[t])) exit(102);
        }
        for (uint32_t t = 0; t < block; t++) pthread_join(th[t], 0);
      }
    free(th);
    free(xs);
  } else {
    for (ctaid_y = 0; ctaid_y < (int32_t)gy; ctaid_y++)
      for (ctaid_x = 0; ctaid_x < (int32_t)gx; ctaid_x++)
        for (tid_x = 0; tid_x < (int32_t)block; tid_x++) call_v(k, nptr, p);
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
};
static void site_run(void *e, uint32_t i) {
  struct site_env *x = e;
  const struct mapal_cand *c = &x->c[i];
  uint64_t p[16];
  memcpy(p, x->ptrs, x->nptr * sizeof *p);
  if (c->zero) mapal_gpu_zero(x->ptrs[x->out], c->zero);
  if (c->ws) p[x->out] = mapal_gpu_alloc(x->ws_slot, x->ws_max);
  if (x->nptr == 3)
    mapal_gpu_launch(c->id, p[0], p[1], p[2], c->gx, c->gy, c->block, c->smem);
  else
    mapal_gpu_launchv(c->id, x->stream, x->nptr, p, c->gx, c->gy, c->block, c->smem);
  if (c->ws) mapal_gpu_launch(c->rid, p[x->out], 0, x->ptrs[x->out], c->rgx, 1, c->rblock, 0);
}
static uint64_t site_timed(void *e, uint32_t i) {
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
  struct site_env x = {cands, stream, nptr, out, ptrs, ws_slot, 0};
  for (uint32_t i = 0; i < n; i++)
    if (cands[i].ws > x.ws_max) x.ws_max = cands[i].ws;
  if (rec.on && getenv("MAPAL_EMU_NS") && !choice_known(cands, n)) rec_flush();
  const char *pick = getenv("MAPAL_GPU_PICK");
  if (pick && !strcmp(pick, "all")) {
    for (uint32_t i = 0; i < n; i++) site_run(&x, i);
    return;
  }
  if (!getenv("MAPAL_EMU_NS")) {
    int i = choice_pick_env();
    site_run(&x, i < 0 ? 0 : (uint32_t)i < n ? (uint32_t)i : n - 1);
    return;
  }
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
  struct rnode *x = rec_node(RN_GEN);
  if (x) {
    x->id = id;
    x->o = out;
    x->n = n;
    return;
  }
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
