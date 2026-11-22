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
 * kernel-merging `calls`: a cooperative launch (`mapal_gpu_launch_coop`, a
 * persistent kernel whose levels meet at `mapal_gsync`) runs every block's
 * threads at once but one block at a time: the running block holds a baton
 * until all its threads reach a grid barrier (or end), then the next block in
 * the run order that may proceed takes it — a block not started yet, or one
 * parked at a barrier every block has reached. A barrier some block never
 * reaches is a deadlock on the device and an exit (105) here. Blocks run at
 * most MAPAL_EMU_GRID (default 3) at once, so each body strides over several
 * of its blocks.
 *
 * Used by crates/backends/llvm/tests/retain_delta.rs; never shipped. */
#include <dlfcn.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mapal_choice.h"

static int32_t ctaid_x, ctaid_y, nctaid_x = 1;
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
enum { S_RUN, S_BAR, S_WARP, S_DONE, S_BARK, S_PARK };
static int s_mode = -1, s_turn;
static unsigned char s_st[1024], s_bid[1024];
/* One condition per thread: a hand-off wakes the one thread the turn goes to.
 * (One shared broadcast woke every waiting thread of the block per hand-off —
 * b_n^2 wake-ups a phase, the serialized modes' whole cost at 256 threads.) */
static pthread_cond_t s_cv[1024] = {[0 ... 1023] = PTHREAD_COND_INITIALIZER};
/* The turn to `t` (lock held; -1: nobody). */
static void s_give(int t) {
  s_turn = t;
  if (t >= 0) pthread_cond_signal(&s_cv[t]);
}
static unsigned s_bar, s_warp[32], s_bark[16];
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
  s_give(next);
}
static void s_wait(void) {
  while (s_turn != tid_x) pthread_cond_wait(&s_cv[tid_x], &b_mu);
}
static void s_start(unsigned n) {
  b_n = n;
  s_bar = 0;
  memset(s_warp, 0, sizeof s_warp);
  memset(s_bark, 0, sizeof s_bark);
  memset(s_st, S_RUN, n);
  s_give(s_first());
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
    s_give(s_first());
  } else {
    s_pass(t);
  }
  s_wait();
  pthread_mutex_unlock(&b_mu);
}
static void s_meet(int warp) { s_meet_n(warp, b_n); }
/* `barrier.sync <id>, n` (id > 0, `calls`' tile boundary): `n` threads meet
 * at barrier `id`, apart from barrier 0's (a padded member's own). */
static void s_meet_k(int id, unsigned n) {
  pthread_mutex_lock(&b_mu);
  int t = tid_x;
  s_st[t] = S_BARK;
  s_bid[t] = (unsigned char)id;
  if (++s_bark[id] == n) {
    s_bark[id] = 0;
    for (unsigned u = 0; u < b_n; u++)
      if (s_st[u] == S_BARK && s_bid[u] == id) s_st[u] = S_RUN;
    s_give(s_first());
  } else {
    s_pass(t);
  }
  s_wait();
  pthread_mutex_unlock(&b_mu);
}
static unsigned k_count[16], k_gen[16];
void emu_barrier_k(int32_t id, int32_t n) {
  if (id <= 0 || id >= 16) {
    fprintf(stderr, "emu: barrier id %d\n", id);
    exit(103);
  }
  if (serial()) {
    s_meet_k(id, (unsigned)n);
    return;
  }
  pthread_mutex_lock(&b_mu);
  unsigned g = k_gen[id];
  if (++k_count[id] == (unsigned)n) {
    k_count[id] = 0;
    k_gen[id]++;
    pthread_cond_broadcast(&b_cv);
  } else {
    while (g == k_gen[id]) pthread_cond_wait(&b_cv, &b_mu);
  }
  pthread_mutex_unlock(&b_mu);
}
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
/* ADR-0042: `ex2.approx.ftz.f32` of the hardware `exp` form (its argument
 * lies in [-1/2, 1/2]: no subnormal in or out). */
float emu_ex2(float a) { return exp2f(a); }

float emu_shfl_bfly(float v, int32_t off) {
  w_x[tid_x] = v;
  warp_meet(tid_x / 32);
  float r = w_x[tid_x ^ off];
  warp_meet(tid_x / 32);
  return r;
}
/* tensor-cores (ADR-0043 D6): the warp's 16-bit matrix instructions. Each lane
 * posts its operands, the warp meets, each lane computes its own fragment from
 * the posts, the warp meets again. Results are read back one register at a
 * time (`emu_frag`), so no struct crosses the C ABI. */
static _Thread_local uint32_t t_frag[4];
static const uint8_t *w_addr[1024];
static uint32_t w_ab[1024][6];
static float w_c[1024][4];
uint32_t emu_frag(int32_t i) { return t_frag[i]; }
float emu_fragf(int32_t i) {
  float f;
  memcpy(&f, &t_frag[i], 4);
  return f;
}
/* `ldmatrix.sync.aligned.m8n8.x<n>[.trans].shared.b16`: matrix q's row r is
 * the 16 bytes at lane 8q + r's address; lane l receives (row l/4, columns
 * 2(l%4), +1), or with `.trans` (rows 2(l%4), +1, column l/4). */
void emu_ldsm(const uint8_t *p, int32_t n, int32_t trans) {
  w_addr[tid_x] = p;
  warp_meet(tid_x / 32);
  int lane = tid_x & 31, base = tid_x & ~31;
  for (int q = 0; q < n; q++) {
    if (!trans) {
      memcpy(&t_frag[q], w_addr[base + q * 8 + lane / 4] + 4 * (lane % 4), 4);
    } else {
      uint16_t lo, hi;
      memcpy(&lo, w_addr[base + q * 8 + 2 * (lane % 4)] + 2 * (lane / 4), 2);
      memcpy(&hi, w_addr[base + q * 8 + 2 * (lane % 4) + 1] + 2 * (lane / 4), 2);
      t_frag[q] = lo | (uint32_t)hi << 16;
    }
  }
  warp_meet(tid_x / 32);
}
static float b16f(uint32_t h, int32_t f16) {
  if (f16) {
    _Float16 x;
    uint16_t b = (uint16_t)h;
    memcpy(&x, &b, 2);
    return (float)x;
  }
  uint32_t w = h << 16;
  float x;
  memcpy(&x, &w, 4);
  return x;
}
/* `cvt.rn.{bf16,f16}x2.f32 d, hi, lo`: each rounded to nearest, ties to even
 * (subnormals kept, NaN quiet), `lo` in the low half. */
static uint32_t to_b16(float x, int32_t f16) {
  if (f16) {
    _Float16 h = (_Float16)x;
    uint16_t b;
    memcpy(&b, &h, 2);
    return b;
  }
  uint32_t b;
  memcpy(&b, &x, 4);
  if ((b & 0x7fffffff) > 0x7f800000) return (b >> 16) | 0x40;
  return (b + 0x7fff + ((b >> 16) & 1)) >> 16;
}
uint32_t emu_cvt_bf16x2(float hi, float lo) { return to_b16(lo, 0) | to_b16(hi, 0) << 16; }
uint32_t emu_cvt_f16x2(float hi, float lo) { return to_b16(lo, 1) | to_b16(hi, 1) << 16; }
/* `mma.sync.aligned.m16n8k16.row.col.f32.{bf16,f16}`: d = c + sum_k a b over
 * the warp's fragments. The products are exact; the sum is formed exactly
 * (double: 16 products of 8- or 11-bit significands and an f32) and truncated
 * toward zero to f32, as the measured part does (`mma_calib.cu`). */
void emu_mma16(int32_t f16, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
               uint32_t b1, float c0, float c1, float c2, float c3) {
  uint32_t *ab = w_ab[tid_x];
  ab[0] = a0, ab[1] = a1, ab[2] = a2, ab[3] = a3, ab[4] = b0, ab[5] = b1;
  w_c[tid_x][0] = c0, w_c[tid_x][1] = c1, w_c[tid_x][2] = c2, w_c[tid_x][3] = c3;
  warp_meet(tid_x / 32);
  int lane = tid_x & 31, base = tid_x & ~31, g = lane / 4, t = lane % 4;
  for (int r = 0; r < 4; r++) {
    int row = g + 8 * (r >> 1), col = 2 * t + (r & 1);
    double s = w_c[tid_x][r];
    for (int k = 0; k < 16; k++) {
      /* A (row, k): lane 4(row%8) + (k%8)/2, register (row/8) + 2(k/8). */
      uint32_t ar = w_ab[base + 4 * (row % 8) + (k % 8) / 2][(row / 8) + 2 * (k / 8)];
      /* B (k, col): lane 4 col + (k%8)/2, register 4 + k/8. */
      uint32_t br = w_ab[base + 4 * col + (k % 8) / 2][4 + k / 8];
      float a = b16f((ar >> (16 * (k % 2))) & 0xffff, f16), b = b16f((br >> (16 * (k % 2))) & 0xffff, f16);
      s += (double)a * (double)b;
    }
    float d = (float)s;
    if (isfinite(d) && fabs((double)d) > fabs(s)) d = nextafterf(d, 0.0f);
    memcpy(&t_frag[r], &d, 4);
  }
  warp_meet(tid_x / 32);
}
int32_t emu_ctaid_x(void) { return ctaid_x; }
int32_t emu_nctaid_x(void) { return nctaid_x; }
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
/* caching-memory `retain-pieces`: rows of `width` bytes placed at a pitch. */
void mapal_gpu_copy2d(uint64_t dst, uint64_t dpitch, uint64_t src,
                      uint64_t spitch, uint64_t width, uint64_t height) {
  for (uint64_t r = 0; r < height; r++)
    memmove((char *)(dst + r * dpitch), (const char *)(src + r * spitch), width);
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
  if (nptr > 32 || out >= nptr || !n) {
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
