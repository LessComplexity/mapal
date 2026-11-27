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
#include <unistd.h>

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
/* One block's shared memory (blocks run one at a time). Aligned to 32 KiB —
 * the most a Mach-O common symbol allows (macOS) — so a swizzle, which reads
 * address bits [4, 10), applies to the same bits the hardware's does. A
 * `wgmma` descriptor's 14-bit start field is (address >> 4) mod 2^14: the
 * decoders below subtract the buffer's own low 18 address bits to recover the
 * offset (`emu_desc_start`), which the device names by its window offset. */
unsigned char emu_smem[1 << 20] __attribute__((aligned(1 << 15)));
/* A matrix descriptor's start field as an offset into `emu_smem`. */
static uint64_t emu_desc_start(uint64_t desc) {
  const uint64_t window = (uint64_t)1 << 18;
  uint64_t base = (uint64_t)(uintptr_t)emu_smem & (window - 1);
  return (((desc & 0x3fff) << 4) + window - base) & (window - 1);
}
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

#include "mapal_gpu_emu_copy.inc"
#include "mapal_gpu_emu_warp.inc"
#include "mapal_gpu_emu_wgmma.inc"
#include "mapal_gpu_emu_host.inc"
#include "mapal_gpu_emu_launch.inc"
#include "mapal_gpu_emu_sites.inc"
