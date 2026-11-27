/* Emulated async copies, bulk and tensor copies, and mbarriers.
 * Included by `mapal_gpu_emu.c` at this place in its translation unit;
 * not compiled on its own. */

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
/* TMA `bulk-stage`: a transaction barrier in its 8 shared bytes — word 0 the
 * phase (bit 31), the arrivals a phase expects (bits 16-30) and those still
 * pending (bits 0-15); word 1 the signed transaction count — and the bulk
 * copies as one deferred queue shared by the block's threads, applied when a
 * thread waits on their barrier: a missing wait reads stale shared memory, a
 * wait on a phase that never completes exits (108) instead of spinning. In a
 * serialized block an unfinished wait yields the turn (the copies it waits
 * for may be a later thread's to issue: a warp spinning while another warp
 * issues). */
#define EMU_BULK_MAX 8192
struct emu_bulk_copy {
  void *d;
  const void *s;
  uint32_t n;
  uint32_t *bar;
  /* hopper-wgmma-tma: a tensor box (`s` its map, `c` its coordinates). */
  int32_t c[5];
};
/* hopper-wgmma-tma: the emulator's tensor map, written into the 128 bytes a
 * kernel takes by value (`emu_tmap_encode`) — the recipe's fields as the box
 * copy reads them. */
struct emu_tmap {
  uint64_t base;
  uint32_t elem, rank, swizzle, pad;
  uint32_t dims[5], box[5];
  uint64_t strides[5];
};
_Static_assert(sizeof(struct emu_tmap) <= 128, "a tensor map is 128 bytes");
/* The `swizzle`-byte pattern of the address `o` (an offset into `emu_smem`):
 * its 16-byte chunk bits [4, 4+b) XOR its bits [7, 7+b), b = log2(span/16). */
static uint64_t emu_swz(uint64_t o, uint32_t swizzle) {
  if (swizzle < 32) return o;
  uint64_t mask = swizzle / 16 - 1;
  return o ^ (((o >> 7) & mask) << 4);
}
/* A box of `t` landing at `d`: each element at its dense offset, the swizzle
 * applied to its shared address; a coordinate outside the tensor reads 0. */
static void emu_box(unsigned char *d, const struct emu_tmap *t, const int32_t *c) {
  uint64_t o0 = (uint64_t)(d - emu_smem);
  if (t->swizzle >= 32 && o0 % (8 * (uint64_t)t->swizzle)) {
    fprintf(stderr, "emu: a %u-byte swizzled box at shared offset %llu (not a whole pattern)\n", t->swizzle,
            (unsigned long long)o0);
    exit(109);
  }
  uint64_t n = 1;
  for (uint32_t i = 0; i < t->rank; i++) n *= t->box[i];
  for (uint64_t e = 0; e < n; e++) {
    uint64_t r = e, src = 0;
    int in = 1;
    for (uint32_t i = 0; i < t->rank; i++) {
      int64_t x = (int64_t)c[i] + (int64_t)(r % t->box[i]);
      r /= t->box[i];
      if (x < 0 || x >= (int64_t)t->dims[i]) in = 0;
      src += (uint64_t)x * t->strides[i];
    }
    unsigned char *q = emu_smem + emu_swz(o0 + e * t->elem, t->swizzle);
    if (in)
      memcpy(q, (const unsigned char *)t->base + src, t->elem);
    else
      memset(q, 0, t->elem);
  }
}
static struct emu_bulk_copy bq[EMU_BULK_MAX];
static uint32_t bq_len;
static pthread_mutex_t bq_mu = PTHREAD_MUTEX_INITIALIZER;
static void mbar_complete(uint32_t *b) {
  if ((b[0] & 0xffffu) == 0 && (int32_t)b[1] == 0) {
    uint32_t count = (b[0] >> 16) & 0x7fffu;
    b[0] = ((b[0] & 0x80000000u) ^ 0x80000000u) | (count << 16) | count;
  }
}
void emu_mbar_init(uint32_t *b, uint32_t count, uint32_t tid) {
  if (tid)
    return;
  pthread_mutex_lock(&bq_mu);
  b[0] = (count << 16) | count;
  b[1] = 0;
  pthread_mutex_unlock(&bq_mu);
}
void emu_mbar_expect(uint32_t *b, uint32_t bytes, uint32_t tid) {
  if (tid)
    return;
  pthread_mutex_lock(&bq_mu);
  if ((b[0] & 0xffffu) == 0) {
    fprintf(stderr, "emu: mbarrier arrival on a complete phase\n");
    exit(107);
  }
  b[1] = (uint32_t)((int32_t)b[1] + (int32_t)bytes);
  b[0]--;
  mbar_complete(b);
  pthread_mutex_unlock(&bq_mu);
}
void emu_bulk(void *d, const void *s, uint32_t n, uint32_t *b, uint32_t on) {
  if (!on)
    return;
  pthread_mutex_lock(&bq_mu);
  if (bq_len == EMU_BULK_MAX) {
    fprintf(stderr, "emu: bulk copy queue full\n");
    exit(103);
  }
  bq[bq_len++] = (struct emu_bulk_copy){d, s, n, b, {INT32_MIN}};
  pthread_mutex_unlock(&bq_mu);
}
/* hopper-wgmma-tma: `cp.async.bulk.tensor.<rank>d ... [d], [map, {c…}], [b]`
 * — queued like a row copy, its bytes the whole box (out-of-bounds elements
 * included, as the engine counts them). */
void emu_bulk_tensor(void *d, uint64_t map, int32_t c0, int32_t c1, int32_t c2, int32_t c3, int32_t c4,
                     uint32_t *b) {
  const struct emu_tmap *t = (const struct emu_tmap *)(uintptr_t)map;
  uint32_t n = t->elem;
  for (uint32_t i = 0; i < t->rank; i++) n *= t->box[i];
  pthread_mutex_lock(&bq_mu);
  if (bq_len == EMU_BULK_MAX) {
    fprintf(stderr, "emu: bulk copy queue full\n");
    exit(103);
  }
  bq[bq_len++] = (struct emu_bulk_copy){d, t, n, b, {c0, c1, c2, c3, c4}};
  pthread_mutex_unlock(&bq_mu);
}
/* `mbarrier.arrive` (no transaction): one of the phase's arrivals. */
void emu_mbar_arrive(uint32_t *b) {
  pthread_mutex_lock(&bq_mu);
  if ((b[0] & 0xffffu) == 0) {
    fprintf(stderr, "emu: mbarrier arrival on a complete phase\n");
    exit(107);
  }
  b[0]--;
  mbar_complete(b);
  pthread_mutex_unlock(&bq_mu);
}
static _Thread_local uint64_t mbar_spins;
uint32_t emu_mbar_try(uint32_t *b, uint32_t parity) {
  pthread_mutex_lock(&bq_mu);
  uint32_t keep = 0;
  for (uint32_t i = 0; i < bq_len; i++) {
    if (bq[i].bar == b) {
      if (bq[i].c[0] != INT32_MIN)
        emu_box(bq[i].d, bq[i].s, bq[i].c);
      else
        memcpy(bq[i].d, bq[i].s, bq[i].n);
      b[1] = (uint32_t)((int32_t)b[1] - (int32_t)bq[i].n);
    } else {
      bq[keep++] = bq[i];
    }
  }
  bq_len = keep;
  mbar_complete(b);
  uint32_t done = (b[0] >> 31) != parity;
  pthread_mutex_unlock(&bq_mu);
  if (done) {
    mbar_spins = 0;
  } else if (serial()) {
    /* A serialized block: the copies this thread waits for are another's
     * to issue — hand the turn on (this thread stays runnable). */
    pthread_mutex_lock(&b_mu);
    s_pass(tid_x);
    s_wait();
    pthread_mutex_unlock(&b_mu);
  }
  if (!done && ++mbar_spins > (1u << 24)) {
    fprintf(stderr, "emu: mbarrier phase %u never completes (state %08x tx %d, %u copies queued)\n", parity,
            b[0], (int32_t)b[1], bq_len);
    exit(108);
  }
  return done;
}

