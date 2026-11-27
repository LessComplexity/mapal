/* Emulated warpgroup and warp matrix instructions.
 * Included by `mapal_gpu_emu.c` at this place in its translation unit;
 * not compiled on its own. */

static uint32_t wg_a[1024][4];
static float wg_c[1024][128];
static uint64_t wg_d[1024];
#define WG_REC_MAX (1 << 17)
static struct { const uint32_t *p; uint32_t v, g; } wg_rec[4][WG_REC_MAX];
static uint32_t wg_nrec[4], wg_group[4];
static const uint32_t *wg_b_at(uint64_t desc, int col, int k) {
  uint64_t start = emu_desc_start(desc), lbo = ((desc >> 16) & 0x3fff) << 4, sbo = ((desc >> 32) & 0x3fff) << 4;
  uint32_t mode = (uint32_t)(desc >> 62) & 3;
  uint32_t span = mode == 1 ? 128 : mode == 2 ? 64 : mode == 3 ? 32 : 0;
  uint64_t kb = (uint64_t)k * 4, o;
  if (span)
    o = emu_swz(start + (uint64_t)(col / 8) * sbo + (uint64_t)(col % 8) * span + kb, span);
  else
    o = start + (uint64_t)(col / 8) * sbo + (uint64_t)(col % 8) * 16 + kb / 16 * lbo + kb % 16;
  if (o + 4 > sizeof emu_smem) {
    fprintf(stderr, "emu: wgmma descriptor reads past shared memory\n");
    exit(110);
  }
  return (const uint32_t *)(emu_smem + o);
}
void emu_wgmma(int32_t n, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {
  int wgi = tid_x / 128, base = wgi * 128, me = tid_x - base;
  wg_a[tid_x][0] = a0, wg_a[tid_x][1] = a1, wg_a[tid_x][2] = a2, wg_a[tid_x][3] = a3;
  memcpy(wg_c[tid_x], t_acc, (size_t)(n / 2) * sizeof(float));
  wg_d[tid_x] = desc;
  emu_barrier_k(12 + wgi, 128);
  int w = me / 32, lane = me % 32, g = lane / 4, t = lane % 4;
  for (int j = 0; j < n / 8; j++)
    for (int h = 0; h < 2; h++)
      for (int e = 0; e < 2; e++) {
        int q = 4 * j + 2 * h + e, row = 16 * w + g + 8 * h, col = 8 * j + 2 * t + e;
        double s = wg_c[tid_x][q];
        for (int k = 0; k < 8; k++) {
          int src = base + 32 * (row / 16) + 4 * (row % 8) + k % 4;
          uint32_t ar = wg_a[src][(row % 16) / 8 + 2 * (k / 4)] & 0xffffe000u;
          uint32_t br = *wg_b_at(wg_d[base], col, k) & 0xffffe000u;
          float a, b;
          memcpy(&a, &ar, 4);
          memcpy(&b, &br, 4);
          s += (double)a * (double)b;
        }
        float d = (float)s;
        if (isfinite(d) && fabs((double)d) > fabs(s)) d = nextafterf(d, 0.0f);
        memcpy(&t_frag[q], &d, 4);
      }
  if (me == 0) {
    for (int col = 0; col < n; col++)
      for (int k = 0; k < 8; k++) {
        if (wg_nrec[wgi] == WG_REC_MAX) {
          fprintf(stderr, "emu: wgmma record full\n");
          exit(110);
        }
        const uint32_t *p = wg_b_at(wg_d[base], col, k);
        wg_rec[wgi][wg_nrec[wgi]].p = p;
        wg_rec[wgi][wg_nrec[wgi]].v = *p;
        wg_rec[wgi][wg_nrec[wgi]++].g = wg_group[wgi];
      }
  }
  emu_barrier_k(12 + wgi, 128);
}
/* wgmma-16bit: `wgmma.mma_async.sync.aligned.m64nNk16.f32.<bf16|f16>` —
 * `a` from registers (`amode` 0: the m64k16 fragments, warp w's rows 16w..,
 * the `mma.m16n8k16` layout of each 16 rows) or a K-major descriptor
 * (`amode` 1), `b` from a descriptor K-major (`tb` 0) or MN-major (`tb` 1,
 * the transpose bit). A descriptor element's byte is its canonical 128-byte
 * swizzled layout's (PTX ISA, "Shared Memory Matrix Layout"): K-major
 * `start + (mn/8)·SBO + (mn%8)·128 + 2k`; MN-major `start + (mn/64)·LBO +
 * (k/8)·SBO + (k%8)·128 + 2·(mn%64)` — the swizzle applied to the address
 * (so a wrong LBO / SBO / transpose bit is a wrong value). Each product
 * exact, each k16 step summed as `mma_sum` models the accumulator (the warp
 * `mma.m16n8k16`'s, `emu_mma16`); the smem elements an instruction read are
 * recorded for `emu_wg_wait` as `emu_wgmma`'s. */
static uint64_t wg_ad[1024];
static const unsigned char *wg16_at(uint64_t desc, int mn, int k, int mn_major) {
  uint64_t start = emu_desc_start(desc), lbo = ((desc >> 16) & 0x3fff) << 4, sbo = ((desc >> 32) & 0x3fff) << 4;
  uint32_t mode = (uint32_t)(desc >> 62) & 3;
  if (mode != 1) {
    fprintf(stderr, "emu: wgmma16 descriptor mode %u (the kernel writes 128-byte swizzled boxes)\n", mode);
    exit(110);
  }
  uint64_t o = mn_major ? start + (uint64_t)(mn / 64) * lbo + (uint64_t)(k / 8) * sbo + (uint64_t)(k % 8) * 128 +
                              (uint64_t)(mn % 64) * 2
                        : start + (uint64_t)(mn / 8) * sbo + (uint64_t)(mn % 8) * 128 + (uint64_t)k * 2;
  o = emu_swz(o, 128);
  if (o + 2 > sizeof emu_smem) {
    fprintf(stderr, "emu: wgmma16 descriptor reads past shared memory\n");
    exit(110);
  }
  return emu_smem + o;
}
static void wg16_rec(int wgi, const unsigned char *e) {
  const uint32_t *p = (const uint32_t *)((uintptr_t)e & ~(uintptr_t)3);
  if (wg_nrec[wgi] == WG_REC_MAX) {
    fprintf(stderr, "emu: wgmma record full\n");
    exit(110);
  }
  wg_rec[wgi][wg_nrec[wgi]].p = p;
  wg_rec[wgi][wg_nrec[wgi]].v = *p;
  wg_rec[wgi][wg_nrec[wgi]++].g = wg_group[wgi];
}
void emu_wgmma16(int32_t n, int32_t f16, int32_t amode, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                 uint64_t adesc, uint64_t bdesc, int32_t tb) {
  int wgi = tid_x / 128, base = wgi * 128, me = tid_x - base;
  wg_a[tid_x][0] = a0, wg_a[tid_x][1] = a1, wg_a[tid_x][2] = a2, wg_a[tid_x][3] = a3;
  memcpy(wg_c[tid_x], t_acc, (size_t)(n / 2) * sizeof(float));
  wg_ad[tid_x] = adesc;
  wg_d[tid_x] = bdesc;
  emu_barrier_k(12 + wgi, 128);
  int w = me / 32, lane = me % 32, g = lane / 4, t = lane % 4;
  for (int j = 0; j < n / 8; j++)
    for (int h = 0; h < 2; h++)
      for (int e = 0; e < 2; e++) {
        int q = 4 * j + 2 * h + e, row = 16 * w + g + 8 * h, col = 8 * j + 2 * t + e;
        double pr[16];
        for (int k = 0; k < 16; k++) {
          uint32_t av, bv;
          if (amode == 0) {
            int src = base + 32 * (row / 16) + 4 * (row % 8) + (k % 8) / 2;
            av = (wg_a[src][(row % 16) / 8 + 2 * (k / 8)] >> (16 * (k % 2))) & 0xffff;
          } else {
            uint16_t x;
            memcpy(&x, wg16_at(wg_ad[base], row, k, 0), 2);
            av = x;
          }
          uint16_t y;
          memcpy(&y, wg16_at(wg_d[base], col, k, tb), 2);
          bv = y;
          pr[k] = (double)b16f(av, f16) * (double)b16f(bv, f16);
        }
        float d = mma_sum(wg_c[tid_x][q], pr, 16);
        memcpy(&t_frag[q], &d, 4);
      }
  if (me == 0) {
    for (int k = 0; k < 16; k++) {
      for (int col = 0; col < n; col++) wg16_rec(wgi, wg16_at(wg_d[base], col, k, tb));
      if (amode)
        for (int row = 0; row < 64; row++) wg16_rec(wgi, wg16_at(wg_ad[base], row, k, 0));
    }
  }
  emu_barrier_k(12 + wgi, 128);
}
void emu_wg_commit(void) {
  if (tid_x % 128 == 0) wg_group[tid_x / 128]++;
}
void emu_wg_wait(int32_t pending) {
  int wgi = tid_x / 128;
  if (tid_x % 128) return;
  uint32_t keep = 0;
  for (uint32_t i = 0; i < wg_nrec[wgi]; i++) {
    if (wg_rec[wgi][i].g + (uint32_t)pending < wg_group[wgi]) {
      if (*wg_rec[wgi][i].p != wg_rec[wgi][i].v) {
        fprintf(stderr, "emu: a wgmma operand changed before its wait_group (group %u)\n", wg_rec[wgi][i].g);
        exit(110);
      }
    } else {
      wg_rec[wgi][keep++] = wg_rec[wgi][i];
    }
  }
  wg_nrec[wgi] = keep;
}
void emu_mma16(int32_t f16, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
               uint32_t b1, float c0, float c1, float c2, float c3) {
  uint32_t *ab = w_ab[tid_x];
  ab[0] = a0, ab[1] = a1, ab[2] = a2, ab[3] = a3, ab[4] = b0, ab[5] = b1;
  w_c[tid_x][0] = c0, w_c[tid_x][1] = c1, w_c[tid_x][2] = c2, w_c[tid_x][3] = c3;
  warp_meet(tid_x / 32);
  int lane = tid_x & 31, base = tid_x & ~31, g = lane / 4, t = lane % 4;
  /* TMA `bulk-stage` tests: `f16 == 2` is `mma.sync.m16n8k8.tf32` — A (row,
   * k) in lane 4(row%8) + k%4, register row/8 + 2(k/4); B (k, col) in lane
   * 4 col + k%4, register 4 + k/4; the operands' 32 bits the TF32 values. */
  if (f16 == 2) {
    for (int r = 0; r < 4; r++) {
      int row = g + 8 * (r >> 1), col = 2 * t + (r & 1);
      double pr[8];
      for (int k = 0; k < 8; k++) {
        uint32_t ar = w_ab[base + 4 * (row % 8) + k % 4][(row / 8) + 2 * (k / 4)];
        uint32_t br = w_ab[base + 4 * col + k % 4][4 + k / 4];
        float a, b;
        memcpy(&a, &ar, 4);
        memcpy(&b, &br, 4);
        pr[k] = (double)a * (double)b;
      }
      float d = mma_sum(w_c[tid_x][r], pr, 8);
      memcpy(&t_frag[r], &d, 4);
    }
    warp_meet(tid_x / 32);
    return;
  }
  for (int r = 0; r < 4; r++) {
    int row = g + 8 * (r >> 1), col = 2 * t + (r & 1);
    double pr[16];
    for (int k = 0; k < 16; k++) {
      /* A (row, k): lane 4(row%8) + (k%8)/2, register (row/8) + 2(k/8). */
      uint32_t ar = w_ab[base + 4 * (row % 8) + (k % 8) / 2][(row / 8) + 2 * (k / 8)];
      /* B (k, col): lane 4 col + (k%8)/2, register 4 + k/8. */
      uint32_t br = w_ab[base + 4 * col + (k % 8) / 2][4 + k / 8];
      float a = b16f((ar >> (16 * (k % 2))) & 0xffff, f16), b = b16f((br >> (16 * (k % 2))) & 0xffff, f16);
      pr[k] = (double)a * (double)b;
    }
    float d = mma_sum(w_c[tid_x][r], pr, 16);
    memcpy(&t_frag[r], &d, 4);
  }
  warp_meet(tid_x / 32);
}
int32_t emu_ctaid_x(void) { return ctaid_x; }
int32_t emu_nctaid_x(void) { return nctaid_x; }
int32_t emu_ctaid_y(void) { return ctaid_y; }
static unsigned long long n_h2d, n_d2h, n_launch, b_in, b_out, n_zero;

