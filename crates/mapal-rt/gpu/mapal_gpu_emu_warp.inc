/* Emulated block threads, warp meets, fragments and the narrow-float conversions.
 * Included by `mapal_gpu_emu.c` at this place in its translation unit;
 * not compiled on its own. */

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
/* ADR-0046: the SM's math units the device module may call. `sqrt.rn` is
 * IEEE `squareRoot` — the host's correctly rounded `sqrtf` / `sqrt`, the same
 * bits; the approximate units are modeled by the host's functions (their
 * measured error is the profile's, not the model's). */
float emu_sqrt_rn(float a) { return sqrtf(a); }
double emu_sqrt_rn64(double a) { return sqrt(a); }
float emu_sqrt_approx(float a) { return sqrtf(a); }
float emu_sin_approx(float a) { return sinf(a); }
float emu_cos_approx(float a) { return cosf(a); }

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
static _Thread_local uint32_t t_frag[128];
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
/* ADR-0043 slice 3: `cvt.rn.satfinite.{e4m3,e5m2}x2.f32` (`hi` to the upper
 * byte): round to nearest even, saturate to the largest finite, NaN to the
 * format's NaN — `mapal_ir::narrow::f8_from_f32`'s integer sequence. */
static uint32_t to_f8(float x, int32_t e5m2) {
  uint32_t b, a, c, drop = e5m2 ? 21 : 20, sign, m;
  uint32_t maxb = e5m2 ? 0x47600000u : 0x43e00000u, minn = e5m2 ? 0x38800000u : 0x3c800000u;
  uint32_t rebias = e5m2 ? 448u : 960u;
  float magic = e5m2 ? 128.0f : 16384.0f, y;
  uint32_t mb;
  memcpy(&b, &x, 4);
  sign = (b >> 24) & 0x80u;
  a = b & 0x7fffffffu;
  if (a > 0x7f800000u)
    return sign | 0x7fu;
  c = a < maxb ? a : maxb;
  if (c >= minn) {
    m = ((c + ((1u << (drop - 1)) - 1) + ((c >> drop) & 1u)) >> drop) - rebias;
  } else {
    memcpy(&y, &c, 4);
    y += magic;
    memcpy(&mb, &y, 4);
    memcpy(&m, &magic, 4);
    m = mb - m;
  }
  return (m | sign) & 0xffu;
}
uint16_t emu_cvt_e4m3x2(float hi, float lo) { return (uint16_t)(to_f8(lo, 0) | to_f8(hi, 0) << 8); }
uint16_t emu_cvt_e5m2x2(float hi, float lo) { return (uint16_t)(to_f8(lo, 1) | to_f8(hi, 1) << 8); }
/* `cvt.rn.f16x2.{e4m3,e5m2}x2`: each byte (the lower to the lower half) to its
 * f16 value, exactly (fp8 ⊂ f16). */
static float of_f8(uint32_t b, int32_t e5m2) {
  uint32_t s = (b & 0x80u) << 24, m = b & 0x7fu, r;
  float f;
  if (e5m2) {
    if (m >= 0x7cu) r = 0x7f800000u | ((m & 3u) << 21);
    else if (m < 4u) { f = (float)m * 0x1p-16f; memcpy(&r, &f, 4); }
    else r = (m << 21) + (112u << 23);
  } else {
    if (m == 0x7fu) r = 0x7fc00000u;
    else if (m < 8u) { f = (float)m * 0x1p-9f; memcpy(&r, &f, 4); }
    else r = (m << 20) + (120u << 23);
  }
  r |= s;
  memcpy(&f, &r, 4);
  return f;
}
uint32_t emu_cvt_f16x2_e4m3x2(uint16_t p) { return to_b16(of_f8(p & 0xffu, 0), 1) | to_b16(of_f8(p >> 8, 0), 1) << 16; }
uint32_t emu_cvt_f16x2_e5m2x2(uint16_t p) { return to_b16(of_f8(p & 0xffu, 1), 1) | to_b16(of_f8(p >> 8, 1), 1) << 16; }
/* `mma.sync.aligned.m16n8k16.row.col.f32.{bf16,f16}`: d = c + sum_k a b over
 * the warp's fragments. The products are exact; the sum is formed exactly
 * (double: 16 products of 8- or 11-bit significands and an f32) and truncated
 * toward zero to f32, as the measured part does (`mma_calib.cu`). */
/* `cvt.rna.tf32.f32`: round to nearest, ties away from zero, to 10 explicit
 * significand bits (the low 13 bits zero); a NaN or infinity unchanged. */
uint32_t emu_cvt_tf32(uint32_t x) {
  if ((x & 0x7f800000u) == 0x7f800000u)
    return x;
  return (x + 0x1000u) & 0xffffe000u;
}
uint32_t emu_cvt_tf32f(float f) {
  uint32_t x;
  memcpy(&x, &f, 4);
  return emu_cvt_tf32(x);
}
/* f32-tensor-cores: the accumulator datapath. MAPAL_EMU_MMA_G = g: the
 * addends (c and the exact products) aligned to the largest one's exponent,
 * each truncated toward zero below 2^(e_max - 23 - g), summed, and the sum
 * truncated toward zero to f32 (`mma_calib.cu`: the RTX 5080 keeps g = 2
 * on every warp format). Unset: the exact sum truncated once (g infinite). */
static int mma_g(void) {
  static int g = -2;
  if (g == -2) {
    const char *e = getenv("MAPAL_EMU_MMA_G");
    g = e ? atoi(e) : -1;
  }
  return g;
}
static float mma_sum(double c, const double *p, int n) {
  int g = mma_g();
  double s = c, mx = fabs(c);
  for (int i = 0; i < n; i++) mx = fmax(mx, fabs(p[i]));
  if (g < 0 || mx == 0 || !isfinite(mx) || isnan(c)) {
    for (int i = 0; i < n; i++) s += p[i];
  } else {
    int e;
    (void)frexp(mx, &e);
    double q = ldexp(1.0, (e - 1) - 23 - g);
    s = trunc(c / q) * q;
    for (int i = 0; i < n; i++) s += isnan(p[i]) ? p[i] : trunc(p[i] / q) * q;
  }
  float d = (float)s;
  if (isfinite(d) && fabs((double)d) > fabs(s)) d = nextafterf(d, 0.0f);
  return d;
}
/* f32-tensor-cores `split-tile`'s guard: `vote.sync.any` over the warp. */
static int32_t w_vote[1024];
int32_t emu_vote_any(int32_t p) {
  w_vote[tid_x] = p != 0;
  warp_meet(tid_x / 32);
  int32_t any = 0, base = tid_x & ~31;
  for (int l = 0; l < 32; l++) any |= w_vote[base + l];
  warp_meet(tid_x / 32);
  return any;
}
/* `cvt.rn.tf32.f32` (ADR-0039 `to_tf32`): to nearest, ties to even. */
uint32_t emu_cvt_rn_tf32f(float f) {
  uint32_t x;
  memcpy(&x, &f, 4);
  if ((x & 0x7f800000u) == 0x7f800000u)
    return x;
  return (x + 0xfffu + ((x >> 13) & 1u)) & 0xffffe000u;
}
/* hopper-wgmma-tma: `wgmma.mma_async.sync.aligned.m64nNk8.f32.tf32.tf32` with
 * `a` from registers and `b` from a matrix descriptor, as a meet of the
 * warpgroup's 128 threads (named barrier 12 + warpgroup). Each thread posts
 * its `a` fragment (warp w's rows 16w.., the m16n8k8 layout) and its
 * accumulators (`emu_acc_set`), then computes its own `d[4j + 2h + e]` =
 * (row 16w + g + 8h, column 8j + 2t + e): the products exact, the sum exact
 * and truncated toward zero to f32 (the `mma` model). Both operands are read
 * as the instruction reads them — the low 13 bits dropped, so an operand not
 * rounded before it is truncated — and `b`'s address comes from the
 * descriptor (start, LBO, SBO, swizzle mode) through the swizzle the box
 * writer applies, so a wrong descriptor is a wrong value.
 *
 * Asynchrony: the first thread of a warpgroup records every `b` element an
 * instruction read and its value; `wgmma.wait_group n` checks that the ones
 * of every group but the newest `n` still hold them (a stage released to the
 * copy engine before the group reading it was waited for is a failure, 110). */
static _Thread_local float t_acc[128];
void emu_acc_set(int32_t i, float v) { t_acc[i] = v; }

