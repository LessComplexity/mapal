/* hopper-launch: the runtime's tensor-map launch path on a real device (any
 * part with the tensor copy engine). A module with one TMA kernel
 * `mapal_tile_00000000000000a1` and its recipe global spelled as NVPTX spells
 * an `[n x i64]` constant (`.b8` bytes); the kernel copies a 32 x 8 box of `a`
 * (128-byte swizzle) and writes its first 8 words to `o`. A plain kernel
 * (`...b2`, BUSY=1) keeps the device busy before the clock as a program's
 * operand generators do; SELF=1 launches the TMA kernel once before the clock
 * (its own first launch paid there).
 *   cc -O2 tma_launch.c mapal_gpu.c -lcuda -lpthread -ldl -lm -o tma_launch
 *   tma_launch <flags>   (mapal_gpu_init's bits: 1 eager-init, 8 preload,
 *                         64 the `tma-warm` OFF arm)
 * Prints the clocked first launch (us; `launch` the host call alone), four
 * launches alternating two buffers (the map cache: no encode), and checks the
 * copied values (exit 1 on a wrong word). */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void mapal_gpu_init(uint32_t f);
uint64_t mapal_gpu_alloc(uint64_t *slot, uint64_t bytes);
void mapal_gpu_h2d(uint64_t d, const void *h, uint64_t bytes);
void mapal_gpu_d2h(void *h, uint64_t d, uint64_t off, uint64_t bytes);
void mapal_gpu_launchv(uint64_t id, uint32_t stream, uint32_t nptr,
                       const uint64_t *ptrs, uint32_t gx, uint32_t gy,
                       uint32_t block, uint32_t smem);
void mapal_gpu_sync(void);

static double us(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}

static const char kernel_ptx[] =
    ".version 8.0\n.target sm_90\n.address_size 64\n"
    ".extern .shared .align 1024 .b8 mapal_smem[];\n"
    "%s\n"
    ".visible .entry mapal_tile_00000000000000a1(.param .u64 a, .param .u64 o, "
    ".param .align 64 .b8 tm[128]) {\n"
    "  .reg .b64 %%rd<6>; .reg .b32 %%r<8>; .reg .f32 %%f<2>; .reg .pred %%p;\n"
    "  mov.u32 %%r0, %%tid.x; setp.ne.u32 %%p, %%r0, 0; @%%p bra done;\n"
    "  mov.b64 %%rd0, tm; cvta.param.u64 %%rd1, %%rd0;\n"
    "  mov.u32 %%r1, mapal_smem; add.u32 %%r2, %%r1, 1024;\n"
    "  mbarrier.init.shared::cta.b64 [%%r2], 1;\n"
    "  fence.proxy.async.shared::cta;\n"
    "  mbarrier.arrive.expect_tx.shared::cta.b64 _, [%%r2], 1024;\n"
    "  mov.u32 %%r3, 0;\n"
    "  cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::"
    "complete_tx::bytes [%%r1], [%%rd1, {%%r3, %%r3}], [%%r2];\n"
    "w: mbarrier.try_wait.parity.shared::cta.b64 %%p, [%%r2], 0; @!%%p bra w;\n"
    "  ld.param.u64 %%rd2, [o];\n"
    "  mov.u32 %%r4, 0;\n"
    "l: shl.b32 %%r5, %%r4, 2; add.u32 %%r6, %%r1, %%r5; ld.shared.f32 %%f1, [%%r6];\n"
    "  mul.wide.u32 %%rd3, %%r4, 4; add.u64 %%rd4, %%rd2, %%rd3; st.global.f32 [%%rd4], %%f1;\n"
    "  add.u32 %%r4, %%r4, 1; setp.lt.u32 %%p, %%r4, 8; @%%p bra l;\n"
    "done: ret;\n}\n"
    ".visible .entry mapal_tile_00000000000000b2(.param .u64 a, .param .u64 o) {\n"
    "  .reg .b32 %%r<4>; .reg .pred %%p;\n"
    "  mov.u32 %%r0, 0;\n"
    "s: add.u32 %%r0, %%r0, 1; setp.lt.u32 %%p, %%r0, 200000; @%%p bra s;\n"
    "  ret;\n}\n";

int main(int argc, char **argv) {
  uint32_t flags = argc > 1 ? (uint32_t)atoi(argv[1]) : 8;
  /* The recipe: 1 map; argument 0, u32, rank 2, swizzle 128, offset 0,
   * dims (32, 8), stride 128 B, box (32, 8). */
  uint64_t rec[21] = {1, 0, 4, 2, 128, 0, 32, 8, 1, 1, 1, 128, 0, 0, 0,
                      32, 8, 1, 1, 1, 0};
  char glob[4096];
  int n = snprintf(glob, sizeof glob,
                   ".visible .global .align 8 .b8 "
                   "mapal_tile_00000000000000a1_tmap[%d] = {",
                   (int)sizeof rec);
  /* NVPTX leaves an initializer's trailing zero bytes out. */
  size_t last = sizeof rec;
  while (last > 1 && !((unsigned char *)rec)[last - 1])
    last--;
  for (size_t i = 0; i < last; i++)
    n += snprintf(glob + n, sizeof glob - n, "%s%u", i ? ", " : "",
                  ((unsigned char *)rec)[i]);
  snprintf(glob + n, sizeof glob - n, "};");
  static char ptx[16384];
  snprintf(ptx, sizeof ptx, kernel_ptx, glob);
  const char *path = getenv("TMA_PTX_OUT");
  if (!path)
    path = "tma_launch.ptx";
  FILE *f = fopen(path, "w");
  fputs(ptx, f);
  fclose(f);
  setenv("MAPAL_PTX", path, 1);

  mapal_gpu_init(flags);
  uint64_t a[2] = {0, 0}, o = 0;
  float h[2][256];
  for (int b = 0; b < 2; b++) {
    mapal_gpu_alloc(&a[b], sizeof h[b]);
    for (int i = 0; i < 256; i++)
      h[b][i] = (float)(1000 * b + i);
    mapal_gpu_h2d(a[b], h[b], sizeof h[b]);
  }
  mapal_gpu_alloc(&o, 64);
  if (getenv("BUSY")) {
    uint64_t pb[2] = {a[1], o};
    mapal_gpu_launchv(0xb2, 0, 2, pb, 132, 1, 128, 0);
  }
  if (getenv("SELF")) {
    uint64_t ps[2] = {a[1], o};
    mapal_gpu_launchv(0xa1, 0, 2, ps, 1, 1, 32, 2048);
  }
  if (getenv("SLEEP_MS")) {
    struct timespec ts = {0, atoi(getenv("SLEEP_MS")) * 1000000L};
    nanosleep(&ts, 0);
  }
  mapal_gpu_sync();
  double t0 = us();
  uint64_t p0[2] = {a[0], o};
  mapal_gpu_launchv(0xa1, 0, 2, p0, 1, 1, 32, getenv("SM") ? atoi(getenv("SM")) : 2048);
  double tl = us();
  mapal_gpu_sync();
  double t1 = us();
  float r[8];
  mapal_gpu_d2h(r, o, 0, sizeof r);
  int bad = 0;
  /* The 128-byte swizzle leaves row 0 in place. */
  for (int i = 0; i < 8; i++)
    bad += r[i] != h[0][i];
  double t2 = us();
  for (int it = 0; it < 4; it++) {
    uint64_t p[2] = {a[it % 2], o};
    mapal_gpu_launchv(0xa1, 0, 2, p, 1, 1, 32, 2048);
  }
  mapal_gpu_sync();
  double t3 = us();
  mapal_gpu_d2h(r, o, 0, sizeof r);
  for (int i = 0; i < 8; i++)
    bad += r[i] != h[1][i];
  printf("flags=%u first_us=%.1f (launch %.1f) alt4_us=%.1f bad=%d\n", flags,
         t1 - t0, tl - t0, t3 - t2, bad);
  return bad != 0;
}
