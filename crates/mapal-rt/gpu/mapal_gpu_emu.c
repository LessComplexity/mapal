/* A host emulation of mapal_gpu.c's entries (S54), for checking the host half
 * of the resident path on a machine with no GPU: device memory is host memory
 * (filled with 0xA5, so a read of a cell nothing wrote shows), and a launch
 * runs the kernel once per thread, in order. The device unit is compiled for
 * the host with its special registers renamed to `emu_*` (the test that links
 * this does it). A staged kernel (smem > 0) needs barriers and exits 102: only
 * the one-thread-per-cell and devgen kernels run here.
 *
 * Used by crates/backends/llvm/tests/retain_delta.rs; never shipped. */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int32_t tid_x, ctaid_x, ctaid_y;
int32_t emu_tid_x(void) { return tid_x; }
int32_t emu_ctaid_x(void) { return ctaid_x; }
int32_t emu_ctaid_y(void) { return ctaid_y; }
static unsigned long long n_h2d, n_d2h, n_launch, b_in, b_out;
static void total(void) {
  fprintf(stderr, "emu_total h2d=%llu d2h=%llu launches=%llu bytes_in=%llu bytes_out=%llu\n", n_h2d, n_d2h,
          n_launch, b_in, b_out);
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
/* `arena` (S54): one allocation per function, made on the first ask. */
uint64_t mapal_gpu_arena(uint64_t *slot, uint64_t bytes) { return mapal_gpu_alloc(slot, bytes); }
void mapal_gpu_h2d(uint64_t d, const void *h, uint64_t bytes) {
  memcpy((void *)d, h, bytes);
  n_h2d++;
  b_in += bytes;
}
void mapal_gpu_d2h(void *h, uint64_t d, uint64_t off, uint64_t bytes) {
  memcpy(h, (void *)(d + off), bytes);
  n_d2h++;
  b_out += bytes;
}
void mapal_gpu_free(uint64_t *slot) {
  if (*slot) free((void *)*slot);
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
void mapal_gpu_launch(uint64_t id, uint64_t a, uint64_t b, uint64_t o, uint32_t gx, uint32_t gy, uint32_t block,
                      uint32_t smem) {
  if (smem) {
    fprintf(stderr, "emu: a staged kernel (smem %u) needs barriers\n", smem);
    exit(102);
  }
  void (*k)(void *, void *, void *) = (void (*)(void *, void *, void *))sym("mapal_tile", id);
  for (ctaid_y = 0; ctaid_y < (int32_t)gy; ctaid_y++)
    for (ctaid_x = 0; ctaid_x < (int32_t)gx; ctaid_x++)
      for (tid_x = 0; tid_x < (int32_t)block; tid_x++) k((void *)a, (void *)b, (void *)o);
  n_launch++;
}
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
