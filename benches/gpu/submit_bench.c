/* S62 submission stage 2: one kernel, its buffers and geometry, launched through
 * the generated runtime entry (`mapal_gpu_launchv` / `mapal_gpu_launch`: name
 * lookup, stream bookkeeping, cuLaunchKernel) and through a minimal direct loop
 * (CUfunction looked up once, cuLaunchKernel only), on the legacy stream like
 * the shipped probes. Per-launch CPU time of each call, no synchronize inside a
 * batch; batches alternate generated / direct / direct / generated (ABBA), each
 * followed by a context synchronize. The two direct batches of a round are each
 * other's null.
 *
 *   cc submit_bench.c mapal_gpu.c -I$CUDA/include -L$CUDA/lib64/stubs -lcuda -lpthread -ldl -lm
 *   MAPAL_PTX=<ptx> ./submit_bench <launch|launchv> <id> <nptr> <bytes> <gx> <gy> <block> <smem> <rounds> <batch>
 */
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

uint64_t mapal_gpu_alloc(uint64_t *slot, uint64_t bytes);
void mapal_gpu_launch(uint64_t id, uint64_t a, uint64_t b, uint64_t out, uint32_t gx, uint32_t gy,
                      uint32_t block, uint32_t smem);
void mapal_gpu_launchv(uint64_t id, uint32_t stream, uint32_t nptr, const uint64_t *ptrs, uint32_t gx,
                       uint32_t gy, uint32_t block, uint32_t smem);
void mapal_gpu_sync(void);

#define CK(x)                                                                                    \
  do {                                                                                           \
    CUresult r_ = (x);                                                                           \
    if (r_ != CUDA_SUCCESS) {                                                                    \
      fprintf(stderr, "submit_bench: %s failed (%d)\n", #x, (int)r_);                            \
      exit(102);                                                                                 \
    }                                                                                            \
  } while (0)

static double now_us(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e6 + t.tv_nsec / 1e3;
}
static int cmp(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return x < y ? -1 : x > y;
}
static double median(double *v, int n) {
  qsort(v, n, sizeof *v, cmp);
  return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;
}

int main(int argc, char **argv) {
  if (argc != 11) {
    fprintf(stderr, "usage: submit_bench <launch|launchv> <id> <nptr> <bytes> <gx> <gy> <block> <smem> <rounds> <batch>\n");
    return 2;
  }
  int v = strcmp(argv[1], "launchv") == 0;
  uint64_t id = strtoull(argv[2], 0, 0);
  uint32_t nptr = atoi(argv[3]), gx = atoi(argv[5]), gy = atoi(argv[6]), block = atoi(argv[7]),
           smem = atoi(argv[8]);
  uint64_t bytes = strtoull(argv[4], 0, 0);
  int rounds = atoi(argv[9]), batch = atoi(argv[10]);
  if (nptr > 16 || (!v && nptr != 3)) return 2;
  uint64_t slots[16] = {0};
  for (uint32_t i = 0; i < nptr; i++) mapal_gpu_alloc(&slots[i], bytes); /* loads the module */

  /* The direct path: its own module from the same PTX, the function looked up once. */
  FILE *f = fopen(getenv("MAPAL_PTX"), "rb");
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *ptx = malloc(n + 1);
  if (fread(ptx, 1, n, f) != (size_t)n) return 102;
  ptx[n] = 0;
  fclose(f);
  CUmodule mod;
  CUfunction fn;
  char name[40];
  CK(cuModuleLoadData(&mod, ptx));
  snprintf(name, sizeof name, "mapal_tile_%016llx", (unsigned long long)id);
  CK(cuModuleGetFunction(&fn, mod, name));
  if (smem > 48 * 1024) CK(cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem));
  CUdeviceptr d[16];
  void *args[16];
  for (uint32_t i = 0; i < nptr; i++) {
    d[i] = slots[i];
    args[i] = &d[i];
  }

  int total = rounds * batch * 2;
  double *g = malloc(total * sizeof *g), *dr = malloc(total * sizeof *dr);
  double *da = malloc(rounds * batch * sizeof *da), *db = malloc(rounds * batch * sizeof *db);
  int ng = 0, nd = 0, na = 0, nb = 0;
  for (int w = 0; w < 2; w++) { /* warm both paths */
    for (int j = 0; j < batch; j++) {
      if (v) mapal_gpu_launchv(id, 0, nptr, slots, gx, gy, block, smem);
      else mapal_gpu_launch(id, slots[0], slots[1], slots[2], gx, gy, block, smem);
      CK(cuLaunchKernel(fn, gx, gy, 1, block, 1, 1, smem, 0, args, 0));
    }
    mapal_gpu_sync();
  }
  for (int r = 0; r < rounds; r++) {
    for (int k = 0; k < 4; k++) { /* A B B A */
      int gen = k == 0 || k == 3;
      for (int j = 0; j < batch; j++) {
        double t0 = now_us();
        if (gen) {
          if (v) mapal_gpu_launchv(id, 0, nptr, slots, gx, gy, block, smem);
          else mapal_gpu_launch(id, slots[0], slots[1], slots[2], gx, gy, block, smem);
        } else {
          CK(cuLaunchKernel(fn, gx, gy, 1, block, 1, 1, smem, 0, args, 0));
        }
        double t = now_us() - t0;
        if (gen) g[ng++] = t;
        else {
          dr[nd++] = t;
          if (k == 1) da[na++] = t;
          else db[nb++] = t;
        }
      }
      mapal_gpu_sync();
    }
  }
  double gm = median(g, ng), dm = median(dr, nd), am = median(da, na), bm = median(db, nb);
  printf("submit_floor kernel=%s entry=%s nptr=%u grid=%ux%u block=%u smem=%u launches=%d "
         "generated_us_med=%.3f generated_us_min=%.3f direct_us_med=%.3f direct_us_min=%.3f "
         "ratio=%.3f null_direct_ratio=%.3f\n",
         name, argv[1], nptr, gx, gy, block, smem, ng, gm, g[0], dm, dr[0], gm / dm, bm / am);
  return 0;
}
