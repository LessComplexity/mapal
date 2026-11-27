/* The stream slots, the zero-fill kernel and the warm fills.
 * Included by `mapal_gpu.c` at this place in its translation unit;
 * not compiled on its own. */

static unsigned slot_of(uint32_t s) {
  return s ? 1 + (s - 1) % MAPAL_GPU_STREAMS : 0;
}
/* gaps-disasm C1 `zero-host`: an output's zero pass is the runtime's own zero
 * kernel — one `cuLaunchKernel` on the host — never the driver's memset API,
 * whose host call cost a one-call program's atomic site 3.7–5.9 us inside its
 * clock against 2.5–3.9 us for a launch (RTX 5080, `gapfix-launch-combine.md`;
 * `gaps-disasm.md` C1). Grid-stride 16-byte stores, then the bytes past the
 * last whole vector; every buffer a site zeroes starts 16-byte aligned
 * (`cuMemAlloc`, the arena's `ARENA_ALIGN`; a misaligned one takes the
 * memset). Loaded from its PTX text by the init of a module whose launches
 * zero (`ZERO_KERNEL`; JIT once, the driver's cache after). Without the flag
 * (`--gpu-off=zero-host`, or a module that zeroes nothing): the memset API. */
static const char zero_ptx[] =
    ".version 6.0\n.target sm_50\n.address_size 64\n"
    ".visible .entry mapal_rt_zero(.param .u64 zp, .param .u64 zn) {\n"
    "  .reg .pred %p<3>;\n  .reg .b16 %h0;\n  .reg .b32 %r<6>;\n  .reg .b64 %d<12>;\n"
    "  ld.param.u64 %d1, [zp];\n  ld.param.u64 %d2, [zn];\n"
    "  cvta.to.global.u64 %d1, %d1;\n"
    "  mov.u32 %r1, %ctaid.x;\n  mov.u32 %r2, %ntid.x;\n  mov.u32 %r3, %tid.x;\n"
    "  mov.u32 %r4, %nctaid.x;\n  mov.u32 %r5, 0;\n  mov.u16 %h0, 0;\n"
    "  mul.wide.u32 %d3, %r1, %r2;\n  cvt.u64.u32 %d4, %r3;\n  add.u64 %d3, %d3, %d4;\n"
    "  mul.wide.u32 %d5, %r4, %r2;\n  shr.u64 %d6, %d2, 4;\n  mov.u64 %d7, %d3;\n"
    "L:\n  setp.ge.u64 %p1, %d7, %d6;\n  @%p1 bra T;\n"
    "  shl.b64 %d8, %d7, 4;\n  add.u64 %d8, %d1, %d8;\n"
    "  st.global.v4.u32 [%d8], {%r5, %r5, %r5, %r5};\n"
    "  add.u64 %d7, %d7, %d5;\n  bra L;\n"
    "T:\n  and.b64 %d9, %d2, 15;\n  setp.ge.u64 %p2, %d3, %d9;\n  @%p2 bra D;\n"
    "  shl.b64 %d10, %d6, 4;\n  add.u64 %d10, %d10, %d3;\n  add.u64 %d10, %d1, %d10;\n"
    "  st.global.u8 [%d10], %h0;\n"
    "D:\n  ret;\n}\n";
static CUfunction zero_fn; /* 0: the memset API */
static uint32_t zero_grid_max;
#define ZERO_THREADS 256
static void zero_load(void) {
  CUmodule zm;
  CUdevice dev;
  int sms = 0;
  if (!(flags & ZERO_KERNEL) || cuModuleLoadData(&zm, zero_ptx) != CUDA_SUCCESS ||
      cuModuleGetFunction(&zero_fn, zm, "mapal_rt_zero") != CUDA_SUCCESS) {
    zero_fn = 0;
    return;
  }
  CK(cuCtxGetDevice(&dev));
  CK(cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
  /* Enough 256-thread blocks to fill every SM's threads (eight a 2048-thread
   * SM), striding past that: a 64 MiB zero is 16 vectors a thread. */
  zero_grid_max = 8 * (uint32_t)(sms > 0 ? sms : 1);
}
/* Zero `bytes` at `d` on stream slot `k`: the zero kernel, or the memset. */
static void zero_on(unsigned k, CUdeviceptr d, uint64_t bytes) {
  if (!bytes)
    return;
  if (zero_fn && !(d & 15)) {
    uint64_t w = bytes / 16, b = (w + ZERO_THREADS - 1) / ZERO_THREADS;
    unsigned grid = (unsigned)(b < zero_grid_max ? (b ? b : 1) : zero_grid_max);
    void *args[] = {&d, &bytes};
    CK(cuLaunchKernel(zero_fn, grid, 1, 1, ZERO_THREADS, 1, 1, 0, streams[k], args, 0));
    last_kernel[k] = 1;
  } else {
    CK(cuMemsetD8Async(d, 0, bytes, streams[k]));
    last_kernel[k] = 0;
  }
}
/* `warm-zero`: a small and a large fill of the scratch buffer. */
static void warm_fills(void) {
  zero_on(0, warm_buf, 4);
  zero_on(0, warm_buf, WARM_ZERO_BYTES);
}

static CUmodule module;
static CUcontext ctx;
/* per-site-choice: the module text's hash (a wisdom key's half). */
static uint64_t module_hash;

