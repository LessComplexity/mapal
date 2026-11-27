/* The emulated host entries: totals, trap word, memory, copies and symbol lookup.
 * Included by `mapal_gpu_emu.c` at this place in its translation unit;
 * not compiled on its own. */

static void total(void) {
  fprintf(stderr, "emu_total h2d=%llu d2h=%llu launches=%llu bytes_in=%llu bytes_out=%llu zeros=%llu\n", n_h2d,
          n_d2h, n_launch, b_in, b_out, n_zero);
}

/* bf16-class `device-trap`: a cell that checks an unproven index records a
 * trap into the module's word `mapal_dev_trap` (i64::MAX: none; else
 * `topo << 8 | kind`). Its launch arms the next host checkpoint (a print, the
 * clock, a trap, the exit: `mapal_rt_checkpoint_hook`), which waits for the
 * device and reads the word — a recorded trap is raised there, the oracle's
 * outcome. Unarmed, a checkpoint costs nothing. */
extern void mapal_trap(uint32_t kind) __attribute__((noreturn));
extern void mapal_rt_checkpoint_hook(void (*f)(void));
static int trap_armed, trap_hooked;
static long long trap_word(void);
static void trap_poll(void) {
  if (!trap_armed)
    return;
  trap_armed = 0;
  long long v = trap_word();
  if (v != 0x7fffffffffffffffLL)
    mapal_trap((uint32_t)(v & 0xff));
}
static void trap_exit(void) {
  if (!trap_armed)
    return;
  trap_armed = 0;
  long long v = trap_word();
  if (v != 0x7fffffffffffffffLL) {
    fflush(stdout);
    fprintf(stderr, "mapal trap: %s\n", (v & 0xff) == 0 ? "div_zero" : "index_oob");
    _exit(101);
  }
}
void mapal_gpu_trap_arm(void) {
  trap_armed = 1;
  if (!trap_hooked) {
    trap_hooked = 1;
    mapal_rt_checkpoint_hook(trap_poll);
    atexit(trap_exit);
  }
}

static long long trap_word(void) {
  /* The emulated device unit's own global (no address spaces here). */
  long long *w = dlsym(RTLD_DEFAULT, "mapal_dev_trap");
  return w ? __atomic_load_n(w, __ATOMIC_SEQ_CST) : 0x7fffffffffffffffLL;
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
/* `device-trap`: a copy-out is a checkpoint (`mapal_gpu.c`). */
void mapal_gpu_d2h(void *h, uint64_t d, uint64_t off, uint64_t bytes) {
  memcpy(h, (void *)(d + off), bytes);
  n_d2h++;
  b_out += bytes;
  trap_poll();
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
void mapal_gpu_launchv(uint64_t id, uint32_t stream, uint32_t nptr, const uint64_t *ptrs, uint32_t gx, uint32_t gy,
                       uint32_t block, uint32_t smem);

