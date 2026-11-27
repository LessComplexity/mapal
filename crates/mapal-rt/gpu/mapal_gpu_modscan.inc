/* Scanning the PTX module: tensor-map records, the kernel index and preload.
 * Included by `mapal_gpu.c` at this place in its translation unit;
 * not compiled on its own. */

/* integration: the kernels whose recipe global `<kernel>_tmap` the loaded PTX
 * declares (hopper-wgmma-tma), read once from the image at load, off any timed
 * path. A launch of any other kernel never asks the driver for a recipe: the
 * failing `cuModuleGetGlobal` per kernel cost ~10 us each inside a program's
 * first timed iteration (RTX 5080, ABBA against the runtime before it: mlp_512
 * 0.040 -> 0.060 ms, lsm_32_8192 0.023 -> 0.049). -1: more names than the table
 * holds (or one too long): every kernel is asked, as before. */
#define TMAP_NAMES 64
#define TMAP_MAX 4
#define TMAP_WORDS 20
#define TMAP_REC_WORDS (1 + TMAP_MAX * TMAP_WORDS)
static char tmap_names[TMAP_NAMES][48];
static int n_tmap_names;
/* hopper-launch: each listed recipe's words, parsed from the global's
 * initializer in the module text at load (`tmap_ok`), so a launch never reads
 * it from the device: the first launch's `cuModuleGetGlobal` + a synchronous
 * pageable `cuMemcpyDtoH` cost 23–30 us on the RTX 5080 inside the program's
 * clock (`hopper-launch.md`). A recipe the text does not spell as parsed here
 * is read from the device at its first launch, as before. */
static uint64_t tmap_rec[TMAP_NAMES][TMAP_REC_WORDS];
static int tmap_ok[TMAP_NAMES];
static int ident_char(char c) {
  return c == '_' || (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') ||
         (c >= 'A' && c <= 'Z');
}
/* The initializer after a global's name (`[N] = {v, v, …}`), its element
 * width read from the declaration's type on the same line (`.b8` / `.u8` or
 * `.b64` / `.u64`: NVPTX spells an `[n x i64]` constant as its bytes), into
 * `rec` as little-endian 64-bit words. 0 where the text is not that shape. */
static int tmap_parse(const char *line, const char *after, uint64_t *rec) {
  int w = 0;
  for (const char *q = line; q < after && !w; q++)
    if (q[0] == '.' && (q[1] == 'b' || q[1] == 'u'))
      w = q[2] == '8' && q[3] == ' '                     ? 1
          : q[2] == '6' && q[3] == '4' && q[4] == ' ' ? 8
                                                        : 0;
  if (!w || *after != '[')
    return 0;
  char *e;
  unsigned long long count = strtoull(after + 1, &e, 10);
  while (*e == ']' || *e == ' ' || *e == '=' || *e == '\t')
    e++;
  if (*e != '{' || !count || count * (unsigned)w > TMAP_REC_WORDS * 8ull)
    return 0;
  memset(rec, 0, TMAP_REC_WORDS * sizeof *rec);
  const char *v = e + 1;
  for (unsigned long long i = 0; i < count; i++) {
    while (*v == ' ' || *v == ',' || *v == '\n' || *v == '\t')
      v++;
    if (*v == '}')
      break; /* NVPTX leaves the trailing zeros out */
    char *ve;
    unsigned long long x = strtoull(v, &ve, 10);
    if (ve == v)
      return 0;
    v = ve;
    if (w == 8)
      rec[i] = x;
    else
      rec[i / 8] |= (uint64_t)(x & 0xff) << (8 * (i % 8));
  }
  return rec[0] <= TMAP_MAX;
}
static void tmap_scan(const char *ptx) {
  for (const char *p = strstr(ptx, "_tmap"); p; p = strstr(p + 5, "_tmap")) {
    const char *s = p;
    while (s > ptx && ident_char(s[-1]))
      s--;
    size_t n = (size_t)(p - s);
    if (!n || n >= sizeof tmap_names[0]) {
      n_tmap_names = -1;
      return;
    }
    int seen = -1;
    for (int i = 0; i < n_tmap_names && seen < 0; i++)
      if (!strncmp(tmap_names[i], s, n) && !tmap_names[i][n])
        seen = i;
    if (seen < 0) {
      if (n_tmap_names == TMAP_NAMES) {
        n_tmap_names = -1;
        return;
      }
      seen = n_tmap_names++;
      memcpy(tmap_names[seen], s, n);
      tmap_names[seen][n] = 0;
    }
    if (!tmap_ok[seen]) {
      const char *line = s;
      while (line > ptx && line[-1] != '\n')
        line--;
      tmap_ok[seen] = tmap_parse(line, p + 5, tmap_rec[seen]);
      tot.n_recipe += (unsigned long long)tmap_ok[seen];
    }
  }
}
/* The listed name's index (-2: not listed; -1: every kernel is asked). */
static int tmap_index(const char *name) {
  if (n_tmap_names < 0)
    return -1;
  for (int i = 0; i < n_tmap_names; i++)
    if (!strcmp(tmap_names[i], name))
      return i;
  return -2;
}

/* `preload`: every `.entry` of the module, resolved at load. `smem` is the
 * dynamic shared-memory limit already set on it (the attribute is set once).
 * gaps-disasm C1 `load-dispatch`: an entry named `<prefix>_<16 hex digits>`
 * (every kernel the emitter names by id) is also found by its id — one probe
 * of `fn_hash` — so a launch neither formats nor compares names (a linear
 * `strcmp` scan, 0.4–0.6 us a launch on the RTX 5080); any other name is
 * scanned as before. */
static struct fn_entry {
  char name[40];
  CUfunction fn;
  uint32_t smem;
  uint32_t plen; /* the prefix's length before `_<id>`; 0: no id */
  uint64_t id;
} *fns;
static size_t n_fns;
static uint32_t *fn_hash; /* index + 1 into `fns`, 0 empty */
static size_t fn_hash_mask;
static size_t fn_slot(uint64_t id) {
  return (size_t)((id * 0x9e3779b97f4a7c15ull) >> 40) & fn_hash_mask;
}
static void fn_index(void) {
  size_t cap = 64;
  while (cap < 2 * n_fns)
    cap *= 2;
  fn_hash = calloc(cap, sizeof *fn_hash);
  if (!fn_hash)
    exit(102);
  fn_hash_mask = cap - 1;
  for (size_t i = 0; i < n_fns; i++) {
    struct fn_entry *e = &fns[i];
    size_t len = strlen(e->name);
    const char *u = strrchr(e->name, '_');
    char *end;
    if (!u || (size_t)(e->name + len - u) != 17)
      continue;
    e->id = strtoull(u + 1, &end, 16);
    if (*end)
      continue;
    e->plen = (uint32_t)(u - e->name);
    size_t h = fn_slot(e->id);
    while (fn_hash[h])
      h = (h + 1) & fn_hash_mask;
    fn_hash[h] = (uint32_t)(i + 1);
  }
}
static struct fn_entry *fn_find(const char *prefix, uint64_t id) {
  if (!fn_hash)
    return 0;
  size_t plen = strlen(prefix);
  for (size_t h = fn_slot(id); fn_hash[h]; h = (h + 1) & fn_hash_mask) {
    struct fn_entry *e = &fns[fn_hash[h] - 1];
    if (e->id == id && e->plen == plen && !memcmp(e->name, prefix, plen))
      return e;
  }
  return 0;
}
static void preload(const char *ptx) {
  size_t cap = 0;
  for (const char *p = strstr(ptx, ".entry "); p; p = strstr(p, ".entry ")) {
    p += 7;
    size_t n = strcspn(p, "( \t\r\n");
    if (!n || n >= sizeof fns->name)
      continue;
    if (n_fns == cap) {
      cap = cap ? 2 * cap : 32;
      fns = realloc(fns, cap * sizeof *fns);
      if (!fns)
        exit(102);
    }
    struct fn_entry *e = &fns[n_fns++];
    memcpy(e->name, p, n);
    e->name[n] = 0;
    e->smem = 48 * 1024;
    e->plen = 0;
    e->id = 0;
    CK(cuModuleGetFunction(&e->fn, module, e->name));
  }
  tot.n_preload = n_fns;
  fn_index();
}

