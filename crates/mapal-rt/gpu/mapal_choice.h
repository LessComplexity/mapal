/* per-site-choice B3: which candidate of a site's table runs, decided once per
 * device and remembered. Shared by the device runtime (`mapal_gpu.c`) and its
 * host emulation (`mapal_gpu_emu.c`), so the tests drive the shipped decision
 * code through a fake clock.
 *
 * A site with candidates (`mapal_gpu_site`) hands the runtime its table: one
 * launch recipe per candidate, candidate 0 the rules' plan. The pick is
 *
 *   MAPAL_GPU_PICK=rules | <index>   that candidate (index clamped); no trial,
 *                                    no wisdom write                 src=forced
 *   wisdom has the site's key        the stored winner               src=wisdom
 *   no entry, RERUN, tuning on       trial, then store               src=trial
 *   otherwise                        candidate 0                     src=rules
 *
 * MAPAL_GPU_PICK unset or `auto` is the table above; `tune` ignores the file's
 * entries (trial again, append); anything else is an error (exit 2).
 *
 * Key = FNV-1a of the table's bytes xor a hash of the module text: any codegen
 * change re-tunes. Wisdom:
 * `${MAPAL_WISDOM:-${XDG_CACHE_HOME:-~/.cache}/mapal}/wisdom-<device>.tsv`, one
 * appended line per decision (`key site n pick reps ns0 .. kernel names`, tab
 * separated); the last line of a key wins, unreadable lines are ignored.
 * `MAPAL_WISDOM=off` reads and writes nothing (candidate 0);
 * `MAPAL_GPU_TUNE=0` reads but never trials.
 *
 * The pick is decided once per table and process: later executions of the
 * site find it by the table's address (the host's constant) in a small
 * direct-mapped memo, so a decided site costs a pointer compare; the trace
 * prints one `op=site` line per decision, none per memo hit. */
#ifndef MAPAL_CHOICE_H
#define MAPAL_CHOICE_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* One candidate's launch recipe: the host's `%mapal_cand` constant
 * (crates/backends/llvm/src/func/gpu.rs `SITE_DECL`). */
struct mapal_cand {
  uint64_t id;   /* main kernel */
  uint64_t rid;  /* reduce kernel (workspace split), else 0 */
  uint64_t zero; /* bytes of ptrs[out] zeroed before the launch, else 0 */
  uint64_t ws;   /* workspace bytes: main writes it, then reduce(ws, 0, out) */
  uint32_t gx, gy, block, smem;
  uint32_t rgx, rblock; /* reduce launch: (rgx, 1, rblock, 0) */
};
_Static_assert(sizeof(struct mapal_cand) == 56, "the host's %mapal_cand");

/* `flags` bit 0: running the site again is value-safe. */
#define MAPAL_SITE_RERUN 1u
/* `flags` bit 1 (kernel-merging `zero-fold`): an earlier kernel of the same
 * stream zeroed the output, so a candidate's zero pass is skipped where the
 * site runs its settled pick alone (a trial still zeroes before each run). */
#define MAPAL_SITE_ZEROED 2u

/* How the decision runs a candidate: `run` enqueues its recipe, `timed` runs
 * it to completion and returns its nanoseconds, `warm` makes what any
 * candidate allocates (the workspace) before a trial times it. */
struct choice_dev {
  void (*run)(void *env, uint32_t i);
  uint64_t (*timed)(void *env, uint32_t i);
  void (*warm)(void *env);
  void *env;
};

#define CHOICE_AUTO (-1)
#define CHOICE_TUNE (-2)

/* MAPAL_GPU_PICK, strictly: unset / `auto` CHOICE_AUTO, `rules` 0, `tune`
 * CHOICE_TUNE, a decimal index; anything else exits (a typo must not force
 * candidate 0). */
static int choice_pick_env(void) {
  const char *p = getenv("MAPAL_GPU_PICK");
  if (!p || !strcmp(p, "auto"))
    return CHOICE_AUTO;
  if (!strcmp(p, "rules"))
    return 0;
  if (!strcmp(p, "tune"))
    return CHOICE_TUNE;
  char *end;
  unsigned long v = strtoul(p, &end, 10);
  if (*p >= '0' && *p <= '9' && !*end && v <= 0x7fffffff)
    return (int)v;
  fprintf(stderr, "mapal: MAPAL_GPU_PICK=%s: expected auto, rules, tune or a candidate index\n", p);
  exit(2);
}

static struct {
  int ready;
  int pick; /* forced index, CHOICE_AUTO, CHOICE_TUNE */
  int tune; /* trials allowed */
  int trace;
  uint64_t module;
  char path[1024]; /* wisdom file; "" = off */
  struct choice_wis {
    uint64_t key;
    uint32_t n, pick;
  } *w;
  size_t nw, cap, nfile; /* entries; the first `nfile` read from the file */
} choice;

/* Decided sites by table address; a collision only re-decides (and a re-decision
 * finds this process's wisdom entry). */
#define CHOICE_MEMO 256
static struct choice_memo {
  const struct mapal_cand *c;
  uint32_t n, pick;
} choice_memo[CHOICE_MEMO];

static uint64_t choice_fnv(const void *p, size_t n, uint64_t h) {
  const unsigned char *b = p;
  for (size_t i = 0; i < n; i++)
    h = (h ^ b[i]) * 0x100000001b3ull;
  return h;
}
#define CHOICE_FNV0 0xcbf29ce484222325ull

static void choice_remember(uint64_t key, uint32_t n, uint32_t pick) {
  if (choice.nw == choice.cap) {
    choice.cap = choice.cap ? 2 * choice.cap : 64;
    choice.w = realloc(choice.w, choice.cap * sizeof *choice.w);
    if (!choice.w)
      exit(102);
  }
  choice.w[choice.nw++] = (struct choice_wis){key, n, pick};
}

/* The wisdom file's decisions, in file order (the last of a key wins). */
static void choice_load(void) {
  FILE *f = fopen(choice.path, "r");
  if (!f)
    return;
  char line[4096];
  while (fgets(line, sizeof line, f)) {
    unsigned long long key, site;
    unsigned n, pick;
    if (sscanf(line, "%llx\t%llx\t%u\t%u\t", &key, &site, &n, &pick) == 4 && pick < n)
      choice_remember(key, n, pick);
  }
  fclose(f);
  choice.nfile = choice.nw;
}

/* `mkdir -p` of the wisdom file's directory; a failure shows at the fopen. */
static void choice_mkdirs(void) {
  char d[sizeof choice.path];
  memcpy(d, choice.path, sizeof d);
  for (char *p = d + 1; *p; p++)
    if (*p == '/') {
      *p = 0;
      mkdir(d, 0755);
      *p = '/';
    }
}

/* Once per process: the environment and the wisdom file. `dev` names the
 * device in the file name (`<name>-sm<arch>`), `module` is the module text's
 * hash; `tune` 0 never trials (the emulator without a fake clock). */
static void choice_init(const char *dev, uint64_t module, int tune, int trace) {
  if (choice.ready)
    return;
  choice.ready = 1;
  choice.module = module;
  choice.trace = trace;
  choice.pick = choice_pick_env();
  const char *t = getenv("MAPAL_GPU_TUNE");
  choice.tune = tune && !(t && !strcmp(t, "0"));
  const char *w = getenv("MAPAL_WISDOM");
  if (w && !strcmp(w, "off"))
    return;
  char dir[sizeof choice.path - 64];
  const char *xdg = getenv("XDG_CACHE_HOME"), *home = getenv("HOME");
  if (w && *w)
    snprintf(dir, sizeof dir, "%s", w);
  else if (xdg && *xdg)
    snprintf(dir, sizeof dir, "%s/mapal", xdg);
  else if (home && *home)
    snprintf(dir, sizeof dir, "%s/.cache/mapal", home);
  else
    return;
  char tag[48];
  size_t i = 0;
  for (; dev[i] && i + 1 < sizeof tag; i++) {
    char c = dev[i];
    int ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
    tag[i] = ok ? c : '-';
  }
  tag[i] = 0;
  snprintf(choice.path, sizeof choice.path, "%s/wisdom-%s.tsv", dir, tag);
  choice_load();
}

/* The site's pick is already decided this process (a memo hit): running it
 * launches its recipe once and nothing else (kernel-merging `zero-fold`). */
static int choice_known(const struct mapal_cand *c, uint32_t n) {
  const struct choice_memo *memo =
      &choice_memo[(((uintptr_t)c * 0x9e3779b97f4a7c15ull) >> 32) % CHOICE_MEMO];
  return memo->c == c && memo->n == n;
}

/* Run site `site`'s pick among its `n` candidates `c`, once. `safe` 0: the
 * runtime saw the output passed as an input too (no trial). Returns the pick. */
static uint32_t choice_site(uint64_t site, uint32_t flags, uint32_t n,
                            const struct mapal_cand *c, int safe,
                            const struct choice_dev *d) {
  struct choice_memo *memo = &choice_memo[(((uintptr_t)c * 0x9e3779b97f4a7c15ull) >> 32) % CHOICE_MEMO];
  if (memo->c == c && memo->n == n) {
    d->run(d->env, memo->pick);
    return memo->pick;
  }
  uint64_t key = 0;
  uint32_t pick = 0;
  const char *src = "rules";
  int found = 0;
  uint32_t reps = 0;
  uint64_t ns0 = 0, nsw = 0;
  if (choice.pick >= 0) {
    /* Forced: no key, no wisdom. */
    pick = (uint32_t)choice.pick < n ? (uint32_t)choice.pick : n - 1;
    src = "forced";
    found = 1;
  } else {
    key = choice_fnv(c, (size_t)n * sizeof *c, CHOICE_FNV0) ^ choice.module;
  }
  /* `tune`: only this process's decisions stand (a site in a loop trials once). */
  size_t lo = choice.pick == CHOICE_TUNE ? choice.nfile : 0;
  for (size_t i = choice.nw; !found && i-- > lo;)
    if (choice.w[i].key == key && choice.w[i].n == n) {
      pick = choice.w[i].pick;
      src = "wisdom";
      found = 1;
    }
  if (!found && (flags & MAPAL_SITE_RERUN) && safe && choice.tune && choice.path[0]) {
    /* The trial: what any candidate allocates is made first; candidate 0 once
     * sizes it (R = 5 below 2 ms, 3 below 50 ms, 2 below 500 ms, else 1); rep
     * r runs the candidates rotated by r, rep 0 a warm-up when R > 1; a
     * candidate's time is its minimum. */
    /* ponytail: the first 64 candidates are trialled; the host caps a table
     * at `--cands` (8). */
    if (d->warm)
      d->warm(d->env);
    uint64_t best[64], t0 = d->timed(d->env, 0);
    uint32_t last = 0;
    reps = t0 < 2000000 ? 5 : t0 < 50000000 ? 3 : t0 < 500000000 ? 2 : 1;
    uint32_t m = n < 64 ? n : 64;
    for (uint32_t i = 0; i < m; i++)
      best[i] = UINT64_MAX;
    for (uint32_t r = 0; r < reps; r++)
      for (uint32_t j = 0; j < m; j++) {
        uint32_t i = (j + r) % m;
        uint64_t ns = d->timed(d->env, i);
        last = i;
        if ((reps == 1 || r > 0) && ns < best[i])
          best[i] = ns;
        if (choice.trace)
          fprintf(stderr,
                  "mapal_gpu_trace op=trial site=%016llx cand=%u kernel=mapal_tile_%016llx "
                  "rep=%u ns=%llu\n",
                  (unsigned long long)site, i, (unsigned long long)c[i].id, r,
                  (unsigned long long)ns);
      }
    for (uint32_t i = 1; i < m; i++)
      if (best[i] < best[pick])
        pick = i;
    /* Candidate 0 stays unless beaten by more than max(3 %, 1 us) — in the
     * trial, then again head to head: 3 alternating pairs (0, w), (w, 0),
     * (0, w), their minima merged with the trial's (one disturbed trial is
     * not stored forever). */
#define CHOICE_MARGIN(t) ((t) * 3 / 100 > 1000 ? (t) * 3 / 100 : 1000)
    if (pick && !(best[pick] + CHOICE_MARGIN(best[0]) < best[0]))
      pick = 0;
    for (uint32_t r = 0; pick && r < 6; r++) {
      uint32_t i = ((r / 2 + r) & 1) ? pick : 0;
      uint64_t ns = d->timed(d->env, i);
      last = i;
      if (ns < best[i])
        best[i] = ns;
      if (choice.trace)
        fprintf(stderr,
                "mapal_gpu_trace op=trial site=%016llx cand=%u kernel=mapal_tile_%016llx "
                "rep=%u ns=%llu confirm=1\n",
                (unsigned long long)site, i, (unsigned long long)c[i].id, reps + r / 2,
                (unsigned long long)ns);
    }
    if (pick && !(best[pick] + CHOICE_MARGIN(best[0]) < best[0]))
      pick = 0;
    ns0 = best[0];
    nsw = best[pick];
    /* The output is the winner's. */
    if (last != pick)
      d->run(d->env, pick);
    choice_remember(key, n, pick);
    choice_mkdirs();
    FILE *f = fopen(choice.path, "a");
    if (f) {
      fprintf(f, "%016llx\t%016llx\t%u\t%u\t%u", (unsigned long long)key,
              (unsigned long long)site, n, pick, reps);
      for (uint32_t i = 0; i < m; i++)
        fprintf(f, "\t%llu", (unsigned long long)best[i]);
      for (uint32_t i = 0; i < m; i++)
        fprintf(f, "\tmapal_tile_%016llx", (unsigned long long)c[i].id);
      fputc('\n', f);
      fclose(f);
    }
    src = "trial";
  } else {
    d->run(d->env, pick);
  }
  /* An unsafe execution's candidate 0 is not a decision: a later one may trial. */
  if (safe || found)
    *memo = (struct choice_memo){c, n, pick};
  if (choice.trace) {
    fprintf(stderr, "mapal_gpu_trace op=site site=%016llx key=%016llx n=%u pick=%u src=%s reps=%u",
            (unsigned long long)site, (unsigned long long)key, n, pick, src, reps);
    if (reps)
      fprintf(stderr, " ns0=%llu nsw=%llu", (unsigned long long)ns0, (unsigned long long)nsw);
    fputc('\n', stderr);
  }
  return pick;
}

#endif
