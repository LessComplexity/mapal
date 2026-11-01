// plan-s51 H2 -- what does the A pack actually cost, in the order the emitter
// actually emits, and can a traversal change fix it?
//
// READ THIS FIRST, IT CORRECTS A STALE INSTRUMENT.
// `benches/sme/packcost.c` says it "replicates the EMITTED nest exactly
// (func/sme.rs::emit_tiled_map_sme)" and runs
//
//     for pk in 0..k:  for pi in 0..32:  ap[pk*32+pi] = a[(i0+pi)*N + pk]
//
// k OUTER, row INNER. **The emitter stopped emitting that in S42**, which
// reordered the pack row-outer (`func/sme.rs:385`, "ROW OUTER, k INNER -- and
// the order is the whole point"; recorded in the S42 roll-up as "the A pack
// reordered row-outer, scalar float loads 51 -> 5"). packcost.c's comment was
// not updated, so its 8.46 ms is the cost of the PRE-S42 pack, and anything
// pricing today's pack from it -- as plan-s51 §4's first draft did -- is
// pricing code that is not in the binary.
//
// WHAT SHIPS TODAY, verbatim from the emitter:
//
//     for pi in 0..32:  for pk in 0..kc:  ap[pk*32 + pi] = a[base + ci*(i0+pi) + ck*pk]
//
// with `ck == 1` (the matmul shape this rung is gated on). So:
//
//   READS  are contiguous along k -- one live row pointer, one line fully
//          consumed before moving on. This side is already right, and it is
//          what S42 bought.
//   WRITES are `ap[pk*32 + pi]`: consecutive pk are 32 floats = 128 B apart, so
//          **every store touches a different line and writes 4 of its 128 B.**
//          The next pi revisits those same lines at +4 B.
//
// So the line waste did not disappear in S42, it MOVED to the store side, and
// whether it costs anything depends on one thing: does `ap` stay in L1D between
// two consecutive rows? `ap` is `32 * k * 4` bytes against a 128 KB L1D:
//
//   k=512  ->  64 KB  -> fits            -> the 32 revisits are free
//   k=1024 -> 128 KB  -> exactly L1D     -> the boundary
//   k=2048 -> 256 KB  -> 2x over         -> every line evicted before reuse
//   k=4096 -> 512 KB  -> 4x over         -> 32x store-side line amplification
//
// PREDICTION, written before the run: `ctl` costs ~the same per element up to
// k=1024 and degrades beyond it; `blk` is flat across all k; they are equal at
// k<=1024 and separate at k>=2048. If `blk` wins at k=512 the model is wrong
// again and this file says so rather than being re-fitted a second time.
//
// ARMS (all move the same N^2 elements and are CHECKSUM-GATED before timing):
//   ctl   row-outer / k-inner        -- what the emitter emits today
//   null  ctl again                  -- this run's own drift
//   old   k-outer / row-inner        -- the pre-S42 order packcost.c measured,
//                                       so the reorder's real worth is a number
//   blk   32x32 tiled                -- both sides local: 32 read lines and 32
//                                       write lines live, 4 KB each
//
// Build: clang -O2 -march=armv8-a benches/sme/packconflict.c -o packconflict
// Run:   benches/perflock.sh ./packconflict [N ...]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

#define PANEL_ROWS 32
#define KB 32   // one 128 B line = 32 floats

// WHAT SHIPS: row outer, k inner. Reads walk a row contiguously; stores stride
// PANEL_ROWS floats.
static void pack_row_outer(const float *a, float *ap, long lda, long i0, long k) {
    for (long pi = 0; pi < PANEL_ROWS; pi++)
        for (long pk = 0; pk < k; pk++)
            ap[pk * PANEL_ROWS + pi] = a[(i0 + pi) * lda + pk];
}

// PRE-S42, and what packcost.c still measures: k outer, row inner.
static void pack_k_outer(const float *a, float *ap, long lda, long i0, long k) {
    for (long pk = 0; pk < k; pk++)
        for (long pi = 0; pi < PANEL_ROWS; pi++)
            ap[pk * PANEL_ROWS + pi] = a[(i0 + pi) * lda + pk];
}

// 32x32 tiled: inside a block both the 32 read lines and the 32 write lines are
// live and each is consumed whole. Independent of k, so independent of whether
// `ap` fits L1.
static void pack_blocked(const float *a, float *ap, long lda, long i0, long k) {
    for (long pk0 = 0; pk0 < k; pk0 += KB) {
        long hi = pk0 + KB < k ? pk0 + KB : k;
        for (long pi = 0; pi < PANEL_ROWS; pi++)
            for (long pk = pk0; pk < hi; pk++)
                ap[pk * PANEL_ROWS + pi] = a[(i0 + pi) * lda + pk];
    }
}

typedef void (*packfn)(const float *, float *, long, long, long);

static double sweep(packfn fn, const float *a, float *ap, long N, int reps) {
    double best = 1e18;
    for (int r = 0; r < reps; r++) {
        double t0 = now_ms();
        for (long i0 = 0; i0 < N; i0 += PANEL_ROWS) fn(a, ap, N, i0, N);
        double d = now_ms() - t0;
        if (d < best) best = d;
    }
    return best;
}

static double checksum(const float *ap, long n) {
    double s = 0;
    for (long t = 0; t < n; t++) s += ap[t] * (double)((t & 7) + 1);
    return s;
}

int main(int argc, char **argv) {
    int reps = getenv("REPS") ? atoi(getenv("REPS")) : 7;
    long sizes[16];
    int ns = 0;
    for (int i = 1; i < argc && ns < 16; i++) sizes[ns++] = atol(argv[i]);
    if (!ns) { long d[] = {128, 256, 512, 1024, 2048, 4096}; for (int i = 0; i < 6; i++) sizes[ns++] = d[i]; }

    printf("plan-s51 H2 -- the A pack in the order the emitter EMITS. 1 thread, no SME.\n");
    printf("best of %d reps per arm. L1D 128 KB, lines 128 B, panel %d rows.\n", reps, PANEL_ROWS);
    printf("NOTE: packcost.c measures the PRE-S42 nest (`old` here), not what ships.\n\n");
    printf("%6s %9s %9s %9s %9s %9s  %8s %8s %8s %9s\n",
           "N", "ap KB", "ctl ms", "null ms", "old ms", "blk ms",
           "blk/ctl", "old/ctl", "drift", "ctl ns/el");
    printf("-----------------------------------------------------------------------------------------------\n");

    for (int s = 0; s < ns; s++) {
        long N = sizes[s];
        float *a = aligned_alloc(4096, (size_t)N * N * 4);
        float *ap = aligned_alloc(4096, (size_t)PANEL_ROWS * N * 4);
        if (!a || !ap) { fprintf(stderr, "alloc failed at N=%ld\n", N); return 1; }
        for (long t = 0; t < N * N; t++) a[t] = (float)((t * 7) % 13) * 0.01f + 1.0f;

        // ---- value gate: every arm writes identical bytes. Checked first.
        long apn = PANEL_ROWS * N;
        memset(ap, 0, (size_t)apn * 4);
        for (long i0 = 0; i0 < N; i0 += PANEL_ROWS) pack_row_outer(a, ap, N, i0, N);
        double ref = checksum(ap, apn);
        packfn others[] = {pack_k_outer, pack_blocked};
        const char *names[] = {"old", "blk"};
        for (int i = 0; i < 2; i++) {
            memset(ap, 0, (size_t)apn * 4);
            for (long i0 = 0; i0 < N; i0 += PANEL_ROWS) others[i](a, ap, N, i0, N);
            if (checksum(ap, apn) != ref) {
                fprintf(stderr, "VALUE GATE FAILED at N=%ld arm %s\n", N, names[i]);
                return 1;
            }
        }

        double ctl = sweep(pack_row_outer, a, ap, N, reps);
        double nul = sweep(pack_row_outer, a, ap, N, reps);
        double old = sweep(pack_k_outer,   a, ap, N, reps);
        double blk = sweep(pack_blocked,   a, ap, N, reps);

        printf("%6ld %9.0f %9.3f %9.3f %9.3f %9.3f  %7.3fx %7.3fx %7.3fx %9.3f\n",
               N, PANEL_ROWS * N * 4 / 1024.0, ctl, nul, old, blk,
               ctl / blk, ctl / old, nul / ctl, ctl * 1e6 / (double)(N * N));

        free(a); free(ap);
    }
    printf("\n  blk/ctl > 1 means blocking WINS over what ships.\n");
    printf("  old/ctl < 1 would mean S42's reorder made it WORSE; > 1 is what it bought.\n");
    printf("  drift is ctl measured twice: any effect not larger than it is not an effect.\n");
    return 0;
}
