// Can a two-level tiled A pack reach the padded-reversed number WITHOUT padding?
// plan-s51 H2 follow-up #2.
//
// WHY NOT "BLOCKED REVERSED". Blocking the outer loop of a two-level nest and
// keeping the same inner index is the identical iteration order:
//
//   for pk0 step B { for pk in pk0..pk0+B { for pi } }  ==  for pk { for pi }
//
// so it is a strict no-op. The reversed order already HAS a 32-line read working
// set: for a fixed `pk` the 32 row-lines it touches serve the next 32 consecutive
// `pk`, because consecutive `pk` are 4 bytes apart in the same row. Nothing is
// left to block on that axis.
//
// WHAT IS ACTUALLY UNTRIED is blocking the ROW axis. Both shipped orders keep all
// 32 rows live at once; §5f showed the reversed order loses 2.0-2.6x to something
// about those 32 power-of-two-strided addresses that L1 set COUNT does not
// explain. If the cause is aliasing among them, touching only R at a time should
// dodge it -- and unlike padding, R is entirely the emitter's to choose.
//
//   for pk0 in 0..k step B:
//     for pi0 in 0..32 step R:
//       <R x B tile, inner order chosen by `inner`>
//
//   read set  = R lines   (rows pi0..pi0+R, each consumed B deep)
//   write set = B lines   (ap lines pk0..pk0+B, each written R at a time)
//
// R=32, B=32, inner=ROW is exactly what H2 shipped, so the sweep contains its own
// baseline and the comparison is like-for-like through ONE loop body.
//
// FAIRNESS. R and B are RUNTIME arguments, so a single loop body serves every
// configuration and no cell can win by being specialized (the S43/S44 rule: one
// loop body, or a size-dependent transformation is possible by construction).
// `blocked_spec` is the shipped compile-time form, reported alongside so the
// generic body's own overhead is visible rather than assumed away.
//
// TARGETS TO BEAT (benches/sme/packsets.c, same machine):
//   N=2048   H2 blocked 0.782 ms   padded-reversed 0.682 ms
//   N=4096   H2 blocked 5.168 ms   padded-reversed 3.167 ms
//
// Build: clang -O2 -march=armv8-a benches/sme/packtile.c -o packtile
// Run:   benches/perflock.sh ./packtile [N ...]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

#define PANEL_ROWS 32
#define PAD 32

enum { INNER_ROW = 0, INNER_K = 1 };

// The generic two-level tile. R and B are runtime, so ONE body serves all cells.
static void pack_tile(const float *a, float *ap, long lda, long i0, long k,
                      long R, long B, int inner) {
    for (long pk0 = 0; pk0 < k; pk0 += B) {
        long khi = pk0 + B < k ? pk0 + B : k;
        for (long pi0 = 0; pi0 < PANEL_ROWS; pi0 += R) {
            long ihi = pi0 + R < PANEL_ROWS ? pi0 + R : PANEL_ROWS;
            if (inner == INNER_ROW) {
                for (long pi = pi0; pi < ihi; pi++)
                    for (long pk = pk0; pk < khi; pk++)
                        ap[pk * PANEL_ROWS + pi] = a[(i0 + pi) * lda + pk];
            } else {
                for (long pk = pk0; pk < khi; pk++)
                    for (long pi = pi0; pi < ihi; pi++)
                        ap[pk * PANEL_ROWS + pi] = a[(i0 + pi) * lda + pk];
            }
        }
    }
}

// The shipped H2 form, compile-time 32x32, row-inner — so the generic body's own
// overhead is measured rather than assumed to be zero.
static void blocked_spec(const float *a, float *ap, long lda, long i0, long k) {
    for (long pk0 = 0; pk0 < k; pk0 += 32) {
        long hi = pk0 + 32 < k ? pk0 + 32 : k;
        for (long pi = 0; pi < PANEL_ROWS; pi++)
            for (long pk = pk0; pk < hi; pk++)
                ap[pk * PANEL_ROWS + pi] = a[(i0 + pi) * lda + pk];
    }
}

static void pack_k_outer(const float *a, float *ap, long lda, long i0, long k) {
    for (long pk = 0; pk < k; pk++)
        for (long pi = 0; pi < PANEL_ROWS; pi++)
            ap[pk * PANEL_ROWS + pi] = a[(i0 + pi) * lda + pk];
}

static double checksum(const float *ap, long n) {
    double s = 0;
    for (long t = 0; t < n; t++) s += ap[t] * (double)((t & 7) + 1);
    return s;
}

static double sweep_tile(const float *a, float *ap, long lda, long N, int reps,
                         long R, long B, int inner) {
    double best = 1e18;
    for (int r = 0; r < reps; r++) {
        double t0 = now_ms();
        for (long i0 = 0; i0 < N; i0 += PANEL_ROWS)
            pack_tile(a, ap, lda, i0, N, R, B, inner);
        double d = now_ms() - t0;
        if (d < best) best = d;
    }
    return best;
}

int main(int argc, char **argv) {
    int reps = getenv("REPS") ? atoi(getenv("REPS")) : 7;
    long sizes[8]; int ns = 0;
    for (int i = 1; i < argc && ns < 8; i++) sizes[ns++] = atol(argv[i]);
    if (!ns) { long d[] = {2048, 4096}; for (int i = 0; i < 2; i++) sizes[ns++] = d[i]; }

    const long RS[] = {4, 8, 16, 32};
    const long BS[] = {32, 64, 128};

    printf("plan-s51 H2 follow-up #2 -- can a two-level tile reach padded-reversed\n");
    printf("without padding? best of %d, 1 thread, no SME. R and B are RUNTIME:\n", reps);
    printf("one loop body serves every cell.\n");

    for (int s = 0; s < ns; s++) {
        long N = sizes[s], lpad = N + PAD;
        float *a    = aligned_alloc(4096, (size_t)N * N * 4);
        float *apad = aligned_alloc(4096, (size_t)N * lpad * 4);
        float *ap   = aligned_alloc(4096, (size_t)PANEL_ROWS * N * 4);
        if (!a || !apad || !ap) { fprintf(stderr, "alloc failed N=%ld\n", N); return 1; }
        for (long i = 0; i < N; i++)
            for (long j = 0; j < N; j++) {
                float v = (float)(((i * N + j) * 7) % 13) * 0.01f + 1.0f;
                a[i * N + j] = v;
                apad[i * lpad + j] = v;
            }
        long apn = PANEL_ROWS * N;

        // ---- value gate FIRST: every configuration must write identical bytes.
        memset(ap, 0, (size_t)apn * 4);
        for (long i0 = 0; i0 < N; i0 += PANEL_ROWS) blocked_spec(a, ap, N, i0, N);
        double ref = checksum(ap, apn);
        for (int ri = 0; ri < 4; ri++)
            for (int bi = 0; bi < 3; bi++)
                for (int inner = 0; inner < 2; inner++) {
                    memset(ap, 0, (size_t)apn * 4);
                    for (long i0 = 0; i0 < N; i0 += PANEL_ROWS)
                        pack_tile(a, ap, N, i0, N, RS[ri], BS[bi], inner);
                    if (checksum(ap, apn) != ref) {
                        fprintf(stderr, "VALUE GATE FAILED N=%ld R=%ld B=%ld inner=%d\n",
                                N, RS[ri], BS[bi], inner);
                        return 1;
                    }
                }

        // ---- references
        double spec = 1e18, ko = 1e18, kop = 1e18, nul = 1e18;
        for (int r = 0; r < reps; r++) {
            double t0 = now_ms();
            for (long i0 = 0; i0 < N; i0 += PANEL_ROWS) blocked_spec(a, ap, N, i0, N);
            double d = now_ms() - t0; if (d < spec) spec = d;
            t0 = now_ms();
            for (long i0 = 0; i0 < N; i0 += PANEL_ROWS) blocked_spec(a, ap, N, i0, N);
            d = now_ms() - t0; if (d < nul) nul = d;
            t0 = now_ms();
            for (long i0 = 0; i0 < N; i0 += PANEL_ROWS) pack_k_outer(a, ap, N, i0, N);
            d = now_ms() - t0; if (d < ko) ko = d;
            t0 = now_ms();
            for (long i0 = 0; i0 < N; i0 += PANEL_ROWS) pack_k_outer(apad, ap, lpad, i0, N);
            d = now_ms() - t0; if (d < kop) kop = d;
        }

        printf("\n=== N=%ld ===\n", N);
        printf("  reference   shipped H2 (32x32 row-inner, specialized) %8.3f ms   drift %.3fx\n",
               spec, nul / spec);
        printf("  reference   reversed, unpadded                       %8.3f ms\n", ko);
        printf("  reference   reversed, PADDED  <- the number to beat   %8.3f ms\n", kop);
        printf("\n  %5s %5s   %10s %10s   %9s %9s\n",
               "R", "B", "row-inner", "k-inner", "vs H2", "vs padded");
        printf("  ---------------------------------------------------------------\n");
        for (int ri = 0; ri < 4; ri++)
            for (int bi = 0; bi < 3; bi++) {
                double t_row = sweep_tile(a, ap, N, N, reps, RS[ri], BS[bi], INNER_ROW);
                double t_k   = sweep_tile(a, ap, N, N, reps, RS[ri], BS[bi], INNER_K);
                double best = t_row < t_k ? t_row : t_k;
                printf("  %5ld %5ld   %10.3f %10.3f   %8.3fx %8.3fx\n",
                       RS[ri], BS[bi], t_row, t_k, spec / best, kop / best);
            }
        free(a); free(apad); free(ap);
    }
    printf("\n  'vs H2' > 1 means the tile BEATS what shipped. 'vs padded' > 1 means it\n");
    printf("  reaches the padded-reversed number WITHOUT padding, which is the point.\n");
    printf("  R=32 B=32 row-inner is H2's own shape through the generic body: the gap\n");
    printf("  between it and the specialized reference is this body's own overhead.\n");
    return 0;
}
