// Is the REVERSED (pre-S42, k-outer) A pack limited by L1 SET CONFLICT on its
// reads? plan-s51 H2 follow-up.
//
// THE CLAIM UNDER TEST. With `pk` outer and `pi` inner, the pack's writes are
// perfect -- ap[pk*32 + 0..31] is one whole 128 B line, written completely
// before moving on. Its READS are 32 addresses `A[(i0+pi)*lda + pk]`, `lda*4`
// bytes apart, and each takes 4 bytes of a 128 B line. Those 32 lines have to
// survive the whole `pk` loop to be reused, and 32 lines is only 4 KB, so
// CAPACITY is not the problem. The suspicion is that they cannot survive
// because they land on too few L1 SETS:
//
//   sets apart = lda*4 / 128;  reachable = min(32, 128 / gcd(sets_apart, 128))
//
//   k=512   -> 16 apart ->  8 sets -> 32/8  =  4 per set  <= 8 ways -> fits
//   k=1024  -> 32 apart ->  4 sets -> 32/4  =  8 per set  == 8 ways -> borderline
//   k=2048  -> 64 apart ->  2 sets -> 32/2  = 16 per set  >  8 ways -> THRASH
//   k=4096  ->128 apart ->  1 set  -> 32/1  = 32 per set  >  8 ways -> SEVERE
//
// THE INSTRUMENT IS S44's, and the reason is that it changes NOTHING about the
// traversal. Padding the row stride by 32 floats (128 B) makes
// gcd(lda*4/128, 128) = 1 at every k here, so all 128 sets become reachable --
// same loop, same instruction sequence, same number of bytes touched, same
// values written. If padding recovers the gap, the mechanism is set conflict.
// If it does not, this hypothesis is dead and the residual is something else.
//
// PRE-REGISTERED, before the first number:
//   H  confirmed iff  k_outer_pad beats k_outer by >=1.5x at k=2048 and k=4096,
//                     with the distributions separated,
//              and    the two are within noise at k=512.
//   NEGATIVE CONTROL (the half that makes it mean anything): row_outer_pad must
//   be within noise of row_outer at EVERY k. Its reads are one contiguous
//   stream, so it has no set pressure to relieve; if padding helps it too, the
//   effect is not set conflict and H is refuted whatever the k_outer arms did.
//
// Arms (identical bytes copied, all CHECKSUM-GATED against row_outer first):
//   k_outer       pre-S42 order, lda = k
//   k_outer_pad   pre-S42 order, lda = k + 32      <- traversal untouched
//   row_outer     what shipped before H2, lda = k
//   row_outer_pad what shipped before H2, lda = k + 32   <- negative control
//   blocked       what ships after H2, lda = k     <- for scale
//   null          k_outer again                    <- this run's drift
//
// Build: clang -O2 -march=armv8-a benches/sme/packsets.c -o packsets
// Run:   benches/perflock.sh ./packsets [N ...]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

#define PANEL_ROWS 32
#define KB 32
#define PAD 32          // +128 B of row stride: gcd(lda*4/128, 128) = 1 for every k here
#define LINE 128
#define SETS 128
#define WAYS 8          // ASSUMED, not measured -- sysctl does not expose it (S44's caveat)

static long gcdl(long a, long b) { while (b) { long t = a % b; a = b; b = t; } return a; }

static long sets_reachable(long lda) {
    long step = (lda * 4 / LINE) % SETS;
    if (step == 0) return 1;
    long cycle = SETS / gcdl(step, SETS);
    return cycle < PANEL_ROWS ? cycle : PANEL_ROWS;
}

static void pack_k_outer(const float *a, float *ap, long lda, long i0, long k) {
    for (long pk = 0; pk < k; pk++)
        for (long pi = 0; pi < PANEL_ROWS; pi++)
            ap[pk * PANEL_ROWS + pi] = a[(i0 + pi) * lda + pk];
}

static void pack_row_outer(const float *a, float *ap, long lda, long i0, long k) {
    for (long pi = 0; pi < PANEL_ROWS; pi++)
        for (long pk = 0; pk < k; pk++)
            ap[pk * PANEL_ROWS + pi] = a[(i0 + pi) * lda + pk];
}

static void pack_blocked(const float *a, float *ap, long lda, long i0, long k) {
    for (long pk0 = 0; pk0 < k; pk0 += KB) {
        long hi = pk0 + KB < k ? pk0 + KB : k;
        for (long pi = 0; pi < PANEL_ROWS; pi++)
            for (long pk = pk0; pk < hi; pk++)
                ap[pk * PANEL_ROWS + pi] = a[(i0 + pi) * lda + pk];
    }
}

typedef void (*packfn)(const float *, float *, long, long, long);

static double sweep(packfn fn, const float *a, float *ap, long lda, long N, int reps) {
    double best = 1e18;
    for (int r = 0; r < reps; r++) {
        double t0 = now_ms();
        for (long i0 = 0; i0 < N; i0 += PANEL_ROWS) fn(a, ap, lda, i0, N);
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
    long sizes[16]; int ns = 0;
    for (int i = 1; i < argc && ns < 16; i++) sizes[ns++] = atol(argv[i]);
    if (!ns) { long d[] = {512, 1024, 2048, 4096}; for (int i = 0; i < 4; i++) sizes[ns++] = d[i]; }

    printf("plan-s51 H2 follow-up -- is the REVERSED pack's read side an L1 set conflict?\n");
    printf("best of %d, 1 thread, no SME. %d B lines, %d sets, %d ways ASSUMED.\n\n",
           reps, LINE, SETS, WAYS);
    printf("%6s %6s %6s %9s %9s %8s   %9s %9s %8s   %9s %8s\n",
           "N", "sets", "sets+", "k_out", "k_out+", "gain", "row_out", "row_out+", "gain",
           "blocked", "drift");
    printf("---------------------------------------------------------------------------------------------------\n");

    for (int s = 0; s < ns; s++) {
        long N = sizes[s], lpad = N + PAD;
        float *a    = aligned_alloc(4096, (size_t)N * N * 4);
        float *apad = aligned_alloc(4096, (size_t)N * lpad * 4);
        float *ap   = aligned_alloc(4096, (size_t)PANEL_ROWS * N * 4);
        if (!a || !apad || !ap) { fprintf(stderr, "alloc failed at N=%ld\n", N); return 1; }
        for (long i = 0; i < N; i++)
            for (long j = 0; j < N; j++) {
                float v = (float)(((i * N + j) * 7) % 13) * 0.01f + 1.0f;
                a[i * N + j] = v;
                apad[i * lpad + j] = v;      // same VALUES, different stride
            }

        // ---- value gate: every arm writes identical bytes. Checked first.
        long apn = PANEL_ROWS * N;
        memset(ap, 0, (size_t)apn * 4);
        for (long i0 = 0; i0 < N; i0 += PANEL_ROWS) pack_row_outer(a, ap, N, i0, N);
        double ref = checksum(ap, apn);
        struct { const char *name; packfn fn; const float *src; long lda; } arms[] = {
            {"k_outer",       pack_k_outer,   a,    N},
            {"k_outer_pad",   pack_k_outer,   apad, lpad},
            {"row_outer_pad", pack_row_outer, apad, lpad},
            {"blocked",       pack_blocked,   a,    N},
        };
        for (int i = 0; i < 4; i++) {
            memset(ap, 0, (size_t)apn * 4);
            for (long i0 = 0; i0 < N; i0 += PANEL_ROWS)
                arms[i].fn(arms[i].src, ap, arms[i].lda, i0, N);
            if (checksum(ap, apn) != ref) {
                fprintf(stderr, "VALUE GATE FAILED at N=%ld arm %s\n", N, arms[i].name);
                return 1;
            }
        }

        double ko   = sweep(pack_k_outer,   a,    ap, N,    N, reps);
        double nul  = sweep(pack_k_outer,   a,    ap, N,    N, reps);
        double kop  = sweep(pack_k_outer,   apad, ap, lpad, N, reps);
        double ro   = sweep(pack_row_outer, a,    ap, N,    N, reps);
        double rop  = sweep(pack_row_outer, apad, ap, lpad, N, reps);
        double blk  = sweep(pack_blocked,   a,    ap, N,    N, reps);

        printf("%6ld %6ld %6ld %9.3f %9.3f %7.3fx   %9.3f %9.3f %7.3fx   %9.3f %7.3fx\n",
               N, sets_reachable(N), sets_reachable(lpad),
               ko, kop, ko / kop, ro, rop, ro / rop, blk, nul / ko);

        free(a); free(apad); free(ap);
    }
    printf("\n  gain > 1 means PADDING WON, i.e. the same traversal got faster from set mapping alone.\n");
    printf("  H needs k_out gain >= 1.5x at 2048/4096, ~1.0 at 512, AND row_out gain ~1.0 everywhere.\n");
    printf("  drift is k_outer measured twice: any effect not larger than it is not an effect.\n");
    return 0;
}
