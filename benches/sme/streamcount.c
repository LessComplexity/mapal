// Does the strided-read penalty scale with the NUMBER OF CONCURRENT STREAMS?
// plan-s51 H2 follow-up #4 -- the prefetch question, without counters.
//
// WHY NOT COUNTERS. The events exist on this part (`L1D_CACHE_MISS_LD` and
// `L1D_CACHE_MISS_LD_NONSPEC` -- their difference IS the speculative/prefetch
// portion), but they are not reachable from the command line: `xctrace record`
// has no event-selection flag, and the stock "CPU Counters" template is pinned
// to Guided/"CPU Bottlenecks" mode. Its trace metadata lists an
// `l1d_miss_sampling` counting mode, so the capability is there -- but the
// template is an NSKeyedArchiver blob and only the GUI selects the mode.
//
// WHAT THIS MEASURES INSTEAD, and it is the causal question rather than a proxy:
// §5h showed the penalty is monotone in how many L1 sets the stride reaches, but
// left a hole -- at 8 sets the 32 rows are only 4 lines per set against 8 ways,
// which should fit. Something inflates demand beyond the 32 live lines, and
// hardware prefetch over 32 simultaneous streams is the obvious suspect.
//
// So: hold the stride FIXED and vary how many rows are interleaved (P). Total
// reads are ROWS*K for every P -- only the number of streams in flight changes.
//
//   if per-set OCCUPANCY drives it -> time grows with P (P rows share `sets`
//                                     sets, so lines-per-set = P/sets)
//   if it is INTRINSIC to the stride (page/DRAM behaviour, not occupancy)
//                                  -> time is flat in P
//
// Run at a stride that reaches few sets (16 lines -> 8 sets, and it is d=512's
// actual stride) against one that reaches all of them (17 lines -> 128 sets).
// The second is the control: it should be flat and fast at every P.
//
// !! RESULT: WEAK. READ BEFORE REUSING !!
// Making P a RUNTIME parameter blocked the unrolling the fixed-PANEL version
// gets, and that washed out most of the phenomenon: at P=32 this reproduces a
// 1.19x stride gap where benches/sme/stridesets.c (compile-time PANEL=32)
// measures 3.05x. So its null result is low-power evidence, not a refutation.
// What it does show, for what that is worth: the gap SHRINKS as P grows
// (1.42-1.83x at P<=4 down to 1.15-1.19x at P>=16), the opposite of what
// prefetch-over-many-streams predicts, and there is already a 1.42x gap at P=1
// where no row can contend with another at all.
//
// Build: clang -O2 -march=armv8-a benches/sme/streamcount.c -o streamcount
// Run:   benches/perflock.sh ./streamcount
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

#define LINE_FLOATS 32
#define SETS 128
#define K 512
#define ROWS 2048
#define PAGE 16384
#define MAXP 32

static volatile float g_sink;
#define BARRIER() __asm__ __volatile__("" ::: "memory")

static long gcdl(long a, long b) { while (b) { long t = a % b; a = b; b = t; } return a; }
static long sets_of(long st) { long s = st % SETS; return s ? SETS / gcdl(s, SETS) : 1; }

// P independent accumulators, so the loads stay on the critical path rather than
// a serial FP chain -- the defect that made an earlier instrument in this
// sequence report a flat 1.17x band at 5.7 GB/s.
__attribute__((noinline))
static float read_streams(const float *a, long stride_floats, long P) {
    float acc[MAXP];
    for (long i = 0; i < MAXP; i++) acc[i] = 0;
    for (long i0 = 0; i0 + P <= ROWS; i0 += P)
        for (long pk = 0; pk < K; pk++)
            for (long pi = 0; pi < P; pi++)
                acc[pi] += a[(i0 + pi) * stride_floats + pk];
    float s = 0;
    for (long i = 0; i < MAXP; i++) s += acc[i];
    return s;
}

int main(void) {
    int reps = getenv("REPS") ? atoi(getenv("REPS")) : 15;
    const long steps[] = {16, 17};
    const long Ps[] = {1, 2, 4, 8, 16, 32};
    const int np = sizeof(Ps) / sizeof(Ps[0]);

    printf("plan-s51 H2 follow-up #4 -- does the penalty scale with STREAM COUNT?\n");
    printf("stride fixed, only the number of interleaved rows varies.\n");
    printf("%ld reads either way. best of %d.\n\n", (long)ROWS * K, reps);

    for (int si = 0; si < 2; si++) {
        long step = steps[si], stride_floats = step * LINE_FLOATS;
        long sets = sets_of(step);
        float *a = aligned_alloc(PAGE, (size_t)ROWS * stride_floats * 4);
        if (!a) { fprintf(stderr, "alloc failed\n"); return 1; }
        for (long r = 0; r < ROWS; r++)
            for (long c = 0; c < K; c++)
                a[r * stride_floats + c] = (float)(((r * 7 + c * 13) % 101) - 50) * 0.01f;

        printf("=== step %ld lines = %ld B stride, reaches %ld of %d sets%s ===\n",
               step, step * LINE_FLOATS * 4, sets, SETS,
               sets == SETS ? "   <- CONTROL" : "");
        printf("  %4s %12s %14s %10s\n", "P", "lines/set", "ms", "vs P=1");
        double base = 0;
        for (int i = 0; i < np; i++) {
            long P = Ps[i];
            volatile float *poke = (volatile float *)a;
            double t = 1e18;
            for (int r = 0; r < reps; r++) {
                *poke = (float)r * 1e-9f;
                BARRIER();
                double t0 = now_ms();
                g_sink += read_streams(a, stride_floats, P);
                double d = now_ms() - t0;
                BARRIER();
                if (d < t) t = d;
            }
            if (i == 0) base = t;
            double lps = (double)P / (double)(P < sets ? P : sets);
            printf("  %4ld %12.2f %14.3f %9.2fx\n", P, lps, t, t / base);
        }
        printf("\n");
        free(a);
    }
    printf("  Growing with P  => per-set OCCUPANCY drives it (and prefetch inflating\n");
    printf("                     demand beyond the live lines is the candidate).\n");
    printf("  Flat in P       => intrinsic to the stride, not to how many streams run.\n");
    return 0;
}
