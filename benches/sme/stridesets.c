// Is the strided-read penalty about SET COUNT, or about the stride being a
// power of two? plan-s51 H2 follow-up #3.
//
// THE PROBLEM WITH EVERY MEASUREMENT SO FAR. Padding the row stride by 128 B
// helps 2.0-2.6x, but padding changes TWO things at once: it changes how many
// L1 sets the rows reach, AND it stops the stride being a power of two (and
// stops it being a multiple of the 16 KB page). Every arm run so far moves both,
// so none of them can say which one matters.
//
// THE DISCRIMINATOR NEEDS NO COUNTERS. Set count is `128 / gcd(step, 128)` where
// `step` is the stride in 128 B lines -- so DIFFERENT strides can share a set
// count. Stepping 16 lines and stepping 48 lines both reach 8 sets, but 16 lines
// is 2048 B (a power of two) and 48 lines is 6144 B (not one). So:
//
//   if SET COUNT is what matters   -> 16 and 48 are equally slow
//   if POWER-OF-TWO is what matters -> 16 is slow and 48 is fast
//
// The two models disagree on 6 of the 10 arms below. A stopwatch settles it.
//
// WHAT IS HELD CONSTANT: the number of rows (32), the elements read per row
// (K), the total bytes touched (512 rows x 512 floats = 1 MB), the loop body,
// and the VALUES read -- the array is filled as f(row, col) rather than from the
// address, so every arm sums the same logical numbers and the checksum is a real
// value gate. Only the stride varies.
//
// This is the READ side only -- no `ap`, no writes. The reversed pack's writes
// are already perfect (one full line at a time), so the suspect is the reads and
// they are what this isolates.
//
// Build: clang -O2 -march=armv8-a benches/sme/stridesets.c -o stridesets
// Run:   benches/perflock.sh ./stridesets
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

#define LINE_FLOATS 32      // 128 B line / 4 B float
#define SETS 128
#define WAYS 8              // ASSUMED -- sysctl does not expose associativity
#define PANEL 32            // rows read together, as the pack does
#define K 512               // elements read per row
#define ROWS 2048           // 64 panels -> 4 MB touched: past L1, inside L2
#define PAGE 16384          // this part's page size

// The reads are pure, so nine identical calls CSE into one and every timed
// region measures nothing (first run of this file printed 0.000 ms for all ten
// arms). The sink plus the barrier is what makes the work observable.
static volatile float g_sink;
#define BARRIER() __asm__ __volatile__("" ::: "memory")

static long gcdl(long a, long b) { while (b) { long t = a % b; a = b; b = t; } return a; }

static long sets_of(long step_lines) {
    long s = step_lines % SETS;
    if (s == 0) return 1;
    return SETS / gcdl(s, SETS);
}

// The reversed pack's read pattern, and nothing else: down the column, 32 rows.
//
// ONE ACCUMULATOR PER ROW, and that is not cosmetic. The first version of this
// file summed into a single `acc`, which is a serial FP dependency chain --
// 262144 adds at ~4 cycles each is 0.26 ms, and the whole sweep came back inside
// a 1.17x band at 5.7 GB/s on a part that does ~250 GB/s. It was timing the
// adder, not the memory, so every arm looked identical. Thirty-two independent
// chains put the loads back on the critical path.
__attribute__((noinline)) static float read_panel(const float *a, long stride_floats) {
    float acc[PANEL];
    for (long i = 0; i < PANEL; i++) acc[i] = 0;
    for (long i0 = 0; i0 < ROWS; i0 += PANEL)
        for (long pk = 0; pk < K; pk++)
            for (long pi = 0; pi < PANEL; pi++)
                acc[pi] += a[(i0 + pi) * stride_floats + pk];
    float s = 0;
    for (long i = 0; i < PANEL; i++) s += acc[i];
    return s;
}

int main(void) {
    int reps = getenv("REPS") ? atoi(getenv("REPS")) : 9;

    // step in LINES, and what each model predicts about it
    long steps[] = {16, 48, 80, 112, 64, 192, 128, 384, 17, 129};
    int n = sizeof(steps) / sizeof(steps[0]);
    // Single-step mode, for running one arm under a profiler.
    if (getenv("ONLY_STEP")) { steps[0] = atol(getenv("ONLY_STEP")); n = 1; }

    printf("plan-s51 H2 follow-up #3 -- SET COUNT or POWER OF TWO?\n");
    printf("32 rows, %d elements each, 1 MB touched, best of %d. Only the stride varies.\n\n",
           K, reps);
    printf("%6s %9s %6s %8s %7s %6s   %9s %8s\n",
           "step", "stride B", "sets", "rows/set", "pow2?", "page?", "ms", "vs best");

    double times[16], best = 1e18;
    float ref = 0;
    for (int i = 0; i < n; i++) {
        long step = steps[i];
        long stride_floats = step * LINE_FLOATS;
        if (stride_floats < K) { printf("  step %ld too small for K=%d\n", step, K); return 1; }
        float *a = aligned_alloc(PAGE, (size_t)ROWS * stride_floats * 4);
        if (!a) { fprintf(stderr, "alloc failed step=%ld\n", step); return 1; }
        // VALUES depend on (row, col), not on the address, so every arm sums the
        // same logical numbers however far apart the rows sit.
        for (long r = 0; r < ROWS; r++)
            for (long c = 0; c < K; c++)
                a[r * stride_floats + c] = (float)(((r * 7 + c * 13) % 101) - 50) * 0.01f;

        BARRIER();
        float acc = read_panel(a, stride_floats);   // warm + value gate
        BARRIER();
        if (i == 0) ref = acc;
        else if (acc != ref) {
            fprintf(stderr, "VALUE GATE FAILED at step=%ld: %.6f vs %.6f\n", step, acc, ref);
            return 1;
        }
        // The poke is what finally defeated CSE: `read_panel` is pure and the
        // array never changed, so N identical calls folded into one and every
        // timed region measured ~0. Writing through a volatile pointer each rep
        // means the compiler cannot prove the result is the same twice.
        volatile float *poke = (volatile float *)a;
        double t = 1e18;
        for (int r = 0; r < reps; r++) {
            *poke = (float)r * 1e-9f;
            BARRIER();
            double t0 = now_ms();
            g_sink += read_panel(a, stride_floats);
            double d = now_ms() - t0;
            BARRIER();
            if (d < t) t = d;
        }
        times[i] = t;
        if (t < best) best = t;
        free(a);
    }

    for (int i = 0; i < n; i++) {
        long step = steps[i], bytes = step * LINE_FLOATS * 4, sets = sets_of(step);
        long per_set = (PANEL + sets - 1) / sets;
        int pow2 = (bytes & (bytes - 1)) == 0;
        int pagey = (bytes % PAGE) == 0;
        printf("%6ld %9ld %6ld %8ld %7s %6s   %9.3f %7.2fx\n",
               step, bytes, sets, per_set, pow2 ? "YES" : "no", pagey ? "YES" : "no",
               times[i], times[i] / best);
        if (i == n - 1)
            printf("\n  sanity: fastest arm moves %.1f MB in %.3f ms = %.0f GB/s\n",
                   (double)ROWS * K * 4 / 1048576.0, best,
                   (double)ROWS * K * 4 / (best * 1e-3) / 1e9);
    }

    printf("\n  SET-COUNT model predicts: {16,48,80,112} equal (8 sets), {64,192} equal (2),\n");
    printf("  {128,384} equal (1), {17,129} fastest (128). Rows/set > %d ways is its\n", WAYS);
    printf("  only failure mode.\n");
    printf("  POWER-OF-TWO model predicts: 16, 64, 128 slow; every other arm fast.\n");
    printf("  They disagree on 48, 80, 112, 192, 384 -- read those rows first.\n");
    return 0;
}
