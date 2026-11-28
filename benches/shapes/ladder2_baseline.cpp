// CPU baselines for the shape-ladder-v2 classes: saxpy, reduce, transpose, gather.
// Build: clang++ -std=c++20 -O3 -march=native -ffp-contract=fast ladder2_baseline.cpp -o ladder2_baseline -pthread
//        (cpu_shapes.sh also builds it with -ffast-math and times whichever is faster)
//
// Every leg mirrors the Mapal shape exactly: the same procedural data generation
// (outside the timer), the same kernel, the same two printed values, and the same
// `iter ms=` line. Generation is deliberately outside the measured region on both
// sides — the S28 gen-boundary finding. The whole-machine leg runs on a pool started
// before the timer (pool.h), as Mapal's does; each kernel is written the way a careful
// C++ programmer would write it, so the comparison is against good C++, not naive C++.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "pool.h"

static std::unique_ptr<Pool> POOL;  // made in main, before any timing, for the mt leg only

static void parallel_for(bool mt, size_t n, const std::function<void(size_t, size_t)>& body) {
    if (!mt || !POOL) { body(0, n); return; }
    POOL->parallel_for(n, body);
}

template <typename Work>
static void run_iters(int iters, Work work) {
    for (int iter = 0; iter < iters; ++iter) {
        const auto start = std::chrono::steady_clock::now();
        work();
        const double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count();
        std::printf("iter ms=%.6f\n", ms);
    }
}

// The Mapal shapes print f32 through Rust's Display. Match the two-value probe
// with %g, which agrees on these exact-in-binary values.
static void probe(float a, float b) { std::printf("%g\n%g\n", a, b); }

// y[i] = 2.5*x[i] + y0[i] — streaming, bandwidth-bound.
static void run_saxpy(bool mt, int iters, size_t n) {
    std::vector<float> x(n), y0(n), y(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = static_cast<float>((static_cast<long>(i) * 7 + 13) % 101 - 50);
        y0[i] = static_cast<float>((static_cast<long>(i) * 5 + 3) % 31 - 15);
    }
    run_iters(iters, [&] {
        parallel_for(mt, n, [&](size_t lo, size_t hi) {
            for (size_t i = lo; i < hi; ++i) y[i] = 2.5f * x[i] + y0[i];
        });
    });
    probe(y[0], y[n - 1]);
}

// total = Σ x — reduction, no output array. Each thread sums its own chunk (by reference — the
// old leg captured the 4 MB vector by value, copying it into every thread on every call), then the
// partials are added in thread order. The strict build keeps one sequential sum per chunk; the
// -ffast-math build may vectorize it (reordering, as Mapal's default float rules allow).
static void run_reduce(bool mt, int iters, size_t n) {
    std::vector<float> x(n);
    for (size_t i = 0; i < n; ++i)
        x[i] = static_cast<float>((static_cast<long>(i) * 7 + 13) % 101 - 50);
    const float* xs = x.data();
    const unsigned width = mt && POOL ? POOL->width() : 1u;
    std::vector<float> partial(width * 16, 0.0f);  // one cache line per thread
    float total = 0.0f;
    run_iters(iters, [&] {
        auto chunk = [&](unsigned t) {
            const size_t lo = n * t / width, hi = n * (t + 1) / width;
            float acc = 0.0f;
            for (size_t i = lo; i < hi; ++i) acc += xs[i];
            partial[t * 16] = acc;
        };
        if (width == 1) chunk(0); else POOL->run(chunk);
        total = 0.0f;
        for (unsigned t = 0; t < width; ++t) total += partial[t * 16];
    });
    std::printf("%g\n", total);
}

// b[t] = a[(t % side)*side + t / side] — pure permutation, zero arithmetic. Done in 32 × 32 blocks so
// both the reads and the writes stay within cache lines (the old leg walked one side with a stride of
// a whole row).
static void run_transpose(bool mt, int iters, size_t side) {
    const size_t n = side * side, tb = 32, nb = (side + tb - 1) / tb;
    std::vector<float> a(n), b(n);
    for (size_t i = 0; i < n; ++i)
        a[i] = static_cast<float>((static_cast<long>(i) * 7 + 13) % 101 - 50);
    run_iters(iters, [&] {
        parallel_for(mt, nb, [&](size_t lo, size_t hi) {
            for (size_t rb = lo; rb < hi; ++rb)
                for (size_t cb = 0; cb < nb; ++cb)
                    for (size_t r = rb * tb; r < std::min(side, rb * tb + tb); ++r)
                        for (size_t c = cb * tb; c < std::min(side, cb * tb + tb); ++c)
                            b[r * side + c] = a[c * side + r];
        });
    });
    probe(b[0], b[n - 1]);
}

// y[i] = x[idx[i]] — data-dependent reads over a stride-1021 permutation.
static void run_gather(bool mt, int iters, size_t n) {
    std::vector<float> x(n), y(n);
    std::vector<int> idx(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = static_cast<float>((static_cast<long>(i) * 7 + 13) % 101 - 50);
        idx[i] = static_cast<int>((static_cast<long>(i) * 1021 + 12347) % static_cast<long>(n));
    }
    run_iters(iters, [&] {
        parallel_for(mt, n, [&](size_t lo, size_t hi) {
            for (size_t i = lo; i < hi; ++i) y[i] = x[idx[i]];
        });
    });
    probe(y[0], y[n - 1]);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
                     "usage: %s <saxpy|reduce|transpose|gather> <1t|mt> <iters> [size]\n", argv[0]);
        return 2;
    }
    const std::string shape = argv[1];
    const bool mt = std::strcmp(argv[2], "mt") == 0;
    const int iters = std::atoi(argv[3]);
    const size_t size = argc > 4 ? std::strtoull(argv[4], nullptr, 10) : 1048576;
    if (mt) POOL = std::make_unique<Pool>(pool_width());

    if (shape == "saxpy") run_saxpy(mt, iters, size);
    else if (shape == "reduce") run_reduce(mt, iters, size);
    else if (shape == "transpose") run_transpose(mt, iters, size);
    else if (shape == "gather") run_gather(mt, iters, size);
    else { std::fprintf(stderr, "unknown shape %s\n", shape.c_str()); return 2; }
    return 0;
}
