// CPU baselines for the fir/conv2d Mapal shapes, size-parameterized (S29 scale-up).
// Build: clang++ -std=c++20 -O3 -march=native -ffp-contract=fast shapes_baseline.cpp -o shapes_baseline -pthread
//        (cpu_shapes.sh also builds it with -ffast-math and times whichever is faster)
//
// Written the way a careful C++ programmer would: outputs in fixed tiles of TILE (a compile-time constant)
// that accumulate in registers across all taps, then are stored once. Every output still adds its terms in
// tap order, bit-identical to the textbook loop. (The textbook loop — outputs outside, taps inside — sums into
// one scalar and cannot vectorize under strict FP: 20 ms against 0.95 ms for FIR 1M on the Threadripper; a
// run-time-sized tile let clang reload every output once per tap: 2.6 ms.) The whole-machine leg runs on a
// pool started before the timer (pool.h), as Mapal's does.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "pool.h"

static std::unique_ptr<Pool> POOL;  // made in main, before any timing, for the mt leg only

// One tile of B outputs (B a compile-time constant) accumulates in registers across all taps, then is
// stored once; a tail shorter than B uses the plain loop. A run-time-sized block instead made clang load and
// store every output once per tap (2.6 ms against 0.95 ms for FIR 1M). Bit-identical to the textbook loop.
constexpr size_t TILE = 256;

// y[t] = Σ_k w[k]·x[t+k] for t in [begin, end).
static void fir_range(const std::vector<float>& x, const std::vector<float>& w,
                      std::vector<float>& y, size_t begin, size_t end) {
    const size_t taps = w.size();
    size_t t0 = begin;
    for (; t0 + TILE <= end; t0 += TILE) {
        float acc[TILE] = {};
        for (size_t k = 0; k < taps; ++k) {
            const float wk = w[k];
            const float* xs = &x[t0 + k];
            for (size_t j = 0; j < TILE; ++j) acc[j] += wk * xs[j];
        }
        for (size_t j = 0; j < TILE; ++j) y[t0 + j] = acc[j];
    }
    for (; t0 < end; ++t0) {
        float a = 0.0f;
        for (size_t k = 0; k < taps; ++k) a += w[k] * x[t0 + k];
        y[t0] = a;
    }
}

// out[i][j] = Σ_k w[k]·img[i + k/3][j + k%3] for rows [row_begin, row_end): each output row in tiles of TILE
// columns, the same way.
static void conv_range(const std::vector<float>& img, const std::vector<float>& w,
                       std::vector<float>& out, size_t side, size_t row_begin, size_t row_end) {
    const size_t stride = side + 2;
    for (size_t i = row_begin; i < row_end; ++i) {
        float* o = &out[i * side];
        size_t j0 = 0;
        for (; j0 + TILE <= side; j0 += TILE) {
            float acc[TILE] = {};
            for (size_t k = 0; k < 9; ++k) {
                const float wk = w[k];
                const float* row = &img[(i + k / 3) * stride + k % 3 + j0];
                for (size_t j = 0; j < TILE; ++j) acc[j] += wk * row[j];
            }
            for (size_t j = 0; j < TILE; ++j) o[j0 + j] = acc[j];
        }
        for (; j0 < side; ++j0) {
            float a = 0.0f;
            for (size_t k = 0; k < 9; ++k) a += w[k] * img[(i + k / 3) * stride + k % 3 + j0];
            o[j0] = a;
        }
    }
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

static void run_fir(int iters, size_t n) {
    std::vector<float> x(n + 63), w(64), y(n);
    for (size_t t = 0; t < x.size(); ++t)
        x[t] = static_cast<float>(static_cast<int>((t * 7 + 13) % 101) - 50);
    for (size_t k = 0; k < w.size(); ++k)
        w[k] = static_cast<float>(static_cast<int>((k * 5 + 3) % 31) - 15);
    run_iters(iters, [&] {
        if (!POOL) { fir_range(x, w, y, 0, y.size()); return; }
        POOL->parallel_for(y.size(), [&](size_t lo, size_t hi) { fir_range(x, w, y, lo, hi); });
    });
    std::printf("%.9g\n%.9g\n", static_cast<double>(y.front()), static_cast<double>(y.back()));
}

static void run_conv(int iters, size_t side) {
    const size_t stride = side + 2;
    std::vector<float> img(stride * stride), w(9), out(side * side);
    for (size_t t = 0; t < img.size(); ++t)
        img[t] = static_cast<float>(static_cast<int>((t * 7 + 13) % 101) - 50);
    for (size_t k = 0; k < w.size(); ++k)
        w[k] = static_cast<float>(static_cast<int>((k * 5 + 3) % 31) - 15);
    run_iters(iters, [&] {
        if (!POOL) { conv_range(img, w, out, side, 0, side); return; }
        POOL->parallel_for(side, [&](size_t lo, size_t hi) { conv_range(img, w, out, side, lo, hi); });
    });
    std::printf("%.9g\n%.9g\n", static_cast<double>(out.front()), static_cast<double>(out.back()));
}

int main(int argc, char** argv) {
    if (argc < 4 || argc > 5 || (std::strcmp(argv[1], "fir") != 0 && std::strcmp(argv[1], "conv2d") != 0) ||
        (std::strcmp(argv[2], "1t") != 0 && std::strcmp(argv[2], "mt") != 0)) {
        std::fprintf(stderr, "usage: %s <fir|conv2d> <1t|mt> <iters> [n|side]\n", argv[0]);
        return 2;
    }
    const int iters = std::atoi(argv[3]);
    if (iters < 1) {
        std::fprintf(stderr, "iters must be >= 1\n");
        return 2;
    }
    const bool fir = std::strcmp(argv[1], "fir") == 0;
    const size_t default_n = fir ? 65536 : 512;
    const size_t n = argc == 5 ? static_cast<size_t>(std::atoll(argv[4])) : default_n;
    if (std::strcmp(argv[2], "mt") == 0) POOL = std::make_unique<Pool>(pool_width());
    if (fir)
        run_fir(iters, n);
    else
        run_conv(iters, n);
}
