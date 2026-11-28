// A thread pool for the C++ baselines' whole-machine legs, started once before any timing — as Mapal's
// runtime starts its own pool before its timer — so a timed call pays a wake-up, never a thread start.
// The calling thread is worker 0. Build with -std=c++20 (std::barrier).
#pragma once
#include <algorithm>
#include <atomic>
#include <barrier>
#include <cstdlib>
#include <functional>
#include <thread>
#include <vector>

class Pool {
  public:
    explicit Pool(unsigned width) : width_(std::max(1u, width)), start_(width_), done_(width_) {
        for (unsigned t = 1; t < width_; ++t)
            workers_.emplace_back([this, t] {
                for (;;) {
                    start_.arrive_and_wait();
                    if (quit_.load(std::memory_order_acquire)) return;
                    job_(t);
                    done_.arrive_and_wait();
                }
            });
    }
    ~Pool() {
        quit_.store(true, std::memory_order_release);
        start_.arrive_and_wait();
        for (auto& w : workers_) w.join();
    }
    unsigned width() const { return width_; }
    // Run `job(t)` on every worker t in [0, width) and wait for all of them.
    void run(const std::function<void(unsigned)>& job) {
        if (width_ == 1) { job(0); return; }
        job_ = job;
        start_.arrive_and_wait();
        job_(0);
        done_.arrive_and_wait();
    }
    // Split [0, n) into `width` contiguous chunks.
    void parallel_for(size_t n, const std::function<void(size_t, size_t)>& body) {
        run([&](unsigned t) {
            const size_t lo = n * t / width_, hi = n * (t + 1) / width_;
            if (lo < hi) body(lo, hi);
        });
    }

  private:
    unsigned width_;
    std::barrier<> start_, done_;
    std::function<void(unsigned)> job_;
    std::atomic<bool> quit_{false};
    std::vector<std::thread> workers_;
};

// THREADS, else the hardware's thread count.
inline unsigned pool_width() {
    if (const char* v = std::getenv("THREADS")) {
        const int t = std::atoi(v);
        if (t >= 1) return static_cast<unsigned>(t);
    }
    const unsigned t = std::thread::hardware_concurrency();
    return t == 0 ? 1u : t;
}
