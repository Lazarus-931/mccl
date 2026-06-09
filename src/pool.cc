#include "pool.h"

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace mccl {

class ThreadPool {
 public:
  explicit ThreadPool(int n) {
    for (int i = 0; i < n; ++i) workers_.emplace_back([this]() { loop(); });
  }
  ~ThreadPool() {
    { std::lock_guard<std::mutex> lk(mu_); stop_ = true; }
    cv_.notify_all();
    for (std::thread& t : workers_) if (t.joinable()) t.join();
  }
  void submit(std::function<void()> task) {
    { std::lock_guard<std::mutex> lk(mu_); q_.push_back(std::move(task)); }
    cv_.notify_one();
  }

 private:
  void loop() {
    for (;;) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [this]() { return stop_ || !q_.empty(); });
        if (stop_ && q_.empty()) return;
        task = std::move(q_.front());
        q_.pop_front();
      }
      task();
    }
  }
  std::vector<std::thread>          workers_;
  std::deque<std::function<void()>> q_;
  std::mutex                        mu_;
  std::condition_variable           cv_;
  bool                              stop_ = false;
};

namespace {
int poolThreads() {
  if (const char* e = std::getenv("MCCL_POOL_THREADS")) { const int v = std::atoi(e); if (v > 0) return v; }
  const unsigned hc = std::thread::hardware_concurrency();
  return hc > 0 ? static_cast<int>(hc) : 8;
}
}

ThreadPool& mcclStripePool() { static ThreadPool p(poolThreads()); return p; }
ThreadPool& mcclFanoutPool() { static ThreadPool p(poolThreads()); return p; }

// Tasks are CLAIMED (atomic counter) by the caller and by pool helpers alike, never assigned: the caller keeps
// claiming until the batch is exhausted, so a saturated pool cannot strand a task behind blocked workers. That
// is the deadlock-freedom argument for blocking I/O tasks — every rank's caller can always reach every task of
// its own op (worst case it runs them all serially), so the peer's matching send/recv always eventually runs.
// Pool helpers that wake after the batch is spent exit via the claim check; shared_ptr keeps state alive for them.
mcclResult mcclParallel(ThreadPool& pool, size_t count, const std::function<mcclResult(size_t)>& fn) {
  if (count == 0) return mcclSuccess;
  if (count == 1) return fn(0);

  struct Batch {
    std::function<mcclResult(size_t)> fn;
    std::vector<mcclResult>           rc;
    std::atomic<size_t>               next{0};
    std::mutex                        mu;
    std::condition_variable           cv;
    size_t                            done = 0;
  };
  auto b = std::make_shared<Batch>();
  b->fn = fn;
  b->rc.assign(count, mcclSuccess);

  auto claimLoop = [count](const std::shared_ptr<Batch>& s) {
    for (;;) {
      const size_t k = s->next.fetch_add(1, std::memory_order_relaxed);
      if (k >= count) return;
      mcclResult r;
      try { r = s->fn(k); } catch (...) { r = mcclInternalError; }
      std::lock_guard<std::mutex> lk(s->mu);
      s->rc[k] = r;
      ++s->done;
      s->cv.notify_one();
    }
  };

  for (size_t k = 1; k < count; ++k) pool.submit([b, claimLoop]() { claimLoop(b); });
  claimLoop(b);
  {
    std::unique_lock<std::mutex> lk(b->mu);
    b->cv.wait(lk, [&]() { return b->done == count; });
  }
  for (mcclResult r : b->rc) if (r != mcclSuccess) return r;
  return mcclSuccess;
}

}
