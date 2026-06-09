#include "pool.h"

#include <condition_variable>
#include <cstdlib>
#include <deque>
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

mcclResult mcclParallel(ThreadPool& pool, size_t count, const std::function<mcclResult(size_t)>& fn) {
  if (count == 0) return mcclSuccess;
  if (count == 1) return fn(0);
  std::vector<mcclResult> rc(count, mcclSuccess);
  std::mutex m;
  std::condition_variable done;
  size_t finished = 0;
  for (size_t k = 1; k < count; ++k)
    pool.submit([&, k]() {
      try { rc[k] = fn(k); } catch (...) { rc[k] = mcclInternalError; }
      std::lock_guard<std::mutex> lk(m);  // notify under the lock: the waiter must not wake + destroy m/done before this returns
      ++finished;
      done.notify_one();
    });
  try { rc[0] = fn(0); } catch (...) { rc[0] = mcclInternalError; }
  { std::unique_lock<std::mutex> lk(m); done.wait(lk, [&]() { return finished == count - 1; }); }
  for (mcclResult r : rc) if (r != mcclSuccess) return r;
  return mcclSuccess;
}

}
