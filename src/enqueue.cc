#include "include/comm.h"

#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>

namespace mccl {

int mcclPickAlgo(const mcclComm* comm, size_t bytes) {
  int a = comm->channel;
  mcclGetAlgoInfo(comm->graphs, comm->nRanks, bytes, comm->enabled, &a);
  return a;
}

struct mcclWorkQueue {
  std::thread                             th;
  std::mutex                              mu;
  std::condition_variable                 cv;
  std::deque<std::function<mcclResult()>> q;
  size_t                                  inFlight   = 0;
  bool                                    stop       = false;
  mcclResult                              firstError = mcclSuccess;  // sticky until Abort: a failed channel is desynced, don't run more over it
  mcclComm*                               comm       = nullptr;
};

namespace {
void workerLoop(mcclWorkQueue* w) {
  for (;;) {
    std::function<mcclResult()> job;
    mcclResult latched;
    {
      std::unique_lock<std::mutex> lk(w->mu);
      w->cv.wait(lk, [w]() { return w->stop || !w->q.empty(); });
      if (w->stop) return;
      job = std::move(w->q.front());
      w->q.pop_front();
      latched = w->firstError;
      ++w->inFlight;
    }
    mcclResult rc = latched;  // already failed fatally: skip the job, just propagate
    if (!mcclResultFatal(latched)) {
      try { rc = job(); }
      catch (...) { rc = mcclInternalError; }  // an escaped exception would std::terminate the worker thread
    }
    {
      std::lock_guard<std::mutex> lk(w->mu);
      --w->inFlight;
      if (mcclResultFatal(rc) && !mcclResultFatal(w->firstError)) w->firstError = rc;  // latch fatal only, not per-call usage errors
      w->comm->asyncState = w->firstError;
      w->cv.notify_all();
    }
  }
}
}

mcclResult mcclWorkerStart(mcclComm* comm) {
  comm->work = new mcclWorkQueue();
  comm->work->comm = comm;
  comm->work->th = std::thread(workerLoop, comm->work);
  return mcclSuccess;
}

void mcclWorkerStop(mcclComm* comm) {
  if (comm->work == nullptr) return;
  mcclWorkQueue* w = comm->work;
  { std::lock_guard<std::mutex> lk(w->mu); w->stop = true; }
  w->cv.notify_all();
  if (w->th.joinable()) w->th.join();
  delete w;
  comm->work = nullptr;
}

mcclResult mcclEnqueue(mcclComm* comm, std::function<mcclResult()> op) {
  mcclWorkQueue* w = comm->work;
  if (w == nullptr) return mcclInternalError;
  {
    std::lock_guard<std::mutex> lk(w->mu);
    if (mcclResultFatal(w->firstError)) return w->firstError;
    w->q.push_back(std::move(op));
  }
  w->cv.notify_one();
  return mcclSuccess;
}

mcclResult mcclCommSynchronize(mcclComm* comm) {
  if (comm == nullptr) return mcclSetLastError(mcclInvalidArgument);
  mcclWorkQueue* w = comm->work;
  if (w == nullptr) return comm->asyncState != mcclSuccess ? mcclSetLastError(comm->asyncState) : mcclSuccess;
  mcclResult rc;
  {
    std::unique_lock<std::mutex> lk(w->mu);
    w->cv.wait(lk, [w]() { return w->q.empty() && w->inFlight == 0; });
    rc = w->firstError;
  }
  return rc == mcclSuccess ? mcclSuccess : mcclSetLastError(rc);
}

mcclResult mcclWorkerAsyncError(mcclComm* comm, mcclResult* out) {
  std::lock_guard<std::mutex> lk(comm->work->mu);
  if (comm->work->firstError != mcclSuccess)                   *out = comm->work->firstError;
  else if (!comm->work->q.empty() || comm->work->inFlight > 0) *out = mcclInProgress;
  else                                                         *out = mcclSuccess;
  return mcclSuccess;
}

}
