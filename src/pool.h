#pragma once

#include <cstddef>
#include <functional>

#include "definitions.h"

namespace mccl {

class ThreadPool;  // process-wide persistent workers; replaces per-op std::thread spawns on the data path

// Two pools at distinct nesting levels so a fan-out task can block on stripe tasks without a same-pool deadlock:
// the fan-out pool runs forEachChild + the ring send/recv pair; those call only the stripe pool (the leaf).
ThreadPool& mcclStripePool();
ThreadPool& mcclFanoutPool();

// Run fn(0..count-1) — index 0 on the caller, the rest on `pool` — block until all finish, return the first failure.
mcclResult mcclParallel(ThreadPool& pool, size_t count, const std::function<mcclResult(size_t)>& fn);

}
