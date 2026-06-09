#pragma once

#include "../definitions.h"
#include "../pool.h"

namespace mccl {

// Run fn(k) for each of nc tree children at once over the fan-out pool — each child has its own link, so the
// hub's transfers proceed in parallel instead of serially. One runs on the caller. Returns the first failure.
template <typename Fn>
inline mcclResult forEachChild(size_t nc, Fn&& fn) {
  return mcclParallel(mcclFanoutPool(), nc, fn);
}

}
