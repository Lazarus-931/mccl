#pragma once

#include <atomic>
#include <cstdint>

namespace mccl {

void mcclLoadParam(const char* env, int64_t deflt, int64_t uninit, int64_t* cache);

}

// MCCL_PARAM(Name) declares int64_t mcclParam##Name() (usable in a header); DEFINE_MCCL_PARAM defines it,
// reading env MCCL_<SUFFIX> once and caching (default must not be -1, which marks "unread"). The cache is
// atomic: the fast-path read happens on transport threads concurrently with the first (mutex-guarded) load —
// a plain int64_t there is a data race. Concurrent first calls both load the env (idempotent) and store the
// same value.
#define MCCL_PARAM(name) int64_t mcclParam##name()

#define DEFINE_MCCL_PARAM(name, env, deflt)                     \
  MCCL_PARAM(name) {                                            \
    static_assert((deflt) != -1LL, "MCCL_PARAM default cannot be -1"); \
    static std::atomic<int64_t> value{-1LL};                    \
    int64_t v = value.load(std::memory_order_acquire);          \
    if (__builtin_expect(v == -1LL, 0)) {                       \
      v = -1LL;                                                 \
      mccl::mcclLoadParam("MCCL_" env, (deflt), -1LL, &v);      \
      value.store(v, std::memory_order_release);                \
    }                                                           \
    return v;                                                   \
  }
