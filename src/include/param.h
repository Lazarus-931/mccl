#pragma once

#include <cstdint>

namespace mccl {

void mcclLoadParam(const char* env, int64_t deflt, int64_t uninit, int64_t* cache);

}

// MCCL_PARAM(Name) declares int64_t mcclParam##Name() (usable in a header); DEFINE_MCCL_PARAM defines it,
// reading env MCCL_<SUFFIX> once and caching (default must not be -1, which marks "unread"). Thread-safe.
#define MCCL_PARAM(name) int64_t mcclParam##name()

#define DEFINE_MCCL_PARAM(name, env, deflt)                     \
  MCCL_PARAM(name) {                                            \
    static_assert((deflt) != -1LL, "MCCL_PARAM default cannot be -1"); \
    static int64_t value = -1LL;                                \
    if (__builtin_expect(value == -1LL, 0))                     \
      mccl::mcclLoadParam("MCCL_" env, (deflt), -1LL, &value);  \
    return value;                                               \
  }
