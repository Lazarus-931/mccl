#include "../include/checks.h"

#include <cstdio>
#include <cstdlib>

namespace mccl {

void mcclCheckLog(const char* file, int line, mcclResult res) {
  if (std::getenv("MCCL_DEBUG") != nullptr)
    std::fprintf(stderr, "mccl %s:%d -> %d\n", file, line, static_cast<int>(res));
}

}
