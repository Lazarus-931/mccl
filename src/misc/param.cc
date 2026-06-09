#include "../include/param.h"

#include <cerrno>
#include <cstdlib>
#include <mutex>

namespace mccl {

void mcclLoadParam(const char* env, int64_t deflt, int64_t uninit, int64_t* cache) {
  static std::mutex mu;
  std::lock_guard<std::mutex> lock(mu);
  if (*cache != uninit) return;
  int64_t value = deflt;
  const char* str = std::getenv(env);
  if (str != nullptr && str[0] != '\0') {
    char* end = nullptr;
    errno = 0;
    long long parsed = std::strtoll(str, &end, 0);
    if (errno == 0 && end != str) value = parsed;
  }
  *cache = value;
}

}
