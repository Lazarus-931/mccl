#include "include/alloc.h"

#include <unistd.h>

#include <cstdlib>
#include <mutex>
#include <vector>

namespace mccl {

struct mcclMemPool {
  size_t             buffSize = 0;
  std::vector<void*> all;
  std::vector<void*> free;
  std::mutex         mu;
};

mcclResult mcclPageAlloc(size_t bytes, void** out) {
  if (out == nullptr) return mcclInvalidArgument;
  *out = nullptr;
  const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));  // 16 KiB on Apple Silicon; page alignment lets Metal wrap the buffer zero-copy
  const size_t aligned = bytes == 0 ? page : ((bytes + page - 1) / page) * page;
  void* p = nullptr;
  if (posix_memalign(&p, page, aligned) != 0) return mcclSystemError;
  *out = p;
  return mcclSuccess;
}

void mcclPageFree(void* ptr) { std::free(ptr); }

mcclResult mcclMemPoolDestroy(mcclMemPool* p) {
  if (p == nullptr) return mcclInvalidArgument;
  for (void* b : p->all) mcclPageFree(b);
  delete p;
  return mcclSuccess;
}

mcclResult mcclMemPoolCreate(size_t buffSize, int count, mcclMemPool** out) {
  if (out == nullptr || buffSize == 0 || count < 0) return mcclInvalidArgument;
  *out = nullptr;
  auto* p = new mcclMemPool();
  p->buffSize = buffSize;
  for (int i = 0; i < count; ++i) {
    void* b = nullptr;
    if (mcclPageAlloc(buffSize, &b) != mcclSuccess) {
      mcclMemPoolDestroy(p);
      return mcclSystemError;
    }
    p->all.push_back(b);
    p->free.push_back(b);
  }
  *out = p;
  return mcclSuccess;
}

mcclResult mcclMemAlloc(mcclMemPool* p, mcclBuf* out) {
  if (p == nullptr || out == nullptr) return mcclInvalidArgument;
  std::lock_guard<std::mutex> lk(p->mu);
  void* b = nullptr;
  if (!p->free.empty()) {
    b = p->free.back();
    p->free.pop_back();
  } else {
    if (mcclPageAlloc(p->buffSize, &b) != mcclSuccess) return mcclSystemError;
    p->all.push_back(b);
  }
  out->host = b;
  out->size = p->buffSize;
  return mcclSuccess;
}

mcclResult mcclMemFree(mcclMemPool* p, const mcclBuf& buf) {
  if (p == nullptr || buf.host == nullptr) return mcclInvalidArgument;
  std::lock_guard<std::mutex> lk(p->mu);
  p->free.push_back(buf.host);
  return mcclSuccess;
}

}

#ifdef MCCL_ALLOC_MAIN
#include <cstdint>
#include <cstdio>

int main() {
  using namespace mccl;
  auto envI = [](const char* k, int d) { const char* v = std::getenv(k); return v ? std::atoi(v) : d; };
  const size_t bs  = static_cast<size_t>(envI("MCCL_BUFFERSIZE", 65536));
  const int    cnt = envI("MCCL_NUM_BUFFERS", 8);
  const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));

  mcclMemPool* p = nullptr;
  if (mcclMemPoolCreate(bs, cnt, &p) != mcclSuccess) { std::printf("create failed\n"); return 1; }

  std::vector<mcclBuf> bufs;
  bool aligned = true, distinct = true;
  for (int i = 0; i < cnt + 2; ++i) {
    mcclBuf b{};
    if (mcclMemAlloc(p, &b) != mcclSuccess) { std::printf("alloc failed\n"); return 1; }
    if (reinterpret_cast<uintptr_t>(b.host) % page != 0) aligned = false;
    for (const mcclBuf& x : bufs) if (x.host == b.host) distinct = false;
    bufs.push_back(b);
  }
  void* freed = bufs[0].host;
  mcclMemFree(p, bufs[0]);
  mcclBuf r{};
  mcclMemAlloc(p, &r);
  const bool reused = (r.host == freed);

  std::printf("[alloc] page=%zu buffSize=%zu count=%d allocated=%zu aligned=%d distinct=%d grew=%d reused=%d\n",
              page, bs, cnt, bufs.size(), aligned, distinct, static_cast<int>(bufs.size() > static_cast<size_t>(cnt)), reused);
  mcclMemPoolDestroy(p);
  return (aligned && distinct && reused) ? 0 : 1;
}
#endif
