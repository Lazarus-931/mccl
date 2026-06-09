#pragma once

#include <cstddef>

#include "../definitions.h"

namespace mccl {

// Page-aligned UMA buffer: one host pointer feeds Metal (wrap-on-use), m2m sockets, and RDMA with no copies.
struct mcclBuf {
  void*  host = nullptr;
  size_t size = 0;
};

struct mcclMemPool;  // fixed-size page-aligned freelist; mcclMemAlloc grows it by one when empty

mcclResult mcclMemPoolCreate(size_t buffSize, int count, mcclMemPool** out);
mcclResult mcclMemAlloc(mcclMemPool* p, mcclBuf* out);
mcclResult mcclMemFree(mcclMemPool* p, const mcclBuf& buf);
mcclResult mcclMemPoolDestroy(mcclMemPool* p);

mcclResult mcclPageAlloc(size_t bytes, void** out);
void       mcclPageFree(void* ptr);

}
