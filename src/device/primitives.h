#pragma once

#include <cstddef>
#include <vector>

#include "../definitions.h"
#include "../include/param.h"

namespace mccl {

struct mcclComm;
struct mcclM2M;

MCCL_PARAM(PipelineChunks);

class Primitives {
 public:
  Primitives(mcclComm* comm, void* buf, mcclDataType dt, mcclRedOp op, size_t maxChunkBytes);
  ~Primitives();
  bool ok() const { return valid_; }

  mcclResult recvReduceSend(size_t sendOff, size_t sendCnt, size_t recvOff, size_t recvCnt);
  mcclResult recvCopySend(size_t sendOff, size_t sendCnt, size_t recvOff, size_t recvCnt);

  mcclResult reduceFromChildren(size_t off, size_t cnt);
  mcclResult sendToParent(size_t off, size_t cnt);
  mcclResult recvFromParent(size_t off, size_t cnt);
  mcclResult sendToChildren(size_t off, size_t cnt);

  void bindTree(mcclM2M* parent, const std::vector<mcclM2M*>* children, void* staging, size_t stagingStride);

  void bindRing(mcclM2M* prev, mcclM2M* next, void* staging, size_t stagingStride);

  mcclResult sendRecv(const void* sp, size_t sbytes, void* rp, size_t rbytes);
  mcclResult sendNext(const void* p, size_t bytes);
  mcclResult recvPrev(void* p, size_t bytes);

 private:
  char*        buf_;
  mcclDataType dt_;
  mcclRedOp    op_;
  size_t       esz_;
  bool         gpu_;
  bool         valid_;
  void*        staging_;
  size_t       stagingStride_;
  mcclM2M*                     tParent_;
  const std::vector<mcclM2M*>* tChildren_;
  mcclM2M*                     rPrev_;
  mcclM2M*                     rNext_;
};

}
