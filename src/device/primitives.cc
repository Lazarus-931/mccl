#include "primitives.h"

#include "fanout.h"
#include "reduce_kernel.h"
#include "../pool.h"
#include "../include/comm.h"
#include "../include/param.h"
#include "../transport/m2m.h"

namespace mccl {

DEFINE_MCCL_PARAM(MetalMinBytes, "METAL_MIN_BYTES", int64_t{1} << 62);
DEFINE_MCCL_PARAM(PipelineChunks, "PIPELINE_CHUNKS", 4);

Primitives::Primitives(mcclComm* comm, void* buf, mcclDataType dt, mcclRedOp op, size_t maxChunkBytes)
    : comm_(comm),
      buf_(static_cast<char*>(buf)),
      dt_(dt),
      op_(op),
      esz_(mcclDataSize(dt)),
      gpu_(pageAligned(buf)),  // Metal availability is probed inside reduce()/reduceMulti(), past the size gate — constructing the Metal context here would put a multi-second XPC stall on every small-message path
      valid_(true),
      staging_(nullptr),
      stagingStride_(maxChunkBytes),
      tParent_(comm->parent),
      tChildren_(&comm->childConns) {
  // One staging slice per child so a hub can receive from all children at once, then fold them in a single
  // pass. Borrowed from the comm's persistent scratch. maxChunkBytes==0 means the caller will bindTree its own.
  const size_t slices = comm->childConns.empty() ? 1 : comm->childConns.size();
  if (maxChunkBytes > 0) valid_ = mcclCommReserveStaging(comm, maxChunkBytes * slices, &staging_) == mcclSuccess;
}

void Primitives::bindTree(mcclM2M* parent, const std::vector<mcclM2M*>* children, void* staging, size_t stagingStride) {
  tParent_ = parent;
  tChildren_ = children;
  staging_ = staging;
  stagingStride_ = stagingStride;
}

Primitives::~Primitives() {}

mcclResult Primitives::recvReduceSend(size_t sendOff, size_t sendCnt, size_t recvOff, size_t recvCnt) {
  const mcclResult rc = mcclParallel(mcclFanoutPool(), 2, [&](size_t k) {
    return k == 0 ? mcclM2MRecv(comm_->prev, staging_, recvCnt * esz_)
                  : mcclM2MSend(comm_->next, buf_ + sendOff * esz_, sendCnt * esz_);
  });
  if (rc != mcclSuccess) return rc;
  return reduce(buf_ + recvOff * esz_, staging_, recvCnt, dt_, op_, gpu_);
}

mcclResult Primitives::recvCopySend(size_t sendOff, size_t sendCnt, size_t recvOff, size_t recvCnt) {
  return mcclParallel(mcclFanoutPool(), 2, [&](size_t k) {
    return k == 0 ? mcclM2MRecv(comm_->prev, buf_ + recvOff * esz_, recvCnt * esz_)
                  : mcclM2MSend(comm_->next, buf_ + sendOff * esz_, sendCnt * esz_);
  });
}

// Receive from every child at once (each over its own link into its own staging slice), then fold them all
// into buf in one sweep / GPU dispatch — the hub's links run in parallel and the fold is one pass, not nc.
mcclResult Primitives::reduceFromChildren(size_t off, size_t cnt) {
  const size_t nc = tChildren_->size();
  if (nc == 0) return mcclSuccess;
  char* stg = static_cast<char*>(staging_);
  const mcclResult rc = forEachChild(nc, [&](size_t k) {
    return mcclM2MRecv((*tChildren_)[k], stg + k * stagingStride_, cnt * esz_);
  });
  if (rc != mcclSuccess) return rc;
  return reduceMulti(buf_ + off * esz_, stg, cnt, nc, stagingStride_ / esz_, dt_, op_, gpu_);
}

mcclResult Primitives::sendToParent(size_t off, size_t cnt) {
  return tParent_ ? mcclM2MSend(tParent_, buf_ + off * esz_, cnt * esz_) : mcclSuccess;
}

mcclResult Primitives::recvFromParent(size_t off, size_t cnt) {
  return tParent_ ? mcclM2MRecv(tParent_, buf_ + off * esz_, cnt * esz_) : mcclSuccess;
}

mcclResult Primitives::sendToChildren(size_t off, size_t cnt) {
  return forEachChild(tChildren_->size(), [&](size_t k) {
    return mcclM2MSend((*tChildren_)[k], buf_ + off * esz_, cnt * esz_);
  });
}

mcclResult Primitives::sendRecv(const void* sp, size_t sbytes, void* rp, size_t rbytes) {
  return mcclParallel(mcclFanoutPool(), 2, [&](size_t k) {
    return k == 0 ? mcclM2MRecv(comm_->prev, rp, rbytes)
                  : mcclM2MSend(comm_->next, sp, sbytes);
  });
}

mcclResult Primitives::sendNext(const void* p, size_t bytes) { return mcclM2MSend(comm_->next, p, bytes); }
mcclResult Primitives::recvPrev(void* p, size_t bytes)       { return mcclM2MRecv(comm_->prev, p, bytes); }

}
