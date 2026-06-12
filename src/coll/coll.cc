#include "../include/coll.h"

#include <cstring>

#include "../include/comm.h"
#include "../include/graph.h"
#include "../include/param.h"
#include "../device/all_reduce.h"
#include "../device/all_gather.h"
#include "../device/reduce_scatter.h"
#include "../device/broadcast.h"
#include "../device/reduce.h"

namespace mccl {

namespace {
DEFINE_MCCL_PARAM(DirectMaxBytes, "DIRECT_MAX_BYTES", 2 << 20);
DEFINE_MCCL_PARAM(DirectMaxRanks, "DIRECT_MAX_RANKS", 8);

bool useDirect(const mcclComm* comm, size_t bytes) {
  if (comm->nRanks <= 1 || comm->nRanks > static_cast<int>(mcclParamDirectMaxRanks())) return false;
  if (bytes > static_cast<size_t>(mcclParamDirectMaxBytes())) return false;
  for (int p = 0; p < comm->nRanks; ++p)
    if (p != comm->rank && !mcclTopoDirectLink(comm->system, comm->rank, p, nullptr, nullptr)) return false;
  return true;
}

}

// Validation lives HERE, before mcclLaunch: on a non-blocking comm the lambda runs on the worker, which
// latches only fatal results — an mcclInvalidArgument raised inside it would be silently swallowed while
// peer ranks block. count == 0 is a successful no-op (NCCL semantics), not an error.
mcclResult mcclAllReduce(mcclComm* comm, const void* sendbuff, void* recvbuff, size_t count, mcclDataType dt, mcclRedOp op) {
  if (comm == nullptr || sendbuff == nullptr || recvbuff == nullptr || mcclDataSize(dt) == 0) return mcclSetLastError(mcclInvalidArgument);
  if (count == 0) return mcclSuccess;
  if (useDirect(comm, count * mcclDataSize(dt)))
    return mcclSetLastError(mcclLaunch(comm, [=]() { return directAllReduce(comm, sendbuff, recvbuff, count, dt, op); }));
  const int algo = mcclPickAlgo(comm, count * mcclDataSize(dt));
  return mcclSetLastError(mcclLaunch(comm, [=]() {
    return algo == MCCL_ALGO_RING ? ringAllReduce(comm, sendbuff, recvbuff, count, dt, op)
                                  : treeAllReduce(comm, sendbuff, recvbuff, count, dt, op);
  }));
}

mcclResult mcclAllGather(mcclComm* comm, const void* sendbuff, void* recvbuff, size_t sendcount, mcclDataType dt) {
  if (comm == nullptr || sendbuff == nullptr || recvbuff == nullptr || mcclDataSize(dt) == 0) return mcclSetLastError(mcclInvalidArgument);
  if (sendcount == 0) return mcclSuccess;
  if (useDirect(comm, static_cast<size_t>(comm->nRanks) * sendcount * mcclDataSize(dt)))
    return mcclSetLastError(mcclLaunch(comm, [=]() { return directAllGather(comm, sendbuff, recvbuff, sendcount, dt); }));
  const int algo = mcclPickAlgo(comm, sendcount * mcclDataSize(dt));
  return mcclSetLastError(mcclLaunch(comm, [=]() {
    return algo == MCCL_ALGO_RING ? ringAllGather(comm, sendbuff, recvbuff, sendcount, dt)
                                  : treeAllGather(comm, sendbuff, recvbuff, sendcount, dt);
  }));
}

mcclResult mcclReduceScatter(mcclComm* comm, const void* sendbuff, void* recvbuff, size_t recvcount, mcclDataType dt, mcclRedOp op) {
  if (comm == nullptr || sendbuff == nullptr || recvbuff == nullptr || mcclDataSize(dt) == 0) return mcclSetLastError(mcclInvalidArgument);
  if (recvcount == 0) return mcclSuccess;
  if (useDirect(comm, static_cast<size_t>(comm->nRanks) * recvcount * mcclDataSize(dt)))
    return mcclSetLastError(mcclLaunch(comm, [=]() { return directReduceScatter(comm, sendbuff, recvbuff, recvcount, dt, op); }));
  const int algo = mcclPickAlgo(comm, recvcount * mcclDataSize(dt));
  return mcclSetLastError(mcclLaunch(comm, [=]() {
    return algo == MCCL_ALGO_RING ? ringReduceScatter(comm, sendbuff, recvbuff, recvcount, dt, op)
                                  : treeReduceScatter(comm, sendbuff, recvbuff, recvcount, dt, op);
  }));
}

mcclResult mcclBroadcast(mcclComm* comm, const void* sendbuff, void* recvbuff, size_t count, mcclDataType dt, int root) {
  if (comm == nullptr || recvbuff == nullptr || mcclDataSize(dt) == 0) return mcclSetLastError(mcclInvalidArgument);
  if (root < 0 || root >= comm->nRanks || (comm->rank == root && sendbuff == nullptr)) return mcclSetLastError(mcclInvalidArgument);
  if (count == 0) return mcclSuccess;
  const int algo = mcclPickAlgo(comm, count * mcclDataSize(dt));
  return mcclSetLastError(mcclLaunch(comm, [=]() {
    return algo == MCCL_ALGO_RING ? ringBroadcast(comm, sendbuff, recvbuff, count, dt, root)
                                  : treeBroadcast(comm, sendbuff, recvbuff, count, dt, root);
  }));
}

mcclResult mcclReduce(mcclComm* comm, const void* sendbuff, void* recvbuff, size_t count, mcclDataType dt, mcclRedOp op, int root) {
  if (comm == nullptr || sendbuff == nullptr || root < 0 || root >= comm->nRanks || mcclDataSize(dt) == 0) return mcclSetLastError(mcclInvalidArgument);
  if (comm->rank == root && recvbuff == nullptr) return mcclSetLastError(mcclInvalidArgument);
  if (count == 0) return mcclSuccess;
  if (useDirect(comm, count * mcclDataSize(dt)))
    return mcclSetLastError(mcclLaunch(comm, [=]() { return directReduce(comm, sendbuff, recvbuff, count, dt, op, root); }));
  const bool ring = mcclPickAlgo(comm, count * mcclDataSize(dt)) == MCCL_ALGO_RING;
  return mcclSetLastError(mcclLaunch(comm, [=]() { return reduceImpl(comm, sendbuff, recvbuff, count, dt, op, root, ring); }));
}

mcclResult mcclAllToAll(mcclComm* comm, const void* sendbuff, void* recvbuff, size_t count, mcclDataType dt) {
  if (comm == nullptr || sendbuff == nullptr || recvbuff == nullptr || mcclDataSize(dt) == 0) return mcclSetLastError(mcclInvalidArgument);
  if (count == 0) return mcclSuccess;
  const size_t stride = count * mcclDataSize(dt);
  const int n = comm->nRanks, r = comm->rank;
  const size_t self = static_cast<size_t>(r) * stride;
  std::memcpy(static_cast<char*>(recvbuff) + self, static_cast<const char*>(sendbuff) + self, stride);
  if (n == 1) return mcclSuccess;

  mcclGroupStart();
  for (int p = 0; p < n; ++p) {
    if (p == r) continue;
    mcclSend(comm, static_cast<const char*>(sendbuff) + static_cast<size_t>(p) * stride, count, dt, p);
    mcclRecv(comm, static_cast<char*>(recvbuff) + static_cast<size_t>(p) * stride, count, dt, p);
  }
  return mcclSetLastError(mcclGroupEnd());
}

}

#ifdef MCCL_COLL_MAIN
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../include/alloc.h"

int main() {
  using namespace mccl;
  auto envI = [](const char* k, int d) { const char* v = std::getenv(k); return v ? std::atoi(v) : d; };
  const int    rank  = envI("MCCL_RANK", 0);
  const int    nn    = envI("MCCL_WORLD_SIZE", 1);
  const size_t count = static_cast<size_t>(envI("MCCL_COUNT", 4096));
  const char*  opn   = std::getenv("MCCL_COLL_OP");
  if (opn == nullptr) opn = "allreduce";

  mcclUniqueId id;
  mcclGetUniqueId(&id);
  mcclComm* comm = nullptr;
  if (mcclCommInitRank(&comm, nn, id, rank) != mcclSuccess) { std::printf("[coll] rank %d init FAIL\n", rank); return 1; }

  bool ok = false;
  if (std::strcmp(opn, "allgather") == 0) {
    void* sb = nullptr;
    void* rb = nullptr;
    mcclPageAlloc(count * sizeof(float), &sb);
    mcclPageAlloc(count * static_cast<size_t>(nn) * sizeof(float), &rb);
    float* s = static_cast<float*>(sb);
    for (size_t i = 0; i < count; ++i) s[i] = static_cast<float>(rank);
    const mcclResult rc = mcclAllGather(comm, sb, rb, count, mcclFloat32);
    float* rr = static_cast<float*>(rb);
    ok = (rc == mcclSuccess);
    for (int b = 0; b < nn && ok; ++b)
      for (size_t j = 0; j < count && ok; ++j)
        if (rr[static_cast<size_t>(b) * count + j] != static_cast<float>(b)) ok = false;
    std::printf("[allgather] rank %d/%d count=%zu slots=%d %s\n", rank, nn, count, nn, ok ? "OK" : "BAD");
    mcclPageFree(sb);
    mcclPageFree(rb);
  } else if (std::strcmp(opn, "reducescatter") == 0) {
    void* sb = nullptr;
    void* rb = nullptr;
    mcclPageAlloc(count * static_cast<size_t>(nn) * sizeof(float), &sb);
    mcclPageAlloc(count * sizeof(float), &rb);
    float* s = static_cast<float*>(sb);
    for (int b = 0; b < nn; ++b) for (size_t j = 0; j < count; ++j) s[static_cast<size_t>(b) * count + j] = static_cast<float>(b);
    const mcclResult rc = mcclReduceScatter(comm, sb, rb, count, mcclFloat32, mcclSum);
    float* rr = static_cast<float*>(rb);
    const float expect = static_cast<float>(rank * nn);
    ok = (rc == mcclSuccess);
    for (size_t j = 0; ok && j < count; ++j) if (rr[j] != expect) ok = false;
    std::printf("[reducescatter] rank %d/%d count=%zu recv[0]=%g expect=%g %s\n", rank, nn, count, rr[0], expect, ok ? "OK" : "BAD");
    mcclPageFree(sb);
    mcclPageFree(rb);
  } else if (std::strcmp(opn, "broadcast") == 0) {
    const int root = envI("MCCL_ROOT", 0);
    void* sb = nullptr;
    void* rb = nullptr;
    mcclPageAlloc(count * sizeof(float), &sb);
    mcclPageAlloc(count * sizeof(float), &rb);
    float* s = static_cast<float*>(sb);
    for (size_t i = 0; i < count; ++i) s[i] = 42.0f;
    const mcclResult rc = mcclBroadcast(comm, sb, rb, count, mcclFloat32, root);
    float* rr = static_cast<float*>(rb);
    ok = (rc == mcclSuccess);
    for (size_t i = 0; ok && i < count; ++i) if (rr[i] != 42.0f) ok = false;
    std::printf("[broadcast] rank %d/%d root=%d recv[0]=%g %s\n", rank, nn, root, rr[0], ok ? "OK" : "BAD");
    mcclPageFree(sb);
    mcclPageFree(rb);
  } else if (std::strcmp(opn, "reduce") == 0) {
    const int root = envI("MCCL_ROOT", 0);
    void* sb = nullptr;
    void* rb = nullptr;
    mcclPageAlloc(count * sizeof(float), &sb);
    mcclPageAlloc(count * sizeof(float), &rb);
    float* s = static_cast<float*>(sb);
    for (size_t i = 0; i < count; ++i) s[i] = static_cast<float>(rank + 1);
    const mcclResult rc = mcclReduce(comm, sb, rb, count, mcclFloat32, mcclSum, root);
    ok = (rc == mcclSuccess);
    if (rank == root) {
      float expect = 0;
      for (int i = 1; i <= nn; ++i) expect += i;
      float* rr = static_cast<float*>(rb);
      for (size_t i = 0; ok && i < count; ++i) if (rr[i] != expect) ok = false;
      std::printf("[reduce] rank %d/%d(root) recv[0]=%g expect=%g %s\n", rank, nn, static_cast<float*>(rb)[0], expect, ok ? "OK" : "BAD");
    } else {
      std::printf("[reduce] rank %d/%d participated rc=%d\n", rank, nn, static_cast<int>(rc));
    }
    mcclPageFree(sb);
    mcclPageFree(rb);
  } else if (std::strcmp(opn, "alltoall") == 0) {
    void* sb = nullptr;
    void* rb = nullptr;
    mcclPageAlloc(count * static_cast<size_t>(nn) * sizeof(float), &sb);
    mcclPageAlloc(count * static_cast<size_t>(nn) * sizeof(float), &rb);
    // Chunk bound for peer p carries this rank's id, tagged with p, so each side can verify provenance.
    float* s = static_cast<float*>(sb);
    for (int p = 0; p < nn; ++p) for (size_t j = 0; j < count; ++j) s[static_cast<size_t>(p) * count + j] = static_cast<float>(rank * nn + p);
    const mcclResult rc = mcclAllToAll(comm, sb, rb, count, mcclFloat32);
    float* rr = static_cast<float*>(rb);
    ok = (rc == mcclSuccess);
    // recv slot b holds the chunk rank b sent to us: value == b*nn + rank.
    for (int b = 0; b < nn && ok; ++b)
      for (size_t j = 0; j < count && ok; ++j)
        if (rr[static_cast<size_t>(b) * count + j] != static_cast<float>(b * nn + rank)) ok = false;
    std::printf("[alltoall] rank %d/%d count=%zu slots=%d %s\n", rank, nn, count, nn, ok ? "OK" : "BAD");
    mcclPageFree(sb);
    mcclPageFree(rb);
  } else {
    void* sb = nullptr;
    void* rb = nullptr;
    mcclPageAlloc(count * sizeof(float), &sb);
    mcclPageAlloc(count * sizeof(float), &rb);
    float* s = static_cast<float*>(sb);
    for (size_t i = 0; i < count; ++i) s[i] = static_cast<float>(rank + 1);
    const mcclResult rc = mcclAllReduce(comm, sb, rb, count, mcclFloat32, mcclSum);
    float expect = 0;
    for (int i = 1; i <= nn; ++i) expect += i;
    float* rr = static_cast<float*>(rb);
    ok = (rc == mcclSuccess);
    for (size_t i = 0; ok && i < count; ++i) if (rr[i] != expect) ok = false;
    std::printf("[allreduce] rank %d/%d count=%zu result[0]=%g expect=%g %s\n", rank, nn, count, rr[0], expect, ok ? "OK" : "BAD");
    mcclPageFree(sb);
    mcclPageFree(rb);
  }
  mcclCommDestroy(comm);
  return ok ? 0 : 1;
}
#endif
