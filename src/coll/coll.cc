#include "../include/coll.h"

#include "../include/comm.h"
#include "../include/graph.h"
#include "../device/all_reduce.h"
#include "../device/all_gather.h"
#include "../device/reduce_scatter.h"
#include "../device/broadcast.h"
#include "../device/reduce.h"

namespace mccl {

mcclResult mcclAllReduce(mcclComm* comm, const void* sendbuff, void* recvbuff, size_t count, mcclDataType dt, mcclRedOp op) {
  if (comm == nullptr || sendbuff == nullptr || recvbuff == nullptr) return mcclSetLastError(mcclInvalidArgument);
  const int algo = mcclPickAlgo(comm, count * mcclDataSize(dt));
  return mcclSetLastError(mcclLaunch(comm, [=]() {
    return algo == MCCL_ALGO_RING ? ringAllReduce(comm, sendbuff, recvbuff, count, dt, op)
                                  : treeAllReduce(comm, sendbuff, recvbuff, count, dt, op);
  }));
}

mcclResult mcclAllGather(mcclComm* comm, const void* sendbuff, void* recvbuff, size_t sendcount, mcclDataType dt) {
  if (comm == nullptr || sendbuff == nullptr || recvbuff == nullptr) return mcclSetLastError(mcclInvalidArgument);
  const int algo = mcclPickAlgo(comm, sendcount * mcclDataSize(dt));
  return mcclSetLastError(mcclLaunch(comm, [=]() {
    return algo == MCCL_ALGO_RING ? ringAllGather(comm, sendbuff, recvbuff, sendcount, dt)
                                  : treeAllGather(comm, sendbuff, recvbuff, sendcount, dt);
  }));
}

mcclResult mcclReduceScatter(mcclComm* comm, const void* sendbuff, void* recvbuff, size_t recvcount, mcclDataType dt, mcclRedOp op) {
  if (comm == nullptr || sendbuff == nullptr || recvbuff == nullptr) return mcclSetLastError(mcclInvalidArgument);
  const int algo = mcclPickAlgo(comm, recvcount * mcclDataSize(dt));
  return mcclSetLastError(mcclLaunch(comm, [=]() {
    return algo == MCCL_ALGO_RING ? ringReduceScatter(comm, sendbuff, recvbuff, recvcount, dt, op)
                                  : treeReduceScatter(comm, sendbuff, recvbuff, recvcount, dt, op);
  }));
}

mcclResult mcclBroadcast(mcclComm* comm, const void* sendbuff, void* recvbuff, size_t count, mcclDataType dt, int root) {
  if (comm == nullptr || recvbuff == nullptr) return mcclSetLastError(mcclInvalidArgument);
  const int algo = mcclPickAlgo(comm, count * mcclDataSize(dt));
  return mcclSetLastError(mcclLaunch(comm, [=]() {
    return algo == MCCL_ALGO_RING ? ringBroadcast(comm, sendbuff, recvbuff, count, dt, root)
                                  : treeBroadcast(comm, sendbuff, recvbuff, count, dt, root);
  }));
}

mcclResult mcclReduce(mcclComm* comm, const void* sendbuff, void* recvbuff, size_t count, mcclDataType dt, mcclRedOp op, int root) {
  if (comm == nullptr || sendbuff == nullptr || root < 0 || root >= comm->nRanks) return mcclSetLastError(mcclInvalidArgument);
  const bool ring = mcclPickAlgo(comm, count * mcclDataSize(dt)) == MCCL_ALGO_RING;
  return mcclSetLastError(mcclLaunch(comm, [=]() { return reduceImpl(comm, sendbuff, recvbuff, count, dt, op, root, ring); }));
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
