#pragma once

#include <cstring>
#include <thread>

#include "all_reduce.h"
#include "fanout.h"
#include "primitives.h"
#include "reduce_kernel.h"
#include "../include/checks.h"
#include "../include/comm.h"
#include "../transport/m2m.h"

namespace mccl {

inline mcclResult ringBroadcastLeg(mcclComm* comm, char* base, size_t count, size_t esz, int root,
                                   mcclDataType dt, int dir, mcclM2M* prev, mcclM2M* next) {
  const int n = comm->nRanks, r = comm->rank;
  Primitives prims(comm, base, dt, mcclSum, 0);
  if (!prims.ok()) return mcclSystemError;
  prims.bindRing(prev, next, nullptr, 0);
  int K = static_cast<int>(mcclParamPipelineChunks());
  if (K < 1) K = 1;
  if (n <= 2 || count * esz < (1u << 20) || static_cast<size_t>(K) > count) K = 1;
  auto off = [&](int i) { return chunkOffElems(count, K, i) * esz; };
  auto len = [&](int i) { return (chunkOffElems(count, K, i + 1) - chunkOffElems(count, K, i)) * esz; };

  if (r == root) {
    for (int i = 0; i < K; ++i) MCCLCHECK(prims.sendNext(base + off(i), len(i)));
    return mcclSuccess;
  }
  const bool forward = (r + dir + 2 * n) % n != root;
  if (!forward) {
    for (int i = 0; i < K; ++i) MCCLCHECK(prims.recvPrev(base + off(i), len(i)));
    return mcclSuccess;
  }
  MCCLCHECK(prims.recvPrev(base + off(0), len(0)));
  for (int i = 0; i + 1 < K; ++i)
    MCCLCHECK(prims.sendRecv(base + off(i), len(i), base + off(i + 1), len(i + 1)));
  return prims.sendNext(base + off(K - 1), len(K - 1));
}

inline mcclResult ringBroadcast(mcclComm* comm, const void* sendbuff, void* recvbuff, size_t count, mcclDataType dt, int root) {
  const size_t esz = mcclDataSize(dt);
  if (esz == 0 || count == 0 || root < 0 || root >= comm->nRanks) return mcclInvalidArgument;
  const int n = comm->nRanks, r = comm->rank;
  const size_t bytes = count * esz;

  if (r == root) {
    if (sendbuff == nullptr) return mcclInvalidArgument;
    if (recvbuff != sendbuff) std::memcpy(recvbuff, sendbuff, bytes);
    if (n == 1) return mcclSuccess;
  }
  char* rb = static_cast<char*>(recvbuff);

  const bool dual = comm->nextB != nullptr && comm->prevB != nullptr && n >= 3 && count >= 2 &&
                    bytes >= 2 * static_cast<size_t>(mcclParamDualRingMinBytes());
  if (dual) {
    const size_t hA = count / 2, hB = count - hA;
    mcclResult rB = mcclSuccess;
    std::thread t([&]() {
      rB = ringBroadcastLeg(comm, rb + hA * esz, hB, esz, root, dt, -1, comm->nextB, comm->prevB);
    });
    const mcclResult rA = ringBroadcastLeg(comm, rb, hA, esz, root, dt, +1, comm->prev, comm->next);
    t.join();
    return rA != mcclSuccess ? rA : rB;
  }
  return ringBroadcastLeg(comm, rb, count, esz, root, dt, +1, comm->prev, comm->next);
}

inline mcclResult treeBroadcast(mcclComm* comm, const void* sendbuff, void* recvbuff, size_t count, mcclDataType dt, int root) {
  const size_t esz = mcclDataSize(dt);
  if (esz == 0 || count == 0 || root < 0 || root >= comm->nRanks) return mcclInvalidArgument;
  const int n = comm->nRanks;
  const size_t bytes = count * esz;
  const bool isRoot = comm->rank == root;
  if (isRoot) {
    if (sendbuff == nullptr) return mcclInvalidArgument;
    if (recvbuff != sendbuff) std::memcpy(recvbuff, sendbuff, bytes);
  }
  if (n == 1) return mcclSuccess;

  // Flat (hub) tree: the data reaches the hub (one hop if the root is a leaf) and the hub pushes it to every
  // child at once — pure data, no reduce. Deep tree: masked sum (zero non-root, then a tree all-reduce).
  if (!comm->chan.flatTree) {
    if (!isRoot) std::memset(recvbuff, 0, bytes);
    return treeAllReduce(comm, recvbuff, recvbuff, count, dt, mcclSum);
  }

  if (comm->parent != nullptr) {  // leaf: a leaf root first pushes its data up to the hub, then everyone receives
    if (isRoot) MCCLCHECK(mcclM2MSend(comm->parent, recvbuff, bytes));
    return mcclM2MRecv(comm->parent, recvbuff, bytes);
  }

  if (!isRoot) {  // hub, but the data originated at a leaf: pull it from that leaf before pushing down
    int idx = -1;
    for (size_t k = 0; k < comm->childRanks.size(); ++k) if (comm->childRanks[k] == root) idx = static_cast<int>(k);
    if (idx < 0) return mcclInternalError;
    MCCLCHECK(mcclM2MRecv(comm->childConns[static_cast<size_t>(idx)], recvbuff, bytes));
  }
  return forEachChild(comm->childConns.size(), [&](size_t k) {
    return mcclM2MSend(comm->childConns[k], recvbuff, bytes);
  });
}

}
