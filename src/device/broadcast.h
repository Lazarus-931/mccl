#pragma once

#include <cstring>

#include "all_reduce.h"
#include "fanout.h"
#include "primitives.h"
#include "reduce_kernel.h"
#include "../include/checks.h"
#include "../include/comm.h"
#include "../transport/m2m.h"

namespace mccl {

// Ring broadcast: a chain root -> root+1 -> ... -> root-1; each rank receives, then forwards unless next is the root.
inline mcclResult ringBroadcast(mcclComm* comm, const void* sendbuff, void* recvbuff, size_t count, mcclDataType dt, int root) {
  const size_t esz = mcclDataSize(dt);
  if (esz == 0 || count == 0 || root < 0 || root >= comm->nRanks) return mcclInvalidArgument;
  const int n = comm->nRanks, r = comm->rank;
  const size_t bytes = count * esz;

  Primitives prims(comm, recvbuff, dt, mcclSum, 0);
  if (r == root) {
    if (sendbuff == nullptr) return mcclInvalidArgument;
    if (recvbuff != sendbuff) std::memcpy(recvbuff, sendbuff, bytes);
    return n > 1 ? prims.sendNext(recvbuff, bytes) : mcclSuccess;
  }
  MCCLCHECK(prims.recvPrev(recvbuff, bytes));
  if ((r + 1) % n != root) return prims.sendNext(recvbuff, bytes);
  return mcclSuccess;
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
