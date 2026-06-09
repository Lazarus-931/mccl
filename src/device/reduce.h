#pragma once

#include <cstring>
#include <set>

#include "all_reduce.h"
#include "primitives.h"
#include "reduce_kernel.h"
#include "../include/alloc.h"
#include "../include/checks.h"
#include "../include/comm.h"
#include "../transport/m2m.h"

namespace mccl {

// Tree reduce: fold every subtree up to the topology hub (no broadcast-down — reduce only needs the result at
// one rank), then if the user's root isn't the hub, the hub ships it there in a single hop. Half the traffic
// of riding all-reduce, which would broadcast the result to every rank.
inline mcclResult treeReduce(mcclComm* comm, const void* sendbuff, void* recvbuff, size_t count, mcclDataType dt, mcclRedOp op, int root) {
  const size_t esz = mcclDataSize(dt);
  if (esz == 0 || count == 0) return mcclInvalidArgument;
  const mcclRedOp redOp = (op == mcclAvg) ? mcclSum : op;
  const int me = comm->rank, n = comm->nRanks;
  const int hub = comm->graphs[MCCL_ALGO_TREE].order.empty() ? 0 : comm->graphs[MCCL_ALGO_TREE].order[0];
  if (me == root && recvbuff == nullptr) return mcclInvalidArgument;

  const bool keepHere = (me == hub && me == root);
  const bool leaf = comm->childConns.empty();  // n>1 means the hub always has children, so a leaf is never the hub
  void* work = recvbuff;
  mcclResult rc = mcclSuccess;
  if (leaf) {
    rc = mcclM2MSend(comm->parent, sendbuff, count * esz);  // up-phase is just "ship my input": no scratch, no copy
  } else {
    if (!keepHere) MCCLCHECK(mcclCommReserveScratch(comm, count * esz, &work));
    if (work != sendbuff) std::memcpy(work, sendbuff, count * esz);  // in-place reduce: src == dst, skip the copy

    Primitives prims(comm, work, dt, redOp, count * esz);
    rc = prims.ok() ? mcclSuccess : mcclSystemError;
    if (rc == mcclSuccess) rc = prims.reduceFromChildren(0, count);
    if (rc == mcclSuccess && me != hub) rc = prims.sendToParent(0, count);
  }

  if (rc == mcclSuccess && hub != root) {  // deliver the hub's sum to the user's root, one hop (find, not at: no throw on the worker thread)
    if (me == hub)  { rc = mcclEnsurePeerConns(comm, {root}); const auto it = comm->peerConns.find(root); if (rc == mcclSuccess) rc = (it != comm->peerConns.end()) ? mcclM2MSend(it->second, work, count * esz) : mcclInternalError; }
    if (me == root) { rc = mcclEnsurePeerConns(comm, {hub});  const auto it = comm->peerConns.find(hub);  if (rc == mcclSuccess) rc = (it != comm->peerConns.end()) ? mcclM2MRecv(it->second, recvbuff, count * esz) : mcclInternalError; }
  }
  if (rc == mcclSuccess && op == mcclAvg && me == root) rc = cpuScale(recvbuff, count, dt, 1.0 / n);
  return rc;
}

inline mcclResult reduceImpl(mcclComm* comm, const void* sendbuff, void* recvbuff, size_t count, mcclDataType dt, mcclRedOp op, int root, bool ring) {
  if (comm->nRanks == 1) {
    if (comm->rank == root) {
      if (recvbuff == nullptr) return mcclInvalidArgument;
      if (recvbuff != sendbuff) std::memcpy(recvbuff, sendbuff, count * mcclDataSize(dt));
    }
    return mcclSuccess;
  }
  if (!ring) return treeReduce(comm, sendbuff, recvbuff, count, dt, op, root);

  // Ring reduce rides all-reduce: ring already moves the bandwidth-optimal 2(n-1) chunks, so the only waste is
  // delivering to every rank instead of just root. A dedicated reduce-scatter + gather-to-root is a refinement.
  if (comm->rank == root) {
    if (recvbuff == nullptr) return mcclInvalidArgument;
    return ringAllReduce(comm, sendbuff, recvbuff, count, dt, op);
  }
  const size_t esz = mcclDataSize(dt);
  if (esz == 0 || count == 0) return mcclInvalidArgument;
  void* tmp = nullptr;
  MCCLCHECK(mcclCommReserveScratch(comm, count * esz, &tmp));
  return ringAllReduce(comm, sendbuff, tmp, count, dt, op);
}

}
