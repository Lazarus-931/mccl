#pragma once

#include <cstring>
#include <set>

#include "all_reduce.h"
#include "primitives.h"
#include "reduce_kernel.h"
#include "../graph/discover.h"
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
  if (rc == mcclSuccess && op == mcclAvg && me == root) rc = scaleBuf(recvbuff, count, dt, 1.0 / n);
  return rc;
}

inline mcclResult directReduce(mcclComm* comm, const void* sendbuff, void* recvbuff, size_t count, mcclDataType dt, mcclRedOp op, int root) {
  const size_t esz = mcclDataSize(dt);
  if (esz == 0 || count == 0) return mcclInvalidArgument;
  const int n = comm->nRanks, r = comm->rank;
  const mcclRedOp wireOp = (op == mcclAvg) ? mcclSum : op;
  const size_t bytes = count * esz;

  if (r != root) {
    MCCLCHECK(mcclEnsurePeerConns(comm, {root}));
    const auto it = comm->peerConns.find(root);
    if (it == comm->peerConns.end() || it->second == nullptr) return mcclInternalError;
    return mcclM2MSend(it->second, sendbuff, bytes);
  }

  std::set<int> peers;
  for (int p = 0; p < n; ++p) if (p != root) peers.insert(p);
  MCCLCHECK(mcclEnsurePeerConns(comm, peers));
  void* stg = nullptr;
  MCCLCHECK(mcclCommReserveStaging(comm, bytes * static_cast<size_t>(n), &stg));
  char* stgB = static_cast<char*>(stg);
  std::vector<int> ps(peers.begin(), peers.end());
  MCCLCHECK(mcclParallel(mcclFanoutPool(), ps.size(), [&](size_t k) -> mcclResult {
    const auto it = comm->peerConns.find(ps[k]);
    if (it == comm->peerConns.end() || it->second == nullptr) return mcclInternalError;
    return mcclM2MRecv(it->second, stgB + static_cast<size_t>(ps[k]) * bytes, bytes);
  }));
  std::memcpy(stgB + static_cast<size_t>(r) * bytes, sendbuff, bytes);
  std::memcpy(recvbuff, stgB, bytes);
  MCCLCHECK(reduceMulti(recvbuff, stgB + bytes, count, static_cast<size_t>(n - 1), count, dt, wireOp, false));
  return op == mcclAvg ? cpuScale(recvbuff, count, dt, 1.0 / n) : mcclSuccess;
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

  const size_t esz = mcclDataSize(dt);
  if (esz == 0 || count == 0) return mcclInvalidArgument;
  const int n = comm->nRanks, r = comm->rank;
  if (r == root && recvbuff == nullptr) return mcclInvalidArgument;

  auto allDirectToRoot = [&]() {
    for (int p = 0; p < n; ++p)
      if (p != root && !mcclTopoDirectLink(comm->system, p, root, nullptr, nullptr)) return false;
    return true;
  };
  if (count >= static_cast<size_t>(n) && allDirectToRoot()) {
    const mcclRedOp ringOp = (op == mcclAvg) ? mcclSum : op;
    void* work = recvbuff;
    if (r != root) MCCLCHECK(mcclCommReserveScratch(comm, count * esz, &work));
    if (work != sendbuff) std::memcpy(work, sendbuff, count * esz);

    std::set<int> peers;
    if (r == root) { for (int p = 0; p < n; ++p) if (p != root) peers.insert(p); }
    else peers.insert(root);
    MCCLCHECK(mcclEnsurePeerConns(comm, peers));

    const bool dual = comm->nextB != nullptr && comm->prevB != nullptr && n >= 3 &&
                      count * esz >= static_cast<size_t>(mcclParamDualRingMinBytes()) &&
                      count >= static_cast<size_t>(2 * n);
    const size_t hA = dual ? count / 2 : count, hB = count - hA;
    auto maxChunkEl = [&](size_t cnt) {
      size_t m = 0;
      for (int i = 0; i < n; ++i) m = std::max(m, chunkOffElems(cnt, n, i + 1) - chunkOffElems(cnt, n, i));
      return m;
    };
    const size_t maxA = maxChunkEl(hA), maxB = dual ? maxChunkEl(hB) : 0;
    void* stg = nullptr;
    MCCLCHECK(mcclCommReserveStaging(comm, (maxA + maxB) * esz, &stg));
    char* w = static_cast<char*>(work);
    auto wrap = [n](int x) { return ((x % n) + n) % n; };
    auto chunk = [&](int q, int dir, size_t* o, size_t* l) {
      const int c = wrap(q + dir);
      const size_t half = dir > 0 ? hA : hB;
      const size_t base = dir > 0 ? 0 : hA;
      const size_t co = chunkOffElems(half, n, c);
      *o = base + co;
      *l = chunkOffElems(half, n, c + 1) - co;
    };
    auto ship = [&](int dir) -> mcclResult {
      if (r != root) {
        const auto it = comm->peerConns.find(root);
        if (it == comm->peerConns.end() || it->second == nullptr) return mcclInternalError;
        size_t o, l;
        chunk(r, dir, &o, &l);
        return l > 0 ? mcclM2MSend(it->second, w + o * esz, l * esz) : mcclSuccess;
      }
      std::vector<int> ps(peers.begin(), peers.end());
      return mcclParallel(mcclFanoutPool(), ps.size(), [&](size_t k) -> mcclResult {
        const auto it = comm->peerConns.find(ps[k]);
        if (it == comm->peerConns.end() || it->second == nullptr) return mcclInternalError;
        size_t o, l;
        chunk(ps[k], dir, &o, &l);
        return l > 0 ? mcclM2MRecv(it->second, w + o * esz, l * esz) : mcclSuccess;
      });
    };
    mcclResult rc;
    if (dual) {
      mcclResult rB = mcclSuccess;
      std::thread t([&]() {
        rB = ringReducePhaseLeg(comm, w + hA * esz, hB, dt, ringOp, -1, comm->nextB, comm->prevB,
                                static_cast<char*>(stg) + maxA * esz, maxB * esz);
      });
      rc = ringReducePhaseLeg(comm, w, hA, dt, ringOp, +1, comm->prev, comm->next, stg, maxA * esz);
      if (rc == mcclSuccess) rc = ship(+1);
      t.join();
      if (rc == mcclSuccess) rc = rB;
      if (rc == mcclSuccess) rc = ship(-1);
    } else {
      rc = ringReducePhaseLeg(comm, w, hA, dt, ringOp, +1, comm->prev, comm->next, stg, maxA * esz);
      if (rc == mcclSuccess) rc = ship(+1);
    }
    if (rc != mcclSuccess) return rc;
    if (r == root && op == mcclAvg) MCCLCHECK(scaleBuf(recvbuff, count, dt, 1.0 / n));
    return mcclSuccess;
  }

  if (r == root) return ringAllReduce(comm, sendbuff, recvbuff, count, dt, op);
  void* tmp = nullptr;
  MCCLCHECK(mcclCommReserveScratch(comm, count * esz, &tmp));
  return ringAllReduce(comm, sendbuff, tmp, count, dt, op);
}

}
