#pragma once

#include <algorithm>
#include <cstring>
#include <set>
#include <thread>
#include <vector>

#include "primitives.h"
#include "reduce_kernel.h"
#include "../include/checks.h"
#include "../include/comm.h"
#include "../transport/m2m.h"

namespace mccl {

inline mcclResult directAllReduce(mcclComm* comm, const void* sendbuff, void* recvbuff, size_t count, mcclDataType dt, mcclRedOp op) {
  const size_t esz = mcclDataSize(dt);
  if (esz == 0 || count == 0) return mcclInvalidArgument;
  const int n = comm->nRanks, r = comm->rank;
  const mcclRedOp wireOp = (op == mcclAvg) ? mcclSum : op;
  const size_t bytes = count * esz;
  if (n == 1) {
    if (sendbuff != recvbuff) std::memcpy(recvbuff, sendbuff, bytes);
    return mcclSuccess;
  }

  std::set<int> peers;
  for (int p = 0; p < n; ++p) if (p != r) peers.insert(p);
  MCCLCHECK(mcclEnsurePeerConns(comm, peers));
  void* stg = nullptr;
  MCCLCHECK(mcclCommReserveStaging(comm, bytes * static_cast<size_t>(n), &stg));

  std::vector<mcclM2M*> conns;
  std::vector<int> peerRank;
  conns.reserve(static_cast<size_t>(n - 1));
  peerRank.reserve(static_cast<size_t>(n - 1));
  for (int p : peers) {
    const auto it = comm->peerConns.find(p);
    if (it == comm->peerConns.end() || it->second == nullptr) return mcclInternalError;
    conns.push_back(it->second);
    peerRank.push_back(p);
  }
  // Slot k holds rank k's buffer (own rank included) so the fold below visits contributions in
  // ascending-rank order on EVERY rank. Folding starting from the local buffer instead makes the
  // fp association differ per rank (rank 2 computes (x2+x0)+x1 while rank 0 computes (x0+x1)+x2),
  // which leaves ranks' results a ULP apart and lets long DDP runs drift.
  char* stgB = static_cast<char*>(stg);
  const mcclResult rc = mcclParallel(mcclFanoutPool(), 2 * static_cast<size_t>(n - 1), [&](size_t k) {
    mcclM2M* c = conns[k / 2];
    return (k & 1) ? mcclM2MSend(c, sendbuff, bytes)
                   : mcclM2MRecv(c, stgB + static_cast<size_t>(peerRank[k / 2]) * bytes, bytes);
  });
  if (rc != mcclSuccess) return rc;

  std::memcpy(stgB + static_cast<size_t>(r) * bytes, sendbuff, bytes);
  std::memcpy(recvbuff, stgB, bytes);
  MCCLCHECK(reduceMulti(recvbuff, stgB + bytes, count, static_cast<size_t>(n - 1), count, dt, wireOp, false));
  return op == mcclAvg ? cpuScale(recvbuff, count, dt, 1.0 / n) : mcclSuccess;
}


inline mcclResult ringAllReduceLeg(mcclComm* comm, void* buf, size_t count, mcclDataType dt, mcclRedOp op,
                                   int dir, mcclM2M* prev, mcclM2M* next, void* stg, size_t stgStride) {
  const int n = comm->nRanks, r = comm->rank;
  Primitives prims(comm, buf, dt, op, 0);
  if (!prims.ok()) return mcclSystemError;
  prims.bindRing(prev, next, stg, stgStride);
  auto wrap = [n](int x) { return ((x % n) + n) % n; };
  auto off = [&](int c) { return chunkOffElems(count, n, c); };
  auto len = [&](int c) { return chunkOffElems(count, n, c + 1) - chunkOffElems(count, n, c); };
  mcclResult rc = mcclSuccess;
  for (int i = 0; i < n - 1 && rc == mcclSuccess; ++i) {
    const int s = wrap(r - dir * i), d = wrap(r - dir * (i + 1));
    rc = prims.recvReduceSend(off(s), len(s), off(d), len(d));
  }
  for (int i = 0; i < n - 1 && rc == mcclSuccess; ++i) {
    const int s = wrap(r - dir * (i - 1)), d = wrap(r - dir * i);
    rc = prims.recvCopySend(off(s), len(s), off(d), len(d));
  }
  return rc;
}


inline mcclResult ringAllReduce(mcclComm* comm, const void* sendbuff, void* recvbuff, size_t count, mcclDataType dt, mcclRedOp op) {
  const size_t esz = mcclDataSize(dt);
  if (esz == 0 || count == 0) return mcclInvalidArgument;
  const int n = comm->nRanks;
  const mcclRedOp ringOp = (op == mcclAvg) ? mcclSum : op;  // sum on the wire, scale to the average at the end
  if (sendbuff != recvbuff) std::memcpy(recvbuff, sendbuff, count * esz);
  if (n == 1) return mcclSuccess;

  auto maxChunkEl = [&](size_t cnt) {
    size_t m = 0;
    for (int i = 0; i < n; ++i) m = std::max(m, chunkOffElems(cnt, n, i + 1) - chunkOffElems(cnt, n, i));
    return m;
  };

  const bool dual = comm->nextB != nullptr && comm->prevB != nullptr && n >= 3 &&
                    count * esz >= static_cast<size_t>(mcclParamDualRingMinBytes()) &&
                    count >= static_cast<size_t>(2 * n);
  mcclResult rc;
  if (dual) {
    const size_t hA = count / 2, hB = count - hA;
    const size_t maxA = maxChunkEl(hA), maxB = maxChunkEl(hB);
    void* stg = nullptr;
    if (mcclCommReserveStaging(comm, (maxA + maxB) * esz, &stg) != mcclSuccess) return mcclSystemError;
    mcclResult rB = mcclSuccess;
    std::thread t([&]() {
      rB = ringAllReduceLeg(comm, static_cast<char*>(recvbuff) + hA * esz, hB, dt, ringOp, -1,
                            comm->nextB, comm->prevB, static_cast<char*>(stg) + maxA * esz, maxB * esz);
    });
    const mcclResult rA = ringAllReduceLeg(comm, recvbuff, hA, dt, ringOp, +1,
                                           comm->prev, comm->next, stg, maxA * esz);
    t.join();
    rc = rA != mcclSuccess ? rA : rB;
  } else {
    const size_t maxEl = maxChunkEl(count);
    void* stg = nullptr;
    if (mcclCommReserveStaging(comm, maxEl * esz, &stg) != mcclSuccess) return mcclSystemError;
    rc = ringAllReduceLeg(comm, recvbuff, count, dt, ringOp, +1, comm->prev, comm->next, stg, maxEl * esz);
  }
  if (rc == mcclSuccess && op == mcclAvg) rc = scaleBuf(recvbuff, count, dt, 1.0 / n);
  return rc;
}

// Tree all-reduce: fold children up to the parent, then push the full result back down. Each Mac is a tree
// node; the root holds the reduced buffer first.
inline mcclResult treeAllReduce(mcclComm* comm, const void* sendbuff, void* recvbuff, size_t count, mcclDataType dt, mcclRedOp op) {
  const size_t esz = mcclDataSize(dt);
  if (esz == 0 || count == 0) return mcclInvalidArgument;
  const mcclRedOp ringOp = (op == mcclAvg) ? mcclSum : op;
  if (comm->nRanks == 1) {
    if (sendbuff != recvbuff) std::memcpy(recvbuff, sendbuff, count * esz);
    return mcclSuccess;
  }
  // A pure leaf sends straight from sendbuff and recvFromParent later overwrites recvbuff, so the upfront
  // copy is dead work there (a full extra buffer sweep on every non-hub rank of the default flat tree).
  // dtree ranks are interior in one of the two trees, so they keep the copy.
  const bool leaf = !comm->chan.dtree && comm->childConns.empty();
  if (sendbuff != recvbuff && !leaf) std::memcpy(recvbuff, sendbuff, count * esz);

  if (comm->chan.dtree) {
    // Double tree: reduce the first half over tree A and the second over tree B concurrently. The trees use
    // disjoint connections (peerConns vs peerConnsB) and disjoint staging, so the two threads never collide,
    // and because every Mac is a leaf in one tree but interior in the other, both directions of each link stay busy.
    const size_t hA = count / 2, hB = count - hA;
    const size_t sliceA = (comm->childConns.empty()  ? 1 : comm->childConns.size())  * hA * esz;
    const size_t sliceB = (comm->childConnsB.empty() ? 1 : comm->childConnsB.size()) * hB * esz;
    void* base = nullptr;
    if (mcclCommReserveStaging(comm, sliceA + sliceB, &base) != mcclSuccess) return mcclSystemError;
    Primitives pA(comm, recvbuff, dt, ringOp, 0);
    Primitives pB(comm, recvbuff, dt, ringOp, 0);
    if (!pA.ok() || !pB.ok()) return mcclSystemError;
    pA.bindTree(comm->parent,  &comm->childConns,  base, hA * esz ? hA * esz : esz);
    pB.bindTree(comm->parentB, &comm->childConnsB, static_cast<char*>(base) + sliceA, hB * esz ? hB * esz : esz);
    auto oneTree = [&](Primitives& p, size_t off, size_t cnt) -> mcclResult {
      if (cnt == 0) return mcclSuccess;
      mcclResult r = p.reduceFromChildren(off, cnt); if (r != mcclSuccess) return r;
      r = p.sendToParent(off, cnt);                  if (r != mcclSuccess) return r;
      r = p.recvFromParent(off, cnt);                if (r != mcclSuccess) return r;
      return p.sendToChildren(off, cnt);
    };
    mcclResult rB = mcclSuccess;
    std::thread t([&]() { rB = oneTree(pB, hA, hB); });
    const mcclResult rA = oneTree(pA, 0, hA);
    t.join();
    mcclResult rc = rA != mcclSuccess ? rA : rB;
    if (rc == mcclSuccess && op == mcclAvg) rc = scaleBuf(recvbuff, count, dt, 1.0 / comm->nRanks);
    return rc;
  }

  // Single tree, pipelined: split into K chunks and overlap chunk i's down with chunk i+1's up, so the parent
  // link carries both directions at once (TB is full-duplex). K=1 is the plain two-phase tree.
  int K = static_cast<int>(mcclParamPipelineChunks());
  if (K < 1) K = 1;
  if (count * esz < (1u << 20) || static_cast<size_t>(K) > count) K = 1;  // pipelining only pays off on big buffers
  size_t maxChunk = 0;
  for (int i = 0; i < K; ++i) maxChunk = std::max(maxChunk, chunkOffElems(count, K, i + 1) - chunkOffElems(count, K, i));
  Primitives prims(comm, recvbuff, dt, ringOp, maxChunk * esz);
  if (!prims.ok()) return mcclSystemError;

  auto chunk = [&](int i, size_t* off, size_t* cnt) { *off = chunkOffElems(count, K, i); *cnt = chunkOffElems(count, K, i + 1) - *off; };
  auto up = [&](int i) -> mcclResult {
    size_t off, cnt; chunk(i, &off, &cnt);
    if (leaf) return mcclM2MSend(comm->parent, static_cast<const char*>(sendbuff) + off * esz, cnt * esz);
    const mcclResult r = prims.reduceFromChildren(off, cnt);
    return r == mcclSuccess ? prims.sendToParent(off, cnt) : r;
  };
  auto down = [&](int i) -> mcclResult {
    size_t off, cnt; chunk(i, &off, &cnt);
    const mcclResult r = prims.recvFromParent(off, cnt);
    return r == mcclSuccess ? prims.sendToChildren(off, cnt) : r;
  };

  mcclResult rc = up(0);
  for (int i = 0; i < K && rc == mcclSuccess; ++i) {
    mcclResult urc = mcclSuccess;
    std::thread t;
    if (i + 1 < K) t = std::thread([&, i]() { urc = up(i + 1); });
    const mcclResult drc = down(i);
    if (t.joinable()) t.join();
    rc = drc != mcclSuccess ? drc : urc;
  }
  if (rc == mcclSuccess && op == mcclAvg) rc = scaleBuf(recvbuff, count, dt, 1.0 / comm->nRanks);
  return rc;
}

}
