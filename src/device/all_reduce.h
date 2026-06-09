#pragma once

#include <algorithm>
#include <cstring>
#include <thread>

#include "primitives.h"
#include "reduce_kernel.h"
#include "../include/comm.h"
#include "../transport/m2m.h"

namespace mccl {

// Ring all-reduce = reduce-scatter (each step sends one chunk, receives + reduces another) then all-gather.
inline mcclResult ringAllReduce(mcclComm* comm, const void* sendbuff, void* recvbuff, size_t count, mcclDataType dt, mcclRedOp op) {
  const size_t esz = mcclDataSize(dt);
  if (esz == 0 || count == 0) return mcclInvalidArgument;
  const int n = comm->nRanks, r = comm->rank;
  const mcclRedOp ringOp = (op == mcclAvg) ? mcclSum : op;  // sum on the wire, scale to the average at the end
  if (sendbuff != recvbuff) std::memcpy(recvbuff, sendbuff, count * esz);
  if (n == 1) return mcclSuccess;

  size_t maxEl = 0;
  for (int i = 0; i < n; ++i) maxEl = std::max(maxEl, chunkOffElems(count, n, i + 1) - chunkOffElems(count, n, i));
  Primitives prims(comm, recvbuff, dt, ringOp, maxEl * esz);
  if (!prims.ok()) return mcclSystemError;

  mcclResult rc = mcclSuccess;
  for (int i = 0; i < n - 1 && rc == mcclSuccess; ++i) {
    const int s = (r - i + n) % n, d = (r - i - 1 + n) % n;
    rc = prims.recvReduceSend(chunkOffElems(count, n, s), chunkOffElems(count, n, s + 1) - chunkOffElems(count, n, s),
                              chunkOffElems(count, n, d), chunkOffElems(count, n, d + 1) - chunkOffElems(count, n, d));
  }
  for (int i = 0; i < n - 1 && rc == mcclSuccess; ++i) {
    const int s = (r - i + 1 + n) % n, d = (r - i + n) % n;
    rc = prims.recvCopySend(chunkOffElems(count, n, s), chunkOffElems(count, n, s + 1) - chunkOffElems(count, n, s),
                            chunkOffElems(count, n, d), chunkOffElems(count, n, d + 1) - chunkOffElems(count, n, d));
  }
  if (rc == mcclSuccess && op == mcclAvg) rc = cpuScale(recvbuff, count, dt, 1.0 / n);
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
    if (rc == mcclSuccess && op == mcclAvg) rc = cpuScale(recvbuff, count, dt, 1.0 / comm->nRanks);
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
  if (rc == mcclSuccess && op == mcclAvg) rc = cpuScale(recvbuff, count, dt, 1.0 / comm->nRanks);
  return rc;
}

}
