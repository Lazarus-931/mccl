#pragma once

#include <algorithm>
#include <cstring>
#include <thread>

#include "all_reduce.h"
#include "fanout.h"
#include "primitives.h"
#include "reduce_kernel.h"
#include "treeflow.h"
#include "../include/alloc.h"
#include "../include/checks.h"
#include "../include/comm.h"
#include "../transport/m2m.h"

namespace mccl {

inline mcclResult directReduceScatter(mcclComm* comm, const void* sendbuff, void* recvbuff, size_t recvcount, mcclDataType dt, mcclRedOp op) {
  const size_t esz = mcclDataSize(dt);
  if (esz == 0 || recvcount == 0) return mcclInvalidArgument;
  const int n = comm->nRanks, r = comm->rank;
  const mcclRedOp wireOp = (op == mcclAvg) ? mcclSum : op;
  const size_t blk = recvcount * esz;
  const char* sb = static_cast<const char*>(sendbuff);
  if (n == 1) { std::memcpy(recvbuff, sendbuff, blk); return mcclSuccess; }

  std::set<int> peers;
  for (int p = 0; p < n; ++p) if (p != r) peers.insert(p);
  MCCLCHECK(mcclEnsurePeerConns(comm, peers));
  void* stg = nullptr;
  MCCLCHECK(mcclCommReserveStaging(comm, blk * static_cast<size_t>(n), &stg));
  char* stgB = static_cast<char*>(stg);
  std::vector<int> ps(peers.begin(), peers.end());
  MCCLCHECK(mcclParallel(mcclFanoutPool(), 2 * ps.size(), [&](size_t k) -> mcclResult {
    const auto it = comm->peerConns.find(ps[k / 2]);
    if (it == comm->peerConns.end() || it->second == nullptr) return mcclInternalError;
    return (k & 1) ? mcclM2MSend(it->second, sb + static_cast<size_t>(ps[k / 2]) * blk, blk)
                   : mcclM2MRecv(it->second, stgB + static_cast<size_t>(ps[k / 2]) * blk, blk);
  }));
  std::memcpy(stgB + static_cast<size_t>(r) * blk, sb + static_cast<size_t>(r) * blk, blk);
  std::memcpy(recvbuff, stgB, blk);
  MCCLCHECK(reduceMulti(recvbuff, stgB + blk, recvcount, static_cast<size_t>(n - 1), recvcount, dt, wireOp, false));
  return op == mcclAvg ? scaleBuf(recvbuff, recvcount, dt, 1.0 / n) : mcclSuccess;
}

inline mcclResult ringReduceScatterLeg(mcclComm* comm, const void* sendbuff, void* recvbuff, size_t recvcount,
                                       size_t legOff, size_t legCnt, mcclDataType dt, mcclRedOp op,
                                       int dir, mcclM2M* prev, mcclM2M* next, char* scratch) {
  const int n = comm->nRanks, r = comm->rank;
  const size_t esz = mcclDataSize(dt);
  const size_t blk = recvcount * esz;
  const size_t legBytes = legCnt * esz;
  char* recvS   = scratch;
  char* fold[2] = {scratch + legBytes, scratch + 2 * legBytes};
  const char* sb = static_cast<const char*>(sendbuff) + legOff * esz;
  char* out = static_cast<char*>(recvbuff) + legOff * esz;

  Primitives prims(comm, recvbuff, dt, op, 0);
  if (!prims.ok()) return mcclSystemError;
  prims.bindRing(prev, next, nullptr, 0);
  auto wrap = [n](int x) { return ((x % n) + n) % n; };
  mcclResult rc = mcclSuccess;
  for (int i = 0; i < n - 1 && rc == mcclSuccess; ++i) {
    const int sIdx = wrap(r - dir * (i + 1));
    const int dIdx = wrap(r - dir * (i + 2));
    const void* sp = (i == 0) ? sb + static_cast<size_t>(sIdx) * blk : fold[(i - 1) & 1];
    rc = prims.sendRecv(sp, legBytes, recvS, legBytes);
    if (rc != mcclSuccess) break;
    void* dst = (i == n - 2) ? static_cast<void*>(out) : static_cast<void*>(fold[i & 1]);
    rc = reduceOut(dst, sb + static_cast<size_t>(dIdx) * blk, recvS, legCnt, dt, op);
  }
  return rc;
}

inline mcclResult ringReduceScatter(mcclComm* comm, const void* sendbuff, void* recvbuff, size_t recvcount, mcclDataType dt, mcclRedOp op) {
  const size_t esz = mcclDataSize(dt);
  if (esz == 0 || recvcount == 0) return mcclInvalidArgument;
  const int n = comm->nRanks;
  const mcclRedOp ringOp = (op == mcclAvg) ? mcclSum : op;
  const size_t blk = recvcount * esz;
  if (n == 1) { std::memcpy(recvbuff, sendbuff, blk); return mcclSuccess; }

  void* scratch = nullptr;
  MCCLCHECK(mcclCommReserveScratch(comm, blk * 3, &scratch));

  const bool dual = comm->nextB != nullptr && comm->prevB != nullptr && n >= 3 && recvcount >= 2 &&
                    blk * static_cast<size_t>(n) >= 4 * static_cast<size_t>(mcclParamDualRingMinBytes());
  mcclResult rc;
  if (dual) {
    const size_t hA = recvcount / 2, hB = recvcount - hA;
    char* sA = static_cast<char*>(scratch);
    char* sB = sA + 3 * hA * esz;
    mcclResult rB = mcclSuccess;
    std::thread t([&]() {
      rB = ringReduceScatterLeg(comm, sendbuff, recvbuff, recvcount, hA, hB, dt, ringOp, -1, comm->nextB, comm->prevB, sB);
    });
    const mcclResult rA = ringReduceScatterLeg(comm, sendbuff, recvbuff, recvcount, 0, hA, dt, ringOp, +1, comm->prev, comm->next, sA);
    t.join();
    rc = rA != mcclSuccess ? rA : rB;
  } else {
    rc = ringReduceScatterLeg(comm, sendbuff, recvbuff, recvcount, 0, recvcount, dt, ringOp, +1, comm->prev, comm->next, static_cast<char*>(scratch));
  }
  if (rc == mcclSuccess && op == mcclAvg) rc = scaleBuf(recvbuff, recvcount, dt, 1.0 / n);
  return rc;
}

inline mcclResult treeReduceScatter(mcclComm* comm, const void* sendbuff, void* recvbuff, size_t recvcount, mcclDataType dt, mcclRedOp op) {
  const size_t esz = mcclDataSize(dt);
  if (esz == 0 || recvcount == 0) return mcclInvalidArgument;
  const int n = comm->nRanks, r = comm->rank;
  const mcclRedOp redOp = (op == mcclAvg) ? mcclSum : op;
  const size_t blk = recvcount * esz;
  if (n == 1) { std::memcpy(recvbuff, sendbuff, blk); return mcclSuccess; }

  // Flat (hub) tree: leaves send their full input up, the root reduces everything and scatters each rank only
  // its own block back — O(1) blocks down, not O(n). Deep tree: masked sum (tree all-reduce, keep my slice).
  if (!comm->chan.flatTree) {
    // Deep tree: reduce every subtree up to the hub (full n-block buffer up each link, as the masked-sum did),
    // then scatter each rank only its own block back down — the down carries just each subtree's blocks instead
    // of masked-sum's full O(n) buffer, so the saving is on the down leg.
    const size_t whole = recvcount * static_cast<size_t>(n);
    void* work = nullptr;
    MCCLCHECK(mcclCommReserveScratch(comm, blk * static_cast<size_t>(n), &work));
    std::memcpy(work, sendbuff, blk * static_cast<size_t>(n));
    Primitives prims(comm, work, dt, redOp, blk * static_cast<size_t>(n));
    if (!prims.ok()) return mcclSystemError;
    MCCLCHECK(prims.reduceFromChildren(0, whole));
    MCCLCHECK(prims.sendToParent(0, whole));

    const std::vector<std::vector<int>> ch = treeChildren(comm->chan.treeParent);
    const std::vector<int> Sx = subtreeRanks(ch, r);
    char* px = static_cast<char*>(work);  // hub: work is the full buffer (sorted-all == rank order); else recv packed Sx here
    if (comm->parent != nullptr) MCCLCHECK(mcclM2MRecv(comm->parent, px, Sx.size() * blk));
    const size_t myPos = static_cast<size_t>(std::lower_bound(Sx.begin(), Sx.end(), r) - Sx.begin());
    std::memcpy(recvbuff, px + myPos * blk, blk);
    if (op == mcclAvg) MCCLCHECK(scaleBuf(recvbuff, recvcount, dt, 1.0 / n));

    const size_t nc = comm->childConns.size();
    std::vector<std::vector<int>> Sc(nc);
    std::vector<size_t> off(nc, 0);
    size_t tot = 0;
    for (size_t k = 0; k < nc; ++k) {
      Sc[k] = subtreeRanks(ch, comm->childRanks[k]);
      off[k] = tot;
      tot += Sc[k].size() * blk;
    }
    char* pc = nullptr;
    if (nc > 0) {
      void* pv = nullptr;
      MCCLCHECK(mcclCommReserveStaging(comm, tot, &pv));
      pc = static_cast<char*>(pv);
    }
    for (size_t k = 0; k < nc; ++k)
      for (size_t j = 0; j < Sc[k].size(); ++j) {
        const size_t pos = static_cast<size_t>(std::lower_bound(Sx.begin(), Sx.end(), Sc[k][j]) - Sx.begin());
        std::memcpy(pc + off[k] + j * blk, px + pos * blk, blk);
      }
    return forEachChild(nc, [&](size_t k) {
      return mcclM2MSend(comm->childConns[k], pc + off[k], Sc[k].size() * blk);
    });
  }

  const size_t whole = recvcount * static_cast<size_t>(n);
  const size_t total = blk * static_cast<size_t>(n);
  const size_t chunkBytes = static_cast<size_t>(mcclParamHubFoldChunkBytes());
  const int nch = total <= chunkBytes ? 1 : static_cast<int>((total + chunkBytes - 1) / chunkBytes);
  auto off = [&](int i) { return chunkOffElems(whole, nch, i); };
  auto len = [&](int i) { return chunkOffElems(whole, nch, i + 1) - chunkOffElems(whole, nch, i); };

  if (comm->parent != nullptr) {
    for (int i = 0; i < nch; ++i)
      MCCLCHECK(mcclM2MSend(comm->parent, static_cast<const char*>(sendbuff) + off(i) * esz, len(i) * esz));
    return mcclM2MRecv(comm->parent, recvbuff, blk);
  }

  const size_t nc = comm->childConns.size();
  size_t maxEl = 0;
  for (int i = 0; i < nch; ++i) maxEl = std::max(maxEl, len(i));
  void* work = nullptr;
  void* stg  = nullptr;  // root uses no Primitives, so staging is free for the per-child gather (both arenas page-aligned for the Metal fold)
  MCCLCHECK(mcclCommReserveScratch(comm, total, &work));
  MCCLCHECK(mcclCommReserveStaging(comm, 2 * maxEl * esz * nc, &stg));
  std::memcpy(work, sendbuff, total);

  auto recvChunk = [&](int i) {
    char* half = static_cast<char*>(stg) + static_cast<size_t>(i & 1) * maxEl * esz * nc;
    return forEachChild(nc, [&](size_t k) {
      return mcclM2MRecv(comm->childConns[k], half + k * maxEl * esz, len(i) * esz);
    });
  };
  mcclResult res = nc == 0 ? mcclSuccess : recvChunk(0);
  for (int i = 0; i < nch && nc > 0 && res == mcclSuccess; ++i) {
    mcclResult nres = mcclSuccess;
    std::thread t;
    if (i + 1 < nch) t = std::thread([&, i]() { nres = recvChunk(i + 1); });
    char* dst = static_cast<char*>(work) + off(i) * esz;
    res = reduceMulti(dst, static_cast<char*>(stg) + static_cast<size_t>(i & 1) * maxEl * esz * nc,
                      len(i), nc, maxEl, dt, redOp, pageAligned(dst));
    if (t.joinable()) t.join();
    if (res == mcclSuccess) res = nres;
  }
  if (res == mcclSuccess && op == mcclAvg) res = scaleBuf(work, whole, dt, 1.0 / n);
  if (res == mcclSuccess) std::memcpy(recvbuff, static_cast<char*>(work) + static_cast<size_t>(r) * blk, blk);
  if (res == mcclSuccess)
    res = forEachChild(nc, [&](size_t k) {
      return mcclM2MSend(comm->childConns[k], static_cast<char*>(work) + static_cast<size_t>(comm->childRanks[k]) * blk, blk);
    });
  return res;
}

}
