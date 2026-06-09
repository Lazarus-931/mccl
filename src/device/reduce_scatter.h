#pragma once

#include <algorithm>
#include <cstring>

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

inline mcclResult ringReduceScatter(mcclComm* comm, const void* sendbuff, void* recvbuff, size_t recvcount, mcclDataType dt, mcclRedOp op) {
  const size_t esz = mcclDataSize(dt);
  if (esz == 0 || recvcount == 0) return mcclInvalidArgument;
  const int n = comm->nRanks, r = comm->rank;
  const mcclRedOp ringOp = (op == mcclAvg) ? mcclSum : op;
  const size_t blk = recvcount * esz;
  if (n == 1) { std::memcpy(recvbuff, sendbuff, blk); return mcclSuccess; }

  void* scratch = nullptr;
  MCCLCHECK(mcclCommReserveScratch(comm, blk * 3, &scratch));
  char* recvS   = static_cast<char*>(scratch);
  char* fold[2] = {recvS + blk, recvS + 2 * blk};
  const char* sb = static_cast<const char*>(sendbuff);

  Primitives prims(comm, recvbuff, dt, ringOp, 0);
  mcclResult rc = mcclSuccess;
  for (int i = 0; i < n - 1 && rc == mcclSuccess; ++i) {
    const int sIdx = (r - i - 1 + 2 * n) % n;
    const int dIdx = (r - i - 2 + 2 * n) % n;
    const void* sp = (i == 0) ? sb + static_cast<size_t>(sIdx) * blk : fold[(i - 1) & 1];
    rc = prims.sendRecv(sp, blk, recvS, blk);
    if (rc != mcclSuccess) break;
    void* dst = (i == n - 2) ? recvbuff : static_cast<void*>(fold[i & 1]);
    rc = reduceOut(dst, sb + static_cast<size_t>(dIdx) * blk, recvS, recvcount, dt, ringOp);
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

    void* pc = nullptr;
    if (!comm->childConns.empty()) MCCLCHECK(mcclCommReserveStaging(comm, blk * static_cast<size_t>(n), &pc));
    for (size_t k = 0; k < comm->childConns.size(); ++k) {
      const std::vector<int> Sc = subtreeRanks(ch, comm->childRanks[k]);
      for (size_t j = 0; j < Sc.size(); ++j) {
        const size_t pos = static_cast<size_t>(std::lower_bound(Sx.begin(), Sx.end(), Sc[j]) - Sx.begin());
        std::memcpy(static_cast<char*>(pc) + j * blk, px + pos * blk, blk);
      }
      MCCLCHECK(mcclM2MSend(comm->childConns[k], pc, Sc.size() * blk));
    }
    return mcclSuccess;
  }

  if (comm->parent != nullptr) {
    MCCLCHECK(mcclM2MSend(comm->parent, sendbuff, blk * static_cast<size_t>(n)));
    return mcclM2MRecv(comm->parent, recvbuff, blk);
  }

  const size_t nc = comm->childConns.size();
  void* work = nullptr;
  void* stg  = nullptr;  // root uses no Primitives, so staging is free for the per-child gather (both arenas page-aligned for the Metal fold)
  MCCLCHECK(mcclCommReserveScratch(comm, blk * static_cast<size_t>(n), &work));
  MCCLCHECK(mcclCommReserveStaging(comm, blk * static_cast<size_t>(n) * nc, &stg));
  std::memcpy(work, sendbuff, blk * static_cast<size_t>(n));
  const bool gpu = pageAligned(work);  // Metal availability is probed lazily in reduceMulti, past the size gate
  const size_t whole = recvcount * static_cast<size_t>(n);

  mcclResult res = forEachChild(nc, [&](size_t k) {  // root: gather every leaf's full input (one link each), fold in one pass, scatter blocks
    return mcclM2MRecv(comm->childConns[k], static_cast<char*>(stg) + k * blk * static_cast<size_t>(n), blk * static_cast<size_t>(n));
  });
  if (res == mcclSuccess) res = reduceMulti(work, stg, whole, nc, whole, dt, redOp, gpu);
  if (res == mcclSuccess && op == mcclAvg) res = scaleBuf(work, whole, dt, 1.0 / n);
  if (res == mcclSuccess) std::memcpy(recvbuff, static_cast<char*>(work) + static_cast<size_t>(r) * blk, blk);
  if (res == mcclSuccess)
    res = forEachChild(nc, [&](size_t k) {
      return mcclM2MSend(comm->childConns[k], static_cast<char*>(work) + static_cast<size_t>(comm->childRanks[k]) * blk, blk);
    });
  return res;
}

}
