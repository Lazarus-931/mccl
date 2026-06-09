#pragma once

#include <algorithm>
#include <cstring>

#include "all_reduce.h"
#include "fanout.h"
#include "primitives.h"
#include "reduce_kernel.h"
#include "treeflow.h"
#include "../include/checks.h"
#include "../include/comm.h"
#include "../transport/m2m.h"

namespace mccl {

// Ring all-gather: place my slot, then circulate slots around the ring — send one while receiving the next.
inline mcclResult ringAllGather(mcclComm* comm, const void* sendbuff, void* recvbuff, size_t sendcount, mcclDataType dt) {
  const size_t esz = mcclDataSize(dt);
  if (esz == 0 || sendcount == 0) return mcclInvalidArgument;
  const int n = comm->nRanks, r = comm->rank;
  char* myslot = static_cast<char*>(recvbuff) + static_cast<size_t>(r) * sendcount * esz;
  if (sendbuff != myslot) std::memcpy(myslot, sendbuff, sendcount * esz);  // in-place: slot already in recvbuff
  if (n == 1) return mcclSuccess;

  Primitives prims(comm, recvbuff, dt, mcclSum, 0);
  mcclResult rc = mcclSuccess;
  for (int i = 0; i < n - 1 && rc == mcclSuccess; ++i) {
    const int s = (r - i + n) % n, d = (r - i - 1 + n) % n;
    rc = prims.recvCopySend(static_cast<size_t>(s) * sendcount, sendcount, static_cast<size_t>(d) * sendcount, sendcount);
  }
  return rc;
}

inline mcclResult treeAllGather(mcclComm* comm, const void* sendbuff, void* recvbuff, size_t sendcount, mcclDataType dt) {
  const size_t esz = mcclDataSize(dt);
  if (esz == 0 || sendcount == 0) return mcclInvalidArgument;
  const int n = comm->nRanks, r = comm->rank;
  const size_t cb = sendcount * esz;
  char* rb = static_cast<char*>(recvbuff);
  if (sendbuff != rb + static_cast<size_t>(r) * cb) std::memcpy(rb + static_cast<size_t>(r) * cb, sendbuff, cb);  // in-place: slot already in place
  if (n == 1) return mcclSuccess;

  // Flat (hub) tree: each leaf sends only its own slot up and the root scatters the whole buffer back down.
  // Deep tree: gather each subtree's slots up to the hub (the up carries only subtree slots, not the full
  // buffer like a masked-sum all-reduce would), then broadcast the assembled buffer back down.
  if (!comm->chan.flatTree) {
    const std::vector<std::vector<int>> ch = treeChildren(comm->chan.treeParent);
    const std::vector<int> Sx = subtreeRanks(ch, r);
    const auto posInSx = [&](int q) { return static_cast<size_t>(std::lower_bound(Sx.begin(), Sx.end(), q) - Sx.begin()); };
    void* pxv = nullptr;
    MCCLCHECK(mcclCommReserveScratch(comm, cb * static_cast<size_t>(n), &pxv));
    char* px = static_cast<char*>(pxv);
    std::memcpy(px + posInSx(r) * cb, rb + static_cast<size_t>(r) * cb, cb);
    void* rbuf = nullptr;
    if (!comm->childConns.empty()) MCCLCHECK(mcclCommReserveStaging(comm, cb * static_cast<size_t>(n), &rbuf));
    for (size_t k = 0; k < comm->childConns.size(); ++k) {
      const std::vector<int> Sc = subtreeRanks(ch, comm->childRanks[k]);
      MCCLCHECK(mcclM2MRecv(comm->childConns[k], rbuf, Sc.size() * cb));
      for (size_t j = 0; j < Sc.size(); ++j)
        std::memcpy(px + posInSx(Sc[j]) * cb, static_cast<char*>(rbuf) + j * cb, cb);
    }
    if (comm->parent != nullptr) MCCLCHECK(mcclM2MSend(comm->parent, px, Sx.size() * cb));
    if (comm->parent == nullptr) std::memcpy(rb, px, cb * static_cast<size_t>(n));
    else MCCLCHECK(mcclM2MRecv(comm->parent, rb, cb * static_cast<size_t>(n)));
    for (size_t k = 0; k < comm->childConns.size(); ++k)
      MCCLCHECK(mcclM2MSend(comm->childConns[k], rb, cb * static_cast<size_t>(n)));
    return mcclSuccess;
  }

  if (comm->parent != nullptr) {  // leaf: send my slot up, receive the whole gathered buffer back
    MCCLCHECK(mcclM2MSend(comm->parent, rb + static_cast<size_t>(r) * cb, cb));
    return mcclM2MRecv(comm->parent, rb, cb * static_cast<size_t>(n));
  }
  const size_t nc = comm->childConns.size();
  const mcclResult grc = forEachChild(nc, [&](size_t k) {
    return mcclM2MRecv(comm->childConns[k], rb + static_cast<size_t>(comm->childRanks[k]) * cb, cb);
  });
  if (grc != mcclSuccess) return grc;
  return forEachChild(nc, [&](size_t k) {
    return mcclM2MSend(comm->childConns[k], rb, cb * static_cast<size_t>(n));
  });
}

}
