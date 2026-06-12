#pragma once

#include <algorithm>
#include <cstring>
#include <thread>

#include "all_reduce.h"
#include "fanout.h"
#include "primitives.h"
#include "reduce_kernel.h"
#include "treeflow.h"
#include "../include/checks.h"
#include "../include/comm.h"
#include "../transport/m2m.h"

namespace mccl {

inline mcclResult directAllGather(mcclComm* comm, const void* sendbuff, void* recvbuff, size_t sendcount, mcclDataType dt) {
  const size_t esz = mcclDataSize(dt);
  if (esz == 0 || sendcount == 0) return mcclInvalidArgument;
  const int n = comm->nRanks, r = comm->rank;
  const size_t slot = sendcount * esz;
  char* rb = static_cast<char*>(recvbuff);
  if (sendbuff != rb + static_cast<size_t>(r) * slot) std::memcpy(rb + static_cast<size_t>(r) * slot, sendbuff, slot);
  if (n == 1) return mcclSuccess;

  std::set<int> peers;
  for (int p = 0; p < n; ++p) if (p != r) peers.insert(p);
  MCCLCHECK(mcclEnsurePeerConns(comm, peers));
  std::vector<int> ps(peers.begin(), peers.end());
  return mcclParallel(mcclFanoutPool(), 2 * ps.size(), [&](size_t k) -> mcclResult {
    const auto it = comm->peerConns.find(ps[k / 2]);
    if (it == comm->peerConns.end() || it->second == nullptr) return mcclInternalError;
    return (k & 1) ? mcclM2MSend(it->second, rb + static_cast<size_t>(r) * slot, slot)
                   : mcclM2MRecv(it->second, rb + static_cast<size_t>(ps[k / 2]) * slot, slot);
  });
}

inline mcclResult ringAllGatherLeg(mcclComm* comm, void* recvbuff, size_t sendcount, size_t legOff, size_t legCnt,
                                   mcclDataType dt, int dir, mcclM2M* prev, mcclM2M* next) {
  const int n = comm->nRanks, r = comm->rank;
  Primitives prims(comm, recvbuff, dt, mcclSum, 0);
  if (!prims.ok()) return mcclSystemError;
  prims.bindRing(prev, next, nullptr, 0);
  auto wrap = [n](int x) { return ((x % n) + n) % n; };
  auto off = [&](int slot) { return static_cast<size_t>(slot) * sendcount + legOff; };
  mcclResult rc = mcclSuccess;
  for (int i = 0; i < n - 1 && rc == mcclSuccess; ++i) {
    const int s = wrap(r - dir * i), d = wrap(r - dir * (i + 1));
    rc = prims.recvCopySend(off(s), legCnt, off(d), legCnt);
  }
  return rc;
}

inline mcclResult ringAllGather(mcclComm* comm, const void* sendbuff, void* recvbuff, size_t sendcount, mcclDataType dt) {
  const size_t esz = mcclDataSize(dt);
  if (esz == 0 || sendcount == 0) return mcclInvalidArgument;
  const int n = comm->nRanks, r = comm->rank;
  char* myslot = static_cast<char*>(recvbuff) + static_cast<size_t>(r) * sendcount * esz;
  if (sendbuff != myslot) std::memcpy(myslot, sendbuff, sendcount * esz);
  if (n == 1) return mcclSuccess;

  const bool dual = comm->nextB != nullptr && comm->prevB != nullptr && n >= 3 && sendcount >= 2 &&
                    static_cast<size_t>(n) * sendcount * esz >= 4 * static_cast<size_t>(mcclParamDualRingMinBytes());
  if (dual) {
    const size_t hA = sendcount / 2;
    mcclResult rB = mcclSuccess;
    std::thread t([&]() {
      rB = ringAllGatherLeg(comm, recvbuff, sendcount, hA, sendcount - hA, dt, -1, comm->nextB, comm->prevB);
    });
    const mcclResult rA = ringAllGatherLeg(comm, recvbuff, sendcount, 0, hA, dt, +1, comm->prev, comm->next);
    t.join();
    return rA != mcclSuccess ? rA : rB;
  }
  return ringAllGatherLeg(comm, recvbuff, sendcount, 0, sendcount, dt, +1, comm->prev, comm->next);
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
    const size_t nc = comm->childConns.size();
    std::vector<std::vector<int>> Sc(nc);
    std::vector<size_t> off(nc, 0);
    size_t tot = 0;
    for (size_t k = 0; k < nc; ++k) {
      Sc[k] = subtreeRanks(ch, comm->childRanks[k]);
      off[k] = tot;
      tot += Sc[k].size() * cb;
    }
    char* rbuf = nullptr;
    if (nc > 0) {
      void* rv = nullptr;
      MCCLCHECK(mcclCommReserveStaging(comm, tot, &rv));
      rbuf = static_cast<char*>(rv);
    }
    MCCLCHECK(forEachChild(nc, [&](size_t k) {
      return mcclM2MRecv(comm->childConns[k], rbuf + off[k], Sc[k].size() * cb);
    }));
    for (size_t k = 0; k < nc; ++k)
      for (size_t j = 0; j < Sc[k].size(); ++j)
        std::memcpy(px + posInSx(Sc[k][j]) * cb, rbuf + off[k] + j * cb, cb);
    if (comm->parent != nullptr) MCCLCHECK(mcclM2MSend(comm->parent, px, Sx.size() * cb));
    if (comm->parent == nullptr) std::memcpy(rb, px, cb * static_cast<size_t>(n));
    else MCCLCHECK(mcclM2MRecv(comm->parent, rb, cb * static_cast<size_t>(n)));
    return forEachChild(nc, [&](size_t k) {
      return mcclM2MSend(comm->childConns[k], rb, cb * static_cast<size_t>(n));
    });
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
