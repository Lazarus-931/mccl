#include "include/coll.h"

#include "include/checks.h"
#include "include/comm.h"
#include "graph/discover.h"
#include "transport/m2m.h"

#include <poll.h>

#include <map>
#include <set>
#include <thread>
#include <utility>
#include <vector>

namespace mccl {

namespace {

struct P2pOp { mcclComm* comm; bool isSend; int peer; void* buf; size_t bytes; };

struct GroupState {
  int                depth = 0;
  mcclResult         error = mcclSuccess;
  std::vector<P2pOp> ops;
};
thread_local GroupState g_group;

mcclResult groupFail(mcclResult rc) {
  if (g_group.depth > 0 && g_group.error == mcclSuccess) g_group.error = rc;
  return mcclSetLastError(rc);
}

uint32_t peerDialIp(const mcclComm* c, int peer) {
  uint32_t ip = 0;
  mcclTopoDirectLink(c->system, c->rank, peer, nullptr, &ip);
  return ip;
}

constexpr uint32_t kConnShared = 0;
constexpr uint32_t kConnP2p    = 1;
struct P2pHello {
  int32_t  rank;
  uint32_t kind;
};

mcclResult acceptRoute(mcclComm* comm, uint32_t* kind, int* peerRank) {
  mcclM2M* conn = nullptr;
  MCCLCHECK(mcclM2MAccept(comm->listener, &conn));
  P2pHello h{};
  const mcclResult rc = mcclM2MRecv(conn, &h, sizeof(h));
  if (rc != mcclSuccess || h.rank < 0 || h.rank >= comm->nRanks) { mcclM2MClose(conn); return rc != mcclSuccess ? rc : mcclInternalError; }
  std::lock_guard<std::mutex> lk(comm->connMu);
  std::map<int, mcclM2M*>& target = h.kind == kConnP2p ? comm->p2pIn : comm->peerConns;
  mcclM2M*& slot = target[h.rank];
  if (slot == nullptr) slot = conn;
  else mcclM2MClose(conn);
  *kind = h.kind;
  *peerRank = h.rank;
  return mcclSuccess;
}

mcclResult dialPeer(mcclComm* comm, int peer, uint32_t kind, std::map<int, mcclM2M*>& target) {
  char ip[16];
  mcclIpStr(peerDialIp(comm, peer), ip);
  mcclM2M* conn = nullptr;
  MCCLCHECK(mcclM2MConnect(ip, comm->peerPorts[static_cast<size_t>(peer)], static_cast<m2mType>(comm->m2mType), &conn));
  const P2pHello h{comm->rank, kind};
  const mcclResult rc = mcclM2MSend(conn, &h, sizeof(h));
  if (rc != mcclSuccess) { mcclM2MClose(conn); return rc; }
  std::lock_guard<std::mutex> lk(comm->connMu);
  mcclM2M*& slot = target[peer];
  if (slot == nullptr) slot = conn;
  else mcclM2MClose(conn);
  return mcclSuccess;
}

mcclM2M* lookupConn(mcclComm* comm, int peer, bool isSend) {
  std::lock_guard<std::mutex> lk(comm->connMu);
  std::map<int, mcclM2M*>& dir = isSend ? comm->p2pOut : comm->p2pIn;
  const auto it = dir.find(peer);
  return it != dir.end() ? it->second : nullptr;
}

mcclResult p2pPrepare(mcclComm* comm, const std::vector<P2pOp>& ops) {
  std::set<int> dialTo;
  std::set<int> awaitFrom;
  for (const P2pOp& o : ops) {
    if (lookupConn(comm, o.peer, o.isSend) != nullptr) continue;
    if (o.isSend) dialTo.insert(o.peer);
    else          awaitFrom.insert(o.peer);
  }
  for (int p : dialTo) MCCLCHECK(dialPeer(comm, p, kConnP2p, comm->p2pOut));
  while (!awaitFrom.empty()) {
    uint32_t kind = 0;
    int peer = -1;
    MCCLCHECK(acceptRoute(comm, &kind, &peer));
    if (kind == kConnP2p) awaitFrom.erase(peer);
  }
  return mcclSuccess;
}


mcclResult runBatch(mcclComm* comm, std::vector<P2pOp> ops) {
  for (const P2pOp& o : ops)
    if (o.peer < 0 || o.peer >= comm->nRanks || o.peer == comm->rank) return mcclInvalidArgument;
  MCCLCHECK(p2pPrepare(comm, ops));

  std::map<int, std::vector<const P2pOp*>> buckets;
  for (const P2pOp& o : ops) buckets[o.peer * 2 + (o.isSend ? 1 : 0)].push_back(&o);

  auto runBucket = [comm](const std::vector<const P2pOp*>& vec) -> mcclResult {
    for (const P2pOp* o : vec) {
      mcclM2M* conn = lookupConn(comm, o->peer, o->isSend);
      if (conn == nullptr) return mcclInternalError;
      const mcclResult r = o->isSend ? mcclM2MSend(conn, o->buf, o->bytes) : mcclM2MRecv(conn, o->buf, o->bytes);
      if (r != mcclSuccess) return r;
    }
    return mcclSuccess;
  };
  if (buckets.size() <= 1)
    return buckets.empty() ? mcclSuccess : runBucket(buckets.begin()->second);

  std::vector<std::thread> ts;
  std::vector<mcclResult>  rc(buckets.size(), mcclSuccess);
  int bi = 0;
  for (auto& kv : buckets) {
    const std::vector<const P2pOp*>* vec = &kv.second;
    const int idx = bi++;
    ts.emplace_back([vec, idx, &rc, &runBucket]() { rc[static_cast<size_t>(idx)] = runBucket(*vec); });
  }
  for (std::thread& t : ts) t.join();
  for (mcclResult r : rc) if (r != mcclSuccess) return r;
  return mcclSuccess;
}

}

// Deadlock-free: the lower rank dials, the higher accepts. The accept thread loops until every EXPECTED peer
// has arrived — never a fixed count, because an early dial from some other lower-ranked peer (for a later op)
// can land first; such a connection is legitimate, so it is kept and merged too, it just doesn't satisfy this
// call. Accepted conns are collected locally and merged after the join; map inserts take comm->connMu so a
// concurrent mcclCommAbort can iterate the maps safely.
mcclResult mcclEnsurePeerConnsInto(mcclComm* comm, const std::set<int>& peers, std::map<int, mcclM2M*>& target) {
  std::vector<int> toDial;
  std::set<int> awaiting;
  for (int p : peers) {
    if (p == comm->rank || target.count(p)) continue;
    if (p > comm->rank) toDial.push_back(p);
    else                awaiting.insert(p);
  }
  if (toDial.empty() && awaiting.empty()) return mcclSuccess;

  std::vector<std::pair<int, mcclM2M*>> accepted;
  mcclResult arc = mcclSuccess;
  std::thread acc;
  if (!awaiting.empty()) acc = std::thread([&]() {
    std::set<int> want = awaiting;
    while (!want.empty()) {
      mcclM2M* conn = nullptr;
      if ((arc = mcclM2MAccept(comm->listener, &conn)) != mcclSuccess) return;
      P2pHello h{};
      if ((arc = mcclM2MRecv(conn, &h, sizeof(h))) != mcclSuccess) { mcclM2MClose(conn); return; }
      if (h.kind == kConnP2p) {
        std::lock_guard<std::mutex> lk(comm->connMu);
        mcclM2M*& slot = comm->p2pIn[h.rank];
        if (slot == nullptr) slot = conn;
        else mcclM2MClose(conn);
        continue;
      }
      accepted.push_back({h.rank, conn});
      want.erase(h.rank);
    }
  });
  mcclResult crc = mcclSuccess;
  for (int p : toDial) {
    char ip[16];
    mcclIpStr(peerDialIp(comm, p), ip);
    mcclM2M* conn = nullptr;
    if ((crc = mcclM2MConnect(ip, comm->peerPorts[static_cast<size_t>(p)], static_cast<m2mType>(comm->m2mType), &conn)) != mcclSuccess) break;
    const P2pHello h{comm->rank, kConnShared};
    if ((crc = mcclM2MSend(conn, &h, sizeof(h))) != mcclSuccess) { mcclM2MClose(conn); break; }
    std::lock_guard<std::mutex> lk(comm->connMu);
    target[p] = conn;
  }
  if (acc.joinable()) acc.join();
  {
    std::lock_guard<std::mutex> lk(comm->connMu);
    for (auto& kv : accepted) {
      mcclM2M*& slot = target[kv.first];
      if (slot == nullptr) slot = kv.second;
      else mcclM2MClose(kv.second);  // duplicate dial for an already-known peer: keep the existing conn
    }
  }
  return crc != mcclSuccess ? crc : arc;
}

mcclResult mcclEnsurePeerConns(mcclComm* comm, const std::set<int>& peers) {
  return mcclEnsurePeerConnsInto(comm, peers, comm->peerConns);
}

mcclResult mcclSend(mcclComm* comm, const void* sendbuff, size_t count, mcclDataType dt, int peer) {
  if (comm == nullptr || sendbuff == nullptr || peer < 0 || peer >= comm->nRanks || peer == comm->rank)
    return groupFail(mcclInvalidArgument);
  const P2pOp op{comm, true, peer, const_cast<void*>(sendbuff), count * mcclDataSize(dt)};
  if (g_group.depth > 0) { g_group.ops.push_back(op); return mcclSuccess; }
  return mcclSetLastError(mcclLaunch(comm, [comm, op]() { return runBatch(comm, {op}); }));
}

mcclResult mcclRecv(mcclComm* comm, void* recvbuff, size_t count, mcclDataType dt, int peer) {
  if (comm == nullptr || recvbuff == nullptr || peer < 0 || peer >= comm->nRanks || peer == comm->rank)
    return groupFail(mcclInvalidArgument);
  const P2pOp op{comm, false, peer, recvbuff, count * mcclDataSize(dt)};
  if (g_group.depth > 0) { g_group.ops.push_back(op); return mcclSuccess; }
  return mcclSetLastError(mcclLaunch(comm, [comm, op]() { return runBatch(comm, {op}); }));
}

mcclResult mcclRecvAny(mcclComm* comm, void* recvbuff, size_t count, mcclDataType dt, int* srcRank) {
  if (comm == nullptr || recvbuff == nullptr || srcRank == nullptr || count == 0 || mcclDataSize(dt) == 0 || comm->nRanks < 2)
    return groupFail(mcclInvalidArgument);
  if (g_group.depth > 0) return groupFail(mcclInvalidUsage);
  if (comm->m2mType == M2M_TYPE_RDMA) return mcclSetLastError(mcclInvalidUsage);
  const size_t bytes = count * mcclDataSize(dt);
  return mcclSetLastError(mcclLaunch(comm, [comm, recvbuff, bytes, srcRank]() -> mcclResult {
    constexpr int kCap = 32;
    std::set<mcclM2M*> dead;
    for (;;) {
      if (comm->aborted) return mcclSystemError;
      std::vector<std::pair<int, mcclM2M*>> conns;
      {
        std::lock_guard<std::mutex> lk(comm->connMu);
        for (auto& kv : comm->p2pIn) if (kv.second != nullptr && dead.count(kv.second) == 0) conns.push_back(kv);
      }
      std::vector<pollfd> pfds;
      std::vector<int> owner;
      for (size_t i = 0; i < conns.size(); ++i) {
        int fds[kCap];
        const int nf = mcclM2MFds(conns[i].second, fds, kCap);
        for (int k = 0; k < nf; ++k) {
          pfds.push_back({fds[k], POLLIN, 0});
          owner.push_back(static_cast<int>(i));
        }
      }
      const int lfd = mcclM2MListenerFd(comm->listener);
      if (lfd >= 0) {
        pfds.push_back({lfd, POLLIN, 0});
        owner.push_back(-1);
      }
      if (lfd < 0 && pfds.empty()) return mcclSystemError;
      const int pr = ::poll(pfds.data(), static_cast<nfds_t>(pfds.size()), 1000);
      if (pr < 0) {
        if (errno == EINTR) continue;
        return mcclSystemError;
      }
      if (pr == 0) continue;
      int ready = -2;
      bool newConn = false;
      for (size_t k = 0; k < pfds.size(); ++k) {
        if (pfds[k].revents == 0) continue;
        if (owner[k] == -1) { newConn = true; continue; }
        if (pfds[k].revents & POLLIN) { ready = owner[k]; break; }
        if (pfds[k].revents & (POLLHUP | POLLERR | POLLNVAL)) dead.insert(conns[static_cast<size_t>(owner[k])].second);
      }
      if (ready >= 0) {
        const mcclResult rc = mcclM2MRecv(conns[static_cast<size_t>(ready)].second, recvbuff, bytes);
        if (rc != mcclSuccess) return rc;
        *srcRank = conns[static_cast<size_t>(ready)].first;
        return mcclSuccess;
      }
      if (newConn) {
        uint32_t kind = 0;
        int peer = -1;
        MCCLCHECK(acceptRoute(comm, &kind, &peer));
      }
    }
  }));
}

mcclResult mcclGroupStart() { ++g_group.depth; return mcclSuccess; }

mcclResult mcclGroupAbort() {
  if (g_group.depth <= 0) return mcclSetLastError(mcclInvalidUsage);
  g_group.depth = 0;
  g_group.error = mcclSuccess;
  g_group.ops.clear();
  return mcclSuccess;
}

mcclResult mcclGroupEnd() {
  if (g_group.depth <= 0) return mcclSetLastError(mcclInvalidUsage);
  if (--g_group.depth > 0) return mcclSuccess;
  if (g_group.error != mcclSuccess) {
    const mcclResult rc = g_group.error;
    g_group.error = mcclSuccess;
    g_group.ops.clear();
    return mcclSetLastError(rc);
  }
  std::map<mcclComm*, std::vector<P2pOp>> byComm;
  for (const P2pOp& o : g_group.ops) byComm[o.comm].push_back(o);
  g_group.ops.clear();
  // All comms' batches launch CONCURRENTLY: byComm's pointer order differs across processes, so launching
  // blocking batches sequentially deadlocks the moment rank A runs comm1-then-comm2 while rank B runs
  // comm2-then-comm1 with transfers too big for the socket buffers. One thread per comm — group calls are
  // rare and the comm count is small; the per-comm batch itself fans out through the pools.
  std::vector<std::pair<mcclComm*, std::vector<P2pOp>>> batches;
  batches.reserve(byComm.size());
  for (auto& kv : byComm) batches.push_back({kv.first, std::move(kv.second)});
  std::vector<mcclResult> rcs(batches.size(), mcclSuccess);
  // The launched lambda owns its ops (value capture): on a non-blocking comm it outlives this frame on the worker.
  auto launchOne = [](std::pair<mcclComm*, std::vector<P2pOp>>& b) {
    mcclComm* c = b.first;
    return mcclLaunch(c, [c, ops = std::move(b.second)]() { return runBatch(c, ops); });
  };
  std::vector<std::thread> ts;
  for (size_t i = 1; i < batches.size(); ++i)
    ts.emplace_back([&, i]() { rcs[i] = launchOne(batches[i]); });
  if (!batches.empty()) rcs[0] = launchOne(batches[0]);
  for (std::thread& t : ts) t.join();
  for (mcclResult r : rcs) if (r != mcclSuccess) return mcclSetLastError(r);
  return mcclSuccess;
}

}
