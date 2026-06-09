#include "include/coll.h"

#include "include/checks.h"
#include "include/comm.h"
#include "graph/discover.h"
#include "transport/m2m.h"

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
  std::vector<P2pOp> ops;
};
thread_local GroupState g_group;

// Directly-reachable IP: a direct TB link, else the peer's LAN. NOT a multi-hop path's first hop, which
// would resolve to the relay between us and the peer, not the peer itself.
uint32_t peerDialIp(const mcclComm* c, int peer) {
  uint32_t ip = 0;
  mcclTopoDirectLink(c->system, c->rank, peer, nullptr, &ip);
  return ip;
}

mcclResult runBatch(mcclComm* comm, std::vector<P2pOp> ops) {
  std::set<int> peers;
  for (const P2pOp& o : ops) {
    if (o.peer < 0 || o.peer >= comm->nRanks || o.peer == comm->rank) return mcclInvalidArgument;
    peers.insert(o.peer);
  }
  MCCLCHECK(mcclEnsurePeerConns(comm, peers));

  std::map<int, std::vector<const P2pOp*>> buckets;
  for (const P2pOp& o : ops) buckets[o.peer * 2 + (o.isSend ? 1 : 0)].push_back(&o);

  auto runBucket = [comm](const std::vector<const P2pOp*>& vec) -> mcclResult {
    for (const P2pOp* o : vec) {
      const auto it = comm->peerConns.find(o->peer);  // find, not at(): a throw on a worker thread would std::terminate
      if (it == comm->peerConns.end() || it->second == nullptr) return mcclInternalError;
      const mcclResult r = o->isSend ? mcclM2MSend(it->second, o->buf, o->bytes) : mcclM2MRecv(it->second, o->buf, o->bytes);
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
      int pr = -1;
      if ((arc = mcclM2MRecv(conn, &pr, sizeof(pr))) != mcclSuccess) { mcclM2MClose(conn); return; }
      accepted.push_back({pr, conn});
      want.erase(pr);
    }
  });
  mcclResult crc = mcclSuccess;
  for (int p : toDial) {
    char ip[16];
    mcclIpStr(peerDialIp(comm, p), ip);
    mcclM2M* conn = nullptr;
    if ((crc = mcclM2MConnect(ip, comm->peerPorts[static_cast<size_t>(p)], static_cast<m2mType>(comm->m2mType), &conn)) != mcclSuccess) break;
    if ((crc = mcclM2MSend(conn, &comm->rank, sizeof(comm->rank))) != mcclSuccess) { mcclM2MClose(conn); break; }
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
    return mcclSetLastError(mcclInvalidArgument);
  const P2pOp op{comm, true, peer, const_cast<void*>(sendbuff), count * mcclDataSize(dt)};
  if (g_group.depth > 0) { g_group.ops.push_back(op); return mcclSuccess; }
  return mcclSetLastError(mcclLaunch(comm, [comm, op]() { return runBatch(comm, {op}); }));
}

mcclResult mcclRecv(mcclComm* comm, void* recvbuff, size_t count, mcclDataType dt, int peer) {
  if (comm == nullptr || recvbuff == nullptr || peer < 0 || peer >= comm->nRanks || peer == comm->rank)
    return mcclSetLastError(mcclInvalidArgument);
  const P2pOp op{comm, false, peer, recvbuff, count * mcclDataSize(dt)};
  if (g_group.depth > 0) { g_group.ops.push_back(op); return mcclSuccess; }
  return mcclSetLastError(mcclLaunch(comm, [comm, op]() { return runBatch(comm, {op}); }));
}

mcclResult mcclGroupStart() { ++g_group.depth; return mcclSuccess; }

mcclResult mcclGroupEnd() {
  if (g_group.depth <= 0) return mcclSetLastError(mcclInvalidUsage);
  if (--g_group.depth > 0) return mcclSuccess;
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
