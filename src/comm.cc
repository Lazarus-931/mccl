#include "include/comm.h"

#include "include/alloc.h"
#include "include/checks.h"
#include "include/coll.h"
#include "include/graph.h"
#include "include/param.h"
#include "bootstrap.h"
#include "init.h"
#include "graph/discover.h"
#include "graph/topo.h"
#include "graph/xml.h"
#include "misc/utils.h"
#include "transport/m2m.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace mccl {

namespace {
DEFINE_MCCL_PARAM(TbGen, "TB_GEN", 4);
constexpr size_t kAlgoPickBytes = 1u << 20;

struct IdData {
  char     ip[64];
  uint16_t port;
};

// All-gathered in one bootstrap round: each Mac's interfaces, its data-listener port, and chip strength.
struct RankInfo {
  mcclIfSet ifs;
  uint16_t  dataPort;
  uint16_t  pad;
  int       gpuCores;
  int       chipCap;
  uint64_t  umaBytes;
};

thread_local mcclResult g_lastError = mcclSuccess;
}

DEFINE_MCCL_PARAM(DualRings, "DUAL_RINGS", 1);
DEFINE_MCCL_PARAM(DualRingMinBytes, "DUAL_RING_MIN_BYTES", int64_t{1} << 20);

mcclResult mcclGetUniqueId(mcclUniqueId* id) {
  if (id == nullptr) return mcclInvalidArgument;
  IdData d{};
  const char* ip = std::getenv("MCCL_BOOTSTRAP_IP");
  const char* p  = std::getenv("MCCL_BOOTSTRAP_PORT");
  std::snprintf(d.ip, sizeof(d.ip), "%s", ip ? ip : "127.0.0.1");
  d.port = static_cast<uint16_t>(p ? std::atoi(p) : 53700);
  static_assert(sizeof(IdData) <= sizeof(id->internal), "mcclUniqueId too small");
  std::memset(id->internal, 0, sizeof(id->internal));
  std::memcpy(id->internal, &d, sizeof(d));
  return mcclSuccess;
}

mcclResult mcclCommInitRankConfig(mcclComm** out, int nRanks, mcclUniqueId commId, int rank, const mcclConfig* config) {
  if (out == nullptr || nRanks < 1 || rank < 0 || rank >= nRanks) return mcclSetLastError(mcclInvalidArgument);
  *out = nullptr;
  IdData d{};
  std::memcpy(&d, commId.internal, sizeof(d));
  const char*    rootIp   = d.ip;
  const uint16_t rootPort = d.port;

  // Resolve the transport up front and refuse here if it's unavailable, so every rank fails together
  // instead of one dialer erroring while its peer blocks the full accept timeout.
  const char* trc = config != nullptr ? config->transport : nullptr;
  const int wantM2M = (trc != nullptr) ? (std::strcmp(trc, "rdma") == 0 ? M2M_TYPE_RDMA : M2M_TYPE_TCP) : m2mTypeSelect();
  if (!mcclM2MAvailable(static_cast<m2mType>(wantM2M))) return mcclSetLastError(mcclInvalidUsage);

  RankInfo me{};
  MCCLCHECK(mcclDiscoverInterfaces(&me.ifs));
  const ChipInfo chip = mcclDiscoverChip();
  me.gpuCores = chip.gpuCores;
  me.chipCap  = chip.chipCap;
  me.umaBytes = chip.unifiedMemBytes;
  mcclM2MListener* lst = nullptr;
  uint16_t myPort = 0;
  MCCLCHECK(mcclM2MListenStart(0, &lst, &myPort));
  me.dataPort = myPort;

  std::vector<RankInfo> infos(static_cast<size_t>(nRanks));
  mcclResult rc = mcclBootstrapAllGather(rootIp, rootPort, rank, nRanks, &me, infos.data(), sizeof(RankInfo));
  if (rc != mcclSuccess) { mcclM2MListenClose(lst); return rc; }

  std::vector<mcclIfSet> sets(static_cast<size_t>(nRanks));
  std::vector<uint16_t>  ports(static_cast<size_t>(nRanks));
  for (int i = 0; i < nRanks; ++i) { sets[static_cast<size_t>(i)] = infos[i].ifs; ports[static_cast<size_t>(i)] = infos[i].dataPort; }

  std::vector<mcclEdge> edges;
  std::vector<uint32_t> lanIp;
  mcclBuildEdges(sets.data(), nRanks, &edges, &lanIp);
  rc = mcclProbeLiveness(edges, rank, nRanks, rootIp, rootPort, static_cast<uint16_t>(rootPort + 1));
  if (rc != mcclSuccess) { mcclM2MListenClose(lst); return rc; }

  auto* comm = new mcclComm();
  comm->rank = rank;
  comm->nRanks = nRanks;
  comm->device = 0;
  if (config != nullptr) comm->config = *config;
  comm->m2mType = wantM2M;
  std::snprintf(comm->rootIp, sizeof(comm->rootIp), "%s", rootIp);
  comm->rootPort = rootPort;
  comm->listener = lst;
  comm->peerPorts = ports;

  rc = mcclTopoGetSystem(nRanks, edges.data(), static_cast<int>(edges.size()),
                         lanIp.empty() ? nullptr : lanIp.data(), static_cast<int>(mcclParamTbGen()), &comm->system);
  if (rc == mcclSuccess)
    for (int i = 0; i < nRanks; ++i) {
      mcclTopoNode& nd = comm->system.nodes[static_cast<size_t>(i)];
      nd.gpuCores = infos[i].gpuCores;
      nd.chipCap  = infos[i].chipCap;
      nd.umaGiB   = infos[i].umaBytes >> 30;
    }
  if (rc == mcclSuccess)
    rc = mcclCalibrateTbLinks(&comm->system, edges.data(), static_cast<int>(edges.size()),
                              rank, nRanks, rootIp, rootPort, static_cast<uint16_t>(rootPort + 2));
  if (rc == mcclSuccess) rc = mcclTopoComputePaths(&comm->system);
  if (rc == mcclSuccess) rc = mcclTopoCompute(comm->system, MCCL_ALGO_RING, &comm->graphs[MCCL_ALGO_RING]);
  if (rc == mcclSuccess) rc = mcclTopoCompute(comm->system, MCCL_ALGO_TREE, &comm->graphs[MCCL_ALGO_TREE]);
  if (rc == mcclSuccess) rc = mcclTopoConnect(comm->system, comm->graphs[MCCL_ALGO_RING], comm->graphs[MCCL_ALGO_TREE], rank, &comm->chan);
  if (rc != mcclSuccess) { mcclCommDestroy(comm); return rc; }
  if (rank == 0)
    if (const char* dump = std::getenv("MCCL_TOPO_DUMP_FILE")) { mcclXml xml; mcclTopoToXml(comm->system, &xml); mcclTopoDumpXmlToFile(dump, &xml); }

  mcclGetEnabledAlgos(comm->enabled);
  int algo = MCCL_ALGO_TREE;
  rc = mcclGetAlgoInfo(comm->graphs, nRanks, kAlgoPickBytes, comm->enabled, &algo);
  if (rc != mcclSuccess) { mcclCommDestroy(comm); return rc; }
  comm->channel = algo;

  if (nRanks > 1) {
    // Connect only channels the cost model can actually pick. The ring/tree crossover is monotonic in size,
    // so the two size extremes bound every per-call choice: a channel that loses at both never runs.
    int small = comm->channel, large = comm->channel;
    mcclGetAlgoInfo(comm->graphs, nRanks, 1, comm->enabled, &small);
    mcclGetAlgoInfo(comm->graphs, nRanks, size_t{1} << 40, comm->enabled, &large);  // cost is linear in bytes, so 1 TiB bounds the pick for ANY size a caller can pass — 1 GiB left real >1 GiB calls picking a never-connected channel
    const mcclChannel& ch = comm->chan;
    const bool useRing = (small == MCCL_ALGO_RING || large == MCCL_ALGO_RING);
    const bool useTree = (small == MCCL_ALGO_TREE || large == MCCL_ALGO_TREE) ||
                         (ch.flatTree && comm->enabled[MCCL_ALGO_TREE] != 0);
    std::set<int> peers;
    if (useRing) {
      if (ch.ring.next >= 0) peers.insert(ch.ring.next);
      if (ch.ring.prev >= 0) peers.insert(ch.ring.prev);
    }
    if (useTree) {
      if (ch.tree.up >= 0) peers.insert(ch.tree.up);
      for (int d : ch.tree.down) peers.insert(d);
    }
    rc = mcclEnsurePeerConns(comm, peers);
    if (rc != mcclSuccess) { mcclCommDestroy(comm); return rc; }
    // next/prev/parent/childConns are non-owning aliases into peerConns (which owns every connection).
    if (useRing) {
      comm->next = ch.ring.next >= 0 ? comm->peerConns.at(ch.ring.next) : nullptr;
      comm->prev = ch.ring.prev >= 0 ? comm->peerConns.at(ch.ring.prev) : nullptr;
    }
    if (useTree) {
      comm->parent = ch.tree.up >= 0 ? comm->peerConns.at(ch.tree.up) : nullptr;
      comm->childRanks = ch.tree.down;
      comm->childConns.clear();
      for (int d : ch.tree.down) comm->childConns.push_back(comm->peerConns.at(d));
    }

    const bool wantRingB = useRing && nRanks >= 3 && mcclParamDualRings() != 0 &&
                           ch.ring.next >= 0 && ch.ring.prev >= 0;
    const bool wantTreeB = useTree && ch.dtree;
    if (wantRingB || wantTreeB) {
      std::set<int> peersB;
      if (wantRingB) { peersB.insert(ch.ring.next); peersB.insert(ch.ring.prev); }
      if (wantTreeB) {
        if (ch.treeB.up >= 0) peersB.insert(ch.treeB.up);
        for (int d : ch.treeB.down) peersB.insert(d);
      }
      int tok = rank;
      std::vector<int> bar(static_cast<size_t>(nRanks));
      rc = mcclBootstrapAllGather(rootIp, rootPort, rank, nRanks, &tok, bar.data(), sizeof(int));
      if (rc == mcclSuccess) rc = mcclEnsurePeerConnsInto(comm, peersB, comm->peerConnsB);
      if (rc != mcclSuccess) { mcclCommDestroy(comm); return rc; }
      if (wantRingB) {
        comm->nextB = comm->peerConnsB.at(ch.ring.next);
        comm->prevB = comm->peerConnsB.at(ch.ring.prev);
      }
      if (wantTreeB) {
        comm->parentB = ch.treeB.up >= 0 ? comm->peerConnsB.at(ch.treeB.up) : nullptr;
        comm->childRanksB = ch.treeB.down;
        comm->childConnsB.clear();
        for (int d : ch.treeB.down) comm->childConnsB.push_back(comm->peerConnsB.at(d));
      }
    }
  }

  if (comm->config.blocking == 0 && nRanks > 1) mcclWorkerStart(comm);
  *out = comm;
  return mcclSuccess;
}

mcclResult mcclCommInitRank(mcclComm** out, int nRanks, mcclUniqueId commId, int rank) {
  return mcclCommInitRankConfig(out, nRanks, commId, rank, nullptr);
}

mcclResult mcclCommReserveStaging(mcclComm* comm, size_t bytes, void** out) {
  if (comm == nullptr || out == nullptr) return mcclInvalidArgument;
  if (bytes > comm->stagingBytes) {
    if (comm->staging) mcclPageFree(comm->staging);
    comm->staging = nullptr; comm->stagingBytes = 0;  // so an alloc failure leaves a consistent (null, 0), not a stale size over a null ptr
    MCCLCHECK(mcclPageAlloc(bytes, &comm->staging));
    comm->stagingBytes = bytes;
  }
  *out = comm->staging;
  return mcclSuccess;
}

// Second persistent arena, disjoint from staging: collective work buffers reuse it instead of mmap-ing per call.
mcclResult mcclCommReserveScratch(mcclComm* comm, size_t bytes, void** out) {
  if (comm == nullptr || out == nullptr) return mcclInvalidArgument;
  if (bytes > comm->scratchBytes) {
    if (comm->scratch) mcclPageFree(comm->scratch);
    comm->scratch = nullptr; comm->scratchBytes = 0;  // so an alloc failure leaves a consistent (null, 0), not a stale size over a null ptr
    MCCLCHECK(mcclPageAlloc(bytes, &comm->scratch));
    comm->scratchBytes = bytes;
  }
  *out = comm->scratch;
  return mcclSuccess;
}

mcclResult mcclCommDestroy(mcclComm* comm) {
  if (comm == nullptr) return mcclInvalidArgument;
  mcclWorkerStop(comm);  // join before freeing the connections the worker uses
  for (auto& kv : comm->peerConns) if (kv.second) mcclM2MClose(kv.second);  // the two maps own everything; the channel pointers only alias
  for (auto& kv : comm->peerConnsB) if (kv.second) mcclM2MClose(kv.second);
  if (comm->listener) mcclM2MListenClose(comm->listener);
  if (comm->staging) mcclPageFree(comm->staging);
  if (comm->scratch) mcclPageFree(comm->scratch);
  delete comm;
  return mcclSuccess;
}

mcclResult mcclCommCount(const mcclComm* comm, int* count) {
  if (comm == nullptr || count == nullptr) return mcclSetLastError(mcclInvalidArgument);
  *count = comm->nRanks;
  return mcclSuccess;
}

mcclResult mcclCommUserRank(const mcclComm* comm, int* rank) {
  if (comm == nullptr || rank == nullptr) return mcclSetLastError(mcclInvalidArgument);
  *rank = comm->rank;
  return mcclSuccess;
}

mcclResult mcclCommDevice(const mcclComm* comm, int* device) {
  if (comm == nullptr || device == nullptr) return mcclSetLastError(mcclInvalidArgument);
  *device = comm->device;
  return mcclSuccess;
}

mcclResult mcclCommGetAsyncError(mcclComm* comm, mcclResult* asyncErr) {
  if (comm == nullptr || asyncErr == nullptr) return mcclSetLastError(mcclInvalidArgument);
  if (comm->aborted) { *asyncErr = mcclInvalidUsage; return mcclSuccess; }
  if (comm->work != nullptr) return mcclWorkerAsyncError(comm, asyncErr);
  *asyncErr = comm->asyncState;
  return mcclSuccess;
}

mcclResult mcclCommFinalize(mcclComm* comm) {
  if (comm == nullptr) return mcclSetLastError(mcclInvalidArgument);
  if (comm->aborted) return mcclSuccess;
  const mcclResult pending = mcclCommSynchronize(comm);
  if (pending != mcclSuccess) { comm->asyncState = pending; return mcclSetLastError(pending); }  // don't barrier a broken comm
  if (comm->nRanks <= 1) return mcclSuccess;
  int tok = 1;
  const mcclResult brc = mcclAllReduce(comm, &tok, &tok, 1, mcclInt32, mcclSum);
  const mcclResult src = mcclCommSynchronize(comm);
  const mcclResult rc  = brc != mcclSuccess ? brc : src;
  comm->asyncState = rc;
  return rc == mcclSuccess ? mcclSuccess : mcclSetLastError(rc);
}

// Fault recovery: SHUTDOWN (never close/free) every connection — including p2p / hub-to-root conns that live
// only in the maps, which a stuck worker may be parked on — plus the listener, so accept-waiters unblock too.
// Freeing stays in mcclCommDestroy: on a blocking comm a user thread can still be inside a collective holding
// these pointers, so Abort closing fds would be a use-after-free / fd-recycle data corruption.
mcclResult mcclCommAbort(mcclComm* comm) {
  if (comm == nullptr) return mcclSetLastError(mcclInvalidArgument);
  comm->aborted = true;
  {
    std::lock_guard<std::mutex> lk(comm->connMu);
    for (auto& kv : comm->peerConns)  if (kv.second) mcclM2MShutdown(kv.second);
    for (auto& kv : comm->peerConnsB) if (kv.second) mcclM2MShutdown(kv.second);
    if (comm->listener) mcclM2MListenShutdown(comm->listener);
  }
  mcclWorkerStop(comm);
  return mcclSuccess;
}

mcclResult mcclSetLastError(mcclResult r) {
  if (r != mcclSuccess) g_lastError = r;
  return r;
}

mcclResult mcclGetLastError(const mcclComm*) { return g_lastError; }

// All ranks all-gather (color, key, LAN addr) over the parent bootstrap; same-color ranks (ordered by key)
// form a fresh comm rooted at the group's rank 0 over a derived port. MCCL_SPLIT_NOCOLOR opts a rank out.
mcclResult mcclCommSplit(mcclComm* comm, int color, int key, mcclComm** newcomm, const mcclConfig* config) {
  if (comm == nullptr || newcomm == nullptr) return mcclSetLastError(mcclInvalidArgument);
  *newcomm = nullptr;
  const int n = comm->nRanks, r = comm->rank;

  struct SplitInfo { int color; int key; uint32_t lanIp; };
  SplitInfo me{color, key, comm->system.nodes[static_cast<size_t>(r)].lanIp};
  std::vector<SplitInfo> all(static_cast<size_t>(n));
  MCCLCHECK(mcclBootstrapAllGather(comm->rootIp, comm->rootPort, r, n, &me, all.data(), sizeof(SplitInfo)));
  if (color == MCCL_SPLIT_NOCOLOR) return mcclSuccess;

  std::vector<std::pair<int, int>> grp;
  for (int i = 0; i < n; ++i) if (all[static_cast<size_t>(i)].color == color) grp.push_back({all[static_cast<size_t>(i)].key, i});
  std::sort(grp.begin(), grp.end());
  const int newN = static_cast<int>(grp.size());
  int newRank = -1;
  for (int i = 0; i < newN; ++i) if (grp[static_cast<size_t>(i)].second == r) newRank = i;
  if (newRank < 0) return mcclSetLastError(mcclInternalError);

  // Port by the color's index among this call's sorted distinct colors — every rank computes the same array, and
  // distinct groups get distinct ports with no modulo aliasing (color & 0x3F collided for colors equal mod 64).
  std::vector<int> distinctColors;
  for (int i = 0; i < n; ++i)
    if (all[static_cast<size_t>(i)].color != MCCL_SPLIT_NOCOLOR) distinctColors.push_back(all[static_cast<size_t>(i)].color);
  std::sort(distinctColors.begin(), distinctColors.end());
  distinctColors.erase(std::unique(distinctColors.begin(), distinctColors.end()), distinctColors.end());
  const size_t groupIdx = static_cast<size_t>(std::lower_bound(distinctColors.begin(), distinctColors.end(), color) - distinctColors.begin());
  const uint16_t subPort = static_cast<uint16_t>(comm->rootPort + 1000 + groupIdx * 8);

  // On a pure point-to-point TB mesh no subnet spans every rank, so lanIp is 0 — fall back to the parent's
  // bootstrap host, which every member could already reach (it bootstrapped through it).
  char subIp[16];
  const uint32_t leadIp = all[static_cast<size_t>(grp[0].second)].lanIp;
  const char* subIpStr = leadIp != 0 ? mcclIpStr(leadIp, subIp) : comm->rootIp;
  mcclUniqueId subId{};
  IdData d{};
  std::snprintf(d.ip, sizeof(d.ip), "%s", subIpStr);
  d.port = subPort;
  std::memcpy(subId.internal, &d, sizeof(d));
  return mcclCommInitRankConfig(newcomm, newN, subId, newRank, config != nullptr ? config : &comm->config);
}

}

#ifdef MCCL_COMM_MAIN
int main() {
  using namespace mccl;
  auto envI = [](const char* k, int d) { const char* v = std::getenv(k); return v ? std::atoi(v) : d; };
  const int rank = envI("MCCL_RANK", 0);
  const int n    = envI("MCCL_WORLD_SIZE", 1);

  mcclUniqueId id;
  mcclGetUniqueId(&id);
  mcclComm* comm = nullptr;
  const mcclResult rc = mcclCommInitRank(&comm, n, id, rank);
  if (rc != mcclSuccess) { std::printf("[comm] rank %d/%d init rc=%d FAIL\n", rank, n, static_cast<int>(rc)); return 1; }

  if (rank == 0) {
    std::printf("[topo] macs=%d lan_all_to_all=%d  ring.bw=%.2f tree.bw=%.2f\n",
                comm->nRanks, comm->system.lanAllToAll ? 1 : 0,
                static_cast<double>(comm->graphs[MCCL_ALGO_RING].bw), static_cast<double>(comm->graphs[MCCL_ALGO_TREE].bw));
    for (const mcclTopoNode& nd : comm->system.nodes) {
      std::printf("[topo]  mac %d links:", nd.rank);
      for (const mcclTopoLink& l : nd.links) std::printf(" %s->%d@%.1f", mcclLinkTypeStr(l.type), l.remote, static_cast<double>(l.bw));
      std::printf("\n");
    }
  }
  std::printf("[comm] rank %d/%d algo=%s  %s\n", rank, n, mcclAlgoStr(comm->channel),
              comm->channel == MCCL_ALGO_RING ? "ring connected"
              : (comm->parent ? "tree leaf/mid" : "tree root"));
  mcclCommDestroy(comm);
  return 0;
}
#endif
