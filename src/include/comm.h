#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <utility>
#include <vector>

#include "../definitions.h"
#include "../graph/topo.h"
#include "graph.h"
#include "param.h"

namespace mccl {

struct mcclM2M;
struct mcclM2MListener;
struct mcclMemPool;
struct mcclWorkQueue;

struct mcclConfig {
  int         blocking  = 1;        // 1: collectives block until done; 0: enqueue + return, drain with mcclCommSynchronize
  const char* transport = nullptr;  // force the m2m backend ("tcp"|"rdma"); null = MCCL_TRANSPORT env, else tcp
};
#define MCCL_CONFIG_INITIALIZER mccl::mcclConfig{}

#define MCCL_SPLIT_NOCOLOR (-1)

struct mcclComm {
  int            rank    = 0;
  int            nRanks  = 1;
  int            device  = 0;
  int            channel = MCCL_ALGO_TREE;
  int            m2mType = 0;
  mcclConfig     config;
  mcclResult     asyncState = mcclSuccess;
  bool           aborted    = false;
  char           rootIp[64] = {0};
  uint16_t       rootPort   = 0;
  int            enabled[MCCL_NUM_ALGOS] = {1, 1};  // MCCL_ALGO mask, read once at init (not per call)
  mcclTopoSystem system;
  mcclTopoGraph  graphs[MCCL_NUM_ALGOS];    // ring + tree search results; the cost model picks between them per call
  mcclChannel    chan;
  mcclM2M*       next   = nullptr;
  mcclM2M*       prev   = nullptr;
  mcclM2M*       nextB  = nullptr;
  mcclM2M*       prevB  = nullptr;  // so the two legs never share one; null when dual rings are off or n < 3
  mcclM2M*       parent = nullptr;
  std::vector<int>      childRanks;
  std::vector<mcclM2M*> childConns;
  mcclM2M*       parentB = nullptr;
  std::vector<int>      childRanksB;
  std::vector<mcclM2M*> childConnsB;
  void*          staging  = nullptr;        // persistent UMA scratch for Primitives recv-staging, grown + reused
  size_t         stagingBytes = 0;
  void*          scratch  = nullptr;        // second persistent arena for collective work buffers (disjoint from staging)
  size_t         scratchBytes = 0;
  mcclWorkQueue* work = nullptr;
  mcclM2MListener*        listener = nullptr;  // stays open past init for on-demand Send/Recv connections
  std::vector<uint16_t>   peerPorts;
  std::mutex              connMu;              // guards map INSERTS (owner thread) vs Abort's iteration (any thread); owner-thread reads need no lock
  std::map<int, mcclM2M*> peerConns;           // owns all connections; next/prev/parent/childConns alias into it
  std::map<int, mcclM2M*> peerConnsB;          // second tree's own connections (dtree), kept separate so the trees never share a socket
};

mcclResult mcclCommReserveStaging(mcclComm* comm, size_t bytes, void** out);
mcclResult mcclCommReserveScratch(mcclComm* comm, size_t bytes, void** out);

MCCL_PARAM(DualRings);
MCCL_PARAM(DualRingMinBytes);

struct mcclUniqueId {
  char internal[128];
};

mcclResult mcclGetUniqueId(mcclUniqueId* id);

mcclResult mcclCommInitRank(mcclComm** out, int nRanks, mcclUniqueId commId, int rank);
mcclResult mcclCommInitRankConfig(mcclComm** out, int nRanks, mcclUniqueId commId, int rank, const mcclConfig* config);
mcclResult mcclCommDestroy(mcclComm* comm);
mcclResult mcclCommFinalize(mcclComm* comm);
mcclResult mcclCommAbort(mcclComm* comm);

mcclResult mcclCommCount(const mcclComm* comm, int* count);
mcclResult mcclCommUserRank(const mcclComm* comm, int* rank);
mcclResult mcclCommDevice(const mcclComm* comm, int* device);
mcclResult mcclCommGetAsyncError(mcclComm* comm, mcclResult* asyncErr);

mcclResult mcclCommSynchronize(mcclComm* comm);

mcclResult mcclEnqueue(mcclComm* comm, std::function<mcclResult()> op);

mcclResult mcclWorkerStart(mcclComm* comm);
void       mcclWorkerStop(mcclComm* comm);
mcclResult mcclWorkerAsyncError(mcclComm* comm, mcclResult* out);
int        mcclPickAlgo(const mcclComm* comm, size_t bytes);

mcclResult mcclEnsurePeerConns(mcclComm* comm, const std::set<int>& peers);
mcclResult mcclEnsurePeerConnsInto(mcclComm* comm, const std::set<int>& peers, std::map<int, mcclM2M*>& target);

// A fatal (transport/system) error desynchronizes the channel and poisons the comm so no later collective
// runs over it; a per-call usage error (bad arg) does not.
inline bool mcclResultFatal(mcclResult r) {
  return r == mcclSystemError || r == mcclInternalError || r == mcclRemoteError;
}

// Run op inline (blocking, the default — zero std::function alloc on the hot path) or hand it to the worker
// (non-blocking) so its transfer overlaps the caller's compute. Drive one comm from one thread at a time.
template <typename Fn>
inline mcclResult mcclLaunch(mcclComm* comm, Fn&& op) {
  if (comm == nullptr) return mcclInvalidArgument;
  if (comm->config.blocking != 0 || comm->nRanks <= 1) {
    if (mcclResultFatal(comm->asyncState)) return comm->asyncState;
    const mcclResult rc = op();
    if (mcclResultFatal(rc)) comm->asyncState = rc;
    return rc;
  }
  return mcclEnqueue(comm, std::function<mcclResult()>(std::forward<Fn>(op)));
}

mcclResult mcclCommSplit(mcclComm* comm, int color, int key, mcclComm** newcomm, const mcclConfig* config);

// Group Send/Recv: ops between Start/End on a comm run as one concurrent batch (a send + recv on a link
// overlap). Collectives still execute eagerly — grouping only batches point-to-point.
mcclResult mcclGroupStart();
mcclResult mcclGroupEnd();

mcclResult mcclGetLastError(const mcclComm* comm);
mcclResult mcclSetLastError(mcclResult r);

}
