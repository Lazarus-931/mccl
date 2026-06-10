#pragma once

#include <cstddef>

#include "../definitions.h"

namespace mccl {

struct mcclComm;

// Collectives over the comm; count is the per-buffer element count. The cost model picks ring vs tree per call.
mcclResult mcclAllReduce(mcclComm* comm, const void* sendbuff, void* recvbuff, size_t count, mcclDataType dt, mcclRedOp op);
mcclResult mcclAllGather(mcclComm* comm, const void* sendbuff, void* recvbuff, size_t sendcount, mcclDataType dt);
mcclResult mcclReduceScatter(mcclComm* comm, const void* sendbuff, void* recvbuff, size_t recvcount, mcclDataType dt, mcclRedOp op);
mcclResult mcclBroadcast(mcclComm* comm, const void* sendbuff, void* recvbuff, size_t count, mcclDataType dt, int root);
mcclResult mcclReduce(mcclComm* comm, const void* sendbuff, void* recvbuff, size_t count, mcclDataType dt, mcclRedOp op, int root);

// Each rank scatters its sendbuff (nRanks chunks of `count` elements) to every rank and gathers a chunk
// from each into recvbuff, both indexed by rank. No reduction; a direct N-by-N exchange over the grouped
// Send/Recv batch. sendbuff and recvbuff each hold count*nRanks elements.
mcclResult mcclAllToAll(mcclComm* comm, const void* sendbuff, void* recvbuff, size_t count, mcclDataType dt);

// Point-to-point to an arbitrary peer. Wrap in mcclGroupStart/End to run a batch (all-to-all, scatter/gather,
// bidirectional exchange) concurrently and deadlock-free; ungrouped, each call is one blocking transfer.
mcclResult mcclSend(mcclComm* comm, const void* sendbuff, size_t count, mcclDataType dt, int peer);
mcclResult mcclRecv(mcclComm* comm, void* recvbuff, size_t count, mcclDataType dt, int peer);

}
