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

// Point-to-point to an arbitrary peer. Wrap in mcclGroupStart/End to run a batch (all-to-all, scatter/gather,
// bidirectional exchange) concurrently and deadlock-free; ungrouped, each call is one blocking transfer.
mcclResult mcclSend(mcclComm* comm, const void* sendbuff, size_t count, mcclDataType dt, int peer);
mcclResult mcclRecv(mcclComm* comm, void* recvbuff, size_t count, mcclDataType dt, int peer);

}
