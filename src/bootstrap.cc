#include "bootstrap.h"

#include "socket.h"
#include "include/checks.h"

#include <cstring>
#include <vector>

namespace mccl {

namespace {
constexpr int kAcceptTimeoutMs  = 30000;
constexpr int kConnectTimeoutMs = 5000;
constexpr int kConnectRetries   = 100;  // ~10s of 100ms backoff to absorb the root coming up after the workers
}

mcclResult mcclBootstrapAllGather(const char* rootIp, uint16_t rootPort, int rank, int nRanks,
                                  const void* sendData, void* recvData, size_t perRankBytes) {
  if (rootIp == nullptr || sendData == nullptr || recvData == nullptr ||
      nRanks < 1 || rank < 0 || rank >= nRanks)
    return mcclInvalidArgument;

  char*       recv = static_cast<char*>(recvData);
  const char* send = static_cast<const char*>(sendData);
  const size_t total = static_cast<size_t>(nRanks) * perRankBytes;

  if (rank == 0) {
    std::memcpy(recv, send, perRankBytes);
    if (nRanks == 1) return mcclSuccess;

    int lst = -1;
    MCCLCHECK(mcclSocketListen(rootPort, nRanks, &lst, nullptr));
    std::vector<int> conns(nRanks, -1);
    mcclResult rc = mcclSuccess;
    for (int i = 0; i < nRanks - 1; ++i) {
      int fd = -1, peerRank = -1;
      if ((rc = mcclSocketAccept(lst, kAcceptTimeoutMs, &fd)) != mcclSuccess) goto done;
      if ((rc = mcclSocketRecv(fd, &peerRank, sizeof(peerRank))) != mcclSuccess) { mcclSocketClose(fd); goto done; }
      if (peerRank <= 0 || peerRank >= nRanks || conns[peerRank] != -1) {
        mcclSocketClose(fd);
        rc = mcclInternalError;
        goto done;
      }
      conns[peerRank] = fd;
      if ((rc = mcclSocketRecv(fd, recv + static_cast<size_t>(peerRank) * perRankBytes, perRankBytes)) != mcclSuccess)
        goto done;
    }
    for (int r = 1; r < nRanks && rc == mcclSuccess; ++r)
      rc = mcclSocketSend(conns[r], recv, total);
  done:
    mcclSocketClose(lst);
    for (int c : conns) mcclSocketClose(c);
    return rc;
  }

  int fd = -1;
  MCCLCHECK(mcclSocketConnect(rootIp, rootPort, kConnectTimeoutMs, kConnectRetries, &fd));
  mcclResult rc = mcclSocketSend(fd, &rank, sizeof(rank));
  if (rc == mcclSuccess) rc = mcclSocketSend(fd, send, perRankBytes);
  if (rc == mcclSuccess) rc = mcclSocketRecv(fd, recv, total);
  mcclSocketClose(fd);
  return rc;
}

}

#ifdef MCCL_BOOTSTRAP_MAIN
#include <cstdio>
#include <cstdlib>

int main() {
  using namespace mccl;
  auto envI = [](const char* k, int d) { const char* v = std::getenv(k); return v ? std::atoi(v) : d; };
  const int      rank = envI("MCCL_RANK", 0);
  const int      n    = envI("MCCL_WORLD_SIZE", 1);
  const char*    ip   = std::getenv("MCCL_BOOTSTRAP_IP");
  const uint16_t port = static_cast<uint16_t>(envI("MCCL_BOOTSTRAP_PORT", 53700));
  if (ip == nullptr) ip = "127.0.0.1";

  const int mine = 1000 + rank;
  std::vector<int> all(n > 0 ? n : 1, -1);
  const mcclResult rc = mcclBootstrapAllGather(ip, port, rank, n, &mine, all.data(), sizeof(int));
  std::printf("[boot] rank %d/%d rc=%d gathered:", rank, n, static_cast<int>(rc));
  for (int x : all) std::printf(" %d", x);
  std::printf("\n");
  return rc == mcclSuccess ? 0 : 1;
}
#endif
