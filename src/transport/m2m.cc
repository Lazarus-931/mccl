#include "m2m.h"

#include "../socket.h"
#include "rdma/rdma.h"
#include "../pool.h"
#include "../include/param.h"
#include "../include/checks.h"

#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>

namespace mccl {

namespace {
// First bytes on every data socket. connId groups one connection's sockets so a listener can demux several
// peers dialing at once (a tree hub's K children); shard is the index within the set; type lets the listener
// tell a TCP striped set (await all shards) from an RDMA connector (one socket, immediate UC bring-up).
struct StripeHdr {
  uint32_t connId;
  uint32_t shard;
  uint32_t type;
};

uint32_t nextConnId() {
  static std::atomic<uint32_t> ctr{0};
  static const uint32_t base = (static_cast<uint32_t>(::getpid()) * 2654435761u) ^
                               static_cast<uint32_t>(std::chrono::steady_clock::now().time_since_epoch().count());
  return base + ctr.fetch_add(1) * 0x9E3779B9u;
}
}

struct m2mTcpResources {
  int* fds;
  int  nSocks;
};

struct m2mRdmaResources {
  mcclRdmaConn* conn;
};

struct m2mResources {
  m2mType type;
  union {
    m2mTcpResources  tcp;
    m2mRdmaResources rdma;
  };
};

struct mcclM2M {
  m2mResources res;
};

m2mType m2mTypeSelect() {
  const char* env = std::getenv("MCCL_TRANSPORT");
  if (env != nullptr && std::strcmp(env, "rdma") == 0) return M2M_TYPE_RDMA;
  return M2M_TYPE_TCP;
}

const char* m2mTypeStr(m2mType t) {
  switch (t) {
    case M2M_TYPE_TCP:  return "tcp";
    case M2M_TYPE_RDMA: return "rdma";
  }
  return "unknown";
}

bool mcclM2MAvailable(m2mType t) { return t == M2M_TYPE_RDMA ? mcclRdmaAvailable() : true; }

namespace {

DEFINE_MCCL_PARAM(ConnectTimeoutMs, "CONNECT_TIMEOUT_MS", 5000);
DEFINE_MCCL_PARAM(AcceptTimeoutMs,  "ACCEPT_TIMEOUT_MS",  30000);
DEFINE_MCCL_PARAM(KeepAliveIdleSec, "KEEPALIVE_IDLE_SEC", 10);
DEFINE_MCCL_PARAM(NSocks,           "NSOCKS",             8);          // sockets per connection (measured best on TB; >=12 stalls connection setup)
DEFINE_MCCL_PARAM(StripeMinBytes,   "STRIPE_MIN_BYTES",   1 << 18);    // below this, one socket: striping's thread-spawn costs more than it saves

constexpr int kConnectRetries = 50;

int dataSocks() {
  const int n = static_cast<int>(mcclParamNSocks());
  return n < 1 ? 1 : (n > 64 ? 64 : n);
}

void keepAlive(int fd) {  // so a blocking recv unblocks if the peer dies (socket.cc sets only NODELAY/NOSIGPIPE)
  int one = 1, idle = static_cast<int>(mcclParamKeepAliveIdleSec());
  setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof(one));
  setsockopt(fd, IPPROTO_TCP, TCP_KEEPALIVE, &idle, sizeof(idle));
}

mcclM2M* newTcp(int* fds, int nSocks) {
  mcclM2M* m = new mcclM2M();
  m->res.type = M2M_TYPE_TCP;
  m->res.tcp.fds = fds;
  m->res.tcp.nSocks = nSocks;
  return m;
}

void closeFds(int* fds, int n) {
  if (fds == nullptr) return;
  for (int i = 0; i < n; ++i) mcclSocketClose(fds[i]);
  delete[] fds;
}

mcclResult tcpConnectN(const char* peerIp, uint16_t port, int nSocks, mcclM2M** out) {
  int* fds = new int[nSocks];
  for (int i = 0; i < nSocks; ++i) fds[i] = -1;
  const uint32_t connId = nextConnId();
  for (int i = 0; i < nSocks; ++i) {
    StripeHdr h{connId, static_cast<uint32_t>(i), M2M_TYPE_TCP};
    if (mcclSocketConnect(peerIp, port, static_cast<int>(mcclParamConnectTimeoutMs()), kConnectRetries, &fds[i]) != mcclSuccess ||
        mcclSocketSend(fds[i], &h, sizeof(h)) != mcclSuccess) {
      closeFds(fds, nSocks);
      return mcclSystemError;
    }
    keepAlive(fds[i]);
  }
  *out = newTcp(fds, nSocks);
  return mcclSuccess;
}

mcclResult acceptStripedSet(int lst, int nSocks, mcclM2M** out) {
  int* fds = new int[nSocks];
  for (int i = 0; i < nSocks; ++i) fds[i] = -1;
  mcclResult rc = mcclSuccess;
  for (int i = 0; i < nSocks; ++i) {
    int fd = -1;
    StripeHdr h{};
    if ((rc = mcclSocketAccept(lst, static_cast<int>(mcclParamAcceptTimeoutMs()), &fd)) != mcclSuccess) break;
    if ((rc = mcclSocketRecv(fd, &h, sizeof(h))) != mcclSuccess) { mcclSocketClose(fd); break; }
    const uint32_t s = h.shard;
    if (static_cast<int>(s) >= nSocks || fds[s] != -1) { mcclSocketClose(fd); rc = mcclInternalError; break; }
    fds[s] = fd;
    keepAlive(fd);
  }
  if (rc != mcclSuccess) { closeFds(fds, nSocks); return rc; }
  *out = newTcp(fds, nSocks);
  return mcclSuccess;
}

mcclResult tcpListenN(uint16_t port, int nSocks, mcclM2M** out) {
  int lst = -1;
  MCCLCHECK(mcclSocketListen(port, nSocks, &lst, nullptr));
  const mcclResult rc = acceptStripedSet(lst, nSocks, out);
  mcclSocketClose(lst);
  return rc;
}

// Split `bytes` into one contiguous shard per socket, transferred concurrently. Both ends derive the same
// split from `bytes`, so no length is exchanged. Small transfers take a single socket (see STRIPE_MIN_BYTES).
mcclResult stripedXfer(const int* fds, int nSocks, void* buf, size_t bytes, bool isSend) {
  char* base = static_cast<char*>(buf);
  if (nSocks <= 1 || bytes < static_cast<size_t>(mcclParamStripeMinBytes()))
    return isSend ? mcclSocketSend(fds[0], base, bytes) : mcclSocketRecv(fds[0], base, bytes);
  return mcclParallel(mcclStripePool(), static_cast<size_t>(nSocks), [=](size_t i) {
    const size_t lo = bytes * i / static_cast<size_t>(nSocks);
    const size_t hi = bytes * (i + 1) / static_cast<size_t>(nSocks);
    return isSend ? mcclSocketSend(fds[i], base + lo, hi - lo)
                  : mcclSocketRecv(fds[i], base + lo, hi - lo);
  });
}

mcclM2M* newRdma(mcclRdmaConn* conn) {
  mcclM2M* m = new mcclM2M();
  m->res.type = M2M_TYPE_RDMA;
  m->res.rdma.conn = conn;
  return m;
}

mcclResult rdmaSetup(int sideFd, mcclM2M** out) {
  const char* device = std::getenv("MCCL_RDMA_DEVICE");
  if (device == nullptr) return mcclInvalidUsage;
  mcclRdmaConn* conn = nullptr;
  if (mcclRdmaOpen(device, &conn) != mcclSuccess) return mcclSystemError;
  mcclRdmaDest local{}, peer{};
  mcclResult rc = mcclSuccess;
  MCCLCHECKGOTO(mcclRdmaLocalDest(conn, &local), rc, fail);
  MCCLCHECKGOTO(mcclSocketSend(sideFd, &local, sizeof(local)), rc, fail);
  MCCLCHECKGOTO(mcclSocketRecv(sideFd, &peer, sizeof(peer)), rc, fail);
  MCCLCHECKGOTO(mcclRdmaConnect(conn, peer), rc, fail);
  *out = newRdma(conn);
  return mcclSuccess;
fail:
  mcclRdmaClose(conn);
  return rc;
}

mcclResult rdmaConnect(const char* peerIp, uint16_t port, mcclM2M** out) {
  int fd = -1;
  StripeHdr h{nextConnId(), 0, M2M_TYPE_RDMA};
  if (mcclSocketConnect(peerIp, port, static_cast<int>(mcclParamConnectTimeoutMs()), kConnectRetries, &fd) != mcclSuccess ||
      mcclSocketSend(fd, &h, sizeof(h)) != mcclSuccess) {
    if (fd >= 0) mcclSocketClose(fd);
    return mcclSystemError;
  }
  const mcclResult rc = rdmaSetup(fd, out);
  mcclSocketClose(fd);
  return rc;
}

mcclResult rdmaListen(uint16_t port, mcclM2M** out) {
  mcclM2M* side = nullptr;
  mcclResult rc = tcpListenN(port, 1, &side);
  if (rc != mcclSuccess) return rc;
  rc = rdmaSetup(side->res.tcp.fds[0], out);
  mcclM2MClose(side);
  return rc;
}

}

mcclResult mcclM2MConnect(const char* peerIp, uint16_t port, m2mType t, mcclM2M** out) {
  if (out == nullptr || peerIp == nullptr) return mcclInvalidArgument;
  *out = nullptr;
  if (t == M2M_TYPE_RDMA) {
    if (!mcclRdmaAvailable()) return mcclInvalidUsage;
    return rdmaConnect(peerIp, port, out);
  }
  return tcpConnectN(peerIp, port, dataSocks(), out);
}

mcclResult mcclM2MListen(uint16_t port, m2mType t, mcclM2M** out) {
  if (out == nullptr) return mcclInvalidArgument;
  *out = nullptr;
  if (t == M2M_TYPE_RDMA) {
    if (!mcclRdmaAvailable()) return mcclInvalidUsage;
    return rdmaListen(port, out);
  }
  return tcpListenN(port, dataSocks(), out);
}

mcclResult mcclM2MSend(mcclM2M* c, const void* buf, size_t bytes) {
  if (c == nullptr || buf == nullptr) return mcclInvalidArgument;
  if (c->res.type == M2M_TYPE_TCP)  return stripedXfer(c->res.tcp.fds, c->res.tcp.nSocks, const_cast<void*>(buf), bytes, true);
  if (c->res.type == M2M_TYPE_RDMA) return mcclRdmaSend(c->res.rdma.conn, buf, bytes);
  return mcclInvalidUsage;
}

mcclResult mcclM2MRecv(mcclM2M* c, void* buf, size_t bytes) {
  if (c == nullptr || buf == nullptr) return mcclInvalidArgument;
  if (c->res.type == M2M_TYPE_TCP)  return stripedXfer(c->res.tcp.fds, c->res.tcp.nSocks, buf, bytes, false);
  if (c->res.type == M2M_TYPE_RDMA) return mcclRdmaRecv(c->res.rdma.conn, buf, bytes);
  return mcclInvalidUsage;
}

// Unblock an in-flight send/recv (e.g. a worker parked on a dead peer) without freeing the fds — Abort calls
// this before joining the worker, then mcclM2MClose frees once nothing can touch the connection.
mcclResult mcclM2MShutdown(mcclM2M* c) {
  if (c == nullptr) return mcclInvalidArgument;
  if (c->res.type == M2M_TYPE_TCP && c->res.tcp.fds != nullptr)
    for (int i = 0; i < c->res.tcp.nSocks; ++i)
      if (c->res.tcp.fds[i] >= 0) ::shutdown(c->res.tcp.fds[i], SHUT_RDWR);
  return mcclSuccess;
}

mcclResult mcclM2MClose(mcclM2M* c) {
  if (c == nullptr) return mcclInvalidArgument;
  if (c->res.type == M2M_TYPE_TCP) closeFds(c->res.tcp.fds, c->res.tcp.nSocks);
  else if (c->res.type == M2M_TYPE_RDMA) mcclRdmaClose(c->res.rdma.conn);
  delete c;
  return mcclSuccess;
}

struct mcclM2MListener {
  int lst;
  int nSocks;
  std::map<uint32_t, std::vector<int>> partial;  // connId -> shards seen so far; lets one accept call complete one peer while others' sockets wait
};

mcclResult mcclM2MListenStart(uint16_t port, mcclM2MListener** out, uint16_t* boundPort) {
  if (out == nullptr) return mcclInvalidArgument;
  *out = nullptr;
  const int nSocks = dataSocks();
  int lst = -1;
  MCCLCHECK(mcclSocketListen(port, 256, &lst, boundPort));  // backlog >> nSocks: a hub gets (peers x nSocks) SYNs at once; 8 dropped them at scale
  auto* l = new mcclM2MListener();
  l->lst = lst;
  l->nSocks = nSocks;
  *out = l;
  return mcclSuccess;
}

// Return one complete connection, demuxing by connId: a socket for another peer's connId is buffered in
// `partial` and picked up by a later call, so a hub can accept its K children even as their sockets interleave.
mcclResult mcclM2MAccept(mcclM2MListener* l, mcclM2M** out) {
  if (l == nullptr || out == nullptr) return mcclInvalidArgument;
  *out = nullptr;
  const int nSocks = l->nSocks;
  for (;;) {
    int fd = -1;
    mcclResult rc = mcclSocketAccept(l->lst, static_cast<int>(mcclParamAcceptTimeoutMs()), &fd);
    if (rc != mcclSuccess) return rc;
    StripeHdr h{};
    if ((rc = mcclSocketRecv(fd, &h, sizeof(h))) != mcclSuccess) { mcclSocketClose(fd); return rc; }
    if (h.type == M2M_TYPE_RDMA) { rc = rdmaSetup(fd, out); mcclSocketClose(fd); return rc; }
    if (static_cast<int>(h.shard) >= nSocks) { mcclSocketClose(fd); return mcclInternalError; }
    std::vector<int>& slots = l->partial[h.connId];
    if (slots.empty()) slots.assign(static_cast<size_t>(nSocks), -1);
    if (slots[h.shard] != -1) { mcclSocketClose(fd); return mcclInternalError; }
    slots[h.shard] = fd;
    keepAlive(fd);
    bool full = true;
    for (int s : slots) if (s == -1) { full = false; break; }
    if (full) {
      int* fds = new int[nSocks];
      for (int i = 0; i < nSocks; ++i) fds[i] = slots[static_cast<size_t>(i)];
      l->partial.erase(h.connId);
      *out = newTcp(fds, nSocks);
      return mcclSuccess;
    }
  }
}

mcclResult mcclM2MListenClose(mcclM2MListener* l) {
  if (l == nullptr) return mcclInvalidArgument;
  for (auto& kv : l->partial) for (int fd : kv.second) if (fd != -1) mcclSocketClose(fd);
  mcclSocketClose(l->lst);
  delete l;
  return mcclSuccess;
}

}

#ifdef MCCL_M2M_MAIN
#include <cstdio>

int main() {
  using namespace mccl;
  auto envI = [](const char* k, int d) { const char* v = std::getenv(k); return v ? std::atoi(v) : d; };
  const char*    role  = std::getenv("MCCL_M2M_ROLE");
  const char*    ip    = std::getenv("MCCL_M2M_PEER");
  const uint16_t port  = static_cast<uint16_t>(envI("MCCL_M2M_PORT", 53760));
  const size_t   bytes = static_cast<size_t>(envI("MCCL_M2M_BYTES", 1 << 20));
  if (role == nullptr) role = "";
  if (ip == nullptr) ip = "127.0.0.1";

  std::vector<unsigned char> data(bytes);
  mcclM2M* c = nullptr;
  if (std::strcmp(role, "server") == 0) {
    mcclResult rc = mcclM2MListen(port, M2M_TYPE_TCP, &c);
    if (rc == mcclSuccess) rc = mcclM2MRecv(c, data.data(), bytes);
    bool ok = (rc == mcclSuccess);
    for (size_t i = 0; ok && i < bytes; ++i)
      if (data[i] != static_cast<unsigned char>(i & 0xff)) ok = false;
    std::printf("[m2m] server rc=%d bytes=%zu integrity=%s\n", static_cast<int>(rc), bytes, ok ? "OK" : "BAD");
    if (c) mcclM2MClose(c);
    return ok ? 0 : 1;
  }
  for (size_t i = 0; i < bytes; ++i) data[i] = static_cast<unsigned char>(i & 0xff);
  mcclResult rc = mcclM2MConnect(ip, port, M2M_TYPE_TCP, &c);
  if (rc == mcclSuccess) rc = mcclM2MSend(c, data.data(), bytes);
  std::printf("[m2m] client rc=%d bytes=%zu\n", static_cast<int>(rc), bytes);
  if (c) mcclM2MClose(c);
  return rc == mcclSuccess ? 0 : 1;
}
#endif
