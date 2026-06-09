
#include <arpa/inet.h>
#include <fcntl.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "init.h"
#include "bootstrap.h"
#include "graph/discover.h"
#include "graph/topo.h"
#include "include/checks.h"

namespace mccl {

namespace {

constexpr size_t kBufBytes = 4u << 20;
constexpr int    kSockBuf  = 4 << 20;

double secondsSince(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void tuneSocket(int fd) {
  int one = 1;
  setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  int sz = kSockBuf;
  setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sz, sizeof(sz));
  setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &sz, sizeof(sz));
}

bool connectTimeout(int fd, const sockaddr_in& addr, int timeoutMs) {
  const int flags = fcntl(fd, F_GETFL, 0);
  if (flags < 0) return false;
  fcntl(fd, F_SETFL, flags | O_NONBLOCK);
  bool ok = false;
  if (::connect(fd, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) == 0) {
    ok = true;
  } else if (errno == EINPROGRESS) {
    fd_set wfds;
    FD_ZERO(&wfds);
    FD_SET(fd, &wfds);
    timeval tv{};
    tv.tv_sec = timeoutMs / 1000;
    tv.tv_usec = (timeoutMs % 1000) * 1000;
    if (select(fd + 1, nullptr, &wfds, nullptr, &tv) > 0) {
      int err = 0;
      socklen_t len = sizeof(err);
      ok = getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len) == 0 && err == 0;
    }
  }
  fcntl(fd, F_SETFL, flags);
  return ok;
}

}

mcclBandwidthResult mcclBandwidthServe(uint16_t port, int acceptTimeoutMs) {
  mcclBandwidthResult r;
  int lst = socket(AF_INET, SOCK_STREAM, 0);
  if (lst < 0) return r;
  int one = 1;
  setsockopt(lst, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(port);
  if (bind(lst, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 || listen(lst, 1) != 0) {
    close(lst);
    return r;
  }

  fd_set rfds;
  FD_ZERO(&rfds);
  FD_SET(lst, &rfds);
  timeval tv{};
  tv.tv_sec = acceptTimeoutMs / 1000;
  tv.tv_usec = (acceptTimeoutMs % 1000) * 1000;
  if (select(lst + 1, &rfds, nullptr, nullptr, &tv) <= 0) {
    close(lst);
    return r;
  }
  int fd = accept(lst, nullptr, nullptr);
  close(lst);
  if (fd < 0) return r;
  tuneSocket(fd);
  timeval rcvto{};
  rcvto.tv_sec = 10;
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &rcvto, sizeof(rcvto));

  std::vector<char> buf(kBufBytes);
  uint64_t total = 0;
  const auto t0 = std::chrono::steady_clock::now();
  for (;;) {
    ssize_t n = recv(fd, buf.data(), buf.size(), 0);
    if (n <= 0) break;
    total += static_cast<uint64_t>(n);
  }
  r.seconds = secondsSince(t0);
  close(fd);

  r.bytes = total;
  r.gbps = r.seconds > 0 ? static_cast<double>(total) / 1e9 / r.seconds : 0.0;
  r.ok = total > 0;
  return r;
}

mcclBandwidthResult mcclBandwidthProbe(const char* host, uint16_t port, int durationMs) {
  mcclBandwidthResult r;
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return r;

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  if (inet_pton(AF_INET, host, &addr.sin_addr) != 1 || !connectTimeout(fd, addr, 5000)) {
    close(fd);
    return r;
  }
  tuneSocket(fd);

  std::vector<char> buf(kBufBytes, 0);
  uint64_t total = 0;
  const auto t0 = std::chrono::steady_clock::now();
  while (secondsSince(t0) * 1000.0 < durationMs) {
    ssize_t n = send(fd, buf.data(), buf.size(), 0);
    if (n <= 0) break;
    total += static_cast<uint64_t>(n);
  }
  r.seconds = secondsSince(t0);
  shutdown(fd, SHUT_WR);
  close(fd);

  r.bytes = total;
  r.gbps = r.seconds > 0 ? static_cast<double>(total) / 1e9 / r.seconds : 0.0;
  r.ok = total > 0;
  return r;
}

std::vector<std::string> mcclIfaceFilter() {
  const char* env = std::getenv("MCCL_IFACE_PRIORITY");
  if (env == nullptr || *env == '\0') return {"bridge"};
  std::vector<std::string> tokens;
  std::string tok;
  for (const char* p = env;; ++p) {
    if (*p == ',' || *p == '\0') {
      if (!tok.empty()) tokens.push_back(tok);
      tok.clear();
      if (*p == '\0') break;
    } else {
      tok.push_back(*p);
    }
  }
  return tokens.empty() ? std::vector<std::string>{"bridge"} : tokens;
}

int mcclLocalThunderboltLinks(std::vector<std::string>* matches = nullptr) {
  const std::vector<std::string> filter = mcclIfaceFilter();
  struct ifaddrs* ifaddr = nullptr;
  if (getifaddrs(&ifaddr) != 0) return 0;
  std::vector<std::string> seen;
  for (struct ifaddrs* ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr || ifa->ifa_addr->sa_family != AF_INET) continue;
    const std::string name = ifa->ifa_name ? ifa->ifa_name : "";
    if (name == "lo0") continue;
    char ip[INET_ADDRSTRLEN] = {0};
    auto* sin = reinterpret_cast<struct sockaddr_in*>(ifa->ifa_addr);
    inet_ntop(AF_INET, &sin->sin_addr, ip, sizeof(ip));
    const std::string addr(ip);

    bool match = false;
    for (const std::string& tok : filter)
      if (name.rfind(tok, 0) == 0 || addr.rfind(tok, 0) == 0) { match = true; break; }
    if (!match) continue;

    bool already = false;
    for (const std::string& s : seen)
      if (s == name) { already = true; break; }
    if (already) continue;
    seen.push_back(name);
    if (matches) matches->push_back(name + " (" + addr + ")");
  }
  freeifaddrs(ifaddr);
  return static_cast<int>(seen.size());
}

int mcclWorldSizeFromEnv() {
  const char* v = std::getenv("MCCL_WORLD_SIZE");
  if (v == nullptr) v = std::getenv("WORLD_SIZE");
  if (v == nullptr) return 0;
  const int n = std::atoi(v);
  return n > 0 ? n : 0;
}

namespace {
constexpr int kCalibrateDurationMs = 1000;

void setTbLinkBw(mcclTopoSystem* sys, int a, int b, float bw) {
  const int n = static_cast<int>(sys->nodes.size());
  if (a < 0 || a >= n || b < 0 || b >= n) return;
  for (mcclTopoLink& l : sys->nodes[static_cast<size_t>(a)].links) if (l.type == LINK_TB && l.remote == b) l.bw = bw;
  for (mcclTopoLink& l : sys->nodes[static_cast<size_t>(b)].links) if (l.type == LINK_TB && l.remote == a) l.bw = bw;
}
}

// Measure each live TB link and overwrite its table bw. Opt-in (MCCL_CALIBRATE) and must be set identically on
// every rank — it adds bootstrap rounds. Per edge the lower endpoint serves, the higher probes; the barrier
// makes all servers listen before any client dials, so a probe can't race ahead of its server.
mcclResult mcclCalibrateTbLinks(mcclTopoSystem* sys, const mcclEdge* edges, int nEdges,
                                int rank, int nRanks, const char* rootIp, uint16_t rootPort, uint16_t basePort) {
  if (sys == nullptr || rootIp == nullptr) return mcclInvalidArgument;
  const char* en = std::getenv("MCCL_CALIBRATE");
  if (en == nullptr || std::atoi(en) == 0) return mcclSuccess;
  if (nEdges <= 0 || edges == nullptr) return mcclSuccess;

  std::vector<std::thread> servers;
  for (int i = 0; i < nEdges; ++i)
    if (edges[i].live && edges[i].a == rank)
      servers.emplace_back([port = static_cast<uint16_t>(basePort + i)]() { mcclBandwidthServe(port, 15000); });

  int tok = rank;
  std::vector<int> bar(static_cast<size_t>(nRanks));
  MCCLCHECK(mcclBootstrapAllGather(rootIp, rootPort, rank, nRanks, &tok, bar.data(), sizeof(int)));  // barrier: servers up before clients dial

  std::vector<double> mine(static_cast<size_t>(nEdges), 0.0);
  for (int i = 0; i < nEdges; ++i)
    if (edges[i].live && edges[i].b == rank) {
      char ip[16];
      mcclIpStr(edges[i].ipA, ip);
      const mcclBandwidthResult r = mcclBandwidthProbe(ip, static_cast<uint16_t>(basePort + i), kCalibrateDurationMs);
      if (r.ok) mine[static_cast<size_t>(i)] = r.gbps;
    }
  for (std::thread& t : servers) t.join();

  std::vector<double> all(static_cast<size_t>(nRanks) * static_cast<size_t>(nEdges), 0.0);
  MCCLCHECK(mcclBootstrapAllGather(rootIp, rootPort, rank, nRanks, mine.data(), all.data(),
                                   static_cast<size_t>(nEdges) * sizeof(double)));
  for (int i = 0; i < nEdges; ++i) {
    double bw = 0.0;
    for (int rr = 0; rr < nRanks; ++rr)
      bw = std::max(bw, all[static_cast<size_t>(rr) * static_cast<size_t>(nEdges) + static_cast<size_t>(i)]);
    if (bw > 0.0) setTbLinkBw(sys, edges[i].a, edges[i].b, static_cast<float>(bw));
  }
  return mcclSuccess;
}

}

#ifdef MCCL_INIT_MAIN
int main() {
  std::vector<std::string> ifaces;
  const int tbLinks = mccl::mcclLocalThunderboltLinks(&ifaces);
  const int world = mccl::mcclWorldSizeFromEnv();

  std::printf("mccl init: cluster discovery\n");
  std::printf("  local Thunderbolt links: %d\n", tbLinks);
  for (const std::string& s : ifaces) std::printf("    - TB iface %s\n", s.c_str());

  if (world > 0)
    std::printf("  world size (from env): %d Mac(s)\n", world);
  else
    std::printf("  world size (from env): unset — assuming 1 Mac (single node)\n");
  const int numMacs = world > 0 ? world : 1;
  std::printf("  -> number of Macs in cluster: %d\n", numMacs);

  std::printf("\nmccl init: Thunderbolt calibration\n");
  const char* role = std::getenv("MCCL_CALIBRATE_ROLE");
  const char* peer = std::getenv("MCCL_CALIBRATE_PEER");
  const char* portStr = std::getenv("MCCL_CALIBRATE_PORT");
  long pv = portStr ? std::strtol(portStr, nullptr, 10) : 53535;
  if (pv < 1 || pv > 65535) {
    std::printf("  invalid MCCL_CALIBRATE_PORT '%s' — using 53535\n", portStr ? portStr : "");
    pv = 53535;
  }
  const uint16_t port = static_cast<uint16_t>(pv);

  if (role != nullptr && std::string(role) == "server") {
    std::printf("  role=server, listening on :%u ...\n", port);
    mccl::mcclBandwidthResult r = mccl::mcclBandwidthServe(port);
    if (r.ok)
      std::printf("  measured: %.2f GB/s (recv %llu bytes in %.3fs)\n",
                  r.gbps, (unsigned long long)r.bytes, r.seconds);
    else
      std::printf("  measurement failed (no client connected before timeout)\n");
  } else if (role != nullptr && std::string(role) == "client" && peer != nullptr) {
    std::printf("  role=client, probing %s:%u ...\n", peer, port);
    mccl::mcclBandwidthResult r = mccl::mcclBandwidthProbe(peer, port);
    if (r.ok)
      std::printf("  measured: %.2f GB/s (sent %llu bytes in %.3fs)\n",
                  r.gbps, (unsigned long long)r.bytes, r.seconds);
    else
      std::printf("  measurement failed (could not reach %s:%u)\n", peer, port);
  } else {
    std::printf("  no peer endpoint (set MCCL_CALIBRATE_ROLE + MCCL_CALIBRATE_PEER);\n");
    std::printf("  each Thunderbolt node's bw (GB/s) stays unmeasured until rendezvous\n");
    std::printf("  provides peers and calibration runs.\n");
  }
  return 0;
}
#endif
