#include "discover.h"

#include "../socket.h"
#include "../bootstrap.h"

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include <atomic>
#include <cstdio>
#include <cstdint>
#include <map>
#include <thread>

namespace mccl {

mcclResult mcclDiscoverInterfaces(mcclIfSet* out) {
  if (out == nullptr) return mcclInvalidArgument;
  out->n = 0;
  struct ifaddrs* ifs = nullptr;
  if (getifaddrs(&ifs) != 0) return mcclSystemError;
  for (struct ifaddrs* p = ifs; p != nullptr && out->n < kMaxIfs; p = p->ifa_next) {
    if (p->ifa_addr == nullptr || p->ifa_addr->sa_family != AF_INET) continue;
    if ((p->ifa_flags & IFF_UP) == 0 || (p->ifa_flags & IFF_LOOPBACK) != 0) continue;
    const uint32_t ip = ntohl(reinterpret_cast<sockaddr_in*>(p->ifa_addr)->sin_addr.s_addr);
    const uint32_t mask = p->ifa_netmask ? ntohl(reinterpret_cast<sockaddr_in*>(p->ifa_netmask)->sin_addr.s_addr) : 0;
    if ((ip >> 24) == 127) continue;     // 127/8 loopback
    if ((ip >> 16) == 0xA9FE) continue;  // 169.254/16 link-local (self-assigned, not a configured link)
    if (mask == 0) continue;
    out->ifs[out->n].ip = ip;
    out->ifs[out->n].mask = mask;
    out->n++;
  }
  freeifaddrs(ifs);
  return mcclSuccess;
}

mcclResult mcclBuildEdges(const mcclIfSet* sets, int nRanks, std::vector<mcclEdge>* edges, std::vector<uint32_t>* lanIp) {
  if (sets == nullptr || edges == nullptr || nRanks < 1) return mcclInvalidArgument;
  edges->clear();
  if (lanIp != nullptr) lanIp->assign(static_cast<size_t>(nRanks), 0);

  std::map<std::pair<uint32_t, uint32_t>, std::vector<std::pair<int, uint32_t>>> bySubnet;
  for (int r = 0; r < nRanks; ++r)
    for (int i = 0; i < sets[r].n; ++i) {
      const uint32_t ip = sets[r].ifs[i].ip, mask = sets[r].ifs[i].mask;
      bySubnet[{ip & mask, mask}].push_back({r, ip});
    }

  for (const auto& kv : bySubnet) {
    const auto& members = kv.second;
    // A point-to-point TB edge: exactly two distinct Macs with distinct IPs. Two identical IPs on a "shared"
    // subnet are a VM bridge numbered the same on each host, not a real link — dialing it loops back to self.
    if (members.size() == 2 && members[0].first != members[1].first &&
        members[0].second != members[1].second) {
      mcclEdge e;
      e.net = kv.first.first;
      e.mask = kv.first.second;
      e.a = members[0].first; e.ipA = members[0].second;
      e.b = members[1].first; e.ipB = members[1].second;
      if (e.a > e.b) { std::swap(e.a, e.b); std::swap(e.ipA, e.ipB); }
      edges->push_back(e);
    } else if (lanIp != nullptr && members.size() >= 2) {  // a subnet reaching every rank -> the all-to-all overlay / fallback medium
      std::vector<uint32_t> ipOf(static_cast<size_t>(nRanks), 0);
      for (const auto& m : members) if (ipOf[static_cast<size_t>(m.first)] == 0) ipOf[static_cast<size_t>(m.first)] = m.second;
      bool all = true;
      for (int r = 0; r < nRanks; ++r) if (ipOf[static_cast<size_t>(r)] == 0) { all = false; break; }
      if (all) *lanIp = ipOf;
    }
  }
  return mcclSuccess;
}

const char* mcclIpStr(uint32_t ip, char buf[16]) {
  std::snprintf(buf, 16, "%u.%u.%u.%u", (ip >> 24) & 255, (ip >> 16) & 255, (ip >> 8) & 255, ip & 255);
  return buf;
}

mcclResult mcclProbeLiveness(std::vector<mcclEdge>& edges, int rank, int nRanks,
                             const char* rootIp, uint16_t rootPort, uint16_t probePort) {
  if (rootIp == nullptr || nRanks < 1) return mcclInvalidArgument;
  const int nE = static_cast<int>(edges.size());
  if (nE == 0) return mcclSuccess;

  // Serve only if some prober will dial us (we are the b endpoint of an edge): two same-host ranks would
  // otherwise both bind probePort and the second would fail init for a server nobody needs.
  bool needServer = false;
  for (int i = 0; i < nE; ++i) if (edges[static_cast<size_t>(i)].b == rank) { needServer = true; break; }
  int lst = -1;
  if (mcclSocketListen(probePort, 16, &lst, nullptr) != mcclSuccess) {
    if (needServer) return mcclSystemError;
    lst = -1;
  }
  std::atomic<bool> stop{false};
  std::thread server;
  if (lst >= 0) server = std::thread([&]() {
    while (!stop.load()) {
      int fd = -1;
      if (mcclSocketAccept(lst, 200, &fd) != mcclSuccess) continue;
      int peer = -1;
      if (mcclSocketRecv(fd, &peer, sizeof(peer)) == mcclSuccess) mcclSocketSend(fd, &rank, sizeof(rank));
      mcclSocketClose(fd);
    }
  });

  std::vector<uint8_t> vote(static_cast<size_t>(nE), 0);  // per edge: 1 = live, 2 = dead
  for (int i = 0; i < nE; ++i) {
    if (edges[i].a != rank) continue;  // only the lower-rank endpoint probes each edge
    char ip[16];
    mcclIpStr(edges[i].ipB, ip);
    int fd = -1;
    bool live = false;
    if (mcclSocketConnect(ip, probePort, 1500, 2, &fd) == mcclSuccess) {
      int peer = -1;
      if (mcclSocketSend(fd, &rank, sizeof(rank)) == mcclSuccess &&
          mcclSocketRecv(fd, &peer, sizeof(peer)) == mcclSuccess && peer == edges[i].b)
        live = true;  // reached the expected peer (not ourselves via a same-IP bridge, nor a stranger)
      mcclSocketClose(fd);
    }
    vote[static_cast<size_t>(i)] = live ? 1 : 2;
  }

  // Serve until the vote all-gather completes — it only returns once EVERY rank has finished probing, so no
  // fixed grace window can cut off a slow prober (a rank with several dead edges probes for many seconds,
  // and a 500ms window falsely marked its live edges dead).
  std::vector<uint8_t> all(static_cast<size_t>(nRanks) * static_cast<size_t>(nE), 0);
  const mcclResult arc = mcclBootstrapAllGather(rootIp, rootPort, rank, nRanks, vote.data(), all.data(), static_cast<size_t>(nE));
  stop.store(true);
  if (server.joinable()) server.join();
  mcclSocketClose(lst);
  if (arc != mcclSuccess) return arc;
  for (int i = 0; i < nE; ++i)
    edges[static_cast<size_t>(i)].live = (all[static_cast<size_t>(edges[i].a) * static_cast<size_t>(nE) + i] == 1);
  return mcclSuccess;
}

}

#ifdef MCCL_DISCOVER_MAIN
#include "../bootstrap.h"
#include <cstdlib>

int main() {
  using namespace mccl;
  auto envI = [](const char* k, int d) { const char* v = std::getenv(k); return v ? std::atoi(v) : d; };
  const int      rank = envI("MCCL_RANK", 0);
  const int      n    = envI("MCCL_WORLD_SIZE", 1);
  const char*    ip   = std::getenv("MCCL_BOOTSTRAP_IP");
  const uint16_t port = static_cast<uint16_t>(envI("MCCL_BOOTSTRAP_PORT", 53700));

  mcclIfSet mine;
  mcclDiscoverInterfaces(&mine);
  char a[16], b[16];
  for (int i = 0; i < mine.n; ++i) {
    mcclIpStr(mine.ifs[i].ip, a);
    mcclIpStr(mine.ifs[i].ip & mine.ifs[i].mask, b);
    std::printf("[discover] rank %d if %s mask %08x net %s\n", rank, a, mine.ifs[i].mask, b);
  }

  std::vector<mcclIfSet> all(n > 0 ? static_cast<size_t>(n) : 1);
  if (mcclBootstrapAllGather(ip ? ip : "127.0.0.1", port, rank, n, &mine, all.data(), sizeof(mcclIfSet)) != mcclSuccess) {
    std::printf("[discover] rank %d allgather FAIL\n", rank);
    return 1;
  }
  std::vector<mcclEdge> edges;
  std::vector<uint32_t> lanIp;
  mcclBuildEdges(all.data(), n, &edges, &lanIp);
  mcclProbeLiveness(edges, rank, n, ip ? ip : "127.0.0.1", port, static_cast<uint16_t>(port + 1));
  if (rank == 0) {
    std::printf("[discover] edges (%zu) after liveness probe:\n", edges.size());
    for (const mcclEdge& e : edges) {
      char x[16], y[16], z[16];
      mcclIpStr(e.ipA, x);
      mcclIpStr(e.ipB, y);
      mcclIpStr(e.net, z);
      std::printf("  rank%d(%s) <-> rank%d(%s)  net %s  %s\n", e.a, x, e.b, y, z, e.live ? "LIVE" : "dead");
    }
    std::printf("[discover] lan-all-to-all: %s\n", (static_cast<int>(lanIp.size()) == n) ? "candidate" : "no");
  }
  return 0;
}
#endif
