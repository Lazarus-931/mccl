#include "topo.h"

#include <algorithm>
#include <vector>

namespace mccl {

namespace {
constexpr int   kInfHops = 1 << 20;
constexpr float kBigBw   = 1e30f;

// Better path: lower type wins, else higher bottleneck bw, else fewer hops.
bool better(const mcclTopoPath& cand, const mcclTopoPath& cur) {
  if (cand.type != cur.type) return cand.type < cur.type;
  if (cand.bw   != cur.bw)   return cand.bw   > cur.bw;
  return cand.count < cur.count;
}
}

mcclResult mcclTopoComputePaths(mcclTopoSystem* sys) {
  if (sys == nullptr) return mcclInvalidArgument;
  const int   n     = static_cast<int>(sys->nodes.size());
  const float lanBw = mcclLanBw();

  for (int s = 0; s < n; ++s) {
    std::vector<mcclTopoPath> dist(static_cast<size_t>(n));
    for (int d = 0; d < n; ++d) dist[static_cast<size_t>(d)].count = kInfHops;
    dist[static_cast<size_t>(s)] = mcclTopoPath{0, kBigBw, PATH_LOC, s, 0};

    bool changed = true;  // relax over TB links until stable (Bellman-Ford; N is small)
    while (changed) {
      changed = false;
      for (int u = 0; u < n; ++u) {
        const mcclTopoPath& du = dist[static_cast<size_t>(u)];
        if (du.type == PATH_DIS) continue;
        for (const mcclTopoLink& l : sys->nodes[static_cast<size_t>(u)].links) {
          if (l.type != LINK_TB) continue;
          const int v = l.remote;
          mcclTopoPath cand;
          cand.count    = du.count + 1;
          cand.bw       = std::min(du.bw, l.bw);
          cand.type     = (cand.count == 1) ? PATH_TB : PATH_TB_HOP;
          cand.firstHop = (u == s) ? v : du.firstHop;
          cand.firstIp  = (u == s) ? l.ipRemote : du.firstIp;
          if (better(cand, dist[static_cast<size_t>(v)])) { dist[static_cast<size_t>(v)] = cand; changed = true; }
        }
      }
    }

    if (sys->lanAllToAll) {  // a direct LAN hop reaches any Mac; wins only where TB is absent or slower
      for (int d = 0; d < n; ++d) {
        if (d == s) continue;
        const mcclTopoPath cand{1, lanBw, PATH_LAN, d, sys->nodes[static_cast<size_t>(d)].lanIp};
        if (better(cand, dist[static_cast<size_t>(d)])) dist[static_cast<size_t>(d)] = cand;
      }
    }
    sys->nodes[static_cast<size_t>(s)].paths = std::move(dist);
  }
  return mcclSuccess;
}

}
