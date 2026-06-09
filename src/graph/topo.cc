#include "topo.h"

#include "discover.h"

#include <cstdlib>

namespace mccl {

const char* mcclLinkTypeStr(mcclLinkType t) {
  switch (t) {
    case LINK_TB:  return "TB";
    case LINK_LAN: return "LAN";
    default:       return "LOC";
  }
}

const char* mcclPathTypeStr(mcclPathType t) {
  switch (t) {
    case PATH_LOC:    return "LOC";
    case PATH_TB:     return "TB";
    case PATH_TB_HOP: return "TB_HOP";
    case PATH_LAN:    return "LAN";
    default:          return "DIS";
  }
}

// GB/s per Thunderbolt link by generation; a starting guess that MCCL_CALIBRATE (init.cc) can replace with a
// measurement, or MCCL_TB_BW can override outright.
float mcclTbGenBw(int gen) {
  if (const char* e = std::getenv("MCCL_TB_BW")) { const double v = std::atof(e); if (v > 0.0) return static_cast<float>(v); }
  switch (gen) {
    case 3:  return 2.7f;
    case 5:  return 11.0f;
    default: return 4.7f;  // TB4
  }
}

float mcclLanBw() {
  if (const char* e = std::getenv("MCCL_LAN_BW")) { const double v = std::atof(e); if (v > 0.0) return static_cast<float>(v); }
  return 1.0f;  // shared-subnet baseline, deliberately below any TB link so the cost model prefers TB
}

mcclResult mcclTopoGetSystem(int nRanks, const mcclEdge* edges, int nEdges,
                             const uint32_t* lanIp, int tbGen, mcclTopoSystem* out) {
  if (out == nullptr || nRanks < 1 || (nEdges > 0 && edges == nullptr)) return mcclInvalidArgument;
  out->nodes.clear();
  out->nodes.resize(static_cast<size_t>(nRanks));
  const float tbBw = mcclTbGenBw(tbGen);
  for (int r = 0; r < nRanks; ++r) {
    out->nodes[static_cast<size_t>(r)].rank = r;
    if (lanIp != nullptr) out->nodes[static_cast<size_t>(r)].lanIp = lanIp[r];
  }

  for (int i = 0; i < nEdges; ++i) {
    const mcclEdge& e = edges[i];
    if (!e.live || e.a < 0 || e.a >= nRanks || e.b < 0 || e.b >= nRanks) continue;
    out->nodes[static_cast<size_t>(e.a)].links.push_back(mcclTopoLink{LINK_TB, tbBw, e.b, e.ipA, e.ipB});
    out->nodes[static_cast<size_t>(e.b)].links.push_back(mcclTopoLink{LINK_TB, tbBw, e.a, e.ipB, e.ipA});
  }

  out->lanAllToAll = (lanIp != nullptr);
  if (out->lanAllToAll)
    for (int r = 0; r < nRanks; ++r)
      if (out->nodes[static_cast<size_t>(r)].lanIp == 0) { out->lanAllToAll = false; break; }
  return mcclSuccess;
}

mcclResult mcclTopoMaxBw(const mcclTopoSystem& sys, float* out) {
  if (out == nullptr) return mcclInvalidArgument;
  float m = 0.0f;
  for (const mcclTopoNode& n : sys.nodes)
    for (const mcclTopoLink& l : n.links)
      if (l.type == LINK_TB && l.bw > m) m = l.bw;
  *out = m;
  return mcclSuccess;
}

mcclResult mcclTopoMinBw(const mcclTopoSystem& sys, float* out) {
  if (out == nullptr) return mcclInvalidArgument;
  float m = 0.0f;
  bool any = false;
  for (const mcclTopoNode& n : sys.nodes)
    for (const mcclTopoLink& l : n.links)
      if (l.type == LINK_TB && (!any || l.bw < m)) { m = l.bw; any = true; }
  *out = any ? m : 0.0f;
  return mcclSuccess;
}

mcclResult mcclTopoTotalBw(const mcclTopoSystem& sys, float* out) {
  if (out == nullptr) return mcclInvalidArgument;
  float t = 0.0f;
  for (const mcclTopoNode& n : sys.nodes)
    for (const mcclTopoLink& l : n.links)
      if (l.type == LINK_TB) t += l.bw;
  *out = t / 2.0f;  // each undirected edge is stored at both endpoints
  return mcclSuccess;
}

int mcclTopoDegree(const mcclTopoSystem& sys, int rank) {
  if (rank < 0 || rank >= static_cast<int>(sys.nodes.size())) return 0;
  int d = 0;
  for (const mcclTopoLink& l : sys.nodes[static_cast<size_t>(rank)].links)
    if (l.type == LINK_TB) ++d;
  return d;
}

bool mcclTopoDirectLink(const mcclTopoSystem& sys, int a, int b, float* bw, uint32_t* ipB) {
  const int n = static_cast<int>(sys.nodes.size());
  if (a < 0 || a >= n || b < 0 || b >= n || a == b) return false;
  for (const mcclTopoLink& l : sys.nodes[static_cast<size_t>(a)].links)
    if (l.type == LINK_TB && l.remote == b) { if (bw) *bw = l.bw; if (ipB) *ipB = l.ipRemote; return true; }
  if (sys.lanAllToAll && sys.nodes[static_cast<size_t>(a)].lanIp != 0 && sys.nodes[static_cast<size_t>(b)].lanIp != 0) {
    if (bw) *bw = mcclLanBw();
    if (ipB) *ipB = sys.nodes[static_cast<size_t>(b)].lanIp;
    return true;
  }
  return false;
}

bool mcclTopoTbAllToAll(const mcclTopoSystem& sys) {
  const int n = static_cast<int>(sys.nodes.size());
  if (n < 2) return false;
  for (int a = 0; a < n; ++a)
    for (int b = 0; b < n; ++b)
      if (a != b) {
        bool direct = false;
        for (const mcclTopoLink& l : sys.nodes[static_cast<size_t>(a)].links)
          if (l.type == LINK_TB && l.remote == b) { direct = true; break; }
        if (!direct) return false;
      }
  return true;
}

float mcclTopoTreeBottleneckBw(const mcclTopoSystem& sys, int root) {
  const int n = static_cast<int>(sys.nodes.size());
  if (n <= 1 || root < 0 || root >= n) return 0.0f;
  std::vector<bool>  inTree(static_cast<size_t>(n), false);
  std::vector<float> best(static_cast<size_t>(n), 0.0f);
  inTree[static_cast<size_t>(root)] = true;
  for (int v = 0; v < n; ++v) { float b = 0.0f; if (v != root && mcclTopoDirectLink(sys, root, v, &b, nullptr)) best[static_cast<size_t>(v)] = b; }
  float bottleneck = 1e30f;
  for (int added = 1; added < n; ++added) {
    int pick = -1;
    float pb = 0.0f;
    for (int v = 0; v < n; ++v) if (!inTree[static_cast<size_t>(v)] && best[static_cast<size_t>(v)] > pb) { pb = best[static_cast<size_t>(v)]; pick = v; }
    if (pick < 0) return 0.0f;
    inTree[static_cast<size_t>(pick)] = true;
    bottleneck = bottleneck < pb ? bottleneck : pb;
    for (int v = 0; v < n; ++v) {
      float b = 0.0f;
      if (!inTree[static_cast<size_t>(v)] && mcclTopoDirectLink(sys, pick, v, &b, nullptr) && b > best[static_cast<size_t>(v)]) best[static_cast<size_t>(v)] = b;
    }
  }
  return bottleneck;
}

}
