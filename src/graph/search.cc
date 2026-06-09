#include "topo.h"
#include "../include/graph.h"

#include <algorithm>
#include <vector>

namespace mccl {

const char* mcclAlgoStr(int algo) {
  switch (algo) {
    case MCCL_ALGO_RING: return "ring";
    case MCCL_ALGO_TREE: return "tree";
    default:             return "?";
  }
}

// A Mac's strength as a relay/root tie-break: newer/bigger chip first, then more GPU cores, then more memory.
// All zero until chip discovery populates the node, in which case the tie falls through to link degree.
float mcclMacScore(const mcclTopoNode& node) {
  return 1000.0f * static_cast<float>(node.chipCap) + 10.0f * static_cast<float>(node.gpuCores) +
         static_cast<float>(node.umaGiB);
}

namespace {

mcclResult computeRing(const mcclTopoSystem& sys, mcclTopoGraph* out) {
  out->algo = MCCL_ALGO_RING;
  out->order.clear();
  out->bw = 0.0f;
  out->nChannels = 1;
  const int n = static_cast<int>(sys.nodes.size());
  if (n == 0) return mcclSuccess;
  if (n == 1) { out->order.push_back(0); return mcclSuccess; }

  // Rank-order ring, viable only if every consecutive pair is one hop apart; a gap leaves order empty so the
  // cost model can't pick ring. bw = the bottleneck direct link. (Max-bw reordering is a later refinement.)
  float minBw = 1e30f;
  for (int i = 0; i < n; ++i) {
    float b = 0.0f;
    if (!mcclTopoDirectLink(sys, i, (i + 1) % n, &b, nullptr)) return mcclSuccess;
    minBw = std::min(minBw, b);
  }
  out->order.resize(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) out->order[static_cast<size_t>(i)] = i;
  out->bw = minBw;
  return mcclSuccess;
}

mcclResult computeTree(const mcclTopoSystem& sys, mcclTopoGraph* out) {
  out->algo = MCCL_ALGO_TREE;
  out->order.clear();
  out->bw = 0.0f;
  out->nChannels = 1;
  const int n = static_cast<int>(sys.nodes.size());
  if (n == 0) return mcclSuccess;

  int root = 0;  // hub = most links, then strongest Mac; connect.cc builds the actual tree from it
  for (int i = 1; i < n; ++i) {
    const int di = mcclTopoDegree(sys, i), dr = mcclTopoDegree(sys, root);
    if (di > dr || (di == dr && mcclMacScore(sys.nodes[static_cast<size_t>(i)]) > mcclMacScore(sys.nodes[static_cast<size_t>(root)]))) root = i;
  }
  out->order.push_back(root);
  for (int i = 0; i < n; ++i) if (i != root) out->order.push_back(i);
  out->bw = mcclTopoTreeBottleneckBw(sys, root);
  return mcclSuccess;
}

}

mcclResult mcclTopoCompute(const mcclTopoSystem& sys, int algo, mcclTopoGraph* out) {
  if (out == nullptr) return mcclInvalidArgument;
  switch (algo) {
    case MCCL_ALGO_RING: return computeRing(sys, out);
    case MCCL_ALGO_TREE: return computeTree(sys, out);
    default:             return mcclInvalidArgument;
  }
}

}
