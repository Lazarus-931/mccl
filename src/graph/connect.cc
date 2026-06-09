#include "topo.h"
#include "../include/graph.h"
#include "../include/trees.h"
#include "../include/param.h"

#include <vector>

namespace mccl {

DEFINE_MCCL_PARAM(DtreeMinRanks, "DTREE_MIN_RANKS", 8);  // double-binary tree kicks in past this many Macs
DEFINE_MCCL_PARAM(Dtree, "DTREE", 0);                    // 1 = force the double tree on (testing/opt-in); 0 = auto. Must match across ranks (it changes the global tree).

namespace {
float linkBw(const mcclTopoSystem& sys, int a, int b) {
  float bw = 0.0f;
  return mcclTopoDirectLink(sys, a, b, &bw, nullptr) ? bw : 0.0f;
}
}

mcclResult mcclTopoConnect(const mcclTopoSystem& sys, const mcclTopoGraph& ring,
                           const mcclTopoGraph& tree, int rank, mcclChannel* out) {
  if (out == nullptr) return mcclInvalidArgument;
  const int n = static_cast<int>(sys.nodes.size());
  *out = mcclChannel{};
  if (rank < 0 || rank >= n) return mcclInvalidArgument;

  if (static_cast<int>(ring.order.size()) == n && n > 1) {
    int pos = -1;
    for (int i = 0; i < n; ++i) if (ring.order[static_cast<size_t>(i)] == rank) pos = i;
    if (pos >= 0) {
      out->ring.next = ring.order[static_cast<size_t>((pos + 1) % n)];
      out->ring.prev = ring.order[static_cast<size_t>((pos - 1 + n) % n)];
      out->ring.userRanks.resize(static_cast<size_t>(n));
      for (int i = 0; i < n; ++i) out->ring.userRanks[static_cast<size_t>(i)] = ring.order[static_cast<size_t>((pos + i) % n)];
    }
  }

  std::vector<int>              parent(static_cast<size_t>(n), -1);
  std::vector<std::vector<int>> children(static_cast<size_t>(n));
  std::vector<int>              parentB(static_cast<size_t>(n), -1);
  std::vector<std::vector<int>> childrenB(static_cast<size_t>(n));
  const int root = tree.order.empty() ? 0 : tree.order[0];

  // Auto-fire only on a dedicated TB all-to-all mesh, where two concurrent trees have private links to use;
  // on a shared-subnet overlay they'd just contend, so the single pipelined tree is better there. MCCL_DTREE forces it.
  const bool useDtree = mcclParamDtree() != 0 ? (n > 1)
                        : (n > static_cast<int>(mcclParamDtreeMinRanks()) && mcclTopoTbAllToAll(sys));
  if (useDtree) {
    // Two complementary binary trees over ordering positions, mapped back to ranks: tree A = tree, tree B =
    // treeB. Every Mac is interior in exactly one tree, so running both keeps each link busy in both directions.
    const std::vector<int>& ord = tree.order;
    for (int p = 0; p < n; ++p) {
      int t0[3], t1[3];
      mcclGetDtree(n, p, t0, t1);
      const int r = ord[static_cast<size_t>(p)];
      if (t0[0] >= 0) parent[static_cast<size_t>(r)]  = ord[static_cast<size_t>(t0[0])];
      if (t0[1] >= 0) children[static_cast<size_t>(r)].push_back(ord[static_cast<size_t>(t0[1])]);
      if (t0[2] >= 0) children[static_cast<size_t>(r)].push_back(ord[static_cast<size_t>(t0[2])]);
      if (t1[0] >= 0) parentB[static_cast<size_t>(r)] = ord[static_cast<size_t>(t1[0])];
      if (t1[1] >= 0) childrenB[static_cast<size_t>(r)].push_back(ord[static_cast<size_t>(t1[1])]);
      if (t1[2] >= 0) childrenB[static_cast<size_t>(r)].push_back(ord[static_cast<size_t>(t1[2])]);
    }
  } else {  // Prim's: grow the max-bottleneck-bw spanning tree from the hub (a star collapses to the hub-tree)
    std::vector<bool>  inTree(static_cast<size_t>(n), false);
    std::vector<float> bestBw(static_cast<size_t>(n), -1.0f);
    std::vector<int>   bestFrom(static_cast<size_t>(n), -1);
    inTree[static_cast<size_t>(root)] = true;
    for (int v = 0; v < n; ++v) if (v != root) { bestBw[static_cast<size_t>(v)] = linkBw(sys, root, v); bestFrom[static_cast<size_t>(v)] = root; }
    for (int added = 1; added < n; ++added) {
      int pick = -1;
      float pb = -1.0f;
      for (int v = 0; v < n; ++v) if (!inTree[static_cast<size_t>(v)] && bestBw[static_cast<size_t>(v)] > pb) { pb = bestBw[static_cast<size_t>(v)]; pick = v; }
      if (pick < 0) break;
      inTree[static_cast<size_t>(pick)] = true;
      parent[static_cast<size_t>(pick)] = bestFrom[static_cast<size_t>(pick)];
      children[static_cast<size_t>(bestFrom[static_cast<size_t>(pick)])].push_back(pick);
      for (int v = 0; v < n; ++v)
        if (!inTree[static_cast<size_t>(v)]) {
          const float b = linkBw(sys, pick, v);
          if (b > bestBw[static_cast<size_t>(v)]) { bestBw[static_cast<size_t>(v)] = b; bestFrom[static_cast<size_t>(v)] = pick; }
        }
    }
  }
  out->tree.up    = parent[static_cast<size_t>(rank)];
  out->tree.down  = children[static_cast<size_t>(rank)];
  out->treeParent = parent;
  out->treeB.up   = parentB[static_cast<size_t>(rank)];
  out->treeB.down = childrenB[static_cast<size_t>(rank)];
  out->dtree      = useDtree;
  bool flat = true;  // depth-1 tree (every non-root attaches to the root) -> the efficient hub gather/scatter path
  for (int v = 0; v < n; ++v) if (v != root && parent[static_cast<size_t>(v)] != root) { flat = false; break; }
  out->flatTree = flat;
  return mcclSuccess;
}

}

#ifdef MCCL_CONNECT_MAIN
#include "discover.h"
#include <cstdio>
#include <cstdlib>

int main() {
  using namespace mccl;
  const char* nv = std::getenv("MCCL_TOPO_NODES");
  const int n = nv ? std::atoi(nv) : 16;
  std::vector<uint32_t> lanIp(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) lanIp[static_cast<size_t>(i)] = 0x0A000000u + static_cast<uint32_t>(i + 1);

  mcclTopoSystem sys;
  mcclTopoGetSystem(n, nullptr, 0, lanIp.data(), 4, &sys);
  mcclTopoComputePaths(&sys);
  mcclTopoGraph ring, tree;
  mcclTopoCompute(sys, MCCL_ALGO_RING, &ring);
  mcclTopoCompute(sys, MCCL_ALGO_TREE, &tree);

  std::vector<int> up(static_cast<size_t>(n), -2);
  std::vector<std::vector<int>> down(static_cast<size_t>(n));
  for (int r = 0; r < n; ++r) {
    mcclChannel ch;
    if (mcclTopoConnect(sys, ring, tree, r, &ch) != mcclSuccess) { std::printf("connect rank %d FAIL\n", r); return 1; }
    up[static_cast<size_t>(r)] = ch.tree.up;
    down[static_cast<size_t>(r)] = ch.tree.down;
  }

  int roots = 0, bad = 0;
  for (int r = 0; r < n; ++r) if (up[static_cast<size_t>(r)] < 0) ++roots;
  for (int r = 0; r < n; ++r)
    for (int c : down[static_cast<size_t>(r)])
      if (c < 0 || c >= n || up[static_cast<size_t>(c)] != r) { std::printf("rank %d child %d not mutual\n", r, c); ++bad; }
  std::vector<bool> seen(static_cast<size_t>(n), false);
  std::vector<int> stack;
  for (int r = 0; r < n; ++r) if (up[static_cast<size_t>(r)] < 0) stack.push_back(r);
  while (!stack.empty()) { const int u = stack.back(); stack.pop_back(); if (seen[static_cast<size_t>(u)]) continue; seen[static_cast<size_t>(u)] = true; for (int c : down[static_cast<size_t>(u)]) stack.push_back(c); }
  int reached = 0;
  for (int r = 0; r < n; ++r) if (seen[static_cast<size_t>(r)]) ++reached;
  const bool dtree = n > static_cast<int>(mcclParamDtreeMinRanks());
  std::printf("connect n=%d (%s): roots=%d reached=%d/%d mutual=%s -> %s\n",
              n, dtree ? "dtree" : "physical", roots, reached, n, bad == 0 ? "ok" : "BAD",
              (roots == 1 && reached == n && bad == 0) ? "OK" : "BAD");
  return (roots == 1 && reached == n && bad == 0) ? 0 : 1;
}
#endif
