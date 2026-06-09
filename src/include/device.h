#pragma once

#include <vector>

namespace mccl {

constexpr int MCCL_MAX_CHANNELS   = 64;
constexpr int MCCL_MAX_TREE_ARITY = 3;  // dtree uses <=2 children; a physical hub fans out via the vector

// This rank's place in one ring: neighbours + the rank order walked, rotated so userRanks[0] = self.
struct mcclRing {
  int              prev = -1;
  int              next = -1;
  std::vector<int> userRanks;
};

// This rank's place in one tree: parent (up, -1 at root) + children (down).
struct mcclTree {
  int              up = -1;
  std::vector<int> down;
};

// A ring view and one or two tree views of the same ranks; the executor uses whichever the chosen algo needs.
struct mcclChannel {
  mcclRing         ring;
  mcclTree         tree;
  mcclTree         treeB;
  std::vector<int> treeParent;  // tree A's parent for every rank (global), so any node can derive subtree membership
  bool             flatTree = false;  // depth-1 tree: enables the efficient hub gather/scatter
  bool             dtree    = false;  // tree + treeB form a double-binary tree (run both for full-duplex)
};

}
