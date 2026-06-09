#pragma once

#include <algorithm>
#include <vector>

namespace mccl {

// Children adjacency from a global parent[] (parent[root] < 0). Index by rank.
inline std::vector<std::vector<int>> treeChildren(const std::vector<int>& parent) {
  std::vector<std::vector<int>> ch(parent.size());
  for (int r = 0; r < static_cast<int>(parent.size()); ++r)
    if (parent[static_cast<size_t>(r)] >= 0) ch[static_cast<size_t>(parent[static_cast<size_t>(r)])].push_back(r);
  return ch;
}

// Ranks in node x's subtree (x included), sorted ascending — the canonical packing order shared by both ends.
inline std::vector<int> subtreeRanks(const std::vector<std::vector<int>>& ch, int x) {
  std::vector<int> out, stack{x};
  while (!stack.empty()) {
    const int u = stack.back();
    stack.pop_back();
    out.push_back(u);
    for (int c : ch[static_cast<size_t>(u)]) stack.push_back(c);
  }
  std::sort(out.begin(), out.end());
  return out;
}

}
