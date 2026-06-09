#include "../include/trees.h"

#include "../include/checks.h"

namespace mccl {

mcclResult mcclGetBtree(int nRanks, int rank, int* up, int* d0, int* d1) {
  if (up == nullptr || d0 == nullptr || d1 == nullptr || nRanks < 1 || rank < 0 || rank >= nRanks)
    return mcclInvalidArgument;

  int bit = 1;  // lowest set bit of rank = its level; leaves (odd ranks) have bit 1, the root's subtrees higher
  while (bit < nRanks) { if (bit & rank) break; bit <<= 1; }

  if (rank == 0) {
    *up = -1;
    *d0 = -1;
    *d1 = nRanks > 1 ? bit >> 1 : -1;
    return mcclSuccess;
  }

  int u = (rank ^ bit) | (bit << 1);
  if (u >= nRanks) u = rank ^ bit;
  *up = u;

  int lowbit = bit >> 1;
  *d0 = lowbit == 0 ? -1 : rank - lowbit;
  int down1 = lowbit == 0 ? -1 : rank + lowbit;
  while (down1 >= nRanks) { lowbit >>= 1; down1 = lowbit == 0 ? -1 : rank + lowbit; }
  *d1 = down1;
  return mcclSuccess;
}

// t0 is the plain btree; t1 is a second btree built so leaves of one are interior in the other. Odd N: shift
// the btree by one rank. Even N: mirror it. Together they keep every rank's links busy in both trees.
mcclResult mcclGetDtree(int nRanks, int rank, int t0[3], int t1[3]) {
  if (t0 == nullptr || t1 == nullptr) return mcclInvalidArgument;
  MCCLCHECK(mcclGetBtree(nRanks, rank, &t0[0], &t0[1], &t0[2]));

  int u = -1, d0 = -1, d1 = -1;
  if (nRanks % 2 == 1) {
    const int shifted = (rank - 1 + nRanks) % nRanks;
    MCCLCHECK(mcclGetBtree(nRanks, shifted, &u, &d0, &d1));
    t1[0] = u  == -1 ? -1 : (u  + 1) % nRanks;
    t1[1] = d0 == -1 ? -1 : (d0 + 1) % nRanks;
    t1[2] = d1 == -1 ? -1 : (d1 + 1) % nRanks;
  } else {
    MCCLCHECK(mcclGetBtree(nRanks, nRanks - 1 - rank, &u, &d0, &d1));
    t1[0] = u  == -1 ? -1 : nRanks - 1 - u;
    t1[1] = d0 == -1 ? -1 : nRanks - 1 - d0;
    t1[2] = d1 == -1 ? -1 : nRanks - 1 - d1;
  }
  return mcclSuccess;
}

}

#ifdef MCCL_TREES_MAIN
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
  using namespace mccl;
  const int maxN = argc > 1 ? std::atoi(argv[1]) : 256;
  int bad = 0;
  for (int n = 1; n <= maxN; ++n) {
    int root0 = -1, root1 = -1;
    for (int r = 0; r < n; ++r) {
      int t0[3], t1[3];
      if (mcclGetDtree(n, r, t0, t1) != mcclSuccess) { std::printf("n=%d r=%d FAIL\n", n, r); ++bad; continue; }
      const bool leaf0 = t0[1] < 0 && t0[2] < 0;
      const bool leaf1 = t1[1] < 0 && t1[2] < 0;
      if (t0[0] < 0) root0 = r;
      if (t1[0] < 0) root1 = r;
      if (n > 2 && leaf0 && leaf1) { std::printf("n=%d r=%d leaf in BOTH trees\n", n, r); ++bad; }
    }
    if (n > 1 && root0 == root1) { std::printf("n=%d roots coincide (%d)\n", n, root0); ++bad; }
  }
  std::printf("dtree check 1..%d: %s\n", maxN, bad == 0 ? "OK" : "BAD");
  return bad == 0 ? 0 : 1;
}
#endif
