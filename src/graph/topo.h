#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "../definitions.h"

namespace mccl {

// UMA makes a whole Mac one node (no internal PCI/GPU hierarchy), so links are Mac-to-Mac: TB or shared subnet.
enum mcclLinkType { LINK_LOC = 0, LINK_TB = 1, LINK_LAN = 2 };
const char* mcclLinkTypeStr(mcclLinkType t);

// Path quality, ordered low = best: local, direct TB, multi-hop TB, shared LAN, unreachable.
enum mcclPathType { PATH_LOC = 0, PATH_TB = 1, PATH_TB_HOP = 2, PATH_LAN = 3, PATH_DIS = 4 };
const char* mcclPathTypeStr(mcclPathType t);

struct mcclTopoLink {
  mcclLinkType type     = LINK_LOC;
  float        bw       = 0.0f;
  int          remote   = -1;
  uint32_t     ipLocal  = 0;
  uint32_t     ipRemote = 0;
};

// Best path to a Mac (filled by paths.cc): hop count, bottleneck bw, worst hop type, and the next Mac to send to.
struct mcclTopoPath {
  int          count    = 0;
  float        bw       = 0.0f;
  mcclPathType type     = PATH_DIS;
  int          firstHop = -1;
  uint32_t     firstIp  = 0;
};

struct mcclTopoNode {
  int                       rank     = -1;
  std::string               host;
  int                       gpuCores = 0;
  uint64_t                  umaGiB   = 0;
  int                       chipCap  = 0;
  uint32_t                  lanIp    = 0;
  std::vector<mcclTopoLink> links;
  std::vector<mcclTopoPath> paths;
};

struct mcclTopoSystem {
  std::vector<mcclTopoNode> nodes;
  bool                      lanAllToAll = false;
};

float mcclTbGenBw(int gen);
float mcclLanBw();

struct mcclEdge;

mcclResult mcclTopoGetSystem(int nRanks, const mcclEdge* edges, int nEdges,
                             const uint32_t* lanIp, int tbGen, mcclTopoSystem* out);

// Fill every node's paths[dst]: widest-path relaxation over TB links (bottleneck bw), preferring better type,
// then higher bw, then fewer hops; plus a one-hop LAN path where the subnet reaches all Macs.
mcclResult mcclTopoComputePaths(mcclTopoSystem* sys);

mcclResult mcclTopoMaxBw(const mcclTopoSystem& sys, float* out);
mcclResult mcclTopoMinBw(const mcclTopoSystem& sys, float* out);
mcclResult mcclTopoTotalBw(const mcclTopoSystem& sys, float* out);
int        mcclTopoDegree(const mcclTopoSystem& sys, int rank);

// One-hop reach a->b (direct TB, else the LAN if all-to-all): fills bw + dial IP. False if b needs a relay.
bool  mcclTopoDirectLink(const mcclTopoSystem& sys, int a, int b, float* bw, uint32_t* ipB);
// True only if every Mac pair has a dedicated direct TB link — the regime where double-binary trees help
// (concurrent trees need private links; on a shared subnet they'd just contend).
bool  mcclTopoTbAllToAll(const mcclTopoSystem& sys);
float mcclTopoTreeBottleneckBw(const mcclTopoSystem& sys, int root);

}
