#pragma once

#include <cstdint>
#include <vector>

#include "../definitions.h"

namespace mccl {

constexpr int kMaxIfs = 8;

struct mcclIfAddr {
  uint32_t ip   = 0;
  uint32_t mask = 0;
};

struct mcclIfSet {
  int        n = 0;
  mcclIfAddr ifs[kMaxIfs];
};

struct mcclEdge {
  int      a = -1, b = -1;
  uint32_t net = 0, mask = 0;
  uint32_t ipA = 0, ipB = 0;
  bool     live = false;
};

mcclResult mcclDiscoverInterfaces(mcclIfSet* out);

mcclResult mcclBuildEdges(const mcclIfSet* sets, int nRanks, std::vector<mcclEdge>* edges, std::vector<uint32_t>* lanIp);

mcclResult mcclProbeLiveness(std::vector<mcclEdge>& edges, int rank, int nRanks,
                             const char* rootIp, uint16_t rootPort, uint16_t probePort);

const char* mcclIpStr(uint32_t ipHostOrder, char buf[16]);

}
