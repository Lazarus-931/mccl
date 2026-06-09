#pragma once

#include <cstdint>

#include "definitions.h"

namespace mccl {

struct mcclTopoSystem;
struct mcclEdge;

struct mcclBandwidthResult {
  bool     ok      = false;
  double   gbps    = 0.0;
  uint64_t bytes   = 0;
  double   seconds = 0.0;
};

mcclBandwidthResult mcclBandwidthServe(uint16_t port, int acceptTimeoutMs = 10000);
mcclBandwidthResult mcclBandwidthProbe(const char* host, uint16_t port, int durationMs = 1000);

mcclResult mcclCalibrateTbLinks(mcclTopoSystem* sys, const mcclEdge* edges, int nEdges,
                                int rank, int nRanks, const char* rootIp, uint16_t rootPort, uint16_t basePort);

}
