#pragma once

#include <cstddef>
#include <vector>

#include "../definitions.h"
#include "device.h"
#include "param.h"

namespace mccl {

struct mcclTopoSystem;
struct mcclTopoNode;

enum mcclAlgo { MCCL_ALGO_RING = 0, MCCL_ALGO_TREE = 1, MCCL_NUM_ALGOS = 2 };
const char* mcclAlgoStr(int algo);

// One algorithm's search result: the rank ordering (ring = cycle, tree = hub-first) and its bottleneck bw.
struct mcclTopoGraph {
  int              algo      = MCCL_ALGO_RING;
  std::vector<int> order;
  float            bw        = 0.0f;
  int              nChannels = 1;
};

float mcclMacScore(const mcclTopoNode& node);

mcclResult mcclTopoCompute(const mcclTopoSystem& sys, int algo, mcclTopoGraph* out);

// Turn the ring + tree search results into this rank's channel (ring prev/next + a hub tree, or a dtree when all-to-all).
mcclResult mcclTopoConnect(const mcclTopoSystem& sys, const mcclTopoGraph& ring,
                           const mcclTopoGraph& tree, int rank, mcclChannel* out);

mcclResult mcclTopoGetAlgoTime(const mcclTopoGraph& graph, int nRanks, size_t nBytes, float* timeUs);
mcclResult mcclGetAlgoInfo(const mcclTopoGraph graphs[MCCL_NUM_ALGOS], int nRanks, size_t nBytes, int* algo);
mcclResult mcclGetAlgoInfo(const mcclTopoGraph graphs[MCCL_NUM_ALGOS], int nRanks, size_t nBytes,
                           const int enabled[MCCL_NUM_ALGOS], int* algo);

mcclResult parseList(const char* str, const char* names[], int n, int* enabled);
mcclResult mcclGetEnabledAlgos(int enabled[MCCL_NUM_ALGOS]);

}
