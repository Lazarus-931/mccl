#include "../include/graph.h"
#include "../include/checks.h"
#include "../include/param.h"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <string>

namespace mccl {

DEFINE_MCCL_PARAM(HopLatencyUs, "HOP_LATENCY_US", 50);  // per-step latency (TB/TCP round trip dominated)

// Predicted time = latency*steps + nBytes/bw. Ring takes 2(n-1) steps, tree ~2*log2(n) — so latency favours
// the tree at scale, bandwidth favours the ring; the crossover is what mcclGetAlgoInfo picks on. bw<=0 = unusable.
mcclResult mcclTopoGetAlgoTime(const mcclTopoGraph& graph, int nRanks, size_t nBytes, float* timeUs) {
  if (timeUs == nullptr) return mcclInvalidArgument;
  if (graph.bw <= 0.0f || nRanks < 1) { *timeUs = -1.0f; return mcclSuccess; }

  const float hop = static_cast<float>(mcclParamHopLatencyUs());
  float steps;
  if (graph.algo == MCCL_ALGO_TREE) {
    int depth = 1;
    while ((1 << depth) < (nRanks > 1 ? nRanks : 1)) ++depth;
    steps = 2.0f * static_cast<float>(depth);
  } else {
    steps = 2.0f * static_cast<float>(nRanks > 1 ? nRanks - 1 : 0);
  }
  *timeUs = hop * steps + static_cast<float>(nBytes) / (1000.0f * graph.bw);
  return mcclSuccess;
}

mcclResult mcclGetAlgoInfo(const mcclTopoGraph graphs[MCCL_NUM_ALGOS], int nRanks, size_t nBytes,
                           const int enabled[MCCL_NUM_ALGOS], int* algo) {
  if (graphs == nullptr || enabled == nullptr || algo == nullptr) return mcclInvalidArgument;
  if (nRanks <= 1) { *algo = MCCL_ALGO_TREE; return mcclSuccess; }  // single rank: every collective is a local copy, channel is moot
  int best = -1;
  float bestT = 0.0f;
  for (int a = 0; a < MCCL_NUM_ALGOS; ++a) {
    if (!enabled[a]) continue;
    float t = -1.0f;
    mcclTopoGetAlgoTime(graphs[a], nRanks, nBytes, &t);
    if (t < 0.0f) continue;
    if (best < 0 || t < bestT) { best = a; bestT = t; }
  }
  if (best < 0) return mcclInvalidUsage;
  *algo = best;
  return mcclSuccess;
}

// Convenience overload that reads MCCL_ALGO every call; the per-call hot path uses the cached-mask overload above.
mcclResult mcclGetAlgoInfo(const mcclTopoGraph graphs[MCCL_NUM_ALGOS], int nRanks, size_t nBytes, int* algo) {
  int enabled[MCCL_NUM_ALGOS];
  MCCLCHECK(mcclGetEnabledAlgos(enabled));
  return mcclGetAlgoInfo(graphs, nRanks, nBytes, enabled, algo);
}

namespace {
bool ieq(const std::string& a, const char* b) {
  if (a.size() != std::strlen(b)) return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (std::tolower((unsigned char)a[i]) != std::tolower((unsigned char)b[i])) return false;
  return true;
}
}

mcclResult parseList(const char* str, const char* names[], int n, int* enabled) {
  if (names == nullptr || enabled == nullptr) return mcclInvalidArgument;
  if (str == nullptr || str[0] == '\0') { for (int i = 0; i < n; ++i) enabled[i] = 1; return mcclSuccess; }
  const bool invert = (str[0] == '^');
  const std::string s(invert ? str + 1 : str);
  for (int i = 0; i < n; ++i) enabled[i] = invert ? 1 : 0;
  for (size_t start = 0;;) {
    const size_t comma = s.find(',', start);
    const std::string tok = s.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
    if (!tok.empty())
      for (int i = 0; i < n; ++i)
        if (ieq(tok, names[i])) { enabled[i] = invert ? 0 : 1; break; }
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  return mcclSuccess;
}

mcclResult mcclGetEnabledAlgos(int enabled[MCCL_NUM_ALGOS]) {
  static const char* names[MCCL_NUM_ALGOS] = { "ring", "tree" };
  return parseList(std::getenv("MCCL_ALGO"), names, MCCL_NUM_ALGOS, enabled);
}

}
