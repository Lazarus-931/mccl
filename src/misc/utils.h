#pragma once

#include <cstdint>
#include <string>

namespace mccl {

enum class ChipTier : int { Base = 0, Pro = 1, Max = 2, Ultra = 3, Unknown = 9 };

struct ChipInfo {
  std::string brand;
  int         generation      = 0;
  ChipTier    tier            = ChipTier::Unknown;
  int         chipCap         = 0;
  int         pCores          = 0;
  int         eCores          = 0;
  int         gpuCores        = 0;
  uint64_t    unifiedMemBytes = 0;
  bool        isAppleSilicon  = false;
};

constexpr int kMcclMinChipCap = 30;

inline bool mcclChipSupported(const ChipInfo& c) { return c.chipCap >= kMcclMinChipCap; }

ChipInfo    mcclDiscoverChip();
std::string mcclChipDescribe(const ChipInfo& c);
const char* mcclChipTierStr(ChipTier t);

}
