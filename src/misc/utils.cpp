#include <sys/sysctl.h>
#include <sys/types.h>
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <sstream>
#include <string>

#include "utils.h"

namespace mccl {

namespace {

std::string sysctlStr(const char* name) {
  size_t len = 0;
  if (sysctlbyname(name, nullptr, &len, nullptr, 0) != 0 || len == 0) return {};
  std::string buf(len, '\0');
  if (sysctlbyname(name, &buf[0], &len, nullptr, 0) != 0) return {};
  while (!buf.empty() && buf.back() == '\0') buf.pop_back();
  return buf;
}

int64_t sysctlInt(const char* name, int64_t fallback = 0) {
  int64_t v = 0;
  size_t len = sizeof(v);
  if (sysctlbyname(name, &v, &len, nullptr, 0) == 0 && len == sizeof(v)) return v;
  int32_t v32 = 0;
  len = sizeof(v32);
  if (sysctlbyname(name, &v32, &len, nullptr, 0) == 0) return v32;
  return fallback;
}

ChipTier parseTier(const std::string& brand) {
  if (brand.find("Ultra") != std::string::npos) return ChipTier::Ultra;
  if (brand.find("Max")   != std::string::npos) return ChipTier::Max;
  if (brand.find("Pro")   != std::string::npos) return ChipTier::Pro;
  return ChipTier::Base;
}

int parseGeneration(const std::string& brand) {
  for (size_t p = brand.find('M'); p != std::string::npos; p = brand.find('M', p + 1))
    if (p + 1 < brand.size() && std::isdigit(static_cast<unsigned char>(brand[p + 1])))
      return brand[p + 1] - '0';
  return 0;
}

// GPU core count isn't a sysctl; it lives in the IORegistry as the IOAccelerator's "gpu-core-count" property.
int gpuCoreCount() {
  int cores = 0;
  io_iterator_t it = IO_OBJECT_NULL;
  if (IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("IOAccelerator"), &it) != KERN_SUCCESS) return 0;
  for (io_object_t o = IOIteratorNext(it); o != IO_OBJECT_NULL; o = IOIteratorNext(it)) {
    CFTypeRef p = IORegistryEntryCreateCFProperty(o, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
    if (p != nullptr) {
      if (CFGetTypeID(p) == CFNumberGetTypeID()) CFNumberGetValue(static_cast<CFNumberRef>(p), kCFNumberIntType, &cores);
      CFRelease(p);
    }
    IOObjectRelease(o);
    if (cores > 0) break;
  }
  IOObjectRelease(it);
  return cores;
}

}

ChipInfo mcclDiscoverChip() {
  ChipInfo c;
  c.brand = sysctlStr("machdep.cpu.brand_string");
  c.isAppleSilicon = sysctlInt("hw.optional.arm64", 0) == 1;

  if (c.isAppleSilicon && c.brand.rfind("Apple M", 0) == 0) {
    c.generation = parseGeneration(c.brand);
    c.tier = parseTier(c.brand);
    if (c.generation > 0) c.chipCap = c.generation * 10 + static_cast<int>(c.tier);
  }

  c.pCores = static_cast<int>(
      sysctlInt("hw.perflevel0.physicalcpu", sysctlInt("hw.physicalcpu")));
  c.eCores = static_cast<int>(sysctlInt("hw.perflevel1.physicalcpu", 0));
  c.unifiedMemBytes = static_cast<uint64_t>(sysctlInt("hw.memsize", 0));
  c.gpuCores = gpuCoreCount();
  return c;
}

std::string mcclChipDescribe(const ChipInfo& c) {
  std::ostringstream os;
  if (!c.isAppleSilicon) {
    os << "non-Apple-Silicon CPU (\"" << c.brand << "\") — unsupported";
    return os.str();
  }
  os << c.brand << " (chipCap=" << c.chipCap << ", " << c.pCores << "P+" << c.eCores << "E";
  if (c.gpuCores) os << ", " << c.gpuCores << "-core GPU";
  if (c.unifiedMemBytes) os << ", " << (c.unifiedMemBytes >> 30) << " GB UMA";
  os << ")";
  if (!mcclChipSupported(c))
    os << "  [UNSUPPORTED: mccl requires M3+ (chipCap>=" << kMcclMinChipCap << ")]";
  return os.str();
}

const char* mcclChipTierStr(ChipTier t) {
  switch (t) {
    case ChipTier::Base:    return "base";
    case ChipTier::Pro:     return "Pro";
    case ChipTier::Max:     return "Max";
    case ChipTier::Ultra:   return "Ultra";
    case ChipTier::Unknown: return "unknown";
  }
  return "unknown";
}

}

#ifdef MCCL_UTILS_MAIN
int main() {
  const mccl::ChipInfo c = mccl::mcclDiscoverChip();
  std::printf("mccl chip discovery\n  %s\n  supported: %s\n",
              mccl::mcclChipDescribe(c).c_str(),
              mccl::mcclChipSupported(c) ? "yes" : "no");
  return mccl::mcclChipSupported(c) ? 0 : 1;
}
#endif
