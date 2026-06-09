#pragma once

#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "../definitions.h"
#include "../include/param.h"
#include "metal.h"

namespace mccl {

MCCL_PARAM(MetalMinBytes);

template <typename T>
inline void cpuReduceT(T* d, const T* s, size_t n, mcclRedOp op) {
  switch (op) {
    case mcclSum:  for (size_t i = 0; i < n; ++i) d[i] = d[i] + s[i]; break;
    case mcclProd: for (size_t i = 0; i < n; ++i) d[i] = d[i] * s[i]; break;
    case mcclMax:  for (size_t i = 0; i < n; ++i) d[i] = d[i] > s[i] ? d[i] : s[i]; break;
    case mcclMin:  for (size_t i = 0; i < n; ++i) d[i] = d[i] < s[i] ? d[i] : s[i]; break;
    default: break;
  }
}

// bf16 has no portable CPU arithmetic; widen to float, operate, narrow back.
inline float    mcclBf16ToF32(uint16_t b) { const uint32_t u = static_cast<uint32_t>(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }
inline uint16_t mcclF32ToBf16(float f) {
  uint32_t u; std::memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return static_cast<uint16_t>((u >> 16) | 0x0040u);  // NaN: force a mantissa bit so truncation can't turn it into Inf
  u += 0x7fffu + ((u >> 16) & 1u);  // round to nearest even before truncating the low 16 bits
  return static_cast<uint16_t>(u >> 16);
}

inline void cpuReduceBf16(uint16_t* d, const uint16_t* s, size_t n, mcclRedOp op) {
  for (size_t i = 0; i < n; ++i) {
    const float a = mcclBf16ToF32(d[i]), b = mcclBf16ToF32(s[i]);
    float r = a;
    switch (op) {
      case mcclSum:  r = a + b; break;
      case mcclProd: r = a * b; break;
      case mcclMax:  r = a > b ? a : b; break;
      case mcclMin:  r = a < b ? a : b; break;
      default: break;
    }
    d[i] = mcclF32ToBf16(r);
  }
}

inline mcclResult cpuReduce(void* d, const void* s, size_t n, mcclDataType dt, mcclRedOp op) {
  switch (dt) {
    case mcclFloat32: cpuReduceT(static_cast<float*>(d),    static_cast<const float*>(s),    n, op); return mcclSuccess;
    case mcclFloat64: cpuReduceT(static_cast<double*>(d),   static_cast<const double*>(s),   n, op); return mcclSuccess;
    case mcclInt8:    cpuReduceT(static_cast<int8_t*>(d),   static_cast<const int8_t*>(s),   n, op); return mcclSuccess;
    case mcclUint8:   cpuReduceT(static_cast<uint8_t*>(d),  static_cast<const uint8_t*>(s),  n, op); return mcclSuccess;
    case mcclInt32:   cpuReduceT(static_cast<int32_t*>(d),  static_cast<const int32_t*>(s),  n, op); return mcclSuccess;
    case mcclUint32:  cpuReduceT(static_cast<uint32_t*>(d), static_cast<const uint32_t*>(s), n, op); return mcclSuccess;
    case mcclInt64:   cpuReduceT(static_cast<int64_t*>(d),  static_cast<const int64_t*>(s),  n, op); return mcclSuccess;
    case mcclUint64:  cpuReduceT(static_cast<uint64_t*>(d), static_cast<const uint64_t*>(s), n, op); return mcclSuccess;
    case mcclFloat16:  cpuReduceT(static_cast<_Float16*>(d), static_cast<const _Float16*>(s), n, op); return mcclSuccess;
    case mcclBfloat16: cpuReduceBf16(static_cast<uint16_t*>(d), static_cast<const uint16_t*>(s), n, op); return mcclSuccess;
    default: return mcclInvalidArgument;
  }
}

inline bool pageAligned(const void* p) {
  static const uintptr_t pg = static_cast<uintptr_t>(sysconf(_SC_PAGESIZE));
  return (reinterpret_cast<uintptr_t>(p) % pg) == 0;
}

// dst = op(dst, src). Metal only above MCCL_METAL_MIN_BYTES — below that, dispatch latency outweighs the
// arithmetic so the CPU wins (and UMA means the CPU reads the same buffer with no copy). CPU is also the
// fallback when there's no GPU kernel for the dtype/op.
inline mcclResult reduce(void* dst, const void* src, size_t count, mcclDataType dt, mcclRedOp op, bool gpu) {
  if (gpu && count * mcclDataSize(dt) >= static_cast<size_t>(mcclParamMetalMinBytes())) {
    const mcclResult rc = mcclMetalReduce(dst, src, count, dt, op);
    if (rc != mcclInvalidUsage) return rc;
  }
  return cpuReduce(dst, src, count, dt, op);
}

// dst = op(dst, src_0..src_{nSrc-1}), source k at srcBase + k*strideElems. Folds every source in one GPU
// dispatch (one buffer sweep) instead of nSrc read-modify-writes — the win is the tree hub's fan-in fold.
inline mcclResult reduceMulti(void* dst, const void* srcBase, size_t count, size_t nSrc, size_t strideElems,
                              mcclDataType dt, mcclRedOp op, bool gpu) {
  if (nSrc == 0 || count == 0) return mcclSuccess;
  if (gpu && count * mcclDataSize(dt) >= static_cast<size_t>(mcclParamMetalMinBytes())) {
    const mcclResult rc = mcclMetalReduceMulti(dst, srcBase, count, nSrc, strideElems, dt, op);
    if (rc != mcclInvalidUsage) return rc;
  }
  const char* base = static_cast<const char*>(srcBase);
  const size_t esz = mcclDataSize(dt);
  for (size_t k = 0; k < nSrc; ++k) {
    const mcclResult rc = cpuReduce(dst, base + k * strideElems * esz, count, dt, op);
    if (rc != mcclSuccess) return rc;
  }
  return mcclSuccess;
}

inline size_t chunkOffElems(size_t count, int n, int i) {
  return count * static_cast<size_t>(i) / static_cast<size_t>(n);
}

inline mcclResult cpuScale(void* buf, size_t count, mcclDataType dt, double factor) {
  switch (dt) {
    case mcclFloat32: { float*    b = static_cast<float*>(buf);    for (size_t i = 0; i < count; ++i) b[i] = static_cast<float>(b[i] * factor); return mcclSuccess; }
    case mcclFloat64: { double*   b = static_cast<double*>(buf);   for (size_t i = 0; i < count; ++i) b[i] = b[i] * factor; return mcclSuccess; }
    case mcclFloat16: { _Float16* b = static_cast<_Float16*>(buf); for (size_t i = 0; i < count; ++i) b[i] = static_cast<_Float16>(static_cast<float>(b[i]) * factor); return mcclSuccess; }
    case mcclBfloat16:{ uint16_t* b = static_cast<uint16_t*>(buf); for (size_t i = 0; i < count; ++i) b[i] = mcclF32ToBf16(static_cast<float>(mcclBf16ToF32(b[i]) * factor)); return mcclSuccess; }
    default: return mcclInvalidUsage;
  }
}

}
