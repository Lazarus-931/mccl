#pragma once

#include <cstddef>

#include "../definitions.h"

namespace mccl {

bool       mcclMetalAvailable();
mcclResult mcclMetalReduce(void* dst, const void* src, size_t count, mcclDataType dt, mcclRedOp op);

mcclResult mcclMetalReduceMulti(void* dst, const void* src, size_t count, size_t nSrc, size_t strideElems,
                                mcclDataType dt, mcclRedOp op);

}
