#pragma once

#include <cstddef>

namespace mccl {

enum mcclResult {
  mcclSuccess         = 0,
  mcclError           = 1,
  mcclSystemError     = 2,
  mcclInternalError   = 3,
  mcclInvalidArgument = 4,
  mcclInvalidUsage    = 5,
  mcclRemoteError     = 6,
  mcclInProgress      = 7,
  mcclNumResults      = 8
};

enum mcclDataType {
  mcclInt8     = 0,
  mcclUint8    = 1,
  mcclInt32    = 2,
  mcclUint32   = 3,
  mcclInt64    = 4,
  mcclUint64   = 5,
  mcclFloat16  = 6,
  mcclBfloat16 = 7,
  mcclFloat32  = 8,
  mcclFloat64  = 9,
  mcclNumTypes = 10
};

enum mcclRedOp {
  mcclSum  = 0,
  mcclProd = 1,
  mcclMax  = 2,
  mcclMin  = 3,
  mcclAvg  = 4,
  mcclNumOps = 5
};

size_t      mcclDataSize(mcclDataType t);
const char* mcclDataTypeStr(mcclDataType t);
const char* mcclRedOpStr(mcclRedOp op);
const char* mcclResultStr(mcclResult r);

}
