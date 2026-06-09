#include "../definitions.h"

namespace mccl {

size_t mcclDataSize(mcclDataType t) {
  switch (t) {
    case mcclInt8:
    case mcclUint8:    return 1;
    case mcclFloat16:
    case mcclBfloat16: return 2;
    case mcclInt32:
    case mcclUint32:
    case mcclFloat32:  return 4;
    case mcclInt64:
    case mcclUint64:
    case mcclFloat64:  return 8;
    case mcclNumTypes: return 0;
  }
  return 0;
}

const char* mcclDataTypeStr(mcclDataType t) {
  switch (t) {
    case mcclInt8:     return "int8";
    case mcclUint8:    return "uint8";
    case mcclInt32:    return "int32";
    case mcclUint32:   return "uint32";
    case mcclInt64:    return "int64";
    case mcclUint64:   return "uint64";
    case mcclFloat16:  return "float16";
    case mcclBfloat16: return "bfloat16";
    case mcclFloat32:  return "float32";
    case mcclFloat64:  return "float64";
    case mcclNumTypes: return "?";
  }
  return "?";
}

const char* mcclRedOpStr(mcclRedOp op) {
  switch (op) {
    case mcclSum:    return "sum";
    case mcclProd:   return "prod";
    case mcclMax:    return "max";
    case mcclMin:    return "min";
    case mcclAvg:    return "avg";
    case mcclNumOps: return "?";
  }
  return "?";
}

const char* mcclResultStr(mcclResult r) {
  switch (r) {
    case mcclSuccess:         return "success";
    case mcclError:           return "error";
    case mcclSystemError:     return "system error";
    case mcclInternalError:   return "internal error";
    case mcclInvalidArgument: return "invalid argument";
    case mcclInvalidUsage:    return "invalid usage";
    case mcclRemoteError:     return "remote error";
    case mcclInProgress:      return "in progress";
    case mcclNumResults:      return "?";
  }
  return "?";
}

}
