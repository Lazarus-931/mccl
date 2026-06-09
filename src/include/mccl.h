#pragma once

// mccl: topology-adaptive collectives for Apple-Silicon clusters over unified memory + Thunderbolt.
// Include this one header for the whole public API.
#include "../definitions.h"
#include "alloc.h"
#include "comm.h"
#include "coll.h"

namespace mccl {

typedef mcclComm*  mcclComm_t;
typedef mcclConfig mcclConfig_t;

inline const char* mcclGetErrorString(mcclResult r) { return mcclResultStr(r); }

}
