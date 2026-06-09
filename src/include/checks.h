#pragma once

#include "../definitions.h"

namespace mccl {

void mcclCheckLog(const char* file, int line, mcclResult res);

}

#define MCCLCHECK(call) do {                                                           \
    mccl::mcclResult mcclCheckRes_ = (call);                                           \
    if (mcclCheckRes_ != mccl::mcclSuccess && mcclCheckRes_ != mccl::mcclInProgress) { \
      mccl::mcclCheckLog(__FILE__, __LINE__, mcclCheckRes_);                           \
      return mcclCheckRes_;                                                            \
    }                                                                                  \
  } while (0)

#define MCCLCHECKGOTO(call, res, label) do {                                           \
    (res) = (call);                                                                    \
    if ((res) != mccl::mcclSuccess && (res) != mccl::mcclInProgress) {                 \
      mccl::mcclCheckLog(__FILE__, __LINE__, (res));                                   \
      goto label;                                                                      \
    }                                                                                  \
  } while (0)
