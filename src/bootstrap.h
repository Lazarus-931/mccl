#pragma once

#include <cstddef>
#include <cstdint>

#include "definitions.h"

namespace mccl {

mcclResult mcclBootstrapAllGather(const char* rootIp, uint16_t rootPort, int rank, int nRanks,
                                  const void* sendData, void* recvData, size_t perRankBytes);

}
