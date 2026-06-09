#pragma once

#include "../definitions.h"

namespace mccl {

mcclResult mcclGetBtree(int nRanks, int rank, int* up, int* d0, int* d1);

mcclResult mcclGetDtree(int nRanks, int rank, int t0[3], int t1[3]);

}
