#include <metal_stdlib>
using namespace metal;

// Fold nSrc sources into dst in one dispatch: dst[i] = op(dst[i], src_0[i], .., src_{nSrc-1}[i]) — the
// accumulate step a ring/tree collective runs, with the hub folding every child at once (source k starts
// at src[k*stride], stride in elements). nSrc=1 is the plain binary reduce dst[i] = op(dst[i], src[i]).
#define MCCL_REDUCE(NAME, T, EXPR)                       \
  kernel void NAME(device T* dst         [[buffer(0)]],  \
                   device const T* src   [[buffer(1)]],  \
                   constant uint& n      [[buffer(2)]],  \
                   constant uint& nSrc   [[buffer(3)]],  \
                   constant uint& stride [[buffer(4)]],  \
                   uint i [[thread_position_in_grid]]) { \
    if (i >= n) return;                                  \
    T a = dst[i];                                        \
    for (uint k = 0; k < nSrc; ++k) { T b = src[k * stride + i]; a = (EXPR); } \
    dst[i] = a;                                          \
  }

#define MCCL_OPS(T, SUF)                  \
  MCCL_REDUCE(reduce_sum_##SUF,  T, a + b) \
  MCCL_REDUCE(reduce_prod_##SUF, T, a * b) \
  MCCL_REDUCE(reduce_max_##SUF,  T, (a > b ? a : b)) \
  MCCL_REDUCE(reduce_min_##SUF,  T, (a < b ? a : b))

MCCL_OPS(float,  f32)
MCCL_OPS(half,   f16)
MCCL_OPS(bfloat, bf16)
MCCL_OPS(int,    i32)
