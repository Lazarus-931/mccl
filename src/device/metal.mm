#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "metal.h"

#include <mach-o/dyld.h>
#include <unistd.h>

#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>

namespace mccl {

namespace {

std::string metalDir() {
  if (const char* e = std::getenv("MCCL_METAL_DIR")) return e;
  char buf[4096];
  uint32_t sz = sizeof(buf);
  if (_NSGetExecutablePath(buf, &sz) == 0) {
    const std::string p(buf);
    const size_t s = p.find_last_of('/');
    if (s != std::string::npos) return p.substr(0, s);
  }
  return ".";
}

struct MetalCtx {
  id<MTLDevice>       device = nil;
  id<MTLCommandQueue> queue  = nil;
  id<MTLLibrary>      lib    = nil;
  std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;
  std::mutex pipeMu;  // the dtree runs two reduces at once; serialize pipeline-cache access
  bool ok = false;

  MetalCtx() {
    device = MTLCreateSystemDefaultDevice();
    if (device == nil) return;
    queue = [device newCommandQueue];
    const std::string path = metalDir() + "/reduce.metal";
    NSError* err = nil;
    NSString* src = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:path.c_str()]
                                              encoding:NSUTF8StringEncoding
                                                 error:&err];
    if (src == nil) return;
    lib = [device newLibraryWithSource:src options:[MTLCompileOptions new] error:&err];
    ok = (queue != nil && lib != nil);
  }
};

MetalCtx& ctx() {
  static MetalCtx c;
  return c;
}

id<MTLComputePipelineState> pipeline(MetalCtx& c, const std::string& name) {
  std::lock_guard<std::mutex> lk(c.pipeMu);
  const auto it = c.pipelines.find(name);
  if (it != c.pipelines.end()) return it->second;
  id<MTLFunction> fn = [c.lib newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
  if (fn == nil) return nil;
  NSError* err = nil;
  id<MTLComputePipelineState> ps = [c.device newComputePipelineStateWithFunction:fn error:&err];
  if (ps != nil) c.pipelines[name] = ps;
  return ps;
}

const char* dtypeSuffix(mcclDataType dt) {
  switch (dt) {
    case mcclFloat32:  return "f32";
    case mcclFloat16:  return "f16";
    case mcclBfloat16: return "bf16";
    case mcclInt32:    return "i32";
    default:           return nullptr;
  }
}

const char* opName(mcclRedOp op) {
  switch (op) {
    case mcclSum:  return "sum";
    case mcclProd: return "prod";
    case mcclMax:  return "max";
    case mcclMin:  return "min";
    default:       return nullptr;
  }
}

size_t pageSize() {
  static const size_t pg = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  return pg;
}

size_t pageRound(size_t b) {
  const size_t pg = pageSize();
  return ((b + pg - 1) / pg) * pg;
}

}

bool mcclMetalAvailable() { return ctx().ok; }

mcclResult mcclMetalReduceMulti(void* dst, const void* src, size_t count, size_t nSrc, size_t strideElems,
                                mcclDataType dt, mcclRedOp op) {
  if (dst == nullptr || src == nullptr) return mcclInvalidArgument;
  if (count == 0 || nSrc == 0) return mcclSuccess;
  MetalCtx& c = ctx();
  if (!c.ok) return mcclInvalidUsage;

  const char* suf = dtypeSuffix(dt);
  const char* on  = opName(op);
  if (suf == nullptr || on == nullptr) return mcclInvalidUsage;
  id<MTLComputePipelineState> ps = pipeline(c, std::string("reduce_") + on + "_" + suf);
  if (ps == nil) return mcclInternalError;

  // dst/src are usually slices inside larger page-aligned buffers. newBufferWithBytesNoCopy needs a page-aligned
  // base, so wrap the page the slice starts in and address the slice via the buffer offset (must be esz-aligned,
  // >=4) — zero-copy over UMA, no staging. A misaligned slice falls back to the CPU.
  const size_t esz = mcclDataSize(dt);
  const size_t pg = pageSize();
  const uintptr_t da = reinterpret_cast<uintptr_t>(dst), sa = reinterpret_cast<uintptr_t>(src);
  const uintptr_t dbase = da & ~(pg - 1), sbase = sa & ~(pg - 1);
  const size_t doff = da - dbase, soff = sa - sbase;
  const size_t need = esz < 4 ? 4 : esz;
  if (doff % need != 0 || soff % need != 0 || (strideElems * esz) % need != 0) return mcclInvalidUsage;
  const size_t srcSpan = soff + (nSrc - 1) * strideElems * esz + count * esz;
  __block mcclResult res = mcclSystemError;
  @autoreleasepool {  // collectives call this in a tight loop; drain the per-dispatch Metal objects each time
    id<MTLBuffer> dstBuf = [c.device newBufferWithBytesNoCopy:reinterpret_cast<void*>(dbase) length:pageRound(doff + count * esz) options:MTLResourceStorageModeShared deallocator:nil];
    id<MTLBuffer> srcBuf = [c.device newBufferWithBytesNoCopy:reinterpret_cast<void*>(sbase) length:pageRound(srcSpan) options:MTLResourceStorageModeShared deallocator:nil];
    if (dstBuf == nil || srcBuf == nil) return mcclSystemError;
    id<MTLCommandBuffer> cb = [c.queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ps];
    [enc setBuffer:dstBuf offset:doff atIndex:0];
    [enc setBuffer:srcBuf offset:soff atIndex:1];
    uint n = static_cast<uint>(count), ns = static_cast<uint>(nSrc), st = static_cast<uint>(strideElems);
    [enc setBytes:&n  length:sizeof(n)  atIndex:2];
    [enc setBytes:&ns length:sizeof(ns) atIndex:3];
    [enc setBytes:&st length:sizeof(st) atIndex:4];
    NSUInteger tg = ps.maxTotalThreadsPerThreadgroup;
    if (tg > count) tg = count;
    [enc dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    res = cb.status == MTLCommandBufferStatusCompleted ? mcclSuccess : mcclSystemError;
  }
  return res;
}

mcclResult mcclMetalReduce(void* dst, const void* src, size_t count, mcclDataType dt, mcclRedOp op) {
  return mcclMetalReduceMulti(dst, src, count, 1, count, dt, op);
}

}

#ifdef MCCL_METAL_MAIN
#include "../include/alloc.h"
#include <cstdio>

int main() {
  using namespace mccl;
  if (!mcclMetalAvailable()) { std::printf("[metal] unavailable (no device or reduce.metal not found)\n"); return 1; }

  const size_t count = 4096;
  void* d = nullptr;
  void* s = nullptr;
  if (mcclPageAlloc(count * sizeof(float), &d) != mcclSuccess ||
      mcclPageAlloc(count * sizeof(float), &s) != mcclSuccess) { std::printf("alloc failed\n"); return 1; }
  float* df = static_cast<float*>(d);
  float* sf = static_cast<float*>(s);

  struct Case { mcclRedOp op; float dv, sv, expect; const char* nm; };
  const Case cases[] = {
      {mcclSum, 1.0f, 2.0f, 3.0f, "sum"},
      {mcclProd, 3.0f, 4.0f, 12.0f, "prod"},
      {mcclMax, 1.0f, 5.0f, 5.0f, "max"},
      {mcclMin, 1.0f, 5.0f, 1.0f, "min"},
  };
  int fails = 0;
  for (const Case& cs : cases) {
    for (size_t i = 0; i < count; ++i) { df[i] = cs.dv; sf[i] = cs.sv; }
    const mcclResult rc = mcclMetalReduce(d, s, count, mcclFloat32, cs.op);
    bool ok = (rc == mcclSuccess);
    for (size_t i = 0; ok && i < count; ++i) if (df[i] != cs.expect) ok = false;
    std::printf("[metal] f32 %-4s rc=%d -> %s\n", cs.nm, static_cast<int>(rc), ok ? "OK" : "BAD");
    fails += !ok;
  }

  int* di = static_cast<int*>(d);
  int* si = static_cast<int*>(s);
  for (size_t i = 0; i < count; ++i) { di[i] = 10; si[i] = 7; }
  const mcclResult rc = mcclMetalReduce(d, s, count, mcclInt32, mcclSum);
  bool ok = (rc == mcclSuccess);
  for (size_t i = 0; ok && i < count; ++i) if (di[i] != 17) ok = false;
  std::printf("[metal] i32 sum  rc=%d -> %s\n", static_cast<int>(rc), ok ? "OK" : "BAD");
  fails += !ok;

  for (size_t i = 0; i < count; ++i) { df[i] = 1.0f; sf[i] = 2.0f; }
  const size_t off = 1024;
  const mcclResult orc = mcclMetalReduce(df + off, sf, 512, mcclFloat32, mcclSum);
  bool ook = (orc == mcclSuccess);
  for (size_t i = 0; ook && i < count; ++i) {
    const float want = (i >= off && i < off + 512) ? 3.0f : 1.0f;
    if (df[i] != want) ook = false;
  }
  std::printf("[metal] f32 offset-sum rc=%d -> %s\n", static_cast<int>(orc), ook ? "OK" : "BAD");
  fails += !ook;

  mcclPageFree(d);
  mcclPageFree(s);
  return fails ? 1 : 0;
}
#endif
