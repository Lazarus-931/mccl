#include "rdma.h"

#if __has_include(<infiniband/verbs.h>)

#include <dlfcn.h>
#include <infiniband/verbs.h>

#include <chrono>
#include <cstring>

namespace mccl {
namespace {

constexpr int kPollTimeoutMs = 10000;

struct IbvApi {
  void* handle = nullptr;
  ibv_device** (*get_device_list)(int*) = nullptr;
  const char* (*get_device_name)(ibv_device*) = nullptr;
  ibv_context* (*open_device)(ibv_device*) = nullptr;
  void (*free_device_list)(ibv_device**) = nullptr;
  int (*close_device)(ibv_context*) = nullptr;
  ibv_pd* (*alloc_pd)(ibv_context*) = nullptr;
  ibv_qp* (*create_qp)(ibv_pd*, ibv_qp_init_attr*) = nullptr;
  ibv_cq* (*create_cq)(ibv_context*, int, void*, ibv_comp_channel*, int) = nullptr;
  int (*destroy_cq)(ibv_cq*) = nullptr;
  int (*destroy_qp)(ibv_qp*) = nullptr;
  int (*dealloc_pd)(ibv_pd*) = nullptr;
  int (*query_port)(ibv_context*, uint8_t, ibv_port_attr*) = nullptr;
  int (*query_gid)(ibv_context*, uint8_t, int, ibv_gid*) = nullptr;
  int (*modify_qp)(ibv_qp*, ibv_qp_attr*, int) = nullptr;
  ibv_mr* (*reg_mr)(ibv_pd*, void*, size_t, int) = nullptr;
  int (*dereg_mr)(ibv_mr*) = nullptr;

  IbvApi() {
    handle = dlopen("librdma.dylib", RTLD_NOW | RTLD_GLOBAL);
    if (handle == nullptr) return;
    bool ok = true;
#define MCCL_DLSYM(f)                                              \
  do {                                                             \
    f = reinterpret_cast<decltype(f)>(dlsym(handle, "ibv_" #f));   \
    if (f == nullptr) ok = false;                                  \
  } while (0)
    MCCL_DLSYM(get_device_list);
    MCCL_DLSYM(get_device_name);
    MCCL_DLSYM(open_device);
    MCCL_DLSYM(free_device_list);
    MCCL_DLSYM(close_device);
    MCCL_DLSYM(alloc_pd);
    MCCL_DLSYM(create_qp);
    MCCL_DLSYM(create_cq);
    MCCL_DLSYM(destroy_cq);
    MCCL_DLSYM(destroy_qp);
    MCCL_DLSYM(dealloc_pd);
    MCCL_DLSYM(query_port);
    MCCL_DLSYM(query_gid);
    MCCL_DLSYM(modify_qp);
    MCCL_DLSYM(reg_mr);
    MCCL_DLSYM(dereg_mr);
#undef MCCL_DLSYM
    if (!ok) {
      dlclose(handle);
      handle = nullptr;
    }
  }
};

IbvApi& ibv() {
  static IbvApi api;  // magic static: dlopen librdma once, thread-safe
  return api;
}

}

struct mcclRdmaConn {
  ibv_context* ctx = nullptr;
  ibv_pd*      pd  = nullptr;
  ibv_cq*      cq  = nullptr;
  ibv_qp*      qp  = nullptr;
  uint32_t     psn = 7;
};

bool mcclRdmaAvailable() { return ibv().handle != nullptr; }

mcclResult mcclRdmaClose(mcclRdmaConn* c) {
  if (c == nullptr) return mcclInvalidArgument;
  IbvApi& v = ibv();
  if (c->qp)  v.destroy_qp(c->qp);
  if (c->cq)  v.destroy_cq(c->cq);
  if (c->pd)  v.dealloc_pd(c->pd);
  if (c->ctx) v.close_device(c->ctx);
  delete c;
  return mcclSuccess;
}

mcclResult mcclRdmaOpen(const char* device, mcclRdmaConn** out) {
  if (out == nullptr || device == nullptr) return mcclInvalidArgument;
  *out = nullptr;
  if (!mcclRdmaAvailable()) return mcclInvalidUsage;
  IbvApi& v = ibv();

  int n = 0;
  ibv_device** list = v.get_device_list(&n);
  ibv_context* ctx = nullptr;
  if (list != nullptr) {
    for (int i = 0; i < n; ++i)
      if (std::strcmp(device, v.get_device_name(list[i])) == 0) { ctx = v.open_device(list[i]); break; }
    v.free_device_list(list);
  }
  if (ctx == nullptr) return mcclSystemError;

  auto* c = new mcclRdmaConn();
  c->ctx = ctx;
  c->pd = v.alloc_pd(ctx);
  c->cq = v.create_cq(ctx, 32, nullptr, nullptr, 0);
  if (c->pd == nullptr || c->cq == nullptr) { mcclRdmaClose(c); return mcclSystemError; }

  ibv_qp_init_attr ia{};
  ia.send_cq = c->cq;
  ia.recv_cq = c->cq;
  ia.cap.max_send_wr = 32;
  ia.cap.max_recv_wr = 32;
  ia.cap.max_send_sge = 1;
  ia.cap.max_recv_sge = 1;
  ia.qp_type = IBV_QPT_UC;
  c->qp = v.create_qp(c->pd, &ia);
  if (c->qp == nullptr) { mcclRdmaClose(c); return mcclSystemError; }

  ibv_qp_attr a{};
  a.qp_state = IBV_QPS_INIT;
  a.port_num = 1;
  a.pkey_index = 0;
  a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
  if (v.modify_qp(c->qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
    mcclRdmaClose(c);
    return mcclSystemError;
  }
  *out = c;
  return mcclSuccess;
}

mcclResult mcclRdmaLocalDest(mcclRdmaConn* c, mcclRdmaDest* out) {
  if (c == nullptr || out == nullptr) return mcclInvalidArgument;
  IbvApi& v = ibv();
  ibv_port_attr pa{};
  if (v.query_port(c->ctx, 1, &pa) != 0) return mcclSystemError;
  ibv_gid gid{};
  for (int i = 0; i < pa.gid_tbl_len; ++i) {
    ibv_gid t;
    if (v.query_gid(c->ctx, 1, i, &t) == 0 && *reinterpret_cast<uint64_t*>(&t.raw[0]) == 0 &&
        *reinterpret_cast<uint16_t*>(&t.raw[8]) == 0 &&
        *reinterpret_cast<uint16_t*>(&t.raw[10]) == 0xffff) {
      gid = t;
      break;
    }
  }
  out->localId = pa.lid;
  out->qpNum = c->qp->qp_num;
  out->psn = c->psn;
  std::memcpy(out->gid, gid.raw, 16);
  return mcclSuccess;
}

mcclResult mcclRdmaConnect(mcclRdmaConn* c, const mcclRdmaDest& peer) {
  if (c == nullptr) return mcclInvalidArgument;
  IbvApi& v = ibv();

  ibv_qp_attr rtr{};
  rtr.qp_state = IBV_QPS_RTR;
  rtr.path_mtu = IBV_MTU_1024;
  rtr.rq_psn = peer.psn;
  rtr.dest_qp_num = peer.qpNum;
  rtr.ah_attr.dlid = static_cast<uint16_t>(peer.localId);
  rtr.ah_attr.port_num = 1;
  ibv_gid pg{};
  std::memcpy(pg.raw, peer.gid, 16);
  if (pg.global.interface_id != 0) {
    rtr.ah_attr.is_global = 1;
    rtr.ah_attr.grh.hop_limit = 1;
    rtr.ah_attr.grh.dgid = pg;
    rtr.ah_attr.grh.sgid_index = 1;
  }
  if (v.modify_qp(c->qp, &rtr,
                  IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN) != 0)
    return mcclSystemError;

  ibv_qp_attr rts{};
  rts.qp_state = IBV_QPS_RTS;
  rts.sq_psn = c->psn;
  if (v.modify_qp(c->qp, &rts, IBV_QP_STATE | IBV_QP_SQ_PSN) != 0) return mcclSystemError;
  return mcclSuccess;
}

namespace {

// Register, post one WR, poll to completion, deregister. TODO: a registered-buffer pool to avoid per-call
// reg_mr, plus a UC recv-before-send protocol, once RDMA can be exercised on TB5 hardware.
mcclResult one_shot(mcclRdmaConn* c, void* buf, size_t bytes, bool isSend) {
  IbvApi& v = ibv();
  ibv_mr* mr = v.reg_mr(c->pd, buf, bytes, IBV_ACCESS_LOCAL_WRITE);
  if (mr == nullptr) return mcclSystemError;

  ibv_sge sge{};
  sge.addr = reinterpret_cast<uintptr_t>(buf);
  sge.length = static_cast<uint32_t>(bytes);
  sge.lkey = mr->lkey;

  int posted;
  if (isSend) {
    ibv_send_wr wr{}, *bad = nullptr;
    wr.wr_id = 1;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_SEND;
    wr.send_flags = IBV_SEND_SIGNALED;
    posted = ibv_post_send(c->qp, &wr, &bad);
  } else {
    ibv_recv_wr wr{}, *bad = nullptr;
    wr.wr_id = 1;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    posted = ibv_post_recv(c->qp, &wr, &bad);
  }

  mcclResult rc = mcclSuccess;
  if (posted != 0) {
    rc = mcclSystemError;
  } else {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(kPollTimeoutMs);
    ibv_wc wc{};
    int got = 0;
    while ((got = ibv_poll_cq(c->cq, 1, &wc)) == 0) {
      if (std::chrono::steady_clock::now() >= deadline) break;
    }
    if (got <= 0 || wc.status != IBV_WC_SUCCESS) rc = mcclSystemError;
  }
  v.dereg_mr(mr);
  return rc;
}

}

mcclResult mcclRdmaSend(mcclRdmaConn* c, const void* buf, size_t bytes) {
  if (c == nullptr || buf == nullptr) return mcclInvalidArgument;
  return one_shot(c, const_cast<void*>(buf), bytes, true);
}

mcclResult mcclRdmaRecv(mcclRdmaConn* c, void* buf, size_t bytes) {
  if (c == nullptr || buf == nullptr) return mcclInvalidArgument;
  return one_shot(c, buf, bytes, false);
}

}

#else  // no <infiniband/verbs.h> (macOS < 26.2 / no TB5): compile a stub so the lib always builds; RDMA is just unavailable

namespace mccl {

bool       mcclRdmaAvailable() { return false; }
mcclResult mcclRdmaOpen(const char*, mcclRdmaConn** out) { if (out) *out = nullptr; return mcclInvalidUsage; }
mcclResult mcclRdmaLocalDest(mcclRdmaConn*, mcclRdmaDest*) { return mcclInvalidUsage; }
mcclResult mcclRdmaConnect(mcclRdmaConn*, const mcclRdmaDest&) { return mcclInvalidUsage; }
mcclResult mcclRdmaSend(mcclRdmaConn*, const void*, size_t) { return mcclInvalidUsage; }
mcclResult mcclRdmaRecv(mcclRdmaConn*, void*, size_t) { return mcclInvalidUsage; }
mcclResult mcclRdmaClose(mcclRdmaConn*) { return mcclSuccess; }

}

#endif
