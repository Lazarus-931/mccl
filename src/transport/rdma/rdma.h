#pragma once

#include <cstddef>
#include <cstdint>

#include "../../definitions.h"

namespace mccl {

// The address a peer needs to reach this queue pair; swapped over a TCP side channel during bring-up.
struct mcclRdmaDest {
  uint32_t localId;   // LID
  uint32_t qpNum;
  uint32_t psn;
  uint8_t  gid[16];   // IPv4-mapped GID (zero = LID-only routing)
};

struct mcclRdmaConn;

bool       mcclRdmaAvailable();                                       // librdma present (macOS 26.2 + TB5); else everything below is a stub
mcclResult mcclRdmaOpen(const char* device, mcclRdmaConn** out);      // open the device, bring a UC queue pair to INIT
mcclResult mcclRdmaLocalDest(mcclRdmaConn* c, mcclRdmaDest* out);
mcclResult mcclRdmaConnect(mcclRdmaConn* c, const mcclRdmaDest& peer);// QP INIT -> RTR -> RTS once the peer's dest is known
mcclResult mcclRdmaSend(mcclRdmaConn* c, const void* buf, size_t bytes);
mcclResult mcclRdmaRecv(mcclRdmaConn* c, void* buf, size_t bytes);
mcclResult mcclRdmaClose(mcclRdmaConn* c);

}
