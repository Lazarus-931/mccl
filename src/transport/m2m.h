#pragma once

#include <cstddef>
#include <cstdint>

#include "../definitions.h"

namespace mccl {

enum m2mType { M2M_TYPE_TCP = 0, M2M_TYPE_RDMA = 1 };

m2mType     m2mTypeSelect();
const char* m2mTypeStr(m2mType t);
bool        mcclM2MAvailable(m2mType t);

struct mcclM2M;

mcclResult mcclM2MConnect(const char* peerIp, uint16_t port, m2mType t, mcclM2M** out);
mcclResult mcclM2MListen(uint16_t port, m2mType t, mcclM2M** out);
mcclResult mcclM2MSend(mcclM2M* c, const void* buf, size_t bytes);
mcclResult mcclM2MRecv(mcclM2M* c, void* buf, size_t bytes);
mcclResult mcclM2MShutdown(mcclM2M* c);
mcclResult mcclM2MClose(mcclM2M* c);

struct mcclM2MListener;
mcclResult mcclM2MListenStart(uint16_t port, mcclM2MListener** out, uint16_t* boundPort);
mcclResult mcclM2MAccept(mcclM2MListener* l, mcclM2M** out);
mcclResult mcclM2MListenClose(mcclM2MListener* l);

}
