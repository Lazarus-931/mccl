#pragma once

#include <cstddef>
#include <cstdint>

#include "definitions.h"
#include "include/param.h"

namespace mccl {

MCCL_PARAM(SocketBufKB);  // shared with m2m: per-connection buffer budget that striping divides per socket

// bufBytes sizes SO_SNDBUF/RCVBUF: <0 = the SOCKET_BUF_KB default, 0 = leave the OS/inherited value, >0 = exact.
// It must be in place BEFORE the TCP handshake advertises a window (connect: on the socket pre-connect;
// accept: set on the LISTENER and inherited) — shrinking later strands an already-advertised larger window and
// the connection collapses into a retransmit storm.
mcclResult mcclSocketListen(uint16_t port, int backlog, int* listenFd, uint16_t* boundPort, int bufBytes = -1);
mcclResult mcclSocketAccept(int listenFd, int timeoutMs, int* connFd, int bufBytes = -1);
mcclResult mcclSocketConnect(const char* ip, uint16_t port, int timeoutMs, int retries, int* connFd, int bufBytes = -1);
mcclResult mcclSocketSend(int fd, const void* buf, size_t bytes);
mcclResult mcclSocketRecv(int fd, void* buf, size_t bytes);
void       mcclSocketClose(int fd);

}
