#pragma once

#include <cstddef>
#include <cstdint>

#include "definitions.h"

namespace mccl {

mcclResult mcclSocketListen(uint16_t port, int backlog, int* listenFd, uint16_t* boundPort);
mcclResult mcclSocketAccept(int listenFd, int timeoutMs, int* connFd);
mcclResult mcclSocketConnect(const char* ip, uint16_t port, int timeoutMs, int retries, int* connFd);
mcclResult mcclSocketSend(int fd, const void* buf, size_t bytes);
mcclResult mcclSocketRecv(int fd, void* buf, size_t bytes);
void       mcclSocketClose(int fd);

}
