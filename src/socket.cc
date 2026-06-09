#include "socket.h"

#include "include/param.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <cerrno>
#include <ctime>

namespace mccl {

namespace {

DEFINE_MCCL_PARAM(SocketBufKB, "SOCKET_BUF_KB", 4096);  // SO_SNDBUF/RCVBUF per socket to keep the TB link full; 0 = OS default

void tune(int fd) {
  int one = 1;
  setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  const int kb = static_cast<int>(mcclParamSocketBufKB());
  if (kb > 0) {
    int b = kb * 1024;
    setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &b, sizeof(b));
    setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &b, sizeof(b));
  }
}

bool waitReady(int fd, bool forWrite, int timeoutMs) {
  fd_set set;
  FD_ZERO(&set);
  FD_SET(fd, &set);
  timeval tv{};
  tv.tv_sec = timeoutMs / 1000;
  tv.tv_usec = (timeoutMs % 1000) * 1000;
  return select(fd + 1, forWrite ? nullptr : &set, forWrite ? &set : nullptr, nullptr, &tv) > 0;
}

bool connectOnce(int fd, const sockaddr_in& addr, int timeoutMs) {
  const int flags = fcntl(fd, F_GETFL, 0);
  if (flags < 0) return false;
  fcntl(fd, F_SETFL, flags | O_NONBLOCK);
  bool ok = false;
  if (::connect(fd, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) == 0) {
    ok = true;
  } else if (errno == EINPROGRESS && waitReady(fd, true, timeoutMs)) {
    int err = 0;
    socklen_t len = sizeof(err);
    ok = getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len) == 0 && err == 0;
  }
  fcntl(fd, F_SETFL, flags);
  return ok;
}

}

mcclResult mcclSocketListen(uint16_t port, int backlog, int* listenFd, uint16_t* boundPort) {
  if (listenFd == nullptr) return mcclInvalidArgument;
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return mcclSystemError;
  int one = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(port);
  if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 || listen(fd, backlog) != 0) {
    close(fd);
    return mcclSystemError;
  }
  if (boundPort != nullptr) {
    sockaddr_in a{};
    socklen_t l = sizeof(a);
    if (getsockname(fd, reinterpret_cast<sockaddr*>(&a), &l) == 0) *boundPort = ntohs(a.sin_port);
  }
  *listenFd = fd;
  return mcclSuccess;
}

mcclResult mcclSocketAccept(int listenFd, int timeoutMs, int* connFd) {
  if (connFd == nullptr) return mcclInvalidArgument;
  if (!waitReady(listenFd, false, timeoutMs)) return mcclSystemError;
  int fd = accept(listenFd, nullptr, nullptr);
  if (fd < 0) return mcclSystemError;
  tune(fd);
  *connFd = fd;
  return mcclSuccess;
}

mcclResult mcclSocketConnect(const char* ip, uint16_t port, int timeoutMs, int retries, int* connFd) {
  if (ip == nullptr || connFd == nullptr) return mcclInvalidArgument;
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  if (inet_pton(AF_INET, ip, &addr.sin_addr) != 1) return mcclInvalidArgument;
  for (int attempt = 0; attempt <= retries; ++attempt) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return mcclSystemError;
    if (connectOnce(fd, addr, timeoutMs)) {
      tune(fd);
      *connFd = fd;
      return mcclSuccess;
    }
    close(fd);
    if (attempt < retries) {
      timespec ts{0, 100 * 1000 * 1000};
      nanosleep(&ts, nullptr);
    }
  }
  return mcclSystemError;
}

mcclResult mcclSocketSend(int fd, const void* buf, size_t bytes) {
  const char* p = static_cast<const char*>(buf);
  for (size_t sent = 0; sent < bytes;) {
    ssize_t n = send(fd, p + sent, bytes - sent, 0);
    if (n <= 0) return mcclSystemError;
    sent += static_cast<size_t>(n);
  }
  return mcclSuccess;
}

mcclResult mcclSocketRecv(int fd, void* buf, size_t bytes) {
  char* p = static_cast<char*>(buf);
  for (size_t got = 0; got < bytes;) {
    ssize_t n = recv(fd, p + got, bytes - got, 0);
    if (n <= 0) return mcclSystemError;
    got += static_cast<size_t>(n);
  }
  return mcclSuccess;
}

void mcclSocketClose(int fd) {
  if (fd >= 0) close(fd);
}

}
