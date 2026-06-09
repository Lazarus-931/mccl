#include "socket.h"

#include "include/param.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <cerrno>
#include <ctime>

namespace mccl {

DEFINE_MCCL_PARAM(SocketBufKB, "SOCKET_BUF_KB", 4096);  // SO_SNDBUF/RCVBUF budget per CONNECTION (m2m splits it across stripes); 0 = OS default

namespace {

void setBufs(int fd, int bufBytes) {
  int b = bufBytes;
  if (b < 0) {
    const int kb = static_cast<int>(mcclParamSocketBufKB());
    if (kb <= 0) return;
    b = kb * 1024;
  }
  if (b == 0) return;  // keep the OS default / listener-inherited size
  setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &b, sizeof(b));
  setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &b, sizeof(b));
}

void tune(int fd) {
  int one = 1;
  setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
  setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
}

// poll, not select: fd_set is a fixed 1024 bits, so FD_SET on a high fd (a hub holds peers x stripes
// descriptors) is stack corruption. poll has no fd-number ceiling.
bool waitReady(int fd, bool forWrite, int timeoutMs) {
  pollfd pf{};
  pf.fd = fd;
  pf.events = forWrite ? POLLOUT : POLLIN;
  for (;;) {
    const int rc = poll(&pf, 1, timeoutMs);
    if (rc > 0) return (pf.revents & (forWrite ? POLLOUT : POLLIN)) != 0 || (pf.revents & (POLLERR | POLLHUP)) != 0;
    if (rc == 0) return false;
    if (errno != EINTR) return false;
  }
}

// A signal (EINTR) or transient kernel buffer exhaustion (ENOBUFS/ENOMEM, seen under many concurrent striped
// transfers) is not a dead peer; failing the transfer would desync the channel and poison the comm fatally.
bool retryErrno() {
  if (errno == EINTR) return true;
  if (errno == ENOBUFS || errno == ENOMEM) {
    timespec ts{0, 2 * 1000 * 1000};  // 2ms: let the kernel drain mbufs before re-offering the data
    nanosleep(&ts, nullptr);
    return true;
  }
  return false;
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

mcclResult mcclSocketListen(uint16_t port, int backlog, int* listenFd, uint16_t* boundPort, int bufBytes) {
  if (listenFd == nullptr) return mcclInvalidArgument;
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return mcclSystemError;
  int one = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
  setBufs(fd, bufBytes);  // accepted sockets inherit these at SYN time — the only safe moment to size them
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

mcclResult mcclSocketAccept(int listenFd, int timeoutMs, int* connFd, int bufBytes) {
  if (connFd == nullptr) return mcclInvalidArgument;
  for (;;) {
    if (!waitReady(listenFd, false, timeoutMs)) return mcclSystemError;
    int fd = accept(listenFd, nullptr, nullptr);
    if (fd < 0) {
      if (errno == EINTR || errno == ECONNABORTED) continue;  // signal / peer gave up while queued: not the listener's failure
      return mcclSystemError;
    }
    tune(fd);
    setBufs(fd, bufBytes);  // growing post-accept is safe; pass 0 to keep what the listener set
    *connFd = fd;
    return mcclSuccess;
  }
}

mcclResult mcclSocketConnect(const char* ip, uint16_t port, int timeoutMs, int retries, int* connFd, int bufBytes) {
  if (ip == nullptr || connFd == nullptr) return mcclInvalidArgument;
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  if (inet_pton(AF_INET, ip, &addr.sin_addr) != 1) return mcclInvalidArgument;
  for (int attempt = 0; attempt <= retries; ++attempt) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return mcclSystemError;
    setBufs(fd, bufBytes);  // before connect: the handshake's window scale and first advertisement come from this
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
    if (n < 0 && retryErrno()) continue;
    if (n <= 0) return mcclSystemError;
    sent += static_cast<size_t>(n);
  }
  return mcclSuccess;
}

mcclResult mcclSocketRecv(int fd, void* buf, size_t bytes) {
  char* p = static_cast<char*>(buf);
  for (size_t got = 0; got < bytes;) {
    ssize_t n = recv(fd, p + got, bytes - got, 0);
    if (n < 0 && retryErrno()) continue;
    if (n <= 0) return mcclSystemError;  // n == 0: peer closed — that one is real
    got += static_cast<size_t>(n);
  }
  return mcclSuccess;
}

void mcclSocketClose(int fd) {
  if (fd >= 0) close(fd);
}

}
