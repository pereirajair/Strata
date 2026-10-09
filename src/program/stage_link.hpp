// src/program/stage_link.hpp - a layer split across processes: the link between two stages.
//
// A model can be split by layer ranges across machines: each machine holds one slice (a POM slice: the layers
// [lo, hi) of the model, read under their numbers in the whole model) and runs it in its own engine.  The LAST
// stage serves the requests (--serve: the head, the sampling and the MTP drafter are there, as on the last GPU of a
// --layer-split); the stage before it runs as a STAGE SERVER (--stage-serve HOST:PORT) and, for every prompt chunk
// and every verify window, hands back its residual rows - the hand-off --layer-split passes between GPUs, here over
// a socket.  The later stage connects with --stage-prev HOST:PORT; both sides take STRATA_STAGE_TOKEN.
//
// Frames are little-endian: u32 length, then the payload.  A request's payload starts with an op byte; an answer's
// with a status byte (0 = ok, then the data; 1 = error, then the message).
//
//   'H' token                        -> u32 lo, u32 hi, u32 window_floats, u32 prompt_floats
//   'Z'                              -> (the stage's session zeroed: a new request starts at position 0)
//   'P' i64 pos0, u32 n, i64 tok[n]  -> f32 rows[n * prompt_floats]   (the prompt chunk read through the stage)
//   'W' i64 pos0, u32 T, i32 tok[T]  -> f32 rows[T * window_floats]   (a verify window read through the stage)
//   'C' u32 n_keep                   -> (the window's first n_keep tokens kept)
//   'Q'                              -> (the stage server ends)
#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace strata::stage {

#if !defined(_WIN32)
inline bool write_all(int fd, const void* data, size_t n) {
    const char* p = static_cast<const char*>(data);
    while (n > 0) {
        const ssize_t w = ::send(fd, p, n, MSG_NOSIGNAL);
        if (w <= 0) return false;
        p += w;
        n -= (size_t) w;
    }
    return true;
}

inline bool read_all(int fd, void* data, size_t n) {
    char* p = static_cast<char*>(data);
    while (n > 0) {
        const ssize_t r = ::recv(fd, p, n, 0);
        if (r <= 0) return false;
        p += r;
        n -= (size_t) r;
    }
    return true;
}

/// One frame: the u32 length, then the bytes.
inline bool send_frame(int fd, const std::vector<char>& payload) {
    const uint32_t n = (uint32_t) payload.size();
    return write_all(fd, &n, 4) && write_all(fd, payload.data(), payload.size());
}

inline bool recv_frame(int fd, std::vector<char>& payload, uint32_t max_bytes = 1u << 31) {
    uint32_t n = 0;
    if (!read_all(fd, &n, 4) || n > max_bytes) return false;
    payload.resize(n);
    return n == 0 || read_all(fd, payload.data(), n);
}

template <class T> void put(std::vector<char>& b, T v) {
    const size_t at = b.size();
    b.resize(at + sizeof(T));
    std::memcpy(b.data() + at, &v, sizeof(T));
}

inline void put_bytes(std::vector<char>& b, const void* p, size_t n) {
    const size_t at = b.size();
    b.resize(at + n);
    if (n) std::memcpy(b.data() + at, p, n);
}

/// Reads a T at `at` of `b`, advancing `at`; false past the end.
template <class T> bool get(const std::vector<char>& b, size_t& at, T& v) {
    if (at + sizeof(T) > b.size()) return false;
    std::memcpy(&v, b.data() + at, sizeof(T));
    at += sizeof(T);
    return true;
}

/// "host:port" -> a connected (or, with `listen_fd`, a listening) TCP socket; -1 with `err`.
inline int tcp_open(const std::string& addr, bool listen, std::string& err, int* bound_port = nullptr) {
    const size_t colon = addr.rfind(':');
    if (colon == std::string::npos) { err = "expected HOST:PORT, not " + addr; return -1; }
    std::string host = addr.substr(0, colon), port = addr.substr(colon + 1);
    if (!host.empty() && host.front() == '[' && host.back() == ']') host = host.substr(1, host.size() - 2);
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    if (listen) hints.ai_flags = AI_PASSIVE;
    addrinfo* res = nullptr;
    if (const int rc = ::getaddrinfo(host.empty() ? nullptr : host.c_str(), port.c_str(), &hints, &res); rc != 0) {
        err = "cannot resolve " + addr + ": " + gai_strerror(rc);
        return -1;
    }
    int fd = -1;
    for (addrinfo* a = res; a != nullptr && fd < 0; a = a->ai_next) {
        fd = ::socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (fd < 0) continue;
        const int one = 1;
        ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        if (listen) {
            ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
            if (::bind(fd, a->ai_addr, a->ai_addrlen) != 0 || ::listen(fd, 4) != 0) { ::close(fd); fd = -1; continue; }
            if (bound_port != nullptr) {
                sockaddr_storage ss{};
                socklen_t sl = sizeof ss;
                ::getsockname(fd, (sockaddr*) &ss, &sl);
                *bound_port = ss.ss_family == AF_INET6 ? ntohs(((sockaddr_in6*) &ss)->sin6_port)
                                                        : ntohs(((sockaddr_in*) &ss)->sin_port);
            }
        } else if (::connect(fd, a->ai_addr, a->ai_addrlen) != 0) {
            ::close(fd);
            fd = -1;
        }
    }
    ::freeaddrinfo(res);
    if (fd < 0) err = std::string(listen ? "cannot listen on " : "cannot connect to ") + addr + ": " + std::strerror(errno);
    return fd;
}

/// The later stage's side: asks the earlier stage, over one connection, for its rows.
class PrevStage {
public:
    ~PrevStage() { if (fd_ >= 0) ::close(fd_); }

    /// Connects and checks that the earlier stage ends where this one starts.
    bool connect(const std::string& addr, const std::string& token, int64_t my_lo, std::string& err) {
        if (fd_ >= 0) ::close(fd_);
        fd_ = tcp_open(addr, false, err);
        if (fd_ < 0) return false;
        std::vector<char> req{'H'};
        put_bytes(req, token.data(), token.size());
        std::vector<char> ans;
        if (!call(req, ans, err)) return false;
        size_t at = 1;
        uint32_t lo = 0, hi = 0, wf = 0, pf = 0;
        if (!get(ans, at, lo) || !get(ans, at, hi) || !get(ans, at, wf) || !get(ans, at, pf)) {
            err = "the earlier stage's hello is malformed";
            return false;
        }
        if ((int64_t) hi != my_lo) {
            err = "the earlier stage runs layers " + std::to_string(lo) + "-" + std::to_string((int64_t) hi - 1) +
                  ", but this stage starts at layer " + std::to_string(my_lo);
            return false;
        }
        window_floats_ = wf;
        prompt_floats_ = pf;
        return true;
    }
    uint32_t window_floats() const { return window_floats_; }
    uint32_t prompt_floats() const { return prompt_floats_; }

    bool reset(std::string& err) {
        std::vector<char> ans;
        return call(std::vector<char>{'Z'}, ans, err);
    }
    bool prefill(const int64_t* tokens, int64_t n, int64_t pos0, float* rows, std::string& err) {
        std::vector<char> req{'P'};
        put<int64_t>(req, pos0);
        put<uint32_t>(req, (uint32_t) n);
        put_bytes(req, tokens, (size_t) n * sizeof(int64_t));
        return rows_call(req, rows, (size_t) n * prompt_floats_, err);
    }
    bool window(int T, const int32_t* tokens, int64_t pos0, float* rows, std::string& err) {
        std::vector<char> req{'W'};
        put<int64_t>(req, pos0);
        put<uint32_t>(req, (uint32_t) T);
        put_bytes(req, tokens, (size_t) T * sizeof(int32_t));
        return rows_call(req, rows, (size_t) T * window_floats_, err);
    }
    bool commit(int n_keep, std::string& err) {
        std::vector<char> req{'C'};
        put<uint32_t>(req, (uint32_t) n_keep);
        std::vector<char> ans;
        return call(req, ans, err);
    }

private:
    int fd_ = -1;
    uint32_t window_floats_ = 0, prompt_floats_ = 0;
    std::vector<char> ans_;

    bool call(const std::vector<char>& req, std::vector<char>& ans, std::string& err) {
        if (fd_ < 0) { err = "the earlier stage is not connected"; return false; }
        if (!send_frame(fd_, req) || !recv_frame(fd_, ans)) {
            err = "the earlier stage's connection was lost";
            ::close(fd_);
            fd_ = -1;
            return false;
        }
        if (ans.empty() || ans[0] != 0) {
            err = "the earlier stage: " + (ans.size() > 1 ? std::string(ans.data() + 1, ans.size() - 1) : std::string("error"));
            return false;
        }
        return true;
    }
    bool rows_call(const std::vector<char>& req, float* rows, size_t floats, std::string& err) {
        if (!call(req, ans_, err)) return false;
        if (ans_.size() != 1 + floats * sizeof(float)) {
            err = "the earlier stage answered " + std::to_string(ans_.size() - 1) + " bytes of rows, expected " +
                  std::to_string(floats * sizeof(float));
            return false;
        }
        std::memcpy(rows, ans_.data() + 1, floats * sizeof(float));
        return true;
    }
};
#endif

}  // namespace strata::stage
