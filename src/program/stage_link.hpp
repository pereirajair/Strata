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
//   'R' u32 n, i32 ids[n]            -> u8 how (1 = its live session, 2 = a checkpoint)   (continue after ids)
//   'S' u32 n, i32 ids[n]            -> (a checkpoint of the session as it is, which holds exactly ids)
//   'P' i64 pos0, u32 n, i64 tok[n], u8 save  -> f32 rows[n * prompt_floats]   (the prompt piece read through the
//                                       stage; save = 1: a checkpoint at its end, as the later stage takes there)
//   'W' i64 pos0, u32 T, i32 tok[T]  -> f32 rows[T * window_floats]   (a verify window read through the stage)
//   'C' u32 n_keep                   -> (the window's first n_keep tokens kept)
//   'Q'                              -> (the stage server ends)
//
// The stage server answers in order, so the later stage may send several 'P' before it reads the first answer: the
// earlier stage reads piece k + 1 while the later one reads piece k (PrevStage::stream_begin, a micro-pipeline).
// THE POM'S LINK FRAMES: every payload above travels in the POM's envelope for an engine's stage link - a kind byte
// first: 1, the bytes as they are; 2, a tensor: u32 n_meta, the first n_meta bytes, then f32 values.  The rows of a
// 'P' or 'W' answer go as a tensor (the status byte is its meta), so the POM node between the machines carries them
// in the deployment's hidden-state precision (f16, f8, int4 ...) and hands them back as f32: the engines never see
// another precision (send_msg, recv_msg).
// 'R' fails when the earlier stage holds no state that ends with exactly those tokens; the later stage then reads
// the prompt from position 0 ('Z').
#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
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

/// A message in the POM's link envelope: `body` as it is, or - with `tensor_from` > 0 - as a tensor whose meta is
/// the first `tensor_from` bytes and whose f32 values are the rest.
inline bool send_msg(int fd, const std::vector<char>& body, size_t tensor_from = 0) {
    char head[9];
    size_t hn = 0;
    const uint32_t n = (uint32_t) (body.size() + (tensor_from > 0 ? 5 : 1));
    std::memcpy(head, &n, 4);
    head[4] = tensor_from > 0 ? 2 : 1;
    hn = 5;
    if (tensor_from > 0) {
        const uint32_t m = (uint32_t) tensor_from;
        std::memcpy(head + 5, &m, 4);
        hn = 9;
    }
    return write_all(fd, head, hn) && write_all(fd, body.data(), body.size());
}

/// The body of a message in the POM's link envelope (a tensor's meta, then its values).
inline bool recv_msg(int fd, std::vector<char>& body, uint32_t max_bytes = 1u << 31) {
    uint32_t n = 0;
    uint8_t kind = 0;
    if (!read_all(fd, &n, 4) || n < 1 || n > max_bytes || !read_all(fd, &kind, 1)) return false;
    n -= 1;
    if (kind == 2) {
        uint32_t m = 0;
        if (n < 4 || !read_all(fd, &m, 4)) return false;
        n -= 4;
        if (m > n || (n - m) % 4 != 0) return false;
    } else if (kind != 1) {
        return false;
    }
    body.resize(n);
    return n == 0 || read_all(fd, body.data(), n);
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
    ~PrevStage() {
        stream_end();
        if (fd_ >= 0) ::close(fd_);
    }

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
    /// The earlier stage continues after `ids` (its live session or a checkpoint of them); `how` is 1 or 2.  False
    /// with `err` when it cannot: the caller reads the prompt from position 0.
    bool resume(const int64_t* ids, int64_t n, int& how, std::string& err) {
        std::vector<char> ans;
        if (!call(ids_request('R', ids, n), ans, err)) return false;
        how = ans.size() > 1 ? (int) (uint8_t) ans[1] : 0;
        return true;
    }
    /// The earlier stage keeps a checkpoint of its session, which holds exactly `ids`.
    bool save(const int64_t* ids, int64_t n, std::string& err) {
        std::vector<char> ans;
        return call(ids_request('S', ids, n), ans, err);
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

    /// A piece of a prompt read through the earlier stage: one 'P' request.
    struct Piece {
        int64_t pos0 = 0, n = 0;
        bool save = false;   ///< the earlier stage takes a checkpoint at its end
    };
    /// THE MICRO-PIPELINE of a prompt read: a thread sends the pieces' requests (`tokens` are the prompt from the
    /// first piece's position) and keeps up to `depth` of them ahead of the ones `stream_take` has taken, so the
    /// earlier stage reads the next pieces while this one computes.  `tokens` must stay valid until `stream_end`.
    bool stream_begin(const int64_t* tokens, std::vector<Piece> pieces, int depth, std::string& err) {
        stream_end();
        if (fd_ < 0) { err = "the earlier stage is not connected"; return false; }
        st_tokens_ = tokens;
        st_pieces_ = std::move(pieces);
        st_depth_ = std::max(1, depth);
        st_sent_ = st_taken_ = 0;
        st_ready_.clear();
        st_err_.clear();
        st_stop_ = false;
        st_failed_ = false;
        st_depth_max_ = 0;
        st_thread_ = std::thread([this] { stream_loop(); });
        return true;
    }
    /// The rows of the next `n` tokens of the stream (whole pieces), in order.
    bool stream_take(int64_t n, float* rows, std::string& err) {
        const size_t pf = prompt_floats_;
        int64_t got = 0;
        while (got < n) {
            std::vector<float> ans;
            int64_t m = 0;
            {
                std::unique_lock<std::mutex> lk(st_mu_);
                st_cv_.wait(lk, [&] { return !st_ready_.empty() || st_failed_; });
                if (st_ready_.empty()) {
                    err = st_err_;
                    return false;
                }
                ans = std::move(st_ready_.front());
                st_ready_.pop_front();
                m = st_pieces_[st_taken_].n;
                ++st_taken_;
            }
            st_cv_.notify_all();
            if (got + m > n) {
                err = "the stage stream's pieces do not end where the chunk ends";
                return false;
            }
            std::memcpy(rows + (size_t) got * pf, ans.data(), (size_t) m * pf * sizeof(float));
            got += m;
        }
        return true;
    }
    /// Ends the stream: no new piece is asked for and the answers already asked for are read (the connection stays
    /// in step).  The greatest number of pieces that were in flight at once.
    int stream_end() {
        if (st_thread_.joinable()) {
            {
                std::lock_guard<std::mutex> lk(st_mu_);
                st_stop_ = true;
            }
            st_cv_.notify_all();
            st_thread_.join();
        }
        st_ready_.clear();
        st_pieces_.clear();
        return st_depth_max_;
    }

private:
    int fd_ = -1;
    uint32_t window_floats_ = 0, prompt_floats_ = 0;
    std::vector<char> ans_;
    // the stream (stream_begin): the pieces, how many were sent and taken, the answers not taken yet
    const int64_t* st_tokens_ = nullptr;
    std::vector<Piece> st_pieces_;
    int st_depth_ = 1, st_depth_max_ = 0;
    size_t st_sent_ = 0, st_taken_ = 0;
    std::deque<std::vector<float>> st_ready_;
    std::string st_err_;
    bool st_stop_ = false, st_failed_ = false;
    std::mutex st_mu_;
    std::condition_variable st_cv_;
    std::thread st_thread_;

    static std::vector<char> ids_request(char op, const int64_t* ids, int64_t n) {
        std::vector<char> req{op};
        put<uint32_t>(req, (uint32_t) n);
        for (int64_t i = 0; i < n; ++i) put<int32_t>(req, (int32_t) ids[i]);
        return req;
    }

    void stream_fail(const std::string& e) {
        {
            std::lock_guard<std::mutex> lk(st_mu_);
            if (!st_failed_) st_err_ = e;
            st_failed_ = true;
        }
        st_cv_.notify_all();
    }

    // The stream's thread: sends while fewer than `depth` pieces are ahead of the taken ones, else reads the oldest
    // answer.  After stream_end it sends nothing more and reads what it has sent.
    void stream_loop() {
        const size_t pf = prompt_floats_;
        size_t received = 0;
        const int64_t base = st_pieces_.empty() ? 0 : st_pieces_.front().pos0;
        for (;;) {
            bool send_one = false;
            {
                std::unique_lock<std::mutex> lk(st_mu_);
                // memory: the answers received and not taken count against the depth as the ones in flight do
                st_cv_.wait(lk, [&] {
                    return st_stop_ || st_failed_ || received < st_sent_ ||
                           (st_sent_ < st_pieces_.size() && st_sent_ - st_taken_ < (size_t) st_depth_);
                });
                if (st_failed_ || (st_stop_ && received == st_sent_)) return;
                send_one = !st_stop_ && st_sent_ < st_pieces_.size() && st_sent_ - st_taken_ < (size_t) st_depth_;
                if (!send_one && received == st_sent_) continue;
            }
            if (send_one) {
                const Piece& p = st_pieces_[st_sent_];
                std::vector<char> req{'P'};
                put<int64_t>(req, p.pos0);
                put<uint32_t>(req, (uint32_t) p.n);
                put_bytes(req, st_tokens_ + (p.pos0 - base), (size_t) p.n * sizeof(int64_t));
                put<uint8_t>(req, p.save ? 1 : 0);
                if (!send_msg(fd_, req)) {
                    lost();
                    return;
                }
                std::lock_guard<std::mutex> lk(st_mu_);
                ++st_sent_;
                st_depth_max_ = std::max(st_depth_max_, (int) (st_sent_ - received));
                // keep sending while the window has room: the answers wait in the socket
                if (st_sent_ < st_pieces_.size() && st_sent_ - st_taken_ < (size_t) st_depth_ && !st_stop_) continue;
            }
            std::vector<char> ans;
            if (!recv_msg(fd_, ans)) {
                lost();
                return;
            }
            const Piece& p = st_pieces_[received++];
            if (ans.empty() || ans[0] != 0) {
                stream_fail("the earlier stage: " +
                            (ans.size() > 1 ? std::string(ans.data() + 1, ans.size() - 1) : std::string("error")));
                drain(received);
                return;
            }
            std::vector<float> rows((size_t) p.n * pf);
            if (ans.size() != 1 + rows.size() * sizeof(float)) {
                stream_fail("the earlier stage answered " + std::to_string(ans.size() - 1) + " bytes of rows, expected " +
                            std::to_string(rows.size() * sizeof(float)));
                drain(received);
                return;
            }
            std::memcpy(rows.data(), ans.data() + 1, rows.size() * sizeof(float));
            {
                std::lock_guard<std::mutex> lk(st_mu_);
                if (!st_stop_) st_ready_.push_back(std::move(rows));
            }
            st_cv_.notify_all();
        }
    }
    // after a failed piece: the answers to the pieces already sent still come, read and dropped
    void drain(size_t received) {
        std::vector<char> ans;
        for (size_t sent = st_sent_; received < sent; ++received)
            if (!recv_msg(fd_, ans)) {
                lost();
                return;
            }
    }
    void lost() {
        stream_fail("the earlier stage's connection was lost");
        ::close(fd_);
        fd_ = -1;
    }

    bool call(const std::vector<char>& req, std::vector<char>& ans, std::string& err) {
        if (fd_ < 0) { err = "the earlier stage is not connected"; return false; }
        if (!send_msg(fd_, req) || !recv_msg(fd_, ans)) {
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
