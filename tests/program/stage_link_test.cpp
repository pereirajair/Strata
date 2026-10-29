// The stage link without a model or a GPU: the POM's envelope round-trips, and a prompt read as a stream keeps
// several pieces in flight against a stage server that answers in order (a fake one here).
#include "../../src/program/stage_link.hpp"

#include <cmath>
#include <sys/socket.h>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace sl = strata::stage;

namespace {
int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", description); std::exit(1); }
}

// The POM's envelope: a tensor message keeps its meta and values apart on the wire and comes back as it went.
void envelope() {
    int sv[2];
    check(::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0, "a socket pair");
    std::vector<char> body{0};
    const float rows[3] = {1.5f, -2.0f, 3.25f};
    sl::put_bytes(body, rows, sizeof rows);
    check(sl::send_msg(sv[0], body, 1), "a tensor message is sent");
    std::vector<char> raw(4 + 1 + 4 + body.size());
    check(sl::read_all(sv[1], raw.data(), raw.size()), "its bytes arrive");
    uint32_t n = 0, m = 0;
    std::memcpy(&n, raw.data(), 4);
    std::memcpy(&m, raw.data() + 5, 4);
    check(n == raw.size() - 4 && raw[4] == 2 && m == 1, "kind 2, then the meta's length");
    check(sl::send_msg(sv[0], body, 1) && sl::send_msg(sv[0], std::vector<char>{'Z'}), "a tensor and an opaque message");
    std::vector<char> got;
    check(sl::recv_msg(sv[1], got) && got == body, "a tensor message comes back whole");
    check(sl::recv_msg(sv[1], got) && got == std::vector<char>{'Z'}, "an opaque message comes back whole");
    const uint32_t bad_n = 2;
    const char bad[2] = {9, 0};
    check(sl::write_all(sv[0], &bad_n, 4) && sl::write_all(sv[0], bad, 2) && !sl::recv_msg(sv[1], got), "an unknown kind is refused");
    ::close(sv[0]);
    ::close(sv[1]);
}

// A fake earlier stage: 'H', then 'P' pieces answered in order with rows = token * 1000 + column, after a
// short pause each (its compute); it counts the most requests that were waiting in its socket at once.
void stream_against_a_fake_stage() {
    const uint32_t pf = 128;
    std::string err;
    int port = 0;
    const int lfd = sl::tcp_open("127.0.0.1:0", true, err, &port);
    check(lfd >= 0, "the fake stage listens");
    std::thread server([&] {
        const int cfd = ::accept(lfd, nullptr, nullptr);
        std::vector<char> req;
        while (sl::recv_msg(cfd, req)) {
            std::vector<char> ans{0};
            size_t at = 1;
            if (req[0] == 'H') {
                sl::put<uint32_t>(ans, 0);
                sl::put<uint32_t>(ans, 10);
                sl::put<uint32_t>(ans, pf);
                sl::put<uint32_t>(ans, pf);
            } else if (req[0] == 'P') {
                int64_t pos0 = 0;
                uint32_t n = 0;
                sl::get(req, at, pos0);
                sl::get(req, at, n);
                std::vector<float> rows((size_t) n * pf);
                for (uint32_t t = 0; t < n; ++t) {
                    int64_t tok = 0;
                    std::memcpy(&tok, req.data() + at + 8 * t, 8);
                    for (uint32_t c = 0; c < pf; ++c) rows[(size_t) t * pf + c] = (float) (tok % 7) + (float) c / 256.0f;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                sl::put_bytes(ans, rows.data(), rows.size() * sizeof(float));
            }
            if (!sl::send_msg(cfd, ans, req[0] == 'P' ? 1 : 0)) break;
        }
        ::close(cfd);
    });
    {
        sl::PrevStage prev;
        check(prev.connect("127.0.0.1:" + std::to_string(port), "", 10, err), "the later stage connects");
        std::vector<int64_t> toks(40);
        for (size_t i = 0; i < toks.size(); ++i) toks[i] = (int64_t) i * 3;
        std::vector<sl::PrevStage::Piece> pieces;
        for (int64_t a = 0; a < 40; a += 4) pieces.push_back({100 + a, 4, false});
        check(prev.stream_begin(toks.data(), pieces, 4, err), "the stream starts");
        std::vector<float> rows(8 * pf);
        bool exact = true;
        for (int64_t c = 0; c < 40; c += 8) {
            check(prev.stream_take(8, rows.data(), err), "a chunk of two pieces is taken");
            std::this_thread::sleep_for(std::chrono::milliseconds(15));   // this stage's compute
            for (int64_t t = 0; t < 8; ++t)
                for (uint32_t col = 0; col < pf; ++col) {
                    const float want = (float) (toks[c + t] % 7) + (float) col / 256.0f;
                    if (std::fabs(rows[t * pf + col] - want) > 0.0f) exact = false;
                }
        }
        check(exact, "the rows come back in the order of the pieces");
        check(prev.stream_end() >= 2, "several pieces were in flight at once");
        // a stream ended early still reads the answers it asked for: the next one starts in step
        check(prev.stream_begin(toks.data(), pieces, 3, err), "a second stream starts");
        check(prev.stream_take(4, rows.data(), err), "its first piece is taken");
        prev.stream_end();
        check(prev.stream_begin(toks.data(), pieces, 2, err), "a third stream starts after an early end");
        check(prev.stream_take(4, rows.data(), err) && rows[0] == (float) (toks[0] % 7),
              "its first piece is its own, not a leftover");
        prev.stream_end();
    }
    server.join();
    ::close(lfd);
}

// The commit folded into the next window: against a fake stage that says it takes 'V', a commit is held back and
// goes with the next window (one round trip per step), and before any other request as a plain 'C'; against one
// that does not (an older server, no features in its hello), every commit is its own 'C'.
void commit_folding(bool server_folds) {
    const uint32_t wf = 16;
    std::string err;
    int port = 0;
    const int lfd = sl::tcp_open("127.0.0.1:0", true, err, &port);
    check(lfd >= 0, "the fake stage listens");
    std::string ops;
    std::vector<uint32_t> keeps;
    std::thread server([&] {
        const int cfd = ::accept(lfd, nullptr, nullptr);
        std::vector<char> req;
        while (sl::recv_msg(cfd, req)) {
            std::vector<char> ans{0};
            size_t at = 1;
            const char op = req[0];
            ops += op;
            if (op == 'H') {
                sl::put<uint32_t>(ans, 0);
                sl::put<uint32_t>(ans, 10);
                sl::put<uint32_t>(ans, wf);
                sl::put<uint32_t>(ans, wf);
                if (server_folds) sl::put<uint32_t>(ans, sl::PrevStage::kFoldCommit);
            } else if (op == 'W' || op == 'V') {
                int64_t pos0 = 0;
                uint32_t T = 0;
                sl::get(req, at, pos0);
                sl::get(req, at, T);
                if (op == 'V') {
                    size_t kat = at + (size_t) T * 4;
                    uint32_t n = 0;
                    sl::get(req, kat, n);
                    keeps.push_back(n);
                    check(req.size() == kat, "a folded window ends with its n_keep");
                } else {
                    check(req.size() == at + (size_t) T * 4, "a plain window ends with its tokens");
                }
                std::vector<float> rows((size_t) T * wf, (float) pos0);
                sl::put_bytes(ans, rows.data(), rows.size() * sizeof(float));
            } else if (op == 'C') {
                uint32_t n = 0;
                sl::get(req, at, n);
                keeps.push_back(n);
            }
            if (!sl::send_msg(cfd, ans, (op == 'W' || op == 'V') ? 1 : 0)) break;
        }
        ::close(cfd);
    });
    {
        sl::PrevStage prev;
        check(prev.connect("127.0.0.1:" + std::to_string(port), "", 10, err), "the later stage connects");
        check(prev.folds_commit() == server_folds, "the hello says whether the commit folds");
        std::vector<float> rows(4 * wf);
        const int32_t toks[4] = {1, 2, 3, 4};
        check(prev.window(3, toks, 50, rows.data(), err) && rows[0] == 50.0f, "the first window");
        check(prev.commit(2, err), "its commit");
        check(prev.window(4, toks, 52, rows.data(), err) && rows[0] == 52.0f, "the next window");
        check(prev.commit(1, err), "its commit");
        int how = 0;
        prev.resume(nullptr, 0, how, err);   // any other request: the held commit goes first
    }
    server.join();
    ::close(lfd);
    if (server_folds) {
        check(ops == "HWVCR", "the commit rides the next window, and goes alone before another request");
    } else {
        check(ops == "HWCWCR", "an older stage gets a 'C' after every window");
    }
    check(keeps == std::vector<uint32_t>({2, 1}), "the commits keep their counts and order");
}
}  // namespace

int main() {
    envelope();
    stream_against_a_fake_stage();
    commit_folding(true);
    commit_folding(false);
    std::printf("stage_link_test: %d checks passed\n", checks);
    return 0;
}
