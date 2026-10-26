// A turn's delta as a file: written from the sender's full image (through tail sources), read back, applied to the
// receiver's base; the result is the sender's full image, and a base of another identity is refused untouched.
#include "strata/core/conversation_file.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace strata::core;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        ++failures;
        std::printf("FAIL: %s\n", what);
    }
}

std::vector<uint8_t> pattern(size_t n, unsigned seed) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = static_cast<uint8_t>((i * 7 + seed * 13 + 3) % 251);
    return v;
}

ConversationBuffer buffer_of(const std::vector<uint8_t>& v, size_t n) {
    ConversationBuffer b;
    b.resize(n);
    b.visit(0, n, [&](uint8_t* p, size_t m, size_t at) {
        std::memcpy(p, v.data() + at, m);
        return true;
    });
    return b;
}

// Two K/V layers, 12 cells (3 pages of 4), 2 heads of 8 bytes; 4 index rows of 5 floats.
struct Layers {
    std::vector<ConversationKv> images;
    std::vector<std::vector<uint8_t>> raw;  // raw[layer * 5 + part]
};

Layers make_layers(int64_t cells, int64_t rows) {
    Layers out;
    for (int layer = 0; layer < 2; ++layer) {
        ConversationKv k;
        k.format = 1;
        k.cells = cells;
        k.heads = 2;
        k.head_dim = 8;
        k.page_size = 4;
        k.pooled_rows = rows;
        k.idx_dim = 5;
        const unsigned s = static_cast<unsigned>(layer * 10);
        const size_t sizes[5] = {size_t(cells) * 2 * 8, size_t(cells) * 2 * 8, size_t(cells) * 2 * 2,
                                 size_t(cells) * 2 * 2, size_t(rows) * 5 * 4};
        std::vector<uint8_t> parts[5];
        for (int part = 0; part < 5; ++part) parts[part] = pattern(sizes[part], s + part + 1);
        k.k = buffer_of(parts[0], sizes[0]);
        k.v = buffer_of(parts[1], sizes[1]);
        k.k_scale = buffer_of(parts[2], sizes[2]);
        k.v_scale = buffer_of(parts[3], sizes[3]);
        k.pooled = buffer_of(parts[4], sizes[4]);
        out.images.push_back(k);
        for (int part = 0; part < 5; ++part) out.raw.push_back(parts[part]);
    }
    return out;
}

// The image's bytes as a SessionKvSource: what the engine's device reader offers.
SessionKvSource source_of(const ConversationKv& k, const std::vector<std::vector<uint8_t>>& raw, size_t layer) {
    SessionKvSource s;
    s.format = k.format;
    s.cells = k.cells;
    s.heads = k.heads;
    s.head_dim = k.head_dim;
    s.page_size = k.page_size;
    s.pooled_rows = k.pooled_rows;
    s.idx_dim = k.idx_dim;
    s.sizes = {raw[layer * 5].size(), raw[layer * 5 + 1].size(), raw[layer * 5 + 2].size(),
               raw[layer * 5 + 3].size(), raw[layer * 5 + 4].size()};
    s.read = [&raw, layer](size_t part, size_t offset, void* dst, size_t n) {
        const auto& v = raw[layer * 5 + part];
        if (offset > v.size() || n > v.size() - offset) return false;
        std::memcpy(dst, v.data() + offset, n);
        return true;
    };
    s.error = [] { return std::string(); };
    return s;
}

bool same_bytes(const ConversationBuffer& b, const std::vector<uint8_t>& v) {
    if (b.size() != v.size()) return false;
    std::vector<uint8_t> x(b.size());
    return b.read(x.data(), 0, b.size()) && x == v;
}

SavedConversation meta_with(const std::vector<int32_t>& ids) {
    SavedConversation m;
    m.geometry[0] = 1;
    m.layer_lo = 0;
    m.layer_hi = 48;
    m.live.ids = ids;
    return m;
}

void file_round_trip_onto_a_partial_base() {
    const Layers full = make_layers(12, 4);
    const std::vector<int32_t> old_ids = {1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<int32_t> new_ids = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const SessionFileIdentity id{7, 9};
    const std::string path = "delta_round_trip.strsdlt";

    // the sender: its base extent is 8 cells and 2 index rows; the delta is everything after
    std::vector<SessionKvSource> tails;
    std::vector<int64_t> base_cells{8, 8}, base_pooled{2, 2};
    for (size_t layer = 0; layer < 2; ++layer)
        tails.push_back(session_delta_tail_source(source_of(full.images[layer], full.raw, layer), 8, 2));
    const uint64_t fp_sender = session_base_fingerprint(old_ids, {8, 8}, {2, 2});
    size_t written = 0;
    std::string error;
    check(session_delta_write(path, meta_with(new_ids), tails, base_cells, base_pooled, fp_sender, id, written, error),
          "write the delta file");

    // the receiver: holds the base image (8 cells, 2 rows, the old ids)
    SessionDelta delta;
    size_t read = 0;
    check(session_delta_read(path, id, delta, read, error), "read the delta file back");
    check(read == written, "the file size is what the writer reported");
    check(delta.base_fingerprint == fp_sender, "the base fingerprint comes back from the file");
    SavedConversation base = meta_with(old_ids);
    base.kv.resize(2);
    for (size_t layer = 0; layer < 2; ++layer) {
        base.kv[layer] = full.images[layer];
        base.kv[layer].cells = 8;
        base.kv[layer].pooled_rows = 2;
        base.kv[layer].k = buffer_of(std::vector<uint8_t>(full.raw[layer * 5].begin(), full.raw[layer * 5].begin() + 8 * 2 * 8), 8 * 2 * 8);
        base.kv[layer].v = buffer_of(std::vector<uint8_t>(full.raw[layer * 5 + 1].begin(), full.raw[layer * 5 + 1].begin() + 8 * 2 * 8), 8 * 2 * 8);
        base.kv[layer].k_scale = buffer_of(std::vector<uint8_t>(full.raw[layer * 5 + 2].begin(), full.raw[layer * 5 + 2].begin() + 8 * 2 * 2), 8 * 2 * 2);
        base.kv[layer].v_scale = buffer_of(std::vector<uint8_t>(full.raw[layer * 5 + 3].begin(), full.raw[layer * 5 + 3].begin() + 8 * 2 * 2), 8 * 2 * 2);
        base.kv[layer].pooled = buffer_of(std::vector<uint8_t>(full.raw[layer * 5 + 4].begin(), full.raw[layer * 5 + 4].begin() + 2 * 5 * 4), 2 * 5 * 4);
    }
    check(session_delta_apply(base, delta, error), "apply the delta to its base");
    check(base.live.ids == new_ids, "the live conversation is the sender's");
    for (size_t layer = 0; layer < 2; ++layer) {
        check(base.kv[layer].cells == 12 && base.kv[layer].pooled_rows == 4, "the layer extents are the full ones");
        check(same_bytes(base.kv[layer].k, full.raw[layer * 5]), "K is the sender's, byte for byte");
        check(same_bytes(base.kv[layer].v, full.raw[layer * 5 + 1]), "V is the sender's, byte for byte");
        check(same_bytes(base.kv[layer].k_scale, full.raw[layer * 5 + 2]), "K scales are the sender's");
        check(same_bytes(base.kv[layer].v_scale, full.raw[layer * 5 + 3]), "V scales are the sender's");
        check(same_bytes(base.kv[layer].pooled, full.raw[layer * 5 + 4]), "index rows are the sender's");
    }
    std::remove(path.c_str());
}

void a_base_of_another_conversation_is_refused_untouched() {
    const Layers full = make_layers(12, 4);
    const SessionFileIdentity id{7, 9};
    const std::string path = "delta_refused.strsdlt";
    std::vector<SessionKvSource> tails;
    for (size_t layer = 0; layer < 2; ++layer)
        tails.push_back(session_delta_tail_source(source_of(full.images[layer], full.raw, layer), 8, 2));
    size_t written = 0;
    std::string error;
    check(session_delta_write(path, meta_with({1, 2, 3}), tails, {8, 8}, {2, 2},
                              session_base_fingerprint({1, 2, 3, 4, 5, 6, 7, 8}, {8, 8}, {2, 2}), id, written, error),
          "write a delta for the refusal case");
    SessionDelta delta;
    size_t read = 0;
    check(session_delta_read(path, id, delta, read, error), "read it back for the refusal case");
    SavedConversation other = meta_with({9, 9, 9, 9, 9, 9, 9, 9});
    other.kv.resize(2);
    for (size_t layer = 0; layer < 2; ++layer) {
        other.kv[layer].format = 1;
        other.kv[layer].cells = 8;
        other.kv[layer].heads = 2;
        other.kv[layer].head_dim = 8;
        other.kv[layer].page_size = 4;
        other.kv[layer].pooled_rows = 2;
        other.kv[layer].idx_dim = 5;
        other.kv[layer].k = buffer_of(std::vector<uint8_t>(8 * 2 * 8, 0), 8 * 2 * 8);
        other.kv[layer].v = buffer_of(std::vector<uint8_t>(8 * 2 * 8, 0), 8 * 2 * 8);
        other.kv[layer].k_scale = buffer_of(std::vector<uint8_t>(8 * 2 * 2, 0), 8 * 2 * 2);
        other.kv[layer].v_scale = buffer_of(std::vector<uint8_t>(8 * 2 * 2, 0), 8 * 2 * 2);
        other.kv[layer].pooled = buffer_of(std::vector<uint8_t>(2 * 5 * 4, 0), 2 * 5 * 4);
    }
    check(!session_delta_apply(other, delta, error), "a base of another conversation is refused");
    check(!error.empty(), "the refusal says why");
    check(other.live.ids.size() == 8 && other.live.ids[0] == 9, "the refused base keeps its live conversation");
    check(other.kv[0].cells == 8, "the refused base keeps its extents");
    std::remove(path.c_str());
}

}  // namespace

int main() {
    file_round_trip_onto_a_partial_base();
    a_base_of_another_conversation_is_refused_untouched();
    if (failures == 0) std::printf("conversation_delta_file_test: all checks passed\n");
    return failures == 0 ? 0 : 1;
}
