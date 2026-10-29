// The turn delta of a K/V image: cut from a full image after a base extent, appended to that base, the full image comes
// back byte for byte; a base of another extent is refused and left as it was.
#include "strata/core/conversation_delta.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using strata::core::ConversationBuffer;
using strata::core::ConversationKv;
using strata::core::KvDelta;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        ++failures;
        std::printf("FAIL: %s\n", what);
    }
}

ConversationBuffer pattern(size_t n, unsigned seed) {
    ConversationBuffer b;
    b.resize(n);
    b.visit(0, n, [&](uint8_t* p, size_t m, size_t at) {
        for (size_t i = 0; i < m; ++i) p[i] = static_cast<uint8_t>(((at + i) * 7 + seed * 13 + 3) % 251);
        return true;
    });
    return b;
}

// The first `n` bytes of `from`.
ConversationBuffer prefix(const ConversationBuffer& from, size_t n) {
    ConversationBuffer b;
    b.resize(n);
    std::vector<uint8_t> tmp(n);
    from.read(tmp.data(), 0, n);
    b.visit(0, n, [&](uint8_t* p, size_t m, size_t at) {
        std::memcpy(p, tmp.data() + at, m);
        return true;
    });
    return b;
}

bool same(const ConversationBuffer& a, const ConversationBuffer& b) {
    if (a.size() != b.size()) return false;
    std::vector<uint8_t> x(a.size()), y(b.size());
    if (!a.read(x.data(), 0, a.size()) || !b.read(y.data(), 0, b.size())) return false;
    return x == y;
}

bool same_image(const ConversationKv& a, const ConversationKv& b) {
    return a.format == b.format && a.cells == b.cells && a.pooled_rows == b.pooled_rows && a.heads == b.heads &&
           a.head_dim == b.head_dim && a.page_size == b.page_size && a.idx_dim == b.idx_dim && same(a.k, b.k) &&
           same(a.v, b.v) && same(a.k_scale, b.k_scale) && same(a.v_scale, b.v_scale) && same(a.pooled, b.pooled);
}

// 12 cells in 3 pages of 4, 2 KV heads of 8 bytes; 4 index rows of 5 floats.
ConversationKv full_image() {
    ConversationKv k;
    k.format = 1;
    k.cells = 12;
    k.heads = 2;
    k.head_dim = 8;
    k.page_size = 4;
    k.pooled_rows = 4;
    k.idx_dim = 5;
    k.k = pattern(12 * 2 * 8, 1);
    k.v = pattern(12 * 2 * 8, 2);
    k.k_scale = pattern(12 * 2 * 2, 3);
    k.v_scale = pattern(12 * 2 * 2, 4);
    k.pooled = pattern(4 * 5 * 4, 5);
    return k;
}

// The image holding the first `cells` cells and `rows` index rows of `full`.
ConversationKv base_of(const ConversationKv& full, int64_t cells, int64_t rows) {
    ConversationKv b = full;
    b.cells = cells;
    b.pooled_rows = rows;
    b.k = prefix(full.k, cells * 2 * 8);
    b.v = prefix(full.v, cells * 2 * 8);
    b.k_scale = prefix(full.k_scale, cells * 2 * 2);
    b.v_scale = prefix(full.v_scale, cells * 2 * 2);
    b.pooled = prefix(full.pooled, rows * 5 * 4);
    return b;
}

void round_trip_from_a_partial_base() {
    const ConversationKv full = full_image();
    ConversationKv base = base_of(full, 8, 2);
    KvDelta delta;
    std::string error;
    check(strata::core::kv_delta_extract(full, 8, 2, delta, error), "extract a tail after a partial base");
    check(delta.tail.k.size() == 4 * 2 * 8 && delta.tail.k_scale.size() == 4 * 2 * 2, "the tail holds cells 8..11");
    check(delta.tail.pooled.size() == 2 * 5 * 4, "the tail holds index rows 2..3");
    check(strata::core::kv_delta_apply(base, delta, error), "apply the tail to its base");
    check(same_image(base, full), "base plus tail is the full image");
}

void round_trip_from_an_empty_base() {
    const ConversationKv full = full_image();
    ConversationKv base = base_of(full, 0, 0);
    KvDelta delta;
    std::string error;
    check(strata::core::kv_delta_extract(full, 0, 0, delta, error), "extract a tail from nothing");
    check(strata::core::kv_delta_apply(base, delta, error), "apply a whole image to an empty base");
    check(same_image(base, full), "an empty base plus the whole image is the image");
}

void no_new_cells_leaves_the_base_alone() {
    const ConversationKv full = full_image();
    ConversationKv base = full;
    KvDelta delta;
    std::string error;
    check(strata::core::kv_delta_extract(full, 12, 4, delta, error), "extract a tail when nothing is new");
    check(delta.tail.k.empty() && delta.tail.pooled.empty(), "an empty tail holds no bytes");
    check(strata::core::kv_delta_apply(base, delta, error), "apply an empty tail");
    check(same_image(base, full), "an empty tail leaves the image as it was");
}

void a_base_of_another_extent_is_refused() {
    const ConversationKv full = full_image();
    KvDelta delta;
    std::string error;
    check(strata::core::kv_delta_extract(full, 8, 2, delta, error), "extract for the refusal cases");
    ConversationKv wrong = base_of(full, 4, 1);
    const ConversationKv before = wrong;
    check(!strata::core::kv_delta_apply(wrong, delta, error), "a base at another extent is refused");
    check(!error.empty(), "the refusal says why");
    check(same_image(wrong, before), "a refused delta leaves the base as it was");
    ConversationKv other_geometry = base_of(full, 8, 2);
    other_geometry.heads = 4;
    check(!strata::core::kv_delta_apply(other_geometry, delta, error), "a base of another geometry is refused");
}

void an_extent_outside_the_image_is_refused() {
    const ConversationKv full = full_image();
    KvDelta delta;
    std::string error;
    check(!strata::core::kv_delta_extract(full, 13, 0, delta, error), "a base past the last cell is refused");
    check(!strata::core::kv_delta_extract(full, 0, 5, delta, error), "a base past the last index row is refused");
    check(!strata::core::kv_delta_extract(full, -1, 0, delta, error), "a negative base is refused");
}

}  // namespace

// The buffers span several segments (16 MiB each): the tail starts and ends inside them.
void round_trip_across_segments() {
    const int64_t cells = 2500000;  // 2.5 M cells of 16 bytes of K: 40 MB, three segments
    ConversationKv full;
    full.format = 1;
    full.cells = cells;
    full.heads = 2;
    full.head_dim = 8;
    full.page_size = 4;
    full.pooled_rows = 4;
    full.idx_dim = 5;
    full.k = pattern(static_cast<size_t>(cells) * 2 * 8, 1);
    full.v = pattern(static_cast<size_t>(cells) * 2 * 8, 2);
    full.k_scale = pattern(static_cast<size_t>(cells) * 2 * 2, 3);
    full.v_scale = pattern(static_cast<size_t>(cells) * 2 * 2, 4);
    full.pooled = pattern(4 * 5 * 4, 5);
    const int64_t base_cells = 1500000;  // the base ends inside the second segment
    ConversationKv base = full;
    base.cells = base_cells;
    base.pooled_rows = 2;
    base.k = prefix(full.k, static_cast<size_t>(base_cells) * 2 * 8);
    base.v = prefix(full.v, static_cast<size_t>(base_cells) * 2 * 8);
    base.k_scale = prefix(full.k_scale, static_cast<size_t>(base_cells) * 2 * 2);
    base.v_scale = prefix(full.v_scale, static_cast<size_t>(base_cells) * 2 * 2);
    base.pooled = prefix(full.pooled, 2 * 5 * 4);
    KvDelta delta;
    std::string error;
    check(strata::core::kv_delta_extract(full, base_cells, 2, delta, error), "extract a tail across segments");
    check(strata::core::kv_delta_apply(base, delta, error), "apply a tail across segments");
    check(same_image(base, full), "a multi-segment base plus tail is the full image");
}

int main() {
    round_trip_across_segments();
    round_trip_from_a_partial_base();
    round_trip_from_an_empty_base();
    no_new_cells_leaves_the_base_alone();
    a_base_of_another_extent_is_refused();
    an_extent_outside_the_image_is_refused();
    if (failures == 0) std::printf("conversation_delta_test: all checks passed\n");
    return failures == 0 ? 0 : 1;
}
