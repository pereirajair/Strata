#include "strata/core/conversation_delta.hpp"

#include <algorithm>
#include <cstring>
#include <utility>
#include <vector>

namespace strata::core {
namespace {

constexpr size_t kChunk = size_t{1} << 20;

// Bytes per unit (a cell, or an index row) of a buffer that holds `units` of them. A buffer of no units holds nothing.
bool unit_bytes(const ConversationBuffer& buffer, int64_t units, size_t& per, std::string& error) {
    if (units <= 0) {
        per = 0;
        if (!buffer.empty()) {
            error = "conversation delta: a buffer holds bytes for no cell";
            return false;
        }
        return true;
    }
    if (buffer.size() % static_cast<size_t>(units) != 0) {
        error = "conversation delta: a buffer is not a whole number of cells";
        return false;
    }
    per = buffer.size() / static_cast<size_t>(units);
    return true;
}

// Writes `n` bytes at `offset` of `to`, which must already be that size. The visitor's `at` is the absolute offset
// within `to`, so the source position is measured from the start of this write.
bool write_at(ConversationBuffer& to, size_t offset, const uint8_t* bytes, size_t n) {
    return to.visit(offset, n, [&](uint8_t* p, size_t m, size_t at) {
        std::memcpy(p, bytes + (at - offset), m);
        return true;
    });
}

// Copies bytes [offset, offset + count) of `from` into `to`, which becomes exactly `count` bytes long.
bool copy_range(const ConversationBuffer& from, size_t offset, size_t count, ConversationBuffer& to) {
    to = ConversationBuffer();
    to.resize(count);
    std::vector<uint8_t> chunk(std::min(count, kChunk));
    for (size_t done = 0; done < count;) {
        const size_t n = std::min(chunk.size(), count - done);
        if (!from.read(chunk.data(), offset + done, n)) return false;
        if (!write_at(to, done, chunk.data(), n)) return false;
        done += n;
    }
    return true;
}

// Appends `count` bytes of `from` (starting at its beginning) to the end of `to`.
bool append(ConversationBuffer& to, const ConversationBuffer& from) {
    const size_t start = to.size();
    const size_t count = from.size();
    to.resize(start + count);
    std::vector<uint8_t> chunk(std::min(count, kChunk));
    for (size_t done = 0; done < count;) {
        const size_t n = std::min(chunk.size(), count - done);
        if (!from.read(chunk.data(), done, n)) return false;
        if (!write_at(to, start + done, chunk.data(), n)) return false;
        done += n;
    }
    return true;
}

// The five buffers of an image, with what each one counts: cells, or index rows for the pooled buffer.
struct Part {
    const ConversationBuffer* full;
    ConversationBuffer* tail;
    bool pooled;
};

}  // namespace

bool kv_delta_extract(const ConversationKv& full, int64_t base_cells, int64_t base_pooled, KvDelta& out,
                      std::string& error) {
    if (base_cells < 0 || base_cells > full.cells || base_pooled < 0 || base_pooled > full.pooled_rows) {
        error = "conversation delta: the base extent is outside the image";
        return false;
    }
    KvDelta delta;
    delta.base_cells = base_cells;
    delta.base_pooled = base_pooled;
    delta.tail.format = full.format;
    delta.tail.cells = full.cells;
    delta.tail.heads = full.heads;
    delta.tail.head_dim = full.head_dim;
    delta.tail.page_size = full.page_size;
    delta.tail.pooled_rows = full.pooled_rows;
    delta.tail.idx_dim = full.idx_dim;
    const Part parts[] = {
        {&full.k, &delta.tail.k, false},
        {&full.v, &delta.tail.v, false},
        {&full.k_scale, &delta.tail.k_scale, false},
        {&full.v_scale, &delta.tail.v_scale, false},
        {&full.pooled, &delta.tail.pooled, true},
    };
    for (const Part& part : parts) {
        const int64_t units = part.pooled ? full.pooled_rows : full.cells;
        const int64_t base = part.pooled ? base_pooled : base_cells;
        size_t per = 0;
        if (!unit_bytes(*part.full, units, per, error)) return false;
        if (!copy_range(*part.full, per * static_cast<size_t>(base), per * static_cast<size_t>(units - base),
                        *part.tail)) {
            error = "conversation delta: could not read the tail of the image";
            return false;
        }
    }
    out = std::move(delta);
    return true;
}

bool kv_delta_apply(ConversationKv& base, const KvDelta& delta, std::string& error) {
    const ConversationKv& tail = delta.tail;
    if (base.cells != delta.base_cells || base.pooled_rows != delta.base_pooled) {
        error = "conversation delta: the base image is not the extent the delta was cut from";
        return false;
    }
    if (base.format != tail.format || base.heads != tail.heads || base.head_dim != tail.head_dim ||
        base.page_size != tail.page_size || base.idx_dim != tail.idx_dim) {
        error = "conversation delta: the base image has another geometry";
        return false;
    }
    if (tail.cells < base.cells || tail.pooled_rows < base.pooled_rows) {
        error = "conversation delta: the delta does not extend the base";
        return false;
    }
    const int64_t tail_cells = tail.cells - base.cells;
    const int64_t tail_rows = tail.pooled_rows - base.pooled_rows;
    struct Check {
        const ConversationBuffer* base;
        const ConversationBuffer* tail;
        int64_t base_units;
        int64_t tail_units;
    };
    const Check checks[] = {
        {&base.k, &tail.k, base.cells, tail_cells},
        {&base.v, &tail.v, base.cells, tail_cells},
        {&base.k_scale, &tail.k_scale, base.cells, tail_cells},
        {&base.v_scale, &tail.v_scale, base.cells, tail_cells},
        {&base.pooled, &tail.pooled, base.pooled_rows, tail_rows},
    };
    // Validate everything before touching the base, so a refused delta leaves the image as it was.
    for (const Check& check : checks) {
        size_t per = 0;
        if (check.tail_units > 0) {
            if (!unit_bytes(*check.tail, check.tail_units, per, error)) return false;
        } else if (!unit_bytes(*check.base, check.base_units, per, error)) {
            return false;
        }
        size_t base_per = 0;
        if (!unit_bytes(*check.base, check.base_units, base_per, error)) return false;
        if (check.tail_units == 0 && !check.tail->empty()) {
            error = "conversation delta: the tail holds bytes for no new cell";
            return false;
        }
        if (check.base_units > 0 && check.tail_units > 0 && base_per != per) {
            error = "conversation delta: the tail and the base disagree on the size of a cell";
            return false;
        }
    }
    ConversationBuffer* targets[] = {&base.k, &base.v, &base.k_scale, &base.v_scale, &base.pooled};
    const ConversationBuffer* sources[] = {&tail.k, &tail.v, &tail.k_scale, &tail.v_scale, &tail.pooled};
    for (size_t i = 0; i < 5; ++i) {
        if (!append(*targets[i], *sources[i])) {
            error = "conversation delta: could not append the tail";
            return false;
        }
    }
    base.cells = tail.cells;
    base.pooled_rows = tail.pooled_rows;
    return true;
}

}  // namespace strata::core
