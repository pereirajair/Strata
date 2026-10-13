#pragma once

#include "strata/core/conversation_cache.hpp"

#include <cstdint>
#include <string>

namespace strata::core {

// A turn's K/V delta: the cells and index rows a full image gained after a base image. The images are cell-major
// (`layout()` in conversation_snapshot.cpp: [cell][kv_head][...]) and index rows are row-major, so each part of the
// delta is one contiguous byte range of the full image. The recurrent state is not here: it changes with every token
// and travels whole with the live checkpoint.
struct KvDelta {
    int64_t base_cells = 0;   // cells the base image holds; the tail starts there
    int64_t base_pooled = 0;  // index rows the base image holds; the tail starts there
    // Same geometry as the full image (format, heads, head_dim, page_size, idx_dim, cells, pooled_rows); each buffer
    // holds only the tail: cells [base_cells, cells) and rows [base_pooled, pooled_rows).
    ConversationKv tail;
};

// The part of `full` after the base extent. Refuses an extent outside the image, and a buffer whose size is not a
// whole number of cells (or rows).
bool kv_delta_extract(const ConversationKv& full, int64_t base_cells, int64_t base_pooled, KvDelta& out,
                      std::string& error);

// Appends `delta` to `base`, which must hold exactly the extent the delta was cut from. Afterwards `base` is the full
// image. A base of another extent, or a tail that does not fit the geometry, is refused and `base` is left as it was.
bool kv_delta_apply(ConversationKv& base, const KvDelta& delta, std::string& error);

}  // namespace strata::core
