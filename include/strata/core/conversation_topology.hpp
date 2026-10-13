#pragma once

#include "strata/core/conversation_cache.hpp"

#include <string>
#include <vector>

namespace strata::core {

// Topology conversion of a conversation image: a layer split saves one image per machine's GPU (the first stage is the
// image's own, the later ones are its stage_images), and a machine with one GPU restores one image with every layer.
// The two forms carry the same bytes: the recurrent blocks (gdn, tails, dead, block_pos) and the K/V layers are per
// layer, in layer order, so joining the stages is concatenation in stage order.  `ple` (the shared history) lives in
// one stage only.  The draft layer's K/V is the last stage's last layer.

// Joins the stages of one split, in layer order (stages[0] holds the first layers), into the image one machine
// restores: its layer range is the union, its K/V list and recurrent blocks are the stages' in order.  Refused, and
// `out` left alone, unless the stages are contiguous, of one model (geometry, cvec), share their tokens and image
// keys, have the same checkpoint chain, carry no stage_parts (a cross-machine split) and no stage images of their own,
// and at most one stage has a `ple` history.
bool conversation_merge_stages(const std::vector<SavedConversation>& stages, SavedConversation& out,
                               std::string& error);

} // namespace strata::core
