#include "strata/core/conversation_topology.hpp"

#include <algorithm>
#include <utility>

namespace strata::core {
namespace {

// The bytes of one recurrent block, joined across stages: stages in order, and at most one of them may hold it.
bool join_blocks(const std::vector<const std::vector<uint8_t>*>& parts, std::vector<uint8_t>& out, const char* what,
                 bool first_stage, std::string& error) {
    (void)what; (void)error;
    out.clear();
    if (first_stage) {
        // the history is per session (not per layer): the first stage's copy is kept; the others are the stages' own
        // histories, which a single session does not join. Equivalence of the answers is checked on the GPU.
        for (const auto* part : parts) {
            if (part->empty()) continue;
            out = *part;
            return true;
        }
        return true;
    }
    for (const auto* part : parts) out.insert(out.end(), part->begin(), part->end());
    return true;
}

bool same_images(const std::vector<ConversationImageKey>& a, const std::vector<ConversationImageKey>& b) {
    return a == b;
}

// The checkpoint chain of every stage joined position by position: the same tokens and image keys at each one.
bool join_checkpoint(const std::vector<const ConversationCheckpoint*>& parts, ConversationCheckpoint& out,
                     std::string& error) {
    const ConversationCheckpoint& first = *parts.front();
    for (const auto* part : parts) {
        if (part->ids != first.ids || !same_images(part->imgs, first.imgs)) {
            error = "topology: the stages' checkpoints are not the same conversation";
            return false;
        }
        if (!part->stage_parts.empty()) {
            error = "topology: a checkpoint of a cross-machine split cannot be joined here";
            return false;
        }
    }
    out = ConversationCheckpoint();
    out.ids = first.ids;
    out.imgs = first.imgs;
    out.used = first.used;
    std::vector<const std::vector<uint8_t>*> gdn, ple, tails, dead, block_pos;
    for (const auto* part : parts) {
        gdn.push_back(&part->gdn); ple.push_back(&part->ple); tails.push_back(&part->tails);
        dead.push_back(&part->dead); block_pos.push_back(&part->block_pos);
    }
    return join_blocks(gdn, out.gdn, "running state", false, error) &&
           join_blocks(ple, out.ple, "ple history", true, error) &&
           join_blocks(tails, out.tails, "tails", false, error) &&
           join_blocks(dead, out.dead, "dead", false, error) &&
           join_blocks(block_pos, out.block_pos, "block positions", false, error);
}

} // namespace

bool conversation_merge_stages(const std::vector<SavedConversation>& stages, SavedConversation& out,
                               std::string& error) {
    if (stages.empty()) {
        error = "topology: no stage to join";
        return false;
    }
    const SavedConversation& first = stages.front();
    for (size_t i = 0; i < stages.size(); ++i) {
        const SavedConversation& s = stages[i];
        if (!s.stage_images.empty()) {
            error = "topology: a stage holds stages of its own";
            return false;
        }
        if (!s.live.stage_parts.empty()) {
            error = "topology: a stage of a cross-machine split cannot be joined here";
            return false;
        }
        if (s.geometry != first.geometry || s.cvec != first.cvec) {
            error = "topology: the stages are not of one model";
            return false;
        }
        if (s.live.ids != first.live.ids || !same_images(s.live.imgs, first.live.imgs)) {
            error = "topology: the stages are not the same conversation";
            return false;
        }
        if (s.checkpoints.size() != first.checkpoints.size()) {
            error = "topology: the stages' checkpoint chains differ in length";
            return false;
        }
        if (i > 0 && s.layer_lo != stages[i - 1].layer_hi) {
            error = "topology: the stages' layer ranges are not contiguous";
            return false;
        }
    }
    SavedConversation joined;
    joined.geometry = first.geometry;
    joined.cvec = first.cvec;
    joined.layer_lo = first.layer_lo;
    joined.layer_hi = stages.back().layer_hi;
    std::vector<const ConversationCheckpoint*> live;
    for (const auto& s : stages) live.push_back(&s.live);
    if (!join_checkpoint(live, joined.live, error)) return false;
    for (size_t c = 0; c < first.checkpoints.size(); ++c) {
        std::vector<const ConversationCheckpoint*> parts;
        for (const auto& s : stages) parts.push_back(&s.checkpoints[c]);
        ConversationCheckpoint chain;
        if (!join_checkpoint(parts, chain, error)) return false;
        joined.checkpoints.push_back(std::move(chain));
    }
    for (const auto& s : stages)
        for (const auto& k : s.kv) joined.kv.push_back(k);
    out = std::move(joined);
    return true;
}

} // namespace strata::core
