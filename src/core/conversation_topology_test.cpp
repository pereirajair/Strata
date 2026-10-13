// Topology conversion: a layer split's stage images join into the one image a single machine restores, in layer order;
// every refusal leaves the output alone.
#include "strata/core/conversation_topology.hpp"

#include <cstdio>
#include <cstdlib>

using namespace strata::core;

namespace {

int checks = 0;

void check(bool ok, const char* label) {
    ++checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", label);
        std::exit(1);
    }
}

std::vector<uint8_t> bytes(std::initializer_list<uint8_t> b) { return std::vector<uint8_t>(b); }

ConversationCheckpoint checkpoint(std::vector<uint8_t> gdn, std::vector<uint8_t> tails, uint64_t used) {
    ConversationCheckpoint c;
    c.ids = {7, 8, 9};
    c.gdn = std::move(gdn);
    c.tails = std::move(tails);
    c.dead = bytes({0xD0});
    c.block_pos = bytes({0xB0});
    c.used = used;
    return c;
}

// A stage that holds layers [lo, hi), with one K/V layer per QSA layer (tagged `tag`) and a `ple` history if asked.
SavedConversation stage(int64_t lo, int64_t hi, uint8_t tag, bool with_ple) {
    SavedConversation s;
    for (size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = int64_t(100 + i);
    s.layer_lo = lo;
    s.layer_hi = hi;
    s.cvec = true;
    s.live.ids = {7, 8, 9};
    s.live.gdn = bytes({tag, uint8_t(tag + 1)});
    s.live.tails = bytes({uint8_t(tag + 2)});
    s.live.dead = bytes({tag});
    s.live.block_pos = bytes({tag});
    if (with_ple) s.live.ple = bytes({0xEE});
    s.live.used = 5;
    s.checkpoints.push_back(checkpoint(bytes({tag}), bytes({tag}), 3));
    for (int i = 0; i < 2; ++i) {
        ConversationKv k;
        k.format = tag * 10 + i;   // the layer's order is readable from its format
        s.kv.push_back(k);
    }
    return s;
}

} // namespace

int main() {
    // the two stages of a 48-layer model split at layer 33
    const std::vector<SavedConversation> split = {stage(0, 33, 1, true), stage(33, 48, 2, false)};
    SavedConversation joined;
    std::string error;
    check(conversation_merge_stages(split, joined, error), "two contiguous stages join");
    check(joined.layer_lo == 0 && joined.layer_hi == 48, "the joined image covers the union of the layer ranges");
    check(joined.geometry == split[0].geometry && joined.cvec, "the joined image is the same model");
    check((joined.live.gdn == bytes({1, 2, 2, 3})), "running state: the stages' rows, in layer order");
    check((joined.live.tails == bytes({3, 4})), "tails: the stages' blocks, in layer order");
    check((joined.live.ple == bytes({0xEE})), "ple: the one stage that holds the history");
    check((joined.live.ids == split[0].live.ids && joined.live.used == 5), "tokens and the clock come from the chain");
    check(joined.checkpoints.size() == 1, "one checkpoint in the chain");
    check(joined.checkpoints[0].gdn == bytes({1, 2}) && joined.checkpoints[0].tails == bytes({1, 2}),
          "the checkpoint chain is joined position by position");
    check(joined.kv.size() == 4, "K/V: every layer of every stage");
    check(joined.kv[0].format == 10 && joined.kv[1].format == 11 && joined.kv[2].format == 20 &&
          joined.kv[3].format == 21, "K/V keep the layer order: the first stage's, then the second's (draft last)");
    check(joined.stage_images.empty(), "the joined image has no stages of its own");

    // a single stage is itself
    SavedConversation one;
    check(conversation_merge_stages({stage(0, 48, 4, true)}, one, error) && one.kv.size() == 2, "one stage joins to itself");

    // refusals: the output is left alone (its sentinel stays)
    auto refused = [&](const std::vector<SavedConversation>& stages, const char* label) {
        SavedConversation out;
        out.layer_lo = -1;
        std::string why;
        check(!conversation_merge_stages(stages, out, why), label);
        check(out.layer_lo == -1 && out.kv.empty(), "a refused join leaves the output alone");
        check(!why.empty(), "the refusal says why");
    };
    {
        auto s = split;
        s[1].layer_lo = 34;   // a gap at layer 33
        refused(s, "non-contiguous stages are refused");
    }
    {
        auto s = split;
        s[1].live.ids = {7, 8};
        refused(s, "stages of other tokens are refused");
    }
    {
        auto s = split;
        s[1].live.ple = bytes({0xEF});
        refused(s, "two stages with a ple history are refused");
    }
    {
        auto s = split;
        s[1].geometry[0] = 999;
        refused(s, "stages of another model are refused");
    }
    {
        auto s = split;
        s[0].stage_images.push_back(stage(33, 48, 9, false));
        refused(s, "a stage with stages of its own is refused");
    }
    {
        auto s = split;
        s[1].live.stage_parts.push_back(checkpoint({}, {}, 1));
        refused(s, "a cross-machine stage is refused");
    }
    {
        auto s = split;
        s[1].checkpoints.clear();
        refused(s, "stages with different checkpoint chains are refused");
    }
    std::printf("conversation_topology_test: %d checks passed\n", checks);
    return 0;
}
