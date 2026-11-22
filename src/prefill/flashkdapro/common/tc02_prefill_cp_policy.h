// SPDX-License-Identifier: BSD-3-Clause
#pragma once

namespace flashkda::host_runtime {

struct Tc02PrefillCpPolicy {
    int mult = 0;
    int min_segment_tiles = 0;
    bool short_k1_lb8 = false;
};

// Inputs are counts of 16-token tiles. The BV64 prescan and short merge
// amortize CP on these 39-CU layouts. Plan construction still caps each
// sequence's split by its own length and the minimum segment length.
inline Tc02PrefillCpPolicy tc02_prefill_cp_policy(
        int heads, int batch, int longest, int shortest, int total) {
    if (longest < 20 || longest > 2048) return {};

    if (heads == 3 && batch <= 9) {
        if (batch == 1 && ((90 <= longest && longest <= 104) ||
                           (156 <= longest && longest <= 208))) return {10, 8, true};
        if (batch == 2) {
            if (longest <= 52) return {5, 8};
            if (longest > 72 && 16 <= shortest && shortest <= 32 &&
                4 * longest >= 3 * total) return {10, 8};
        }
        if (total <= 96 * batch) {
            if (4 * longest <= total) return {4, 4};
            if (batch >= 3 && 5 * batch * longest <= 6 * total) return {6, 8, true};
            return {longest <= 72 ? 6 : 8, 8, true};
        }
        if (5 * longest > 2 * total) return {10, 8};
        return {4 * longest <= total ? 8 : 6, 8};
    }

    if ((heads == 4 && batch <= 7) ||
        (heads == 6 && batch <= 6) || (heads == 8 && batch <= 5)) {
        if (longest > 144) return {8, 8};
        if (heads == 4) {
            if (longest <= 56) return {5, 8};
            return total > 256 ? Tc02PrefillCpPolicy{6, 8}
                               : Tc02PrefillCpPolicy{8, 12};
        }
        if (heads == 6) {
            if (50 * longest <= 27 * total)
                return batch <= 5 ? Tc02PrefillCpPolicy{4, 8} : Tc02PrefillCpPolicy{};
            return longest <= 56 ? Tc02PrefillCpPolicy{5, 8}
                                 : Tc02PrefillCpPolicy{8, 12};
        }
        if (10 * longest <= 3 * total) return {};
        if (batch == 1 && longest <= 41) return {4, 8};
        return shortest <= 48 ? Tc02PrefillCpPolicy{6, 8}
                              : Tc02PrefillCpPolicy{8, 12};
    }

    if (heads == 12 && batch <= 4) {
        if (longest > 160) return {8, 8};
        if (batch <= 3) return {total <= 144 ? 5 : 4, 8};
    }
    if (heads == 16 && batch <= 3) {
        // On the small single-sequence grid, three BV64 segments beat four.
        if (batch == 1 && longest <= 34) return {3, 8, true};
        // With 16 heads, short CP amortizes prescan and merge when one
        // sequence dominates a batch of two.
        if (batch == 2 && longest <= 128) {
            if (shortest > 16 || 3 * longest < 2 * total) return {};
            if (longest <= 60) return {8, 12, true};
            if (longest <= 96) return {6, 8, true};
        }
        if (25 * longest <= 12 * total)
            return longest > 256 ? Tc02PrefillCpPolicy{8, 8} : Tc02PrefillCpPolicy{};
        return {total <= 256 ? 4 : 8, 8};
    }
    if (heads == 24 && batch <= 3 && total > 192 * batch &&
        (batch == 1 || longest >= 512)) return {8, 8};
    if (heads == 32 && batch >= 2 && batch <= 3 && (
        total > 64 * batch || (batch == 2 && longest >= 80 && 5 * longest >= 4 * total)))
        return {longest <= 384 ? 4 : 8, 8};
    return {};
}

}  // namespace flashkda::host_runtime
