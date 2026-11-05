// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "speculative_decoding/stateful/fast_draft_sampling.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

#include "openvino/core/except.hpp"

namespace ov {
namespace genai {
namespace speculative_sampling {

namespace {

constexpr double kTwoPow26 = 67108864.0;
constexpr double kTwoPowMinus53 = 1.0 / 9007199254740992.0;
constexpr size_t kTopPFirstPartialSortSize = 16;
constexpr size_t kNucleusFirstCandidates = 1024;
constexpr double kMinRawKeyTemperature = 0x1p-895;
constexpr double kMaxRawKeyTemperature = 0x1p873;
/**
 * @brief Relative slack on the smallest candidate probability, absorbing last-place jitter of std::exp.
 *
 * The nucleus shortcut assumes exp() is monotone up to this slack. Checked (opt_num2/exp_monotonic.md) with
 * MSVC 19.29 UCRT on an FMA3-only x64 CPU: no violation over all 2^32 floats at T=1, worst exp error about
 * 1.03 * 2^-53, derived inversion slack about 5.6e-16, so 2^-40 leaves a margin of at least 1600. This is a
 * measurement for that platform only, not a guarantee for other libms.
 */
constexpr double kMonotoneSlack = 0x1p-40;
/**
 * @brief Below this probability exp() results are subnormal-range and carry no usable relative accuracy.
 */
constexpr double kMinTrustedProbability = 0x1p-1022;
constexpr size_t kRadixSelectMinDivisor = 16;
constexpr size_t kLargeSelection = 16384;
constexpr size_t kBlockSize = 16;
constexpr size_t kMaxTrackedBlocks = 512;
constexpr size_t kDenseProbeBlocks = 64;
constexpr float kPosInf = std::numeric_limits<float>::infinity();
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

/**
 * @brief Throws unless u lies in the half-open unit interval.
 */
void check_unit_interval(double u, const char* name) {
    OPENVINO_ASSERT(u >= 0.0 && u < 1.0, "Speculative sampling: ", name, " must be in [0, 1), got ", u);
}

/**
 * @brief True when any of the kBlockSize doubles at ptr has a non-zero bit pattern.
 *
 * -0.0 and NaN count as non-zero, so a block reported as empty holds only +0.0.
 */
bool block_has_bits(const double* ptr) {
    uint64_t bits[kBlockSize];
    std::memcpy(bits, ptr, sizeof(bits));
    uint64_t any = 0;
    for (size_t j = 0; j < kBlockSize; ++j) {
        any |= bits[j];
    }
    return any != 0;
}

/**
 * @brief Indices of the first non-empty blocks found while totalling, so the draw pass need not rescan.
 *
 * When more than kMaxTrackedBlocks blocks are non-empty the list is incomplete and overflow is set; the draw
 * pass then rescans.
 */
struct BlockHits {
    std::array<uint32_t, kMaxTrackedBlocks> index;
    size_t count = 0;
    bool overflow = false;
    bool dense = false;
};

/**
 * @brief Sum of weight_of(i) in index order, skipping blocks for which has_mass(first) is false.
 *
 * Adding +0.0 to a running sum that starts at +0.0 never changes it (the sum cannot become -0.0), so skipping
 * blocks whose weights are all exactly +0.0 leaves every partial sum, and therefore the result, bit-identical
 * to the plain index-order loop.
 */
template <typename HasMass, typename WeightFn>
double total_weight(size_t size, HasMass has_mass, WeightFn weight_of, BlockHits& hits) {
    hits.count = 0;
    hits.overflow = false;
    hits.dense = false;
    double total = 0.0;
    const size_t full_blocks = size / kBlockSize;
    for (size_t b = 0; b < full_blocks; ++b) {
        const size_t first = b * kBlockSize;
        if (b == kDenseProbeBlocks && hits.count * 8 >= kDenseProbeBlocks * 7) {
            hits.dense = true;
        }
        if (hits.dense) {
            for (size_t j = 0; j < kBlockSize; ++j) {
                total += weight_of(first + j);
            }
            continue;
        }
        if (!has_mass(first)) {
            continue;
        }
        if (hits.count < kMaxTrackedBlocks) {
            hits.index[hits.count++] = static_cast<uint32_t>(b);
        } else {
            hits.overflow = true;
        }
        for (size_t j = 0; j < kBlockSize; ++j) {
            total += weight_of(first + j);
        }
    }
    for (size_t i = full_blocks * kBlockSize; i < size; ++i) {
        total += weight_of(i);
    }
    return total;
}

/**
 * @brief Inverse-CDF draw over non-negative weights produced by weight_of(i), visiting only recorded blocks.
 *
 * Zero-weight entries never change the running sum, so the strict comparison can never select them; when
 * rounding makes the threshold reach the total, the last positive entry is returned.
 */
template <typename HasMass, typename WeightFn>
size_t draw_index(size_t size, HasMass has_mass, WeightFn weight_of, double u, double total, const BlockHits& hits) {
    const double threshold = u * total;
    double cumulative = 0.0;
    size_t last_positive = size;
    const auto visit = [&](size_t i) {
        const double w = weight_of(i);
        if (w > 0.0) {
            cumulative += w;
            last_positive = i;
            return cumulative > threshold;
        }
        return false;
    };
    const auto visit_block = [&](size_t first) {
        for (size_t j = 0; j < kBlockSize; ++j) {
            if (visit(first + j)) {
                return first + j;
            }
        }
        return size;
    };
    const size_t full_blocks = size / kBlockSize;
    if (hits.dense) {
        for (size_t i = 0; i < full_blocks * kBlockSize; ++i) {
            if (visit(i)) {
                return i;
            }
        }
    }
    const bool all_blocks = hits.overflow || hits.dense;
    const size_t visited = hits.dense ? 0 : (all_blocks ? full_blocks : hits.count);
    for (size_t k = 0; k < visited; ++k) {
        const size_t first = (all_blocks ? k : static_cast<size_t>(hits.index[k])) * kBlockSize;
        if (hits.overflow && !has_mass(first)) {
            continue;
        }
        const size_t found = visit_block(first);
        if (found != size) {
            return found;
        }
    }
    for (size_t i = full_blocks * kBlockSize; i < size; ++i) {
        if (visit(i)) {
            return i;
        }
    }
    OPENVINO_ASSERT(last_positive < size, "Speculative sampling: distribution has no positive mass");
    return last_positive;
}

/**
 * @brief Result of the validation sweeps over the valid logits.
 */
struct LogitScan {
    float max_logit;
    bool invalid;
};

/**
 * @brief Maximum logit and whether any logit is NaN or +inf, in two simple loops the compiler vectorises.
 *
 * NaN is detected by x != x (C++17 [expr.eq]); +inf shows up as a maximum of +inf. std::max ignores NaN, which
 * is harmless because a NaN row is rejected by the flag anyway. Both results are order-independent.
 */
LogitScan scan_logits(const float* logits, size_t count) {
    float max_logit = kNegInf;
    for (size_t i = 0; i < count; ++i) {
        max_logit = std::max(max_logit, logits[i]);
    }
    uint32_t has_nan = 0;
    for (size_t i = 0; i < count; ++i) {
        has_nan |= static_cast<uint32_t>(logits[i] != logits[i]);
    }
    return {max_logit, has_nan != 0 || max_logit == kPosInf};
}

/**
 * @brief Number of logits above -inf (NaN is excluded beforehand by scan_logits).
 */
size_t count_finite(const float* logits, size_t count) {
    uint32_t finite = 0;
    for (size_t i = 0; i < count; ++i) {
        finite += static_cast<uint32_t>(logits[i] > kNegInf);
    }
    return finite;
}

/**
 * @brief Throws the NaN / +inf diagnostic for the first offending logit.
 */
[[noreturn]] void throw_invalid_logit(const float* logits, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        const double value = static_cast<double>(logits[i]);
        OPENVINO_ASSERT(!std::isnan(value) && value != std::numeric_limits<double>::infinity(),
                        "Speculative sampling: logit ",
                        i,
                        " is NaN or +inf");
    }
    OPENVINO_THROW("Speculative sampling: invalid logits");
}

/**
 * @brief Collects the ids of the k largest values into ids, unordered; fewer only if fewer values exceed -inf.
 *
 * The order is (value desc, id asc). Ids are visited in ascending order and only a value strictly greater
 * than the current k-th best is appended, so an equal value seen later (larger id) never displaces an
 * earlier one. The buffer is cut back to k with nth_element whenever it reaches 2k entries, which gives
 * O(n) expected time plus O(k log(n / k)) comparator work. Values must not be NaN.
 */
template <typename ValueFn>
void select_top_k(size_t count, size_t k, ValueFn value, std::vector<uint32_t>& ids) {
    using Value = decltype(value(size_t{0}));
    const auto better = [&value](uint32_t a, uint32_t b) {
        const Value va = value(a);
        const Value vb = value(b);
        return va > vb || (va == vb && a < b);
    };
    const size_t capacity = std::max<size_t>(2 * k, 64);
    const auto cut_to_k = [&]() {
        std::nth_element(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(k - 1), ids.end(), better);
        ids.resize(k);
        return value(ids[k - 1]);
    };
    ids.clear();
    ids.reserve(capacity);
    Value gate = -std::numeric_limits<Value>::infinity();
    for (size_t i = 0; i < count; ++i) {
        if (value(i) > gate) {
            ids.push_back(static_cast<uint32_t>(i));
            if (ids.size() == capacity) {
                gate = cut_to_k();
            }
        }
    }
    if (ids.size() > k) {
        cut_to_k();
    }
}

/**
 * @brief True when comparing raw float logits orders them exactly as logit / temperature does in double.
 *
 * Division by a temperature in [2^-895, 2^873] maps distinct floats to distinct doubles and never overflows
 * to +-inf or underflows to zero for any float, so both orders and both tie sets coincide.
 */
bool raw_logit_order_is_exact(double temperature) {
    return temperature >= kMinRawKeyTemperature && temperature <= kMaxRawKeyTemperature;
}

/**
 * @brief Order-preserving 32-bit key of a non-NaN float: a larger value gives a larger key, -0.0 equals +0.0.
 */
uint32_t order_key(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits = bits == 0x80000000u ? 0u : bits;
    return (bits & 0x80000000u) != 0 ? ~bits : (bits | 0x80000000u);
}

/**
 * @brief Key of the k-th largest logit and how many entries equal to it belong to the top k.
 */
struct KthKey {
    uint32_t key;
    size_t equal_quota;
};

/**
 * @brief Exact k-th largest key by three radix passes (11 + 11 + 10 bits) over the logits.
 *
 * Each pass histograms the next digit of the keys that match the digits fixed so far, then walks the digits
 * from the top to find the one holding rank k. Cost is 3n cheap reads and 8 KiB of stack, with no data-dependent
 * branching on the values. Requires 1 <= k <= count and no NaN.
 */
KthKey find_kth_largest_key(const float* logits, size_t count, size_t k) {
    struct Pass {
        unsigned shift;
        uint32_t mask;
    };
    constexpr Pass kPasses[] = {{21, 0x7FFu}, {10, 0x7FFu}, {0, 0x3FFu}};
    std::array<uint32_t, 2048> histogram;
    uint32_t prefix = 0;
    uint32_t prefix_mask = 0;
    size_t remaining = k;
    for (const Pass& pass : kPasses) {
        histogram.fill(0);
        for (size_t i = 0; i < count; ++i) {
            const uint32_t key = order_key(logits[i]);
            if ((key & prefix_mask) == prefix) {
                ++histogram[(key >> pass.shift) & pass.mask];
            }
        }
        size_t digit = static_cast<size_t>(pass.mask) + 1;
        bool found = false;
        while (digit-- > 0) {
            if (histogram[digit] >= remaining) {
                found = true;
                break;
            }
            remaining -= histogram[digit];
        }
        OPENVINO_ASSERT(found, "Speculative sampling: top_k radix selection lost its rank");
        prefix |= static_cast<uint32_t>(digit) << pass.shift;
        prefix_mask |= pass.mask << pass.shift;
    }
    return {prefix, remaining};
}

/**
 * @brief Selects the ids of the k largest logits in ascending id order without sorting.
 *
 * Entries with a key above the k-th key are taken, and among entries equal to it only the first equal_quota
 * by ascending id, which is the (value desc, id asc) tie rule. The final pass is branch-free.
 */
void select_top_k_ascending(const float* logits, size_t count, size_t k, std::vector<uint32_t>& ids) {
    const KthKey kth = find_kth_largest_key(logits, count, k);
    ids.resize(k + 1);
    size_t taken = 0;
    size_t quota = kth.equal_quota;
    for (size_t i = 0; i < count; ++i) {
        const uint32_t key = order_key(logits[i]);
        const size_t is_equal = key == kth.key ? 1 : 0;
        const size_t take = (key > kth.key ? 1 : 0) | (is_equal & (quota > 0 ? 1 : 0));
        ids[taken] = static_cast<uint32_t>(i);
        taken += take;
        quota -= is_equal & take;
    }
    OPENVINO_ASSERT(taken == k, "Speculative sampling: top_k radix selection produced ", taken, " ids, expected ", k);
    ids.resize(k);
}

/**
 * @brief Collects the top_k ids in ascending id order; the order the probabilities are summed in.
 */
void select_top_k_ids(const float* logits, size_t count, size_t k, double temperature, std::vector<uint32_t>& ids) {
    if (!raw_logit_order_is_exact(temperature)) {
        const auto scaled = [logits, temperature](size_t i) {
            return static_cast<double>(logits[i]) / temperature;
        };
        select_top_k(count, k, scaled, ids);
        std::sort(ids.begin(), ids.end());
    } else if (k >= count / kRadixSelectMinDivisor) {
        select_top_k_ascending(logits, count, k, ids);
    } else {
        select_top_k(
            count,
            k,
            [logits](size_t i) {
                return logits[i];
            },
            ids);
        std::sort(ids.begin(), ids.end());
    }
}

/**
 * @brief Collects the ids of the `want` largest scaled logits, unordered, for nucleus search.
 */
void select_leading_ids(const float* logits,
                        size_t count,
                        size_t want,
                        double temperature,
                        std::vector<uint32_t>& ids) {
    if (raw_logit_order_is_exact(temperature)) {
        select_top_k(
            count,
            want,
            [logits](size_t i) {
                return logits[i];
            },
            ids);
        return;
    }
    const auto scaled = [logits, temperature](size_t i) {
        return static_cast<double>(logits[i]) / temperature;
    };
    select_top_k(count, want, scaled, ids);
}

/**
 * @brief Sorts order by descending probability (lower id first on ties) and returns the nucleus size.
 *
 * The result is the smallest prefix whose cumulative probability exceeds mass_limit, crossing token included,
 * or order.size() if the mass never exceeds it. When order is only a candidate subset, every id outside it
 * has probability <= floor, so a crossing at a probability not above floor is not trusted and 0 is returned
 * (use floor < 0 for a complete set).
 */
double trusted_floor(bool complete, double smallest) {
    if (complete) {
        return -1.0;
    }
    return smallest < kMinTrustedProbability ? std::numeric_limits<double>::infinity()
                                             : smallest * (1.0 + kMonotoneSlack);
}

size_t nucleus_size(std::vector<uint32_t>& order, const double* prob, double mass_limit, double floor) {
    const auto by_prob = [prob](uint32_t a, uint32_t b) {
        return prob[a] > prob[b] || (prob[a] == prob[b] && a < b);
    };
    for (size_t window = kTopPFirstPartialSortSize;; window *= 64) {
        const size_t sorted = std::min(window, order.size());
        if (sorted == order.size()) {
            std::sort(order.begin(), order.end(), by_prob);
        } else {
            std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(sorted), order.end(), by_prob);
        }
        double cumulative = 0.0;
        for (size_t i = 0; i < sorted; ++i) {
            cumulative += prob[order[i]];
            if (cumulative > mass_limit) {
                return prob[order[i]] > floor ? i + 1 : 0;
            }
        }
        if (sorted == order.size()) {
            return floor < 0.0 ? order.size() : 0;
        }
    }
}

/**
 * @brief Row-wide constants shared by the three compute_probs paths.
 */
struct RowContext {
    const float* logits;
    size_t vocab_size;
    size_t valid_vocab_size;
    double scale;
    double max_scaled;
    double top_p;
};

double exp_term(const RowContext& ctx, size_t i) {
    return std::exp(static_cast<double>(ctx.logits[i]) / ctx.scale - ctx.max_scaled);
}

/**
 * @brief Plain softmax over the finite logits, summed in ascending id order, written over the whole row.
 */
void write_softmax_row(const RowContext& ctx, ProbRow& out) {
    out.prob.resize(ctx.vocab_size);
    double* prob = out.prob.data();
    double sum = 0.0;
    for (size_t i = 0; i < ctx.valid_vocab_size; ++i) {
        const double e = ctx.logits[i] > kNegInf ? exp_term(ctx, i) : 0.0;
        prob[i] = e;
        sum += e;
    }
    for (size_t i = 0; i < ctx.valid_vocab_size; ++i) {
        prob[i] /= sum;
    }
    std::fill(prob + ctx.valid_vocab_size, prob + ctx.vocab_size, 0.0);
}

/**
 * @brief Zeroes the candidates outside the nucleus and renormalises the kept ones by their kept mass.
 *
 * The kept mass is accumulated in sorted order exactly as the reference does; the divisor is that mass and
 * each kept value was already divided by the full softmax sum.
 */
void finish_nucleus(std::vector<uint32_t>& order, size_t keep, double* prob, Scratch& scratch) {
    auto& kept = scratch.scaled;
    kept.resize(keep);
    double kept_sum = 0.0;
    for (size_t i = 0; i < keep; ++i) {
        kept[i] = prob[order[i]];
        kept_sum += kept[i];
    }
    for (size_t i = keep; i < order.size(); ++i) {
        prob[order[i]] = 0.0;
    }
    for (size_t i = 0; i < keep; ++i) {
        prob[order[i]] = kept[i] / kept_sum;
    }
}

/**
 * @brief Top_k active: the k selected ids carry the whole distribution, optionally cut to the nucleus.
 *
 * Probabilities are summed over the selected ids in ascending id order. The reference sums them in the
 * unspecified order std::nth_element leaves behind, so equality with it holds to the standard summation bound
 * rather than bit for bit.
 */
size_t nucleus_within_selected(const RowContext& ctx, std::vector<uint32_t>& order, double* prob);

void finish_top_k_row(const RowContext& ctx, ProbRow& out, Scratch& scratch) {
    auto& order = scratch.order;
    out.prob.assign(ctx.vocab_size, 0.0);
    double* prob = out.prob.data();
    double sum = 0.0;
    for (uint32_t id : order) {
        const double e = exp_term(ctx, id);
        prob[id] = e;
        sum += e;
    }
    for (uint32_t id : order) {
        prob[id] /= sum;
    }
    if (ctx.top_p < 1.0) {
        const size_t keep = order.size() > kLargeSelection && raw_logit_order_is_exact(ctx.scale)
                                ? nucleus_within_selected(ctx, order, prob)
                                : nucleus_size(order, prob, ctx.top_p, -1.0);
        finish_nucleus(order, keep, prob, scratch);
    }
}

/**
 * @brief Large top_k: radix or double-keyed selection of the ids, then the shared finishing step.
 */
void compute_top_k_row(const RowContext& ctx, size_t top_k, ProbRow& out, Scratch& scratch) {
    select_top_k_ids(ctx.logits, ctx.valid_vocab_size, top_k, ctx.scale, scratch.order);
    finish_top_k_row(ctx, out, scratch);
}

/**
 * @brief Sum of exp terms over the finite logits in ascending id order, without storing them.
 */
double sum_exp_terms(const RowContext& ctx) {
    double sum = 0.0;
    for (size_t i = 0; i < ctx.valid_vocab_size; ++i) {
        sum += ctx.logits[i] > kNegInf ? exp_term(ctx, i) : 0.0;
    }
    return sum;
}

/**
 * @brief Top_k off, top_p < 1: only the nucleus candidates are ever turned into probabilities.
 *
 * The softmax sum is accumulated in ascending id order (identical to the reference), then candidates are the
 * `want` largest logits, widened until the nucleus provably lies inside them. A crossing is trusted only if its
 * probability exceeds the smallest candidate probability by a margin that covers last-place exp() jitter, so
 * every id left out sorts after it. Output values equal the reference bit for bit.
 */
void compute_nucleus_row_generic(const RowContext& ctx, ProbRow& out, Scratch& scratch) {
    const double sum = sum_exp_terms(ctx);
    out.prob.assign(ctx.vocab_size, 0.0);
    double* prob = out.prob.data();
    auto& order = scratch.order;
    size_t keep = 0;
    for (size_t want = kNucleusFirstCandidates; keep == 0; want = std::min(want * 8, ctx.valid_vocab_size)) {
        select_leading_ids(ctx.logits, ctx.valid_vocab_size, want, ctx.scale, order);
        const bool complete = want >= ctx.valid_vocab_size || order.size() < want;
        double floor = complete ? -1.0 : std::numeric_limits<double>::infinity();
        double mass = 0.0;
        for (uint32_t id : order) {
            prob[id] = exp_term(ctx, id) / sum;
            floor = complete ? floor : std::min(floor, prob[id]);
            mass += prob[id];
        }
        if (complete || mass > ctx.top_p) {
            keep = nucleus_size(order, prob, ctx.top_p, trusted_floor(complete, floor));
        }
        if (keep == 0) {
            for (uint32_t id : order) {
                prob[id] = 0.0;
            }
        }
    }
    finish_nucleus(order, keep, prob, scratch);
}

constexpr size_t kMassBins = 2048;
constexpr double kBinsPerNat = 16.0;
constexpr size_t kNucleusMargins[] = {2, 34, 130};
constexpr size_t kSortAllBelow = 16384;

/**
 * @brief Sorts the candidates by (probability desc, id asc) and returns the nucleus size, 0 if untrusted.
 *
 * Small candidate sets are fully sorted in one go; larger ones use the windowed partial sort of nucleus_size.
 */
size_t nucleus_of_candidates(std::vector<uint32_t>& order, const double* prob, double top_p, double floor) {
    if (order.size() > kSortAllBelow) {
        return nucleus_size(order, prob, top_p, floor);
    }
    std::sort(order.begin(), order.end(), [prob](uint32_t a, uint32_t b) {
        return prob[a] > prob[b] || (prob[a] == prob[b] && a < b);
    });
    double cumulative = 0.0;
    for (size_t i = 0; i < order.size(); ++i) {
        cumulative += prob[order[i]];
        if (cumulative > top_p) {
            return prob[order[i]] > floor ? i + 1 : 0;
        }
    }
    return floor < 0.0 ? order.size() : 0;
}

/**
 * @brief Top_k off, top_p < 1, raw-logit order exact: candidates come from an exp-mass histogram threshold.
 *
 * One ascending-id pass accumulates the softmax sum (bit-identical to the reference) and a histogram of the
 * exp mass per 1/16 nat of scaled logit. The bin where the cumulative mass passes top_p fixes a logit cut-off
 * with a safety margin; the candidates are all ids whose logit is at or above it. That is an upper set of the
 * logit order, so the candidate-nucleus argument applies (proofs_P1): a crossing is accepted only when its
 * probability exceeds the smallest candidate probability (times 1 + kMonotoneSlack). The histogram only steers
 * the choice, so a wrong guess costs a wider margin, never a wrong result; after the last margin the whole
 * finite set is used, which equals the reference computation.
 */
void compute_nucleus_row(const RowContext& ctx, ProbRow& out, Scratch& scratch) {
    const float* logits = ctx.logits;
    const size_t n = ctx.valid_vocab_size;
    std::array<double, kMassBins> mass;
    mass.fill(0.0);
    double sum = 0.0;
    size_t finite = 0;
    for (size_t i = 0; i < n; ++i) {
        if (logits[i] > kNegInf) {
            const double s = static_cast<double>(logits[i]) / ctx.scale;
            const double e = std::exp(s - ctx.max_scaled);
            sum += e;
            const double d = (ctx.max_scaled - s) * kBinsPerNat;
            mass[d < static_cast<double>(kMassBins - 1) ? static_cast<size_t>(d) : kMassBins - 1] += e;
            ++finite;
        }
    }
    size_t cut_bin = kMassBins;
    double cumulative = 0.0;
    for (size_t b = 0; b < kMassBins; ++b) {
        cumulative += mass[b];
        if (cumulative / sum > ctx.top_p + 1e-9) {
            cut_bin = b;
            break;
        }
    }
    out.prob.assign(ctx.vocab_size, 0.0);
    double* prob = out.prob.data();
    auto& order = scratch.order;
    size_t keep = 0;
    for (size_t attempt = 0; keep == 0; ++attempt) {
        const size_t margin =
            attempt < sizeof(kNucleusMargins) / sizeof(kNucleusMargins[0]) ? kNucleusMargins[attempt] : kMassBins;
        float cut = kNegInf;
        if (cut_bin + margin < kMassBins) {
            const double cut_scaled = ctx.max_scaled - static_cast<double>(cut_bin + margin) / kBinsPerNat - 1e-9;
            cut = std::nextafter(static_cast<float>(cut_scaled * ctx.scale), kNegInf);
        }
        order.clear();
        for (size_t i = 0; i < n; ++i) {
            if (logits[i] > kNegInf && logits[i] >= cut) {
                order.push_back(static_cast<uint32_t>(i));
            }
        }
        const bool complete = order.size() == finite;
        double floor = complete ? -1.0 : std::numeric_limits<double>::infinity();
        for (uint32_t id : order) {
            prob[id] = exp_term(ctx, id) / sum;
            floor = complete ? floor : std::min(floor, prob[id]);
        }
        keep = nucleus_of_candidates(order, prob, ctx.top_p, trusted_floor(complete, floor));
        if (keep == 0) {
            for (uint32_t id : order) {
                prob[id] = 0.0;
            }
        }
    }
    finish_nucleus(order, keep, prob, scratch);
}

/**
 * @brief Nucleus of a large top_k selection, found without sorting every selected id.
 *
 * Probabilities of all ids in order are already in prob (normalised over the selection). A mass histogram over
 * them gives a logit cut-off as in compute_nucleus_row; ids below it are zeroed and dropped from order, the
 * rest are sorted and the crossing is accepted only above the smallest candidate probability, with the same
 * widening and final full-set fallback. Returns the nucleus size; order then holds the candidates, the first
 * `keep` of them sorted as the nucleus.
 */
size_t nucleus_within_selected(const RowContext& ctx, std::vector<uint32_t>& order, double* prob) {
    const float* logits = ctx.logits;
    std::array<double, kMassBins> mass;
    mass.fill(0.0);
    for (uint32_t id : order) {
        const double d = (ctx.max_scaled - static_cast<double>(logits[id]) / ctx.scale) * kBinsPerNat;
        mass[d < static_cast<double>(kMassBins - 1) ? static_cast<size_t>(d) : kMassBins - 1] += prob[id];
    }
    size_t cut_bin = kMassBins;
    double cumulative = 0.0;
    for (size_t b = 0; b < kMassBins; ++b) {
        cumulative += mass[b];
        if (cumulative > ctx.top_p + 1e-9) {
            cut_bin = b;
            break;
        }
    }
    const size_t total = order.size();
    const auto by_prob = [prob](uint32_t x, uint32_t y) {
        return prob[x] > prob[y] || (prob[x] == prob[y] && x < y);
    };
    for (size_t attempt = 0;; ++attempt) {
        const size_t margin =
            attempt < sizeof(kNucleusMargins) / sizeof(kNucleusMargins[0]) ? kNucleusMargins[attempt] : kMassBins;
        float cut = kNegInf;
        if (cut_bin + margin < kMassBins) {
            const double cut_scaled = ctx.max_scaled - static_cast<double>(cut_bin + margin) / kBinsPerNat - 1e-9;
            cut = std::nextafter(static_cast<float>(cut_scaled * ctx.scale), kNegInf);
        }
        const auto split = std::partition(order.begin(), order.end(), [logits, cut](uint32_t id) {
            return logits[id] >= cut;
        });
        const size_t m = static_cast<size_t>(split - order.begin());
        const bool complete = m == total;
        double floor = complete ? -1.0 : std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < m; ++i) {
            floor = complete ? floor : std::min(floor, prob[order[i]]);
        }
        floor = trusted_floor(complete, floor);
        std::sort(order.begin(), split, by_prob);
        size_t keep = complete ? m : 0;
        double running = 0.0;
        for (size_t i = 0; i < m; ++i) {
            running += prob[order[i]];
            if (running > ctx.top_p) {
                keep = prob[order[i]] > floor ? i + 1 : 0;
                break;
            }
        }
        if (keep != 0) {
            for (size_t i = m; i < total; ++i) {
                prob[order[i]] = 0.0;
            }
            order.resize(m);
            return keep;
        }
    }
}

/**
 * @brief Streams the logits once, collecting the ids of the k largest and flagging NaN.
 *
 * Same selection as select_top_k; the NaN test rides along in the single pass so small k needs no separate
 * validation sweep. Returns true if a NaN was seen.
 */
template <typename ValueFn>
bool select_top_k_checked(size_t count, size_t k, ValueFn value, std::vector<uint32_t>& ids) {
    using Value = decltype(value(size_t{0}));
    const auto better = [&value](uint32_t a, uint32_t b) {
        const Value va = value(a);
        const Value vb = value(b);
        return va > vb || (va == vb && a < b);
    };
    const size_t capacity = std::max<size_t>(2 * k, 64);
    const auto cut_to_k = [&]() {
        std::nth_element(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(k - 1), ids.end(), better);
        ids.resize(k);
        return value(ids[k - 1]);
    };
    ids.clear();
    ids.reserve(capacity);
    Value gate = -std::numeric_limits<Value>::infinity();
    bool has_nan = false;
    for (size_t i = 0; i < count; ++i) {
        const Value x = value(i);
        has_nan |= x != x;
        if (x > gate) {
            ids.push_back(static_cast<uint32_t>(i));
            if (ids.size() == capacity) {
                gate = cut_to_k();
            }
        }
    }
    if (ids.size() > k) {
        cut_to_k();
    }
    return has_nan;
}

/**
 * @brief Small top_k: one streaming pass selects, validates and finds the maximum.
 *
 * +inf, if present, is always among the selected ids because nothing exceeds it, so the maximum and the +inf
 * test are taken over the selected ids only. A row with no logit above -inf leaves the selection empty.
 */
void compute_small_top_k_row(const float* logits,
                             size_t vocab_size,
                             size_t valid_vocab_size,
                             const SamplingParams& params,
                             ProbRow& out,
                             Scratch& scratch) {
    auto& order = scratch.order;
    bool has_nan;
    if (raw_logit_order_is_exact(params.temperature)) {
        has_nan = select_top_k_checked(
            valid_vocab_size,
            params.top_k,
            [logits](size_t i) {
                return logits[i];
            },
            order);
    } else {
        const double scale = params.temperature;
        has_nan = select_top_k_checked(
            valid_vocab_size,
            params.top_k,
            [logits, scale](size_t i) {
                return static_cast<double>(logits[i]) / scale;
            },
            order);
    }
    float max_logit = kNegInf;
    bool has_pos_inf = false;
    for (uint32_t id : order) {
        max_logit = std::max(max_logit, logits[id]);
        has_pos_inf |= logits[id] == kPosInf;
    }
    const bool invalid = has_nan || has_pos_inf;
    if (invalid || order.empty()) {
        out.prob.assign(vocab_size, 0.0);
    }
    if (invalid) {
        throw_invalid_logit(logits, valid_vocab_size);
    }
    OPENVINO_ASSERT(!order.empty(), "Speculative sampling: no finite logit in the valid vocabulary range");
    std::sort(order.begin(), order.end());
    const RowContext ctx{logits,
                         vocab_size,
                         valid_vocab_size,
                         params.temperature,
                         static_cast<double>(max_logit) / params.temperature,
                         params.top_p};
    finish_top_k_row(ctx, out, scratch);
}

}  // namespace

SamplingParams make_sampling_params(const GenerationConfig& config) {
    OPENVINO_ASSERT(std::isfinite(config.temperature) && config.temperature > 0.0f,
                    "Speculative sampling requires 'temperature' > 0, got ",
                    config.temperature);
    OPENVINO_ASSERT(config.top_p > 0.0f && config.top_p <= 1.0f,
                    "Speculative sampling requires 'top_p' in (0, 1], got ",
                    config.top_p);
    OPENVINO_ASSERT(config.top_k >= 1, "Speculative sampling requires 'top_k' >= 1");
    OPENVINO_ASSERT(config.repetition_penalty == 1.0f, "Speculative sampling does not support 'repetition_penalty'");
    OPENVINO_ASSERT(config.presence_penalty == 0.0f, "Speculative sampling does not support 'presence_penalty'");
    OPENVINO_ASSERT(config.frequency_penalty == 0.0f, "Speculative sampling does not support 'frequency_penalty'");
    OPENVINO_ASSERT(config.min_p == 0.0f, "Speculative sampling does not support 'min_p'");
    OPENVINO_ASSERT(config.logprobs == 0, "Speculative sampling does not support 'logprobs'");
    OPENVINO_ASSERT(config.min_new_tokens == 0, "Speculative sampling does not support 'min_new_tokens'");
    OPENVINO_ASSERT(config.num_beams == 1, "Speculative sampling does not support beam search ('num_beams' must be 1)");
    OPENVINO_ASSERT(config.num_return_sequences == 1,
                    "Speculative sampling requires 'num_return_sequences' equal to 1");
    OPENVINO_ASSERT(!config.structured_output_config.has_value(),
                    "Speculative sampling does not support 'structured_output_config'");

    SamplingParams params;
    params.temperature = static_cast<double>(config.temperature);
    params.top_k = config.top_k;
    params.top_p = static_cast<double>(config.top_p);
    return params;
}

double uniform01(std::mt19937& rng) {
    const uint64_t high = static_cast<uint64_t>(rng()) >> 5;
    const uint64_t low = static_cast<uint64_t>(rng()) >> 6;
    return (static_cast<double>(high) * kTwoPow26 + static_cast<double>(low)) * kTwoPowMinus53;
}

void compute_probs(const float* logits,
                   size_t vocab_size,
                   size_t valid_vocab_size,
                   const SamplingParams& params,
                   ProbRow& out,
                   Scratch& scratch) {
    OPENVINO_ASSERT(logits != nullptr, "Speculative sampling: logits pointer is null");
    OPENVINO_ASSERT(vocab_size > 0, "Speculative sampling: vocabulary size must be greater than 0");
    OPENVINO_ASSERT(valid_vocab_size > 0 && valid_vocab_size <= vocab_size,
                    "Speculative sampling: invalid valid vocabulary size ",
                    valid_vocab_size,
                    " for vocabulary ",
                    vocab_size);
    OPENVINO_ASSERT(std::isfinite(params.temperature) && params.temperature > 0.0,
                    "Speculative sampling: temperature must be > 0");
    OPENVINO_ASSERT(params.top_k >= 1, "Speculative sampling: top_k must be >= 1");
    OPENVINO_ASSERT(params.top_p > 0.0 && params.top_p <= 1.0, "Speculative sampling: top_p must be in (0, 1]");

    if (params.top_k < valid_vocab_size / kRadixSelectMinDivisor) {
        compute_small_top_k_row(logits, vocab_size, valid_vocab_size, params, out, scratch);
        return;
    }

    const LogitScan scan = scan_logits(logits, valid_vocab_size);
    const bool no_finite = !scan.invalid && scan.max_logit == kNegInf;
    if (scan.invalid || no_finite) {
        out.prob.assign(vocab_size, 0.0);
    }
    if (scan.invalid) {
        throw_invalid_logit(logits, valid_vocab_size);
    }
    OPENVINO_ASSERT(!no_finite, "Speculative sampling: no finite logit in the valid vocabulary range");

    const RowContext ctx{logits,
                         vocab_size,
                         valid_vocab_size,
                         params.temperature,
                         static_cast<double>(scan.max_logit) / params.temperature,
                         params.top_p};
    if (params.top_k < valid_vocab_size && params.top_k < count_finite(logits, valid_vocab_size)) {
        compute_top_k_row(ctx, params.top_k, out, scratch);
    } else if (params.top_p < 1.0) {
        if (raw_logit_order_is_exact(params.temperature)) {
            compute_nucleus_row(ctx, out, scratch);
        } else {
            compute_nucleus_row_generic(ctx, out, scratch);
        }
    } else {
        write_softmax_row(ctx, out);
    }
}

size_t sample_from(const ProbRow& row, double u) {
    check_unit_interval(u, "uniform number");
    const double* prob = row.prob.data();
    const size_t size = row.prob.size();
    const auto has_mass = [prob](size_t first) {
        return block_has_bits(prob + first);
    };
    const auto weight_of = [prob](size_t i) {
        return prob[i];
    };
    BlockHits hits;
    const double total = total_weight(size, has_mass, weight_of, hits);
    OPENVINO_ASSERT(size > 0 && std::isfinite(total) && total > 0.0,
                    "Speculative sampling: distribution has no positive finite mass");
    return draw_index(size, has_mass, weight_of, u, total, hits);
}

namespace {

/**
 * @brief The single accept/resample loop shared by every verification entry point.
 *
 * target_row(i) must return q_i and is called for position i only when verification reaches it: once per
 * position in ascending order, row 0 first, row K only after all K candidates were accepted. Rows are
 * therefore materialised lazily by verify_block and simply looked up by verify_with_uniforms.
 *
 * @param vocab_size Width every draft and target row must have; checked by the callers.
 */
template <typename TargetRowFn>
VerifyResult verify_core(const std::vector<int64_t>& candidates,
                         const std::vector<ProbRow>& draft_rows,
                         TargetRowFn target_row,
                         const std::vector<double>& accept_u,
                         double final_u,
                         size_t vocab_size) {
    const size_t num_candidates = candidates.size();
    for (size_t i = 0; i < num_candidates; ++i) {
        const int64_t token = candidates[i];
        OPENVINO_ASSERT(token >= 0 && static_cast<size_t>(token) < vocab_size,
                        "Speculative sampling: candidate ",
                        i,
                        " has out-of-range token id ",
                        token);
        check_unit_interval(accept_u[i], "acceptance uniform number");

        const ProbRow& target = target_row(i);
        const double* p = draft_rows[i].prob.data();
        const double* q = target.prob.data();
        const double draft_prob = p[static_cast<size_t>(token)];
        const double target_prob = q[static_cast<size_t>(token)];
        const double acceptance = draft_prob > 0.0 ? std::min(1.0, target_prob / draft_prob) : 0.0;
        if (accept_u[i] < acceptance) {
            continue;
        }

        const auto has_mass = [p, q](size_t first) {
            return block_has_bits(q + first) || block_has_bits(p + first);
        };
        const auto residual_weight = [p, q](size_t j) {
            return std::max(0.0, q[j] - p[j]);
        };
        BlockHits hits;
        const double residual_total = total_weight(vocab_size, has_mass, residual_weight, hits);
        VerifyResult result;
        result.num_accepted = i;
        result.all_accepted = false;
        if (residual_total > 0.0) {
            result.final_token =
                static_cast<int64_t>(draw_index(vocab_size, has_mass, residual_weight, final_u, residual_total, hits));
        } else {
            result.final_token = static_cast<int64_t>(sample_from(target, final_u));
        }
        return result;
    }

    VerifyResult result;
    result.num_accepted = num_candidates;
    result.all_accepted = true;
    result.final_token = static_cast<int64_t>(sample_from(target_row(num_candidates), final_u));
    return result;
}

}  // namespace

VerifyResult verify_with_uniforms(const std::vector<int64_t>& candidates,
                                  const std::vector<ProbRow>& draft_rows,
                                  const std::vector<ProbRow>& target_rows,
                                  const std::vector<double>& accept_u,
                                  double final_u) {
    const size_t num_candidates = candidates.size();
    OPENVINO_ASSERT(draft_rows.size() >= num_candidates,
                    "Speculative sampling: expected at least ",
                    num_candidates,
                    " draft rows, got ",
                    draft_rows.size());
    OPENVINO_ASSERT(target_rows.size() >= num_candidates + 1,
                    "Speculative sampling: expected at least ",
                    num_candidates + 1,
                    " target rows, got ",
                    target_rows.size());
    OPENVINO_ASSERT(accept_u.size() == num_candidates,
                    "Speculative sampling: expected ",
                    num_candidates,
                    " acceptance uniforms, got ",
                    accept_u.size());
    check_unit_interval(final_u, "final uniform number");

    const size_t vocab_size = target_rows.front().prob.size();
    OPENVINO_ASSERT(vocab_size > 0, "Speculative sampling: empty target row");
    for (size_t i = 0; i <= num_candidates; ++i) {
        OPENVINO_ASSERT(target_rows[i].prob.size() == vocab_size, "Speculative sampling: target rows differ in size");
    }
    for (size_t i = 0; i < num_candidates; ++i) {
        OPENVINO_ASSERT(draft_rows[i].prob.size() == vocab_size,
                        "Speculative sampling: draft and target rows differ in size");
    }
    return verify_core(
        candidates,
        draft_rows,
        [&target_rows](size_t i) -> const ProbRow& {
            return target_rows[i];
        },
        accept_u,
        final_u,
        vocab_size);
}

VerifyResult verify_candidates(const std::vector<int64_t>& candidates,
                               const std::vector<ProbRow>& draft_rows,
                               const std::vector<ProbRow>& target_rows,
                               std::mt19937& rng) {
    std::vector<double> accept_u(candidates.size());
    for (double& u : accept_u) {
        u = uniform01(rng);
    }
    const double final_u = uniform01(rng);
    return verify_with_uniforms(candidates, draft_rows, target_rows, accept_u, final_u);
}

CommitPlan plan_commit(size_t num_candidates, size_t num_accepted, size_t stop_index) {
    OPENVINO_ASSERT(num_accepted <= num_candidates,
                    "Speculative sampling: accepted ",
                    num_accepted,
                    " exceeds candidates ",
                    num_candidates);
    OPENVINO_ASSERT(stop_index == kNoStopToken || stop_index <= num_accepted,
                    "Speculative sampling: stop index ",
                    stop_index,
                    " is outside the validated block");
    const size_t effective_accepted = stop_index == kNoStopToken ? num_accepted : stop_index;
    const size_t discarded = num_candidates - effective_accepted;
    CommitPlan plan;
    plan.output_length = effective_accepted + 1;
    plan.main_trim = discarded;
    plan.draft_trim = discarded > 0 ? discarded - 1 : 0;
    plan.drop_draft_prefix = discarded > 0;
    return plan;
}

std::vector<int64_t> draft_candidates(size_t num_candidates,
                                      int64_t first_token,
                                      const DraftStepFn& step,
                                      std::vector<ProbRow>& draft_rows,
                                      std::mt19937& rng) {
    OPENVINO_ASSERT(num_candidates <= draft_rows.size(),
                    "Speculative sampling: ",
                    num_candidates,
                    " candidates need as many draft rows, got ",
                    draft_rows.size());
    std::vector<int64_t> candidates;
    candidates.reserve(num_candidates);
    int64_t token = first_token;
    for (size_t i = 0; i < num_candidates; ++i) {
        step(token, i, draft_rows[i]);
        token = static_cast<int64_t>(sample_from(draft_rows[i], uniform01(rng)));
        candidates.push_back(token);
    }
    return candidates;
}

VerifyResult verify_block(const float* main_logits,
                          size_t seq_len,
                          size_t vocab_size,
                          const std::vector<int64_t>& candidates,
                          const std::vector<ProbRow>& draft_rows,
                          std::vector<ProbRow>& target_rows,
                          const SamplingParams& params,
                          Scratch& scratch,
                          std::mt19937& rng) {
    const size_t rows_needed = candidates.size() + 1;
    OPENVINO_ASSERT(seq_len >= rows_needed && target_rows.size() >= rows_needed,
                    "Speculative sampling: ",
                    rows_needed,
                    " main rows are needed, logits hold ",
                    seq_len,
                    " and the buffer holds ",
                    target_rows.size());
    OPENVINO_ASSERT(vocab_size > 0, "Speculative sampling: empty target row");
    for (size_t i = 0; i < candidates.size(); ++i) {
        OPENVINO_ASSERT(draft_rows.size() > i && draft_rows[i].prob.size() == vocab_size,
                        "Speculative sampling: draft and target rows differ in size");
    }
    std::vector<double> accept_u(candidates.size());
    for (double& u : accept_u) {
        u = uniform01(rng);
    }
    const double final_u = uniform01(rng);
    size_t computed = 0;
    return verify_core(
        candidates,
        draft_rows,
        [&](size_t i) -> const ProbRow& {
            for (; computed <= i; ++computed) {
                const float* row = main_logits + (seq_len - rows_needed + computed) * vocab_size;
                compute_probs(row, vocab_size, vocab_size, params, target_rows[computed], scratch);
            }
            return target_rows[i];
        },
        accept_u,
        final_u,
        vocab_size);
}

}  // namespace speculative_sampling
}  // namespace genai
}  // namespace ov
