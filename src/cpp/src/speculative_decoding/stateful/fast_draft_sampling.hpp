// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include "openvino/genai/generation_config.hpp"

namespace ov {
namespace genai {
namespace speculative_sampling {

/**
 * @brief Sampling knobs shared by the draft and the main model.
 *
 * Both models MUST be processed with the same instance: the accept/resample rule of
 * speculative sampling is exact only when the draft and target distributions are built
 * by the same filter chain (temperature, top_k, top_p, renormalisation).
 */
struct SamplingParams {
    double temperature = 1.0;
    size_t top_k = std::numeric_limits<size_t>::max();
    double top_p = 1.0;
};

/**
 * @brief Extracts the sampling knobs from a user generation config.
 *
 * @throws ov::Exception if the config requests an option that speculative sampling does not
 *         support (non-positive temperature, top_p outside (0, 1], penalties, min_p, beam search,
 *         num_return_sequences != 1, logprobs).
 */
SamplingParams make_sampling_params(const GenerationConfig& config);

/**
 * @brief Dense probability row over the vocabulary, in double precision.
 *
 * Entries outside the support are exactly 0. The row sums to 1 up to rounding of the final
 * normalisation.
 */
struct ProbRow {
    std::vector<double> prob;
};

/**
 * @brief Reusable scratch memory so that steady-state calls do not allocate.
 */
struct Scratch {
    std::vector<uint32_t> order;
    std::vector<double> scaled;
};

/**
 * @brief Draws a uniform number in [0, 1) on a 53-bit grid from two raw mt19937 outputs.
 *
 * Uses the standardised mt19937 stream only (no std::uniform_real_distribution), so results
 * are identical across standard libraries.
 */
double uniform01(std::mt19937& rng);

/**
 * @brief Builds the filtered, renormalised distribution of one logits row.
 *
 * Order: temperature scaling, top_k, softmax, top_p (smallest descending-probability prefix whose
 * cumulative mass exceeds top_p, the crossing token included), renormalisation. Ties are broken
 * by the lower token index.
 *
 * @param logits Pointer to vocab_size float32 logits. Not modified.
 * @param vocab_size Number of logits in the row; must be greater than 0.
 * @param valid_vocab_size Tokens with id >= valid_vocab_size are masked out (padded logits).
 * @param params Sampling knobs; must be shared between draft and main rows.
 * @param out Receives the distribution (resized to vocab_size).
 * @param scratch Reusable temporary buffers.
 * @throws ov::Exception on NaN or +inf logits, or if no finite logit exists in the valid range.
 */
void compute_probs(const float* logits,
                   size_t vocab_size,
                   size_t valid_vocab_size,
                   const SamplingParams& params,
                   ProbRow& out,
                   Scratch& scratch);

/**
 * @brief Inverse-CDF draw from a distribution using a precomputed uniform number.
 *
 * Accumulates in double and clamps to the last index with positive mass, so a token with zero
 * probability can never be returned.
 *
 * @throws ov::Exception if the row has no positive mass.
 */
size_t sample_from(const ProbRow& row, double u);

/**
 * @brief Outcome of verifying one draft block.
 */
struct VerifyResult {
    size_t num_accepted = 0;
    int64_t final_token = -1;
    bool all_accepted = false;
};

/**
 * @brief Leviathan et al. (2023) accept/resample verification with caller-supplied uniforms.
 *
 * Candidate i is accepted iff accept_u[i] < min(1, q_i(x_i) / p_i(x_i)) (strict; a candidate with
 * p_i(x_i) == 0 is rejected). At the first rejection the returned token is drawn from
 * normalise(max(0, q_i - p_i)) using final_u; if no mass is left in the residual the token is drawn
 * from q_i instead. If every candidate is accepted the returned token is a bonus draw from q_K using
 * final_u. The emitted sequence is distributed exactly as autoregressive sampling from the target.
 *
 * @param candidates The K drafted tokens.
 * @param draft_rows At least K rows; the first K are p_0..p_{K-1}, p_i being the distribution candidate i
 *        was drawn from. Extra rows are ignored so callers can reuse a larger buffer.
 * @param target_rows At least K + 1 rows; the first K + 1 are q_0..q_K from the main model, built by the
 *        same filter chain. Extra rows are ignored.
 * @param accept_u K uniform numbers in [0, 1), one per candidate.
 * @param final_u One uniform number in [0, 1) for the residual or bonus draw.
 * @throws ov::Exception on inconsistent sizes, out-of-range candidate ids or u outside [0, 1).
 */
VerifyResult verify_with_uniforms(const std::vector<int64_t>& candidates,
                                  const std::vector<ProbRow>& draft_rows,
                                  const std::vector<ProbRow>& target_rows,
                                  const std::vector<double>& accept_u,
                                  double final_u);

/**
 * @brief Same as verify_with_uniforms, drawing K + 1 uniforms from rng in a fixed order.
 *
 * The generator is consumed identically on every call (accept_u[0..K-1] then final_u), so results
 * are reproducible per seed.
 */
VerifyResult verify_candidates(const std::vector<int64_t>& candidates,
                               const std::vector<ProbRow>& draft_rows,
                               const std::vector<ProbRow>& target_rows,
                               std::mt19937& rng);

constexpr size_t kNoStopToken = std::numeric_limits<size_t>::max();

/**
 * @brief How one verified block is committed to the output and to both KV caches.
 */
struct CommitPlan {
    size_t output_length = 0;
    size_t main_trim = 0;
    size_t draft_trim = 0;
    bool drop_draft_prefix = false;
};

/**
 * @brief Plans the commit of a verified block, truncating after an EOS or stop token.
 *
 * The validated block is the first num_accepted candidates followed by one more token, so it has
 * num_accepted + 1 entries. A stop token at index stop_index <= num_accepted ends the output there and
 * counts as if only stop_index candidates were accepted: the candidates after it are trimmed from both
 * caches exactly like rejected ones. The last output token is never stored in either cache.
 *
 * @param num_candidates Number K of drafted candidates.
 * @param num_accepted Accepted candidates, at most K.
 * @param stop_index Index of the first stop token in the validated block, or kNoStopToken.
 * @throws ov::Exception if num_accepted > num_candidates or stop_index > num_accepted.
 */
CommitPlan plan_commit(size_t num_candidates, size_t num_accepted, size_t stop_index);

/**
 * @brief Callback that runs one draft step and fills the draft row for the next token.
 *
 * @param token Token fed to the draft model for this step.
 * @param index Zero-based position of the candidate being drafted.
 * @param row Receives the filtered distribution the candidate is drawn from.
 */
using DraftStepFn = std::function<void(int64_t token, size_t index, ProbRow& row)>;

/**
 * @brief Autoregressively drafts num_candidates tokens, each drawn from the row stored for it.
 *
 * The token is sampled from the same row that is kept in draft_rows[i], so the p used by the
 * accept test is the p the candidate came from. One uniform number per candidate is consumed from rng.
 *
 * @param num_candidates Number K of candidates to draw; must not exceed draft_rows.size().
 * @param first_token Token that precedes the first candidate.
 * @param step Draft step callback.
 * @param draft_rows Receives p_0..p_{K-1}.
 * @param rng Generator shared with verification.
 * @return The K drafted tokens.
 * @throws ov::Exception if draft_rows is too small.
 */
std::vector<int64_t> draft_candidates(size_t num_candidates,
                                      int64_t first_token,
                                      const DraftStepFn& step,
                                      std::vector<ProbRow>& draft_rows,
                                      std::mt19937& rng);

/**
 * @brief Builds the target rows from the last rows of a logits buffer, lazily, and verifies the block.
 *
 * Row q_i is taken from logits row (seq_len - (K + 1) + i), so the buffer may hold more leading rows
 * than K + 1 (for example padded NPU output). All rows go through the same filter chain as the draft.
 * The K + 1 uniforms are drawn from rng up front, in the order of verify_candidates, so the generator is
 * consumed exactly as before. Row q_i is computed only when verification reaches position i: row 0 always,
 * row i after candidates 0..i-1 were accepted, row K only if all K were accepted. After the call
 * target_rows[0..num_accepted] (target_rows[0..K] if all_accepted) hold the rows that were used; later
 * entries are unspecified (stale from earlier calls). Consequently an invalid logits row (NaN, +inf) after
 * the first rejection no longer throws, and when a row does throw the uniforms have already been drawn.
 *
 * @param main_logits Pointer to seq_len x vocab_size float32 logits of the main model. Not modified.
 * @param seq_len Number of logits rows in the buffer; must be at least K + 1.
 * @param vocab_size Row width.
 * @param candidates The K drafted tokens.
 * @param draft_rows Rows p_0..p_{K-1} from draft_candidates.
 * @param target_rows Scratch rows, at least K + 1 entries; computed rows are overwritten (see above).
 * @param params Sampling knobs shared with the draft.
 * @param scratch Reusable buffers.
 * @param rng Generator shared with drafting.
 * @throws ov::Exception on inconsistent sizes.
 */
VerifyResult verify_block(const float* main_logits,
                          size_t seq_len,
                          size_t vocab_size,
                          const std::vector<int64_t>& candidates,
                          const std::vector<ProbRow>& draft_rows,
                          std::vector<ProbRow>& target_rows,
                          const SamplingParams& params,
                          Scratch& scratch,
                          std::mt19937& rng);

}  // namespace speculative_sampling
}  // namespace genai
}  // namespace ov
