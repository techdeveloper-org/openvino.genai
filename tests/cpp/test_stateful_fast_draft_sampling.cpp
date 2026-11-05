// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/genai/generation_config.hpp"
#include "speculative_decoding/stateful/fast_draft_sampling.hpp"

namespace {

using namespace ov::genai::speculative_sampling;

constexpr double kExactTol = 1e-12;
constexpr double kInf = std::numeric_limits<double>::infinity();

ProbRow make_row(std::vector<double> prob) {
    ProbRow row;
    row.prob = std::move(prob);
    return row;
}

std::vector<double> reference_softmax(const std::vector<double>& logits, double temperature) {
    double max_logit = -kInf;
    for (double v : logits) {
        max_logit = std::max(max_logit, v / temperature);
    }
    std::vector<double> out(logits.size());
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        out[i] = std::exp(logits[i] / temperature - max_logit);
        sum += out[i];
    }
    for (double& v : out) {
        v /= sum;
    }
    return out;
}

std::vector<float> to_float(const std::vector<double>& v) {
    std::vector<float> out;
    out.reserve(v.size());
    for (double value : v) {
        out.push_back(static_cast<float>(value));
    }
    return out;
}

std::vector<double> to_double_vector(const std::vector<float>& v) {
    return std::vector<double>(v.begin(), v.end());
}

ProbRow probs_of(const std::vector<double>& logits, const SamplingParams& params = SamplingParams{}) {
    std::vector<float> f = to_float(logits);
    ProbRow row;
    Scratch scratch;
    compute_probs(f.data(), f.size(), f.size(), params, row, scratch);
    return row;
}

double sum_of(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) {
        s += x;
    }
    return s;
}

std::vector<double> random_simplex(std::mt19937& rng, size_t size, double floor_mass = 0.05) {
    std::vector<double> v(size);
    for (double& x : v) {
        x = floor_mass + uniform01(rng);
    }
    const double s = sum_of(v);
    for (double& x : v) {
        x /= s;
    }
    return v;
}

// Pearson chi-square against exact probabilities. Bins with expected count < 20 are pooled; an
// observation in a zero-probability bin is an immediate failure (returned as infinity).
double chi_square(const std::vector<size_t>& counts, const std::vector<double>& expected_prob, size_t n) {
    double stat = 0.0;
    double pooled_obs = 0.0;
    double pooled_exp = 0.0;
    for (size_t i = 0; i < counts.size(); ++i) {
        const double expected = expected_prob[i] * static_cast<double>(n);
        if (expected_prob[i] == 0.0) {
            if (counts[i] != 0) {
                return kInf;
            }
            continue;
        }
        if (expected < 20.0) {
            pooled_obs += static_cast<double>(counts[i]);
            pooled_exp += expected;
            continue;
        }
        const double d = static_cast<double>(counts[i]) - expected;
        stat += d * d / expected;
    }
    if (pooled_exp > 0.0) {
        const double d = pooled_obs - pooled_exp;
        stat += d * d / pooled_exp;
    }
    return stat;
}

ov::genai::GenerationConfig sampling_config() {
    ov::genai::GenerationConfig config;
    config.do_sample = true;
    config.temperature = 0.7f;
    config.top_p = 0.9f;
    config.top_k = 40;
    config.max_new_tokens = 16;
    config.num_assistant_tokens = 5;
    return config;
}

}  // namespace

TEST(StatefulSpeculativeSamplingUniform, Uniform01_AllDraws_InHalfOpenUnitInterval) {
    std::mt19937 rng(1234);
    double sum = 0.0;
    const size_t n = 200000;
    for (size_t i = 0; i < n; ++i) {
        const double u = uniform01(rng);
        ASSERT_GE(u, 0.0);
        ASSERT_LT(u, 1.0);
        sum += u;
    }
    const double mean = sum / n;
    EXPECT_NEAR(mean, 0.5, 5.0 * std::sqrt(1.0 / 12.0 / n));
}

TEST(StatefulSpeculativeSamplingUniform, Uniform01_SameSeed_SameStreamAndConsumesTwoOutputs) {
    std::mt19937 a(7);
    std::mt19937 b(7);
    std::mt19937 c(7);
    EXPECT_EQ(uniform01(a), uniform01(b));
    static_cast<void>(c());
    static_cast<void>(c());
    EXPECT_EQ(a(), c());
}

TEST(StatefulSpeculativeSamplingUniform, Uniform01_KnownRawOutputs_MatchesManualConversion) {
    std::mt19937 rng(42);
    std::mt19937 ref(42);
    const uint32_t hi = ref() >> 5;
    const uint32_t lo = ref() >> 6;
    const double expected =
        (static_cast<double>(hi) * 67108864.0 + static_cast<double>(lo)) * (1.0 / 9007199254740992.0);
    EXPECT_EQ(uniform01(rng), expected);
}

TEST(StatefulSpeculativeSamplingParams, MakeSamplingParams_ValidConfig_CopiesKnobs) {
    const auto params = make_sampling_params(sampling_config());
    EXPECT_NEAR(params.temperature, 0.7, 1e-6);
    EXPECT_NEAR(params.top_p, 0.9, 1e-6);
    EXPECT_EQ(params.top_k, 40u);
}

TEST(StatefulSpeculativeSamplingParams, MakeSamplingParams_DefaultTopKAndTopP_AreDisabled) {
    auto config = sampling_config();
    config.top_k = std::numeric_limits<size_t>::max();
    config.top_p = 1.0f;
    const auto params = make_sampling_params(config);
    EXPECT_EQ(params.top_k, std::numeric_limits<size_t>::max());
    EXPECT_EQ(params.top_p, 1.0);
}

TEST(StatefulSpeculativeSamplingParams, MakeSamplingParams_UnsupportedOptions_ThrowWithOptionName) {
    struct Case {
        std::string name;
        std::function<void(ov::genai::GenerationConfig&)> mutate;
    };
    const std::vector<Case> cases = {
        {"repetition_penalty",
         [](auto& c) {
             c.repetition_penalty = 1.2f;
         }},
        {"presence_penalty",
         [](auto& c) {
             c.presence_penalty = 0.5f;
         }},
        {"frequency_penalty",
         [](auto& c) {
             c.frequency_penalty = 0.5f;
         }},
        {"min_p",
         [](auto& c) {
             c.min_p = 0.1f;
         }},
        {"logprobs",
         [](auto& c) {
             c.logprobs = 1;
         }},
        {"min_new_tokens",
         [](auto& c) {
             c.min_new_tokens = 2;
         }},
        {"num_beams",
         [](auto& c) {
             c.num_beams = 2;
         }},
        {"num_return_sequences",
         [](auto& c) {
             c.num_return_sequences = 2;
         }},
        {"temperature",
         [](auto& c) {
             c.temperature = 0.0f;
         }},
        {"temperature",
         [](auto& c) {
             c.temperature = -1.0f;
         }},
        {"top_p",
         [](auto& c) {
             c.top_p = 0.0f;
         }},
        {"top_p",
         [](auto& c) {
             c.top_p = 1.5f;
         }},
    };
    for (const auto& tc : cases) {
        auto config = sampling_config();
        tc.mutate(config);
        try {
            static_cast<void>(make_sampling_params(config));
            FAIL() << "expected exception for " << tc.name;
        } catch (const ov::Exception& e) {
            EXPECT_NE(std::string(e.what()).find(tc.name), std::string::npos) << e.what();
        }
    }
}

TEST(StatefulSpeculativeSamplingProbs, ComputeProbs_PlainSoftmax_MatchesReference) {
    const std::vector<double> logits = {1.0, 2.0, 3.0, 0.5, -1.0};
    const auto row = probs_of(logits);
    const auto ref = reference_softmax(to_double_vector(to_float(logits)), 1.0);
    ASSERT_EQ(row.prob.size(), logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        EXPECT_NEAR(row.prob[i], ref[i], kExactTol);
    }
    EXPECT_NEAR(sum_of(row.prob), 1.0, kExactTol);
}

TEST(StatefulSpeculativeSamplingProbs, ComputeProbs_Temperature_ScalesLogits) {
    const std::vector<double> logits = {1.0, 2.0, 3.0, 0.5};
    SamplingParams params;
    params.temperature = 0.5;
    const auto row = probs_of(logits, params);
    const auto ref = reference_softmax(to_double_vector(to_float(logits)), 0.5);
    for (size_t i = 0; i < logits.size(); ++i) {
        EXPECT_NEAR(row.prob[i], ref[i], kExactTol);
    }
}

TEST(StatefulSpeculativeSamplingProbs, ComputeProbs_TopK_KeepsLargestAndRenormalises) {
    const std::vector<double> logits = {0.0, 3.0, 1.0, 2.0, -2.0};
    SamplingParams params;
    params.top_k = 2;
    const auto row = probs_of(logits, params);
    const double e3 = std::exp(3.0);
    const double e2 = std::exp(2.0);
    EXPECT_EQ(row.prob[0], 0.0);
    EXPECT_EQ(row.prob[2], 0.0);
    EXPECT_EQ(row.prob[4], 0.0);
    EXPECT_NEAR(row.prob[1], e3 / (e3 + e2), kExactTol);
    EXPECT_NEAR(row.prob[3], e2 / (e3 + e2), kExactTol);
}

TEST(StatefulSpeculativeSamplingProbs, ComputeProbs_TopKAtLeastVocab_IsNoOp) {
    const std::vector<double> logits = {0.0, 1.0, 2.0};
    SamplingParams params;
    params.top_k = 3;
    const auto row = probs_of(logits, params);
    const auto plain = probs_of(logits);
    EXPECT_EQ(row.prob, plain.prob);
}

TEST(StatefulSpeculativeSamplingProbs, ComputeProbs_TopKOne_IsOneHotLowestIndexOnTie) {
    SamplingParams params;
    params.top_k = 1;
    const auto row = probs_of({1.0, 5.0, 5.0, 2.0}, params);
    EXPECT_EQ(row.prob, (std::vector<double>{0.0, 1.0, 0.0, 0.0}));
}

TEST(StatefulSpeculativeSamplingProbs, ComputeProbs_TopP_CrossingTokenIncludedAndRenormalised) {
    const std::vector<double> logits = {std::log(0.5), std::log(0.3), std::log(0.2)};
    SamplingParams params;
    params.top_p = 0.55;
    auto row = probs_of(logits, params);
    EXPECT_NEAR(row.prob[0], 0.5 / 0.8, 1e-6);
    EXPECT_NEAR(row.prob[1], 0.3 / 0.8, 1e-6);
    EXPECT_EQ(row.prob[2], 0.0);

    params.top_p = 0.4;
    row = probs_of(logits, params);
    EXPECT_EQ(row.prob, (std::vector<double>{1.0, 0.0, 0.0}));
}

TEST(StatefulSpeculativeSamplingProbs, ComputeProbs_TopPOne_KeepsEverything) {
    const auto row = probs_of({0.1, 0.2, 0.3});
    for (double p : row.prob) {
        EXPECT_GT(p, 0.0);
    }
}

size_t count_nonzero(const std::vector<double>& v) {
    return static_cast<size_t>(std::count_if(v.begin(), v.end(), [](double x) {
        return x > 0.0;
    }));
}

TEST(StatefulSpeculativeSamplingProbs, ComputeProbs_TopPCumulativeExactlyAtThreshold_KeepsNextToken_FullSortPath) {
    SamplingParams params;
    params.top_p = 0.5;
    const auto row = probs_of(std::vector<double>(8, 0.0), params);
    EXPECT_EQ(count_nonzero(row.prob), 5u);
    for (size_t i = 0; i < row.prob.size(); ++i) {
        if (i < 5) {
            EXPECT_NEAR(row.prob[i], 1.0 / 5.0, kExactTol) << "index " << i;
        } else {
            EXPECT_EQ(row.prob[i], 0.0) << "index " << i;
        }
    }
}

TEST(StatefulSpeculativeSamplingProbs, ComputeProbs_TopPCumulativeExactlyAtThreshold_KeepsNextToken_PartialSortPath) {
    SamplingParams params;
    params.top_p = 0.5;
    const auto row = probs_of(std::vector<double>(32, 0.0), params);
    EXPECT_EQ(count_nonzero(row.prob), 17u);
    for (size_t i = 0; i < row.prob.size(); ++i) {
        EXPECT_EQ(row.prob[i] > 0.0, i < 17) << "index " << i;
    }
}

TEST(StatefulSpeculativeSamplingProbs, ComputeProbs_TopPInsidePartialSortWindow_StopsAtCrossingToken) {
    SamplingParams params;
    params.top_p = 0.3;
    const auto row = probs_of(std::vector<double>(64, 0.0), params);
    EXPECT_EQ(count_nonzero(row.prob), 20u);
}

TEST(StatefulSpeculativeSamplingProbs, ComputeProbs_TopPPartialAndFullSortPaths_AgreeOnRandomLogits) {
    std::mt19937 rng(17);
    for (size_t trial = 0; trial < 20; ++trial) {
        std::vector<float> logits(3000);
        for (float& v : logits) {
            v = static_cast<float>(uniform01(rng) * 8.0 - 4.0);
        }
        SamplingParams params;
        params.top_p = 0.2 + 0.04 * static_cast<double>(trial);
        ProbRow row;
        Scratch scratch;
        compute_probs(logits.data(), logits.size(), logits.size(), params, row, scratch);

        std::vector<size_t> idx(logits.size());
        for (size_t i = 0; i < idx.size(); ++i) {
            idx[i] = i;
        }
        std::vector<double> full(logits.begin(), logits.end());
        const auto soft = reference_softmax(full, 1.0);
        std::sort(idx.begin(), idx.end(), [&soft](size_t a, size_t b) {
            return soft[a] > soft[b] || (soft[a] == soft[b] && a < b);
        });
        double cumulative = 0.0;
        size_t keep = 0;
        for (size_t i = 0; i < idx.size(); ++i) {
            cumulative += soft[idx[i]];
            keep = i + 1;
            if (cumulative > params.top_p) {
                break;
            }
        }
        EXPECT_EQ(count_nonzero(row.prob), keep) << "trial " << trial;
    }
}

TEST(StatefulSpeculativeSamplingProbs, ComputeProbs_TopKThenTopP_ComposeLikeConfigOrder) {
    const std::vector<double> logits = {std::log(0.4), std::log(0.3), std::log(0.2), std::log(0.1)};
    SamplingParams params;
    params.top_k = 3;
    params.top_p = 0.6;
    const auto row = probs_of(logits, params);
    const double kept = 0.4 + 0.3;
    EXPECT_NEAR(row.prob[0], 0.4 / kept, 1e-6);
    EXPECT_NEAR(row.prob[1], 0.3 / kept, 1e-6);
    EXPECT_EQ(row.prob[2], 0.0);
    EXPECT_EQ(row.prob[3], 0.0);
}

TEST(StatefulSpeculativeSamplingProbs, ComputeProbs_PaddedVocab_MasksIdsAboveValidSize) {
    const std::vector<float> logits = {1.0f, 2.0f, 100.0f, 100.0f};
    ProbRow row;
    Scratch scratch;
    compute_probs(logits.data(), logits.size(), 2, SamplingParams{}, row, scratch);
    ASSERT_EQ(row.prob.size(), 4u);
    EXPECT_EQ(row.prob[2], 0.0);
    EXPECT_EQ(row.prob[3], 0.0);
    EXPECT_NEAR(row.prob[0] + row.prob[1], 1.0, kExactTol);
}

TEST(StatefulSpeculativeSamplingProbs, ComputeProbs_NegativeInfinityLogit_GetsZeroMass) {
    const auto row = probs_of({0.0, -kInf, 1.0});
    EXPECT_EQ(row.prob[1], 0.0);
    EXPECT_NEAR(row.prob[0] + row.prob[2], 1.0, kExactTol);
}

TEST(StatefulSpeculativeSamplingProbs, ComputeProbs_ExtremeLogits_DoNotOverflow) {
    const auto row = probs_of({1000.0, 999.0, -1000.0});
    EXPECT_NEAR(sum_of(row.prob), 1.0, kExactTol);
    EXPECT_GT(row.prob[0], row.prob[1]);
    EXPECT_NEAR(row.prob[2], 0.0, 1e-300);
}

TEST(StatefulSpeculativeSamplingProbs, ComputeProbs_SingleToken_IsOne) {
    EXPECT_EQ(probs_of({-3.0}).prob, (std::vector<double>{1.0}));
}

TEST(StatefulSpeculativeSamplingProbs, ComputeProbs_InvalidInput_Throws) {
    ProbRow row;
    Scratch scratch;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const std::vector<float> with_nan = {0.0f, nan};
    const std::vector<float> with_pos_inf = {0.0f, inf};
    const std::vector<float> all_neg_inf = {-inf, -inf};
    const std::vector<float> ok = {0.0f, 1.0f};
    EXPECT_THROW(compute_probs(with_nan.data(), 2, 2, SamplingParams{}, row, scratch), ov::Exception);
    EXPECT_THROW(compute_probs(with_pos_inf.data(), 2, 2, SamplingParams{}, row, scratch), ov::Exception);
    EXPECT_THROW(compute_probs(all_neg_inf.data(), 2, 2, SamplingParams{}, row, scratch), ov::Exception);
    EXPECT_THROW(compute_probs(ok.data(), 0, 0, SamplingParams{}, row, scratch), ov::Exception);
    EXPECT_THROW(compute_probs(ok.data(), 2, 0, SamplingParams{}, row, scratch), ov::Exception);
    EXPECT_THROW(compute_probs(ok.data(), 2, 3, SamplingParams{}, row, scratch), ov::Exception);
    EXPECT_THROW(compute_probs(nullptr, 2, 2, SamplingParams{}, row, scratch), ov::Exception);
}

TEST(StatefulSpeculativeSamplingProbs, ComputeProbs_ReusedScratchAndRow_GiveIdenticalResults) {
    const std::vector<float> a = {0.1f, 0.5f, 0.2f, 0.9f};
    const std::vector<float> b = {3.0f, -1.0f, 0.0f, 0.4f};
    SamplingParams params;
    params.top_k = 3;
    params.top_p = 0.8;
    ProbRow reused;
    Scratch scratch;
    compute_probs(a.data(), 4, 4, params, reused, scratch);
    compute_probs(b.data(), 4, 4, params, reused, scratch);
    ProbRow fresh;
    Scratch fresh_scratch;
    compute_probs(b.data(), 4, 4, params, fresh, fresh_scratch);
    EXPECT_EQ(reused.prob, fresh.prob);
}

TEST(StatefulSpeculativeSamplingProbs, ComputeProbs_IdenticalLogitsTwice_BitIdenticalRows) {
    std::mt19937 rng(5);
    std::vector<float> logits(257);
    for (float& v : logits) {
        v = static_cast<float>(uniform01(rng) * 12.0 - 6.0);
    }
    SamplingParams params;
    params.temperature = 0.8;
    params.top_k = 50;
    params.top_p = 0.92;
    ProbRow draft_slot;
    ProbRow target_slot;
    Scratch s1;
    Scratch s2;
    compute_probs(logits.data(), logits.size(), logits.size(), params, draft_slot, s1);
    compute_probs(logits.data(), logits.size(), logits.size(), params, target_slot, s2);
    EXPECT_EQ(draft_slot.prob, target_slot.prob);
}

TEST(StatefulSpeculativeSamplingSample, SampleFrom_ScriptedUniforms_PickCorrectBin) {
    const auto row = make_row({0.25, 0.25, 0.5});
    EXPECT_EQ(sample_from(row, 0.0), 0u);
    EXPECT_EQ(sample_from(row, 0.2499999), 0u);
    EXPECT_EQ(sample_from(row, 0.25), 1u);
    EXPECT_EQ(sample_from(row, 0.4999999), 1u);
    EXPECT_EQ(sample_from(row, 0.5), 2u);
    EXPECT_EQ(sample_from(row, std::nextafter(1.0, 0.0)), 2u);
}

TEST(StatefulSpeculativeSamplingSample, SampleFrom_ZeroMassBins_NeverSelected) {
    const auto row = make_row({0.0, 0.5, 0.0, 0.0, 0.5, 0.0});
    for (double u : {0.0, 1e-18, 0.25, 0.4999999999, 0.5, 0.75, 0.9999999999, std::nextafter(1.0, 0.0)}) {
        const size_t idx = sample_from(row, u);
        EXPECT_TRUE(idx == 1u || idx == 4u) << "u=" << u << " idx=" << idx;
    }
}

TEST(StatefulSpeculativeSamplingSample, SampleFrom_UniformNearOne_PicksLastPositiveMass) {
    const auto row = make_row({0.1, 0.2, 0.7, 0.0, 0.0});
    EXPECT_EQ(sample_from(row, std::nextafter(1.0, 0.0)), 2u);
}

TEST(StatefulSpeculativeSamplingSample, SampleFrom_InvalidInput_Throws) {
    EXPECT_THROW(sample_from(make_row({0.0, 0.0}), 0.5), ov::Exception);
    EXPECT_THROW(sample_from(make_row({}), 0.5), ov::Exception);
    EXPECT_THROW(sample_from(make_row({1.0}), 1.0), ov::Exception);
    EXPECT_THROW(sample_from(make_row({1.0}), -0.1), ov::Exception);
}

TEST(StatefulSpeculativeSamplingVerify, VerifyWithUniforms_RatioAtLeastOne_AlwaysAccepted) {
    const auto p = make_row({0.5, 0.5});
    const auto q = make_row({0.8, 0.2});
    const auto result = verify_with_uniforms({0}, {p}, {q, q}, {std::nextafter(1.0, 0.0)}, 0.0);
    EXPECT_EQ(result.num_accepted, 1u);
    EXPECT_TRUE(result.all_accepted);
}

TEST(StatefulSpeculativeSamplingVerify, VerifyWithUniforms_UniformAtRatio_RejectsStrictly) {
    const auto p = make_row({0.5, 0.5});
    const auto q = make_row({0.8, 0.2});
    const auto below = verify_with_uniforms({1}, {p}, {q, q}, {std::nextafter(0.4, 0.0)}, 0.0);
    EXPECT_EQ(below.num_accepted, 1u);
    const auto at = verify_with_uniforms({1}, {p}, {q, q}, {0.4}, 0.0);
    EXPECT_EQ(at.num_accepted, 0u);
    EXPECT_FALSE(at.all_accepted);
    EXPECT_EQ(at.final_token, 0);
}

TEST(StatefulSpeculativeSamplingVerify, VerifyWithUniforms_TargetZeroMass_RejectedEvenAtUniformZero) {
    const auto p = make_row({0.5, 0.5});
    const auto q = make_row({1.0, 0.0});
    const auto result = verify_with_uniforms({1}, {p}, {q, q}, {0.0}, 0.9);
    EXPECT_EQ(result.num_accepted, 0u);
    EXPECT_EQ(result.final_token, 0);
}

TEST(StatefulSpeculativeSamplingVerify, VerifyWithUniforms_DraftZeroMass_RejectedWithoutDividingByZero) {
    const auto p = make_row({1.0, 0.0});
    const auto q = make_row({0.5, 0.5});
    const auto result = verify_with_uniforms({1}, {p}, {q, q}, {0.0}, 0.0);
    EXPECT_EQ(result.num_accepted, 0u);
    EXPECT_EQ(result.final_token, 1);
}

TEST(StatefulSpeculativeSamplingVerify, VerifyWithUniforms_ResidualOnlyOnTokensWhereTargetExceedsDraft) {
    const auto p = make_row({0.5, 0.3, 0.2});
    const auto q = make_row({0.2, 0.5, 0.3});
    const double tiny = 1e-12;
    const auto first = verify_with_uniforms({0}, {p}, {q, q}, {0.99}, tiny);
    EXPECT_EQ(first.final_token, 1);
    const auto second = verify_with_uniforms({0}, {p}, {q, q}, {0.99}, 0.5);
    EXPECT_EQ(second.final_token, 1);
    const auto third = verify_with_uniforms({0}, {p}, {q, q}, {0.99}, 0.7);
    EXPECT_EQ(third.final_token, 2);
    for (double u : {0.0, 0.1, 0.3, 0.66, 0.67, 0.99}) {
        EXPECT_NE(verify_with_uniforms({0}, {p}, {q, q}, {0.99}, u).final_token, 0);
    }
}

TEST(StatefulSpeculativeSamplingVerify, VerifyWithUniforms_AllAccepted_BonusDrawnFromLastTargetRow) {
    const auto bonus = make_row({0.0, 0.25, 0.75});
    const auto padded_p = make_row({0.5, 0.5, 0.0});
    const auto padded_q = make_row({0.5, 0.5, 0.0});
    const auto low = verify_with_uniforms({0, 1}, {padded_p, padded_p}, {padded_q, padded_q, bonus}, {0.1, 0.1}, 0.1);
    EXPECT_TRUE(low.all_accepted);
    EXPECT_EQ(low.num_accepted, 2u);
    EXPECT_EQ(low.final_token, 1);
    const auto high = verify_with_uniforms({0, 1}, {padded_p, padded_p}, {padded_q, padded_q, bonus}, {0.1, 0.1}, 0.9);
    EXPECT_EQ(high.final_token, 2);
}

TEST(StatefulSpeculativeSamplingVerify, VerifyWithUniforms_FirstRejectionStopsScan) {
    const auto p = make_row({0.5, 0.5});
    const auto q = make_row({0.9, 0.1});
    const auto result = verify_with_uniforms({0, 1, 0}, {p, p, p}, {q, q, q, q}, {0.1, 0.99, 0.0}, 0.0);
    EXPECT_EQ(result.num_accepted, 1u);
    EXPECT_FALSE(result.all_accepted);
}

TEST(StatefulSpeculativeSamplingVerify, VerifyWithUniforms_ZeroResidualMass_FallsBackToTargetRow) {
    const auto p = make_row({0.5, 0.5});
    const auto q = make_row({0.5, 0.4999});
    const auto result = verify_with_uniforms({1}, {p}, {q, q}, {std::nextafter(1.0, 0.0)}, 0.0);
    EXPECT_EQ(result.num_accepted, 0u);
    EXPECT_EQ(result.final_token, 0);
}

TEST(StatefulSpeculativeSamplingVerify, VerifyWithUniforms_NoCandidates_BonusFromFirstTargetRow) {
    const auto q = make_row({0.0, 1.0});
    const auto result = verify_with_uniforms({}, {}, {q}, {}, 0.3);
    EXPECT_EQ(result.num_accepted, 0u);
    EXPECT_TRUE(result.all_accepted);
    EXPECT_EQ(result.final_token, 1);
}

TEST(StatefulSpeculativeSamplingVerify, VerifyWithUniforms_InvalidInput_Throws) {
    const auto p = make_row({0.5, 0.5});
    const auto q = make_row({0.5, 0.5});
    EXPECT_THROW(verify_with_uniforms({0}, {p}, {q}, {0.1}, 0.1), ov::Exception);
    EXPECT_THROW(verify_with_uniforms({0, 0}, {p}, {q, q, q}, {0.1, 0.1}, 0.1), ov::Exception);
    EXPECT_NO_THROW(verify_with_uniforms({0}, {p, p}, {q, q, q}, {0.1}, 0.1));
    EXPECT_THROW(verify_with_uniforms({0}, {p}, {q, q}, {0.1, 0.2}, 0.1), ov::Exception);
    EXPECT_THROW(verify_with_uniforms({2}, {p}, {q, q}, {0.1}, 0.1), ov::Exception);
    EXPECT_THROW(verify_with_uniforms({-1}, {p}, {q, q}, {0.1}, 0.1), ov::Exception);
    EXPECT_THROW(verify_with_uniforms({0}, {p}, {q, q}, {1.0}, 0.1), ov::Exception);
    EXPECT_THROW(verify_with_uniforms({0}, {p}, {q, q}, {-0.1}, 0.1), ov::Exception);
    EXPECT_THROW(verify_with_uniforms({0}, {p}, {q, q}, {0.1}, 1.0), ov::Exception);
    EXPECT_THROW(verify_with_uniforms({0}, {p}, {q, make_row({1.0})}, {0.1}, 0.1), ov::Exception);
}

TEST(StatefulSpeculativeSamplingVerify, VerifyCandidates_SameSeed_SameResultAndFixedConsumption) {
    const auto p = make_row({0.6, 0.4});
    const auto q = make_row({0.3, 0.7});
    std::mt19937 a(99);
    std::mt19937 b(99);
    const auto ra = verify_candidates({0, 1}, {p, p}, {q, q, q}, a);
    const auto rb = verify_candidates({0, 1}, {p, p}, {q, q, q}, b);
    EXPECT_EQ(ra.num_accepted, rb.num_accepted);
    EXPECT_EQ(ra.final_token, rb.final_token);
    std::mt19937 c(99);
    for (int i = 0; i < 3 * 2; ++i) {
        static_cast<void>(c());
    }
    EXPECT_EQ(a(), c());
}

TEST(StatefulSpeculativeSamplingStatistics, VerifyCandidates_SinglePosition_OutputLawEqualsTarget) {
    struct Case {
        size_t vocab;
        size_t samples;
        double critical;
    };
    const std::vector<Case> cases = {{4, 1000000, 37.662}, {8, 1000000, 48.147}, {16, 1000000, 65.061}};
    for (const auto& tc : cases) {
        std::mt19937 setup(static_cast<uint32_t>(100 + tc.vocab));
        const auto p = make_row(random_simplex(setup, tc.vocab));
        const auto q = make_row(random_simplex(setup, tc.vocab));
        std::mt19937 rng(static_cast<uint32_t>(2024 + tc.vocab));
        std::vector<size_t> counts(tc.vocab, 0);
        for (size_t i = 0; i < tc.samples; ++i) {
            const size_t x = sample_from(p, uniform01(rng));
            const auto r = verify_candidates({static_cast<int64_t>(x)}, {p}, {q, q}, rng);
            const size_t emitted = r.num_accepted == 1 ? x : static_cast<size_t>(r.final_token);
            ++counts[emitted];
        }
        EXPECT_LT(chi_square(counts, q.prob, tc.samples), tc.critical) << "vocab=" << tc.vocab;
    }
}

TEST(StatefulSpeculativeSamplingStatistics, VerifyCandidates_ZerosAndDisjointSupport_NeverEmitZeroMassToken) {
    const auto p = make_row({0.5, 0.5, 0.0, 0.0});
    const auto q = make_row({0.0, 0.0, 0.3, 0.7});
    std::mt19937 rng(11);
    std::vector<size_t> counts(4, 0);
    const size_t n = 200000;
    for (size_t i = 0; i < n; ++i) {
        const size_t x = sample_from(p, uniform01(rng));
        const auto r = verify_candidates({static_cast<int64_t>(x)}, {p}, {q, q}, rng);
        ASSERT_EQ(r.num_accepted, 0u);
        ++counts[r.final_token];
    }
    EXPECT_EQ(counts[0] + counts[1], 0u);
    EXPECT_LT(chi_square(counts, q.prob, n), 37.662);
}

TEST(StatefulSpeculativeSamplingStatistics, VerifyCandidates_IdenticalRows_AlwaysAccepts) {
    std::mt19937 setup(3);
    const auto p = make_row(random_simplex(setup, 8));
    std::mt19937 rng(21);
    for (size_t i = 0; i < 100000; ++i) {
        const size_t x = sample_from(p, uniform01(rng));
        const auto r = verify_candidates({static_cast<int64_t>(x)}, {p}, {p, p}, rng);
        ASSERT_EQ(r.num_accepted, 1u);
    }
}

TEST(StatefulSpeculativeSamplingStatistics, VerifyCandidates_AcceptanceRate_MatchesOverlapWithinWilsonInterval) {
    std::mt19937 setup(8);
    const auto p = make_row(random_simplex(setup, 8));
    const auto q = make_row(random_simplex(setup, 8));
    double beta = 0.0;
    for (size_t i = 0; i < 8; ++i) {
        beta += std::min(p.prob[i], q.prob[i]);
    }
    std::mt19937 rng(77);
    const size_t n = 400000;
    size_t accepted = 0;
    for (size_t i = 0; i < n; ++i) {
        const size_t x = sample_from(p, uniform01(rng));
        accepted += verify_candidates({static_cast<int64_t>(x)}, {p}, {q, q}, rng).num_accepted;
    }
    const double z = 5.5;
    const double phat = static_cast<double>(accepted) / n;
    const double denom = 1.0 + z * z / n;
    const double centre = (phat + z * z / (2.0 * n)) / denom;
    const double half = z * std::sqrt(phat * (1.0 - phat) / n + z * z / (4.0 * n * n)) / denom;
    EXPECT_GE(beta, centre - half);
    EXPECT_LE(beta, centre + half);
}

namespace {

constexpr size_t kToyVocab = 4;

std::vector<double> toy_row_values(uint64_t salt, const std::vector<int64_t>& prefix) {
    uint64_t h = 1469598103934665603ull ^ salt;
    for (int64_t t : prefix) {
        h = (h ^ static_cast<uint64_t>(t + 1)) * 1099511628211ull;
    }
    std::mt19937 gen(static_cast<uint32_t>(h ^ (h >> 32)));
    return random_simplex(gen, kToyVocab, 0.1);
}

ProbRow toy_target(const std::vector<int64_t>& prefix) {
    return make_row(toy_row_values(1, prefix));
}

ProbRow toy_draft(const std::vector<int64_t>& prefix) {
    const auto a = toy_row_values(1, prefix);
    const auto b = toy_row_values(2, prefix);
    std::vector<double> mix(kToyVocab);
    for (size_t i = 0; i < kToyVocab; ++i) {
        mix[i] = 0.5 * a[i] + 0.5 * b[i];
    }
    return make_row(mix);
}

}  // namespace

TEST(StatefulSpeculativeSamplingStatistics, VerifyCandidates_ThreeStepStream_JointLawEqualsAutoregressiveTarget) {
    const size_t block = 3;
    const size_t stream_len = 3;
    const size_t n = 300000;
    std::mt19937 rng(31337);
    std::vector<size_t> counts(64, 0);
    for (size_t it = 0; it < n; ++it) {
        std::vector<int64_t> stream;
        while (stream.size() < stream_len) {
            std::vector<int64_t> draft;
            std::vector<ProbRow> p_rows;
            for (size_t i = 0; i < block; ++i) {
                std::vector<int64_t> prefix = stream;
                prefix.insert(prefix.end(), draft.begin(), draft.end());
                p_rows.push_back(toy_draft(prefix));
                draft.push_back(static_cast<int64_t>(sample_from(p_rows.back(), uniform01(rng))));
            }
            std::vector<ProbRow> q_rows;
            for (size_t i = 0; i <= block; ++i) {
                std::vector<int64_t> prefix = stream;
                prefix.insert(prefix.end(), draft.begin(), draft.begin() + i);
                q_rows.push_back(toy_target(prefix));
            }
            const auto r = verify_candidates(draft, p_rows, q_rows, rng);
            stream.insert(stream.end(), draft.begin(), draft.begin() + r.num_accepted);
            stream.push_back(r.final_token);
        }
        ++counts[stream[0] * 16 + stream[1] * 4 + stream[2]];
    }
    std::vector<double> expected(64, 0.0);
    for (int64_t a = 0; a < 4; ++a) {
        for (int64_t b = 0; b < 4; ++b) {
            for (int64_t c = 0; c < 4; ++c) {
                expected[a * 16 + b * 4 + c] =
                    toy_target({}).prob[a] * toy_target({a}).prob[b] * toy_target({a, b}).prob[c];
            }
        }
    }
    EXPECT_LT(chi_square(counts, expected, n), 143.380);
}

namespace {

TEST(StatefulSpeculativeCommitPlan, PlanCommit_AllBlockShapes_KeepsKvLengthsConsistent) {
    for (size_t k = 0; k <= 10; ++k) {
        for (size_t accepted = 0; accepted <= k; ++accepted) {
            std::vector<size_t> stops = {kNoStopToken};
            for (size_t s = 0; s <= accepted; ++s) {
                stops.push_back(s);
            }
            for (size_t stop : stops) {
                const size_t effective = stop == kNoStopToken ? accepted : stop;
                const auto plan = plan_commit(k, accepted, stop);
                const size_t main_before = 100;
                const size_t draft_before = 99;
                const size_t main_after = main_before + k + 1 - plan.main_trim;
                const size_t draft_after = draft_before + k - plan.draft_trim;
                EXPECT_EQ(plan.output_length, effective + 1);
                EXPECT_EQ(main_after, main_before + 1 + effective);
                EXPECT_EQ(draft_after, draft_before + effective + (effective < k ? 1u : 0u));
                EXPECT_EQ(plan.drop_draft_prefix, effective < k);
                EXPECT_LE(plan.draft_trim, plan.main_trim);
            }
        }
    }
}

TEST(StatefulSpeculativeCommitPlan, PlanCommit_NoStopMatchesGreedyTrimRule) {
    for (size_t k = 1; k <= 10; ++k) {
        for (size_t accepted = 0; accepted <= k; ++accepted) {
            const auto plan = plan_commit(k, accepted, kNoStopToken);
            const size_t mismatched = k - accepted;
            EXPECT_EQ(plan.main_trim, mismatched);
            EXPECT_EQ(plan.draft_trim, mismatched > 0 ? mismatched - 1 : 0u);
            EXPECT_EQ(plan.output_length, accepted + 1);
        }
    }
}

TEST(StatefulSpeculativeCommitPlan, PlanCommit_StopOnFinalToken_AddsNoExtraTrim) {
    const auto plan = plan_commit(5, 3, 3);
    EXPECT_EQ(plan.output_length, 4u);
    EXPECT_EQ(plan.main_trim, 2u);
    EXPECT_EQ(plan.draft_trim, 1u);
}

TEST(StatefulSpeculativeCommitPlan, PlanCommit_StopInsideFullyAcceptedBlock_TrimsTail) {
    const auto plan = plan_commit(5, 5, 1);
    EXPECT_EQ(plan.output_length, 2u);
    EXPECT_EQ(plan.main_trim, 4u);
    EXPECT_EQ(plan.draft_trim, 3u);
    EXPECT_TRUE(plan.drop_draft_prefix);
}

TEST(StatefulSpeculativeCommitPlan, PlanCommit_ZeroCandidates_IsNoOp) {
    const auto plan = plan_commit(0, 0, kNoStopToken);
    EXPECT_EQ(plan.output_length, 1u);
    EXPECT_EQ(plan.main_trim, 0u);
    EXPECT_EQ(plan.draft_trim, 0u);
    EXPECT_FALSE(plan.drop_draft_prefix);
}

TEST(StatefulSpeculativeCommitPlan, PlanCommit_InvalidArguments_Throw) {
    EXPECT_THROW(plan_commit(3, 4, kNoStopToken), ov::Exception);
    EXPECT_THROW(plan_commit(3, 2, 3), ov::Exception);
}

}  // namespace

namespace {

constexpr size_t kWireVocab = 8;
constexpr size_t kWireCandidates = 3;
constexpr size_t kWireTokens = 3;
constexpr size_t kWireJunkRows = 2;
constexpr int64_t kWireStartToken = 0;
constexpr size_t kWireTrials = 1000000;
constexpr double kWireAlpha = 1e-6;
constexpr double kWireMinExpected = 5.0;

struct WireModel {
    std::vector<std::vector<float>> draft_logits;
    std::vector<std::vector<float>> target_logits;
    SamplingParams params;
};

WireModel make_wire_model() {
    WireModel model;
    model.params.temperature = 0.8;
    model.params.top_k = 6;
    model.draft_logits.assign(kWireVocab, std::vector<float>(kWireVocab));
    model.target_logits.assign(kWireVocab, std::vector<float>(kWireVocab));
    for (size_t t = 0; t < kWireVocab; ++t) {
        for (size_t v = 0; v < kWireVocab; ++v) {
            const double ft = static_cast<double>(t);
            const double fv = static_cast<double>(v);
            model.target_logits[t][v] =
                static_cast<float>(1.7 * std::sin(1.3 * ft + 2.1 * fv + 0.4) + 0.3 * std::cos(3.7 * fv - ft));
            model.draft_logits[t][v] = static_cast<float>(2.0 * std::sin(2.9 * ft - 1.7 * fv + 1.1) + 0.05 * fv);
        }
    }
    return model;
}

/** Independent reference of the filter chain: temperature, top_k, softmax, renormalise. */
std::vector<double> reference_filtered(const std::vector<float>& logits, double temperature, size_t top_k) {
    std::vector<size_t> order(logits.size());
    for (size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return logits[a] > logits[b];
    });
    const size_t kept = std::min(top_k, logits.size());
    std::vector<double> out(logits.size(), 0.0);
    const double peak = static_cast<double>(logits[order[0]]) / temperature;
    double total = 0.0;
    for (size_t r = 0; r < kept; ++r) {
        out[order[r]] = std::exp(static_cast<double>(logits[order[r]]) / temperature - peak);
        total += out[order[r]];
    }
    for (double& p : out) {
        p /= total;
    }
    return out;
}

/** Exact law of the first kWireTokens tokens: the autoregressive product of the target rows. */
std::vector<double> analytic_wire_law(const WireModel& model, double temperature) {
    std::vector<std::vector<double>> q;
    for (const auto& logits : model.target_logits) {
        q.push_back(reference_filtered(logits, temperature, model.params.top_k));
    }
    std::vector<double> law(kWireVocab * kWireVocab * kWireVocab, 0.0);
    const auto first = q[static_cast<size_t>(kWireStartToken)];
    for (size_t a = 0; a < kWireVocab; ++a) {
        for (size_t b = 0; b < kWireVocab; ++b) {
            for (size_t c = 0; c < kWireVocab; ++c) {
                law[(a * kWireVocab + b) * kWireVocab + c] = first[a] * q[a][b] * q[b][c];
            }
        }
    }
    return law;
}

struct WireStats {
    size_t cycles = 0;
    size_t accepted = 0;
    std::vector<size_t> accepted_histogram = std::vector<size_t>(kWireCandidates + 1, 0);
};

/**
 * Mirrors one generate() call of the stochastic pipeline over scripted logits: first token from the
 * main row, then draft_candidates, a main logits buffer with leading junk rows, verify_block and
 * plan_commit, with the shared generator.
 */
std::vector<int64_t> run_wire_trial(const WireModel& model,
                                    std::mt19937& rng,
                                    std::vector<ProbRow>& draft_rows,
                                    std::vector<ProbRow>& target_rows,
                                    Scratch& scratch,
                                    WireStats& stats) {
    std::vector<int64_t> output;
    ProbRow row;
    compute_probs(model.target_logits[static_cast<size_t>(kWireStartToken)].data(),
                  kWireVocab,
                  kWireVocab,
                  model.params,
                  row,
                  scratch);
    int64_t last = static_cast<int64_t>(sample_from(row, uniform01(rng)));
    output.push_back(last);

    const size_t seq_len = kWireCandidates + 1 + kWireJunkRows;
    std::vector<float> main_logits(seq_len * kWireVocab);
    while (output.size() < kWireTokens) {
        const auto step = [&](int64_t token, size_t, ProbRow& out) {
            compute_probs(model.draft_logits[static_cast<size_t>(token)].data(),
                          kWireVocab,
                          kWireVocab,
                          model.params,
                          out,
                          scratch);
        };
        const auto candidates = draft_candidates(kWireCandidates, last, step, draft_rows, rng);

        std::vector<int64_t> input_for_main = {last};
        input_for_main.insert(input_for_main.end(), candidates.begin(), candidates.end());
        for (size_t r = 0; r < seq_len; ++r) {
            const bool junk = r < kWireJunkRows;
            const auto& source = junk ? model.draft_logits[(r + 3) % kWireVocab]
                                      : model.target_logits[static_cast<size_t>(input_for_main[r - kWireJunkRows])];
            std::copy(source.begin(), source.end(), main_logits.begin() + static_cast<std::ptrdiff_t>(r * kWireVocab));
        }

        const auto verdict = verify_block(main_logits.data(),
                                          seq_len,
                                          kWireVocab,
                                          candidates,
                                          draft_rows,
                                          target_rows,
                                          model.params,
                                          scratch,
                                          rng);
        std::vector<int64_t> validated(candidates.begin(),
                                       candidates.begin() + static_cast<std::ptrdiff_t>(verdict.num_accepted));
        validated.push_back(verdict.final_token);
        validated.resize(plan_commit(kWireCandidates, verdict.num_accepted, kNoStopToken).output_length);
        last = validated.back();
        output.insert(output.end(), validated.begin(), validated.end());
        stats.cycles += 1;
        stats.accepted += verdict.num_accepted;
        stats.accepted_histogram[verdict.num_accepted] += 1;
    }
    output.resize(kWireTokens);
    return output;
}

/** Regularised upper incomplete gamma Q(a, x) (series and continued fraction). */
double regularized_gamma_q(double a, double x) {
    if (x <= 0.0) {
        return 1.0;
    }
    const double log_prefix = -x + a * std::log(x) - std::lgamma(a);
    if (x < a + 1.0) {
        double term = 1.0 / a;
        double sum = term;
        for (int n = 1; n < 10000; ++n) {
            term *= x / (a + n);
            sum += term;
            if (std::fabs(term) < std::fabs(sum) * 1e-16) {
                break;
            }
        }
        return 1.0 - sum * std::exp(log_prefix);
    }
    const double tiny = 1e-300;
    double b = x + 1.0 - a;
    double c = 1.0 / tiny;
    double d = 1.0 / b;
    double h = d;
    for (int n = 1; n < 10000; ++n) {
        const double an = -n * (n - a);
        b += 2.0;
        d = an * d + b;
        d = std::fabs(d) < tiny ? tiny : d;
        c = b + an / c;
        c = std::fabs(c) < tiny ? tiny : c;
        d = 1.0 / d;
        const double delta = d * c;
        h *= delta;
        if (std::fabs(delta - 1.0) < 1e-16) {
            break;
        }
    }
    return std::exp(log_prefix) * h;
}

struct ChiSquare {
    double statistic = 0.0;
    size_t bins = 0;
    double p_value = 1.0;
    size_t impossible_hits = 0;
};

/** One-sample chi-square goodness of fit; bins with expected count below kWireMinExpected are merged. */
ChiSquare goodness_of_fit(const std::vector<size_t>& counts, const std::vector<double>& law, size_t total) {
    ChiSquare result;
    double pooled_expected = 0.0;
    double pooled_observed = 0.0;
    for (size_t i = 0; i < law.size(); ++i) {
        const double expected = law[i] * static_cast<double>(total);
        if (law[i] == 0.0) {
            result.impossible_hits += counts[i];
        } else if (expected < kWireMinExpected) {
            pooled_expected += expected;
            pooled_observed += static_cast<double>(counts[i]);
        } else {
            const double diff = static_cast<double>(counts[i]) - expected;
            result.statistic += diff * diff / expected;
            result.bins += 1;
        }
    }
    if (pooled_expected > 0.0) {
        const double diff = pooled_observed - pooled_expected;
        result.statistic += diff * diff / pooled_expected;
        result.bins += 1;
    }
    const double dof = static_cast<double>(result.bins) - 1.0;
    result.p_value = regularized_gamma_q(dof / 2.0, result.statistic / 2.0);
    return result;
}

}  // namespace

TEST(StatefulSpeculativeWiring, ReferenceLaw_IsNormalisedAndHasImpossibleCells) {
    const auto model = make_wire_model();
    const auto law = analytic_wire_law(model, model.params.temperature);
    double sum = 0.0;
    size_t zero_cells = 0;
    for (double p : law) {
        sum += p;
        zero_cells += p == 0.0 ? 1 : 0;
    }
    EXPECT_NEAR(sum, 1.0, 1e-12);
    EXPECT_GT(zero_cells, 0u) << "top_k=6 of 8 must leave impossible cells so masking is exercised";
}

TEST(StatefulSpeculativeWiring, GoodnessOfFit_KnownSampleFromLaw_IsNotRejected) {
    const auto model = make_wire_model();
    const auto law = analytic_wire_law(model, model.params.temperature);
    std::discrete_distribution<size_t> direct(law.begin(), law.end());
    std::mt19937 rng(777);
    std::vector<size_t> counts(law.size(), 0);
    for (size_t i = 0; i < kWireTrials; ++i) {
        counts[direct(rng)] += 1;
    }
    const auto fit = goodness_of_fit(counts, law, kWireTrials);
    std::cout << "[wiring] control direct sampling: chi2=" << fit.statistic << " bins=" << fit.bins
              << " p=" << fit.p_value << std::endl;
    EXPECT_EQ(fit.impossible_hits, 0u);
    EXPECT_GT(fit.p_value, kWireAlpha);
}

TEST(StatefulSpeculativeWiring, SpeculativePipelineWiring_MatchesAnalyticLaw_AtOneMillionTrials) {
    const auto model = make_wire_model();
    const auto law = analytic_wire_law(model, model.params.temperature);
    std::mt19937 rng(20260101);
    std::vector<ProbRow> draft_rows(2 * kWireCandidates);
    std::vector<ProbRow> target_rows(2 * kWireCandidates + 1);
    Scratch scratch;
    WireStats stats;
    std::vector<size_t> counts(law.size(), 0);
    for (size_t i = 0; i < kWireTrials; ++i) {
        const auto tokens = run_wire_trial(model, rng, draft_rows, target_rows, scratch, stats);
        counts[(static_cast<size_t>(tokens[0]) * kWireVocab + static_cast<size_t>(tokens[1])) * kWireVocab +
               static_cast<size_t>(tokens[2])] += 1;
    }
    const auto fit = goodness_of_fit(counts, law, kWireTrials);
    const double rate = static_cast<double>(stats.accepted) / static_cast<double>(stats.cycles * kWireCandidates);
    std::cout << "[wiring] speculative: chi2=" << fit.statistic << " bins=" << fit.bins << " p=" << fit.p_value
              << " impossible_hits=" << fit.impossible_hits << " cycles=" << stats.cycles << " acceptance_rate=" << rate
              << std::endl;
    for (size_t k = 0; k <= kWireCandidates; ++k) {
        std::cout << "[wiring] accepted==" << k << ": " << stats.accepted_histogram[k] << std::endl;
        EXPECT_GT(stats.accepted_histogram[k], 0u) << "every accepted-count must occur so all branches run";
    }
    EXPECT_EQ(fit.impossible_hits, 0u);
    EXPECT_GT(fit.p_value, kWireAlpha);

    const auto wrong_law = analytic_wire_law(model, 0.85);
    const auto power = goodness_of_fit(counts, wrong_law, kWireTrials);
    std::cout << "[wiring] power control (reference T=0.85): chi2=" << power.statistic << " p=" << power.p_value
              << std::endl;
    EXPECT_LT(power.p_value, kWireAlpha) << "the test must be able to reject a law that is slightly wrong";
}

TEST(StatefulSpeculativeWiring, DraftCandidates_StepReceivesPreviousTokenChain) {
    const auto model = make_wire_model();
    std::vector<ProbRow> rows(kWireCandidates);
    Scratch scratch;
    std::mt19937 rng(5);
    std::vector<int64_t> fed;
    std::vector<size_t> indices;
    const auto step = [&](int64_t token, size_t index, ProbRow& out) {
        fed.push_back(token);
        indices.push_back(index);
        compute_probs(model.draft_logits[static_cast<size_t>(token)].data(),
                      kWireVocab,
                      kWireVocab,
                      model.params,
                      out,
                      scratch);
    };
    const auto candidates = draft_candidates(kWireCandidates, 5, step, rows, rng);
    ASSERT_EQ(candidates.size(), kWireCandidates);
    EXPECT_EQ(fed, (std::vector<int64_t>{5, candidates[0], candidates[1]}));
    EXPECT_EQ(indices, (std::vector<size_t>{0, 1, 2}));
}

TEST(StatefulSpeculativeWiring, DraftCandidates_TooFewRows_Throws) {
    std::vector<ProbRow> rows(1);
    std::mt19937 rng(5);
    EXPECT_THROW(draft_candidates(2, 0, [](int64_t, size_t, ProbRow&) {}, rows, rng), ov::Exception);
}

namespace {

constexpr size_t kOracleNoTopK = std::numeric_limits<size_t>::max();
constexpr double kOracleUnitRoundoff = 0x1p-53;

/**
 * Logits on the grid i / 64 with |i| <= range. A small range makes equal logits frequent, so the lower-index tie rule
 * decides the top_k and top_p boundaries.
 */
std::vector<float> oracle_grid_logits(std::mt19937& gen, size_t count, uint32_t range) {
    std::vector<float> logits(count);
    for (float& x : logits) {
        x = static_cast<float>(static_cast<int>(gen() % (2 * range + 1)) - static_cast<int>(range)) / 64.0f;
    }
    return logits;
}

/**
 * Logits spread over [-amplitude, amplitude) on a 2^-23 grid, so exact ties are rare.
 */
std::vector<float> oracle_spread_logits(std::mt19937& gen, size_t count, float amplitude) {
    std::vector<float> logits(count);
    for (float& x : logits) {
        x = (static_cast<float>(gen() >> 8) / 8388608.0f - 1.0f) * amplitude;
    }
    return logits;
}

/**
 * Uniform number on a 2^-26 grid built from integer arithmetic only, independent of uniform01.
 */
double oracle_unit(std::mt19937& gen) {
    return static_cast<double>(gen() >> 6) * 0x1p-26;
}

/**
 * Candidate set of the documented filter chain: tokens whose scaled logit is above -inf, the top_k largest by
 * (scaled logit descending, index ascending), returned in ascending index order, plus the largest scaled logit.
 */
struct OracleSelection {
    std::vector<double> scaled;
    std::vector<uint32_t> ids;
    double peak = -kInf;
};

OracleSelection oracle_select(const std::vector<float>& logits, size_t valid, double temperature, size_t top_k) {
    OracleSelection selection;
    selection.scaled.resize(valid);
    for (size_t i = 0; i < valid; ++i) {
        selection.scaled[i] = static_cast<double>(logits[i]) / temperature;
        if (selection.scaled[i] > -kInf) {
            selection.ids.push_back(static_cast<uint32_t>(i));
            selection.peak = std::max(selection.peak, selection.scaled[i]);
        }
    }
    const auto& scaled = selection.scaled;
    std::stable_sort(selection.ids.begin(), selection.ids.end(), [&scaled](uint32_t a, uint32_t b) {
        return scaled[a] > scaled[b];
    });
    if (top_k < selection.ids.size()) {
        selection.ids.resize(top_k);
    }
    std::sort(selection.ids.begin(), selection.ids.end());
    return selection;
}

/**
 * Independent double-precision model of compute_probs.
 *
 * It follows the documented order (temperature, top_k, softmax, top_p, renormalisation) with plain full sorts and
 * the normalising sums taken in ascending token-id order, the top_p cumulative and kept sums in (probability
 * descending, index ascending) order.
 *
 * Why the result is expected to be bit-identical to compute_probs and not merely close: every exp argument is the
 * same IEEE-754 expression logit / temperature - max(logit / temperature), std::exp is the same function of the same
 * argument in the same process, additions are performed in the same fixed order (no reassociation without
 * -ffast-math), and the sorts break every tie by the lower index, so the kept sets coincide exactly.
 */
std::vector<double> oracle_probs(const std::vector<float>& logits,
                                 size_t valid,
                                 double temperature,
                                 size_t top_k,
                                 double top_p) {
    const OracleSelection selection = oracle_select(logits, valid, temperature, top_k);
    std::vector<uint32_t> ids = selection.ids;
    std::vector<double> prob(logits.size(), 0.0);
    double sum = 0.0;
    for (uint32_t id : ids) {
        prob[id] = std::exp(selection.scaled[id] - selection.peak);
        sum += prob[id];
    }
    for (uint32_t id : ids) {
        prob[id] /= sum;
    }
    if (top_p < 1.0) {
        std::stable_sort(ids.begin(), ids.end(), [&prob](uint32_t a, uint32_t b) {
            return prob[a] > prob[b];
        });
        size_t keep = ids.size();
        double cumulative = 0.0;
        for (size_t i = 0; i < ids.size(); ++i) {
            cumulative += prob[ids[i]];
            if (cumulative > top_p) {
                keep = i + 1;
                break;
            }
        }
        double kept_sum = 0.0;
        for (size_t i = 0; i < keep; ++i) {
            kept_sum += prob[ids[i]];
        }
        for (size_t i = keep; i < ids.size(); ++i) {
            prob[ids[i]] = 0.0;
        }
        for (size_t i = 0; i < keep; ++i) {
            prob[ids[i]] /= kept_sum;
        }
    }
    return prob;
}

std::string oracle_label(const char* what, size_t vocab, const SamplingParams& params) {
    return std::string(what) + " vocab=" + std::to_string(vocab) +
           " top_k=" + (params.top_k == kOracleNoTopK ? std::string("max") : std::to_string(params.top_k)) +
           " log2(T)=" + std::to_string(std::log2(params.temperature)) + " top_p=" + std::to_string(params.top_p);
}

SamplingParams oracle_params(double temperature, size_t top_k, double top_p) {
    SamplingParams params;
    params.temperature = temperature;
    params.top_k = top_k;
    params.top_p = top_p;
    return params;
}

void oracle_expect_rows_equal(const std::vector<double>& actual,
                              const std::vector<double>& expected,
                              const std::string& label) {
    ASSERT_EQ(actual.size(), expected.size()) << label;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!(actual[i] == expected[i])) {
            EXPECT_EQ(actual[i], expected[i]) << label << ", first mismatch at index " << i;
            return;
        }
    }
}

void oracle_expect_probs(const std::vector<float>& logits,
                         size_t valid,
                         const SamplingParams& params,
                         ProbRow& row,
                         Scratch& scratch,
                         const std::string& label) {
    compute_probs(logits.data(), logits.size(), valid, params, row, scratch);
    oracle_expect_rows_equal(row.prob,
                             oracle_probs(logits, valid, params.temperature, params.top_k, params.top_p),
                             label);
}

std::vector<size_t> oracle_support(const std::vector<double>& prob) {
    std::vector<size_t> ids;
    for (size_t i = 0; i < prob.size(); ++i) {
        if (!(prob[i] == 0.0)) {
            ids.push_back(i);
        }
    }
    return ids;
}

std::vector<size_t> oracle_iota(size_t first, size_t last) {
    std::vector<size_t> ids;
    for (size_t i = first; i < last; ++i) {
        ids.push_back(i);
    }
    return ids;
}

/**
 * Neumaier compensated sum. Its result is within 2 * 2^-53 relative of the exact sum of non-negative terms whatever
 * the order of the terms, which makes it an order-independent oracle for the normalising sum.
 */
double oracle_compensated_sum(const std::vector<double>& terms) {
    double sum = 0.0;
    double carry = 0.0;
    for (double x : terms) {
        const double t = sum + x;
        carry += std::fabs(sum) >= std::fabs(x) ? (sum - t) + x : (x - t) + sum;
        sum = t;
    }
    return sum + carry;
}

}  // namespace

/**
 * Order-independent check of the top_k rows.
 *
 * With u = 2^-53 and n selected terms, plain recursive summation of non-negative terms gives
 * |S_hat - S| <= gamma_(n-1) * S with gamma_m = m * u / (1 - m * u) (Higham, Accuracy and Stability of Numerical
 * Algorithms, Theorem 4.2). The compensated reference is within 2u of S, and the division adds one rounding of u, so
 * every probability satisfies |p_hat - e / S| <= gamma_(n+3) * e / S. The bound does not depend on the summation
 * order, so unlike the bit-exact tests it also holds for a build that reassociates the sum.
 */
TEST(StatefulSpeculativeSamplingOracleProbs, ComputeProbs_TopKRows_MatchCompensatedSumWithinDerivedBound) {
    std::mt19937 gen(4242);
    ProbRow row;
    Scratch scratch;
    for (size_t top_k : {size_t(1), size_t(7), size_t(60), size_t(500), size_t(1999), kOracleNoTopK}) {
        for (double temperature : {0.6, 1.0, 2.5}) {
            const auto logits = oracle_spread_logits(gen, 2000, 6.0f);
            const auto params = oracle_params(temperature, top_k, 1.0);
            compute_probs(logits.data(), logits.size(), logits.size(), params, row, scratch);
            const OracleSelection selection = oracle_select(logits, logits.size(), temperature, top_k);
            std::vector<double> terms;
            for (uint32_t id : selection.ids) {
                terms.push_back(std::exp(selection.scaled[id] - selection.peak));
            }
            const double exact_sum = oracle_compensated_sum(terms);
            const double m = static_cast<double>(selection.ids.size() + 3) * kOracleUnitRoundoff;
            const double bound = m / (1.0 - m);
            ASSERT_EQ(row.prob.size(), logits.size());
            for (size_t j = 0; j < selection.ids.size(); ++j) {
                const double expected = terms[j] / exact_sum;
                EXPECT_LE(std::fabs(row.prob[selection.ids[j]] - expected), bound * expected)
                    << oracle_label("compensated", logits.size(), params) << " id " << selection.ids[j];
            }
            EXPECT_EQ(oracle_support(row.prob).size(), selection.ids.size());
        }
    }
}

TEST(StatefulSpeculativeSamplingOracleProbs, ComputeProbs_FilterChainRows_BitIdenticalToAscendingSumModel) {
    struct Shape {
        size_t vocab;
        std::vector<size_t> top_ks;
    };
    const std::vector<Shape> shapes = {
        {2000, {1, 2, 50, 124, 125, 126, 1000, 1999, 2000, 2001, 100000, kOracleNoTopK}},
        {20000, {1249, 1250, 5000, 19999}},
    };
    std::mt19937 gen(20260401);
    ProbRow row;
    row.prob.assign(37, 7.0);
    Scratch scratch;
    for (const Shape& shape : shapes) {
        for (size_t top_k : shape.top_ks) {
            for (double temperature : {0.5, 1.0, 3.0}) {
                for (double top_p : {1.0, 0.9, 0.35}) {
                    for (bool tied : {false, true}) {
                        const auto logits = tied ? oracle_grid_logits(gen, shape.vocab, 12)
                                                 : oracle_spread_logits(gen, shape.vocab, 8.0f);
                        const auto params = oracle_params(temperature, top_k, top_p);
                        oracle_expect_probs(logits,
                                            logits.size(),
                                            params,
                                            row,
                                            scratch,
                                            oracle_label(tied ? "tied" : "spread", shape.vocab, params));
                    }
                }
            }
        }
    }
}

TEST(StatefulSpeculativeSamplingOracleProbs, ComputeProbs_SmallTopKSummedInAscendingIdOrder_BitIdenticalToModel) {
    std::mt19937 gen(99);
    ProbRow row;
    Scratch scratch;
    for (size_t trial = 0; trial < 40; ++trial) {
        const auto logits = oracle_spread_logits(gen, 1000 + trial * 37, 5.0f);
        const auto params = oracle_params(1.0, 20 + trial % 30, 1.0);
        oracle_expect_probs(logits, logits.size(), params, row, scratch, oracle_label("small", logits.size(), params));
    }
}

TEST(StatefulSpeculativeSamplingOracleProbs, ComputeProbs_LargeSelectionNucleus_BitIdenticalToModel) {
    struct Shape {
        size_t vocab;
        size_t top_k;
    };
    const std::vector<Shape> shapes = {{40000, 20000}, {40000, 17000}, {151936, 100000}};
    std::mt19937 gen(31);
    ProbRow row;
    Scratch scratch;
    for (const Shape& shape : shapes) {
        for (double top_p : {0.9, 0.35}) {
            for (bool tied : {false, true}) {
                const auto logits =
                    tied ? oracle_grid_logits(gen, shape.vocab, 40) : oracle_spread_logits(gen, shape.vocab, 9.0f);
                const auto params = oracle_params(1.0, shape.top_k, top_p);
                oracle_expect_probs(logits,
                                    logits.size(),
                                    params,
                                    row,
                                    scratch,
                                    oracle_label(tied ? "large-tied" : "large-spread", shape.vocab, params));
            }
        }
    }
}

TEST(StatefulSpeculativeSamplingOracleProbs, ComputeProbs_LargeSelectionAtExtremeTemperature_BitIdenticalToModel) {
    std::mt19937 gen(32);
    ProbRow row;
    Scratch scratch;
    for (double temperature : {0x1p874, 1e300, 0x1p-896, 1e-300}) {
        for (double top_p : {0.9, 0.35}) {
            for (bool tied : {false, true}) {
                const auto logits = tied ? oracle_grid_logits(gen, 40000, 6) : oracle_spread_logits(gen, 40000, 9.0f);
                const auto params = oracle_params(temperature, 20000, top_p);
                oracle_expect_probs(logits,
                                    40000,
                                    params,
                                    row,
                                    scratch,
                                    oracle_label("large-extreme-T", 40000, params));
            }
        }
    }
}

TEST(StatefulSpeculativeSamplingOracleProbs, ComputeProbs_TopKGateBoundaries_BitIdenticalToModel) {
    std::mt19937 gen(515);
    ProbRow row;
    Scratch scratch;
    for (size_t top_k : {size_t(99), size_t(100), size_t(101)}) {
        const auto logits = oracle_spread_logits(gen, 1600, 4.0f);
        const auto params = oracle_params(1.0, top_k, 1.0);
        oracle_expect_probs(logits, logits.size(), params, row, scratch, oracle_label("radix-gate", 1600, params));
    }
    for (size_t top_k : {size_t(449), size_t(450), size_t(451)}) {
        for (double top_p : {1.0, 0.8}) {
            auto logits = oracle_spread_logits(gen, 500, 4.0f);
            for (size_t i = 0; i < 50; ++i) {
                logits[(i * 10 + 3) % 500] = -std::numeric_limits<float>::infinity();
            }
            const auto params = oracle_params(1.0, top_k, top_p);
            oracle_expect_probs(logits, logits.size(), params, row, scratch, oracle_label("finite-gate", 500, params));
        }
    }
}

TEST(StatefulSpeculativeSamplingOracleProbs, ComputeProbs_TinyAndSingleTokenVocabularies_BitIdenticalToModel) {
    std::mt19937 gen(7);
    ProbRow row;
    Scratch scratch;
    for (size_t vocab : {size_t(1), size_t(2), size_t(3), size_t(15), size_t(16), size_t(17)}) {
        for (size_t top_k : {size_t(1), size_t(2), size_t(100), kOracleNoTopK}) {
            for (double top_p : {1.0, 0.5, 0.01}) {
                for (double temperature : {1.0, 0.7}) {
                    for (bool tied : {false, true}) {
                        const auto logits =
                            tied ? oracle_grid_logits(gen, vocab, 2) : oracle_spread_logits(gen, vocab, 3.0f);
                        const size_t valid = vocab > 1 ? vocab - 1 : vocab;
                        const auto params = oracle_params(temperature, top_k, top_p);
                        oracle_expect_probs(logits, vocab, params, row, scratch, oracle_label("tiny", vocab, params));
                        oracle_expect_probs(logits,
                                            valid,
                                            params,
                                            row,
                                            scratch,
                                            oracle_label("tiny-masked", valid, params));
                    }
                }
            }
        }
    }
    EXPECT_EQ(probs_of({-3.0}, oracle_params(0.3, 1, 0.2)).prob, (std::vector<double>{1.0}));
    EXPECT_EQ(probs_of({4.0}, oracle_params(1.0, kOracleNoTopK, 0.2)).prob, (std::vector<double>{1.0}));
}

TEST(StatefulSpeculativeSamplingOracleProbs, ComputeProbs_ReusedRowAndScratchWithGarbage_MatchModelForEveryShape) {
    std::mt19937 gen(1234);
    ProbRow row;
    Scratch scratch;
    const size_t vocabs[] = {17, 64, 1000, 3000, 70, 20000, 5, 2048, 64};
    const size_t top_ks[] = {1, 3, 20, 125, 700, 2999, 100000, kOracleNoTopK};
    const double temperatures[] = {0.5, 1.0, 2.0};
    const double top_ps[] = {1.0, 0.9, 0.2};
    for (size_t round = 0; round < 60; ++round) {
        const size_t vocab = vocabs[round % 9];
        const size_t valid = round % 4 == 3 ? vocab - vocab / 5 : vocab;
        const auto logits =
            round % 2 == 0 ? oracle_spread_logits(gen, vocab, 7.0f) : oracle_grid_logits(gen, vocab, 20);
        const auto params = oracle_params(temperatures[gen() % 3], top_ks[gen() % 8], top_ps[gen() % 3]);
        row.prob.assign(1 + gen() % 5000, std::numeric_limits<double>::quiet_NaN());
        scratch.order.assign(1 + gen() % 3000, 0xDEADBEEFu);
        scratch.scaled.assign(1 + gen() % 3000, -1.0);
        const std::string label = oracle_label("reused", vocab, params) + " valid=" + std::to_string(valid);
        oracle_expect_probs(logits, valid, params, row, scratch, label);
        ProbRow fresh;
        Scratch fresh_scratch;
        compute_probs(logits.data(), vocab, valid, params, fresh, fresh_scratch);
        oracle_expect_rows_equal(row.prob, fresh.prob, label + " fresh");
    }
}

TEST(StatefulSpeculativeSamplingOracleProbs, ComputeProbs_PaddedTailWithInvalidValues_IsIgnoredAndStaysZero) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    std::mt19937 gen(8);
    const std::vector<SamplingParams> configs = {oracle_params(1.0, kOracleNoTopK, 1.0),
                                                 oracle_params(1.0, 3, 1.0),
                                                 oracle_params(1.0, 10, 0.9),
                                                 oracle_params(0.5, kOracleNoTopK, 0.6),
                                                 oracle_params(0x1p900, 10, 1.0)};
    for (const SamplingParams& params : configs) {
        for (float bad : {nan, inf, -inf}) {
            auto logits = oracle_spread_logits(gen, 64, 4.0f);
            for (size_t i = 40; i < 64; ++i) {
                logits[i] = bad;
            }
            ProbRow row;
            Scratch scratch;
            oracle_expect_probs(logits, 40, params, row, scratch, oracle_label("padded-tail", 64, params));
        }
    }
}

TEST(StatefulSpeculativeSamplingOracleProbs, ComputeProbs_InvalidRow_ThrowsAndLeavesZeroRowOfVocabularySize) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    struct Config {
        size_t vocab;
        SamplingParams params;
    };
    const std::vector<Config> configs = {{64, oracle_params(1.0, kOracleNoTopK, 1.0)},
                                         {64, oracle_params(1.0, 3, 1.0)},
                                         {64, oracle_params(1.0, 10, 1.0)},
                                         {64, oracle_params(0.5, 40, 0.9)},
                                         {1000, oracle_params(1.0, 3, 1.0)},
                                         {1000, oracle_params(1.0, 500, 0.5)},
                                         {1000, oracle_params(0x1p900, 3, 1.0)},
                                         {1000, oracle_params(0x1p900, 500, 1.0)}};
    std::mt19937 gen(3);
    for (const Config& config : configs) {
        for (size_t kind = 0; kind < 4; ++kind) {
            for (size_t position : {size_t(0), config.vocab / 2, config.vocab - 1}) {
                for (size_t stale_size : {size_t(7), size_t(200), config.vocab, size_t(5000)}) {
                    auto logits = oracle_spread_logits(gen, config.vocab, 3.0f);
                    if (kind == 0) {
                        logits[position] = nan;
                    } else if (kind == 1) {
                        logits[position] = inf;
                    } else if (kind == 2) {
                        logits.assign(config.vocab, -inf);
                    } else {
                        logits.assign(config.vocab, -inf);
                        logits[position] = nan;
                    }
                    ProbRow row;
                    row.prob.assign(stale_size, 7.0);
                    Scratch scratch;
                    const std::string label =
                        oracle_label("invalid", config.vocab, config.params) + " kind=" + std::to_string(kind) +
                        " position=" + std::to_string(position) + " stale=" + std::to_string(stale_size);
                    EXPECT_THROW(
                        compute_probs(logits.data(), logits.size(), logits.size(), config.params, row, scratch),
                        ov::Exception)
                        << label;
                    ASSERT_EQ(row.prob.size(), config.vocab) << label;
                    EXPECT_TRUE(std::all_of(row.prob.begin(), row.prob.end(), [](double p) {
                        return p == 0.0;
                    })) << label;
                }
            }
        }
    }
}

TEST(StatefulSpeculativeSamplingOracleProbs, ComputeProbs_ValidCallAfterThrow_IsNotAffectedByTheFailedCall) {
    const std::vector<float> bad = {0.0f, std::numeric_limits<float>::quiet_NaN(), 1.0f, 2.0f};
    const std::vector<float> good = {0.5f, -1.0f, 1.5f, 0.0f};
    ProbRow row;
    Scratch scratch;
    const SamplingParams params = oracle_params(1.0, 2, 0.9);
    EXPECT_THROW(compute_probs(bad.data(), 4, 4, params, row, scratch), ov::Exception);
    const auto expected = oracle_probs(good, 4, 1.0, 2, 0.9);
    compute_probs(good.data(), 4, 4, params, row, scratch);
    oracle_expect_rows_equal(row.prob, expected, "after throw");
}

/**
 * Above 2^873 the raw float order and the order of logit / temperature may differ, so top_k must rank the scaled
 * values. There every scaled logit is within 2^-860 of zero, so exp gives exactly 1 for all of them, every selected
 * token has probability 1 / n, and the support of the row is exactly the selected set: any wrong selection or wrong
 * tie rule shows up as a different support, not just as a rounding difference.
 */
TEST(StatefulSpeculativeSamplingOracleProbs, ComputeProbs_TemperatureAboveExactKeyRange_SelectsTopKByLowerIndex) {
    std::mt19937 gen(606);
    ProbRow row;
    Scratch scratch;
    for (double temperature : {0x1p873, 0x1p874, 0x1p900}) {
        for (size_t top_k : {size_t(125), size_t(200), size_t(300), size_t(1000), size_t(1999)}) {
            for (double top_p : {1.0, 0.5}) {
                for (bool tied : {false, true}) {
                    const auto logits = tied ? oracle_grid_logits(gen, 2000, 3) : oracle_spread_logits(gen, 2000, 8.0f);
                    const auto params = oracle_params(temperature, top_k, top_p);
                    oracle_expect_probs(logits, 2000, params, row, scratch, oracle_label("huge-T", 2000, params));
                }
            }
        }
    }
}

TEST(StatefulSpeculativeSamplingOracleProbs, ComputeProbs_HugeTemperatureTies_KeepLowestIndicesOfTheBoundaryLevel) {
    std::vector<float> logits(1000, 0.0f);
    for (size_t i = 0; i < logits.size(); ++i) {
        logits[i] = i < 10 ? 1.0f : (i % 3 == 0 ? 0.5f : -1.0f);
    }
    ProbRow row;
    Scratch scratch;
    compute_probs(logits.data(), logits.size(), logits.size(), oracle_params(0x1p900, 200, 1.0), row, scratch);
    std::vector<size_t> expected = oracle_iota(0, 10);
    for (size_t i = 10; expected.size() < 200; ++i) {
        if (i % 3 == 0) {
            expected.push_back(i);
        }
    }
    EXPECT_EQ(oracle_support(row.prob), expected);
    for (size_t id : expected) {
        EXPECT_EQ(row.prob[id], 1.0 / 200.0);
    }
}

/**
 * The streaming selection keeps a gate equal to the k-th best value after every trim. A trim that leaves a better
 * element at position k - 1 raises the gate and silently drops later candidates, which happens in only a few percent of
 * the rows, so many rows are checked. At temperature 2^900 all probabilities of the selected tokens are equal, so the
 * support of each row is exactly the selected set.
 */
TEST(StatefulSpeculativeSamplingOracleProbs, ComputeProbs_HugeTemperatureManyRows_SupportIsExactTopKSet) {
    std::mt19937 gen(12);
    ProbRow row;
    Scratch scratch;
    struct Shape {
        size_t vocab;
        size_t top_k;
    };
    for (const Shape& shape : {Shape{600, 40}, Shape{1000, 63}, Shape{1000, 100}}) {
        for (size_t trial = 0; trial < 400; ++trial) {
            const auto logits = oracle_spread_logits(gen, shape.vocab, 8.0f);
            const auto params = oracle_params(0x1p900, shape.top_k, 1.0);
            oracle_expect_probs(logits,
                                shape.vocab,
                                params,
                                row,
                                scratch,
                                oracle_label("many-rows", shape.vocab, params));
            if (::testing::Test::HasFailure()) {
                return;
            }
        }
    }
}

/**
 * Below 2^-895 the scaled logits are huge, so exp underflows to 0 for everything but the maximum. With logits on
 * four levels a quarter of the tokens tie for the maximum, and top_k must keep the lowest indices among them.
 */
TEST(StatefulSpeculativeSamplingOracleProbs, ComputeProbs_TemperatureBelowExactKeyRange_KeepsLowestIndexTies) {
    std::mt19937 gen(77);
    std::vector<float> logits(1000);
    for (float& x : logits) {
        x = static_cast<float>(gen() % 4);
    }
    std::vector<size_t> maxima;
    for (size_t i = 0; i < logits.size(); ++i) {
        if (logits[i] == 3.0f) {
            maxima.push_back(i);
        }
    }
    ASSERT_GT(maxima.size(), 150u);
    for (double temperature : {0x1p-896, 0x1p-900, 1e-300}) {
        for (size_t top_k : {size_t(1), size_t(100), size_t(500), kOracleNoTopK}) {
            for (double top_p : {1.0, 0.5}) {
                ProbRow row;
                Scratch scratch;
                const auto params = oracle_params(temperature, top_k, top_p);
                oracle_expect_probs(logits, 1000, params, row, scratch, oracle_label("tiny-T", 1000, params));
            }
        }
        ProbRow row;
        Scratch scratch;
        compute_probs(logits.data(), 1000, 1000, oracle_params(temperature, 100, 1.0), row, scratch);
        EXPECT_EQ(oracle_support(row.prob), std::vector<size_t>(maxima.begin(), maxima.begin() + 100));
        for (size_t id : oracle_support(row.prob)) {
            EXPECT_EQ(row.prob[id], 0.01);
        }
    }
}

/**
 * With temperature 2^-900 every logit of at least 2^124 scales to +inf, so those tokens tie even though their float
 * values differ, and the lowest indices must win. The probabilities of the tied tokens are not meaningful (inf - inf
 * is NaN), so only the support is asserted.
 */
TEST(StatefulSpeculativeSamplingOracleProbs, ComputeProbs_ScaledLogitsOverflowingToInfinity_TieByLowerIndex) {
    std::vector<float> logits(1000, -1.0f);
    for (size_t i = 0; i < 300; ++i) {
        logits[i] = 3.0e37f + static_cast<float>(i) * 1.0e35f;
    }
    ProbRow row;
    Scratch scratch;
    compute_probs(logits.data(), logits.size(), logits.size(), oracle_params(0x1p-900, 100, 1.0), row, scratch);
    ASSERT_EQ(row.prob.size(), logits.size());
    EXPECT_EQ(oracle_support(row.prob), oracle_iota(0, 100));
}

/**
 * Above 2^873 a scaled logit may underflow to zero. Tiny denormal logits of distinct float values then all scale to
 * 0.0 and tie, so the lowest indices win although the raw float order would prefer the larger magnitudes.
 */
TEST(StatefulSpeculativeSamplingOracleProbs, ComputeProbs_UnderflowingScaledLogits_TieByLowerIndex) {
    std::vector<float> logits(1000);
    for (size_t i = 0; i < logits.size(); ++i) {
        const float magnitude = std::ldexp(static_cast<float>(1 + (i * 7919) % 8000), -149);
        logits[i] = i % 2 == 0 ? magnitude : -magnitude;
    }
    for (double top_p : {1.0, 0.5}) {
        ProbRow row;
        Scratch scratch;
        const auto params = oracle_params(0x1p940, 100, top_p);
        oracle_expect_probs(logits, 1000, params, row, scratch, oracle_label("underflow", 1000, params));
        const auto support = oracle_support(row.prob);
        EXPECT_EQ(support, oracle_iota(0, support.size()));
        EXPECT_EQ(support.size() == 100, top_p == 1.0);
    }
    ProbRow row;
    Scratch scratch;
    compute_probs(logits.data(), 1000, 1000, oracle_params(0x1p940, 100, 1.0), row, scratch);
    EXPECT_EQ(oracle_support(row.prob), oracle_iota(0, 100));
}

TEST(StatefulSpeculativeSamplingOracleProbs, ComputeProbs_TemperatureAtExactKeyRangeEdges_BitIdenticalToModel) {
    std::mt19937 gen(404);
    ProbRow row;
    Scratch scratch;
    const double temperatures[] = {0x1p-895, 0x1p-894, 0x1p-896, 0x1p872, 0x1p873, 0x1p874};
    for (double temperature : temperatures) {
        for (size_t top_k : {size_t(10), size_t(300), size_t(1999), kOracleNoTopK}) {
            for (double top_p : {1.0, 0.7}) {
                const auto logits = oracle_spread_logits(gen, 2000, 100.0f);
                const auto params = oracle_params(temperature, top_k, top_p);
                oracle_expect_probs(logits, 2000, params, row, scratch, oracle_label("edge-T", 2000, params));
            }
        }
    }
}

TEST(StatefulSpeculativeSamplingOracleProbs, ComputeProbs_ThreeLevelRows_BitIdenticalToModelAcrossNucleusFallbacks) {
    const size_t vocab = 151936;
    std::mt19937 gen(2718);
    std::vector<uint32_t> perm(vocab);
    for (size_t i = 0; i < vocab; ++i) {
        perm[i] = static_cast<uint32_t>(i);
    }
    std::shuffle(perm.begin(), perm.end(), gen);
    std::vector<float> logits(vocab, -20.0f);
    for (size_t i = 0; i < 10; ++i) {
        logits[perm[i]] = 5.0f;
    }
    for (size_t i = 10; i < 60010; ++i) {
        logits[perm[i]] = 0.0f;
    }
    ProbRow row;
    Scratch scratch;
    for (size_t top_k : {kOracleNoTopK, size_t(70000)}) {
        for (double top_p : {0.9, 0.999}) {
            for (double temperature : {1.0, 0.5, 0x1p874}) {
                const auto params = oracle_params(temperature, top_k, top_p);
                oracle_expect_probs(logits, vocab, params, row, scratch, oracle_label("three-level", vocab, params));
            }
        }
    }
}

/**
 * exp(x) for -745 < x < -708 is subnormal, so such probabilities carry fewer than 53 significant bits. The model
 * ranks them with the same exp, so the nucleus must still equal the model bit for bit.
 */
TEST(StatefulSpeculativeSamplingOracleProbs, ComputeProbs_ArgumentsBelowSubnormalThreshold_BitIdenticalToModel) {
    std::mt19937 gen(1);
    ProbRow row;
    Scratch scratch;
    for (double temperature : {1.0, 0.25}) {
        for (size_t top_k : {size_t(2500), kOracleNoTopK}) {
            for (double top_p : {0.5, 0.999999, 1.0}) {
                std::vector<float> logits(5000);
                for (float& x : logits) {
                    x = -static_cast<float>(gen() % 7500) / 10.0f * static_cast<float>(temperature);
                }
                const auto params = oracle_params(temperature, top_k, top_p);
                oracle_expect_probs(logits, 5000, params, row, scratch, oracle_label("subnormal", 5000, params));
            }
        }
    }
}

namespace {

/**
 * Index of the first positive weight whose running sum exceeds u * total, or of the last positive weight if rounding
 * keeps the running sum from exceeding it; the inverse-CDF rule of sample_from in plain index order.
 */
size_t oracle_inverse_cdf(const std::vector<double>& weights, double u) {
    double total = 0.0;
    for (double w : weights) {
        total += w;
    }
    const double threshold = u * total;
    double cumulative = 0.0;
    size_t last_positive = weights.size();
    for (size_t i = 0; i < weights.size(); ++i) {
        if (weights[i] > 0.0) {
            cumulative += weights[i];
            last_positive = i;
            if (cumulative > threshold) {
                return i;
            }
        }
    }
    return last_positive;
}

std::vector<double> oracle_sparse_row(std::mt19937& gen, size_t size, uint32_t density_percent) {
    std::vector<double> weights(size, 0.0);
    for (double& w : weights) {
        if (gen() % 100 < density_percent) {
            w = static_cast<double>(1 + gen() % 1000) * 0x1p-10;
        }
    }
    if (oracle_support(weights).empty()) {
        weights[gen() % size] = 1.0;
    }
    return weights;
}

}  // namespace

/**
 * The weights {1/4, 1/4, 1/2} and every u below are dyadic, so the running sums and u * total are exact and the
 * expected index follows from the cumulative edges 1/4, 1/2 and 1 alone: the smallest i with edge_i > u, so a u equal
 * to an edge belongs to the next token, u = 0 skips leading zeros and u just below 1 never reaches a trailing zero.
 */
TEST(StatefulSpeculativeSamplingOracleSample, SampleFrom_DyadicEdgesAtEveryBlockOffset_PicksStrictlyAboveEdge) {
    const double us[] = {0.0,
                         0x1p-3,
                         std::nextafter(0.25, 0.0),
                         0.25,
                         std::nextafter(0.25, 1.0),
                         std::nextafter(0.5, 0.0),
                         0.5,
                         std::nextafter(0.5, 1.0),
                         0.75,
                         std::nextafter(1.0, 0.0)};
    const size_t expected_offsets[] = {0, 0, 0, 1, 1, 1, 2, 2, 2, 2};
    for (size_t size : {size_t(4), size_t(17), size_t(40), size_t(1000), size_t(5000)}) {
        for (size_t shift : {size_t(0),
                             size_t(1),
                             size_t(13),
                             size_t(14),
                             size_t(15),
                             size_t(16),
                             size_t(17),
                             size_t(31),
                             size_t(33)}) {
            if (shift + 3 > size) {
                continue;
            }
            std::vector<double> weights(size, 0.0);
            weights[shift] = 0.25;
            weights[shift + 1] = 0.25;
            weights[shift + 2] = 0.5;
            const ProbRow row = make_row(weights);
            for (size_t j = 0; j < 10; ++j) {
                EXPECT_EQ(sample_from(row, us[j]), shift + expected_offsets[j])
                    << "size=" << size << " shift=" << shift << " u=" << us[j];
            }
        }
    }
}

/**
 * 16384 equal weights 2^-14 sum to exactly 1 and u = j * 2^-14 is exact, so the cumulative edge j * 2^-14 puts that
 * u on token j and the u just below it on token j - 1. The row is longer than any block-tracking table and has no
 * empty block, which makes every block non-empty.
 */
TEST(StatefulSpeculativeSamplingOracleSample, SampleFrom_UniformDyadicWeights_IndexIsFloorOfUTimesCount) {
    std::vector<double> weights(20000, 0.0);
    for (size_t i = 0; i < 16384; ++i) {
        weights[i] = 0x1p-14;
    }
    const ProbRow row = make_row(weights);
    for (size_t j : {size_t(0),
                     size_t(1),
                     size_t(15),
                     size_t(16),
                     size_t(17),
                     size_t(1023),
                     size_t(1024),
                     size_t(8191),
                     size_t(8192),
                     size_t(16383)}) {
        const double edge = static_cast<double>(j) * 0x1p-14;
        EXPECT_EQ(sample_from(row, edge), j);
        if (j > 0) {
            EXPECT_EQ(sample_from(row, std::nextafter(edge, 0.0)), j - 1);
        }
    }
    EXPECT_EQ(sample_from(row, std::nextafter(1.0, 0.0)), 16383u);
}

TEST(StatefulSpeculativeSamplingOracleSample, SampleFrom_SparseThenDenseBlocks_MatchesPlainInverseCdf) {
    std::vector<double> weights(24000, 0.0);
    for (size_t block = 0; block < 1500; ++block) {
        if (block >= 64 || block % 2 == 0) {
            weights[block * 16 + 7] = 0x1p-11;
        }
    }
    const ProbRow row = make_row(weights);
    std::mt19937 gen(5);
    std::vector<double> us = {0.0, std::nextafter(1.0, 0.0), 0.5, 0.999999};
    for (size_t i = 0; i < 300; ++i) {
        us.push_back(oracle_unit(gen));
    }
    for (double u : us) {
        EXPECT_EQ(sample_from(row, u), oracle_inverse_cdf(weights, u)) << "u=" << u;
    }
}

TEST(StatefulSpeculativeSamplingOracleSample, SampleFrom_RandomSparseRows_MatchesPlainInverseCdf) {
    std::mt19937 gen(60);
    for (size_t size : {size_t(1),
                        size_t(5),
                        size_t(15),
                        size_t(16),
                        size_t(17),
                        size_t(31),
                        size_t(32),
                        size_t(33),
                        size_t(100),
                        size_t(1000),
                        size_t(20000)}) {
        for (uint32_t density : {1u, 10u, 60u, 100u}) {
            const auto weights = oracle_sparse_row(gen, size, density);
            const ProbRow row = make_row(weights);
            std::vector<double> us = {0.0, std::nextafter(1.0, 0.0), 0.5};
            for (size_t i = 0; i < 40; ++i) {
                us.push_back(oracle_unit(gen));
            }
            for (double u : us) {
                EXPECT_EQ(sample_from(row, u), oracle_inverse_cdf(weights, u))
                    << "size=" << size << " density=" << density << " u=" << u;
            }
        }
    }
}

/**
 * Three weights of 2^-1074 sum to 3 * 2^-1074. For u = 1 - 2^-53 the exact product u * total is below the total by
 * far less than half the subnormal spacing, so it rounds to the total and the strict comparison never succeeds. The
 * specification then asks for the last token with positive mass, never for a trailing zero-mass token.
 */
TEST(StatefulSpeculativeSamplingOracleSample,
     SampleFrom_SubnormalTotalWhereThresholdRoundsToTotal_ReturnsLastPositive) {
    for (size_t size : {size_t(40), size_t(5), size_t(1000)}) {
        std::vector<double> weights(size, 0.0);
        const size_t first = 2;
        for (size_t i = first; i < first + 3; ++i) {
            weights[i] = std::numeric_limits<double>::denorm_min();
        }
        const ProbRow row = make_row(weights);
        EXPECT_EQ(sample_from(row, std::nextafter(1.0, 0.0)), first + 2) << "size=" << size;
        EXPECT_EQ(sample_from(row, 0.0), first) << "size=" << size;
    }
}

TEST(StatefulSpeculativeSamplingOracleSample, SampleFrom_NegativeZeroAndZeroBlocks_NeverSelected) {
    std::vector<double> weights(64, 0.0);
    for (size_t i = 0; i < 48; ++i) {
        weights[i] = -0.0;
    }
    weights[50] = 0.5;
    weights[63] = 0.5;
    const ProbRow row = make_row(weights);
    EXPECT_EQ(sample_from(row, 0.0), 50u);
    EXPECT_EQ(sample_from(row, 0.49), 50u);
    EXPECT_EQ(sample_from(row, 0.5), 63u);
    EXPECT_THROW(sample_from(make_row(std::vector<double>(64, -0.0)), 0.5), ov::Exception);
}

namespace {

struct OracleVerdict {
    size_t num_accepted;
    int64_t final_token;
};

/**
 * Leviathan et al. accept/resample rule written as plain loops: candidate i is accepted iff u_i < min(1, q/p) with
 * p = 0 rejecting, the first rejection draws from normalise(max(0, q - p)) or from q when that residual is empty,
 * and a fully accepted block draws the bonus token from the last target row.
 */
OracleVerdict oracle_verify(const std::vector<int64_t>& candidates,
                            const std::vector<std::vector<double>>& draft,
                            const std::vector<std::vector<double>>& target,
                            const std::vector<double>& accept_u,
                            double final_u) {
    for (size_t i = 0; i < candidates.size(); ++i) {
        const size_t token = static_cast<size_t>(candidates[i]);
        const double p = draft[i][token];
        const double q = target[i][token];
        const double ratio = p > 0.0 ? std::min(1.0, q / p) : 0.0;
        if (accept_u[i] < ratio) {
            continue;
        }
        std::vector<double> residual(target[i].size());
        double total = 0.0;
        for (size_t j = 0; j < residual.size(); ++j) {
            residual[j] = std::max(0.0, target[i][j] - draft[i][j]);
            total += residual[j];
        }
        const size_t drawn =
            total > 0.0 ? oracle_inverse_cdf(residual, final_u) : oracle_inverse_cdf(target[i], final_u);
        return {i, static_cast<int64_t>(drawn)};
    }
    return {candidates.size(), static_cast<int64_t>(oracle_inverse_cdf(target[candidates.size()], final_u))};
}

std::vector<ProbRow> oracle_rows(const std::vector<std::vector<double>>& values) {
    std::vector<ProbRow> rows;
    for (const auto& v : values) {
        rows.push_back(make_row(v));
    }
    return rows;
}

}  // namespace

/**
 * Probabilities 1/2 and ratio * 1/2 with ratio a dyadic fraction are exact, so q/p equals the ratio exactly and the
 * test can place accept_u on the acceptance boundary: u equal to the ratio rejects (strict comparison), the
 * predecessor of the ratio accepts, the successor rejects.
 */
TEST(StatefulSpeculativeSamplingOracleVerify, VerifyWithUniforms_UniformEqualToAcceptanceRatio_Rejects) {
    for (size_t vocab : {size_t(2), size_t(20), size_t(40)}) {
        for (size_t token : {size_t(0), vocab - 1}) {
            const size_t other = token == 0 ? vocab - 1 : 0;
            for (double ratio : {0.125, 0.25, 0.5, 0.75}) {
                std::vector<double> p(vocab, 0.0);
                std::vector<double> q(vocab, 0.0);
                p[token] = 0.5;
                p[other] = 0.5;
                q[token] = 0.5 * ratio;
                q[other] = 1.0 - q[token];
                const auto pr = make_row(p);
                const auto qr = make_row(q);
                const auto at = verify_with_uniforms({static_cast<int64_t>(token)}, {pr}, {qr, qr}, {ratio}, 0.0);
                EXPECT_EQ(at.num_accepted, 0u) << "ratio=" << ratio;
                EXPECT_FALSE(at.all_accepted);
                EXPECT_EQ(at.final_token, static_cast<int64_t>(other));
                const auto below = verify_with_uniforms({static_cast<int64_t>(token)},
                                                        {pr},
                                                        {qr, qr},
                                                        {std::nextafter(ratio, 0.0)},
                                                        0.0);
                EXPECT_EQ(below.num_accepted, 1u) << "ratio=" << ratio;
                EXPECT_TRUE(below.all_accepted);
                const auto above = verify_with_uniforms({static_cast<int64_t>(token)},
                                                        {pr},
                                                        {qr, qr},
                                                        {std::nextafter(ratio, 1.0)},
                                                        0.0);
                EXPECT_EQ(above.num_accepted, 0u) << "ratio=" << ratio;
            }
        }
    }
}

TEST(StatefulSpeculativeSamplingOracleVerify, VerifyWithUniforms_TargetAtLeastDraft_AcceptedForEveryUniformBelowOne) {
    const auto p = make_row({0.25, 0.75});
    const auto q = make_row({0.75, 0.25});
    for (double u : {0.0, 0.5, std::nextafter(1.0, 0.0)}) {
        EXPECT_EQ(verify_with_uniforms({0}, {p}, {q, q}, {u}, 0.0).num_accepted, 1u) << "u=" << u;
        EXPECT_EQ(verify_with_uniforms({0}, {q}, {q, q}, {u}, 0.0).num_accepted, 1u) << "u=" << u;
    }
}

TEST(StatefulSpeculativeSamplingOracleVerify, VerifyWithUniforms_ZeroDraftProbability_RejectedForEveryUniform) {
    for (size_t vocab : {size_t(2), size_t(40)}) {
        std::vector<double> p(vocab, 0.0);
        std::vector<double> q(vocab, 0.0);
        p[0] = 1.0;
        q[vocab - 1] = 1.0;
        const auto pr = make_row(p);
        const auto qr = make_row(q);
        for (double u : {0.0, 0.5, std::nextafter(1.0, 0.0)}) {
            const auto result = verify_with_uniforms({static_cast<int64_t>(vocab - 1)}, {pr}, {qr, qr}, {u}, 0.0);
            EXPECT_EQ(result.num_accepted, 0u) << "u=" << u;
            EXPECT_FALSE(result.all_accepted);
            EXPECT_EQ(result.final_token, static_cast<int64_t>(vocab - 1));
        }
    }
}

/**
 * Both rows are exactly dyadic. The residual max(0, q - p) is {0, 1/8, 1/8, 0} with total 1/4, so final_u = 1/2 puts
 * the threshold on the cumulative edge 1/8 and must pick the second positive entry, the predecessor of 1/2 the first.
 */
TEST(StatefulSpeculativeSamplingOracleVerify, VerifyWithUniforms_ResidualEdge_PicksTokenAboveCumulativeEdge) {
    const auto p = make_row({0.5, 0.125, 0.125, 0.25});
    const auto q = make_row({0.25, 0.25, 0.25, 0.25});
    const auto edge = verify_with_uniforms({0}, {p}, {q, q}, {0.5}, 0.5);
    EXPECT_EQ(edge.num_accepted, 0u);
    EXPECT_EQ(edge.final_token, 2);
    const auto below = verify_with_uniforms({0}, {p}, {q, q}, {0.5}, std::nextafter(0.5, 0.0));
    EXPECT_EQ(below.final_token, 1);
    const auto top = verify_with_uniforms({0}, {p}, {q, q}, {0.5}, std::nextafter(1.0, 0.0));
    EXPECT_EQ(top.final_token, 2);
}

/**
 * Candidate 0 of the second position is rejected (u equals its ratio 1/2) while q <= p everywhere, so the residual is
 * empty and the token comes from q_1 itself: q_1 has total 1/2, final_u = 0.9 puts the threshold at 0.45 and selects
 * token 1. The neighbouring rows p_1 and q_2 would give 2 and 3, so a fallback to a wrong row is detected.
 */
TEST(StatefulSpeculativeSamplingOracleVerify, VerifyWithUniforms_EmptyResidual_FallsBackToTargetRowOfThatPosition) {
    const auto p0 = make_row({0.25, 0.25, 0.25, 0.25});
    const auto q0 = make_row({0.25, 0.25, 0.25, 0.25});
    const auto p1 = make_row({0.5, 0.25, 0.25, 0.0});
    const auto q1 = make_row({0.25, 0.25, 0.0, 0.0});
    const auto q2 = make_row({0.0, 0.0, 0.0, 1.0});
    const auto result = verify_with_uniforms({1, 0}, {p0, p1}, {q0, q1, q2}, {0.0, 0.5}, 0.9);
    EXPECT_EQ(result.num_accepted, 1u);
    EXPECT_FALSE(result.all_accepted);
    EXPECT_EQ(result.final_token, 1);
    const auto low = verify_with_uniforms({1, 0}, {p0, p1}, {q0, q1, q2}, {0.0, 0.5}, 0.4);
    EXPECT_EQ(low.final_token, 0);
}

TEST(StatefulSpeculativeSamplingOracleVerify, VerifyWithUniforms_RandomSparseRows_MatchPlainLoopModel) {
    std::mt19937 gen(31337);
    for (size_t iteration = 0; iteration < 3000; ++iteration) {
        const size_t vocab = std::vector<size_t>{3, 17, 40, 300}[iteration % 4];
        const size_t block = gen() % 5;
        std::vector<std::vector<double>> draft;
        std::vector<std::vector<double>> target;
        std::vector<int64_t> candidates;
        std::vector<double> accept_u;
        for (size_t i = 0; i < block; ++i) {
            draft.push_back(oracle_sparse_row(gen, vocab, 30));
            target.push_back(oracle_sparse_row(gen, vocab, 30));
            candidates.push_back(static_cast<int64_t>(gen() % vocab));
            accept_u.push_back(gen() % 5 == 0 ? 0.0 : oracle_unit(gen));
        }
        target.push_back(oracle_sparse_row(gen, vocab, 30));
        const double final_u = gen() % 7 == 0 ? std::nextafter(1.0, 0.0) : oracle_unit(gen);
        const OracleVerdict expected = oracle_verify(candidates, draft, target, accept_u, final_u);
        const auto actual =
            verify_with_uniforms(candidates, oracle_rows(draft), oracle_rows(target), accept_u, final_u);
        ASSERT_EQ(actual.num_accepted, expected.num_accepted) << "iteration " << iteration;
        ASSERT_EQ(actual.final_token, expected.final_token) << "iteration " << iteration;
        ASSERT_EQ(actual.all_accepted, expected.num_accepted == block) << "iteration " << iteration;
    }
}

/**
 * mt19937 yields 32-bit words w1, w2. The 53-bit mantissa is (w1 >> 5) * 2^26 + (w2 >> 6), 27 + 26 bits, and the
 * draw is that integer times 2^-53. The expectation below assembles the integer with shifts and ldexp, without the
 * multiplications used by the implementation.
 */
TEST(StatefulSpeculativeSamplingOracleUniform, Uniform01_SeededStream_EqualsIndependentFiftyThreeBitAssembly) {
    for (uint32_t seed : {0u, 1u, 42u, 5489u, 123456789u, 4294967295u}) {
        std::mt19937 rng(seed);
        std::mt19937 ref(seed);
        for (size_t i = 0; i < 2000; ++i) {
            const uint64_t high = ref();
            const uint64_t low = ref();
            const uint64_t mantissa = ((high >> 5) << 26) | (low >> 6);
            const double expected = std::ldexp(static_cast<double>(mantissa), -53);
            ASSERT_EQ(uniform01(rng), expected) << "seed=" << seed << " draw=" << i;
        }
    }
}

/**
 * The C++ standard fixes the 10000th output of a default-constructed mt19937 to 4123659995. The 5000th draw consumes
 * outputs 9999 and 10000, so that value anchors its low word independently of any helper in this file.
 */
TEST(StatefulSpeculativeSamplingOracleUniform, Uniform01_StandardTenThousandthOutput_FixesLowWord) {
    std::mt19937 rng;
    for (size_t i = 0; i < 4999; ++i) {
        static_cast<void>(uniform01(rng));
    }
    std::mt19937 peek = rng;
    const uint64_t high = peek();
    const uint64_t low = peek();
    ASSERT_EQ(low, 4123659995u);
    const uint64_t mantissa = ((high >> 5) << 26) | (low >> 6);
    EXPECT_EQ(uniform01(rng), std::ldexp(static_cast<double>(mantissa), -53));
    EXPECT_EQ(rng(), peek());
}

TEST(StatefulSpeculativeSamplingOracleUniform, Uniform01_AllDraws_LieOnTheFiftyThreeBitGrid) {
    std::mt19937 rng(9);
    for (size_t i = 0; i < 20000; ++i) {
        const double u = uniform01(rng);
        const double scaled = std::ldexp(u, 53);
        ASSERT_GE(u, 0.0);
        ASSERT_LT(u, 1.0);
        ASSERT_EQ(scaled, std::floor(scaled));
    }
}
