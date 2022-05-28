#include <gtest/gtest.h>
#include <ghostbasil/optimization/basil.hpp>
#include <optimization/basil_util.hpp>
#include <testutil/data_util.hpp>
#include <testutil/macros.hpp>

namespace ghostbasil {
namespace {

using namespace basil_util;
    
static constexpr double tol = 2e-8;
static constexpr double thr = 1e-16;
static constexpr size_t max_cds = 10000;
static constexpr size_t max_n_lambdas = 3;
static constexpr size_t n_lambdas_iter = 2;
static const size_t n_threads = std::thread::hardware_concurrency();

template <class GenerateFType>
void test_basil(
        GenerateFType generate_dataset,
        size_t max_strong_size = 100,
        size_t strong_size = 1,
        size_t delta_strong_size = 1,
        bool do_user = false)
{
    auto input = make_basil_input(generate_dataset, max_strong_size, do_user);
    auto& A = std::get<0>(input);
    auto& r = std::get<1>(input);
    auto& s = std::get<2>(input);
    auto& user_lmdas = std::get<3>(input);
    auto& max_strong_size_local = std::get<4>(input);
    auto& betas = std::get<5>(input);
    auto& lmdas = std::get<6>(input);
    auto& rsqs = std::get<7>(input);
    auto& expected_betas = std::get<8>(input);
    auto& expected_lmdas = std::get<9>(input);
    auto& expected_objs = std::get<10>(input);

    try {
        basil(A, r, s, user_lmdas, max_n_lambdas, n_lambdas_iter,
              strong_size, delta_strong_size, max_strong_size_local, max_cds, thr, n_threads,
              betas, lmdas, rsqs);
#ifdef MAKE_LMDAS
        for (size_t i = 0; i < lmdas.size(); ++i) {
            PRINT(lmdas[i]);
        }
        return;
#endif
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return;
    }

    EXPECT_EQ(expected_lmdas.size(), expected_objs.size());
    EXPECT_EQ(betas.size(), lmdas.size());

    for (size_t i = 0; i < lmdas.size(); ++i) {
        const auto& betas_i = betas[i];
        const auto& lmdas_i = lmdas[i];
        auto expected = expected_betas.col(i);

        EXPECT_DOUBLE_EQ(lmdas_i, expected_lmdas[i]);
        EXPECT_NEAR(expected_objs[i], objective(A, r, s, lmdas_i, betas_i), tol);
        EXPECT_EQ(betas_i.size(), expected.size());
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_NEAR(expected[i], betas_i.coeff(i), tol);
        }
    }
    EXPECT_EQ(expected_lmdas.size(), lmdas.size());
    EXPECT_EQ(expected_betas.cols(), lmdas.size());
    EXPECT_EQ(expected_objs.size(), lmdas.size());
}

#ifndef TEST_BASIL_FN
#define TEST_BASIL_FN(n) \
    []() { \
        return testutil::generate_dataset("basil_" STRINGIFY(n));\
    }
#endif

TEST(Basil, basil_n_ge_p)
{
    test_basil(TEST_BASIL_FN(1));
}

TEST(Basil, basil_n_le_p)
{
    test_basil(TEST_BASIL_FN(2));
}

TEST(Basil, basil_p_large)
{
    size_t max_strong_size = 1000;
    size_t strong_size = 100;
    size_t delta_strong_size = 50;
    test_basil(TEST_BASIL_FN(3),
            max_strong_size,
            strong_size,
            delta_strong_size);
}
    
} // namespace
} // namespace ghostbasil
