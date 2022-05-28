#include <gtest/gtest.h>
#include <ghostbasil/optimization/lasso.hpp>
#include <optimization/lasso_util.hpp>
#include <testutil/data_util.hpp>
#include <testutil/macros.hpp>

namespace ghostbasil {
namespace {

using namespace lasso_util;

static constexpr double tol = 2e-8;
static constexpr double thr = 1e-16;
static constexpr size_t max_cds = 1000;

template <class GenerateFType>
void test_lasso(GenerateFType generate_dataset)
{
    auto input = make_lasso_input(generate_dataset);
    auto& A = std::get<0>(input);
    auto& r = std::get<1>(input);
    auto& s = std::get<2>(input);
    auto& strong_set = std::get<3>(input);
    auto& strong_A_diag = std::get<4>(input);
    auto& lmdas = std::get<5>(input);
    auto& rsq = std::get<6>(input);
    auto& strong_beta = std::get<7>(input);
    auto& strong_grad = std::get<8>(input);
    auto& active_set = std::get<9>(input);
    auto& is_active = std::get<10>(input);
    auto& betas = std::get<11>(input);
    auto& rsqs = std::get<12>(input);
    auto& n_cds = std::get<13>(input);
    auto& n_lmdas = std::get<14>(input);
    auto& expected_betas = std::get<15>(input);
    auto& expected_objs = std::get<16>(input);

    lasso(A, s, strong_set, strong_A_diag, lmdas, max_cds, thr, rsq, strong_beta, 
          strong_grad, active_set, is_active, betas, rsqs, n_cds, n_lmdas);

    EXPECT_LE(n_cds, max_cds);

    EXPECT_EQ(betas.size(), lmdas.size());
    EXPECT_EQ(expected_betas.cols(), lmdas.size());
    EXPECT_EQ(expected_objs.size(), lmdas.size());

    for (size_t i = 0; i < lmdas.size(); ++i) {
        const auto& actual = betas[i];
        auto expected = expected_betas.col(i);

        EXPECT_NEAR(expected_objs[i], objective(A, r, s, lmdas[i], actual), tol);
        EXPECT_EQ(actual.size(), expected.size());
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_NEAR(expected[i], actual.coeff(i), tol);
        }
    }
}

#ifndef TEST_LASSO
#define TEST_LASSO(n) \
    test_lasso([]() { \
        return testutil::generate_dataset("lasso_" STRINGIFY(n));\
    })
#endif

TEST(Lasso, lasso_n_ge_p_full)
{
    TEST_LASSO(1);
}

TEST(Lasso, lasso_n_ge_p_partial)
{
    TEST_LASSO(2);
}

TEST(Lasso, lasso_n_le_p_full)
{
    TEST_LASSO(3);
}

TEST(Lasso, lasso_n_le_p_partial)
{
    TEST_LASSO(4);
}

TEST(Lasso, lasso_p_large)
{
    TEST_LASSO(5);
}

#undef TEST_LASSO

}
} // namespace ghostbasil
