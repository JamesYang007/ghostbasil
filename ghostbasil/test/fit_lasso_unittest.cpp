#include <gtest/gtest.h>
#include <ghostbasil/lasso.hpp>
#include <testutil/fit_lasso_util.hpp>

namespace ghostbasil {
namespace {

using namespace fit_lasso_util;

static constexpr double tol = 1e-8;
static constexpr double thr = 1e-16;
static constexpr size_t max_cds = 1000;

template <class GenerateFType>
void test_fit_lasso(GenerateFType generate_dataset)
{
    auto dataset = generate_dataset();
    auto& A = std::get<0>(dataset);
    auto& r = std::get<1>(dataset);
    auto& s = std::get<2>(dataset);
    auto& strong_set = std::get<3>(dataset);
    auto& lmdas = std::get<4>(dataset);
    auto& expected_betas = std::get<5>(dataset);
    auto& expected_objs = std::get<6>(dataset);

    auto output = make_lasso_output(strong_set, r);
    auto& warm_start = std::get<0>(output);
    auto& betas = std::get<1>(output);
    auto& strong_grad = std::get<2>(output);
    auto& active_set = std::get<3>(output);
    auto& is_active = std::get<4>(output);
    auto& n_cds = std::get<5>(output);
    auto& n_lmdas = std::get<6>(output);

    fit_lasso(A, s, strong_set, lmdas, max_cds, thr, warm_start, 
                betas, strong_grad, active_set, is_active, n_cds, n_lmdas);

    EXPECT_LE(n_cds, max_cds);

    EXPECT_EQ(betas.cols(), lmdas.size());
    EXPECT_EQ(expected_betas.cols(), lmdas.size());
    EXPECT_EQ(expected_objs.size(), lmdas.size());

    for (size_t i = 0; i < lmdas.size(); ++i) {
        auto actual = betas.col(i);
        auto expected = expected_betas.col(i);

        EXPECT_NEAR(expected_objs[i], objective(A, r, s, lmdas[i], actual), tol);
        EXPECT_EQ(actual.size(), expected.size());
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_NEAR(expected[i], actual.coeff(i), tol);
        }
    }
}

TEST(FitLasso, fit_lasso_n_ge_p_full)
{
    test_fit_lasso(generate_dataset_1);
}

TEST(FitLasso, fit_lasso_n_ge_p_partial)
{
    test_fit_lasso(generate_dataset_2);
}

TEST(FitLasso, fit_lasso_n_le_p_full)
{
    test_fit_lasso(generate_dataset_3);
}

TEST(FitLasso, fit_lasso_n_le_p_partial)
{
    test_fit_lasso(generate_dataset_4);
}

TEST(FitLasso, fit_lasso_p_large)
{
    test_fit_lasso(generate_dataset_5);
}

}
} // namespace ghostbasil
