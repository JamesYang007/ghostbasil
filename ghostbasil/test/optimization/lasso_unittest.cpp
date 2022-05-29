#include <gtest/gtest.h>
#include <ghostbasil/optimization/lasso.hpp>
#include <tools/data_util.hpp>
#include <tools/macros.hpp>

namespace ghostbasil {
namespace {

struct LassoFixture
    : ::testing::Test
{
    const double tol = 1e-8;
    const double thr = 1e-16;
    const size_t max_cds = 1000;

    template <class F>
    auto make_input(F generate_dataset)
    {
        auto dataset = generate_dataset();
        auto&& A = std::get<0>(dataset);
        auto&& r = std::get<1>(dataset);
        auto&& s = std::get<2>(dataset);
        auto&& strong_set = std::get<3>(dataset);
        auto&& lmdas = std::get<4>(dataset);
        auto&& expected_betas = std::get<5>(dataset);
        auto&& expected_objs = std::get<6>(dataset);

        size_t p = r.size();

        std::vector<double> strong_grad(strong_set.size());
        for (size_t i = 0; i < strong_grad.size(); ++i) {
            strong_grad[i] = r[strong_set[i]];
        }

        Eigen::Vector<double, Eigen::Dynamic> strong_beta(strong_set.size());
        strong_beta.setZero();
        
        Eigen::Vector<double, Eigen::Dynamic> strong_A_diag(strong_set.size());
        for (int i = 0; i < strong_A_diag.size(); ++i) {
            auto k = strong_set[i];
            strong_A_diag[i] = A(k,k);
        }

        using sp_vec_t = Eigen::SparseVector<double>;
        Eigen::Vector<sp_vec_t, Eigen::Dynamic> betas(lmdas.size());
        betas.fill(sp_vec_t(p));
        std::vector<uint32_t> active_set;
        std::vector<bool> is_active(strong_set.size(), false);
        std::vector<double> rsqs(lmdas.size());
        size_t n_cds = 0;
        size_t n_lmdas = 0;

        double rsq = 0;

        return std::make_tuple(
                std::move(A), std::move(r), std::move(s),
                std::move(strong_set), std::move(strong_A_diag), 
                std::move(lmdas), rsq, 
                std::move(strong_beta),
                std::move(strong_grad),
                std::move(active_set),
                std::move(is_active),
                std::move(betas),
                std::move(rsqs),
                n_cds, n_lmdas,
                std::move(expected_betas),
                std::move(expected_objs));
    }

    template <class GenerateFType>
    void test(GenerateFType generate_dataset)
    {
        auto input = make_input(generate_dataset);
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
};

#ifndef TEST_LASSO
#define TEST_LASSO(n) \
    []() { \
        return tools::generate_dataset("lasso_" STRINGIFY(n));\
    }
#endif

TEST_F(LassoFixture, lasso_n_ge_p_full)
{
    test(TEST_LASSO(1));
}

TEST_F(LassoFixture, lasso_n_ge_p_partial)
{
    test(TEST_LASSO(2));
}

TEST_F(LassoFixture, lasso_n_le_p_full)
{
    test(TEST_LASSO(3));
}

TEST_F(LassoFixture, lasso_n_le_p_partial)
{
    test(TEST_LASSO(4));
}

TEST_F(LassoFixture, lasso_p_large)
{
    test(TEST_LASSO(5));
}

#undef TEST_LASSO

}
} // namespace ghostbasil
