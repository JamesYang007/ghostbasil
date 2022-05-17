#include <gtest/gtest.h>
#include <ghostbasil/lasso.hpp>
#include <testutil/basil_generate_data.hpp>

namespace ghostbasil {
namespace {
    
constexpr double tol = 1e-8;
constexpr double thr = 1e-16;
constexpr size_t max_cds = 1000;

auto make_basil_output()
{
    return std::make_tuple(
            std::vector<Eigen::SparseMatrix<double>>(),
            std::vector<Eigen::VectorXd>());
}

template <class GenerateFType>
void test_fit_basil(
        GenerateFType generate_dataset)
{
    auto dataset = generate_dataset();
    auto& A = std::get<0>(dataset);
    auto& r = std::get<1>(dataset);
    auto& s = std::get<2>(dataset);
    auto& expected_lmdas = std::get<3>(dataset);
    auto& expected_betas = std::get<4>(dataset);
    auto& expected_objs = std::get<5>(dataset);

    auto output = make_basil_output();
    auto& betas = std::get<0>(output);
    auto& lmdas = std::get<1>(output);

    size_t n_knockoffs = 0;
    size_t n_lambdas = 3;
    size_t n_lambdas_iter = 2;
    size_t strong_size = 1;
    size_t delta_strong_size = 1;
    size_t n_iters = 1000;

    try {
        fit_basil(A, r, s, n_knockoffs, n_lambdas, n_lambdas_iter,
                  strong_size, delta_strong_size, n_iters, max_cds, thr,
                  betas, lmdas);
#ifdef MAKE_LMDAS
        for (size_t i = 0; i < lmdas.size(); ++i) {
            PRINT(lmdas[i]);
        }
#endif
    }
    catch (const max_basil_iters_error& e) {
        std::cerr << e.what() << std::endl;
    }

    size_t pos = 0;
    size_t n_lmdas_total = 0;
    for (size_t i = 0; i < lmdas.size(); ++i) {
        auto& betas_i = betas[i];
        auto& lmdas_i = lmdas[i];

        n_lmdas_total += lmdas_i.size();
        
        EXPECT_EQ(betas_i.cols(), lmdas_i.size());

        for (size_t j = 0; j < lmdas_i.size(); ++j, ++pos) {
            EXPECT_DOUBLE_EQ(lmdas_i[j], expected_lmdas[pos]);

            auto actual = betas_i.col(j);
            auto expected = expected_betas.col(pos);

            EXPECT_NEAR(expected_objs[pos], objective(A, r, s, lmdas_i[j], actual), tol);
            EXPECT_EQ(actual.size(), expected.size());
            for (size_t i = 0; i < expected.size(); ++i) {
                EXPECT_NEAR(expected[i], actual.coeff(i), tol);
            }
        }
   }
    EXPECT_EQ(expected_lmdas.size(), n_lmdas_total);
    EXPECT_EQ(expected_betas.cols(), n_lmdas_total);
    EXPECT_EQ(expected_objs.size(), n_lmdas_total);
}


TEST(FitBasil, fit_basil_n_ge_p)
{
    test_fit_basil(generate_dataset_1);
}

TEST(FitBasil, fit_basil_n_le_p)
{
    test_fit_basil(generate_dataset_2);
}
    
} // namespace
} // namespace ghostbasil
