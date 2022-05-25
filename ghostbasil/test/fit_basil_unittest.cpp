#include <gtest/gtest.h>
#include <ghostbasil/lasso.hpp>
#include <testutil/fit_basil_util.hpp>
#include <thread>

namespace ghostbasil {
namespace {

using namespace fit_basil_util;
    
static constexpr double tol = 2e-8;
static constexpr double thr = 1e-16;
static constexpr size_t max_cds = 10000;
static constexpr size_t max_n_lambdas = 3;
static constexpr size_t n_lambdas_iter = 2;

template <class GenerateFType>
void test_fit_basil(
        GenerateFType generate_dataset,
        size_t max_strong_size = 100,
        size_t strong_size = 1,
        size_t delta_strong_size = 1,
        bool do_user = false)
{
    auto dataset = generate_dataset();
    auto& A = std::get<0>(dataset);
    auto& r = std::get<1>(dataset);
    auto& s = std::get<2>(dataset);
    auto& expected_lmdas = std::get<3>(dataset);
    auto& expected_betas = std::get<4>(dataset);
    auto& expected_objs = std::get<5>(dataset);
    size_t n_threads = std::thread::hardware_concurrency();

    std::vector<double> user_lmdas;
    size_t max_strong_size_local = max_strong_size;
    if (do_user) {
        user_lmdas.resize(expected_lmdas.size());
        Eigen::Map<Eigen::VectorXd>(user_lmdas.data(), user_lmdas.size())
            = expected_lmdas;
        max_strong_size_local = r.size(); // force full strong set
    }

    auto output = make_basil_output();
    auto& betas = std::get<0>(output);
    auto& lmdas = std::get<1>(output);

    try {
        fit_basil(A, r, s, user_lmdas, max_n_lambdas, n_lambdas_iter,
                  strong_size, delta_strong_size, max_strong_size_local, max_cds, thr, n_threads,
                  betas, lmdas);
#ifdef MAKE_LMDAS
        for (size_t i = 0; i < lmdas.size(); ++i) {
            PRINT(lmdas[i]);
        }
        return;
#endif
    }
    catch (const max_cds_error& e) {
        std::cerr << e.what() << std::endl;
        return;
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return;
    }

    EXPECT_EQ(expected_lmdas.size(), expected_objs.size());
    EXPECT_EQ(betas.size(), lmdas.size());

    size_t pos = 0;
    size_t n_lmdas_total = 0;
    for (size_t i = 0; i < lmdas.size(); ++i) {
        auto& betas_i = betas[i];
        auto& lmdas_i = lmdas[i];

        n_lmdas_total += lmdas_i.size();
        
        EXPECT_EQ(betas_i.cols(), lmdas_i.size());

        for (size_t j = 0; j < lmdas_i.size(); ++j, ++pos) {
            EXPECT_LT(pos, expected_lmdas.size());
            EXPECT_LT(pos, expected_betas.cols());
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

TEST(FitBasil, fit_basil_p_large)
{
    size_t max_strong_size = 1000;
    size_t strong_size = 100;
    size_t delta_strong_size = 50;
    test_fit_basil(generate_dataset_3,
            max_strong_size,
            strong_size,
            delta_strong_size);
}
    
} // namespace
} // namespace ghostbasil
