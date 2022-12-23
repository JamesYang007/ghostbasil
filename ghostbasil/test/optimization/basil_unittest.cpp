#include <gtest/gtest.h>
#include <ghostbasil/optimization/basil.hpp>
#include <ghostbasil/matrix/block_matrix.hpp>
#include <ghostbasil/matrix/ghost_matrix.hpp>
#include <tools/data_util.hpp>
#include <tools/macros.hpp>
#include <tools/matrix/block_matrix.hpp>
#include <tools/matrix/ghost_matrix.hpp>
#include <thread>

namespace ghostbasil {
namespace lasso {
namespace {

struct BasilFixture
    : ::testing::Test
{
    const double tol = 2e-8;
    const double thr = 1e-24;
    const size_t max_cds = 10000;
    const size_t max_n_lambdas = 3;
    const size_t n_lambdas_iter = 2;
    const size_t n_threads = std::thread::hardware_concurrency();

    template <class F>
    auto make_input(
            F generate_dataset,
            size_t max_strong_size,
            bool do_user)
    {
        auto dataset = generate_dataset();
        auto& A = std::get<0>(dataset);
        auto& r = std::get<1>(dataset);
        auto& alpha = std::get<2>(dataset);
        auto& penalty = std::get<3>(dataset);
        auto& expected_lmdas = std::get<5>(dataset);
        auto& expected_betas = std::get<6>(dataset);
        auto& expected_objs = std::get<7>(dataset);

        util::vec_type<double> user_lmdas;
        size_t max_strong_size_local = max_strong_size;
        if (do_user) {
            user_lmdas.resize(expected_lmdas.size());
            Eigen::Map<Eigen::VectorXd>(user_lmdas.data(), user_lmdas.size())
                = expected_lmdas;
            max_strong_size_local = r.size(); // force full strong set
        }
        
        std::vector<Eigen::SparseVector<double>> betas;
        std::vector<double> lmdas;
        std::vector<double> rsqs;
        return std::make_tuple(
                A, r, alpha, penalty, user_lmdas, max_strong_size_local,
                betas, lmdas, rsqs,
                expected_betas, expected_lmdas, expected_objs);
    }

    template <class GenerateFType>
    void test(
            GenerateFType generate_dataset,
            size_t max_strong_size = 100,
            size_t delta_strong_size = 1,
            double min_ratio = 7e-1,
            bool do_user = false)
    {
        auto input = make_input(generate_dataset, max_strong_size, do_user);
        auto& A = std::get<0>(input);
        auto& r = std::get<1>(input);
        auto& alpha = std::get<2>(input);
        auto& penalty = std::get<3>(input);
        auto& user_lmdas = std::get<4>(input);
        max_strong_size = std::get<5>(input);
        auto& betas = std::get<6>(input);
        auto& lmdas = std::get<7>(input);
        auto& rsqs = std::get<8>(input);
        auto& expected_betas = std::get<9>(input);
        auto& expected_lmdas = std::get<10>(input);
        auto& expected_objs = std::get<11>(input);

        try {
            basil(A, r, alpha, penalty, user_lmdas, max_n_lambdas, n_lambdas_iter,
                  delta_strong_size, max_strong_size, max_cds, thr, 
                  min_ratio, n_threads,
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
            EXPECT_TRUE(false);
            return;
        }

        EXPECT_EQ(expected_lmdas.size(), expected_objs.size());
        EXPECT_EQ(betas.size(), lmdas.size());

        for (size_t i = 0; i < lmdas.size(); ++i) {
            const auto& betas_i = betas[i];
            const auto& lmdas_i = lmdas[i];
            auto expected = expected_betas.col(i);

            EXPECT_DOUBLE_EQ(lmdas_i, expected_lmdas[i]);
            const auto obj = objective(A, r, penalty, alpha, lmdas_i, betas_i);
            EXPECT_NEAR(expected_objs[i], obj, tol);
            EXPECT_EQ(betas_i.size(), expected.size());
            for (size_t i = 0; i < expected.size(); ++i) {
                EXPECT_NEAR(expected[i], betas_i.coeff(i), tol);
            }
        }
        EXPECT_EQ(expected_lmdas.size(), lmdas.size());
        EXPECT_EQ(expected_betas.cols(), lmdas.size());
        EXPECT_EQ(expected_objs.size(), lmdas.size());
    }
};

#ifndef TEST_BASIL_FN
#define TEST_BASIL_FN(n) \
    []() { \
        return tools::generate_dataset("basil_" STRINGIFY(n));\
    }
#endif

TEST_F(BasilFixture, basil_n_ge_p)
{
    test(TEST_BASIL_FN(1));
}

TEST_F(BasilFixture, basil_n_le_p)
{
    test(TEST_BASIL_FN(2));
}

TEST_F(BasilFixture, basil_p_large)
{
    size_t max_strong_size = 1000;
    size_t delta_strong_size = 50;
    test(TEST_BASIL_FN(3),
            max_strong_size,
            delta_strong_size);
}

/*
 * This fixture contains common routines used to compare
 * other matrix with dense matrix.
 */
struct BasilCompareFixture
    : BasilFixture
{
    template <class F>
    auto make_input(F generate_dataset)
    {
        auto dataset = generate_dataset();
        auto&& A = std::get<0>(dataset);
        auto&& r = std::get<1>(dataset);
        auto&& alpha = std::get<2>(dataset);
        auto&& penalty = std::get<3>(dataset);
        auto&& delta_strong_size = std::get<4>(dataset);
        auto&& max_strong_size = std::get<5>(dataset);
        auto&& min_ratio = std::get<6>(dataset);

        util::vec_type<double> user_lmdas;
        std::vector<Eigen::SparseVector<double>> betas;
        std::vector<double> lmdas;
        std::vector<double> rsqs;

        return std::make_tuple(
                A, r, alpha, penalty, user_lmdas, delta_strong_size, 
                max_strong_size, min_ratio,
                betas, lmdas, rsqs);
    }

    template <class DataSetType>
    auto run(const DataSetType& dataset)
    {
        auto&& A = std::get<0>(dataset);
        auto&& r = std::get<1>(dataset);
        auto&& alpha = std::get<2>(dataset);
        auto&& penalty = std::get<3>(dataset);
        auto&& user_lmdas = std::get<4>(dataset);
        auto&& delta_strong_size = std::get<5>(dataset);
        auto&& max_strong_size = std::get<6>(dataset);
        auto&& min_ratio = std::get<7>(dataset);
        auto betas = std::get<8>(dataset);
        auto lmdas = std::get<9>(dataset);
        auto rsqs = std::get<10>(dataset);

        basil(A, r, alpha, penalty, user_lmdas, max_n_lambdas, n_lambdas_iter,
              delta_strong_size, max_strong_size, max_cds, thr, 
              min_ratio, n_threads,
              betas, lmdas, rsqs);
        return std::make_tuple(betas, lmdas, rsqs);
    }

    template <class F1, class F2>
    void test(F1 generate_dataset_other,
              F2 generate_dataset_dense)
    {
        auto&& actual_dataset = make_input(generate_dataset_other);
        auto&& expected_dataset = make_input(generate_dataset_dense);

        auto&& expected = run(expected_dataset);
        auto&& actual = run(actual_dataset);

        auto&& expected_betas = std::get<0>(expected);
        auto&& expected_lmdas = std::get<1>(expected);
        auto&& expected_rsqs = std::get<2>(expected);
        auto&& actual_betas = std::get<0>(actual);
        auto&& actual_lmdas = std::get<1>(actual);
        auto&& actual_rsqs = std::get<2>(actual);
        
        EXPECT_EQ(expected_betas.size(), expected_lmdas.size());
        EXPECT_EQ(expected_betas.size(), expected_rsqs.size());

        EXPECT_EQ(actual_betas.size(), expected_betas.size());
        EXPECT_EQ(actual_lmdas.size(), expected_lmdas.size());
        EXPECT_EQ(actual_rsqs.size(), expected_rsqs.size());
        for (size_t i = 0; i < expected_lmdas.size(); ++i) {
            const auto& expected_beta_i = expected_betas[i];
            const auto& actual_beta_i = actual_betas[i];
            EXPECT_EQ(actual_beta_i.size(), expected_beta_i.size());
            for (size_t j = 0; j < expected_beta_i.size(); ++j) {
                EXPECT_NEAR(actual_beta_i.coeff(j), expected_beta_i.coeff(j), 1e-13);
            }
            EXPECT_NEAR(actual_rsqs[i], expected_rsqs[i], 1e-13);
            EXPECT_NEAR(actual_lmdas[i], expected_lmdas[i], 1e-13);
        }
    }
};

// ================================================================
// TEST Block<Dense> vs. Dense
// ================================================================

struct BasilBlockFixture
    : BasilCompareFixture,
      tools::BlockMatrixUtil,
      ::testing::WithParamInterface<
        std::tuple<size_t, size_t, size_t> >
{
    using butil = tools::BlockMatrixUtil;
    using value_t = double;
    using mat_t = util::mat_type<value_t>;
    using bmat_t = BlockMatrix<mat_t>;

    // Generates a block matrix and a corresponding dense matrix
    // and other data that make_input needs.
    auto generate(
            size_t seed,
            size_t L,
            size_t p,
            size_t delta_strong_size = 3,
            size_t max_strong_size = 100,
            double min_ratio = 7e-1)
    {
        auto&& out = butil::generate_data(seed, L, p, 0, true, false);
        auto&& mat_list = std::get<0>(out);

        auto&& A_dense = std::get<3>(out);
        bmat_t A(mat_list); 

        std::mt19937 gen(seed);
        std::normal_distribution<> norm(0., 1.);
        size_t n_cols = A.cols();
        Eigen::VectorXd beta(n_cols); 
        beta.setZero();
        std::uniform_int_distribution<> unif(0, n_cols-1);
        for (size_t k = 0; k < 10; ++k) {
            beta[unif(gen)] = norm(gen);
        }

        Eigen::VectorXd r = A_dense * beta + Eigen::VectorXd::NullaryExpr(n_cols,
                [&](auto) { return 0.2 * norm(gen); });

        value_t alpha = 1;
        Eigen::VectorXd penalty(n_cols);
        penalty.setOnes();

        return std::make_tuple(
                std::move(mat_list), // must return also since A references it
                std::move(A),
                std::move(A_dense),
                std::move(r),
                std::move(alpha),
                std::move(penalty),
                delta_strong_size,
                max_strong_size,
                min_ratio);
    }

    template <class DatasetType>
    auto generate_datasets(const DatasetType& dataset)
    {
        auto generate_actual_pack = [&]() {
            return std::make_tuple(
                std::get<1>(dataset), // A (block)
                std::get<3>(dataset), // r
                std::get<4>(dataset), // alpha
                std::get<5>(dataset), // penalty
                std::get<6>(dataset),
                std::get<7>(dataset),
                std::get<8>(dataset)
                );
        };
        auto generate_expected_pack = [&]() {
            return std::make_tuple(
                std::get<2>(dataset), // A_dense
                std::get<3>(dataset), // r
                std::get<4>(dataset), // alpha
                std::get<5>(dataset), // penalty
                std::get<6>(dataset),
                std::get<7>(dataset),
                std::get<8>(dataset)
                );
        };
        return std::make_tuple(generate_actual_pack, generate_expected_pack);
    }
};

TEST_P(BasilBlockFixture, basil_block)
{
    size_t seed;
    size_t L;
    size_t p;
    std::tie(seed, L, p) = GetParam();
    auto&& dataset = generate(seed, L, p);
    auto fs = generate_datasets(dataset);
    test(std::get<0>(fs), std::get<1>(fs));
}

INSTANTIATE_TEST_SUITE_P(
        BasilBlockSuite,
        BasilBlockFixture,
        testing::Values(
            std::make_tuple(0,      1, 2),
            std::make_tuple(0,      2, 2),
            std::make_tuple(124,    3, 10),
            std::make_tuple(321,    5, 20),
            std::make_tuple(9382,   10, 7),
            std::make_tuple(3,      20, 3),
            std::make_tuple(6,      4, 20)
            )
    );

// ================================================================
// TEST Ghost<Dense, Dense> vs. Dense
// ================================================================

struct BasilGhostFixture
    : BasilCompareFixture,
      tools::GhostMatrixUtil,
      ::testing::WithParamInterface<
        std::tuple<size_t, size_t, size_t> >
{
    using gutil = tools::GhostMatrixUtil;
    using value_t = double;
    using mat_t = util::mat_type<value_t>;
    using vec_t = util::vec_type<value_t>;
    using gmat_t = GhostMatrix<mat_t, vec_t>;

    // Generates a ghost matrix and a corresponding dense matrix
    // and other data that make_input needs.
    auto generate(
            size_t seed,
            size_t p,
            size_t n_groups,
            size_t delta_strong_size = 3,
            size_t max_strong_size = 100,
            double min_ratio = 7e-1)
    {
        auto&& out = gutil::generate_data(seed, p, n_groups, 0, true, false);
        auto&& mat = std::get<0>(out);
        auto&& vec = std::get<1>(out);
        auto&& A_dense = std::get<4>(out);
        gmat_t A(mat, vec, n_groups); 

        std::mt19937 gen(seed);
        std::normal_distribution<> norm(0., 1.);
        size_t n_cols = A.cols();
        Eigen::VectorXd beta(n_cols); 
        beta.setZero();
        std::uniform_int_distribution<> unif(0, n_cols-1);
        for (size_t k = 0; k < 10; ++k) {
            beta[unif(gen)] = norm(gen);
        }

        Eigen::VectorXd r = A_dense * beta + Eigen::VectorXd::NullaryExpr(n_cols,
                [&](auto) { return 0.2 * norm(gen); });

        value_t alpha = 1;
        Eigen::VectorXd penalty(n_cols);
        penalty.setOnes();

        return std::make_tuple(
                std::move(mat), // must return also since A references it
                std::move(vec), // must return also since A references it
                std::move(A),
                std::move(A_dense),
                std::move(r),
                std::move(alpha),
                std::move(penalty),
                delta_strong_size,
                max_strong_size,
                min_ratio);
    }

    template <class DatasetType>
    auto generate_datasets(const DatasetType& dataset)
    {
        auto generate_actual_pack = [&]() {
            return std::make_tuple(
                std::get<2>(dataset), // A (ghost)
                std::get<4>(dataset), // r
                std::get<5>(dataset), // alpha
                std::get<6>(dataset), // penalty
                std::get<7>(dataset),
                std::get<8>(dataset),
                std::get<9>(dataset)
                );
        };
        auto generate_expected_pack = [&]() {
            return std::make_tuple(
                std::get<3>(dataset), // A_dense
                std::get<4>(dataset), // r
                std::get<5>(dataset), // alpha
                std::get<6>(dataset), // penalty
                std::get<7>(dataset),
                std::get<8>(dataset),
                std::get<9>(dataset)
                );
        };
        return std::make_tuple(generate_actual_pack, generate_expected_pack);
    }
};

TEST_P(BasilGhostFixture, basil_ghost)
{
    size_t seed;
    size_t p;
    size_t n_groups;
    std::tie(seed, p, n_groups) = GetParam();
    auto&& dataset = generate(seed, p, n_groups);
    auto fs = generate_datasets(dataset);
    test(std::get<0>(fs), std::get<1>(fs));
}

INSTANTIATE_TEST_SUITE_P(
        BasilGhostSuite,
        BasilGhostFixture,
        testing::Values(
            std::make_tuple(0,      2, 2),
            std::make_tuple(124,    3, 3),
            std::make_tuple(321,    5, 4),
            std::make_tuple(9382,   10, 2),
            std::make_tuple(3,      20, 5),
            std::make_tuple(6,      4, 10)
            )
    );

// ================================================================
// TEST Block<Ghost<Dense, Dense>> vs. Dense
// ================================================================

struct BasilBlockGhostFixture
    : BasilCompareFixture,
      tools::BlockMatrixUtil,
      tools::GhostMatrixUtil,
      ::testing::WithParamInterface<
        std::tuple<size_t, size_t, size_t, size_t> >
{
    using gutil = tools::GhostMatrixUtil;
    using butil = tools::BlockMatrixUtil;
    using value_t = double;
    using mat_t = util::mat_type<value_t>;
    using vec_t = util::vec_type<value_t>;
    using gmat_t = GhostMatrix<mat_t, vec_t>;
    using bmat_t = BlockMatrix<gmat_t>;

    // Generates a block matrix and a corresponding dense matrix
    // and other data that make_input needs.
    auto generate(
            size_t seed,
            size_t L,
            size_t p,
            size_t n_groups,
            size_t delta_strong_size = 3,
            size_t max_strong_size = 100,
            double min_ratio = 7e-1)
    {
        std::vector<mat_t> mat_list(L);
        std::vector<vec_t> vec_list(L);
        std::vector<gmat_t> gmat_list;
        std::vector<mat_t> dense_list(L);
        for (size_t i = 0; i < L; ++i) {
            auto&& out = gutil::generate_data(seed, p, n_groups, 0, true, false);
            mat_list[i] = std::move(std::get<0>(out));
            vec_list[i] = std::move(std::get<1>(out));
            gmat_list.emplace_back(mat_list[i], vec_list[i], n_groups);
            dense_list[i] = gutil::make_dense(mat_list[i], vec_list[i], n_groups);
        }
        bmat_t A(gmat_list); 
        mat_t A_dense = butil::make_dense(dense_list);

        std::mt19937 gen(seed);
        std::normal_distribution<> norm(0., 1.);
        size_t n_cols = A.cols();
        Eigen::VectorXd beta(n_cols); 
        beta.setZero();
        std::uniform_int_distribution<> unif(0, n_cols-1);
        for (size_t k = 0; k < 10; ++k) {
            beta[unif(gen)] = norm(gen);
        }

        Eigen::VectorXd r = A_dense * beta + Eigen::VectorXd::NullaryExpr(n_cols,
                [&](auto) { return 0.2 * norm(gen); });

        value_t alpha = 1;
        Eigen::VectorXd penalty(n_cols);
        penalty.setOnes();

        return std::make_tuple(
                std::move(mat_list), // must return also since A references it
                std::move(vec_list), // must return also since A references it
                std::move(gmat_list), // must return also since A references it
                std::move(A),
                std::move(A_dense),
                std::move(r),
                std::move(alpha),
                std::move(penalty),
                delta_strong_size,
                max_strong_size,
                min_ratio);
    }

    template <class DatasetType>
    auto generate_datasets(const DatasetType& dataset)
    {
        auto generate_actual_pack = [&]() {
            return std::make_tuple(
                std::get<3>(dataset), // A (block)
                std::get<5>(dataset), // r
                std::get<6>(dataset), // alpha
                std::get<7>(dataset), // penalty
                std::get<8>(dataset),
                std::get<9>(dataset),
                std::get<10>(dataset)
                );
        };
        auto generate_expected_pack = [&]() {
            return std::make_tuple(
                std::get<4>(dataset), // A (dense)
                std::get<5>(dataset), // r
                std::get<6>(dataset), // alpha
                std::get<7>(dataset), // penalty
                std::get<8>(dataset),
                std::get<9>(dataset),
                std::get<10>(dataset)
                );
        };
        return std::make_tuple(generate_actual_pack, generate_expected_pack);
    }
};

TEST_P(BasilBlockGhostFixture, basil_block_ghost)
{
    size_t seed;
    size_t L;
    size_t p;
    size_t n_groups;
    std::tie(seed, L, p, n_groups) = GetParam();
    auto&& dataset = generate(seed, L, p, n_groups);
    auto fs = generate_datasets(dataset);
    test(std::get<0>(fs), std::get<1>(fs));
}

INSTANTIATE_TEST_SUITE_P(
        BasilBlockGhostSuite,
        BasilBlockGhostFixture,
        testing::Values(
            std::make_tuple(0,      2, 2, 2),
            std::make_tuple(124,    3, 3, 3),
            std::make_tuple(321,    4, 5, 4),
            std::make_tuple(9382,   2, 10, 2),
            std::make_tuple(3,      2, 20, 5),
            std::make_tuple(6,      2, 4, 10)
            )
    );
    
} // namespace
} // namespace lasso
} // namespace ghostbasil
