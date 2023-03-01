#include <gtest/gtest.h>
#include <ghostbasil/optimization/lasso.hpp>
#include <ghostbasil/matrix/block_matrix.hpp>
#include <ghostbasil/matrix/ghost_matrix.hpp>
#include <ghostbasil/matrix/group_ghost_matrix.hpp>
#include <ghostbasil/matrix/block_group_ghost_matrix.hpp>
#include <tools/data_util.hpp>
#include <tools/macros.hpp>
#include <tools/matrix/block_matrix.hpp>
#include <tools/matrix/ghost_matrix.hpp>
#include <tools/matrix/group_ghost_matrix.hpp>
#include <tools/matrix/block_group_ghost_matrix.hpp>

namespace ghostbasil {
namespace lasso {
namespace {

struct LassoFixture
    : ::testing::Test
{
    const double tol = 1e-8;
    const double thr = 1e-24;
    const size_t max_cds = 10000;

    template <class F>
    auto run(F generate_dataset)
    {
        auto dataset = generate_dataset();
        auto&& A = std::get<0>(dataset);
        auto&& r = std::get<1>(dataset);
        auto&& alpha = std::get<2>(dataset);
        auto&& penalty = std::get<3>(dataset);
        auto&& strong_set = std::get<4>(dataset);
        auto&& lmdas = std::get<5>(dataset);
        auto&& expected_betas = std::get<6>(dataset);
        auto&& expected_objs = std::get<7>(dataset);

        std::vector<int> strong_order(strong_set.size());
        std::iota(strong_order.begin(), strong_order.end(), 0);
        std::sort(strong_order.begin(), strong_order.end(),
                  [&](auto x, auto y) { return strong_set[x] < strong_set[y]; });

        std::vector<double> strong_grad(strong_set.size());
        for (size_t i = 0; i < strong_grad.size(); ++i) {
            strong_grad[i] = r[strong_set[i]];
        }

        util::vec_type<double> strong_beta(strong_set.size());
        strong_beta.setZero();
        
        util::vec_type<double> strong_A_diag(strong_set.size());
        for (int i = 0; i < strong_A_diag.size(); ++i) {
            auto k = strong_set[i];
            strong_A_diag[i] = A.coeff(k,k);
        }

        using sp_vec_value_t = Eigen::SparseVector<double>;
        std::vector<sp_vec_value_t> betas(lmdas.size());
        std::vector<int> active_set;
        std::vector<int> active_order;
        std::vector<int> active_set_ordered;
        std::vector<int> is_active(strong_set.size(), false);
        std::vector<double> rsqs(lmdas.size());
        size_t n_cds = 0;
        size_t n_lmdas = 0;
        double rsq = 0;

        LassoParamPack<
            std::decay_t<decltype(A)>, double, int, int
        > pack(
            A, alpha, penalty, strong_set, strong_order, strong_A_diag,
            lmdas, max_cds, thr, rsq, strong_beta, strong_grad,
            active_set, active_order, active_set_ordered,
            is_active, betas, rsqs, n_cds, n_lmdas
        );

        fit(pack);

        return std::make_tuple(
            A, r, alpha, penalty, lmdas, betas, rsqs,
            n_cds, n_lmdas, expected_betas, expected_objs
        );
    }

    template <class F>
    void test(F generate_data)
    {
        const auto& pack = run(generate_data);

        const auto& A = std::get<0>(pack);
        const auto& r = std::get<1>(pack);
        const auto alpha = std::get<2>(pack);
        const auto& penalty = std::get<3>(pack);
        const auto& lmdas = std::get<4>(pack);
        const auto& betas = std::get<5>(pack);
        const auto n_cds = std::get<7>(pack);
        const auto n_lmdas = std::get<8>(pack);
        const auto& expected_betas = std::get<9>(pack);
        const auto& expected_objs = std::get<10>(pack);

        EXPECT_LE(n_cds, max_cds);

        EXPECT_EQ(betas.size(), lmdas.size());
        EXPECT_EQ(expected_betas.cols(), lmdas.size());
        EXPECT_EQ(expected_objs.size(), lmdas.size());
        EXPECT_LE(0, n_lmdas);
        EXPECT_LE(n_lmdas, lmdas.size());

        for (size_t i = 0; i < n_lmdas; ++i) {
            const auto& actual = betas[i];
            auto expected = expected_betas.col(i);

            const auto obj = objective(A, r, penalty, alpha, lmdas[i], actual);
            EXPECT_NEAR(expected_objs[i], obj, tol);
            EXPECT_EQ(actual.size(), expected.size());
            for (size_t i = 0; i < expected.size(); ++i) {
                EXPECT_NEAR(expected[i], actual.coeff(i), tol);
            }
        }
    }
};

#ifndef GENERATE_DATASET
#define GENERATE_DATASET(n) \
    []() { \
        return tools::generate_dataset("lasso_" STRINGIFY(n));\
    }
#endif

TEST_F(LassoFixture, lasso_n_ge_p_full)
{
    test(GENERATE_DATASET(1));
}

TEST_F(LassoFixture, lasso_n_ge_p_partial)
{
    test(GENERATE_DATASET(2));
}

TEST_F(LassoFixture, lasso_n_le_p_full)
{
    test(GENERATE_DATASET(3));
}

TEST_F(LassoFixture, lasso_n_le_p_partial)
{
    test(GENERATE_DATASET(4));
}

TEST_F(LassoFixture, lasso_p_large)
{
    test(GENERATE_DATASET(5));
}

// ================================================================
// TEST Block<Dense> vs. Dense
// ================================================================

struct LassoCompareFixture
    : LassoFixture
{
    template <class F1, class F2>
    void test(F1 generate_dataset_actual,
              F2 generate_dataset_expected,
              bool do_eq = true,
              double coeff_tol = 1e-12,
              double rsq_tol = 1e-12)
    {
        auto&& actual = run(generate_dataset_actual);
        auto&& expected = run(generate_dataset_expected);

        auto&& expected_betas = std::get<5>(expected);
        auto&& expected_rsqs = std::get<6>(expected);
        auto&& expected_n_cds = std::get<7>(expected);
        auto&& expected_n_lmdas = std::get<8>(expected);
        auto&& actual_betas = std::get<5>(actual);
        auto&& actual_rsqs = std::get<6>(actual);
        auto&& actual_n_cds = std::get<7>(actual);
        auto&& actual_n_lmdas = std::get<8>(actual);
        
        EXPECT_EQ(actual_betas.size(), expected_betas.size());
        EXPECT_EQ(actual_rsqs.size(), expected_rsqs.size());
        EXPECT_EQ(actual_n_lmdas, expected_n_lmdas);
        EXPECT_EQ(actual_n_cds, expected_n_cds);
        for (size_t i = 0; i < expected_n_lmdas; ++i) {
            auto& expected_beta_i = expected_betas[i];
            auto& actual_beta_i = actual_betas[i];
            EXPECT_EQ(actual_beta_i.size(), expected_beta_i.size());
            for (size_t j = 0; j < expected_beta_i.size(); ++j) {
                if (do_eq) {
                    EXPECT_DOUBLE_EQ(actual_beta_i.coeff(j), expected_beta_i.coeff(j));
                } else {
                    EXPECT_NEAR(actual_beta_i.coeff(j), expected_beta_i.coeff(j), coeff_tol);
                }
            }
            if (do_eq) {
                EXPECT_DOUBLE_EQ(actual_rsqs[i], expected_rsqs[i]);
            } else {
                EXPECT_NEAR(actual_rsqs[i], expected_rsqs[i], rsq_tol);
            }
        }
    }
};

struct LassoBlockFixture
    : LassoCompareFixture,
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
            size_t p)
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

        value_t alpha = 0.1;
        
        Eigen::VectorXd penalty(n_cols);
        penalty.setOnes();

        std::vector<int> strong_set(n_cols);
        std::iota(strong_set.begin(), strong_set.end(), 0);

        util::vec_type<value_t> lmdas(3);
        lmdas[0] = r.array().abs().maxCoeff();
        for (int i = 1; i < lmdas.size(); ++i) {
            lmdas[i] = lmdas[i-1] * 0.7;
        }

        return std::make_tuple(
                std::move(mat_list), // must return also since A references it
                std::move(A),
                std::move(A_dense),
                std::move(r),
                std::move(alpha),
                std::move(penalty),
                std::move(strong_set),
                std::move(lmdas),
                0, 0); // dummy variables for expected betas and objs
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
                std::get<6>(dataset), // strong_set
                std::get<7>(dataset), // lmdas
                std::get<8>(dataset), // dummy: expected_betas
                std::get<9>(dataset)  // dummy: expected_objs
                );
        };
        auto generate_expected_pack = [&]() {
            return std::make_tuple(
                std::get<2>(dataset), // A_dense
                std::get<3>(dataset), // r
                std::get<4>(dataset), // alpha
                std::get<5>(dataset), // penalty
                std::get<6>(dataset), // strong_set
                std::get<7>(dataset), // lmdas
                std::get<8>(dataset), // dummy: expected_betas
                std::get<9>(dataset)  // dummy: expected_objs
                );
        };
        return std::make_tuple(generate_actual_pack, generate_expected_pack);
    }
};

TEST_P(LassoBlockFixture, lasso_block)
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
        LassoBlockSuite,
        LassoBlockFixture,
        testing::Values(
            std::make_tuple(0,      1, 2),
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

struct LassoGhostFixture
    : LassoCompareFixture,
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
            size_t n_groups)
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

        value_t alpha = 0.1;
        
        Eigen::VectorXd penalty(n_cols);
        penalty.setOnes();

        std::vector<int> strong_set(n_cols);
        std::iota(strong_set.begin(), strong_set.end(), 0);

        util::vec_type<value_t> lmdas(3);
        lmdas[0] = r.array().abs().maxCoeff();
        for (int i = 1; i < lmdas.size(); ++i) {
            lmdas[i] = lmdas[i-1] * 0.7;
        }

        return std::make_tuple(
                std::move(mat), // must return also since A references it
                std::move(vec), // must return also since A references it
                std::move(A),
                std::move(A_dense),
                std::move(r),
                std::move(alpha),
                std::move(penalty),
                std::move(strong_set),
                std::move(lmdas)); 
    }

    template <class DatasetType>
    auto generate_datasets(const DatasetType& dataset)
    {
        auto generate_actual_pack = [&]() {
            return std::make_tuple(
                std::get<2>(dataset), // A (block)
                std::get<4>(dataset), // r
                std::get<5>(dataset), // alpha
                std::get<6>(dataset), // penalty
                std::get<7>(dataset), // strong_set
                std::get<8>(dataset), // lmdas
                0, // dummy: expected_betas
                0  // dummy: expected_objs
                );
        };
        auto generate_expected_pack = [&]() {
            return std::make_tuple(
                std::get<3>(dataset), // A (block)
                std::get<4>(dataset), // r
                std::get<5>(dataset), // alpha
                std::get<6>(dataset), // penalty
                std::get<7>(dataset), // strong_set
                std::get<8>(dataset), // lmdas
                0, // dummy: expected_betas
                0  // dummy: expected_objs
                );
        };
        return std::make_tuple(generate_actual_pack, generate_expected_pack);
    }
};

TEST_P(LassoGhostFixture, lasso_ghost)
{
    size_t seed;
    size_t p;
    size_t n_groups;
    std::tie(seed, p, n_groups) = GetParam();
    auto&& dataset = generate(seed, p, n_groups);
    auto fs = generate_datasets(dataset);
    test(std::get<0>(fs), std::get<1>(fs), false, 1e-12, 1e-12);
}

INSTANTIATE_TEST_SUITE_P(
        LassoGhostSuite,
        LassoGhostFixture,
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
// TEST BlockGroupGhost<Dense> vs. Dense
// ================================================================

struct LassoBlockGroupGhostFixture
    : LassoCompareFixture,
      tools::BlockGroupGhostMatrixUtil,
      ::testing::WithParamInterface<
        std::tuple<size_t, size_t, size_t, size_t> >
{
    using gutil = tools::BlockGroupGhostMatrixUtil;
    using value_t = double;
    using mat_t = util::mat_type<value_t>;
    using vec_t = util::vec_type<value_t>;
    using gmat_t = BlockGroupGhostMatrix<mat_t>;
    using bmat_t = BlockMatrix<mat_t>;

    // Generates a block group ghost matrix and a corresponding dense matrix
    // and other data that make_input needs.
    auto generate(
            size_t seed,
            size_t p,
            size_t n_blocks,
            size_t n_groups)
    {
        auto&& out = gutil::generate_data(seed, p, n_blocks, n_groups, 0, true, false, 0.01);
        auto&& S = std::get<0>(out);
        auto&& dl = std::get<1>(out);
        auto&& A_dense = std::get<4>(out);

        bmat_t D(dl);
        gmat_t A(S, D, n_groups); 

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

        value_t alpha = 0.1;
        
        Eigen::VectorXd penalty(n_cols);
        penalty.setOnes();

        std::vector<int> strong_set(n_cols);
        std::iota(strong_set.begin(), strong_set.end(), 0);

        util::vec_type<value_t> lmdas(3);
        lmdas[0] = r.array().abs().maxCoeff();
        for (int i = 1; i < lmdas.size(); ++i) {
            lmdas[i] = lmdas[i-1] * 0.7;
        }

        return std::make_tuple(
                std::move(S), // must return also since A references it
                std::move(dl), // must return also since A references it
                std::move(A),
                std::move(A_dense),
                std::move(r),
                std::move(alpha),
                std::move(penalty),
                std::move(strong_set),
                std::move(lmdas)); 
    }

    template <class DatasetType>
    auto generate_datasets(const DatasetType& dataset)
    {
        auto generate_actual_pack = [&]() {
            return std::make_tuple(
                std::get<2>(dataset), // A (block)
                std::get<4>(dataset), // r
                std::get<5>(dataset), // alpha
                std::get<6>(dataset), // penalty
                std::get<7>(dataset), // strong_set
                std::get<8>(dataset), // lmdas
                0, // dummy: expected_betas
                0  // dummy: expected_objs
                );
        };
        auto generate_expected_pack = [&]() {
            return std::make_tuple(
                std::get<3>(dataset), // A (block)
                std::get<4>(dataset), // r
                std::get<5>(dataset), // alpha
                std::get<6>(dataset), // penalty
                std::get<7>(dataset), // strong_set
                std::get<8>(dataset), // lmdas
                0, // dummy: expected_betas
                0  // dummy: expected_objs
                );
        };
        return std::make_tuple(generate_actual_pack, generate_expected_pack);
    }
};

TEST_P(LassoBlockGroupGhostFixture, lasso_block_group_ghost)
{
    size_t seed;
    size_t p;
    size_t n_blocks;
    size_t n_groups;
    std::tie(seed, p, n_blocks, n_groups) = GetParam();
    auto&& dataset = generate(seed, p, n_blocks, n_groups);
    auto fs = generate_datasets(dataset);
    test(std::get<0>(fs), std::get<1>(fs), false, 1e-12, 1e-12);
}

INSTANTIATE_TEST_SUITE_P(
        LassoBlockGroupGhostSuite,
        LassoBlockGroupGhostFixture,
        testing::Values(
            std::make_tuple(0,      2,   1, 2),
            std::make_tuple(124,    3,   3, 3),
            std::make_tuple(321,    5,   3, 4),
            std::make_tuple(9382,   10,  3, 2),
            std::make_tuple(3,      20,  3, 5),
            std::make_tuple(6,      6,  3, 10)
            )
    );

// ================================================================
// TEST GroupGhost<Dense> vs. Dense
// ================================================================

struct LassoGroupGhostFixture
    : LassoCompareFixture,
      tools::GroupGhostMatrixUtil,
      ::testing::WithParamInterface<
        std::tuple<size_t, size_t, size_t> >
{
    using gutil = tools::GroupGhostMatrixUtil;
    using value_t = double;
    using mat_t = util::mat_type<value_t>;
    using vec_t = util::vec_type<value_t>;
    using gmat_t = GroupGhostMatrix<mat_t>;

    // Generates a group ghost matrix and a corresponding dense matrix
    // and other data that make_input needs.
    auto generate(
            size_t seed,
            size_t p,
            size_t n_groups)
    {
        auto&& out = gutil::generate_data(seed, p, n_groups, 0, true, false);
        auto&& S = std::get<0>(out);
        auto&& D = std::get<1>(out);
        auto&& A_dense = std::get<4>(out);
        gmat_t A(S, D, n_groups); 

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

        value_t alpha = 0.1;
        
        Eigen::VectorXd penalty(n_cols);
        penalty.setOnes();

        std::vector<int> strong_set(n_cols);
        std::iota(strong_set.begin(), strong_set.end(), 0);

        util::vec_type<value_t> lmdas(3);
        lmdas[0] = r.array().abs().maxCoeff();
        for (int i = 1; i < lmdas.size(); ++i) {
            lmdas[i] = lmdas[i-1] * 0.7;
        }

        return std::make_tuple(
                std::move(S), // must return also since A references it
                std::move(D), // must return also since A references it
                std::move(A),
                std::move(A_dense),
                std::move(r),
                std::move(alpha),
                std::move(penalty),
                std::move(strong_set),
                std::move(lmdas)); 
    }

    template <class DatasetType>
    auto generate_datasets(const DatasetType& dataset)
    {
        auto generate_actual_pack = [&]() {
            return std::make_tuple(
                std::get<2>(dataset), // A (block)
                std::get<4>(dataset), // r
                std::get<5>(dataset), // alpha
                std::get<6>(dataset), // penalty
                std::get<7>(dataset), // strong_set
                std::get<8>(dataset), // lmdas
                0, // dummy: expected_betas
                0  // dummy: expected_objs
                );
        };
        auto generate_expected_pack = [&]() {
            return std::make_tuple(
                std::get<3>(dataset), // A (block)
                std::get<4>(dataset), // r
                std::get<5>(dataset), // alpha
                std::get<6>(dataset), // penalty
                std::get<7>(dataset), // strong_set
                std::get<8>(dataset), // lmdas
                0, // dummy: expected_betas
                0  // dummy: expected_objs
                );
        };
        return std::make_tuple(generate_actual_pack, generate_expected_pack);
    }
};

TEST_P(LassoGroupGhostFixture, lasso_group_ghost)
{
    size_t seed;
    size_t p;
    size_t n_groups;
    std::tie(seed, p, n_groups) = GetParam();
    auto&& dataset = generate(seed, p, n_groups);
    auto fs = generate_datasets(dataset);
    test(std::get<0>(fs), std::get<1>(fs), false, 1e-12, 1e-12);
}

INSTANTIATE_TEST_SUITE_P(
        LassoGroupGhostSuite,
        LassoGroupGhostFixture,
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

struct LassoBlockGhostFixture
    : LassoCompareFixture,
      tools::GhostMatrixUtil,
      tools::BlockMatrixUtil,
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

    // Generates a ghost matrix and a corresponding dense matrix
    // and other data that make_input needs.
    auto generate(
            size_t seed,
            size_t L,
            size_t p,
            size_t n_groups)
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

        value_t alpha = 0.1;
        
        Eigen::VectorXd penalty(n_cols);
        penalty.setOnes();

        std::vector<int> strong_set(n_cols);
        std::iota(strong_set.begin(), strong_set.end(), 0);

        util::vec_type<value_t> lmdas(3);
        lmdas[0] = r.array().abs().maxCoeff();
        for (int i = 1; i < lmdas.size(); ++i) {
            lmdas[i] = lmdas[i-1] * 0.7;
        }

        return std::make_tuple(
                std::move(mat_list), // must return also since A references it
                std::move(vec_list), // must return also since A references it
                std::move(gmat_list), // must return also since A references it
                std::move(A),
                std::move(A_dense),
                std::move(r),
                std::move(alpha),
                std::move(penalty),
                std::move(strong_set),
                std::move(lmdas)); 
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
                std::get<8>(dataset), // strong_set
                std::get<9>(dataset), // lmdas
                0, // dummy: expected_betas
                0  // dummy: expected_objs
                );
        };
        auto generate_expected_pack = [&]() {
            return std::make_tuple(
                std::get<4>(dataset), // A (dense)
                std::get<5>(dataset), // r
                std::get<6>(dataset), // alpha
                std::get<7>(dataset), // penalty
                std::get<8>(dataset), // strong_set
                std::get<9>(dataset), // lmdas
                0, // dummy: expected_betas
                0  // dummy: expected_objs
                );
        };
        return std::make_tuple(generate_actual_pack, generate_expected_pack);
    }
};

TEST_P(LassoBlockGhostFixture, lasso_block_ghost)
{
    size_t seed;
    size_t L;
    size_t p;
    size_t n_groups;
    std::tie(seed, L, p, n_groups) = GetParam();
    auto&& dataset = generate(seed, L, p, n_groups);
    auto fs = generate_datasets(dataset);
    test(std::get<0>(fs), std::get<1>(fs), false, 4e-15, 3e-14);
}

INSTANTIATE_TEST_SUITE_P(
        LassoBlockGhostSuite,
        LassoBlockGhostFixture,
        testing::Values(
            std::make_tuple(0,      2, 2, 2),
            std::make_tuple(124,    3, 3, 3),
            std::make_tuple(321,    4, 5, 4),
            std::make_tuple(9382,   2, 10, 2),
            std::make_tuple(3,      2, 20, 5),
            std::make_tuple(6,      2, 4, 10)
            )
    );

}
} // namespace lasso
} // namespace ghostbasil
  
#undef GENERATE_DATASET
