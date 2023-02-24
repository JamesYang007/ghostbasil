#include <gtest/gtest.h>
#include <ghostbasil/matrix/block_group_ghost_matrix.hpp>
#include <tools/matrix/block_group_ghost_matrix.hpp>

namespace ghostbasil {
namespace {

static constexpr double tol = 1e-14; 

struct BlockGroupGhostMatrixFixture
    : ::testing::Test,
      tools::BlockGroupGhostMatrixUtil
{
    using bmat_t = BlockMatrix<mat_t>;
    using gmat_t = BlockGroupGhostMatrix<mat_t>;
};

// ========================================================
// TEST Constructor
// ========================================================

TEST_F(BlockGroupGhostMatrixFixture, ctor_less_than_2_groups)
{
    mat_t m(1,1);
    std::vector<mat_t> dl(1);
    dl[0].resize(1,1);
    bmat_t d(dl);
    EXPECT_THROW(gmat_t gm(m, d, 0), std::runtime_error); 
    EXPECT_THROW(gmat_t gm(m, d, 1), std::runtime_error); 
}

TEST_F(BlockGroupGhostMatrixFixture, ctor_diff_S_D_size)
{
    mat_t m(2,2);
    std::vector<mat_t> dl(1);
    dl[0].resize(3,3);
    bmat_t d(dl);
    EXPECT_THROW(gmat_t gm(m, d, 2), std::runtime_error); 
}

TEST_F(BlockGroupGhostMatrixFixture, ctor_S_not_square)
{
    mat_t m(2,3);
    std::vector<mat_t> dl(1);
    dl[0].resize(2,2);
    bmat_t d(dl);
    EXPECT_THROW(gmat_t gm(m, d, 2), std::runtime_error); 
}

TEST_F(BlockGroupGhostMatrixFixture, ctor_valid)
{
    mat_t m(3,3);
    std::vector<mat_t> dl(1);
    dl[0].resize(3,3);
    bmat_t d(dl);
    EXPECT_NO_THROW(gmat_t gm(m, d, 2)); 
}

// ========================================================
// TEST col_dot
// ========================================================

struct BlockGroupGhostMatrixColDotFixture
    : BlockGroupGhostMatrixFixture,
      testing::WithParamInterface<
        std::tuple<size_t, size_t, size_t, size_t> >
{
    template <class VecType>
    void test(
            const gmat_t& gmat,
            const mat_t& dense,
            const VecType& v)
    {
        for (size_t i = 0; i < dense.cols(); ++i) {
            auto actual = gmat.col_dot(i, v);
            auto expected = v.dot(dense.col(i));
            EXPECT_NEAR(actual, expected, tol);
        }
    }
};

TEST_P(BlockGroupGhostMatrixColDotFixture, col_dot)
{
    auto param = GetParam();
    size_t seed = std::get<0>(param);
    size_t p = std::get<1>(param);
    size_t L = std::get<2>(param);
    size_t n_groups = std::get<3>(param);

    auto out = generate_data(seed, p, L, n_groups);
    auto& S = std::get<0>(out);
    auto& D = std::get<1>(out);
    auto& v = std::get<2>(out);
    auto& vs = std::get<3>(out);
    auto& dense = std::get<4>(out);

    gmat_t gmat(S, D, n_groups);

    test(gmat, dense, v);
    test(gmat, dense, vs);
}

INSTANTIATE_TEST_SUITE_P(
        BlockGroupGhostMatrixColDotSuite,
        BlockGroupGhostMatrixColDotFixture,
        testing::Values(
            std::make_tuple(0, 4, 1, 2),
            std::make_tuple(124, 3, 2, 2),
            std::make_tuple(321, 1, 1, 3),
            std::make_tuple(9382, 9, 4, 2))
    );

// ========================================================
// TEST quad_form
// ========================================================

struct BlockGroupGhostMatrixQuadFormFixture
    : BlockGroupGhostMatrixFixture,
      testing::WithParamInterface<
        std::tuple<size_t, size_t, size_t, size_t> >
{
    template <class VecType>
    void test(
            const gmat_t& gmat,
            const mat_t& dense,
            const VecType& v)
    {
        auto actual = gmat.quad_form(v);
        auto expected = v.dot(dense * v);
        EXPECT_NEAR(actual, expected, tol);
    }
};

TEST_P(BlockGroupGhostMatrixQuadFormFixture, quad_form)
{
    auto param = GetParam();
    size_t seed = std::get<0>(param);
    size_t p = std::get<1>(param);
    size_t L = std::get<2>(param);
    size_t n_groups = std::get<3>(param);

    auto out = generate_data(seed, p, L, n_groups);
    auto& S = std::get<0>(out);
    auto& D = std::get<1>(out);
    auto& v = std::get<2>(out);
    auto& vs = std::get<3>(out);
    auto& dense = std::get<4>(out);

    gmat_t gmat(S, D, n_groups);

    test(gmat, dense, v);
    test(gmat, dense, vs);
}

INSTANTIATE_TEST_SUITE_P(
        BlockGroupGhostMatrixQuadFormSuite,
        BlockGroupGhostMatrixQuadFormFixture,
        testing::Values(
            std::make_tuple(0, 4, 1, 2),
            std::make_tuple(124, 3, 2, 2),
            std::make_tuple(321, 1, 1, 3),
            std::make_tuple(9382, 9, 4, 2))
    );

// ========================================================
// TEST coeff
// ========================================================

struct BlockGroupGhostMatrixCoeffFixture
    : BlockGroupGhostMatrixFixture,
      testing::WithParamInterface<
        std::tuple<size_t, size_t, size_t, size_t> >
{
    void test(
            const gmat_t& gmat,
            const mat_t& dense)
    {
        EXPECT_EQ(gmat.rows(), dense.rows());
        EXPECT_EQ(gmat.cols(), dense.cols());

        for (size_t j = 0; j < gmat.cols(); ++j) {
            for (size_t i = 0; i < gmat.rows(); ++i) {
                auto actual = gmat.coeff(i, j);
                auto expected = dense.coeff(i, j);
                EXPECT_DOUBLE_EQ(actual, expected);
            }
        }
    }
};

TEST_P(BlockGroupGhostMatrixCoeffFixture, coeff)
{
    auto param = GetParam();
    size_t seed = std::get<0>(param);
    size_t p = std::get<1>(param);
    size_t L = std::get<2>(param);
    size_t n_groups = std::get<3>(param);

    auto out = generate_data(seed, p, L, n_groups);
    auto& S = std::get<0>(out);
    auto& D = std::get<1>(out);
    auto& dense = std::get<4>(out);

    gmat_t gmat(S, D, n_groups);

    test(gmat, dense);
    test(gmat, dense);
}

INSTANTIATE_TEST_SUITE_P(
        BlockGroupGhostMatrixCoeffSuite,
        BlockGroupGhostMatrixCoeffFixture,
        testing::Values(
            std::make_tuple(0, 4, 1, 2),
            std::make_tuple(124, 3, 2, 2),
            std::make_tuple(321, 1, 1, 3),
            std::make_tuple(9382, 9, 4, 2))
    );

} 
} // namespace ghostbasil
