#include <gtest/gtest.h>
#include <ghostbasil/matrix/group_ghost_matrix.hpp>
#include <tools/matrix/group_ghost_matrix.hpp>

namespace ghostbasil {
namespace {

static constexpr double tol = 1e-14; 

struct GroupGhostMatrixFixture
    : ::testing::Test,
      tools::GroupGhostMatrixUtil
{
    using gmat_t = GroupGhostMatrix<mat_t>;
};

// ========================================================
// TEST Constructor
// ========================================================

TEST_F(GroupGhostMatrixFixture, ctor_empty)
{
    mat_t m, d;
    EXPECT_THROW(gmat_t gm(m, d, 2), std::runtime_error); 
}

TEST_F(GroupGhostMatrixFixture, ctor_less_than_2_groups)
{
    mat_t m(1,1);
    mat_t d(1,1);
    EXPECT_THROW(gmat_t gm(m, d, 0), std::runtime_error); 
    EXPECT_THROW(gmat_t gm(m, d, 1), std::runtime_error); 
}

TEST_F(GroupGhostMatrixFixture, ctor_diff_S_D_size)
{
    mat_t m(2,2);
    mat_t d(3,3);
    EXPECT_THROW(gmat_t gm(m, d, 2), std::runtime_error); 
}

TEST_F(GroupGhostMatrixFixture, ctor_S_not_square)
{
    mat_t m(2,3);
    mat_t d(2,2);
    EXPECT_THROW(gmat_t gm(m, d, 2), std::runtime_error); 
}

TEST_F(GroupGhostMatrixFixture, ctor_valid)
{
    mat_t m(3,3);
    mat_t d(3,3);
    EXPECT_NO_THROW(gmat_t gm(m, d, 2)); 
}

// ========================================================
// TEST col_dot
// ========================================================

struct GroupGhostMatrixColDotFixture
    : GroupGhostMatrixFixture,
      testing::WithParamInterface<
        std::tuple<size_t, size_t, size_t> >
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

TEST_P(GroupGhostMatrixColDotFixture, col_dot)
{
    auto param = GetParam();
    size_t seed = std::get<0>(param);
    size_t p = std::get<1>(param);
    size_t n_groups = std::get<2>(param);

    auto out = generate_data(seed, p, n_groups);
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
        GroupGhostMatrixColDotSuite,
        GroupGhostMatrixColDotFixture,
        testing::Values(
            std::make_tuple(0, 4, 2),
            std::make_tuple(124, 3, 3),
            std::make_tuple(321, 1, 3),
            std::make_tuple(9382, 9, 2))
    );

// ========================================================
// TEST quad_form
// ========================================================

struct GroupGhostMatrixQuadFormFixture
    : GroupGhostMatrixFixture,
      testing::WithParamInterface<
        std::tuple<size_t, size_t, size_t> >
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

TEST_P(GroupGhostMatrixQuadFormFixture, quad_form)
{
    auto param = GetParam();
    size_t seed = std::get<0>(param);
    size_t p = std::get<1>(param);
    size_t n_groups = std::get<2>(param);

    auto out = generate_data(seed, p, n_groups);
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
        GroupGhostMatrixQuadFormSuite,
        GroupGhostMatrixQuadFormFixture,
        testing::Values(
            std::make_tuple(0, 4, 2),
            std::make_tuple(124, 3, 2),
            std::make_tuple(321, 1, 3),
            std::make_tuple(9382, 9, 2))
    );

// ========================================================
// TEST coeff
// ========================================================

struct GroupGhostMatrixCoeffFixture
    : GroupGhostMatrixFixture,
      testing::WithParamInterface<
        std::tuple<size_t, size_t, size_t> >
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

TEST_P(GroupGhostMatrixCoeffFixture, coeff)
{
    auto param = GetParam();
    size_t seed = std::get<0>(param);
    size_t p = std::get<1>(param);
    size_t n_groups = std::get<2>(param);

    auto out = generate_data(seed, p, n_groups);
    auto& S = std::get<0>(out);
    auto& D = std::get<1>(out);
    auto& dense = std::get<4>(out);

    gmat_t gmat(S, D, n_groups);

    test(gmat, dense);
    test(gmat, dense);
}

INSTANTIATE_TEST_SUITE_P(
        GroupGhostMatrixCoeffSuite,
        GroupGhostMatrixCoeffFixture,
        testing::Values(
            std::make_tuple(0, 4, 2),
            std::make_tuple(124, 3, 2),
            std::make_tuple(321, 1, 3),
            std::make_tuple(9382, 9, 2))
    );

} 
} // namespace ghostbasil
