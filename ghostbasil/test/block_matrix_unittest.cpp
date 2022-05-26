#include <gtest/gtest.h>
#include <testutil/block_matrix_util.hpp>

namespace ghostbasil {
namespace {

using namespace block_matrix_util;

static constexpr double tol = 1e-14; 

// ========================================================
// TEST Constructor
// ========================================================

TEST(BlockMatrixTest, ctor_empty_list)
{
    mat_list_t ml;
    EXPECT_NO_THROW(bmat_t m(ml)); 
}

TEST(BlockMatrixTest, ctor_m_not_square)
{
    mat_t m(2,3);
    mat_list_t ml(2, m);
    EXPECT_THROW(bmat_t m(ml), std::runtime_error); 
}

TEST(BlockMatrixTest, ctor_valid)
{
    mat_t m(2,2);
    mat_list_t ml(2, m);
    EXPECT_NO_THROW(bmat_t m(ml)); 
}

// ========================================================
// TEST col_dot
// ========================================================

struct BlockMatrixColDotFixture
    : testing::Test,
      testing::WithParamInterface<
        std::tuple<size_t, size_t, size_t> >
{};

template <class VecType>
static inline void test_col_dot(
        const bmat_t& bmat,
        const mat_t& dense,
        const VecType& v)
{
    for (size_t i = 0; i < dense.cols(); ++i) {
        auto actual = bmat.col_dot(i, v);
        auto expected = v.dot(dense.col(i));
        EXPECT_NEAR(actual, expected, tol);
    }
}

TEST_P(BlockMatrixColDotFixture, col_dot)
{
    auto param = GetParam();
    size_t seed = std::get<0>(param);
    size_t L = std::get<1>(param);
    size_t p = std::get<2>(param);

    auto out = generate_data(seed, L, p);
    auto& ml = std::get<0>(out);
    auto& v = std::get<1>(out);
    auto& vs = std::get<2>(out);
    auto& dense = std::get<3>(out);

    bmat_t bmat(ml);

    test_col_dot(bmat, dense, v);
    test_col_dot(bmat, dense, vs);
}

INSTANTIATE_TEST_SUITE_P(
        BlockMatrixColDotSuite,
        BlockMatrixColDotFixture,
        testing::Values(
            std::make_tuple(0, 1, 4),
            std::make_tuple(124, 2, 3),
            std::make_tuple(321, 5, 1),
            std::make_tuple(9382, 3, 9))
    );

// ========================================================
// TEST quad_form
// ========================================================

struct BlockMatrixQuadFormFixture
    : testing::Test,
      testing::WithParamInterface<
        std::tuple<size_t, size_t, size_t> >
{};

template <class VecType>
static inline void test_quad_form(
        const bmat_t& bmat,
        const mat_t& dense,
        const VecType& v)
{
    auto actual = bmat.quad_form(v);
    auto expected = v.dot(dense * v);
    EXPECT_NEAR(actual, expected, tol);
}

TEST_P(BlockMatrixQuadFormFixture, quad_form)
{
    auto param = GetParam();
    size_t seed = std::get<0>(param);
    size_t L = std::get<1>(param);
    size_t p = std::get<2>(param);

    auto out = generate_data(seed, L, p);
    auto& ml = std::get<0>(out);
    auto& v = std::get<1>(out);
    auto& vs = std::get<2>(out);
    auto& dense = std::get<3>(out);

    bmat_t bmat(ml);

    test_quad_form(bmat, dense, v);
    test_quad_form(bmat, dense, vs);
}

INSTANTIATE_TEST_SUITE_P(
        BlockMatrixQuadFormSuite,
        BlockMatrixQuadFormFixture,
        testing::Values(
            std::make_tuple(0, 1, 4),
            std::make_tuple(124, 2, 3),
            std::make_tuple(321, 5, 1),
            std::make_tuple(9382, 3, 9))
    );

// ========================================================
// TEST inv_quad_form
// ========================================================

struct BlockMatrixInvQuadFormFixture
    : testing::Test,
      testing::WithParamInterface<
        std::tuple<size_t, size_t, size_t, double> >
{};

template <class VecType>
static inline void test_inv_quad_form(
        const bmat_t& bmat,
        const mat_t& dense,
        double s,
        const VecType& v)
{
    auto actual = bmat.inv_quad_form(s, v);
    mat_t T = (1-s) * dense;
    T.diagonal().array() += s;
    Eigen::FullPivLU<mat_t> lu(T);
    vec_t vd = v;
    auto expected = v.dot(lu.solve(vd));
    EXPECT_NEAR(actual, expected, std::abs(expected * tol));
}

TEST_P(BlockMatrixInvQuadFormFixture, inv_quad_form)
{
    auto param = GetParam();
    size_t seed = std::get<0>(param);
    size_t L = std::get<1>(param);
    size_t p = std::get<2>(param);
    double s = std::get<3>(param);

    auto out = generate_data(seed, L, p);
    auto& ml = std::get<0>(out);
    auto& v = std::get<1>(out);
    auto& vs = std::get<2>(out);
    auto& dense = std::get<3>(out);

    bmat_t bmat(ml);

    test_inv_quad_form(bmat, dense, s, v);
    test_inv_quad_form(bmat, dense, s, vs);
}

INSTANTIATE_TEST_SUITE_P(
        BlockMatrixInvQuadFormSuite,
        BlockMatrixInvQuadFormFixture,
        testing::Values(
            std::make_tuple(0, 1, 2, 0.5),
            std::make_tuple(124, 1, 3, 0.3),
            std::make_tuple(321, 1, 4, 0.9),
            std::make_tuple(9382, 1, 5, 0.01),
            std::make_tuple(341111, 3, 9, 0.2)
            )
    );

} 
} // namespace ghostbasil
