#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <ghostbasil/matrix/block_matrix.hpp>
#include <tools/matrix/block_matrix.hpp>
#include <random>

namespace ghostbasil {
namespace {

static constexpr double tol = 1e-14; 

struct BlockMatrixFixture
    : ::testing::Test,
      tools::BlockMatrixUtil
{
    using util_t = tools::BlockMatrixUtil;

    using vec_t = Eigen::VectorXd;
    using sp_vec_t = Eigen::SparseVector<value_t>;
    using bmat_t = BlockMatrix<mat_t>;
    using mat_list_t = std::vector<mat_t>;

    auto generate_data(
            size_t seed,
            size_t L,
            size_t p,
            double density = 0.5,
            bool do_dense = true,
            bool do_v = true)
    {
        srand(seed);
        mat_list_t mat_list(L);
        for (size_t l = 0; l < L; ++l) {
            mat_list[l].setRandom(p, p);
            mat_list[l] = (mat_list[l] + mat_list[l].transpose()) / 2;
        }
        mat_t dense;
        if (do_dense) {
            dense = util_t::make_dense(mat_list);
        }

        vec_t v;
        sp_vec_t vs;
        if (do_v) {
            v.setRandom(dense.cols());
            vs.resize(v.size());
            std::bernoulli_distribution bern(density);
            std::mt19937 gen(seed);
            for (size_t i = 0; i < vs.size(); ++i) {
                if (bern(gen)) vs.coeffRef(i) = v[i];
            }
        }

        return std::make_tuple(mat_list, v, vs, dense);
    }
};

// ========================================================
// TEST Constructor
// ========================================================

TEST_F(BlockMatrixFixture, ctor_empty_list)
{
    mat_list_t ml;
    EXPECT_NO_THROW(bmat_t m(ml)); 
}

TEST_F(BlockMatrixFixture, ctor_m_not_square)
{
    mat_t m(2,3);
    mat_list_t ml(2, m);
    EXPECT_THROW(bmat_t m(ml), std::runtime_error); 
}

TEST_F(BlockMatrixFixture, ctor_valid)
{
    mat_t m(2,2);
    mat_list_t ml(2, m);
    EXPECT_NO_THROW(bmat_t m(ml)); 
}

// ========================================================
// TEST col_dot
// ========================================================

struct BlockMatrixColDotFixture
    : BlockMatrixFixture,
      testing::WithParamInterface<
        std::tuple<size_t, size_t, size_t> >
{
    template <class VecType>
    void test(
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
};

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

    test(bmat, dense, v);
    test(bmat, dense, vs);
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
    : BlockMatrixFixture,
      testing::WithParamInterface<
        std::tuple<size_t, size_t, size_t> >
{
    template <class VecType>
    void test(
            const bmat_t& bmat,
            const mat_t& dense,
            const VecType& v)
    {
        auto actual = bmat.quad_form(v);
        auto expected = v.dot(dense * v);
        EXPECT_NEAR(actual, expected, tol);
    }
};

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

    test(bmat, dense, v);
    test(bmat, dense, vs);
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
    : BlockMatrixFixture,
      testing::WithParamInterface<
        std::tuple<size_t, size_t, size_t, double> >
{
    template <class VecType>
    void test(
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
};

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

    test(bmat, dense, s, v);
    test(bmat, dense, s, vs);
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

// ========================================================
// TEST ConstBlockIterator
// ========================================================

struct BlockMatrixConstBlockIteratorFormFixture
    : BlockMatrixFixture,
      testing::WithParamInterface<
        std::tuple<size_t, size_t, size_t> >
{
    void test(
            const bmat_t& bmat,
            const mat_t& dense)
    {
        const auto it = bmat.block_begin();
        {
            auto expected = it;
            auto actual = it;
            actual.advance_at(0);
            EXPECT_EQ(actual, expected);
        }
        {
            auto next_idx = dense.cols() / 2; 
            auto expected = it;
            const auto& strides = bmat.strides();
            auto stride_it = std::upper_bound(strides.begin(), strides.end(), next_idx);
            auto to_iterate = std::distance(strides.begin(), stride_it)-1;
            for (size_t i = 0; i < to_iterate; ++expected, ++i);
            auto actual = it;
            actual.advance_at(next_idx);
            EXPECT_EQ(actual, expected);
        }
        {
            auto next_idx = dense.cols() / 2 + dense.cols() / 4; 
            auto expected = it;
            const auto& strides = bmat.strides();
            auto stride_it = std::upper_bound(strides.begin(), strides.end(), next_idx);
            auto to_iterate = std::distance(strides.begin(), stride_it)-1;
            for (size_t i = 0; i < to_iterate; ++expected, ++i);
            auto actual = it;
            actual.advance_at(next_idx);
            EXPECT_EQ(actual, expected);
        }
    }
};

TEST_P(BlockMatrixConstBlockIteratorFormFixture, const_block_iterator)
{
    auto param = GetParam();
    size_t seed = std::get<0>(param);
    size_t L = std::get<1>(param);
    size_t p = std::get<2>(param);

    auto out = generate_data(seed, L, p);
    auto& ml = std::get<0>(out);
    auto& dense = std::get<3>(out);

    bmat_t bmat(ml);

    test(bmat, dense);
    test(bmat, dense);
}

INSTANTIATE_TEST_SUITE_P(
        BlockMatrixConstBlockIteratorFormSuite,
        BlockMatrixConstBlockIteratorFormFixture,
        testing::Values(
            std::make_tuple(0, 1, 2),
            std::make_tuple(124, 1, 3),
            std::make_tuple(321, 1, 4),
            std::make_tuple(9382, 1, 5),
            std::make_tuple(341111, 3, 9)
            )
    );

} 
} // namespace ghostbasil
