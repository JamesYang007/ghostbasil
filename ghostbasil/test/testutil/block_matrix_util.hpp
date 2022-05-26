#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <ghostbasil/block_matrix.hpp>
#include <random>

namespace ghostbasil {
namespace block_matrix_util {

using value_t = double;
using mat_t = Eigen::MatrixXd;
using vec_t = Eigen::VectorXd;
using sp_vec_t = Eigen::SparseVector<value_t>;
using bmat_t = BlockMatrix<mat_t>;
using mat_list_t = std::vector<mat_t>;

struct Assignment
{
    template <class XType, class YType>
    auto&& operator()(XType&& x, const YType& y)
    {
        return x = y;
    }
};

template <class MatListType, class AssignType=Assignment>
inline auto make_dense(
        const MatListType& ml,
        AssignType assign = AssignType())
{
    size_t p = 0;
    for (size_t i = 0; i < ml.size(); ++i) {
        const auto& B = ml[i];
        assert(B.rows() == B.cols());
        p += B.rows();
    }

    mat_t dense(p, p);
    dense.setZero();

    size_t pi_cum = 0;
    for (size_t i = 0; i < ml.size(); ++i) {
        const auto& B = ml[i];
        size_t pi = B.rows();
        auto dense_i = dense.block(pi_cum, pi_cum, pi, pi);
        assign(dense_i, B);
        pi_cum += pi;
    }

    return dense;
}

inline auto generate_data(
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
        dense = make_dense(mat_list);
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

} // namespace block_matrix_util
} // namespace ghostbasil
