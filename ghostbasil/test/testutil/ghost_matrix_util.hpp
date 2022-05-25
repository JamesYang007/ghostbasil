#pragma once 
#include <Eigen/SparseCore>
#include <ghostbasil/ghost_matrix.hpp>
#include <random>

namespace ghostbasil {
namespace ghost_matrix_util {

using value_t = double;
using mat_t = Eigen::MatrixXd;
using vec_t = Eigen::VectorXd;
using sp_vec_t = Eigen::SparseVector<value_t>;
using gmat_t = GhostMatrix<mat_t, vec_t>;
using mat_list_t = std::vector<mat_t>;
using vec_list_t = std::vector<vec_t>;

template <class MatListType, class VecListType>
inline auto make_dense(
        const MatListType& ml,
        const VecListType& vl,
        size_t K)
{
    assert(ml.size() == vl.size());

    size_t p = 0;
    for (size_t i = 0; i < ml.size(); ++i) {
        const auto& B = ml[i];
        const auto& D = vl[i];

        assert(B.rows() == B.cols());
        assert(B.rows() == D.size());

        p += B.rows() * K;
    }

    mat_t dense(p, p);
    dense.setZero();

    size_t pi_cum = 0;
    for (size_t i = 0; i < ml.size(); ++i) {
        const auto& B = ml[i];
        const auto& D = vl[i];
        size_t pi = B.rows();
        auto dense_i = dense.block(pi_cum, pi_cum, pi*K, pi*K);

        for (size_t k = 0; k < K; ++k) {
            for (size_t l = 0; l < K; ++l) {
                auto dense_ikl = dense_i.block(pi*k, pi*l, pi, pi);
                dense_ikl = B;
                if (k == l) continue;
                dense_ikl.diagonal() -= D;
            }
        }

        pi_cum += pi*K;
    }

    return dense;
}

inline auto generate_data(
        size_t seed,
        size_t L,
        size_t p,
        size_t n_knockoffs,
        double density = 0.5)
{
    srand(seed);
    mat_list_t mat_list(L);
    vec_list_t vec_list(L);
    for (size_t l = 0; l < L; ++l) {
        mat_list[l].setRandom(p, p);
        mat_list[l] = (mat_list[l] + mat_list[l].transpose()) / 2;
        vec_list[l].setRandom(p);
    }
    auto dense = make_dense(mat_list, vec_list, n_knockoffs+1);
    vec_t v = vec_t::Random(dense.cols());
    sp_vec_t vs(v.size());
    std::bernoulli_distribution bern(density);
    std::mt19937 gen(seed);
    for (size_t i = 0; i < vs.size(); ++i) {
        if (bern(gen)) vs.coeffRef(i) = v[i];
    }
    return std::make_tuple(mat_list, vec_list, v, vs, dense);
}

} // namespace ghost_matrix_util
} // namespace ghostbasil
