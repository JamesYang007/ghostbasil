#pragma once 
#include <Eigen/SparseCore>
#include <ghostbasil/matrix/ghost_matrix.hpp>
#include <random>

namespace ghostbasil {
namespace ghost_matrix_util {

using value_t = double;
using mat_t = Eigen::MatrixXd;
using vec_t = Eigen::VectorXd;
using sp_vec_t = Eigen::SparseVector<value_t>;
using gmat_t = GhostMatrix<mat_t, vec_t>;

template <class MatType, class VecType>
inline auto make_dense(
        const MatType& m,
        const VecType& v,
        size_t K)
{
    size_t p = m.rows() * K;

    assert(m.rows() == m.cols());
    assert(m.rows() == v.size());

    mat_t dense(p, p);
    dense.setZero();

    size_t pi = m.rows();

    for (size_t k = 0; k < K; ++k) {
        for (size_t l = 0; l < K; ++l) {
            auto dense_kl = dense.block(pi*k, pi*l, pi, pi);
            dense_kl = m;
            if (k == l) continue;
            dense_kl.diagonal() -= v;
        }
    }

    return dense;
}

inline auto make_dense(const gmat_t& gmat)
{
    return make_dense(gmat.matrix(), gmat.vector(), gmat.n_groups());
}

inline auto generate_data(
        size_t seed,
        size_t p,
        size_t n_groups,
        double density = 0.5,
        bool do_dense = true,
        bool do_v = true)
{
    srand(seed);
    mat_t mat;
    vec_t vec;
    mat.setRandom(p, p);
    mat = (mat + mat.transpose()) / 2;
    vec.setRandom(p);

    mat_t dense;
    if (do_dense) dense = make_dense(mat, vec, n_groups);

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

    return std::make_tuple(mat, vec, v, vs, dense);
}

} // namespace ghost_matrix_util
} // namespace ghostbasil
