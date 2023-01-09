#pragma once
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <random>

namespace ghostbasil {
namespace tools {

struct GroupGhostMatrixUtil
{
    using value_t = double;
    using mat_t = Eigen::MatrixXd;
    using vec_t = Eigen::VectorXd;
    using sp_vec_t = Eigen::SparseVector<value_t>;

    template <class MatType>
    static auto make_dense(
            const MatType& m,
            const MatType& d,
            size_t K)
    {
        size_t p = m.rows() * K;

        assert(m.rows() == m.cols());
        assert(m.rows() == d.cols());

        mat_t dense(p, p);
        dense.setZero();

        size_t pi = m.rows();

        for (size_t k = 0; k < K; ++k) {
            for (size_t l = 0; l < K; ++l) {
                auto dense_kl = dense.block(pi*k, pi*l, pi, pi);
                dense_kl = m;
                if (k == l) {
                    dense_kl += d;
                }
            }
        }

        return dense;
    }
    
    template <class GMatType>
    static auto make_dense(const GMatType& gmat)
    {
        return make_dense(
                gmat.get_S(), gmat.get_D(), gmat.n_groups());
    }

    static auto generate_data(
            size_t seed,
            size_t p,
            size_t n_groups,
            double density = 0.5,
            bool do_dense = true,
            bool do_v = true,
            double eps = 0.1)
    {
        assert(eps > 0);
        srand(seed);
        mat_t mat;
        mat_t d(p, p);
        mat.setRandom(p, p);
        d.setZero();
        mat = (mat.transpose() * mat) / p;
        d.diagonal().array() = eps; // make sure (g+1)/g mat >= vec >= 0

        mat_t dense;
        if (do_dense) {
            dense = make_dense(mat, d, n_groups);
        }

        size_t full_size = p * n_groups;
        vec_t v;
        sp_vec_t vs;
        if (do_v) {
            v.setRandom(full_size);
            vs.resize(v.size());
            std::bernoulli_distribution bern(density);
            std::mt19937 gen(seed);
            for (size_t i = 0; i < vs.size(); ++i) {
                if (bern(gen)) vs.coeffRef(i) = v[i];
            }
        }

        return std::make_tuple(mat, d, v, vs, dense);
    }
};

} // namespace tools
} // namespace ghostbasil
