#pragma once
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <random>
#include <tools/matrix/block_matrix.hpp>

namespace ghostbasil {
namespace tools {

struct BlockGroupGhostMatrixUtil
{
    using value_t = double;
    using mat_t = Eigen::MatrixXd;
    using vec_t = Eigen::VectorXd;
    using sp_vec_t = Eigen::SparseVector<value_t>;

    template <class MatType, class VecMatType>
    static auto make_dense(
            const MatType& m,
            const VecMatType& dl,
            size_t K)
    {
        size_t p = m.rows() * K;

        assert(m.rows() == m.cols());
        
        mat_t d_dense = BlockMatrixUtil::make_dense(dl);

        mat_t dense(p, p);
        dense.setZero();

        size_t pi = m.rows();

        for (size_t k = 0; k < K; ++k) {
            for (size_t l = 0; l < K; ++l) {
                auto dense_kl = dense.block(pi*k, pi*l, pi, pi);
                dense_kl = m;
                if (k == l) {
                    dense_kl += d_dense;
                }
            }
        }

        return dense;
    }
    
    template <class GMatType>
    static auto make_dense(const GMatType& gmat)
    {
        const auto& D = gmat.get_D();
        std::vector<const Eigen::Map<mat_t>> dl;
        dl.reserve(D.n_blocks());
        for (size_t i = 0; i < D.n_blocks(); ++i) {
            dl.emplace_back(
                D.blocks()[i].data(),
                D.blocks()[i].rows(),
                D.blocks()[i].cols()
            );
        }
        return make_dense(
                gmat.get_S(), dl, gmat.n_groups());
    }

    static auto generate_data(
            size_t seed,
            size_t p,
            size_t n_blocks,
            size_t n_groups,
            double density = 0.5,
            bool do_dense = true,
            bool do_v = true,
            double eps = 0.1)
    {
        assert(eps > 0);
        srand(seed);
        mat_t mat;
        mat.setRandom(p, p);
        mat = (mat.transpose() * mat) / p;

        const auto block_size = p / n_blocks;
        std::vector<mat_t> dl(n_blocks, {block_size, block_size});
        if (p % n_blocks != 0) {
            dl.push_back({p % n_blocks, p % n_blocks});
        }
        for (auto& d : dl) {
            d.setRandom();
            d = eps * (d.transpose() * d) / d.cols();
        }

        mat_t dense;
        if (do_dense) {
            dense = make_dense(mat, dl, n_groups);
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

        return std::make_tuple(mat, dl, v, vs, dense);
    }
};

} // namespace tools
} // namespace ghostbasil
