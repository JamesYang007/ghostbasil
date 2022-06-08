#include <benchmark/benchmark.h>
#include <ghostbasil/matrix/block_matrix.hpp>
#include <ghostbasil/matrix/ghost_matrix.hpp>
#include <tools/matrix/ghost_matrix.hpp>
#include <tools/matrix/block_matrix.hpp>

namespace ghostbasil {
namespace {

struct GhostMatrixColDotFixture
    : benchmark::Fixture,
      tools::GhostMatrixUtil,
      tools::BlockMatrixUtil
{
    using gutil = tools::GhostMatrixUtil;
    using butil = tools::BlockMatrixUtil;

    using value_t = double;
    using mat_t = Eigen::MatrixXd;
    using vec_t = Eigen::VectorXd;
    using sp_vec_t = Eigen::SparseVector<value_t>;
    using gmat_t = GhostMatrix<mat_t, vec_t>;
    using bmat_t = BlockMatrix<gmat_t>;
    using gmat_list_t = std::vector<gmat_t>;
    using mat_list_t = std::vector<mat_t>;
    using vec_list_t = std::vector<vec_t>;

    auto generate_data(
            size_t seed,
            size_t L,
            size_t p,
            size_t n_groups,
            double density)
    {
        mat_list_t mat_list;
        vec_list_t vec_list;

        for (size_t l = 0; l < L; ++l) {
            auto input = gutil::generate_data(seed+l, p, n_groups, density, false, false);
            auto& S = std::get<0>(input);
            auto& D = std::get<1>(input);
            mat_list.emplace_back(S);
            vec_list.emplace_back(D);
        }

        size_t p_tot = 0;
        for (size_t l = 0; l < L; ++l) {
            p_tot += mat_list[l].cols();
        }

        sp_vec_t vs(p_tot);
        std::normal_distribution<> norm(0., 1.);
        std::bernoulli_distribution bern(density);
        std::mt19937 gen(seed);
        for (size_t i = 0; i < vs.size(); ++i) {
            if (bern(gen)) vs.coeffRef(i) = norm(gen);
        }

        return std::make_tuple(mat_list, vec_list, vs);    
    }
};

// ==================================================================
// BENCHMARK BlockMatrix<GhostMatrix> vs. Dense vs. BlockMatrix<Dense>
// ==================================================================

BENCHMARK_DEFINE_F(GhostMatrixColDotFixture, ghost_matrix)(benchmark::State& state) 
{
    size_t seed = state.range(0);
    size_t L = state.range(1);
    size_t p = state.range(2);
    size_t n_groups = state.range(3);
    double density = 0.1;

    auto input = generate_data(seed, L, p, n_groups, density);
    auto& ml = std::get<0>(input);
    auto& vl = std::get<1>(input);
    auto& vs = std::get<2>(input);

    gmat_list_t gmat_list;
    for (size_t i = 0; i < ml.size(); ++i) {
        gmat_list.emplace_back(ml[i], vl[i], n_groups);
    }

    bmat_t bmat(gmat_list);
    
    value_t res = 0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(
            res = bmat.col_dot(p/2, vs)
        );
    }
    state.counters["dp"] = res;
}

BENCHMARK_REGISTER_F(GhostMatrixColDotFixture, ghost_matrix)
    // p large (large blocks)
    -> Args({0, 10, 10, 2})
    -> Args({0, 10, 20, 2})
    -> Args({0, 10, 30, 2})
    -> Args({0, 10, 50, 2})
    -> Args({0, 10, 80, 2})
    -> Args({0, 10, 100, 2})
    -> Args({0, 10, 200, 2})
    -> Args({0, 10, 300, 2})
    -> Args({0, 10, 500, 2})
    -> Args({0, 10, 1000, 2})
    -> Args({0, 10, 2000, 2})

    // L large (many blocks)
    -> Args({9, 100, 10, 2})
    -> Args({9, 100, 20, 2})
    -> Args({9, 100, 30, 2})
    -> Args({9, 100, 50, 2})
    -> Args({9, 100, 100, 2})
    -> Args({9, 100, 500, 2})
    -> Args({9, 100, 1000, 2})
    ;

BENCHMARK_DEFINE_F(GhostMatrixColDotFixture, dense_matrix)(benchmark::State& state) 
{
    size_t seed = state.range(0);
    size_t L = state.range(1);
    size_t p = state.range(2);
    size_t n_groups = state.range(3);
    double density = 0.1;

    auto input = generate_data(seed, L, p, n_groups, density);
    auto& ml = std::get<0>(input);
    auto& vl = std::get<1>(input);
    auto& vs = std::get<2>(input);

    gmat_list_t gmat_list;
    for (size_t i = 0; i < ml.size(); ++i) {
        gmat_list.emplace_back(ml[i], vl[i], n_groups);
    }

    auto dense = butil::make_dense(gmat_list, 
            [](auto& x, const auto& y) {
                x = gutil::make_dense(y);
            });
    
    value_t res = 0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(
            res = vs.dot(dense.col(p/2))
        );
    }
    state.counters["dp"] = res;
}

BENCHMARK_REGISTER_F(GhostMatrixColDotFixture, dense_matrix)
    // p large (large blocks)
    -> Args({0, 10, 10, 2})
    -> Args({0, 10, 20, 2})
    -> Args({0, 10, 30, 2})
    -> Args({0, 10, 50, 2})
    -> Args({0, 10, 80, 2})
    -> Args({0, 10, 100, 2})
    -> Args({0, 10, 200, 2})
    -> Args({0, 10, 300, 2})

    // L large (many blocks)
    -> Args({9, 100, 10, 2})
    -> Args({9, 100, 20, 2})
    -> Args({9, 100, 30, 2})
    -> Args({9, 100, 50, 2})
    ;

BENCHMARK_DEFINE_F(GhostMatrixColDotFixture, block_dense_matrix)(benchmark::State& state) 
{
    size_t seed = state.range(0);
    size_t L = state.range(1);
    size_t p = state.range(2);
    size_t n_groups = state.range(3);
    double density = 0.1;

    auto input = generate_data(seed, L, p, n_groups, density);
    auto& ml = std::get<0>(input);
    auto& vl = std::get<1>(input);
    auto& vs = std::get<2>(input);

    mat_list_t gmat_dense_list(ml.size());
    for (size_t i = 0; i < gmat_dense_list.size(); ++i) {
        gmat_dense_list[i] = gutil::make_dense(ml[i], vl[i], n_groups);
    }

    BlockMatrix<mat_t> bmat(gmat_dense_list);

    value_t res = 0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(
            res = bmat.col_dot(p/2, vs)
        );
    }
    state.counters["dp"] = res;
}

BENCHMARK_REGISTER_F(GhostMatrixColDotFixture, block_dense_matrix)
    // p large (large blocks)
    -> Args({0, 10, 10, 2})
    -> Args({0, 10, 20, 2})
    -> Args({0, 10, 30, 2})
    -> Args({0, 10, 50, 2})
    -> Args({0, 10, 80, 2})
    -> Args({0, 10, 100, 2})
    -> Args({0, 10, 200, 2})
    -> Args({0, 10, 300, 2})
    -> Args({0, 10, 500, 2})
    -> Args({0, 10, 1000, 2})
    -> Args({0, 10, 2000, 2})

    // L large (many blocks)
    -> Args({9, 100, 10, 2})
    -> Args({9, 100, 20, 2})
    -> Args({9, 100, 30, 2})
    -> Args({9, 100, 50, 2})
    -> Args({9, 100, 100, 2})
    -> Args({9, 100, 500, 2})
    -> Args({9, 100, 1000, 2})
    ;

// ==================================================================
// BENCHMARK GhostMatrix vs. Dense
// ==================================================================

struct GhostMatrixColDotFixture2
    : GhostMatrixColDotFixture
{};

BENCHMARK_DEFINE_F(GhostMatrixColDotFixture2, ghost_matrix)(benchmark::State& state) 
{
    size_t seed = state.range(0);
    size_t p = state.range(1);
    size_t L = 1;
    size_t n_groups = 2;
    double density = 0.1;

    auto input = generate_data(seed, L, p, n_groups, density);
    auto& ml = std::get<0>(input);
    auto& vl = std::get<1>(input);
    auto& vs = std::get<2>(input);

    gmat_t gmat(ml[0], vl[0], n_groups);

    value_t res = 0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(
            res = gmat.col_dot(p/2, vs)
        );
    }
    state.counters["dp"] = res;
}

BENCHMARK_REGISTER_F(GhostMatrixColDotFixture2, ghost_matrix)
    -> Args({0, 10})
    -> Args({0, 50})
    -> Args({0, 100})
    -> Args({0, 500})
    ;

BENCHMARK_DEFINE_F(GhostMatrixColDotFixture2, dense_matrix)(benchmark::State& state) 
{
    size_t seed = state.range(0);
    size_t p = state.range(1);
    size_t L = 1;
    size_t n_groups = 2;
    double density = 0.1;

    auto input = generate_data(seed, L, p, n_groups, density);
    auto& ml = std::get<0>(input);
    auto& vl = std::get<1>(input);
    auto& vs = std::get<2>(input);

    auto dense = gutil::make_dense(ml[0], vl[0], n_groups);
    
    value_t res = 0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(
            res = dense.col_dot(p/2, vs)
        );
    }
    state.counters["dp"] = res;
}

BENCHMARK_REGISTER_F(GhostMatrixColDotFixture2, dense_matrix)
    -> Args({0, 10})
    -> Args({0, 50})
    -> Args({0, 100})
    -> Args({0, 500})
    ;

// ==================================================================
// BENCHMARK GhostMatrix vs. Dense Coeff
// ==================================================================

struct GhostMatrixCoeffFixture
    : GhostMatrixColDotFixture
{};

BENCHMARK_DEFINE_F(GhostMatrixCoeffFixture, ghost_matrix)(benchmark::State& state) 
{
    size_t seed = state.range(0);
    size_t p = state.range(1);
    size_t L = 1;
    size_t n_groups = 2;
    double density = 0.1;

    auto input = generate_data(seed, L, p, n_groups, density);
    auto& ml = std::get<0>(input);
    auto& vl = std::get<1>(input);

    gmat_t gmat(ml[0], vl[0], n_groups);

    value_t res = 0;
    for (auto _ : state) {
        res = 0;
        for (size_t j = 0; j < gmat.cols(); ++j) {
            for (size_t i = 0; i < gmat.rows(); ++i) {
                benchmark::DoNotOptimize(
                    res += gmat.coeff(i,j)
                );
            }
        }
    }
    state.counters["dp"] = res;
}

BENCHMARK_REGISTER_F(GhostMatrixCoeffFixture, ghost_matrix)
    -> Args({0, 10})
    -> Args({0, 50})
    -> Args({0, 100})
    -> Args({0, 500})
    ;

BENCHMARK_DEFINE_F(GhostMatrixCoeffFixture, dense_matrix)(benchmark::State& state) 
{
    size_t seed = state.range(0);
    size_t p = state.range(1);
    size_t L = 1;
    size_t n_groups = 2;
    double density = 0.1;

    auto input = generate_data(seed, L, p, n_groups, density);
    auto& ml = std::get<0>(input);
    auto& vl = std::get<1>(input);

    auto dense = gutil::make_dense(ml[0], vl[0], n_groups);
    
    value_t res = 0;
    for (auto _ : state) {
        res = 0;
        for (size_t j = 0; j < dense.cols(); ++j) {
            for (size_t i = 0; i < dense.rows(); ++i) {
                benchmark::DoNotOptimize(
                    res += dense.coeff(i,j)
                );
            }
        }
    }
    state.counters["dp"] = res;
}

BENCHMARK_REGISTER_F(GhostMatrixCoeffFixture, dense_matrix)
    -> Args({0, 10})
    -> Args({0, 50})
    -> Args({0, 100})
    -> Args({0, 500})
    ;
}
} // namespace ghostbasil
