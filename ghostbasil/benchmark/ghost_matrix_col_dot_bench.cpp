#include <benchmark/benchmark.h>
#include <ghostbasil/ghost_matrix.hpp>
#include <testutil/ghost_matrix_util.hpp>

namespace ghostbasil {
namespace {

using namespace ghost_matrix_util;

static void BM_ghost_matrix_col_dot(benchmark::State& state) 
{
    size_t seed = state.range(0);
    size_t L = state.range(1);
    size_t p = state.range(2);
    size_t n_knockoffs = state.range(3);
    double density = 0.1;

    auto input = generate_data(seed, L, p, n_knockoffs, density);
    auto& ml = std::get<0>(input);
    auto& vl = std::get<1>(input);
    auto& vs = std::get<3>(input);

    gmat_t gmat(ml, vl, n_knockoffs);
    
    value_t res = 0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(
            res = gmat.col_dot(p/2, vs)
        );
    }
}

BENCHMARK(BM_ghost_matrix_col_dot)
    // p large (large blocks)
    -> Args({0, 10, 10, 1})
    -> Args({0, 10, 20, 1})
    -> Args({0, 10, 30, 1})
    -> Args({0, 10, 50, 1})
    -> Args({0, 10, 80, 1})
    -> Args({0, 10, 100, 1})
    -> Args({0, 10, 200, 1})
    -> Args({0, 10, 300, 1})

    // L large (many blocks)
    -> Args({9, 100, 10, 1})
    -> Args({9, 100, 20, 1})
    -> Args({9, 100, 30, 1})
    -> Args({9, 100, 50, 1})
    ;

static void BM_matrix_col_dot(benchmark::State& state) 
{
    size_t seed = state.range(0);
    size_t L = state.range(1);
    size_t p = state.range(2);
    size_t n_knockoffs = state.range(3);
    double density = 0.1;

    auto input = generate_data(seed, L, p, n_knockoffs, density);
    auto& vs = std::get<3>(input);
    auto& dense = std::get<4>(input);
    
    value_t res = 0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(
            res = vs.dot(dense.col(p/2))
        );
    }
}

BENCHMARK(BM_matrix_col_dot)
    // p large (large blocks)
    -> Args({0, 10, 10, 1})
    -> Args({0, 10, 20, 1})
    -> Args({0, 10, 30, 1})
    -> Args({0, 10, 50, 1})
    -> Args({0, 10, 80, 1})
    -> Args({0, 10, 100, 1})
    -> Args({0, 10, 200, 1})
    -> Args({0, 10, 300, 1})

    // L large (many blocks)
    -> Args({9, 100, 10, 1})
    -> Args({9, 100, 20, 1})
    -> Args({9, 100, 30, 1})
    -> Args({9, 100, 50, 1})
    ;

}
} // namespace ghostbasil
