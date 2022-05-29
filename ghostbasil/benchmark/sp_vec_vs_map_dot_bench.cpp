// This benchmark tests the speed difference between
// performing sparse vector dot product with dense vector
// vs. Eigen::Map<SparseVector> with dense vector.
#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <ghostbasil/util/eigen/map_sparsevector.hpp>
#include <random>

namespace ghostbasil {
namespace {

struct Fixture
    : benchmark::Fixture
{
    using vec_t = Eigen::Vector<double, Eigen::Dynamic>;
    using sp_vec_t = Eigen::SparseVector<double>;
    using map_sp_vec_t = Eigen::Map<sp_vec_t>;

    auto make_input(
            size_t seed,
            size_t p,
            double density)
    {
        std::mt19937 gen(seed);
        vec_t beta(p); 
        beta.setRandom();

        sp_vec_t vs(p);
        std::normal_distribution<> norm(0., 1.);
        std::bernoulli_distribution bern(density);
        for (size_t i = 0; i < vs.size(); ++i) {
            if (bern(gen)) vs.coeffRef(i) = norm(gen);
        }

        map_sp_vec_t vs_map(vs.size(), vs.nonZeros(), vs.innerIndexPtr(), vs.valuePtr());

        return std::make_tuple(beta, vs, vs_map);
    }
};

BENCHMARK_DEFINE_F(Fixture, sparse)(benchmark::State& state)
{
    size_t seed = state.range(0);
    size_t p = state.range(1);
    double density = 0.7;

    auto input = make_input(seed, p, density);
    auto& beta = std::get<0>(input);
    auto& vs = std::get<1>(input);

    double res = 0;

    for (auto _ : state) {
        benchmark::DoNotOptimize(
                res = vs.dot(beta)
                );
    }
}

BENCHMARK_REGISTER_F(Fixture, sparse)
    -> Args({0, 10})
    -> Args({0, 100})
    -> Args({0, 500})
    -> Args({0, 1000})
    ;

BENCHMARK_DEFINE_F(Fixture, sparse_map)(benchmark::State& state)
{
    size_t seed = state.range(0);
    size_t p = state.range(1);
    double density = 0.7;

    auto input = make_input(seed, p, density);
    auto& beta = std::get<0>(input);
    auto& vs_map = std::get<2>(input);

    double res = 0;

    for (auto _ : state) {
        benchmark::DoNotOptimize(
                res = vs_map.dot(beta)
                );
    }
}

BENCHMARK_REGISTER_F(Fixture, sparse_map)
    -> Args({0, 10})
    -> Args({0, 100})
    -> Args({0, 500})
    -> Args({0, 1000})
    ;

}
} // namespace ghostbasil
