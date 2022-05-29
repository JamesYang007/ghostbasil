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
    using value_t = double;
    using index_t = uint32_t;
    using vec_t = Eigen::Vector<value_t, Eigen::Dynamic>;
    using sp_vec_t = Eigen::SparseVector<value_t>;
    using sp_vec_our_t = Eigen::SparseVector<value_t, Eigen::ColMajor, index_t>;
    using map_sp_vec_our_t = Eigen::Map<const sp_vec_our_t>;

    auto make_input(
            size_t seed,
            size_t p,
            double density)
    {
        std::mt19937 gen(seed);
        vec_t beta(p); 
        beta.setRandom();

        std::vector<index_t> innerIndex;
        std::vector<value_t> values;
        std::normal_distribution<> norm(0., 1.);
        std::bernoulli_distribution bern(density);
        for (size_t i = 0; i < p; ++i) {
            if (bern(gen)) {
                innerIndex.push_back(i);
                values.push_back(norm(gen));
            }
        }

        return std::make_tuple(beta, innerIndex, values);
    }
};

BENCHMARK_DEFINE_F(Fixture, sparse)(benchmark::State& state)
{
    size_t seed = state.range(0);
    size_t p = state.range(1);
    double density = 0.3;

    auto input = make_input(seed, p, density);
    auto& beta = std::get<0>(input);
    auto& innerIndex = std::get<1>(input);
    auto& values = std::get<2>(input);

    map_sp_vec_our_t vmap(beta.size(), innerIndex.size(), innerIndex.data(), values.data());
    sp_vec_t vs = vmap;

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
    double density = 0.3;

    auto input = make_input(seed, p, density);
    auto& beta = std::get<0>(input);
    auto& innerIndex = std::get<1>(input);
    auto& values = std::get<2>(input);

    map_sp_vec_our_t vs_our_map(
            beta.size(), 
            innerIndex.size(), 
            innerIndex.data(),
            values.data());

    double res = 0;

    for (auto _ : state) {
        benchmark::DoNotOptimize(
                res = vs_our_map.dot(beta)
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
