// This benchmark tests whether writing a manual loop
// over a range of a subvector vs. using segment.
//
// TLDR:
// - Sparse vector dot-product internally constructs an InnerIterator
//   for the vector. If a block is used, then its InnerIterator's constructor
//   (in EvalIterator) does a while-loop to iterate the inner index until 
//   it is at the first real index >= begin.
//   This loop adds a lot of cost!
#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <ghostbasil/util/eigen/map_sparsevector.hpp>
#include <random>

namespace ghostbasil {
namespace {

struct SpVecDotFixture
    : benchmark::Fixture
{
    auto make_input(size_t p, double density=0.9, size_t seed=124)
    {
        srand(seed);
        Eigen::SparseVector<double> v(p);
        std::bernoulli_distribution bern(density);
        std::mt19937 gen(seed);
        for (size_t i = 0; i < p; ++i) {
            if (bern(gen)) v.coeffRef(i) = i;
        }
        Eigen::VectorXd x(p);
        x.setRandom();
        return std::make_tuple(v, x);
    }
};

BENCHMARK_DEFINE_F(SpVecDotFixture, subset)(benchmark::State& state)
{
    size_t p = state.range(0);
    size_t k = state.range(1);
    double res = 0;
    auto input = make_input(p);
    auto v = std::get<0>(input);
    auto x = std::get<1>(input);
    auto group_size = p / k;
    for (auto _ : state) {
        res = 0;
        for (size_t i = 0; i < k; ++i) {
            res += v.segment(i*group_size, group_size).dot(
                x.segment(i*group_size, group_size));
        }
    }
    state.counters["res"] = res;
}

BENCHMARK_DEFINE_F(SpVecDotFixture, manual)(benchmark::State& state)
{
    size_t p = state.range(0);
    size_t k = state.range(1);
    double res = 0;
    auto input = make_input(p);
    auto v = std::get<0>(input);
    auto x = std::get<1>(input);
    auto group_size = p / k;
    for (auto _ : state) {
        // iterate over each group
        res = 0;
        size_t seg_pos = 0;
        const auto inner = v.innerIndexPtr();
        const auto values = v.valuePtr();
        const auto nnz = v.nonZeros();
        for (size_t i = 0; i < k; ++i) {
            const auto end = (i+1)*group_size;
            auto it = std::lower_bound(inner+seg_pos, inner+nnz, end);
            auto seg_end = std::distance(inner, it);
            // compute dot product for that group
            for (; seg_pos < seg_end; ++seg_pos) {
                auto v_idx = inner[seg_pos];
                res += values[seg_pos] * x.coeff(v_idx);
            }
        }
    }
    state.counters["res"] = res;
}

// This version is the same as manual, but
// does not provide a hint that the next segment inner indices
// are shrinking in possibility.
BENCHMARK_DEFINE_F(SpVecDotFixture, manual_no_hint)(benchmark::State& state)
{
    size_t p = state.range(0);
    size_t k = state.range(1);
    double res = 0;
    auto input = make_input(p);
    auto v = std::get<0>(input);
    auto x = std::get<1>(input);
    auto group_size = p / k;
    for (auto _ : state) {
        // iterate over each group
        res = 0;
        const auto inner = v.innerIndexPtr();
        const auto values = v.valuePtr();
        const auto nnz = v.nonZeros();
        for (size_t i = 0; i < k; ++i) {
            const auto begin = i*group_size;
            auto it = std::lower_bound(inner, inner+nnz, begin);
            auto seg_begin = std::distance(inner, it);
            it = std::lower_bound(inner+seg_begin, inner+nnz, begin+group_size);
            auto seg_end = std::distance(inner, it);
            // compute dot product for that group
            for (size_t seg_pos = seg_begin; seg_pos < seg_end; ++seg_pos) {
                auto v_idx = inner[seg_pos];
                res += values[seg_pos] * x.coeff(v_idx);
            }
        }
    }
    state.counters["res"] = res;
}

BENCHMARK_DEFINE_F(SpVecDotFixture, optimal)(benchmark::State& state)
{
    size_t p = state.range(0);
    double res = 0;
    auto input = make_input(p);
    auto v = std::get<0>(input);
    auto x = std::get<1>(input);
    for (auto _ : state) {
        res = v.dot(x);
    }
    state.counters["res"] = res;
}

BENCHMARK_DEFINE_F(SpVecDotFixture, optimal_map)(benchmark::State& state)
{
    size_t p = state.range(0);
    double res = 0;
    auto input = make_input(p);
    auto v = std::get<0>(input);
    auto x = std::get<1>(input);
    Eigen::Map<Eigen::SparseVector<double>> v_map(
            v.size(), v.nonZeros(), v.innerIndexPtr(),
            v.valuePtr());
    for (auto _ : state) {
        res = v_map.dot(x);
    }
    state.counters["res"] = res;
}

BENCHMARK_REGISTER_F(SpVecDotFixture, subset)
    -> Args({10, 2})
    -> Args({100, 2})
    -> Args({500, 2})
    -> Args({1000, 2})
    -> Args({5000, 2})
    -> Args({10000, 2})

    -> Args({10, 1})
    -> Args({100, 10})
    -> Args({500, 50})
    -> Args({1000, 100})
    -> Args({5000, 500})
    -> Args({10000, 1000})

    -> Args({1000000, 2000})
    ;

BENCHMARK_REGISTER_F(SpVecDotFixture, manual)
    -> Args({10, 2})
    -> Args({100, 2})
    -> Args({500, 2})
    -> Args({1000, 2})
    -> Args({5000, 2})
    -> Args({10000, 2})

    -> Args({10, 1})
    -> Args({100, 10})
    -> Args({500, 50})
    -> Args({1000, 100})
    -> Args({5000, 500})
    -> Args({10000, 1000})

    -> Args({1000000, 2000})
    ;

BENCHMARK_REGISTER_F(SpVecDotFixture, manual_no_hint)
    -> Args({10, 2})
    -> Args({100, 2})
    -> Args({500, 2})
    -> Args({1000, 2})
    -> Args({5000, 2})
    -> Args({10000, 2})

    -> Args({10, 1})
    -> Args({100, 10})
    -> Args({500, 50})
    -> Args({1000, 100})
    -> Args({5000, 500})
    -> Args({10000, 1000})

    -> Args({1000000, 2000})
    ;

BENCHMARK_REGISTER_F(SpVecDotFixture, optimal)
    -> Args({10, 2})
    -> Args({100, 2})
    -> Args({500, 2})
    -> Args({1000, 2})
    -> Args({5000, 2})
    -> Args({10000, 2})

    -> Args({10, 1})
    -> Args({100, 10})
    -> Args({500, 50})
    -> Args({1000, 100})
    -> Args({5000, 500})
    -> Args({10000, 1000})

    -> Args({1000000, 2000})
    ;

BENCHMARK_REGISTER_F(SpVecDotFixture, optimal_map)
    -> Args({10, 2})
    -> Args({100, 2})
    -> Args({500, 2})
    -> Args({1000, 2})
    -> Args({5000, 2})
    -> Args({10000, 2})

    -> Args({10, 1})
    -> Args({100, 10})
    -> Args({500, 50})
    -> Args({1000, 100})
    -> Args({5000, 500})
    -> Args({10000, 1000})

    -> Args({1000000, 2000})
    ;

}
} // namespace ghostbasil
