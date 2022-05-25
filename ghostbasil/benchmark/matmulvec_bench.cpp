#include <benchmark/benchmark.h>
#include <random>
#include <Eigen/Dense>

namespace ghostbasil {
namespace {

static inline auto make_input(
        size_t p, size_t seed)
{
    std::mt19937 gen(seed);
    std::normal_distribution<> norm(0., 1.);
    Eigen::MatrixXd X = Eigen::MatrixXd::NullaryExpr(p, p,
            [&](auto, auto) { return norm(gen); });
    Eigen::MatrixXd A = X.transpose() * X / p;
    Eigen::VectorXd r = Eigen::VectorXd::Random(p);
    return std::make_tuple(A, r);
}

static void BM_matmulvec(benchmark::State& state) 
{
    size_t p = state.range(0);
    size_t seed = 30;

    auto&& out = make_input(p, seed);
    auto& A = std::get<0>(out);
    auto& r = std::get<1>(out);
    Eigen::VectorXd t;

    for (auto _ : state) {
        benchmark::DoNotOptimize(t.noalias() = A * r);
    }
}

BENCHMARK(BM_matmulvec)
    -> Arg(10)
    -> Arg(50)
    -> Arg(100)
    -> Arg(500)
    -> Arg(1000);

}
} // namespace ghostbasil
