#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <random>

namespace {
    
static void BM_swap(benchmark::State& state)
{
    const auto p = state.range(0);
    Eigen::MatrixXd A(p, p);
    A.setRandom(); 
    std::vector<int> order(p);
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 gen(0);
    std::shuffle(order.begin(), order.end(), gen);
    for (auto _ : state) {
        A = A(order, order);
    }
}
    
BENCHMARK(BM_swap)
    -> Arg(10)
    -> Arg(50)
    -> Arg(100)
    -> Arg(500)
    -> Arg(1000)
    ;

} // namespace