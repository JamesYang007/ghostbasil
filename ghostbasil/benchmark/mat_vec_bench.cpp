// This benchmark tests whether manual loop of dot-products is 
// faster than matrix-vector product.
#include <benchmark/benchmark.h>
#include <Eigen/Core>
#include <random>

namespace ghostbasil {
namespace {

struct MatVecFixture
    : benchmark::Fixture
{
    auto make_input(size_t n, size_t p, size_t seed=124)
    {
        srand(seed);
        Eigen::MatrixXd A(n, p);
        Eigen::VectorXd v(p);
        A.setRandom();
        v.setRandom();
        return std::make_tuple(A, v);
    }
};

BENCHMARK_DEFINE_F(MatVecFixture, manual)(benchmark::State& state)
{
    size_t n = state.range(0);
    size_t p = state.range(1);
    auto input = make_input(n, p);
    auto A = std::get<0>(input);
    auto v = std::get<1>(input);
    Eigen::VectorXd out(n);
    for (auto _ : state) {
        out.setZero();
        for (size_t i = 0; i < p; ++i) {
            out += A.col(i) * v[i];
        }
    }
}

BENCHMARK_DEFINE_F(MatVecFixture, product)(benchmark::State& state)
{
    size_t n = state.range(0);
    size_t p = state.range(1);
    auto input = make_input(n, p);
    auto A = std::get<0>(input);
    auto v = std::get<1>(input);
    Eigen::VectorXd out(n);
    for (auto _ : state) {
        out.setZero();
        out = A * v;
    }
}

BENCHMARK_REGISTER_F(MatVecFixture, manual)
    -> Args({100, 2})
    -> Args({100, 20})
    -> Args({100, 25})
    -> Args({100, 50})
    -> Args({100, 75})
    -> Args({100, 100})

    //-> Args({10, 2})
    //-> Args({100, 2})
    //-> Args({250, 2})
    //-> Args({500, 2})
    //-> Args({750, 2})
    //-> Args({1000, 2})

    //-> Args({1000, 1})
    //-> Args({1000, 10})
    //-> Args({1000, 50})
    //-> Args({1000, 100})
    //-> Args({1000, 500})
    //-> Args({1000, 1000})

    //-> Args({20, 1})
    //-> Args({20, 10})
    //-> Args({20, 50})
    //-> Args({20, 100})
    //-> Args({20, 500})
    //-> Args({20, 1000})
    ;

BENCHMARK_REGISTER_F(MatVecFixture, product)
    -> Args({100, 2})
    -> Args({100, 20})
    -> Args({100, 25})
    -> Args({100, 50})
    -> Args({100, 75})
    -> Args({100, 100})

    //-> Args({10, 2})
    //-> Args({100, 2})
    //-> Args({250, 2})
    //-> Args({500, 2})
    //-> Args({750, 2})
    //-> Args({1000, 2})

    //-> Args({1000, 1})
    //-> Args({1000, 10})
    //-> Args({1000, 50})
    //-> Args({1000, 100})
    //-> Args({1000, 500})
    //-> Args({1000, 1000})

    //-> Args({20, 1})
    //-> Args({20, 10})
    //-> Args({20, 50})
    //-> Args({20, 100})
    //-> Args({20, 500})
    //-> Args({20, 1000})
    ;

}
} // namespace ghostbasil
