#include <benchmark/benchmark.h>
#include <ghostbasil/lasso.hpp>
#include <random>
#include <iostream>
#include <iomanip>

namespace ghostbasil {
namespace {

static inline auto make_input(
        size_t n, size_t p, size_t seed)
{
    std::mt19937 gen(seed);
    std::normal_distribution<> norm(0., 1.);
    Eigen::MatrixXd X = Eigen::MatrixXd::NullaryExpr(n, p,
            [&](auto, auto) { return norm(gen); });
    Eigen::VectorXd beta(p); 
    beta.setZero();
    std::uniform_int_distribution<> unif(0, p-1);
    for (size_t k = 0; k < 10; ++k) {
        beta[unif(gen)] = norm(gen);
    }
    Eigen::VectorXd y = X * beta + Eigen::VectorXd::NullaryExpr(n,
            [&](auto) { return norm(gen); });
    Eigen::MatrixXd A = X.transpose() * X / n;
    Eigen::VectorXd r = X.transpose() * y / n;
    return std::make_tuple(A, r);
}

template <class SGType, class StrongBetaType, class BetasType,
          class ASType, class IAType>
static inline void reset(
        const SGType& orig_strong_grad,
        StrongBetaType& strong_beta,
        BetasType& betas,
        SGType& strong_grad,
        ASType& active_set,
        IAType& is_active,
        size_t& n_cds,
        size_t& n_lmdas)
{
    strong_beta.setZero();
    betas.setZero();
    strong_grad = orig_strong_grad;
    active_set.clear();
    std::fill(is_active.begin(), is_active.end(), false);
    n_cds = 0;
    n_lmdas = 0;
}

static void BM_fit_lasso(benchmark::State& state) 
{
    size_t p = state.range(0);
    size_t n = 100;
    size_t seed = 30;
    size_t max_cds = 100000;
    double thr = 1e-7;
    double s = 0.5;

    auto input = make_input(n, p, seed);
    auto& A = std::get<0>(input);
    auto& r = std::get<1>(input);

    std::vector<double> lmdas(100);
    double factor = std::pow(1e-6, 1./(lmdas.size()-1));
    lmdas[0] = r.array().abs().maxCoeff();
    for (size_t i = 1; i < lmdas.size(); ++i) {
        lmdas[i] = lmdas[i-1] * factor;
    }

    std::vector<uint32_t> strong_set(p);
    std::iota(strong_set.begin(), strong_set.end(), 0);

    Eigen::SparseVector<double> strong_beta(p);
    Eigen::SparseMatrix<double> betas(p, lmdas.size());
    std::vector<double> strong_grad(strong_set.size());
    for (size_t i = 0; i < strong_grad.size(); ++i) {
        strong_grad[i] = r[strong_set[i]];
    }
    auto orig_strong_grad = strong_grad;
    std::vector<uint32_t> active_set;
    std::vector<bool> is_active(strong_set.size(), false);

    size_t n_cds = 0;
    size_t n_lmdas = 0;
        
    for (auto _ : state) {
        state.PauseTiming();
        reset(orig_strong_grad, strong_beta, betas, strong_grad, active_set,
              is_active, n_cds, n_lmdas);
        state.ResumeTiming();
        fit_lasso(A, s, strong_set, lmdas, max_cds, thr, strong_beta, betas, 
                  strong_grad, active_set, is_active, 
                  n_cds, n_lmdas);
    }

    state.counters["n_cds"] = n_cds;
}

BENCHMARK(BM_fit_lasso)
    -> Arg(10)
    -> Arg(50)
    -> Arg(100)
    -> Arg(500)
    -> Arg(1000)
    -> Arg(2000);

}
} // namespace ghostbasil
