#include <benchmark/benchmark.h>
#include <random>
#include <iostream>
#include <iomanip>
#include <ghostbasil/optimization/lasso.hpp>

namespace ghostbasil {
namespace {

struct LassoFixture
    : benchmark::Fixture
{
    static auto make_input(
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

    template <class SGType, class StrongBetaType,
              class ASType, class AOType, class ASOType, class IAType>
    static void reset(
            const SGType& orig_strong_grad,
            StrongBetaType& strong_beta,
            SGType& strong_grad,
            ASType& active_set,
            AOType& active_order,
            ASOType& active_set_ordered,
            IAType& is_active,
            size_t& n_cds,
            size_t& n_lmdas,
            double& rsq)
    {
        Eigen::Map<Eigen::VectorXd> strong_beta_view(
                strong_beta.data(), strong_beta.size());
        strong_beta_view.setZero();
        strong_grad = orig_strong_grad;
        active_set.clear();
        active_order.clear();
        active_set_ordered.clear();
        std::fill(is_active.begin(), is_active.end(), false);
        n_cds = 0;
        n_lmdas = 0;
        rsq = 0;
    }
};

BENCHMARK_DEFINE_F(LassoFixture, lasso_bench)(benchmark::State& state) 
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

    auto strong_order = strong_set;

    const Eigen::VectorXd strong_A_diag = A.diagonal();

    std::vector<double> strong_beta(strong_set.size(), 0);
    std::vector<double> strong_grad(strong_set.size());
    for (size_t i = 0; i < strong_grad.size(); ++i) {
        strong_grad[i] = r[strong_set[i]];
    }
    std::vector<uint32_t> active_set;
    std::vector<uint32_t> active_order;
    std::vector<uint32_t> active_set_ordered;
    std::vector<bool> is_active(strong_set.size(), false);

    const auto orig_strong_grad = strong_grad;

    std::vector<Eigen::SparseVector<double>> betas(lmdas.size());
    std::vector<double> rsqs(lmdas.size());

    size_t n_cds = 0;
    size_t n_lmdas = 0;
    double rsq = 0;
        
    for (auto _ : state) {
        state.PauseTiming();
        reset(orig_strong_grad, strong_beta, strong_grad, active_set,
              active_order, active_set_ordered, is_active, n_cds, n_lmdas, rsq);
        state.ResumeTiming();
        lasso(A, s, strong_set, strong_order, 
              strong_A_diag, lmdas, max_cds, thr, rsq, strong_beta, 
              strong_grad, active_set, active_order, active_set_ordered,
              is_active, betas, rsqs, 
              n_cds, n_lmdas);
    }

    state.counters["n_cds"] = n_cds;
    state.counters["n_active_max"] = active_set.size();
}

BENCHMARK_REGISTER_F(LassoFixture, lasso_bench)
    -> Arg(10)
    -> Arg(50)
    -> Arg(100)
    -> Arg(500)
    -> Arg(1000)
    -> Arg(2000);

}
} // namespace ghostbasil
