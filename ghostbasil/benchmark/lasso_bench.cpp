#include <benchmark/benchmark.h>
#include <random>
#include <iostream>
#include <iomanip>
#include <ghostbasil/optimization/lasso.hpp>
#include <ghostbasil/matrix/block_matrix.hpp>
#include <tools/matrix/block_matrix.hpp>

namespace ghostbasil {
namespace lasso {
namespace {

struct LassoFixture
    : benchmark::Fixture
{
    static auto generate_data(
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

    template <class AType, class RType>
    static auto make_input(const AType& A, const RType& r)
    {
        size_t p = A.cols();

        std::vector<double> lmdas(100);
        double factor = std::pow(1e-6, 1./(lmdas.size()-1));
        lmdas[0] = r.array().abs().maxCoeff();
        for (size_t i = 1; i < lmdas.size(); ++i) {
            lmdas[i] = lmdas[i-1] * factor;
        }

        std::vector<int32_t> strong_set(p);
        std::iota(strong_set.begin(), strong_set.end(), 0);

        auto strong_order = strong_set;

        Eigen::VectorXd strong_A_diag(p);
        for (size_t i = 0; i < strong_A_diag.size(); ++i) {
            strong_A_diag[i] = A.coeff(i,i);
        }

        std::vector<double> strong_beta(strong_set.size(), 0);
        std::vector<double> strong_grad(strong_set.size());
        for (size_t i = 0; i < strong_grad.size(); ++i) {
            strong_grad[i] = r[strong_set[i]];
        }
        std::vector<int32_t> active_set;
        std::vector<int32_t> active_order;
        std::vector<int32_t> active_set_ordered;
        std::vector<int32_t> is_active(strong_set.size(), false);

        const auto orig_strong_grad = strong_grad;

        std::vector<Eigen::SparseVector<double>> betas(lmdas.size());
        std::vector<double> rsqs(lmdas.size());

        size_t n_cds = 0;
        size_t n_lmdas = 0;
        double rsq = 0;
        size_t max_cds = 100000;
        double thr = 1e-7;

        return std::make_tuple(
                strong_set, strong_order, 
                strong_A_diag, lmdas, max_cds, thr, rsq, strong_beta, 
                strong_grad, active_set, active_order, active_set_ordered,
                is_active, betas, rsqs, 
                n_cds, n_lmdas);
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

BENCHMARK_DEFINE_F(LassoFixture, dense)(benchmark::State& state) 
{
    size_t p = state.range(0);
    size_t n = 100;
    size_t seed = 30;
    double s = 0.5;

    auto&& data = generate_data(n, p, seed);
    auto&& A = std::get<0>(data);
    auto&& r = std::get<1>(data);

    auto&& input = make_input(A, r);
    auto&& strong_set = std::get<0>(input);
    auto&& strong_order = std::get<1>(input);
    auto&& strong_A_diag = std::get<2>(input);
    auto&& lmdas = std::get<3>(input);
    auto&& max_cds = std::get<4>(input);
    auto&& thr = std::get<5>(input);
    auto&& rsq = std::get<6>(input);
    auto&& strong_beta = std::get<7>(input);
    auto&& strong_grad = std::get<8>(input);
    auto orig_strong_grad = strong_grad;
    auto&& active_set = std::get<9>(input);
    auto&& active_order = std::get<10>(input);
    auto&& active_set_ordered = std::get<11>(input);
    auto&& is_active = std::get<12>(input);
    auto&& betas = std::get<13>(input);
    auto&& rsqs = std::get<14>(input);
    auto&& n_cds = std::get<15>(input);
    auto&& n_lmdas = std::get<16>(input);

    for (auto _ : state) {
        state.PauseTiming();
        reset(orig_strong_grad, strong_beta, strong_grad, active_set,
              active_order, active_set_ordered, is_active, n_cds, n_lmdas, rsq);
        LassoParamPack<
            std::decay_t<decltype(A)>, double, int, int
        > pack(
            A, s, strong_set, strong_order, strong_A_diag,
            lmdas, max_cds, thr, rsq, strong_beta, strong_grad,
            active_set, active_order, active_set_ordered,
            is_active, betas, rsqs, n_cds, n_lmdas            
        );
        state.ResumeTiming();
        fit(pack);
        n_cds = pack.n_cds;
    }

    state.counters["n_cds"] = n_cds;
    state.counters["n_active_max"] = active_set.size();
}

BENCHMARK_REGISTER_F(LassoFixture, dense)
    -> Arg(10)
    -> Arg(50)
    -> Arg(100)
    -> Arg(500)
    -> Arg(1000)
    -> Arg(2000);

// ==========================================
// BENCHMARK BlockMatrix<Dense>
// ==========================================

struct LassoBlockFixture
    : LassoFixture,
      tools::BlockMatrixUtil
{
    using butil = tools::BlockMatrixUtil;
    using bmat_t = BlockMatrix<Eigen::MatrixXd>;

    static auto generate_data(size_t n, size_t p, size_t L, size_t seed)
    {
        std::vector<Eigen::MatrixXd> mat_list(L);
        Eigen::VectorXd r(L * p);
        for (size_t l = 0; l < L; ++l) {
            auto&& out = LassoFixture::generate_data(n, p, seed);
            mat_list[l] = std::move(std::get<0>(out));
            r.segment(l*p, p) = std::get<1>(out);
        }
        return std::make_tuple(mat_list, r);
    }
};

BENCHMARK_DEFINE_F(LassoBlockFixture, block_dense)(benchmark::State& state) 
{
    size_t L = state.range(0);
    size_t p = state.range(1);
    size_t n = 100;
    size_t seed = 30;
    double s = 0.5;

    auto&& data = generate_data(n, p, L, seed);
    auto&& mat_list = std::get<0>(data);
    bmat_t A(mat_list);
    auto&& r = std::get<1>(data);

    auto&& input = make_input(A, r);
    auto&& strong_set = std::get<0>(input);
    auto&& strong_order = std::get<1>(input);
    auto&& strong_A_diag = std::get<2>(input);
    auto&& lmdas = std::get<3>(input);
    auto&& max_cds = std::get<4>(input);
    auto&& thr = std::get<5>(input);
    auto&& rsq = std::get<6>(input);
    auto&& strong_beta = std::get<7>(input);
    auto&& strong_grad = std::get<8>(input);
    auto orig_strong_grad = strong_grad;
    auto&& active_set = std::get<9>(input);
    auto&& active_order = std::get<10>(input);
    auto&& active_set_ordered = std::get<11>(input);
    auto&& is_active = std::get<12>(input);
    auto&& betas = std::get<13>(input);
    auto&& rsqs = std::get<14>(input);
    auto&& n_cds = std::get<15>(input);
    auto&& n_lmdas = std::get<16>(input);
        
    for (auto _ : state) {
        state.PauseTiming();
        reset(orig_strong_grad, strong_beta, strong_grad, active_set,
              active_order, active_set_ordered, is_active, n_cds, n_lmdas, rsq);
        LassoParamPack<
            std::decay_t<decltype(A)>, double, int, int
        > pack(
            A, s, strong_set, strong_order, strong_A_diag,
            lmdas, max_cds, thr, rsq, strong_beta, strong_grad,
            active_set, active_order, active_set_ordered,
            is_active, betas, rsqs, n_cds, n_lmdas            
        );
        state.ResumeTiming();
        fit(pack);
        n_cds = pack.n_cds;
    }

    state.counters["n_cds"] = n_cds;
    state.counters["n_active_max"] = active_set.size();
}

BENCHMARK_REGISTER_F(LassoBlockFixture, block_dense)
    -> Args({2, 10})
    -> Args({2, 50})
    -> Args({2, 100})
    -> Args({2, 500})

    -> Args({2,  100})
    -> Args({16, 100})
    -> Args({32, 100})
    -> Args({64, 100})
    ;
}
} // namespace lasso
} // namespace ghostbasil
