#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

namespace ghostbasil {
namespace lasso_util {

template <class F>
auto make_lasso_input(F generate_dataset)
{
    auto dataset = generate_dataset();
    auto&& A = std::get<0>(dataset);
    auto&& r = std::get<1>(dataset);
    auto&& s = std::get<2>(dataset);
    auto&& strong_set = std::get<3>(dataset);
    auto&& lmdas = std::get<4>(dataset);
    auto&& expected_betas = std::get<5>(dataset);
    auto&& expected_objs = std::get<6>(dataset);

    size_t p = r.size();

    std::vector<double> strong_grad(strong_set.size());
    for (size_t i = 0; i < strong_grad.size(); ++i) {
        strong_grad[i] = r[strong_set[i]];
    }

    Eigen::Vector<double, Eigen::Dynamic> strong_beta(strong_set.size());
    strong_beta.setZero();
    
    Eigen::Vector<double, Eigen::Dynamic> strong_A_diag(strong_set.size());
    for (int i = 0; i < strong_A_diag.size(); ++i) {
        auto k = strong_set[i];
        strong_A_diag[i] = A(k,k);
    }

    using sp_vec_t = Eigen::SparseVector<double>;
    Eigen::Vector<sp_vec_t, Eigen::Dynamic> betas(lmdas.size());
    betas.fill(sp_vec_t(p));
    std::vector<uint32_t> active_set;
    std::vector<bool> is_active(strong_set.size(), false);
    std::vector<double> rsqs(lmdas.size());
    size_t n_cds = 0;
    size_t n_lmdas = 0;

    double rsq = 0;

    return std::make_tuple(
            std::move(A), std::move(r), std::move(s),
            std::move(strong_set), std::move(strong_A_diag), 
            std::move(lmdas), rsq, 
            std::move(strong_beta),
            std::move(strong_grad),
            std::move(active_set),
            std::move(is_active),
            std::move(betas),
            std::move(rsqs),
            n_cds, n_lmdas,
            std::move(expected_betas),
            std::move(expected_objs));
}

} // namespace lasso_util
} // namespace ghostbasil
