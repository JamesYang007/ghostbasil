#include <Rcpp.h>
#include <RcppEigen.h>
#include <ghostbasil/optimization/group_lasso.hpp>

using namespace Rcpp;

// [[Rcpp::export]]
List update_group_coeffs__(
    const Eigen::Map<Eigen::VectorXd> L,
    const Eigen::Map<Eigen::VectorXd> v,
    double lmda,
    double s,
    double tol=1e-8,
    size_t max_iters=1000
)
{
    Eigen::VectorXd x_sol;
    size_t iters = 0;
    ghostbasil::GroupLasso<double> gl(L.size(), L.size());
    gl.update_group_coefficients(
        L, v, lmda, s, tol, max_iters, x_sol, iters
    );
    return List::create(
        Named("beta", x_sol),
        Named("iters", iters)
    );
}

// [[Rcpp::export]]
List group_lasso__(
    const Eigen::Map<Eigen::MatrixXd> A,
    const Eigen::Map<Eigen::VectorXi> groups,
    double s,
    const Eigen::Map<Eigen::VectorXi> strong_set,
    const Eigen::Map<Eigen::VectorXi> strong_begins,
    const Eigen::Map<Eigen::VectorXd> strong_A_diag,
    const Eigen::Map<Eigen::VectorXd> lmdas,
    size_t max_cds,
    double thr,
    double newton_tol,
    size_t newton_max_iters,
    double rsq,
    Eigen::Map<Eigen::VectorXd> strong_beta, 
    Eigen::Map<Eigen::VectorXd> strong_grad,
    std::vector<int> active_set,
    std::vector<int> active_begins,
    std::vector<int> active_order,
    Eigen::Map<Eigen::VectorXi> is_active
)
{
    const auto p = A.cols();
    ghostbasil::GroupLasso<double> gl(p, p);
    
    active_set.reserve(strong_set.size());
    active_begins.reserve(strong_set.size());
    active_order.reserve(strong_set.size());

    std::vector<Eigen::SparseVector<double>> betas(lmdas.size());
    std::vector<double> rsqs(lmdas.size());
    size_t n_cds = 0;
    size_t n_lmdas = 0;

    std::string error;

    auto check_user_interrupt = [&](auto n_cds) {
        if (n_cds % 100 == 0) {
            Rcpp::checkUserInterrupt();
        }
    };

    try {
        gl.group_lasso(
            A, groups, s, strong_set, strong_begins, strong_A_diag,
            lmdas, max_cds, thr, newton_tol, newton_max_iters, rsq,
            strong_beta, strong_grad, active_set, active_begins, active_order,
            is_active, betas, rsqs, n_cds, n_lmdas, check_user_interrupt
        );
    } catch (const std::exception& e) {
        error = e.what();
    }

    // convert the list of sparse vectors into a sparse matrix
    Eigen::SparseMatrix<double> mat_betas;
    if (betas.size()) {
        auto p = betas[0].size();
        mat_betas.resize(p, betas.size());
        for (size_t i = 0; i < betas.size(); ++i) {
            mat_betas.col(i) = betas[i];
        }
    }

    return List::create(
        Named("betas")=std::move(mat_betas),
        Named("rsqs")=std::move(rsqs),
        Named("error")=std::move(error)
    );
}