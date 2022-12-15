#include <Rcpp.h>
#include <RcppEigen.h>
#include <ghostbasil/optimization/lasso.hpp>

using namespace Rcpp;

// [[Rcpp::export]]
List lasso__(
    const Eigen::Map<Eigen::MatrixXd> A, 
    double s, 
    const Eigen::Map<Eigen::VectorXi> strong_set, 
    const std::vector<int>& strong_order,
    const Eigen::Map<Eigen::VectorXd> strong_A_diag,
    const Eigen::Map<Eigen::VectorXd> lmdas, 
    size_t max_cds,
    double thr,
    double rsq,
    Eigen::Map<Eigen::VectorXd> strong_beta, 
    Eigen::Map<Eigen::VectorXd> strong_grad,
    std::vector<int> active_set,
    std::vector<int> active_order,
    std::vector<int> active_set_ordered,
    Eigen::Map<Eigen::VectorXi> is_active
)
{
    using namespace ghostbasil::lasso;

    active_set.reserve(strong_set.size());
    active_order.reserve(strong_set.size());
    active_set_ordered.reserve(strong_set.size());

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
        fit(
            A, s, strong_set, strong_order, strong_A_diag,
            lmdas, max_cds, thr, rsq,
            strong_beta, strong_grad, active_set, active_order, active_set_ordered,
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