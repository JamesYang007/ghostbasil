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
    using namespace ghostbasil::group_lasso;

    Eigen::VectorXd x_sol;
    size_t iters = 0;
    GroupLassoBufferPack<double> buffer_pack(
        L.size(), L.size(), L.size()
    );
    update_coefficients(
        L, v, lmda, s, tol, max_iters, x_sol, iters, 
        buffer_pack.buffer1, buffer_pack.buffer2
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
    const Eigen::Map<Eigen::VectorXi> group_sizes,
    double s,
    const Eigen::Map<Eigen::VectorXi> strong_set,
    const Eigen::Map<Eigen::VectorXi> strong_g1,
    const Eigen::Map<Eigen::VectorXi> strong_g2,
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
    std::vector<int> active_g1,
    std::vector<int> active_g2,
    std::vector<int> active_begins,
    std::vector<int> active_order,
    Eigen::Map<Eigen::VectorXi> is_active
)
{
    using namespace ghostbasil::group_lasso;

    active_set.reserve(strong_set.size());
    active_g1.reserve(strong_set.size());
    active_g2.reserve(strong_set.size());
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

    GroupLassoParamPack<
        Eigen::Map<Eigen::MatrixXd>,
        double,
        int,
        int
    > pack(
        A, groups, group_sizes, s, strong_set, strong_g1, strong_g2, strong_begins,
        strong_A_diag, lmdas, max_cds, thr, newton_tol, newton_max_iters,
        rsq, strong_beta, strong_grad, active_set, active_g1, active_g2,
        active_begins, active_order,
        is_active, betas, rsqs, n_cds, n_lmdas
    );

    try {
        fit(pack, check_user_interrupt);
    } catch (const std::exception& e) {
        error = e.what();
    }

    // convert the list of sparse vectors into a sparse matrix
    Eigen::SparseMatrix<double> mat_betas;
    if (pack.n_lmdas) {
        auto p = betas[0].size();
        mat_betas.resize(p, pack.n_lmdas);
        for (size_t i = 0; i < pack.n_lmdas; ++i) {
            mat_betas.col(i) = betas[i];
        }
    }

    return List::create(
        Named("n_cds")=pack.n_cds,
        Named("betas")=std::move(mat_betas),
        Named("rsqs")=std::move(rsqs),
        Named("error")=std::move(error)
    );
}