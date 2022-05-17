#include <Rcpp.h>
#include <RcppEigen.h>
#include <ghostbasil/lasso.hpp>

using namespace Rcpp;

// [[Rcpp::export]]
List fit_basil__(
        const Eigen::Map<Eigen::MatrixXd> A, // TODO: change type of A?
        const Eigen::Map<Eigen::VectorXd> y,
        double s,
        size_t n_knockoffs,
        size_t n_lambdas,
        size_t n_lambdas_iter,
        size_t strong_size,
        size_t delta_strong_size,
        size_t n_iters,
        size_t max_cds,
        double thr)
{
    std::vector<Eigen::SparseMatrix<double>> betas;
    std::vector<Eigen::VectorXd> lmdas;
    ghostbasil::fit_basil(
            A, y, s, n_knockoffs, n_lambdas, n_lambdas_iter,
            strong_size, delta_strong_size, n_iters, max_cds, thr,
              betas, lmdas);

    return List::create(
            Named("betas")=std::move(betas),
            Named("lmdas")=std::move(lmdas));
}
