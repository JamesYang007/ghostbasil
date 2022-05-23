#include <Rcpp.h>
#include <RcppEigen.h>
#include <ghostbasil/lasso.hpp>

using namespace Rcpp;

// [[Rcpp::export]]
List fit_basil__(
        const Eigen::Map<Eigen::MatrixXd> A, // TODO: change type of A?
        const Eigen::Map<Eigen::VectorXd> y,
        double s,
        const Eigen::Map<Eigen::VectorXd> user_lmdas,
        size_t max_n_lambdas,
        size_t n_lambdas_iter,
        size_t strong_size,
        size_t delta_strong_size,
        size_t max_strong_size,
        size_t max_n_cds,
        double thr)
{
    std::vector<Eigen::SparseMatrix<double>> betas;
    std::vector<Eigen::VectorXd> lmdas;
    std::string error;

    try {
        ghostbasil::fit_basil(
                A, y, s, user_lmdas, max_n_lambdas, n_lambdas_iter,
                strong_size, delta_strong_size, max_strong_size, max_n_cds, thr,
                betas, lmdas);
    }
    catch (const std::exception& e) {
        error = e.what();
    }

    return List::create(
            Named("betas")=std::move(betas),
            Named("lmdas")=std::move(lmdas),
            Named("error")=std::move(error));
}

// [[Rcpp::export]]
List objective_sparse__(
        const Eigen::Map<Eigen::MatrixXd> A,
        const Eigen::Map<Eigen::VectorXd> r,
        double s,
        double lmda,
        const Eigen::Map<Eigen::SparseMatrix<double>> beta)
{
    double out = 0;
    std::string error;
    try {
        out = ghostbasil::objective(A, r, s, lmda, beta.col(0));
    }
    catch (const std::exception& e) {
        error = e.what();
    }

    return List::create(
            Named("objective")=out,
            Named("error")=error);
}

// [[Rcpp::export]]
List objective_dense__(
        const Eigen::Map<Eigen::MatrixXd> A,
        const Eigen::Map<Eigen::VectorXd> r,
        double s,
        double lmda,
        const Eigen::Map<Eigen::VectorXd> beta)
{
    double out = 0;
    std::string error;
    try {
        out = ghostbasil::objective(A, r, s, lmda, beta);
    }
    catch (const std::exception& e) {
        error = e.what();
    }

    return List::create(
            Named("objective")=out,
            Named("error")=error);
}
