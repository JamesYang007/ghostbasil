#include <Rcpp.h>
#include <RcppEigen.h>
#include <ghostbasil/optimization/lasso.hpp>
#include <ghostbasil/optimization/basil.hpp>
#include <thread>

using namespace Rcpp;

// [[Rcpp::export]]
List basil__(
        const Eigen::Map<Eigen::MatrixXd> A, // TODO: change type of A?
        const Eigen::Map<Eigen::VectorXd> r,
        double s,
        const Eigen::Map<Eigen::VectorXd> user_lmdas,
        size_t max_n_lambdas,
        size_t n_lambdas_iter,
        size_t strong_size,
        size_t delta_strong_size,
        size_t max_strong_size,
        size_t max_n_cds,
        double thr,
        size_t n_threads)
{
    std::vector<Eigen::SparseVector<double>> betas;
    std::vector<double> lmdas;
    std::vector<double> rsqs;

    // slight optimization: reserve spaces ahead of time
    constexpr size_t capacity = 100;
    betas.reserve(capacity);
    lmdas.reserve(capacity);
    rsqs.reserve(capacity);

    std::string error;

    if (n_threads == -1) {
        n_threads = std::thread::hardware_concurrency();
    }

    try {
        ghostbasil::basil(
                A, r, s, user_lmdas, max_n_lambdas, n_lambdas_iter,
                strong_size, delta_strong_size, max_strong_size, max_n_cds, thr, n_threads,
                betas, lmdas, rsqs);
    }
    catch (const std::exception& e) {
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
            Named("lmdas")=std::move(lmdas),
            Named("rsqs")=std::move(rsqs),
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
