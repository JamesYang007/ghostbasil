#include <Rcpp.h>
#include <RcppEigen.h>
#include <ghostbasil/optimization/basil.hpp>
#include <thread>
#include "rcpp_block_matrix.hpp"

using namespace Rcpp;

// [[Rcpp::plugins(openmp)]]

template <class AType>
List basil__(
        const AType& A, 
        const Eigen::Map<Eigen::VectorXd> r,
        double alpha,
        const Eigen::Map<Eigen::VectorXd> penalty,
        const Eigen::Map<Eigen::VectorXd> user_lmdas,
        size_t max_n_lambdas,
        size_t n_lambdas_iter,
        size_t delta_strong_size,
        size_t max_strong_size,
        size_t max_n_cds,
        double thr,
        double min_ratio,
        size_t n_threads)
{
    using namespace ghostbasil::lasso;

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

    auto check_user_interrupt = [&](auto n_cds) {
        if (n_cds % 100 == 0) {
            Rcpp::checkUserInterrupt();
        }
    };

    try {
        basil(
                A, r, alpha, penalty, user_lmdas, max_n_lambdas, n_lambdas_iter,
                delta_strong_size, max_strong_size, max_n_cds, thr, 
                min_ratio, n_threads,
                betas, lmdas, rsqs,
                check_user_interrupt);
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
List basil_dense__(
        const Eigen::Map<Eigen::MatrixXd> A,
        const Eigen::Map<Eigen::VectorXd> r,
        double alpha,
        const Eigen::Map<Eigen::VectorXd> penalty,
        const Eigen::Map<Eigen::VectorXd> user_lmdas,
        size_t max_n_lambdas,
        size_t n_lambdas_iter,
        size_t delta_strong_size,
        size_t max_strong_size,
        size_t max_n_cds,
        double thr,
        double min_ratio,
        size_t n_threads)
{
    return basil__(A, r, alpha, penalty, user_lmdas, max_n_lambdas,
            n_lambdas_iter, delta_strong_size,
            max_strong_size, max_n_cds, thr, min_ratio, n_threads);
}

// [[Rcpp::export]]
List basil_block_dense__(
        SEXP A,
        const Eigen::Map<Eigen::VectorXd> r,
        double alpha,
        const Eigen::Map<Eigen::VectorXd> penalty,
        const Eigen::Map<Eigen::VectorXd> user_lmdas,
        size_t max_n_lambdas,
        size_t n_lambdas_iter,
        size_t delta_strong_size,
        size_t max_strong_size,
        size_t max_n_cds,
        double thr,
        double min_ratio,
        size_t n_threads)
{
    auto bmw = Rcpp::as<BlockMatrixWrap>(A);

    // gets the internal BlockMatrix class
    const auto& bm = bmw.internal();
    
    return basil__(
            bm, r, alpha, penalty, user_lmdas, max_n_lambdas,
            n_lambdas_iter, delta_strong_size,
            max_strong_size, max_n_cds, thr, min_ratio, n_threads);
}

// [[Rcpp::export]]
List basil_ghost__(
        SEXP A,
        const Eigen::Map<Eigen::VectorXd> r,
        double alpha,
        const Eigen::Map<Eigen::VectorXd> penalty,
        const Eigen::Map<Eigen::VectorXd> user_lmdas,
        size_t max_n_lambdas,
        size_t n_lambdas_iter,
        size_t delta_strong_size,
        size_t max_strong_size,
        size_t max_n_cds,
        double thr,
        double min_ratio,
        size_t n_threads)
{
    auto gmw = Rcpp::as<GhostMatrixWrap>(A);

    // gets the internal GhostMatrix class
    const auto& gm = gmw.internal();
    
    return basil__(
            gm, r, alpha, penalty, user_lmdas, max_n_lambdas,
            n_lambdas_iter, delta_strong_size,
            max_strong_size, max_n_cds, thr, min_ratio, n_threads);
}

// [[Rcpp::export]]
List basil_block_ghost__(
        SEXP A,
        const Eigen::Map<Eigen::VectorXd> r,
        double alpha,
        const Eigen::Map<Eigen::VectorXd> penalty,
        const Eigen::Map<Eigen::VectorXd> user_lmdas,
        size_t max_n_lambdas,
        size_t n_lambdas_iter,
        size_t delta_strong_size,
        size_t max_strong_size,
        size_t max_n_cds,
        double thr,
        double min_ratio,
        size_t n_threads)
{
    auto bgmw = Rcpp::as<BlockGhostMatrixWrap>(A);

    // gets the internal GhostMatrix class
    const auto& bgm = bgmw.internal();
    
    return basil__(
            bgm, r, alpha, penalty, user_lmdas, max_n_lambdas,
            n_lambdas_iter, delta_strong_size,
            max_strong_size, max_n_cds, thr, min_ratio, n_threads);
}

// [[Rcpp::export]]
List objective_sparse__(
        const Eigen::Map<Eigen::MatrixXd> A,
        const Eigen::Map<Eigen::VectorXd> r,
        const Eigen::Map<Eigen::VectorXd> penalty,
        double alpha,
        double lmda,
        const Eigen::Map<Eigen::SparseMatrix<double>> beta)
{
    using namespace ghostbasil::lasso;

    double out = 0;
    std::string error;
    try {
        out = objective(A, r, penalty, alpha, lmda, beta.col(0));
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
        const Eigen::Map<Eigen::VectorXd> penalty,
        double alpha,
        double lmda,
        const Eigen::Map<Eigen::VectorXd> beta)
{
    using namespace ghostbasil::lasso;

    double out = 0;
    std::string error;
    try {
        out = objective(A, r, penalty, alpha, lmda, beta);
    }
    catch (const std::exception& e) {
        error = e.what();
    }

    return List::create(
            Named("objective")=out,
            Named("error")=error);
}
