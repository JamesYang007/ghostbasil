#include <Rcpp.h>
#include <RcppEigen.h>
#include <ghostbasil/optimization/basil.hpp>
#include <thread>
#include "rcpp_block_matrix.hpp"

using namespace Rcpp;
using ckpt_t = ghostbasil::lasso::BasilCheckpoint<double, int, int>;
using value_t = typename ckpt_t::value_t;
using vec_value_t = typename ckpt_t::vec_value_t;
using dyn_vec_value_t = typename ckpt_t::dyn_vec_value_t;
using dyn_vec_index_t = typename ckpt_t::dyn_vec_index_t;
using dyn_vec_bool_t = typename ckpt_t::dyn_vec_bool_t;

// [[Rcpp::plugins(openmp)]]

auto list_to_checkpoint(List checkpoint)
{
    if (checkpoint.size() == 0) return ckpt_t();
    return ckpt_t(
        as<dyn_vec_index_t>(checkpoint["strong_set"]),
        as<dyn_vec_index_t>(checkpoint["strong_order"]),
        as<dyn_vec_value_t>(checkpoint["strong_beta"]),
        as<dyn_vec_value_t>(checkpoint["strong_grad"]),
        as<dyn_vec_value_t>(checkpoint["strong_A_diag"]),
        as<dyn_vec_index_t>(checkpoint["active_set"]),
        as<dyn_vec_index_t>(checkpoint["active_order"]),
        as<dyn_vec_index_t>(checkpoint["active_set_ordered"]),
        as<dyn_vec_bool_t>(checkpoint["is_active"]),
        as<Eigen::Map<vec_value_t>>(checkpoint["grad"]),
        as<value_t>(checkpoint["rsq"])
    );
}

auto checkpoint_to_list(const ckpt_t& checkpoint)
{
    return List::create(
        Named("strong_set")=checkpoint.strong_set,
        Named("strong_order")=checkpoint.strong_order,
        Named("strong_beta")=checkpoint.strong_beta,
        Named("strong_grad")=checkpoint.strong_grad,
        Named("strong_A_diag")=checkpoint.strong_A_diag,
        Named("active_set")=checkpoint.active_set,
        Named("active_order")=checkpoint.active_order,
        Named("active_set_ordered")=checkpoint.active_set_ordered,
        Named("is_active")=checkpoint.is_active,
        Named("grad")=checkpoint.grad,
        Named("rsq")=checkpoint.rsq
    );
}

auto list_to_checkpoint(ListOf<List> checkpoints)
{
    using ckpt_t = ghostbasil::lasso::BasilCheckpoint<double, int, int>;
    std::vector<ckpt_t> checkpoints_cvt;
    checkpoints_cvt.reserve(checkpoints.size());
    for (size_t i = 0; i < checkpoints.size(); ++i) {
        checkpoints_cvt.emplace_back(
            list_to_checkpoint(checkpoints[i])
        );
    }
    return checkpoints_cvt;
}

auto checkpoint_to_list(const std::vector<ckpt_t>& checkpoints)
{
    ListOf<List> lst(checkpoints.size());
    for (size_t i = 0; i < checkpoints.size(); ++i) {
        lst[i] = checkpoint_to_list(checkpoints[i]);
    }
    return lst;
}

template <class AType, class CheckpointType>
List basil__(
        const AType& A, 
        const Eigen::Map<Eigen::VectorXd> r,
        double alpha,
        const Eigen::Map<Eigen::VectorXd> penalty,
        const Eigen::Map<Eigen::VectorXd> user_lmdas,
        size_t max_n_lambdas,
        size_t n_lambdas_iter,
        bool use_strong_rule,
        size_t delta_strong_size,
        size_t max_strong_size,
        size_t max_n_cds,
        double thr,
        double min_ratio,
        size_t n_threads,
        CheckpointType&& checkpoint)
{
    using namespace ghostbasil::lasso;

    std::vector<Eigen::SparseVector<double>> betas;
    std::vector<double> lmdas;
    std::vector<double> rsqs;

    // slight optimization: reserve spaces ahead of time
    const size_t capacity = std::max(
        max_n_lambdas, static_cast<size_t>(user_lmdas.size())
    );
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
                use_strong_rule, delta_strong_size, max_strong_size, max_n_cds, thr, 
                min_ratio, n_threads,
                betas, lmdas, rsqs, checkpoint,
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
        Named("error")=std::move(error),
        Named("checkpoint")=checkpoint_to_list(checkpoint)
    );
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
        bool use_strong_rule,
        size_t delta_strong_size,
        size_t max_strong_size,
        size_t max_n_cds,
        double thr,
        double min_ratio,
        size_t n_threads,
        List checkpoint)
{
    auto&& checkpoint_cvt = list_to_checkpoint(checkpoint);
    return basil__(A, r, alpha, penalty, user_lmdas, max_n_lambdas,
            n_lambdas_iter, use_strong_rule, delta_strong_size,
            max_strong_size, max_n_cds, thr, min_ratio, n_threads, checkpoint_cvt);
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
        bool use_strong_rule,
        size_t delta_strong_size,
        size_t max_strong_size,
        size_t max_n_cds,
        double thr,
        double min_ratio,
        size_t n_threads,
        ListOf<List> checkpoints)
{
    auto bmw = Rcpp::as<BlockMatrixWrap>(A);

    // gets the internal BlockMatrix class
    const auto& bm = bmw.internal();
    auto&& checkpoints_cvt = list_to_checkpoint(checkpoints);
    
    return basil__(
            bm, r, alpha, penalty, user_lmdas, max_n_lambdas,
            n_lambdas_iter, use_strong_rule, delta_strong_size,
            max_strong_size, max_n_cds, thr, min_ratio, n_threads,
            checkpoints_cvt);
}

//// [[Rcpp::export]]
//List basil_ghost__(
//        SEXP A,
//        const Eigen::Map<Eigen::VectorXd> r,
//        double alpha,
//        const Eigen::Map<Eigen::VectorXd> penalty,
//        const Eigen::Map<Eigen::VectorXd> user_lmdas,
//        size_t max_n_lambdas,
//        size_t n_lambdas_iter,
//        bool use_strong_rule,
//        size_t delta_strong_size,
//        size_t max_strong_size,
//        size_t max_n_cds,
//        double thr,
//        double min_ratio,
//        size_t n_threads)
//{
//    auto gmw = Rcpp::as<GhostMatrixWrap>(A);
//
//    // gets the internal GhostMatrix class
//    const auto& gm = gmw.internal();
//    
//    return basil__(
//            gm, r, alpha, penalty, user_lmdas, max_n_lambdas,
//            n_lambdas_iter, use_strong_rule, delta_strong_size,
//            max_strong_size, max_n_cds, thr, min_ratio, n_threads);
//}
//
//// [[Rcpp::export]]
//List basil_block_ghost__(
//        SEXP A,
//        const Eigen::Map<Eigen::VectorXd> r,
//        double alpha,
//        const Eigen::Map<Eigen::VectorXd> penalty,
//        const Eigen::Map<Eigen::VectorXd> user_lmdas,
//        size_t max_n_lambdas,
//        size_t n_lambdas_iter,
//        bool use_strong_rule,
//        size_t delta_strong_size,
//        size_t max_strong_size,
//        size_t max_n_cds,
//        double thr,
//        double min_ratio,
//        size_t n_threads)
//{
//    auto bgmw = Rcpp::as<BlockGhostMatrixWrap>(A);
//
//    // gets the internal GhostMatrix class
//    const auto& bgm = bgmw.internal();
//    
//    return basil__(
//            bgm, r, alpha, penalty, user_lmdas, max_n_lambdas,
//            n_lambdas_iter, use_strong_rule, delta_strong_size,
//            max_strong_size, max_n_cds, thr, min_ratio, n_threads);
//}

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
