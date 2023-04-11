// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// basil_dense__
List basil_dense__(const Eigen::Map<Eigen::MatrixXd> A, const Eigen::Map<Eigen::VectorXd> r, double alpha, const Eigen::Map<Eigen::VectorXd> penalty, const Eigen::Map<Eigen::VectorXd> user_lmdas, size_t max_n_lambdas, size_t n_lambdas_iter, bool use_strong_rule, bool do_early_exit, size_t delta_strong_size, size_t max_strong_size, size_t max_n_cds, double thr, double min_ratio, size_t n_threads, List checkpoint);
RcppExport SEXP _ghostbasil_basil_dense__(SEXP ASEXP, SEXP rSEXP, SEXP alphaSEXP, SEXP penaltySEXP, SEXP user_lmdasSEXP, SEXP max_n_lambdasSEXP, SEXP n_lambdas_iterSEXP, SEXP use_strong_ruleSEXP, SEXP do_early_exitSEXP, SEXP delta_strong_sizeSEXP, SEXP max_strong_sizeSEXP, SEXP max_n_cdsSEXP, SEXP thrSEXP, SEXP min_ratioSEXP, SEXP n_threadsSEXP, SEXP checkpointSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd> >::type A(ASEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type r(rSEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type penalty(penaltySEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type user_lmdas(user_lmdasSEXP);
    Rcpp::traits::input_parameter< size_t >::type max_n_lambdas(max_n_lambdasSEXP);
    Rcpp::traits::input_parameter< size_t >::type n_lambdas_iter(n_lambdas_iterSEXP);
    Rcpp::traits::input_parameter< bool >::type use_strong_rule(use_strong_ruleSEXP);
    Rcpp::traits::input_parameter< bool >::type do_early_exit(do_early_exitSEXP);
    Rcpp::traits::input_parameter< size_t >::type delta_strong_size(delta_strong_sizeSEXP);
    Rcpp::traits::input_parameter< size_t >::type max_strong_size(max_strong_sizeSEXP);
    Rcpp::traits::input_parameter< size_t >::type max_n_cds(max_n_cdsSEXP);
    Rcpp::traits::input_parameter< double >::type thr(thrSEXP);
    Rcpp::traits::input_parameter< double >::type min_ratio(min_ratioSEXP);
    Rcpp::traits::input_parameter< size_t >::type n_threads(n_threadsSEXP);
    Rcpp::traits::input_parameter< List >::type checkpoint(checkpointSEXP);
    rcpp_result_gen = Rcpp::wrap(basil_dense__(A, r, alpha, penalty, user_lmdas, max_n_lambdas, n_lambdas_iter, use_strong_rule, do_early_exit, delta_strong_size, max_strong_size, max_n_cds, thr, min_ratio, n_threads, checkpoint));
    return rcpp_result_gen;
END_RCPP
}
// basil_block_dense__
List basil_block_dense__(SEXP A, const Eigen::Map<Eigen::VectorXd> r, double alpha, const Eigen::Map<Eigen::VectorXd> penalty, const Eigen::Map<Eigen::VectorXd> user_lmdas, size_t max_n_lambdas, size_t n_lambdas_iter, bool use_strong_rule, bool do_early_exit, size_t delta_strong_size, size_t max_strong_size, size_t max_n_cds, double thr, double min_ratio, size_t n_threads, ListOf<List> checkpoints);
RcppExport SEXP _ghostbasil_basil_block_dense__(SEXP ASEXP, SEXP rSEXP, SEXP alphaSEXP, SEXP penaltySEXP, SEXP user_lmdasSEXP, SEXP max_n_lambdasSEXP, SEXP n_lambdas_iterSEXP, SEXP use_strong_ruleSEXP, SEXP do_early_exitSEXP, SEXP delta_strong_sizeSEXP, SEXP max_strong_sizeSEXP, SEXP max_n_cdsSEXP, SEXP thrSEXP, SEXP min_ratioSEXP, SEXP n_threadsSEXP, SEXP checkpointsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type A(ASEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type r(rSEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type penalty(penaltySEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type user_lmdas(user_lmdasSEXP);
    Rcpp::traits::input_parameter< size_t >::type max_n_lambdas(max_n_lambdasSEXP);
    Rcpp::traits::input_parameter< size_t >::type n_lambdas_iter(n_lambdas_iterSEXP);
    Rcpp::traits::input_parameter< bool >::type use_strong_rule(use_strong_ruleSEXP);
    Rcpp::traits::input_parameter< bool >::type do_early_exit(do_early_exitSEXP);
    Rcpp::traits::input_parameter< size_t >::type delta_strong_size(delta_strong_sizeSEXP);
    Rcpp::traits::input_parameter< size_t >::type max_strong_size(max_strong_sizeSEXP);
    Rcpp::traits::input_parameter< size_t >::type max_n_cds(max_n_cdsSEXP);
    Rcpp::traits::input_parameter< double >::type thr(thrSEXP);
    Rcpp::traits::input_parameter< double >::type min_ratio(min_ratioSEXP);
    Rcpp::traits::input_parameter< size_t >::type n_threads(n_threadsSEXP);
    Rcpp::traits::input_parameter< ListOf<List> >::type checkpoints(checkpointsSEXP);
    rcpp_result_gen = Rcpp::wrap(basil_block_dense__(A, r, alpha, penalty, user_lmdas, max_n_lambdas, n_lambdas_iter, use_strong_rule, do_early_exit, delta_strong_size, max_strong_size, max_n_cds, thr, min_ratio, n_threads, checkpoints));
    return rcpp_result_gen;
END_RCPP
}
// basil_ghost__
List basil_ghost__(SEXP A, const Eigen::Map<Eigen::VectorXd> r, double alpha, const Eigen::Map<Eigen::VectorXd> penalty, const Eigen::Map<Eigen::VectorXd> user_lmdas, size_t max_n_lambdas, size_t n_lambdas_iter, bool use_strong_rule, bool do_early_exit, size_t delta_strong_size, size_t max_strong_size, size_t max_n_cds, double thr, double min_ratio, size_t n_threads, List checkpoint);
RcppExport SEXP _ghostbasil_basil_ghost__(SEXP ASEXP, SEXP rSEXP, SEXP alphaSEXP, SEXP penaltySEXP, SEXP user_lmdasSEXP, SEXP max_n_lambdasSEXP, SEXP n_lambdas_iterSEXP, SEXP use_strong_ruleSEXP, SEXP do_early_exitSEXP, SEXP delta_strong_sizeSEXP, SEXP max_strong_sizeSEXP, SEXP max_n_cdsSEXP, SEXP thrSEXP, SEXP min_ratioSEXP, SEXP n_threadsSEXP, SEXP checkpointSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type A(ASEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type r(rSEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type penalty(penaltySEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type user_lmdas(user_lmdasSEXP);
    Rcpp::traits::input_parameter< size_t >::type max_n_lambdas(max_n_lambdasSEXP);
    Rcpp::traits::input_parameter< size_t >::type n_lambdas_iter(n_lambdas_iterSEXP);
    Rcpp::traits::input_parameter< bool >::type use_strong_rule(use_strong_ruleSEXP);
    Rcpp::traits::input_parameter< bool >::type do_early_exit(do_early_exitSEXP);
    Rcpp::traits::input_parameter< size_t >::type delta_strong_size(delta_strong_sizeSEXP);
    Rcpp::traits::input_parameter< size_t >::type max_strong_size(max_strong_sizeSEXP);
    Rcpp::traits::input_parameter< size_t >::type max_n_cds(max_n_cdsSEXP);
    Rcpp::traits::input_parameter< double >::type thr(thrSEXP);
    Rcpp::traits::input_parameter< double >::type min_ratio(min_ratioSEXP);
    Rcpp::traits::input_parameter< size_t >::type n_threads(n_threadsSEXP);
    Rcpp::traits::input_parameter< List >::type checkpoint(checkpointSEXP);
    rcpp_result_gen = Rcpp::wrap(basil_ghost__(A, r, alpha, penalty, user_lmdas, max_n_lambdas, n_lambdas_iter, use_strong_rule, do_early_exit, delta_strong_size, max_strong_size, max_n_cds, thr, min_ratio, n_threads, checkpoint));
    return rcpp_result_gen;
END_RCPP
}
// basil_block_ghost__
List basil_block_ghost__(SEXP A, const Eigen::Map<Eigen::VectorXd> r, double alpha, const Eigen::Map<Eigen::VectorXd> penalty, const Eigen::Map<Eigen::VectorXd> user_lmdas, size_t max_n_lambdas, size_t n_lambdas_iter, bool use_strong_rule, bool do_early_exit, size_t delta_strong_size, size_t max_strong_size, size_t max_n_cds, double thr, double min_ratio, size_t n_threads, ListOf<List> checkpoints);
RcppExport SEXP _ghostbasil_basil_block_ghost__(SEXP ASEXP, SEXP rSEXP, SEXP alphaSEXP, SEXP penaltySEXP, SEXP user_lmdasSEXP, SEXP max_n_lambdasSEXP, SEXP n_lambdas_iterSEXP, SEXP use_strong_ruleSEXP, SEXP do_early_exitSEXP, SEXP delta_strong_sizeSEXP, SEXP max_strong_sizeSEXP, SEXP max_n_cdsSEXP, SEXP thrSEXP, SEXP min_ratioSEXP, SEXP n_threadsSEXP, SEXP checkpointsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type A(ASEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type r(rSEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type penalty(penaltySEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type user_lmdas(user_lmdasSEXP);
    Rcpp::traits::input_parameter< size_t >::type max_n_lambdas(max_n_lambdasSEXP);
    Rcpp::traits::input_parameter< size_t >::type n_lambdas_iter(n_lambdas_iterSEXP);
    Rcpp::traits::input_parameter< bool >::type use_strong_rule(use_strong_ruleSEXP);
    Rcpp::traits::input_parameter< bool >::type do_early_exit(do_early_exitSEXP);
    Rcpp::traits::input_parameter< size_t >::type delta_strong_size(delta_strong_sizeSEXP);
    Rcpp::traits::input_parameter< size_t >::type max_strong_size(max_strong_sizeSEXP);
    Rcpp::traits::input_parameter< size_t >::type max_n_cds(max_n_cdsSEXP);
    Rcpp::traits::input_parameter< double >::type thr(thrSEXP);
    Rcpp::traits::input_parameter< double >::type min_ratio(min_ratioSEXP);
    Rcpp::traits::input_parameter< size_t >::type n_threads(n_threadsSEXP);
    Rcpp::traits::input_parameter< ListOf<List> >::type checkpoints(checkpointsSEXP);
    rcpp_result_gen = Rcpp::wrap(basil_block_ghost__(A, r, alpha, penalty, user_lmdas, max_n_lambdas, n_lambdas_iter, use_strong_rule, do_early_exit, delta_strong_size, max_strong_size, max_n_cds, thr, min_ratio, n_threads, checkpoints));
    return rcpp_result_gen;
END_RCPP
}
// basil_block_group_ghost__
List basil_block_group_ghost__(SEXP A, const Eigen::Map<Eigen::VectorXd> r, double alpha, const Eigen::Map<Eigen::VectorXd> penalty, const Eigen::Map<Eigen::VectorXd> user_lmdas, size_t max_n_lambdas, size_t n_lambdas_iter, bool use_strong_rule, bool do_early_exit, size_t delta_strong_size, size_t max_strong_size, size_t max_n_cds, double thr, double min_ratio, size_t n_threads, List checkpoint);
RcppExport SEXP _ghostbasil_basil_block_group_ghost__(SEXP ASEXP, SEXP rSEXP, SEXP alphaSEXP, SEXP penaltySEXP, SEXP user_lmdasSEXP, SEXP max_n_lambdasSEXP, SEXP n_lambdas_iterSEXP, SEXP use_strong_ruleSEXP, SEXP do_early_exitSEXP, SEXP delta_strong_sizeSEXP, SEXP max_strong_sizeSEXP, SEXP max_n_cdsSEXP, SEXP thrSEXP, SEXP min_ratioSEXP, SEXP n_threadsSEXP, SEXP checkpointSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type A(ASEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type r(rSEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type penalty(penaltySEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type user_lmdas(user_lmdasSEXP);
    Rcpp::traits::input_parameter< size_t >::type max_n_lambdas(max_n_lambdasSEXP);
    Rcpp::traits::input_parameter< size_t >::type n_lambdas_iter(n_lambdas_iterSEXP);
    Rcpp::traits::input_parameter< bool >::type use_strong_rule(use_strong_ruleSEXP);
    Rcpp::traits::input_parameter< bool >::type do_early_exit(do_early_exitSEXP);
    Rcpp::traits::input_parameter< size_t >::type delta_strong_size(delta_strong_sizeSEXP);
    Rcpp::traits::input_parameter< size_t >::type max_strong_size(max_strong_sizeSEXP);
    Rcpp::traits::input_parameter< size_t >::type max_n_cds(max_n_cdsSEXP);
    Rcpp::traits::input_parameter< double >::type thr(thrSEXP);
    Rcpp::traits::input_parameter< double >::type min_ratio(min_ratioSEXP);
    Rcpp::traits::input_parameter< size_t >::type n_threads(n_threadsSEXP);
    Rcpp::traits::input_parameter< List >::type checkpoint(checkpointSEXP);
    rcpp_result_gen = Rcpp::wrap(basil_block_group_ghost__(A, r, alpha, penalty, user_lmdas, max_n_lambdas, n_lambdas_iter, use_strong_rule, do_early_exit, delta_strong_size, max_strong_size, max_n_cds, thr, min_ratio, n_threads, checkpoint));
    return rcpp_result_gen;
END_RCPP
}
// basil_block_block_group_ghost__
List basil_block_block_group_ghost__(SEXP A, const Eigen::Map<Eigen::VectorXd> r, double alpha, const Eigen::Map<Eigen::VectorXd> penalty, const Eigen::Map<Eigen::VectorXd> user_lmdas, size_t max_n_lambdas, size_t n_lambdas_iter, bool use_strong_rule, bool do_early_exit, size_t delta_strong_size, size_t max_strong_size, size_t max_n_cds, double thr, double min_ratio, size_t n_threads, ListOf<List> checkpoints);
RcppExport SEXP _ghostbasil_basil_block_block_group_ghost__(SEXP ASEXP, SEXP rSEXP, SEXP alphaSEXP, SEXP penaltySEXP, SEXP user_lmdasSEXP, SEXP max_n_lambdasSEXP, SEXP n_lambdas_iterSEXP, SEXP use_strong_ruleSEXP, SEXP do_early_exitSEXP, SEXP delta_strong_sizeSEXP, SEXP max_strong_sizeSEXP, SEXP max_n_cdsSEXP, SEXP thrSEXP, SEXP min_ratioSEXP, SEXP n_threadsSEXP, SEXP checkpointsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< SEXP >::type A(ASEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type r(rSEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type penalty(penaltySEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type user_lmdas(user_lmdasSEXP);
    Rcpp::traits::input_parameter< size_t >::type max_n_lambdas(max_n_lambdasSEXP);
    Rcpp::traits::input_parameter< size_t >::type n_lambdas_iter(n_lambdas_iterSEXP);
    Rcpp::traits::input_parameter< bool >::type use_strong_rule(use_strong_ruleSEXP);
    Rcpp::traits::input_parameter< bool >::type do_early_exit(do_early_exitSEXP);
    Rcpp::traits::input_parameter< size_t >::type delta_strong_size(delta_strong_sizeSEXP);
    Rcpp::traits::input_parameter< size_t >::type max_strong_size(max_strong_sizeSEXP);
    Rcpp::traits::input_parameter< size_t >::type max_n_cds(max_n_cdsSEXP);
    Rcpp::traits::input_parameter< double >::type thr(thrSEXP);
    Rcpp::traits::input_parameter< double >::type min_ratio(min_ratioSEXP);
    Rcpp::traits::input_parameter< size_t >::type n_threads(n_threadsSEXP);
    Rcpp::traits::input_parameter< ListOf<List> >::type checkpoints(checkpointsSEXP);
    rcpp_result_gen = Rcpp::wrap(basil_block_block_group_ghost__(A, r, alpha, penalty, user_lmdas, max_n_lambdas, n_lambdas_iter, use_strong_rule, do_early_exit, delta_strong_size, max_strong_size, max_n_cds, thr, min_ratio, n_threads, checkpoints));
    return rcpp_result_gen;
END_RCPP
}
// objective_sparse__
List objective_sparse__(const Eigen::Map<Eigen::MatrixXd> A, const Eigen::Map<Eigen::VectorXd> r, const Eigen::Map<Eigen::VectorXd> penalty, double alpha, double lmda, const Eigen::Map<Eigen::SparseMatrix<double>> beta);
RcppExport SEXP _ghostbasil_objective_sparse__(SEXP ASEXP, SEXP rSEXP, SEXP penaltySEXP, SEXP alphaSEXP, SEXP lmdaSEXP, SEXP betaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd> >::type A(ASEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type r(rSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type penalty(penaltySEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< double >::type lmda(lmdaSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::SparseMatrix<double>> >::type beta(betaSEXP);
    rcpp_result_gen = Rcpp::wrap(objective_sparse__(A, r, penalty, alpha, lmda, beta));
    return rcpp_result_gen;
END_RCPP
}
// objective_dense__
List objective_dense__(const Eigen::Map<Eigen::MatrixXd> A, const Eigen::Map<Eigen::VectorXd> r, const Eigen::Map<Eigen::VectorXd> penalty, double alpha, double lmda, const Eigen::Map<Eigen::VectorXd> beta);
RcppExport SEXP _ghostbasil_objective_dense__(SEXP ASEXP, SEXP rSEXP, SEXP penaltySEXP, SEXP alphaSEXP, SEXP lmdaSEXP, SEXP betaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd> >::type A(ASEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type r(rSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type penalty(penaltySEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< double >::type lmda(lmdaSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type beta(betaSEXP);
    rcpp_result_gen = Rcpp::wrap(objective_dense__(A, r, penalty, alpha, lmda, beta));
    return rcpp_result_gen;
END_RCPP
}
// update_group_coeffs__
List update_group_coeffs__(const Eigen::Map<Eigen::VectorXd> L, const Eigen::Map<Eigen::VectorXd> v, double l1, double l2, double tol, size_t max_iters);
RcppExport SEXP _ghostbasil_update_group_coeffs__(SEXP LSEXP, SEXP vSEXP, SEXP l1SEXP, SEXP l2SEXP, SEXP tolSEXP, SEXP max_itersSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type L(LSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type v(vSEXP);
    Rcpp::traits::input_parameter< double >::type l1(l1SEXP);
    Rcpp::traits::input_parameter< double >::type l2(l2SEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< size_t >::type max_iters(max_itersSEXP);
    rcpp_result_gen = Rcpp::wrap(update_group_coeffs__(L, v, l1, l2, tol, max_iters));
    return rcpp_result_gen;
END_RCPP
}
// group_lasso__
List group_lasso__(const Eigen::Map<Eigen::MatrixXd> A, const Eigen::Map<Eigen::VectorXi> groups, const Eigen::Map<Eigen::VectorXi> group_sizes, double alpha, const Eigen::Map<Eigen::VectorXd> penalty, const Eigen::Map<Eigen::VectorXi> strong_set, const Eigen::Map<Eigen::VectorXi> strong_g1, const Eigen::Map<Eigen::VectorXi> strong_g2, const Eigen::Map<Eigen::VectorXi> strong_begins, const Eigen::Map<Eigen::VectorXd> strong_A_diag, const Eigen::Map<Eigen::VectorXd> lmdas, size_t max_cds, double thr, double newton_tol, size_t newton_max_iters, double rsq, Eigen::Map<Eigen::VectorXd> strong_beta, Eigen::Map<Eigen::VectorXd> strong_grad, std::vector<int> active_set, std::vector<int> active_g1, std::vector<int> active_g2, std::vector<int> active_begins, std::vector<int> active_order, Eigen::Map<Eigen::VectorXi> is_active);
RcppExport SEXP _ghostbasil_group_lasso__(SEXP ASEXP, SEXP groupsSEXP, SEXP group_sizesSEXP, SEXP alphaSEXP, SEXP penaltySEXP, SEXP strong_setSEXP, SEXP strong_g1SEXP, SEXP strong_g2SEXP, SEXP strong_beginsSEXP, SEXP strong_A_diagSEXP, SEXP lmdasSEXP, SEXP max_cdsSEXP, SEXP thrSEXP, SEXP newton_tolSEXP, SEXP newton_max_itersSEXP, SEXP rsqSEXP, SEXP strong_betaSEXP, SEXP strong_gradSEXP, SEXP active_setSEXP, SEXP active_g1SEXP, SEXP active_g2SEXP, SEXP active_beginsSEXP, SEXP active_orderSEXP, SEXP is_activeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd> >::type A(ASEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXi> >::type groups(groupsSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXi> >::type group_sizes(group_sizesSEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type penalty(penaltySEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXi> >::type strong_set(strong_setSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXi> >::type strong_g1(strong_g1SEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXi> >::type strong_g2(strong_g2SEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXi> >::type strong_begins(strong_beginsSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type strong_A_diag(strong_A_diagSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type lmdas(lmdasSEXP);
    Rcpp::traits::input_parameter< size_t >::type max_cds(max_cdsSEXP);
    Rcpp::traits::input_parameter< double >::type thr(thrSEXP);
    Rcpp::traits::input_parameter< double >::type newton_tol(newton_tolSEXP);
    Rcpp::traits::input_parameter< size_t >::type newton_max_iters(newton_max_itersSEXP);
    Rcpp::traits::input_parameter< double >::type rsq(rsqSEXP);
    Rcpp::traits::input_parameter< Eigen::Map<Eigen::VectorXd> >::type strong_beta(strong_betaSEXP);
    Rcpp::traits::input_parameter< Eigen::Map<Eigen::VectorXd> >::type strong_grad(strong_gradSEXP);
    Rcpp::traits::input_parameter< std::vector<int> >::type active_set(active_setSEXP);
    Rcpp::traits::input_parameter< std::vector<int> >::type active_g1(active_g1SEXP);
    Rcpp::traits::input_parameter< std::vector<int> >::type active_g2(active_g2SEXP);
    Rcpp::traits::input_parameter< std::vector<int> >::type active_begins(active_beginsSEXP);
    Rcpp::traits::input_parameter< std::vector<int> >::type active_order(active_orderSEXP);
    Rcpp::traits::input_parameter< Eigen::Map<Eigen::VectorXi> >::type is_active(is_activeSEXP);
    rcpp_result_gen = Rcpp::wrap(group_lasso__(A, groups, group_sizes, alpha, penalty, strong_set, strong_g1, strong_g2, strong_begins, strong_A_diag, lmdas, max_cds, thr, newton_tol, newton_max_iters, rsq, strong_beta, strong_grad, active_set, active_g1, active_g2, active_begins, active_order, is_active));
    return rcpp_result_gen;
END_RCPP
}
// lasso__
List lasso__(const Eigen::Map<Eigen::MatrixXd> A, double alpha, const Eigen::Map<Eigen::VectorXd> penalty, const Eigen::Map<Eigen::VectorXi> strong_set, const std::vector<int>& strong_order, const Eigen::Map<Eigen::VectorXd> strong_A_diag, const Eigen::Map<Eigen::VectorXd> lmdas, size_t max_cds, double thr, double rsq, Eigen::Map<Eigen::VectorXd> strong_beta, Eigen::Map<Eigen::VectorXd> strong_grad, std::vector<int> active_set, std::vector<int> active_order, std::vector<int> active_set_ordered, Eigen::Map<Eigen::VectorXi> is_active);
RcppExport SEXP _ghostbasil_lasso__(SEXP ASEXP, SEXP alphaSEXP, SEXP penaltySEXP, SEXP strong_setSEXP, SEXP strong_orderSEXP, SEXP strong_A_diagSEXP, SEXP lmdasSEXP, SEXP max_cdsSEXP, SEXP thrSEXP, SEXP rsqSEXP, SEXP strong_betaSEXP, SEXP strong_gradSEXP, SEXP active_setSEXP, SEXP active_orderSEXP, SEXP active_set_orderedSEXP, SEXP is_activeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::MatrixXd> >::type A(ASEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type penalty(penaltySEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXi> >::type strong_set(strong_setSEXP);
    Rcpp::traits::input_parameter< const std::vector<int>& >::type strong_order(strong_orderSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type strong_A_diag(strong_A_diagSEXP);
    Rcpp::traits::input_parameter< const Eigen::Map<Eigen::VectorXd> >::type lmdas(lmdasSEXP);
    Rcpp::traits::input_parameter< size_t >::type max_cds(max_cdsSEXP);
    Rcpp::traits::input_parameter< double >::type thr(thrSEXP);
    Rcpp::traits::input_parameter< double >::type rsq(rsqSEXP);
    Rcpp::traits::input_parameter< Eigen::Map<Eigen::VectorXd> >::type strong_beta(strong_betaSEXP);
    Rcpp::traits::input_parameter< Eigen::Map<Eigen::VectorXd> >::type strong_grad(strong_gradSEXP);
    Rcpp::traits::input_parameter< std::vector<int> >::type active_set(active_setSEXP);
    Rcpp::traits::input_parameter< std::vector<int> >::type active_order(active_orderSEXP);
    Rcpp::traits::input_parameter< std::vector<int> >::type active_set_ordered(active_set_orderedSEXP);
    Rcpp::traits::input_parameter< Eigen::Map<Eigen::VectorXi> >::type is_active(is_activeSEXP);
    rcpp_result_gen = Rcpp::wrap(lasso__(A, alpha, penalty, strong_set, strong_order, strong_A_diag, lmdas, max_cds, thr, rsq, strong_beta, strong_grad, active_set, active_order, active_set_ordered, is_active));
    return rcpp_result_gen;
END_RCPP
}

RcppExport SEXP _rcpp_module_boot_core_module();

static const R_CallMethodDef CallEntries[] = {
    {"_ghostbasil_basil_dense__", (DL_FUNC) &_ghostbasil_basil_dense__, 16},
    {"_ghostbasil_basil_block_dense__", (DL_FUNC) &_ghostbasil_basil_block_dense__, 16},
    {"_ghostbasil_basil_ghost__", (DL_FUNC) &_ghostbasil_basil_ghost__, 16},
    {"_ghostbasil_basil_block_ghost__", (DL_FUNC) &_ghostbasil_basil_block_ghost__, 16},
    {"_ghostbasil_basil_block_group_ghost__", (DL_FUNC) &_ghostbasil_basil_block_group_ghost__, 16},
    {"_ghostbasil_basil_block_block_group_ghost__", (DL_FUNC) &_ghostbasil_basil_block_block_group_ghost__, 16},
    {"_ghostbasil_objective_sparse__", (DL_FUNC) &_ghostbasil_objective_sparse__, 6},
    {"_ghostbasil_objective_dense__", (DL_FUNC) &_ghostbasil_objective_dense__, 6},
    {"_ghostbasil_update_group_coeffs__", (DL_FUNC) &_ghostbasil_update_group_coeffs__, 6},
    {"_ghostbasil_group_lasso__", (DL_FUNC) &_ghostbasil_group_lasso__, 24},
    {"_ghostbasil_lasso__", (DL_FUNC) &_ghostbasil_lasso__, 16},
    {"_rcpp_module_boot_core_module", (DL_FUNC) &_rcpp_module_boot_core_module, 0},
    {NULL, NULL, 0}
};

RcppExport void R_init_ghostbasil(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
