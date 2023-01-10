#include <Rcpp.h>
#include <RcppEigen.h>
#include "rcpp_block_matrix.hpp"
#include "rcpp_ghost_matrix.hpp"

RCPP_MODULE(core_module) {
    Rcpp::class_<BlockMatrixWrap>("BlockMatrix__")
        .constructor<Rcpp::List>()
        .method("col_dot", &BlockMatrixWrap::col_dot_exp)
        .method("quad_form", &BlockMatrixWrap::quad_form_exp)
        .method("inv_quad_form", &BlockMatrixWrap::inv_quad_form_exp)
        .property("matrices", &BlockMatrixWrap::get_mat_list_exp)
        .property("dim", &BlockMatrixWrap::dim_exp)
        ;

    Rcpp::class_<GhostMatrixWrap>("GhostMatrix__")
        .constructor<SEXP, SEXP, size_t>()
        .method("col_dot", &GhostMatrixWrap::col_dot_exp)
        .method("quad_form", &GhostMatrixWrap::quad_form_exp)
        .method("inv_quad_form", &GhostMatrixWrap::inv_quad_form_exp)
        .property("get_S", &GhostMatrixWrap::matrix_exp)
        .property("get_D", &GhostMatrixWrap::vector_exp)
        .property("n_groups", &GhostMatrixWrap::n_groups_exp)
        .property("dim", &GhostMatrixWrap::dim_exp)
        ;

    Rcpp::class_<GroupGhostMatrixWrap>("GroupGhostMatrix__")
        .constructor<SEXP, SEXP, size_t>()
        .method("col_dot", &GroupGhostMatrixWrap::col_dot_exp)
        .method("quad_form", &GroupGhostMatrixWrap::quad_form_exp)
        .property("get_S", &GroupGhostMatrixWrap::get_S_exp)
        .property("get_D", &GroupGhostMatrixWrap::get_D_exp)
        .property("n_groups", &GroupGhostMatrixWrap::n_groups_exp)
        .property("dim", &GroupGhostMatrixWrap::dim_exp)
        ;

    Rcpp::class_<BlockGhostMatrixWrap>("BlockGhostMatrix__")
        .constructor<Rcpp::List>()
        .method("col_dot", &BlockGhostMatrixWrap::col_dot_exp)
        .method("quad_form", &BlockGhostMatrixWrap::quad_form_exp)
        .method("inv_quad_form", &BlockGhostMatrixWrap::inv_quad_form_exp)
        .property("matrices", &BlockGhostMatrixWrap::get_mat_list_exp)
        .property("dim", &BlockGhostMatrixWrap::dim_exp)
        ;

    Rcpp::class_<BlockGroupGhostMatrixWrap>("BlockGroupGhostMatrix__")
        .constructor<Rcpp::List>()
        .method("col_dot", &BlockGroupGhostMatrixWrap::col_dot_exp)
        .method("quad_form", &BlockGroupGhostMatrixWrap::quad_form_exp)
        .property("matrices", &BlockGroupGhostMatrixWrap::get_mat_list_exp)
        .property("dim", &BlockGroupGhostMatrixWrap::dim_exp)
        ;
}
