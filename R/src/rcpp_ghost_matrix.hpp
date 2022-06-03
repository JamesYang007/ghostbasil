#pragma once
#include <Rcpp.h>
#include <RcppEigen.h>
#include <ghostbasil/matrix/ghost_matrix.hpp>

class GhostMatrixWrap: 
    public ghostbasil::GhostMatrix<
        Eigen::MatrixXd, Eigen::VectorXd>
{
    using mat_t = Eigen::MatrixXd;
    using vec_t = Eigen::VectorXd;
    using const_mat_map_t = Eigen::Map<const mat_t>;
    using const_vec_map_t = Eigen::Map<const vec_t>;
    using mat_map_t = Eigen::Map<mat_t>;
    using vec_map_t = Eigen::Map<vec_t>;
    using ghost_mat_t = ghostbasil::GhostMatrix<mat_t, vec_t>;
    using dim_t = Eigen::Array<size_t, 2, 1>;

    const dim_t dim_;

public: 
    using typename ghost_mat_t::Scalar;

    GhostMatrixWrap(SEXP mat,
                    SEXP vec,
                    size_t n_groups)
        : ghost_mat_t(
                Rcpp::as<mat_map_t>(mat), 
                Rcpp::as<vec_map_t>(vec), 
                n_groups),
          dim_(ghost_mat_t::rows(), ghost_mat_t::cols())
    {}

    // Expose base versions for building the rest of the bindings.
    using ghost_mat_t::col_dot;
    using ghost_mat_t::quad_form;
    using ghost_mat_t::inv_quad_form;

    // For export only
    dim_t dim_exp() const { return dim_; }
    const_mat_map_t matrix_exp() const { return ghost_mat_t::matrix(); }
    const_vec_map_t vector_exp() const { return ghost_mat_t::vector(); }
    size_t n_groups_exp() const { return ghost_mat_t::n_groups(); }
    double col_dot_exp(size_t k, const vec_map_t v) const {
        return ghost_mat_t::col_dot(k, v);
    }
    double quad_form_exp(const vec_map_t v) const {
        return ghost_mat_t::quad_form(v);
    }
    double inv_quad_form_exp(double s, const vec_map_t v) const {
        return ghost_mat_t::inv_quad_form(s, v);
    }
};

RCPP_EXPOSED_AS(GhostMatrixWrap)
