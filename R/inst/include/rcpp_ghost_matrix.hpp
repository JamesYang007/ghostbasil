#pragma once
#include <Rcpp.h>
#include <RcppEigen.h>
#include <ghostbasil/matrix/ghost_matrix.hpp>

class GhostMatrixWrap
{
    using mat_t = Eigen::MatrixXd;
    using vec_t = Eigen::VectorXd;
    using const_mat_map_t = Eigen::Map<const mat_t>;
    using const_vec_map_t = Eigen::Map<const vec_t>;
    using mat_map_t = Eigen::Map<mat_t>;
    using vec_map_t = Eigen::Map<vec_t>;
    using gmat_t = ghostbasil::GhostMatrix<mat_t, vec_t>;
    using dim_t = Eigen::Array<size_t, 2, 1>;

    gmat_t gmat_;
    const dim_t dim_;

public: 
    GhostMatrixWrap(SEXP mat,
                    SEXP vec,
                    size_t n_groups)
        : gmat_(Rcpp::as<mat_map_t>(mat), 
                Rcpp::as<vec_map_t>(vec), 
                n_groups),
          dim_(gmat_.rows(), gmat_.cols())
    {}

    GHOSTBASIL_STRONG_INLINE
    const auto& internal() const { return gmat_; }

    // For export only
    dim_t dim_exp() const { return dim_; }
    const_mat_map_t matrix_exp() const { return gmat_.matrix(); }
    const_vec_map_t vector_exp() const { return gmat_.vector(); }
    size_t n_groups_exp() const { return gmat_.n_groups(); }
    double col_dot_exp(size_t k, const vec_map_t v) const {
        return gmat_.col_dot(k, v);
    }
    double quad_form_exp(const vec_map_t v) const {
        return gmat_.quad_form(v);
    }
    double inv_quad_form_exp(double s, const vec_map_t v) const {
        return gmat_.inv_quad_form(s, v);
    }
};

// MUST be in header since it needs to exist in every cpp file that includes class definition
RCPP_EXPOSED_AS(GhostMatrixWrap)