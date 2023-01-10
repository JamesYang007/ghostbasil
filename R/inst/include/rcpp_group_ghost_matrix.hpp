#pragma once
#include <Rcpp.h>
#include <RcppEigen.h>
#include <ghostbasil/matrix/group_ghost_matrix.hpp>

class GroupGhostMatrixWrap
{
    using mat_t = Eigen::MatrixXd;
    using vec_t = Eigen::VectorXd;
    using const_mat_map_t = Eigen::Map<const mat_t>;
    using mat_map_t = Eigen::Map<mat_t>;
    using vec_map_t = Eigen::Map<vec_t>;
    using gmat_t = ghostbasil::GroupGhostMatrix<mat_t>;
    using dim_t = Eigen::Array<size_t, 2, 1>;

    gmat_t gmat_;
    const dim_t dim_;

public: 
    GroupGhostMatrixWrap(
        SEXP S,
        SEXP D,
        size_t n_groups
    )
        : gmat_(Rcpp::as<mat_map_t>(S), 
                Rcpp::as<mat_map_t>(D), 
                n_groups),
          dim_(gmat_.rows(), gmat_.cols())
    {}

    GHOSTBASIL_STRONG_INLINE
    const auto& internal() const { return gmat_; }

    // For export only
    dim_t dim_exp() const { return dim_; }
    const_mat_map_t get_S_exp() const { return gmat_.get_S(); }
    const_mat_map_t get_D_exp() const { return gmat_.get_D(); }
    size_t n_groups_exp() const { return gmat_.n_groups(); }
    double col_dot_exp(size_t k, const vec_map_t v) const {
        return gmat_.col_dot(k, v);
    }
    double quad_form_exp(const vec_map_t v) const {
        return gmat_.quad_form(v);
    }
};

RCPP_EXPOSED_AS(GroupGhostMatrixWrap)
