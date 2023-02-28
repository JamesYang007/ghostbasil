#pragma once
#include <Rcpp.h>
#include <RcppEigen.h>
#include <ghostbasil/matrix/block_group_ghost_matrix.hpp>
#include "rcpp_block_matrix.hpp"

class BlockGroupGhostMatrixWrap
{
public:
    using mat_t = Eigen::MatrixXd;
    using vec_t = Eigen::VectorXd;
    using const_mat_map_t = Eigen::Map<const mat_t>;
    using mat_map_t = Eigen::Map<mat_t>;
    using vec_map_t = Eigen::Map<vec_t>;
    using bmat_t = ghostbasil::BlockMatrix<mat_map_t>;
    using gmat_t = ghostbasil::BlockGroupGhostMatrix<mat_t, bmat_t>;
    using dim_t = Eigen::Array<size_t, 2, 1>;

private:
    const mat_map_t orig_S_;
    const BlockMatrixWrap orig_D_;
    gmat_t gmat_;
    const dim_t dim_;

public: 
    BlockGroupGhostMatrixWrap(
        SEXP S,
        SEXP D,
        size_t n_groups
    )
        : orig_S_(Rcpp::as<mat_map_t>(S)),
          orig_D_(Rcpp::as<BlockMatrixWrap>(D)),
          gmat_(orig_S_, 
                orig_D_.internal(), 
                n_groups),
          dim_(gmat_.rows(), gmat_.cols())
    {}

    GHOSTBASIL_STRONG_INLINE
    const auto& internal() const { return gmat_; }

    // For export only
    dim_t dim_exp() const { return dim_; }
    mat_map_t get_S_exp() const { return orig_S_; }
    Rcpp::List get_D_exp() const { return orig_D_.get_mat_list_exp(); }
    size_t n_groups_exp() const { return gmat_.n_groups(); }
    double col_dot_exp(size_t k, const vec_map_t v) const {
        return gmat_.col_dot(k, v);
    }
    double quad_form_exp(const vec_map_t v) const {
        return gmat_.quad_form(v);
    }
    mat_t to_dense_exp() const { return gmat_.to_dense(); }
};

// MUST be in header since it needs to exist in every cpp file that includes class definition
RCPP_EXPOSED_AS(BlockGroupGhostMatrixWrap)