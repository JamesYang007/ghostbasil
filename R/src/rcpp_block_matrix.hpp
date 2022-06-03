#pragma once
#include <Rcpp.h>
#include <RcppEigen.h>
#include <ghostbasil/matrix/block_matrix.hpp>
#include "rcpp_ghost_matrix.hpp"

class BlockMatrixWrap
{
    using mat_t = Eigen::MatrixXd;
    using vec_t = Eigen::VectorXd;
    using mat_map_t = Eigen::Map<mat_t>;
    using vec_map_t = Eigen::Map<vec_t>;
    using mat_list_t = std::vector<const mat_map_t>;
    using block_mat_t = ghostbasil::BlockMatrix<mat_map_t>;
    using dim_t = Eigen::Array<size_t, 2, 1>;

    mat_list_t mat_list_;
    Rcpp::List orig_list_;
    block_mat_t block_mat_;
    const dim_t dim_;

    static mat_list_t init_mat_list(Rcpp::List mat_list) 
    {
        mat_list_t mat_list_;
        mat_list_.reserve(mat_list.size()); // important to not invoke resize?
        for (size_t i = 0; i < mat_list.size(); ++i) {
            mat_list_.emplace_back(Rcpp::as<mat_map_t>(mat_list[i]));
        }
        return mat_list_;
    }

public: 
    using Scalar = typename block_mat_t::Scalar;
    using Index = typename block_mat_t::Index;

    BlockMatrixWrap(Rcpp::List mat_list)
        : mat_list_(init_mat_list(mat_list)),
          orig_list_(mat_list),
          block_mat_(mat_list_),
          dim_(block_mat_.rows(), block_mat_.cols())
    {}

    GHOSTBASIL_STRONG_INLINE
    const auto& internal() const { return block_mat_; }

    // For export only
    dim_t dim_exp() const { return dim_; }

    double col_dot_exp(size_t k, const vec_map_t v) const
    {
        return block_mat_.col_dot(k, v);
    }

    double quad_form_exp(const vec_map_t v) const
    {
        return block_mat_.quad_form(v);
    }

    double inv_quad_form_exp(double s, const vec_map_t v) const
    {
        return block_mat_.inv_quad_form(s, v);
    }

    Rcpp::List get_mat_list_exp() const { return orig_list_; }
};

class BlockGhostMatrixWrap
{
    using vec_t = Eigen::VectorXd;
    using mat_map_t = GhostMatrixWrap;
    using vec_map_t = Eigen::Map<vec_t>;
    using mat_list_t = std::vector<const mat_map_t>;
    using bmat_t = ghostbasil::BlockMatrix<mat_map_t>;
    using dim_t = Eigen::Array<size_t, 2, 1>;

    mat_list_t mat_list_;
    Rcpp::List orig_list_;
    bmat_t bmat_;
    const dim_t dim_;

    static mat_list_t init_mat_list(Rcpp::List mat_list) 
    {
        mat_list_t mat_list_;
        for (size_t i = 0; i < mat_list.size(); ++i) {
            mat_list_.emplace_back(Rcpp::as<mat_map_t>(mat_list[i]));
        }
        return mat_list_;
    }

public: 
    using Scalar = typename bmat_t::Scalar;
    using Index = typename bmat_t::Index;

    BlockGhostMatrixWrap(Rcpp::List mat_list)
        : mat_list_(init_mat_list(mat_list)),
          orig_list_(mat_list),
          bmat_(mat_list_),
          dim_(bmat_.rows(), bmat_.cols())
    {}

    GHOSTBASIL_STRONG_INLINE Index cols() const { 
        return bmat_.cols();
    }
    
    template <class VecType>
    GHOSTBASIL_STRONG_INLINE
    auto col_dot(size_t k, const VecType& v) const
    {
        return bmat_.col_dot(k, v);
    }

    GHOSTBASIL_STRONG_INLINE 
    Scalar coeff(Index i, Index j) const 
    {
        return bmat_.coeff(i, j);
    }

    // For export only
    dim_t dim_exp() const { return dim_; }

    double col_dot_exp(size_t k, const vec_map_t v) const
    {
        return bmat_.col_dot(k, v);
    }

    double quad_form_exp(const vec_map_t v) const
    {
        return bmat_.quad_form(v);
    }

    double inv_quad_form_exp(double s, const vec_map_t v) const
    {
        return bmat_.inv_quad_form(s, v);
    }

    Rcpp::List get_mat_list_exp() const { return orig_list_; }
};

RCPP_EXPOSED_AS(BlockMatrixWrap)
RCPP_EXPOSED_AS(BlockGhostMatrixWrap)
