#pragma once
#include <Rcpp.h>
#include <RcppEigen.h>
#include <ghostbasil/matrix/block_matrix.hpp>
#include "rcpp_ghost_matrix.hpp"
#include "rcpp_group_ghost_matrix.hpp"

class BlockMatrixWrap
{
    using vec_t = Eigen::VectorXd;
    using block_as_t = Eigen::Map<Eigen::MatrixXd>;
    using vec_map_t = Eigen::Map<vec_t>;
    using mat_list_t = std::vector<block_as_t>;
    using block_mat_t = ghostbasil::BlockMatrix<block_as_t>;
    using dim_t = Eigen::Array<size_t, 2, 1>;

    mat_list_t mat_list_;
    Rcpp::List orig_list_;
    block_mat_t block_mat_;
    const dim_t dim_;

    static auto init_mat_list(Rcpp::List mat_list) 
    {
        std::vector<block_as_t> mat_list_;
        mat_list_.reserve(mat_list.size()); // important to not invoke resize?
        for (size_t i = 0; i < mat_list.size(); ++i) {
            mat_list_.emplace_back(Rcpp::as<block_as_t>(mat_list[i]));
        }
        return mat_list_;
    }

public: 
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
    using mat_t = Eigen::MatrixXd;
    using vec_t = Eigen::VectorXd;
    using block_as_t = ghostbasil::GhostMatrix<mat_t, vec_t>;
    using vec_map_t = Eigen::Map<vec_t>;
    using mat_list_t = std::vector<block_as_t>;
    using block_mat_t = ghostbasil::BlockMatrix<block_as_t>;
    using dim_t = Eigen::Array<size_t, 2, 1>;

    mat_list_t mat_list_;
    Rcpp::List orig_list_;
    block_mat_t block_mat_;
    const dim_t dim_;

    static auto init_mat_list(Rcpp::List mat_list) 
    {
        using as_t = GhostMatrixWrap;
        std::vector<block_as_t> mat_list_;
        mat_list_.reserve(mat_list.size()); // important to not invoke resize?
        for (size_t i = 0; i < mat_list.size(); ++i) {
            mat_list_.emplace_back(
                    Rcpp::as<as_t>(mat_list[i]).internal());
        }
        return mat_list_;
    }

public: 
    BlockGhostMatrixWrap(Rcpp::List mat_list)
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

class BlockGroupGhostMatrixWrap
{
    using mat_t = Eigen::MatrixXd;
    using vec_t = Eigen::VectorXd;
    using block_as_t = ghostbasil::GroupGhostMatrix<mat_t>;
    using vec_map_t = Eigen::Map<vec_t>;
    using mat_list_t = std::vector<block_as_t>;
    using block_mat_t = ghostbasil::BlockMatrix<block_as_t>;
    using dim_t = Eigen::Array<size_t, 2, 1>;

    mat_list_t mat_list_;
    Rcpp::List orig_list_;
    block_mat_t block_mat_;
    const dim_t dim_;

    static auto init_mat_list(Rcpp::List mat_list) 
    {
        using as_t = GroupGhostMatrixWrap;
        std::vector<block_as_t> mat_list_;
        mat_list_.reserve(mat_list.size()); // important to not invoke resize?
        for (size_t i = 0; i < mat_list.size(); ++i) {
            mat_list_.emplace_back(
                    Rcpp::as<as_t>(mat_list[i]).internal());
        }
        return mat_list_;
    }

public: 
    BlockGroupGhostMatrixWrap(Rcpp::List mat_list)
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

    Rcpp::List get_mat_list_exp() const { return orig_list_; }
};

RCPP_EXPOSED_AS(BlockMatrixWrap)
RCPP_EXPOSED_AS(BlockGhostMatrixWrap)
RCPP_EXPOSED_AS(BlockGroupGhostMatrixWrap)
