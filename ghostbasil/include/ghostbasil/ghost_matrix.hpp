#pragma once
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <ghostbasil/macros.hpp>

namespace ghostbasil {

template <class MatrixType, class VectorType>
class GhostMatrix
{
    using mat_t = std::decay_t<MatrixType>;
    using vec_t = std::decay_t<VectorType>;
    using value_t = typename mat_t::Scalar;
    using colvec_t = Eigen::Matrix<value_t, Eigen::Dynamic, 1>;
    using sp_mat_t = Eigen::SparseMatrix<value_t>;

    static_assert(
        std::is_same<value_t, typename vec_t::Scalar>::value,
        "Matrix and vector underlying value type must be the same."
    );

    const mat_t* mat_list_;
    const vec_t* vec_list_;
    size_t n_;
    size_t n_multiplier_;
    std::vector<uint32_t> n_cum_sum_; // [n_cum_sum_[i], n_cum_sum_[i+1])
                                      // is the range of features for block i.

    GHOSTBASIL_STRONG_INLINE
    auto n_features() const { return n_cum_sum_.back(); }
    GHOSTBASIL_STRONG_INLINE
    auto n_orig_features(size_t i) const { return mat_list_[i].cols(); }

    template <class XType, class BType, class DType,
              class BufferType, class TType, class SType>
    GHOSTBASIL_STRONG_INLINE
    void compute_TS(
            const XType& x,
            const BType& B,
            const DType& D,
            BufferType& buffer,
            TType& T,
            SType& S) const
    {
        assert(T.size() == S.rows());
        assert(S.cols() == n_multiplier_);
        size_t group_size = T.size();
        size_t x_k_begin = 0;
        for (size_t k = 0; k < n_multiplier_; ++k, x_k_begin += group_size) {
            const auto x_k = x.segment(x_k_begin, group_size);
            const auto R_k = B * x_k;
            const auto S_k = x_k.cwiseProduct(D);
            buffer = R_k; // load into common buffer to avoid memory alloc underneath.
                          // helps in sparse x_k case also so that the next step is vectorized.
            S.col(k) = S_k; // save S_k for later
            T += buffer;
            T -= S.col(k);
        }
    }

    template <class XType, class TType, class SType>
    GHOSTBASIL_STRONG_INLINE
    value_t compute_quadform(
            const XType& x, 
            const TType& T,
            const SType& S) const
    {
        assert(T.size() == S.rows());
        size_t group_size = T.size();
        size_t x_k_begin = 0;
        value_t quadform = 0;
        for (size_t k = 0; k < n_multiplier_; ++k, x_k_begin += group_size) {
            const auto x_k = x.segment(x_k_begin, group_size);
            quadform += x_k.dot(T + S.col(k));
        }
        return quadform;
    }

public:
    template <class MatrixListType,
              class VectorListType>
    GhostMatrix(const MatrixListType& matrix_list,
                const VectorListType& vector_list,
                size_t n_knockoffs)
        : mat_list_(matrix_list.data()),
          vec_list_(vector_list.data()),
          n_(matrix_list.size()),
          n_multiplier_(n_knockoffs + 1)
    {
        // Check that number of knockoffs is at least 1.
        if (n_knockoffs < 1) {
            throw std::runtime_error(
                "Number of knockoffs must be at least 1. "
                "If number of knockoffs is 0, use BlockMatrix instead.");
        }

        // Check that matrix list and vector list is the same length.
        if ((matrix_list.size() == 0) ||
            (matrix_list.size() != vector_list.size())) {
            throw std::runtime_error(
                "List of matrix and list of vectors must have the same length and nonzero. ");
        }

        // Check that each matrix, vector pair have the same dimensions
        // and the matrix is square.
        for (size_t i = 0; i < n_; ++i) {
            const auto& B = mat_list_[i];
            const auto& D = vec_list_[i];
            if (B.rows() != B.cols()) {
                std::string error = "Matrix at index " + std::to_string(i) + " is not square.";
                throw std::runtime_error(error);
            }
            if (B.rows() != D.size()) {
                std::string error = 
                    "Matrix and vector pair at index " + std::to_string(i) + 
                    " do not have same dimensions. " +
                    "Matrix has dimensions " + std::to_string(B.rows()) + " x " + std::to_string(B.cols()) + " and " +
                    "vector has length " + std::to_string(D.size()) + ". ";
                throw std::runtime_error(error);
            }
            if (B.rows() <= 0) {
                std::string error =
                    "Matrix and vector pair at index " + std::to_string(i) +
                    " must have dimension/length > 0.";
                throw std::runtime_error(error);
            }
        }

        // Compute the cumulative number of features.
        n_cum_sum_.resize(n_+1);
        n_cum_sum_[0] = 0;
        for (size_t i = 0; i < n_; ++i) {
            n_cum_sum_[i+1] = n_cum_sum_[i] + mat_list_[i].cols() * n_multiplier_;
        }
    }

    /*
     * Computes the dot product between kth column of the matrix with v: 
     *      A[:,k]^T * v
     */
    template <class VecType>
    value_t col_dot(size_t k, const VecType& v) const
    {
        assert(k < n_features());

        // Find the i(k) which is the closest index to k:
        // n_cum_sum_[i(k)] <= k < n_cum_sum_[i(k)+1]
        const auto ik_end = std::upper_bound(
                n_cum_sum_.begin(),
                n_cum_sum_.end(),
                k);
        const auto ik_begin = std::next(ik_end, -1);
        const auto ik = std::distance(n_cum_sum_.begin(), ik_begin);  
        assert((ik+1) < n_cum_sum_.size());

        // Find i(k)th block matrix, diagonal matrix, and size.
        const auto& B = mat_list_[ik];
        const auto& D = vec_list_[ik];
        const size_t group_size = n_orig_features(ik);

        // Find v_{i(k)}, i(k)th block of vector. 
        const auto vi = v.segment(n_cum_sum_[ik], n_cum_sum_[ik+1]-n_cum_sum_[ik]);

        // Find the shifted k relative to A_{i(k)}).
        const size_t k_shifted = k - n_cum_sum_[ik];

        // Find the index to block of K features relative to A_{i(k)} containing k_shifted.
        const size_t k_shifted_block_begin = (k_shifted / group_size) * group_size;

        // Find the relative k to 
        const size_t k_shifted_block = k_shifted - k_shifted_block_begin;

        // Get quantities for reuse.
        value_t D_kk = D[k_shifted_block];

        // Compute the dot product.
        value_t dp = 0;
        size_t vi_j_begin = 0;
        for (size_t j = 0; j < n_multiplier_; ++j, vi_j_begin += group_size) {
            const auto vi_j = vi.segment(vi_j_begin, group_size);
            const auto B_k = B.col(k_shifted_block);
            dp += vi_j.dot(B_k) - D_kk * vi_j.coeff(k_shifted_block);
        }
        dp += D_kk * vi.coeff(k_shifted);

        return dp;
    }

    /*
     * Computes the quadratic form of the matrix with v:
     *      v^T A v
     */
    template <class VecType>
    value_t quad_form(const VecType& v) const
    {   
        // Notation:
        // K = n_multiplier_
        // B = a block matrix
        // D = corresponding diagonal matrix
        // x = subvector corresponding to B and D
        // x_k = kth block vector corresponding to a group
        // R_k = B x_k (columns of R)
        // S_k = D x_k (columns of S)
        // T = \sum\limits_{k=1}^K R_k - \sum\limits_{k=1}^K S_k

        // Choose type of S based on whether v is dense or sparse.
        using S_t = std::conditional_t<
            std::is_base_of<Eigen::DenseBase<VecType>, VecType>::value,
            mat_t, sp_mat_t>;

        colvec_t buffer;
        colvec_t T; 
        S_t S;
        value_t quadform = 0;
        
        for (size_t i = 0; i < n_; ++i) {
            const auto& B = mat_list_[i];
            const auto& D = vec_list_[i];
            const auto x = v.segment(n_cum_sum_[i], n_cum_sum_[i+1]-n_cum_sum_[i]);
            const size_t group_size = n_orig_features(i);

            T.setZero(group_size);
            S.resize(group_size, n_multiplier_);

            // Compute T and S
            compute_TS(x, B, D, buffer, T, S);

            // Compute quadratic form of current block
            quadform += compute_quadform(x, T, S);
        }

        return quadform;
    }

    /*
     * Computes an _estimate_ of inverse quadratic form:
     *      v^T [(1-s)A + sI]^{-1} v
     * where 0 <= s <= 1.
     * Note that it is undefined behavior if A
     * is not positive semi-definite.
     * If v == 0, then the result is 0.
     * If s == 0, A is not positive definite but semi-definite,
     * and v != 0 is in the kernel of A, then the result is Inf.
     * If s == 0, A is not positive definite, and not the previous cases,
     * it is undefined behavior.
     * In all other cases, the function will attempt to compute the desired quantity,
     * and is well-defined.
     */
    template <class VecType>
    value_t inv_quad_form(value_t s, const VecType& v) const
    {
        // Compute ||v||^6
        const auto v_norm_sq = v.squaredNorm();

        assert(0 <= s && s <= 1);

        if (v_norm_sq <= 0) return 0;

        // Choose type of S based on whether v is dense or sparse.
        using S_t = std::conditional_t<
            std::is_base_of<Eigen::DenseBase<VecType>, VecType>::value,
            mat_t, sp_mat_t>;

        colvec_t buffer; 
        colvec_t T; 
        S_t S;
        value_t Av_norm_sq = 0;
        value_t vTAv = 0;

        for (size_t i = 0; i < n_; ++i) {
            const auto& B = mat_list_[i];
            const auto& D = vec_list_[i];
            const auto x = v.segment(n_cum_sum_[i], n_cum_sum_[i+1]-n_cum_sum_[i]);
            const size_t group_size = n_orig_features(i);

            T.setZero(group_size);
            S.resize(group_size, n_multiplier_);

            // Compute T and S
            compute_TS(x, B, D, buffer, T, S);

            // Compute Av_norm_sq
            for (size_t l = 0; l < n_multiplier_; ++l) {
                Av_norm_sq += (T + S.col(l)).squaredNorm();
            }

            // Compute quadratic form of current block
            vTAv += compute_quadform(x, T, S);
        }

        const auto sc = 1-s;
        const auto s_sq = s * s;
        const auto sc_sq = sc * sc;
        const auto denom = (sc * vTAv + s * v_norm_sq);
        if (denom <= 0) return std::numeric_limits<value_t>::infinity();
        const auto factor = v_norm_sq / denom;
        const auto factor_pow3 = factor * factor * factor;
        value_t inv_quad_form = 
            factor_pow3 * (sc_sq * Av_norm_sq + 2*s*denom - s_sq*v_norm_sq);
        return inv_quad_form;
    }
};

} // namespace ghostbasil
