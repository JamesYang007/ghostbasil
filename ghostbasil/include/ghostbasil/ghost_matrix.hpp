#pragma once
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <ghostbasil/macros.hpp>

namespace ghostbasil {

/*
 * This class represents a matrix of the form:
 *
 *      S   S-D ... S-D
 *      S-D S   .   .
 *      .   .   S   .
 *      S-D .   ... S
 *
 * which is the typical form for the Gaussian covariance matrix
 * under multiple knockoffs framework.
 * We say that a GhostMatrix has n groups if it is of size
 * (n*p) x (n*p) where S and D are p x p.
 *
 * @tparam  MatrixType  type of matrix that represents S.
 * @tparam  VectorType  type of vector that represents the diagonal of D.
 */
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

    const mat_t* mat_ptr_;  // ptr to matrix S
    const vec_t* vec_ptr_;  // ptr to diagonal vector D
    size_t n_groups_;       // number of groups
    Eigen::Index rows_;

    GHOSTBASIL_STRONG_INLINE
    auto compute_group_size() const { return rows_ / n_groups_; }

    template <class XType, class SType, class DType,
              class BufferType, class TType, class QType>
    GHOSTBASIL_STRONG_INLINE
    void compute_TQ(
            const XType& x,
            const SType& S,
            const DType& D,
            BufferType& buffer,
            TType& T,
            QType& Q) const
    {
        assert(T.size() == Q.rows());
        assert(Q.cols() == n_groups_);
        size_t group_size = T.size();
        size_t x_k_begin = 0;
        for (size_t k = 0; k < n_groups_; ++k, x_k_begin += group_size) {
            const auto x_k = x.segment(x_k_begin, group_size);
            const auto R_k = S * x_k;
            const auto Q_k = x_k.cwiseProduct(D);
            buffer = R_k; // load into common buffer to avoid memory alloc underneath.
                          // helps in sparse x_k case also so that the next step is vectorized.
            Q.col(k) = Q_k; // save S_k for later
            T += buffer;
            T -= Q.col(k);
        }
    }

    template <class XType, class TType, class QType>
    GHOSTBASIL_STRONG_INLINE
    value_t compute_quadform(
            const XType& x, 
            const TType& T,
            const QType& Q) const
    {
        assert(T.size() == Q.rows());
        size_t group_size = T.size();
        size_t x_k_begin = 0;
        value_t quadform = 0;
        for (size_t k = 0; k < n_groups_; ++k, x_k_begin += group_size) {
            const auto x_k = x.segment(x_k_begin, group_size);
            quadform += x_k.dot(T + Q.col(k));
        }
        return quadform;
    }

public:
    using Scalar = value_t;
    using Index = Eigen::Index;
    
    GhostMatrix(const mat_t& mat,
                const vec_t& vec,
                size_t n_groups)
        : mat_ptr_(&mat),
          vec_ptr_(&vec),
          n_groups_(n_groups),
          rows_(mat.cols() * n_groups)
    {
        // Check that number of groups is at least 2.
        if (n_groups_ < 2) {
            throw std::runtime_error(
                "Number of groups must be at least 2. "
                "If number of groups <= 1, use Eigen::Matrix instead, "
                "since GhostMatrix degenerates to the top-left corner matrix. ");
        }

        // Check that the matrix is square and non-empty.
        const auto& B = matrix();
        if (B.rows() != B.cols()) {
            throw std::runtime_error("Matrix is not square.");
        }
        if (B.rows() <= 0) {
            throw std::runtime_error("Matrix and vector must have dimension/length > 0.");
        }

        // Check that the matrix and vector agree in size.
        const auto& D = vector();
        if (B.rows() != D.size()) {
            std::string error = 
                "Matrix and vector do not have same dimensions. "                
                "Matrix has dimensions " + std::to_string(B.rows()) + " x " + std::to_string(B.cols()) + " and " +
                "vector has length " + std::to_string(D.size()) + ". ";
            throw std::runtime_error(error);
        }
    }

    GHOSTBASIL_STRONG_INLINE Index rows() const { return rows_; }
    GHOSTBASIL_STRONG_INLINE Index cols() const { return rows(); }
    GHOSTBASIL_STRONG_INLINE
    const auto& matrix() const { return *mat_ptr_; }
    GHOSTBASIL_STRONG_INLINE
    const auto& vector() const { return *vec_ptr_; }
    GHOSTBASIL_STRONG_INLINE
    auto n_groups() const { return n_groups_; }

    /*
     * Computes the dot product between kth column of the matrix with v: 
     *      A[:,k]^T * v
     */
    template <class VecType>
    value_t col_dot(size_t k, const VecType& v) const
    {
        assert(k < cols());

        const size_t group_size = compute_group_size();
        const auto& S = matrix();
        const auto& D = vector();

        // Find the index to block of K features containing k.
        const size_t k_block_begin = (k / group_size) * group_size;

        // Find the index relative to k_block_begin.
        const size_t k_block = k - k_block_begin;

        // Get quantities for reuse.
        value_t D_kk = D[k_block];

        // Compute the dot product.
        value_t dp = 0;
        size_t v_j_begin = 0;
        for (size_t j = 0; j < n_groups_; ++j, v_j_begin += group_size) {
            const auto v_j = v.segment(v_j_begin, group_size);
            const auto S_k = S.col(k_block);
            dp += v_j.dot(S_k) - D_kk * v_j.coeff(k_block);
        }
        dp += D_kk * v.coeff(k);

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
        // K = n_groups_
        // S = top-left corner matrix
        // D = S's corresponding diagonal matrix
        // v_k = kth subset of v (of length group_size)
        // R_k = S v_k (columns of R)
        // Q_k = D v_k (columns of Q)
        // T = \sum\limits_{k=1}^K R_k - \sum\limits_{k=1}^K Q_k

        // Choose type of Q based on whether v is dense or sparse.
        using Q_t = std::conditional_t<
            std::is_base_of<Eigen::DenseBase<VecType>, VecType>::value,
            mat_t, sp_mat_t>;

        const auto& S = matrix();
        const auto& D = vector();
        const size_t group_size = compute_group_size();

        colvec_t buffer;
        colvec_t T(group_size); 
        T.setZero();
        Q_t Q(group_size, n_groups_);

        // Compute T and S
        compute_TQ(v, S, D, buffer, T, Q);

        // Compute quadratic form of current block
        value_t quadform = compute_quadform(v, T, Q);

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
        // Compute ||v||^6.
        const auto v_norm_sq = v.squaredNorm();

        assert(0 <= s && s <= 1);

        if (v_norm_sq <= 0) return 0;

        // Choose type of Q based on whether v is dense or sparse.
        using Q_t = std::conditional_t<
            std::is_base_of<Eigen::DenseBase<VecType>, VecType>::value,
            mat_t, sp_mat_t>;

        const auto& S = matrix();
        const auto& D = vector();
        const size_t group_size = compute_group_size();

        colvec_t buffer; 
        colvec_t T(group_size); 
        T.setZero();
        Q_t Q(group_size, n_groups_);
        value_t Av_norm_sq = 0;
        value_t vTAv = 0;

        // Compute T and Q.
        compute_TQ(v, S, D, buffer, T, Q);

        // Compute Av_norm_sq.
        for (size_t l = 0; l < n_groups_; ++l) {
            Av_norm_sq += (T + Q.col(l)).squaredNorm();
        }

        // Compute quadratic form of current block.
        vTAv += compute_quadform(v, T, Q);

        // Compute the inverse quadratic form estimate.
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
