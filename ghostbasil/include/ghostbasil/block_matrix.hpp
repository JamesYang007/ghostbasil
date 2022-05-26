#pragma once
#include <Eigen/Core>
#include <vector>
#include <string>
#include <ghostbasil/macros.hpp>

namespace ghostbasil {

/*
 * This class represents a block-diagonal matrix
 * where each block has the same type MatrixType.
 * A block-diagonal matrix is of the form:
 *
 *      A_1 0   ... 0
 *      0   A_2 ... 0
 *      .   .   ... 0
 *      0   .   ... A_L
 *
 * where the above has L blocks of square matrices A_l.
 */
template <class MatrixType>
class BlockMatrix
{
    using mat_t = MatrixType;
    using value_t = typename mat_t::Scalar;

    const mat_t* mat_list_;
    size_t n_mats_;
    std::vector<uint32_t> n_cum_sum_; // [n_cum_sum_[i], n_cum_sum_[i+1])
                                      // is the range of features for block i.

    GHOSTBASIL_STRONG_INLINE
    auto n_features() const { return n_cum_sum_.back(); }

public:
    using Scalar = value_t;
    using Index = typename Eigen::Index;

    template <class MatrixListType>
    BlockMatrix(const MatrixListType& mat_list)
        : mat_list_(mat_list.data()),
          n_mats_(mat_list.size())
    {
        // Check that each matrix is square.
        for (size_t i = 0; i < n_mats_; ++i) {
            const auto& B = mat_list_[i];
            if (B.rows() != B.cols()) {
                std::string error = 
                    "Matrix at index " + std::to_string(i) + " is not square. ";
                throw std::runtime_error(error);
            }
        }

        // Compute the cumulative number of features.
        n_cum_sum_.resize(n_mats_+1);
        n_cum_sum_[0] = 0;
        for (size_t i = 0; i < n_mats_; ++i) {
            n_cum_sum_[i+1] = n_cum_sum_[i] + mat_list_[i].cols();
        }
    }

    GHOSTBASIL_STRONG_INLINE Index rows() const { return n_features(); }
    GHOSTBASIL_STRONG_INLINE Index cols() const { return n_features(); }

    template <class VecType>
    GHOSTBASIL_STRONG_INLINE
    value_t col_dot(size_t k, const VecType& v) const
    {
        assert(k < cols());

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

        // Find v_{i(k)}, i(k)th block of vector. 
        const auto vi = v.segment(n_cum_sum_[ik], n_cum_sum_[ik+1]-n_cum_sum_[ik]);

        // Find the shifted k relative to A_{i(k)}).
        const size_t k_shifted = k - n_cum_sum_[ik];

        return B.col_dot(k_shifted, vi);
    }

    template <class VecType>
    GHOSTBASIL_STRONG_INLINE
    value_t quad_form(const VecType& v) const
    {
        value_t quadform = 0;
        for (size_t i = 0; i < n_mats_; ++i) {
            const auto& B = mat_list_[i];
            const auto vi = v.segment(n_cum_sum_[i], n_cum_sum_[i+1]-n_cum_sum_[i]);
            quadform += B.quad_form(vi);
        }
        return quadform;
    }

    template <class VecType>
    GHOSTBASIL_STRONG_INLINE
    value_t inv_quad_form(value_t s, const VecType& v) const
    {
        value_t inv_quadform = 0;
        for (size_t i = 0; i < n_mats_; ++i) {
            const auto& B = mat_list_[i];
            const auto vi = v.segment(n_cum_sum_[i], n_cum_sum_[i+1]-n_cum_sum_[i]);
            inv_quadform += B.inv_quad_form(s, vi);
        }
        return inv_quadform;
    }
};

} // namespace ghostbasil
