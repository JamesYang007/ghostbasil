#pragma once
#include <ghostbasil/util/exceptions.hpp>
#include <ghostbasil/util/functional.hpp>
#include <ghostbasil/util/macros.hpp>
#include <ghostbasil/util/types.hpp>
#include <ghostbasil/util/counting_iterator.hpp>
#include <ghostbasil/util/eigen/map_sparsevector.hpp>
#include <ghostbasil/matrix/forward_decl.hpp>

namespace ghostbasil {
namespace internal {

template <class AType, class ValueType, class SSType,
          class ASType, class IAType, class ADiagType,
          class SBType, class SGType>
GHOSTBASIL_STRONG_INLINE
void lasso_assert_valid_inputs(
        const AType& A,
        ValueType s,
        const SSType& strong_set,
        const ASType& active_set,
        const IAType& is_active,
        const ADiagType& strong_A_diag,
        const SBType& strong_beta,
        const SGType& strong_grad)
{
#ifndef NDEBUG
    // check that A is square
    assert((A.rows() == A.cols()) && A.size());

    // check that s is in [0,1]
    assert((0 <= s) && (s <= 1));

    {
        // check that strong set contains values in [0, p)
        using ss_value_t = typename std::decay_t<SSType>::value_type;
        Eigen::Map<const util::vec_type<ss_value_t>> ss_view(
                strong_set.data(),
                strong_set.size());
        assert(!ss_view.size() || (0 <= ss_view.minCoeff() && ss_view.maxCoeff() < A.cols()));

        // check that strong set is sorted.
        util::vec_type<ss_value_t> ss_copy = ss_view;
        std::sort(ss_copy.data(), ss_copy.data()+ss_copy.size());
        assert((ss_copy.array() == ss_view.array()).all());
    }

    {
        // check that strong_A_diag satisfies the conditions.
        assert(strong_A_diag.size() == strong_set.size());
        for (size_t i = 0; i < strong_A_diag.size(); ++i) {
            auto k = strong_set[i];
            auto A_kk = A.coeff(k, k);
            assert(strong_A_diag[i] == A_kk);
        }
    }

    {
        // check that active set is of right size and contains values in [0, strong_set.size()).
        using as_value_t = typename std::decay_t<ASType>::value_type;
        Eigen::Map<const util::vec_type<as_value_t>> as_view(
                active_set.data(),
                active_set.size());
        assert(active_set.size() <= strong_set.size());
        assert(!active_set.size() || (0 <= as_view.minCoeff() && as_view.maxCoeff() < strong_set.size()));

        // check that active set is sorted.
        util::vec_type<as_value_t> as_copy = as_view;
        std::sort(as_copy.data(), as_copy.data()+as_copy.size());
        assert((as_copy.array() == as_view.array()).all());
    }

    // check that is_active is right size and contains correct active set variables.
    { 
        assert(is_active.size() == strong_set.size());
        size_t n_active = 0;
        for (size_t i = 0; i < active_set.size(); ++i) {
            if (is_active[active_set[i]]) ++n_active;
        }
        assert(n_active == active_set.size());
    }

    // check that strong_beta and strong_grad agree in size with strong_set.
    {
        assert(strong_beta.size() == strong_set.size());
        assert(strong_grad.size() == strong_set.size());
    }
#endif
}

} // namespace internal

/*
 * Computes the objective that we wish to minimize.
 * The objective is the quadratic loss + regularization:
 *
 *      (1-s)/2 \beta^\top A \beta - \beta^\top r + s/2 ||\beta||_2^2 
 *          + \lambda ||\beta||_1
 */
template <class AType, class RType, class ValueType, class BetaType>
GHOSTBASIL_STRONG_INLINE 
auto objective(
        const AType& A,
        const RType& r,
        ValueType s,
        ValueType lmda,
        const BetaType& beta)
{
    return (1-s)/2 * A.quad_form(beta) - beta.dot(r) + s/2 * beta.squaredNorm()
        + lmda * beta.cwiseAbs().sum();
}

/*
 * Updates the coefficient given the current state via coordinate descent rule.
 *
 * @param   coeff   current coefficient to update.
 * @param   x_var   variance of feature. A[k,k] where k is the feature corresponding to coeff.
 * @param   grad    current (negative) gradient for coeff.
 * @param   s       regularization of A towards identity.
 * @param   sc      1-s (for optimization purposes, assume user can provide pre-computed value).
 * @param   lmda    L1 regularization value.
 */
template <class ValueType>
GHOSTBASIL_STRONG_INLINE
void update_coefficient(
        ValueType& coeff,
        ValueType x_var,
        ValueType grad,
        ValueType s,
        ValueType sc,
        ValueType lmda)
{
    auto denom = sc * x_var + s;
    auto u = grad + coeff * denom;
    auto v = std::abs(u) - lmda;
    coeff = (v > 0.0) ? std::copysign(v,u)/denom : 0;
}

/*
 * Updates the convergence measure using variance of feature and coefficient change.
 *
 * @param   convg_measure   current convergence measure to update.
 * @param   coeff_diff      new coefficient minus old coefficient.
 * @param   x_var           variance of feature. A[k,k] where k is the feature corresponding to coeff_diff.
 */
template <class ValueType>
GHOSTBASIL_STRONG_INLINE
void update_convergence_measure(
        ValueType& convg_measure,
        ValueType coeff_diff,
        ValueType x_var)
{
    auto convg_measure_curr = x_var * coeff_diff * coeff_diff;
    convg_measure = std::max(convg_measure_curr, convg_measure);
}

/*
 * Updates the R^2 under the current state.
 *
 * @param   rsq         R^2 to update.
 * @param   old_coeff   old coefficient.
 * @param   new_coeff   new coefficient.
 * @param   x_var       variance of feature. A[k,k].
 * @param   grad        (negative) gradient corresponding to the coefficient.
 * @param   s           regularization of A towards identity.
 * @param   sc          1-s.
 */
template <class ValueType>
GHOSTBASIL_STRONG_INLINE
void update_rsq(
        ValueType& rsq, 
        ValueType old_coeff, 
        ValueType new_coeff, 
        ValueType x_var, 
        ValueType grad, 
        ValueType s, 
        ValueType sc)
{
    auto del = new_coeff - old_coeff;
    auto x_var_reg = sc * x_var + s;
    rsq += del * (2 * grad - del * x_var_reg);
}

/*
 * Coordinate descent (one loop) over a possibly subset of strong variables.
 * The iterators begin, end should return indices into strong_set.
 *
 * @param   begin           begin iterator to strong variables.
 *                          Note that this iterator can subset the strong set.
 * @param   end             end iterator to strong variables.
 * @praam   strong_set      strong set.
 * @param   strong_A_diag   diagonal of A corresponding to strong_set.
 * @param   A               covariance matrix.
 * @param   s               regularization of A towards identity.
 * @param   lmda            L1 regularization value.
 * @param   strong_beta     coefficients corresponding to strong_set.
 * @param   strong_grad     (negative) gradient corresponding to strong_beta.
 * @param   convg_measure   stores the convergence measure of the call.
 * @param   rsq             R^2 of current beta. Stores the updated value after the call.
 */
template <class Iter, class StrongSetType, 
          class StrongADiagType, class AType, class ValueType,
          class StrongBetaType, class StrongGradType,
          class AdditionalStepType=util::no_op>
GHOSTBASIL_STRONG_INLINE
void coordinate_descent(
        Iter begin,
        Iter end,
        const StrongSetType& strong_set,
        const StrongADiagType& strong_A_diag,
        const AType& A,
        ValueType s,
        ValueType lmda,
        StrongBetaType& strong_beta,
        StrongGradType& strong_grad,
        ValueType& convg_measure,
        ValueType& rsq,
        AdditionalStepType additional_step=AdditionalStepType())
{
    const auto sc = 1-s;

    convg_measure = 0;
    for (auto it = begin; it != end; ++it) {
        auto ss_idx = *it;              // index to strong set
        auto k = strong_set[ss_idx];    // actual feature index
        auto ak = strong_beta[ss_idx];  // corresponding beta
        auto gk = strong_grad[ss_idx];  // corresponding gradient
        auto A_kk = strong_A_diag[ss_idx];  // corresponding A diagonal element
                                    
        auto& ak_ref = strong_beta[ss_idx];
        update_coefficient(ak_ref, A_kk, gk, s, sc, lmda);

        if (ak_ref == ak) continue;

        auto del = ak_ref - ak;

        // update measure of convergence
        update_convergence_measure(convg_measure, del, A_kk);

        // update rsq
        update_rsq(rsq, ak, ak_ref, A_kk, gk, s, sc);

        // update gradient
        strong_grad[ss_idx] -= s * del;
        for (auto jt = begin; jt != end; ++jt) {
            auto ss_idx_j = *jt;
            auto j = strong_set[ss_idx_j];
            auto A_jk = A.coeff(j, k);
            strong_grad[ss_idx_j] -= sc * A_jk * del;
        }

        // additional step
        additional_step(ss_idx);
    }
}

template <class Iter, class StrongSetType, 
          class StrongADiagType, class MatType, class ValueType,
          class StrongBetaType, class StrongGradType,
          class AdditionalStepType=util::no_op>
GHOSTBASIL_STRONG_INLINE
void coordinate_descent(
        Iter begin,
        Iter end,
        const StrongSetType& strong_set,
        const StrongADiagType& strong_A_diag,
        const BlockMatrix<MatType>& A,
        ValueType s,
        ValueType lmda,
        StrongBetaType& strong_beta,
        StrongGradType& strong_grad,
        ValueType& convg_measure,
        ValueType& rsq,
        AdditionalStepType additional_step=AdditionalStepType())
{
    const auto sc = 1-s;

    // Note: here we really assume sorted-ness!
    // Since begin->end results in increasing sequence of strong set indices,
    // we can update the block iterator as we increment begin.
    auto block_it = A.block_begin();

    convg_measure = 0;
    for (auto it = begin; it != end; ++it) {
        auto ss_idx = *it;              // index to strong set
        auto k = strong_set[ss_idx];    // actual feature index
        auto ak = strong_beta[ss_idx];  // corresponding beta
        auto gk = strong_grad[ss_idx];  // corresponding gradient
        auto A_kk = strong_A_diag[ss_idx];  // corresponding A diagonal element
                                    
        auto& ak_ref = strong_beta[ss_idx];
        update_coefficient(ak_ref, A_kk, gk, s, sc, lmda);

        if (ak_ref == ak) continue;

        auto del = ak_ref - ak;

        // update measure of convergence
        update_convergence_measure(convg_measure, del, A_kk);

        // update rsq
        update_rsq(rsq, ak, ak_ref, A_kk, gk, s, sc);

        // update A block stride pointer if current feature is not in the block.
        if (!block_it.is_in_block(k)) block_it.advance_at(k);
        auto k_shifted = block_it.shift(k);
        const auto& block = block_it.block();
        
        // update gradient
        strong_grad[ss_idx] -= s * del;
        for (auto jt = begin; jt != end; ++jt) {
            auto ss_idx_j = *jt;
            auto j = strong_set[ss_idx_j];
            // optimization: if j is not in the current block, no update.
            if (!block_it.is_in_block(j)) continue;
            const auto j_shifted = block_it.shift(j);
            auto A_jk = block.coeff(j_shifted, k_shifted);
            strong_grad[ss_idx_j] -= sc * A_jk * del;
        }

        // additional step
        additional_step(ss_idx);
    }
}

/*
 * Coordinate descent on the active set to minimize the ghostbasil objective.
 * See "objective" function for the objective of interest.
 *
 * @param   A           covariance matrix (p x p). 
 * @param   s           regularization of A towards identity. 
 *                      It is undefined behavior is s is not in [0,1].
 * @param   strong_set  strong set as a dense vector of sorted indices in [0, p).
 *                      strong_set[i] = ith strong feature.
 * @param   active_set  active set as a dense vector of sorted indices in [0, strong_set.size()).
 *                      strong_set[active_set[i]] = ith active feature.
 * @param   is_active   dense vector of bool of size strong_set.size(). 
 *                      is_active[i] = true if feature strong_set[i] is active.
 *                      active_set should contain i.
 * @param   strong_A_diag       dense vector representing diagonal of A restricted to strong_set.
 *                              strong_A_diag[i] = A[k,k] where k = strong_set[i].
 *                              It is of size strong_set.size().
 * @param   lmda_idx    index into the lambda sequence for logging purposes.
 * @param   lmda        L1 regularization of loss.
 * @param   max_cds     max number of coordinate descents.
 * @param   thr         convergence threshold.
 * @param   strong_beta dense vector of coefficients of size strong_set.size().
 *                      strong_beta[i] = coefficient for feature strong_set[i].
 *                      The updated coefficients will be stored here.
 * @param   strong_grad dense vector of (negative) gradient of objective of size strong_set.size().
 *                      strong_grad[i] = (negative) gradient for feature strong_set[i] at beta.
 *                      The updated gradients will be stored here.
 * @param   rsq         unnormalized R^2 estimate (same as negative loss function).
 *                      It is only well-defined if it is (approximately) 
 *                      the R^2 that corresponds to beta.
 *                      The updated R^2 estimate will be stored here.
 * @param   n_cds       stores the number of coordinate descents from this call.
 * @param   sg_update   functor that updates strong_gradient.
 */
template <class AType, class ValueType, class SSType, 
          class ASType, class IAType, class StrongADiagType, 
          class SBType, class SGType, class SGUpdateType>
GHOSTBASIL_STRONG_INLINE
void lasso_active_impl(
    const AType& A,
    ValueType s,
    const SSType& strong_set,
    const ASType& active_set,
    const IAType& is_active,
    const StrongADiagType& strong_A_diag,
    size_t lmda_idx,
    ValueType lmda,
    size_t max_cds,
    ValueType thr,
    SBType& strong_beta,
    SGType& strong_grad,
    ValueType& rsq,
    size_t& n_cds,
    SGUpdateType sg_update)
{
    using value_t = ValueType;

    internal::lasso_assert_valid_inputs(
            A, s, strong_set, active_set, is_active, strong_A_diag,
            strong_beta, strong_grad);

    auto strong_beta_diff = strong_beta;

    while (1) {
        ++n_cds;
        value_t convg_measure;
        coordinate_descent(
                active_set.begin(), active_set.end(),
                strong_set, strong_A_diag, A, s, lmda,
                strong_beta, strong_grad,
                convg_measure, rsq);
        if (convg_measure < thr) break;
        if (n_cds >= max_cds) throw util::max_cds_error(lmda_idx);
    }
    
    Eigen::Map<const util::vec_type<value_t>> sb_view(
            strong_beta.data(), strong_beta.size());
    Eigen::Map<util::vec_type<value_t>> sb_diff_view(
            strong_beta_diff.data(), strong_beta_diff.size());

    sb_diff_view = sb_view - sb_diff_view;

    // update strong gradient
    sg_update(A, strong_set, is_active, strong_beta_diff, strong_grad);
}


/*
 * Calls lasso_active_impl with a specialized gradient update routine
 * for a generic matrix.
 */
template <class AType, class ValueType, class SSType, 
          class ASType, class IAType, class StrongADiagType, 
          class SBType, class SGType>
GHOSTBASIL_STRONG_INLINE 
void lasso_active(
    const AType& A,
    ValueType s,
    const SSType& strong_set,
    const ASType& active_set,
    const IAType& is_active,
    const StrongADiagType& strong_A_diag,
    size_t lmda_idx,
    ValueType lmda,
    size_t max_cds,
    ValueType thr,
    SBType& strong_beta,
    SGType& strong_grad,
    ValueType& rsq,
    size_t& n_cds)
{
    auto sg_update = [s](const auto& A_,
                         const auto& strong_set_,
                         const auto& is_active_,
                         const auto& beta_diff_,
                         auto& strong_grad_) {
        using value_t = ValueType;
        using index_t = typename std::decay_t<decltype(strong_set_[0])>;
        using sp_vec_t = util::sp_vec_type<value_t, Eigen::ColMajor, index_t>;
        const auto sc = 1-s;
        Eigen::Map<const sp_vec_t> beta_diff_map(
                A_.cols(),
                strong_set_.size(),
                strong_set_.data(),
                beta_diff_.data());
        // update gradient in non-active positions
        for (size_t ss_idx = 0; ss_idx < strong_set_.size(); ++ss_idx) {
            if (is_active_[ss_idx]) continue;
            auto k = strong_set_[ss_idx];
            strong_grad_[ss_idx] -= sc * A_.col_dot(k, beta_diff_map);
        }
    };

    lasso_active_impl(
            A, s, strong_set, active_set, is_active, strong_A_diag,
            lmda_idx, lmda, max_cds, thr, 
            strong_beta, strong_grad, rsq, n_cds, sg_update);
}

/*
 * Calls lasso_active_impl with a specialized gradient update routine
 * for BlockMatrix.
 */
template <class MatType, class ValueType, class SSType, 
          class ASType, class IAType, class StrongADiagType, 
          class SBType, class SGType>
GHOSTBASIL_STRONG_INLINE 
void lasso_active(
    const BlockMatrix<MatType>& A,
    ValueType s,
    const SSType& strong_set,
    const ASType& active_set,
    const IAType& is_active,
    const StrongADiagType& strong_A_diag,
    size_t lmda_idx,
    ValueType lmda,
    size_t max_cds,
    ValueType thr,
    SBType& strong_beta,
    SGType& strong_grad,
    ValueType& rsq,
    size_t& n_cds)
{
    auto sg_update = [s](const auto& A_,
                         const auto& strong_set_,
                         const auto& is_active_,
                         const auto& beta_diff_,
                         auto& strong_grad_) {
        using value_t = ValueType;
        using index_t = std::decay_t<decltype(strong_set_[0])>;
        using sp_vec_t = util::sp_vec_type<value_t, Eigen::ColMajor, index_t>;
        const auto sc = 1-s;
        // TODO: check this? with segment?
        Eigen::Map<const sp_vec_t> beta_diff_map(
                A_.cols(),
                strong_set_.size(),
                strong_set_.data(),
                beta_diff_.data());

        auto block_it = A_.block_begin();

        // update gradient in non-active positions
        for (size_t ss_idx = 0; ss_idx < strong_set_.size(); ++ss_idx) {
            if (is_active_[ss_idx]) continue;
            auto k = strong_set_[ss_idx];
            // update A block stride pointer if current feature is not in the block.
            if (!block_it.is_in_block(k)) block_it.advance_at(k);
            const auto k_shifted = block_it.shift(k);
            const auto stride_begin = block_it.stride();
            const auto& A_block = block_it.block();
            const auto beta_diff_map_seg = 
                beta_diff_map.segment(stride_begin, A_block.cols());
            strong_grad_[ss_idx] -= sc * A_block.col_dot(k_shifted, beta_diff_map_seg);
        }
    };

    lasso_active_impl(
            A, s, strong_set, active_set, is_active, strong_A_diag,
            lmda_idx, lmda, max_cds, thr, 
            strong_beta, strong_grad, rsq, n_cds, sg_update);
}

/*
 * Minimizes the objective described in the function "objective"
 * with the additional constraint:
 * \[
 *      \beta_{i} = 0, \, \forall i \notin \{strong_set\}
 * \]
 * i.e. all betas not in strong_set are fixed to be 0.
 *
 * See lasso_active for parameter descriptions.
 *
 * @param   lmdas       regularization parameter lambda sequence.
 * @param   betas       output coefficient sparse matrix (p x lmdas.size()).
 *                      betas(i,j) = ith coefficient for jth lambda.
 * @param   rsqs        dense vector to store R^2 values for each lambda in lmdas.
 */
template <class AType, class ValueType,
          class SSType, class SADType,
          class LmdasType,
          class SBType, class SGType,
          class ASType, class IAType,
          class BetasType, class RsqsType>
inline void lasso(
    const AType& A, 
    ValueType s, 
    const SSType& strong_set, 
    const SADType& strong_A_diag,
    const LmdasType& lmdas, 
    size_t max_cds,
    ValueType thr,
    ValueType rsq,
    SBType& strong_beta, 
    SGType& strong_grad,
    ASType& active_set,
    IAType& is_active,
    BetasType& betas, 
    RsqsType& rsqs,
    size_t& n_cds,
    size_t& n_lmdas)
{
    internal::lasso_assert_valid_inputs(
            A, s, strong_set, active_set, is_active, strong_A_diag,
            strong_beta, strong_grad);

    assert(betas.size() == lmdas.size());
    assert(rsqs.size() == lmdas.size());

    using value_t = ValueType;
    using index_t = std::decay_t<decltype(strong_set[0])>;
    using sp_vec_t = util::sp_vec_type<value_t, Eigen::ColMajor, index_t>;
    
    bool lasso_active_called = false;

    n_cds = 0;
    n_lmdas = 0;

    Eigen::Map<const sp_vec_t> strong_beta_map(
            A.cols(),
            strong_set.size(),
            strong_set.data(),
            strong_beta.data());

    auto add_active_set = [&](auto ss_idx) {
        if (!is_active[ss_idx]) {
            is_active[ss_idx] = true;
            active_set.push_back(ss_idx);
        }
    };

    auto lasso_active_and_update = [&](size_t l, auto lmda) {
        lasso_active(
                A, s, strong_set, active_set, is_active, strong_A_diag,
                l, lmda, max_cds, thr, 
                strong_beta, strong_grad, rsq, n_cds);
        lasso_active_called = true;
    };

    for (size_t l = 0; l < lmdas.size(); ++l) {
        const auto lmda = lmdas[l];
        const auto rsq_prev = rsq;

        if (lasso_active_called) {
            lasso_active_and_update(l, lmda);
        }

        while (1) {
            ++n_cds;
            value_t convg_measure;
            coordinate_descent(
                    util::counting_iterator<>(0),
                    util::counting_iterator<>(strong_set.size()),
                    strong_set, strong_A_diag, A, s, lmda,
                    strong_beta, strong_grad,
                    convg_measure, rsq, add_active_set);
            if (convg_measure < thr) break;
            if (n_cds >= max_cds) throw util::max_cds_error(l);

            // since coordinate descent could have added new active variables,
            // we sort active set to preserve invariant.
            // NOTE: speed shouldn't be affected since most of the array is already sorted
            // and std::sort is really fast in this setting!
            // See: https://solarianprogrammer.com/2012/10/24/cpp-11-sort-benchmark/
            std::sort(active_set.begin(), active_set.end());

            lasso_active_and_update(l, lmda);
        }

        betas[l] = strong_beta_map;
        rsqs[l] = rsq;
        ++n_lmdas;

        if (l == 0) continue;
        if (rsq-rsq_prev < 1e-5*rsq) break;
    }
}

} // namespace ghostbasil
