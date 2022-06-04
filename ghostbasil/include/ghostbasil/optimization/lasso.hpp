#pragma once
#include <numeric>
#include <ghostbasil/util/exceptions.hpp>
#include <ghostbasil/util/functional.hpp>
#include <ghostbasil/util/macros.hpp>
#include <ghostbasil/util/types.hpp>
#include <ghostbasil/util/functor_iterator.hpp>
#include <ghostbasil/util/eigen/map_sparsevector.hpp>
#include <ghostbasil/matrix/forward_decl.hpp>

namespace ghostbasil {
namespace internal {

template <class AType, class ValueType, class SSType, class SOType,
          class ASType, class AOType, class ASOType, class IAType, class ADiagType,
          class SBType, class SGType>
GHOSTBASIL_STRONG_INLINE
void lasso_assert_valid_inputs(
        const AType& A,
        ValueType s,
        const SSType& strong_set,
        const SOType& strong_order,
        const ASType& active_set,
        const AOType& active_order,
        const ASOType& active_set_ordered,
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

        // check that strong order results in sorted strong set
        util::vec_type<ss_value_t> ss_copy = ss_view;
        std::sort(ss_copy.data(), ss_copy.data()+ss_copy.size());
        assert(strong_order.size() == strong_set.size());
        assert((ss_copy.array() == 
                    util::vec_type<ss_value_t>::NullaryExpr(ss_view.size(),
                        [&](auto i) { return ss_view[strong_order[i]]; }
                        ).array()).all());
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

        // check that active order is sorted.
        using ao_value_t = typename std::decay_t<AOType>::value_type;
        Eigen::Map<const util::vec_type<ao_value_t>> ao_view(
                active_order.data(),
                active_order.size());
        util::vec_type<ao_value_t> ao_copy = ao_view;
        std::sort(ao_copy.data(), ao_copy.data()+ao_copy.size(),
                  [&](auto i, auto j) { 
                        return strong_set[active_set[i]] < strong_set[active_set[j]]; 
                    });
        assert((ao_copy.array() == ao_view.array()).all());

        // check that active set contains at least the non-zero betas
        for (size_t i = 0; i < strong_set.size(); ++i) {
            if (strong_beta[i] == 0) continue;
            auto it = std::find(active_set.begin(), active_set.end(), i);
            assert(it != active_set.end());
        }

        // check that active_set_ordered is truly ordered
        using aso_value_t = typename std::decay_t<ASOType>::value_type;
        util::vec_type<aso_value_t> aso_copy =
            util::vec_type<aso_value_t>::NullaryExpr(
                    active_order.size(),
                    [&](auto i) { return strong_set[active_set[active_order[i]]]; });
        Eigen::Map<const util::vec_type<aso_value_t>> aso_view(
                active_set_ordered.data(),
                active_set_ordered.size());
        assert((aso_copy.array() == aso_view.template cast<as_value_t>().array()).all());
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

template<class ForwardIt, class F, class T>
ForwardIt lower_bound(ForwardIt first, ForwardIt last, F f, T value)
{
    ForwardIt it = first;
    typename std::iterator_traits<ForwardIt>::difference_type count, step;
    count = std::distance(first, last);

    while (count > 0) {
        it = first;
        step = count / 2;
        std::advance(it, step);
        if (f(*it) < value) {
            first = ++it;
            count -= step + 1;
        }
        else
            count = step;
    }
    return first;
}

} // namespace internal
  
/*
 * Checks early stopping based on R^2 values.
 * Returns true (early stopping should occur) if both are true:
 *
 *      delta_u := (R^2_u - R^2_m)/R^2_u
 *      delta_m := (R^2_m - R^2_l)/R^2_m 
 *      delta_u < cond_0_thresh 
 *      AND
 *      (delta_u - delta_m) < cond_1_thresh
 *
 * @param   rsq_l   third to last R^2 value.
 * @param   rsq_m   second to last R^2 value.
 * @param   rsq_u   last R^2 value.
 */
template <class ValueType>
GHOSTBASIL_STRONG_INLINE
bool check_early_stop_rsq(
        ValueType rsq_l,
        ValueType rsq_m,
        ValueType rsq_u,
        ValueType cond_0_thresh = 1e-5,
        ValueType cond_1_thresh = 1e-5)
{
    const auto delta_u = (rsq_u-rsq_m);
    const auto delta_m = (rsq_m-rsq_l);
    return ((delta_u < cond_0_thresh*rsq_u) &&
            ((delta_m*rsq_u-delta_u*rsq_m) < cond_1_thresh*rsq_m*rsq_u));
}

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
    const auto denom = sc * x_var + s;
    const auto u = grad + coeff * denom;
    const auto v = std::abs(u) - lmda;
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
    const auto convg_measure_curr = x_var * coeff_diff * coeff_diff;
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
    const auto del = new_coeff - old_coeff;
    const auto x_var_reg = sc * x_var + s;
    rsq += del * (2 * grad - del * x_var_reg);
}

/*
 * Coordinate descent (one loop) over a possibly subset of strong variables.
 * See "lasso_active" for more descriptions of arguments.
 *
 * @param   begin           begin iterator to indices into strong set, i.e.
 *                          strong_set[*begin] is the current feature to descend.
 *                          The resulting sequence of indices from calling 
 *                          strong_set[*(begin++)] MUST be ordered.
 * @param   end             end iterator to indices into strong set.
 * @praam   strong_set      strong set of indices of features.
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
          class StrongADiagType, class DerivedType, class ValueType,
          class StrongBetaType, class StrongGradType,
          class AdditionalStepType=util::no_op>
GHOSTBASIL_STRONG_INLINE
void coordinate_descent(
        Iter begin,
        Iter end,
        const StrongSetType& strong_set,
        const StrongADiagType& strong_A_diag,
        const Eigen::DenseBase<DerivedType>& A_base,
        ValueType s,
        ValueType lmda,
        StrongBetaType& strong_beta,
        StrongGradType& strong_grad,
        ValueType& convg_measure,
        ValueType& rsq,
        AdditionalStepType additional_step=AdditionalStepType())
{
    const auto& A = A_base.derived();
    const auto sc = 1-s;

    convg_measure = 0;
    for (auto it = begin; it != end; ++it) {
        const auto ss_idx = *it;              // index to strong set
        const auto k = strong_set[ss_idx];    // actual feature index
        const auto ak = strong_beta[ss_idx];  // corresponding beta
        const auto gk = strong_grad[ss_idx];  // corresponding gradient
        const auto A_kk = strong_A_diag[ss_idx];  // corresponding A diagonal element
                                    
        auto& ak_ref = strong_beta[ss_idx];
        update_coefficient(ak_ref, A_kk, gk, s, sc, lmda);

        if (ak_ref == ak) continue;

        const auto del = ak_ref - ak;

        // update measure of convergence
        update_convergence_measure(convg_measure, del, A_kk);

        // update rsq
        update_rsq(rsq, ak, ak_ref, A_kk, gk, s, sc);

        // update gradient
        strong_grad[ss_idx] -= s * del;
        const auto sc_del = sc * del;
        for (auto jt = begin; jt != end; ++jt) {
            const auto ss_idx_j = *jt;
            const auto j = strong_set[ss_idx_j];
            const auto A_jk = A.coeff(j, k);
            strong_grad[ss_idx_j] -= sc_del * A_jk;
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
    const auto get_strong_set = [&](auto i) { return strong_set[i]; };

    // Note: here we really assume sorted-ness!
    // Since begin->end results in increasing sequence of strong set indices,
    // we can update the block iterator as we increment begin.
    auto block_it = A.block_begin();

    // We can also keep track of the range of strong set indices
    // that produce indices within the current block.
    auto range_begin = begin;
    auto range_end = internal::lower_bound(
            range_begin, end, get_strong_set,
            block_it.stride() + block_it.block().cols());

    convg_measure = 0;
    for (auto it = begin; it != end; ++it) {
        const auto ss_idx = *it;              // index to strong set
        const auto k = strong_set[ss_idx];    // actual feature index
        const auto ak = strong_beta[ss_idx];  // corresponding beta
        const auto gk = strong_grad[ss_idx];  // corresponding gradient
        const auto A_kk = strong_A_diag[ss_idx];  // corresponding A diagonal element
                                    
        auto& ak_ref = strong_beta[ss_idx];
        update_coefficient(ak_ref, A_kk, gk, s, sc, lmda);

        if (ak_ref == ak) continue;

        const auto del = ak_ref - ak;

        // update measure of convergence
        update_convergence_measure(convg_measure, del, A_kk);

        // update rsq
        update_rsq(rsq, ak, ak_ref, A_kk, gk, s, sc);

        // update outer iterators to preserve invariant
        if (!block_it.is_in_block(k)) {
            block_it.advance_at(k);
            range_begin = internal::lower_bound(
                    range_end, end, get_strong_set,
                    block_it.stride());
            range_end = internal::lower_bound(
                    range_begin, end, get_strong_set,
                    block_it.stride() + block_it.block().cols());
        }
        const auto k_shifted = block_it.shift(k);
        const auto& block = block_it.block();
        
        // update gradient
        strong_grad[ss_idx] -= s * del;
        const auto sc_del = sc * del;
        for (auto jt = range_begin; jt != range_end; ++jt) {
            const auto ss_idx_j = *jt;
            const auto j = strong_set[ss_idx_j];
            const auto j_shifted = block_it.shift(j);
            const auto A_jk = block.coeff(j_shifted, k_shifted);
            strong_grad[ss_idx_j] -= sc_del * A_jk;
        }

        // additional step
        additional_step(ss_idx);
    }
}

/*
 * Applies multiple coordinate descent on the active set 
 * to minimize the ghostbasil objective.
 * See "objective" function for the objective of interest.
 *
 * @param   A           covariance matrix (p x p). 
 * @param   s           regularization of A towards identity. 
 *                      It is undefined behavior is s is not in [0,1].
 * @param   strong_set  strong set as a dense vector of indices in [0, p).
 *                      strong_set[i] = ith strong feature.
 * @param   strong_order    order of strong_set that results in sorted (ascending) values.
 *                          strong_set[strong_order[i]] < strong_set[strong_order[j]] if i < j.
 * @param   active_set  active set as a dense vector of indices in [0, strong_set.size()).
 *                      strong_set[active_set[i]] = ith active feature.
 *                      This set must at least contain ALL indices into strong_set 
 *                      where the corresponding strong_beta is non-zero, that is,
 *                      if strong_beta[strong_set[i]] != 0, then i is in active_set.
 * @param   active_order    order of active_set that results in sorted (ascending) 
 *                          values of strong_set.
 *                          strong_set[active_set[active_order[i]]] < 
 *                              strong_set[active_set[active_order[j]]] if i < j.
 * @param   active_set_ordered  ordered *features corresponding to* active_set.
 *                              active_set_ordered[i] == strong_set[active_set[active_order[i]]].
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
 * @param   active_beta_diff_ordered    dense vector to store coefficient difference corresponding
 *                                      to the active set between
 *                                      the new coefficients and the current coefficients
 *                                      after performing coordinate descent on the active set.
 *                                      It must be initialized to be of size active_order.size(),
 *                                      though the values need not be initialized.
 *                                      active_beta_diff_ordered[i] = 
 *                                          (new minus old of) 
 *                                          strong_beta[active_set[active_order[i]]].
 * @param   rsq         unnormalized R^2 estimate (same as negative loss function).
 *                      It is only well-defined if it is (approximately) 
 *                      the R^2 that corresponds to beta.
 *                      The updated R^2 estimate will be stored here.
 * @param   n_cds       stores the number of coordinate descents from this call.
 * @param   sg_update   functor that updates strong_gradient.
 */
template <class AType, class ValueType, class SSType, class SOType, 
          class ASType, class AOType, class ASOType,class IAType, class StrongADiagType, 
          class SBType, class SGType, class ABDiffOType, class SGUpdateType,
          class CUIType = util::no_op>
GHOSTBASIL_STRONG_INLINE
void lasso_active_impl(
    const AType& A,
    ValueType s,
    const SSType& strong_set,
    const SOType& strong_order,
    const ASType& active_set,
    const AOType& active_order,
    const ASOType& active_set_ordered,
    const IAType& is_active,
    const StrongADiagType& strong_A_diag,
    size_t lmda_idx,
    ValueType lmda,
    size_t max_cds,
    ValueType thr,
    SBType& strong_beta,
    SGType& strong_grad,
    ABDiffOType& active_beta_diff_ordered,
    ValueType& rsq,
    size_t& n_cds,
    SGUpdateType sg_update,
    CUIType check_user_interrupt = CUIType())
{
    using value_t = ValueType;
    using ao_value_t = typename std::decay_t<AOType>::value_type;
    using aso_value_t= typename std::decay_t<ASOType>::value_type;
    using sp_vec_t = util::sp_vec_type<value_t, Eigen::ColMajor, aso_value_t>;

    internal::lasso_assert_valid_inputs(
            A, s, strong_set, strong_order, 
            active_set, active_order, active_set_ordered,
            is_active, strong_A_diag,
            strong_beta, strong_grad);

    assert(active_beta_diff_ordered.size() == active_order.size());
    Eigen::Map<util::vec_type<value_t>> ab_diff_o_view(
            active_beta_diff_ordered.data(), 
            active_beta_diff_ordered.size());
    const auto active_beta_ordered_expr = 
        util::vec_type<value_t>::NullaryExpr(
            active_order.size(),
            [&](auto i) { return strong_beta[active_set[active_order[i]]]; });
    ab_diff_o_view = active_beta_ordered_expr;

    const auto active_set_iter_f = [&](auto i) {
        return active_set[active_order[i]];
    };

    while (1) {
        check_user_interrupt(n_cds);
        ++n_cds;
        value_t convg_measure;
        coordinate_descent(
                util::make_functor_iterator<ao_value_t>(
                    0, active_set_iter_f),
                util::make_functor_iterator<ao_value_t>(
                    active_order.size(), active_set_iter_f),
                strong_set, strong_A_diag, A, s, lmda,
                strong_beta, strong_grad,
                convg_measure, rsq);
        if (convg_measure < thr) break;
        if (n_cds >= max_cds) throw util::max_cds_error(lmda_idx);
    }
    
    ab_diff_o_view = active_beta_ordered_expr - ab_diff_o_view;

    Eigen::Map<const sp_vec_t> active_beta_diff_map(
            A.cols(),
            active_set_ordered.size(),
            active_set_ordered.data(),
            active_beta_diff_ordered.data());

    // update strong gradient
    sg_update(active_beta_diff_map);
}


/*
 * Calls lasso_active_impl with a specialized gradient update routine
 * for a generic matrix.
 */
template <class DerivedType, class ValueType, class SSType, class SOType,
          class ASType, class AOType, class ASOType, class IAType, 
          class StrongADiagType, class SBType, class SGType,
          class ABDiffOType, class CUIType = util::no_op>
GHOSTBASIL_STRONG_INLINE 
void lasso_active(
    const Eigen::DenseBase<DerivedType>& A_base,
    ValueType s,
    const SSType& strong_set,
    const SOType& strong_order,
    const ASType& active_set,
    const AOType& active_order,
    const ASOType& active_set_ordered,
    const IAType& is_active,
    const StrongADiagType& strong_A_diag,
    size_t lmda_idx,
    ValueType lmda,
    size_t max_cds,
    ValueType thr,
    SBType& strong_beta,
    SGType& strong_grad,
    ABDiffOType& active_beta_diff_ordered,
    ValueType& rsq,
    size_t& n_cds,
    CUIType check_user_interrupt = CUIType())
{
    const auto& A = A_base.derived();
    const auto sg_update = [&](const auto& sp_beta_diff) {
        if (sp_beta_diff.nonZeros() == 0) return;

        const auto sc = 1-s;
        // update gradient in non-active positions
        for (size_t ss_idx = 0; ss_idx < strong_set.size(); ++ss_idx) {
            if (is_active[ss_idx]) continue;
            const auto k = strong_set[ss_idx];
            strong_grad[ss_idx] -= sc * A.col_dot(k, sp_beta_diff);
        }
    };

    lasso_active_impl(
            A, s, strong_set, strong_order, 
            active_set, active_order, active_set_ordered,
            is_active, strong_A_diag,
            lmda_idx, lmda, max_cds, thr, 
            strong_beta, strong_grad, 
            active_beta_diff_ordered, rsq, n_cds, sg_update,
            check_user_interrupt);
}

/*
 * Calls lasso_active_impl with a specialized gradient update routine
 * for BlockMatrix.
 */
template <class MatType, class ValueType, class SSType, class SOType, 
          class ASType, class AOType, class ASOType, class IAType, 
          class StrongADiagType, class SBType, class SGType,
          class ABDiffOType, class CUIType=util::no_op>
GHOSTBASIL_STRONG_INLINE 
void lasso_active(
    const BlockMatrix<MatType>& A,
    ValueType s,
    const SSType& strong_set,
    const SOType& strong_order,
    const ASType& active_set,
    const AOType& active_order,
    const ASOType& active_set_ordered,
    const IAType& is_active,
    const StrongADiagType& strong_A_diag,
    size_t lmda_idx,
    ValueType lmda,
    size_t max_cds,
    ValueType thr,
    SBType& strong_beta,
    SGType& strong_grad,
    ABDiffOType& active_beta_diff_ordered,
    ValueType& rsq,
    size_t& n_cds,
    CUIType check_user_interrupt = CUIType())
{
    auto sg_update = [&](const auto& sp_beta_diff) {
        using value_t = ValueType;
        const auto sc = 1-s;

        auto block_it = A.block_begin();
        const auto bd_inner = sp_beta_diff.innerIndexPtr();
        const auto bd_value = sp_beta_diff.valuePtr();
        const auto bd_nnz = sp_beta_diff.nonZeros();

        if (bd_nnz == 0) return;

        // initialized below
        size_t bd_seg_begin;
        size_t bd_seg_end; 

        {
            const auto curr_stride = block_it.stride(); // should be 0
            assert(curr_stride == 0);

            // find first time a non-zero index of beta is inside the current block
            const auto it = std::lower_bound(bd_inner, bd_inner+bd_nnz, curr_stride);
            bd_seg_begin = std::distance(bd_inner, it);

            // find first time a non-zero index of beta is outside the current block
            const auto next_stride = curr_stride+block_it.block().cols();
            const auto end = std::lower_bound(bd_inner+bd_seg_begin, bd_inner+bd_nnz, next_stride);
            bd_seg_end = std::distance(bd_inner, end);

            // these two define the index of bd_inner and size to read
            // to perform any dot product with a column of current block.
        }

        // update gradient in non-active positions
        for (size_t ss_idx : strong_order) {
            if (is_active[ss_idx]) continue;
            const auto k = strong_set[ss_idx];
            // update A block stride pointer if current feature is not in the block.
            if (!block_it.is_in_block(k)) {
                block_it.advance_at(k);

                const auto curr_stride = block_it.stride();
                const auto it = std::lower_bound(
                        bd_inner+bd_seg_end, 
                        bd_inner+bd_nnz, 
                        curr_stride);
                bd_seg_begin = std::distance(bd_inner, it);

                const auto next_stride = curr_stride+block_it.block().cols();
                const auto end = std::lower_bound(
                        bd_inner+bd_seg_begin, 
                        bd_inner+bd_nnz, 
                        next_stride);
                bd_seg_end = std::distance(bd_inner, end);
            }
            const auto k_shifted = block_it.shift(k);
            const auto& A_block = block_it.block();
            
            value_t dp = 0;
            for (auto i = bd_seg_begin; i < bd_seg_end; ++i) {
                dp += A_block.coeff(block_it.shift(bd_inner[i]), k_shifted) * bd_value[i];
            }
            strong_grad[ss_idx] -= sc * dp;
        }
    };

    lasso_active_impl(
            A, s, strong_set, strong_order, 
            active_set, active_order, active_set_ordered,
            is_active, strong_A_diag,
            lmda_idx, lmda, max_cds, thr, 
            strong_beta, strong_grad, 
            active_beta_diff_ordered, rsq, n_cds, sg_update,
            check_user_interrupt);
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
          class SSType, class SOType,
          class SADType, class LmdasType,
          class SBType, class SGType,
          class ASType, class AOType, class ASOType, 
          class IAType, class BetasType, class RsqsType,
          class CUIType = util::no_op>
inline void lasso(
    const AType& A, 
    ValueType s, 
    const SSType& strong_set, 
    const SOType& strong_order,
    const SADType& strong_A_diag,
    const LmdasType& lmdas, 
    size_t max_cds,
    ValueType thr,
    ValueType rsq,
    SBType& strong_beta, 
    SGType& strong_grad,
    ASType& active_set,
    AOType& active_order,
    ASOType& active_set_ordered,
    IAType& is_active,
    BetasType& betas, 
    RsqsType& rsqs,
    size_t& n_cds,
    size_t& n_lmdas,
    CUIType check_user_interrupt = CUIType())
{
    internal::lasso_assert_valid_inputs(
            A, s, strong_set, strong_order, 
            active_set, active_order, active_set_ordered, 
            is_active, strong_A_diag,
            strong_beta, strong_grad);

    assert(betas.size() == lmdas.size());
    assert(rsqs.size() == lmdas.size());

    using value_t = ValueType;
    using aso_value_t = typename std::decay_t<ASOType>::value_type;
    using beta_value_t = typename std::decay_t<SBType>::value_type;
    using sp_vec_t = util::sp_vec_type<beta_value_t, Eigen::ColMajor, aso_value_t>;

    // common buffers for the routine
    std::vector<beta_value_t> active_beta_ordered;
    std::vector<beta_value_t> active_beta_diff_ordered;
    active_beta_ordered.reserve(strong_set.size());
    active_beta_diff_ordered.reserve(strong_set.size());
    active_beta_diff_ordered.resize(active_order.size());
    
    bool lasso_active_called = false;
    n_cds = 0;
    n_lmdas = 0;

    const auto add_active_set = [&](auto ss_idx) {
        if (!is_active[ss_idx]) {
            is_active[ss_idx] = true;
            active_set.push_back(ss_idx);
        }
    };

    const auto lasso_active_and_update = [&](size_t l, auto lmda) {
        lasso_active(
                A, s, strong_set, strong_order, 
                active_set, active_order, active_set_ordered,
                is_active, strong_A_diag,
                l, lmda, max_cds, thr, 
                strong_beta, strong_grad, 
                active_beta_diff_ordered, rsq, n_cds,
                check_user_interrupt);
        lasso_active_called = true;
    };

    for (size_t l = 0; l < lmdas.size(); ++l) {
        const auto lmda = lmdas[l];

        if (lasso_active_called) {
            lasso_active_and_update(l, lmda);
        }

        size_t old_active_size;
        while (1) {
            check_user_interrupt(n_cds);
            ++n_cds;
            value_t convg_measure;
            old_active_size = active_set.size();
            coordinate_descent(
                    strong_order.begin(), strong_order.end(),
                    strong_set, strong_A_diag, A, s, lmda,
                    strong_beta, strong_grad,
                    convg_measure, rsq, add_active_set);
            const bool new_active_added = (old_active_size < active_set.size());

            // since coordinate descent could have added new active variables,
            // we update active_order and active_set_ordered to preserve invariant.
            // NOTE: speed shouldn't be affected since most of the array is already sorted
            // and std::sort is really fast in this setting!
            // See: https://solarianprogrammer.com/2012/10/24/cpp-11-sort-benchmark/
            if (new_active_added) {
                active_order.resize(active_set.size());
                std::iota(std::next(active_order.begin(), old_active_size), 
                          active_order.end(), 
                          old_active_size);
                std::sort(active_order.begin(), active_order.end(),
                          [&](auto i, auto j) { 
                                return strong_set[active_set[i]] < strong_set[active_set[j]];
                            });

                active_set_ordered.resize(active_set.size());
                Eigen::Map<util::vec_type<aso_value_t>> aso_map(
                        active_set_ordered.data(),
                        active_set_ordered.size());
                aso_map = util::vec_type<aso_value_t>::NullaryExpr(
                        active_order.size(),
                        [&](auto i) { return strong_set[active_set[active_order[i]]]; });

                active_beta_diff_ordered.resize(active_order.size());
            }

            if (convg_measure < thr) break;
            if (n_cds >= max_cds) throw util::max_cds_error(l);

            lasso_active_and_update(l, lmda);
        }

        // order the strong betas 
        active_beta_ordered.resize(active_order.size());
        Eigen::Map<util::vec_type<beta_value_t>> ab_o_view(
                active_beta_ordered.data(),
                active_beta_ordered.size());
        ab_o_view = util::vec_type<beta_value_t>::NullaryExpr(
                active_order.size(),
                [&](auto i) { return strong_beta[active_set[active_order[i]]]; });
        assert(active_set_ordered.size() == active_order.size());
        Eigen::Map<const sp_vec_t> beta_map(
                A.cols(),
                active_set_ordered.size(),
                active_set_ordered.data(),
                active_beta_ordered.data());

        betas[l] = beta_map;
        rsqs[l] = rsq;
        ++n_lmdas;

        // make sure to do at least 3 lambdas.
        if (l < 2) continue;

        // early stop if R^2 criterion is fulfilled.
        if (check_early_stop_rsq(rsqs[l-2], rsqs[l-1], rsqs[l])) break;
    }
}

} // namespace ghostbasil
