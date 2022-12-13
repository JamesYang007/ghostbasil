#pragma once
#include <array>
#include <numeric>
#include <ghostbasil/util/exceptions.hpp>
#include <ghostbasil/util/functional.hpp>
#include <ghostbasil/util/macros.hpp>
#include <ghostbasil/util/types.hpp>
#include <ghostbasil/util/counting_iterator.hpp>
#include <ghostbasil/util/eigen/map_sparsevector.hpp>
#include <ghostbasil/matrix/forward_decl.hpp>
#include <ghostbasil/optimization/lasso_base.hpp>

namespace ghostbasil {
    
template <class ValueType>
class GroupLasso : public LassoBase
{
    using value_t = ValueType;
    
    util::vec_type<value_t> vec_buffer_ugc_;
    util::vec_type<value_t> vec_buffer_bcd_;

public:
    explicit GroupLasso(
        size_t vec_buffer_ugc_size,
        size_t vec_buffer_bcd_size)
        : vec_buffer_ugc_(vec_buffer_ugc_size),
          vec_buffer_bcd_(vec_buffer_bcd_size)
    {}

    /*
     * Constructs active (feature) indices in increasing order
     * expanding group ranges as a dense vector.
     * The result is stored in out.
     */
    template <class AOType, class ASType, class SSType, 
              class GroupsType, class OutType>
    GHOSTBASIL_STRONG_INLINE
    static void get_active_indices(
        const AOType& active_ordered, 
        const ASType& active_set, 
        const SSType& strong_set, 
        const GroupsType& groups,
        OutType& out
    )
    {
        using out_value_t = typename std::decay_t<OutType>::value_type;
        auto out_begin = out.data();
        for (size_t i = 0; i < active_ordered.size(); ++i) {
            const auto ss_idx = active_set[active_ordered[i]];
            const auto group = strong_set[ss_idx];
            const auto group_size = groups[group+1] - groups[group];
            Eigen::Map<util::vec_type<out_value_t>> seg(
                out_begin, group_size
            );
            seg = util::vec_type<out_value_t>::LinSpaced(
                group_size, groups[group], groups[group+1] - 1
            );
            out_begin += group_size;
        }
        assert(out.size() == std::distance(out.data(), out_begin));
    }

    /*
     * Constructs active (feature) values in increasing index order.
     * The result is stored in out.
     */
    template <class AOType, class ASType, class SSType, 
              class GroupsType, class SBetaType, class SBeginsType,
              class OutType>
    GHOSTBASIL_STRONG_INLINE
    static void get_active_values(
        const AOType& active_ordered, 
        const ASType& active_set, 
        const SSType& strong_set, 
        const GroupsType& groups,
        const SBetaType& strong_beta, 
        const SBeginsType& strong_begins, 
        OutType& out 
    )
    {
        using out_value_t = typename std::decay_t<OutType>::value_type;
        auto out_begin = out.data();
        for (size_t i = 0; i < active_ordered.size(); ++i) {
            const auto ss_idx = active_set[active_ordered[i]];
            const auto group = strong_set[ss_idx];
            const auto group_size = groups[group+1] - groups[group];
            Eigen::Map<util::vec_type<out_value_t>> seg(
                out_begin, group_size
            );
            seg = strong_beta.segment(strong_begins[ss_idx], group_size);
            out_begin += group_size;
        }        
        assert(out.size() == std::distance(out.data(), out_begin));
    }
    
    /*
     * Computes the objective that we wish to minimize.
     * The objective is the quadratic loss + regularization:
     *
     *      (1-s)/2 \sum_{ij} x_i^T A_{ij} x_j - \sum_{i} x_i^T r 
     *          + lmda \sum_i ||x_i||_2 + s/2 \sum_i ||x_i||_2^2
     *          
     * @param   A       any square (n,n) matrix. 
     * @param   r       any vector (n,).
     * @param   groups  vector of indices defining the beginning index of groups.
     * @param   s       PGR regularization.
     * @param   lmda    group-lasso regularization.
     * @param   beta    coefficient vector.
     */
    template <class AType, class RType, class GroupsType,
              class BetaType>
    GHOSTBASIL_STRONG_INLINE 
    static auto objective(
        const AType& A,
        const RType& r,
        const GroupsType& groups,
        value_t s,
        value_t lmda,
        const BetaType& beta)
    {
        value_t penalty = 0.0;
        for (size_t j = 0; j < groups.size()-1; ++j) {
            const auto begin = groups[j];
            const auto end = groups[j+1];
            penalty += beta.segment(begin, end-begin).norm();
        }
        penalty *= lmda;
        return (1-s)/2 * A.quad_form(beta) - beta.dot(r) 
                + s/2 * beta.squaredNorm() + penalty;
    }

    /*
     * Updates the convergence measure using variance of each direction.
     * 
     * @param   convg_measure   convergence measure to update.
     * @param   del             vector difference in a group coefficient.
     * @param   var             vector of variance along each direction of coefficient.
     */
    template <class DelType, class VarType>
    GHOSTBASIL_STRONG_INLINE 
    static void update_convergence_measure(
        value_t& convg_measure, 
        const DelType& del, 
        const VarType& var)
    {
        const auto convg_measure_curr = del.dot(var.cwiseProduct(del)) / del.size();
        convg_measure = std::max(convg_measure, convg_measure_curr);
    }
    
    /*
     * Updates R^2 given the group variance vector, coefficient difference
     * (old minus new), and the current residual correlation vector.
     * 
     * @param   rsq     R^2 to update.
     * @param   del     new coefficient minus old coefficient.
     * @param   var     variance along each coordinate of group.
     * @param   r       current residual correlation vector for group.
     */
    template <class DelType, class CoeffNewType,
              class VarType, class RType>
    GHOSTBASIL_STRONG_INLINE
    static void update_rsq(
        value_t& rsq,
        const DelType& del,
        const CoeffNewType& coeff_new,
        const VarType& var,
        const RType& r,
        value_t s
    )
    {
        const auto sum = 2 * coeff_new - del;
        rsq += ((-2 * r.array() + ((1-s) * var.array() + s) * sum.array()) * del.array()).sum();
    }

    /*
     * Solves the solution for the equation (w.r.t. x):
     *      minimize_x (1-s)/2 x^T L x - x^T v + lmda ||x||_2 + s/2 ||x||_2^2
     *      
     * @param  L       vector representing a diagonal PSD matrix.
     *                 Must have max(L + s) > 0. 
     *                 L.size() <= vec_buffer_ugc_.size().
     * @param  v       any vector.  
     * @param  lmda    group-lasso regularization value. Must be >= 0.
     * @param  s       PGR regularization value. Must be in [0,1].
     * @param  tol     Newton's method tolerance of closeness to 0.
     * @param  max_iters   maximum number of iterations of Newton's method.
     * @param  x           solution vector.
     * @param  iters       number of Newton's method iterations.
     */
    template <class LType, class VType, class XType>
    GHOSTBASIL_STRONG_INLINE
    void update_group_coefficients(
        const LType& L,
        const VType& v,
        value_t lmda,
        value_t s,
        value_t tol,
        size_t max_iters,
        XType& x,
        size_t& iters
    )
    {
        iters = 0;

        // Easy case: ||v||_2 <= lmda -> x = 0
        const auto v_l2 = v.norm();
        if (v_l2 <= lmda) {
            x.setZero();
            return;
        }
        
        if (v.size() == 1) {
            assert(L.size() == 1);
            assert(x.size() == 1);
            x[0] = std::copysign(1, v[0]) * (std::abs(v[0]) - lmda) / ((1-s) * L[0] + s);
            return;
        }
        
        // Difficult case: ||v||_2 > lmda

        // First solve for h := ||x||_2
        auto buffer = vec_buffer_ugc_.head(L.size());

        // Find good initialization
        // The following helps tremendously if the resulting h > 0.
        buffer.array() = ((1-s) * L.array() + s);
        const value_t b = lmda * buffer.sum();
        const value_t a = buffer.array().square().sum();
        const value_t v_l1 = v.array().abs().sum();
        const value_t c = lmda * lmda * L.size() - v_l1 * v_l1;
        const value_t zero = 0.0;
        const value_t discr = b*b - a*c;
        auto h = (discr > -1e-12) ? 
            (-b + std::sqrt(std::max(discr, zero))) / a : 0.0;
        
        // Otherwise, if h <= 0, we know at least 0 is a reasonable solution.
        // The only case h <= 0 is when 0 is already close to the solution.
        h = std::max(h, zero);

        // Newton initialization
        // We use x as a buffer.
        x.array() = (v.array() / (buffer.array() * h + lmda)).square();
        auto fh = x.sum() - 1;
        auto dfh = -2 * (
            x.array() * (buffer.array() / (buffer.array() * h + lmda))
        ).sum();

        while (std::abs(fh) > tol && iters < max_iters) {
            // Newton update 
            h -= fh / dfh;
            
            // update next states
            x.array() = (v.array() / (buffer.array() * h + lmda)).square();
            fh = x.sum() - 1;
            dfh = -2 * (
                x.array() * (buffer.array() / (buffer.array() * h + lmda))
            ).sum();
            ++iters;
        }
        
        // numerical stability
        h = std::max(h, 1e-14);

        // final solution
        x.array() = v.array() / (buffer.array() + lmda / h);
    }

    /*
     * One Blockwise-Coordinate-Descent loop to solve the objective.
     *  
     * @param   begin           begin iterator to indices into strong set, i.e.
     *                          strong_set[*begin] is the current group to descend.
     *                          The resulting sequence of indices from calling 
     *                          strong_set[*(begin++)] MUST be ordered.
     * @param   end             end iterator to indices into strong set.
     * @param   strong_set      strong set of indices of groups.
     * @param   strong_begins   beginning index into strong_set corresponding values.
     * @param   strong_A_diag   diagonal of A corresponding to groups in strong_set.
     * @param   A       PSD matrix with 
     *                  A[groups[i]:groups[i+1], groups[i]:groups[i+1]] diagonal.
     * @param   r       any vector.
     * @param   groups  vector of indices where groups[i] defines the beginning index of group i
     *                  so that [groups[i], groups[i+1]) is the range of indices belonging to group i.
     *                  MUST have that vec_buffer_bcd_.size() >= largest group size.
     * @param   lmda    group-lasso regularization value. Must be >= 0.
     * @param   s       PGR regularization value. Must be >= 0.
     * @param   tol     see update_group_coefficients__.
     * @param   max_iters   see update_group_coefficients__.
     * @param   strong_beta     coefficients corresponding to strong_set.
     *                          strong_beta[
     *                              strong_begins[i] : strong_begins[i] + groups[strong_set[i]+1]-groups[strong_set[i]]
     *                          ] is the beta coefficient for group strong_set[i].
     * @param   strong_grad     (negative) gradient corresponding to strong_beta.
     *                          strong_grad[
     *                              strong_begins[i] : strong_begins[i] + groups[strong_set[i]+1]-groups[strong_set[i]]
     *                          ] is the gradient for group strong_set[i].
     * @param   convg_measure   stores the convergence measure of the call.
     * @param   rsq             R^2 of current beta. Stores the updated value after the call.
     */
    template <class Iter, class StrongSetType, class StrongBeginsType,
              class StrongADiagType, class AType, class GroupsType,
              class StrongBetaType, class StrongGradType,
              class AdditionalStepType=util::no_op>
    GHOSTBASIL_STRONG_INLINE
    void blockwise_coordinate_descent(
        Iter begin,
        Iter end,
        const StrongSetType& strong_set,
        const StrongBeginsType& strong_begins,
        const StrongADiagType& strong_A_diag,
        const AType& A,
        const GroupsType& groups,
        value_t lmda,
        value_t s,
        value_t tol,
        size_t max_iters,
        StrongBetaType& strong_beta,
        StrongGradType& strong_grad,
        value_t& convg_measure,
        value_t& rsq,
        AdditionalStepType additional_step=AdditionalStepType()
    )
    {
        Eigen::Map<const util::vec_type<value_t>> A_diag_map(
            strong_A_diag.data(), strong_A_diag.size()
        );
        Eigen::Map<util::vec_type<value_t>> sb_map(
            strong_beta.data(), strong_beta.size()
        );
        Eigen::Map<util::vec_type<value_t>> sg_map(
            strong_grad.data(), strong_grad.size()
        );
        const auto sc = 1-s;

        convg_measure = 0;
        for (auto it = begin; it != end; ++it) {
            const auto ss_idx = *it;              // index to strong set
            const auto k = strong_set[ss_idx];    // actual group index
            const auto ss_value_begin = strong_begins[ss_idx]; // value begin index at ss_idx
            const auto gsize = groups[k+1] - groups[k]; // group size  
            auto ak = sb_map.segment(ss_value_begin, gsize); // corresponding beta
            auto ak_old = vec_buffer_bcd_.head(ak.size());
            ak_old = ak; // save old beta in buffer
            auto gk = sg_map.segment(ss_value_begin, gsize); // corresponding gradient
            const auto A_kk = A_diag_map.segment(ss_value_begin, gsize);  // corresponding A diagonal 
                                        
            // update group coefficients
            size_t iters;
            update_group_coefficients(
                A_kk, gk, lmda, s, tol, max_iters, ak, iters
            );
            if (iters >= max_iters) {
                throw util::group_lasso_max_newton_iters();
            }

            if ((ak_old.array() == ak.array()).all()) continue;

            // use same buffer as ak_old to store difference
            auto& del = ak_old;
            del = ak - ak_old;

            // update measure of convergence
            update_convergence_measure(convg_measure, del, A_kk);

            // update rsq
            update_rsq(rsq, del, ak, A_kk, gk, s);

            // update gradient-like quantity
            // we split the for-loop to encourage compiler vectorization.
            del.array() *= sc;
            auto loop_body = [&](auto jt) {
                const auto ss_idx_j = *jt;
                const auto j = strong_set[ss_idx_j];
                const auto groupj_size = groups[j+1] - groups[j];
                const auto A_jk = A.block(
                    groups[j], groups[k], 
                    groupj_size, gsize
                );
                auto sg_idx_j = sg_map.segment(
                    strong_begins[ss_idx_j], groupj_size
                );
                auto buffer = vec_buffer_ugc_.head(groupj_size);
                buffer.noalias() = A_jk * del;
                sg_idx_j -= buffer;
            };
            auto jt = begin;
            for (; jt != it; ++jt) { loop_body(jt); }
            ++jt;
            for (; jt != end; ++jt) { loop_body(jt); }
            
            // additional step
            additional_step(ss_idx);
        }
    }

    /*
     * Applies multiple blockwise coordinate descent on the active set 
     * to minimize the ghostbasil objective with group-lasso penalty.
     * See "objective" function for the objective of interest.
     *
     * @param   A           PSD matrix (p, p) with diagonal blocks A_{ii}. 
     * @param   s           regularization of A towards identity. 
     *                      It is undefined behavior is s is not in [0,1].
     * @param   strong_set  strong set as a dense vector of indices in [0, I),
     *                      where I is the total number of groups.
     *                      strong_set[i] = ith strong group.
     * @param   strong_begins   vector of indices that define the beginning index to values
     *                          corresponding the strong groups.
     *                          MUST have strong_begins.size() == strong_set.size().
     * @param   strong_order    order of strong_set that results in sorted (ascending) values.
     *                          strong_set[strong_order[i]] < strong_set[strong_order[j]] if i < j.
     * @param   active_set  active set as a dense vector of indices in [0, strong_set.size()).
     *                      strong_set[active_set[i]] = ith active group.
     *                      This set must at least contain ALL indices into strong_set 
     *                      where the corresponding strong_beta is non-zero, that is,
     *                      if strong_beta[
     *                          strong_begins[i] : strong_begins[i] + groups[strong_set[i+1]] - groups[strong_set[i]]
     *                      ] != 0, then i is in active_set.
     * @param   active_order    order of active_set that results in sorted (ascending) 
     *                          values of strong_set.
     *                          strong_set[active_set[active_order[i]]] < 
     *                              strong_set[active_set[active_order[j]]] if i < j.
     * @param   active_set_ordered  ordered *groups corresponding to* active_set.
     *                              active_set_ordered[i] == strong_set[active_set[active_order[i]]].
     * @param   is_active   dense vector of bool of size strong_set.size(). 
     *                      is_active[i] = true if group strong_set[i] is active.
     *                      active_set should contain i.
     * @param   strong_A_diag       dense vector representing diagonal of A restricted to strong_set.
     *                              strong_A_diag[
     *                                  strong_begins[i] : strong_begins[i] + groups[strong_set[i+1]] - groups[strong_set[i]]
     *                              ] = (block matrix) A_{kk} where k = strong_set[i].
     * @param   lmda_idx    index into the lambda sequence for logging purposes.
     * @param   lmda        L1 regularization of loss.
     * @param   max_cds     max number of coordinate descents.
     * @param   thr         convergence threshold.
     * @param   strong_beta dense vector of coefficients of size strong_A_diag.size().
     *                      strong_beta[b_j:(b_j+p_i)] = coefficient for group i,
     *                      where i := strong_set[j], b_j := strong_begins[j], and p_i = size of group i.
     *                      The updated coefficients will be stored here.
     * @param   strong_grad dense vector of residuals.
     *                      strong_grad[b_j:(b_j+p_i)] = residuals for group i,
     *                      where i := strong_set[j], b_j := strong_begins[j], and p_i = size of group i.
     *                      The updated gradients will be stored here.
     * @param   active_beta_diff_ordered    dense vector to store coefficient difference corresponding
     *                                      to the active set between
     *                                      the new coefficients and the current coefficients
     *                                      after performing coordinate descent on the active set.
     *                                      It must be initialized to be of size active_order.size(),
     *                                      though the values need not be initialized.
     * @param   rsq         unnormalized R^2 estimate.
     *                      It is only well-defined if it is (approximately) 
     *                      the R^2 that corresponds to beta.
     *                      The updated R^2 estimate will be stored here.
     * @param   n_cds       stores the number of coordinate descents from this call.
     * @param   sg_update   functor that updates strong_gradient.
     */
    template <class AType, class GroupsType, class SSType, class SBeginsType, //class SOType, 
              class ASType, class ABType, 
              //class AOType, class ASOType,
              class IAType, class StrongADiagType, 
              class SBType, class SGType, 
              //class ABDiffOType, 
              class ABDiffType, 
              //class SGUpdateType,
              class CUIType = util::no_op>
    GHOSTBASIL_STRONG_INLINE
    void group_lasso_active(
        const AType& A,
        const GroupsType& groups,
        value_t s,
        const SSType& strong_set,
        const SBeginsType& strong_begins,
        //const SOType& strong_order,
        const ASType& active_set,
        const ABType& active_begins,
        //const AOType& active_order,
        //const ASOType& active_set_ordered,
        const IAType& is_active,
        const StrongADiagType& strong_A_diag,
        size_t lmda_idx,
        value_t lmda,
        size_t max_cds,
        value_t thr,
        value_t newton_tol,
        size_t newton_max_iters,
        SBType& strong_beta,
        SGType& strong_grad,
        //ABDiffOType& active_beta_diff_ordered,
        ABDiffType& active_beta_diff,
        value_t& rsq,
        size_t& n_cds,
        //SGUpdateType sg_update,
        CUIType check_user_interrupt = CUIType())
    {
        //using ao_value_t = typename std::decay_t<AOType>::value_type;
        //using aso_value_t= typename std::decay_t<ASOType>::value_type;
        //using as_value_t= typename std::decay_t<ASType>::value_type;

        Eigen::Map<const util::vec_type<value_t>> A_diag_map(
            strong_A_diag.data(), strong_A_diag.size()
        );
        Eigen::Map<util::vec_type<value_t>> sb_map(
            strong_beta.data(), strong_beta.size()
        );
        Eigen::Map<util::vec_type<value_t>> sg_map(
            strong_grad.data(), strong_grad.size()
        );

        // TODO: write some similar assert checks
        //internal::lasso_assert_valid_inputs(
        //        A, s, strong_set, strong_order, 
        //        active_set, active_order, active_set_ordered,
        //        is_active, strong_A_diag,
        //        strong_beta, strong_grad);

        Eigen::Map<util::vec_type<value_t>> ab_diff_view(
            active_beta_diff.data(), active_beta_diff.size()
        );
        
        // save old active beta
        for (size_t i = 0; i < active_set.size(); ++i) {
            const auto ss_idx_group = active_set[i];
            const auto ss_group = strong_set[ss_idx_group];
            const auto ss_group_size = groups[ss_group + 1] - groups[ss_group];
            const auto sb_begin = strong_begins[ss_idx_group];
            const auto sb = sb_map.segment(sb_begin, ss_group_size);
            const auto ab_begin = active_begins[i];
            ab_diff_view.segment(ab_begin, ss_group_size) = sb;
        }

        while (1) {
            check_user_interrupt(n_cds);
            ++n_cds;
            value_t convg_measure;
            blockwise_coordinate_descent(
                active_set.begin(),
                active_set.end(),
                strong_set, strong_begins, strong_A_diag, 
                A, groups, lmda, s, newton_tol, newton_max_iters,
                strong_beta, strong_grad,
                convg_measure, rsq
            );
            if (convg_measure < thr) break;
            if (n_cds >= max_cds) throw util::max_cds_error(lmda_idx);
        }
        
        // compute new active beta - old active beta
        for (size_t i = 0; i < active_set.size(); ++i) {
            const auto ss_idx_group = active_set[i];
            const auto ss_group = strong_set[ss_idx_group];
            const auto ss_group_size = groups[ss_group + 1] - groups[ss_group];
            const auto sb_begin = strong_begins[ss_idx_group];
            const auto sb = sb_map.segment(sb_begin, ss_group_size);
            const auto ab_begin = active_begins[i];
            auto ab_diff_view_curr = ab_diff_view.segment(ab_begin, ss_group_size);
            ab_diff_view_curr = sb - ab_diff_view_curr;
        }

        // update strong gradient for non-active strong variables
        const auto sc = 1-s;
        ab_diff_view.array() *= sc;
        for (size_t i_idx = 0; i_idx < active_set.size(); ++i_idx) {
            const auto ss_idx_i = active_set[i_idx];
            const auto i = strong_set[ss_idx_i];
            const auto groupi_size = groups[i+1] - groups[i];
            const auto ab_begin = active_begins[i_idx];
            const auto ab_diff_view_curr = ab_diff_view.segment(ab_begin, groupi_size);

            for (size_t j_idx = 0; j_idx < strong_set.size(); ++j_idx) {
                if (is_active[j_idx]) continue;
                const auto j = strong_set[j_idx];
                const auto groupj_size = groups[j+1] - groups[j];
                const auto A_ji = A.block(
                    groups[j], groups[i], 
                    groupj_size, groupi_size
                );
                auto sg_j = sg_map.segment(
                    strong_begins[j_idx], groupj_size
                );
                auto buffer = vec_buffer_bcd_.head(groupj_size);
                buffer.noalias() = A_ji * ab_diff_view_curr;
                sg_j -= buffer;
            }
        }
    }

    /*
     * Minimizes the objective described in the function objective()
     * with the additional constraint:
     * \[
     *      \x_{i} = 0, \, \forall i \notin \{strong_set\}
     * \]
     * i.e. all betas not in strong_set are fixed to be 0.
     *
     * See group_lasso_active() for descriptions of common parameters.
     *
     * @param   lmdas       regularization parameter lambda sequence.
     * @param   betas       vector of length lmdas.size() of 
     *                      output coefficient sparse vector (p,).
     *                      betas[j](i) = ith coefficient for jth lambda.
     * @param   rsqs        vector of length lmdas.size() 
     *                      to store R^2 values for each lambda in lmdas.
     */
    template <class AType, class GroupsType,
              class SSType, class SBeginsType, //class SOType,
              class SADType, class LmdasType,
              class SBType, class SGType,
              class ASType, class ABType, class AOType, //class ASOType, 
              class IAType, class BetasType, class RsqsType,
              class CUIType = util::no_op>
    void group_lasso(
        const AType& A,
        const GroupsType& groups, 
        value_t s, 
        const SSType& strong_set, 
        const SBeginsType& strong_begins, 
        //const SOType& strong_order,
        const SADType& strong_A_diag,
        const LmdasType& lmdas, 
        size_t max_cds,
        value_t thr,
        value_t newton_tol,
        size_t newton_max_iters,
        value_t rsq,
        SBType& strong_beta, 
        SGType& strong_grad,
        ASType& active_set,
        ABType& active_begins,
        AOType& active_order,
        //ASOType& active_set_ordered,
        IAType& is_active,
        BetasType& betas, 
        RsqsType& rsqs,
        size_t& n_cds,
        size_t& n_lmdas,
        CUIType check_user_interrupt = CUIType())
    {
        //internal::lasso_assert_valid_inputs(
        //        A, s, strong_set, strong_order, 
        //        active_set, active_order, active_set_ordered, 
        //        is_active, strong_A_diag,
        //        strong_beta, strong_grad);

        assert(betas.size() == lmdas.size());
        assert(rsqs.size() == lmdas.size());

        using as_value_t = typename std::decay_t<ASType>::value_type;
        using beta_value_t = typename std::decay_t<SBType>::value_type;
        using sp_vec_t = util::sp_vec_type<beta_value_t, Eigen::ColMajor, as_value_t>;

        // buffers for the routine
        
        // buffer to store final result
        std::vector<as_value_t> active_beta_indices;
        std::vector<beta_value_t> active_beta_ordered;
        
        // buffer for internal routine group_lasso_active
        std::vector<beta_value_t> active_beta_diff;
        
        // compute size of active_beta_diff (needs to be exact)
        size_t active_beta_size = 0;
        if (active_set.size()) {
            const auto last_idx = active_set.size()-1;
            const auto last_group = strong_set[active_set[last_idx]];
            const auto group_size = groups[last_group+1] - groups[last_group];
            active_beta_size = active_begins[last_idx] + group_size;
        }

        // allocate buffers with optimization
        active_beta_indices.reserve(strong_beta.size());
        active_beta_ordered.reserve(strong_beta.size());
        active_beta_diff.reserve(strong_beta.size());
        active_beta_diff.resize(active_beta_size);
        
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
            group_lasso_active(
                    A, groups, s, strong_set, strong_begins, //strong_order, 
                    active_set, active_begins, //active_order, active_set_ordered,
                    is_active, strong_A_diag,
                    l, lmda, max_cds, thr, newton_tol, newton_max_iters,
                    strong_beta, strong_grad, 
                    active_beta_diff, rsq, n_cds,
                    check_user_interrupt);
            lasso_active_called = true;
        };

        for (size_t l = 0; l < lmdas.size(); ++l) {
            const auto lmda = lmdas[l];

            if (lasso_active_called) {
                lasso_active_and_update(l, lmda);
            }

            while (1) {
                check_user_interrupt(n_cds);
                ++n_cds;
                value_t convg_measure;
                const auto old_active_size = active_set.size();
                blockwise_coordinate_descent(
                        util::counting_iterator<size_t>(0),
                        util::counting_iterator<size_t>(strong_set.size()),
                        strong_set, strong_begins, strong_A_diag, 
                        A, groups, lmda, s, newton_tol, newton_max_iters,
                        strong_beta, strong_grad,
                        convg_measure, rsq, add_active_set);
                const bool new_active_added = (old_active_size < active_set.size());

                if (new_active_added) {
                    // update active_begins
                    auto new_abd_size = active_beta_diff.size();
                    active_begins.resize(active_set.size());
                    for (size_t i = old_active_size; i < active_begins.size(); ++i) {
                        active_begins[i] = new_abd_size;
                        const auto curr_group = strong_set[active_set[i]];
                        const auto curr_size = groups[curr_group+1] - groups[curr_group];
                        new_abd_size += curr_size;
                    }

                    // update active_beta_diff size
                    active_beta_diff.resize(new_abd_size);
                }

                if (convg_measure < thr) break;
                if (n_cds >= max_cds) throw util::max_cds_error(l);

                lasso_active_and_update(l, lmda);
            }

            // update active_order
            const auto old_active_size = active_order.size();
            active_order.resize(active_set.size());
            std::iota(std::next(active_order.begin(), old_active_size), 
                      active_order.end(), 
                      old_active_size);
            std::sort(active_order.begin(), active_order.end(),
                      [&](auto i, auto j) { 
                            return strong_set[active_set[i]] < strong_set[active_set[j]];
                        });
            
            // update active_beta_indices
            active_beta_indices.resize(active_beta_diff.size());
            get_active_indices(
                active_order, active_set, strong_set, groups,
                active_beta_indices
            );
            
            // update active_beta_ordered
            active_beta_ordered.resize(active_beta_diff.size());
            get_active_values(
                active_order, active_set, strong_set, groups,
                strong_beta, strong_begins, 
                active_beta_ordered
            );

            // order the strong betas 
            Eigen::Map<const sp_vec_t> beta_map(
                    A.cols(),
                    active_beta_indices.size(),
                    active_beta_indices.data(),
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

}; // class GroupLasso
    
} // namespace ghostbasil