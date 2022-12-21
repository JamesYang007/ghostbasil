#pragma once
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <ghostbasil/util/algorithm.hpp>
#include <ghostbasil/util/exceptions.hpp>
#include <ghostbasil/util/macros.hpp>
#include <ghostbasil/util/types.hpp>
#include <ghostbasil/optimization/lasso.hpp>
#include <atomic>
#include <cmath>
#include <unordered_set>
#include <vector>

namespace ghostbasil {
namespace lasso {

/*
 * Computes and stores the next sequence of lambdas.
 * Given the max absolute gradient max_abs_grad,
 * it will be the first value in the sequence
 * and the subsequent values decrease down by factor.
 * If lmdas is empty, nothing occurs.
 */
template <class ValueType, class LmdasType>
GHOSTBASIL_STRONG_INLINE
void next_lambdas(
        ValueType max_abs_grad,
        ValueType factor,
        LmdasType& lmdas)
{
    if (lmdas.size() == 0) return;
    lmdas[0] = max_abs_grad;
    for (size_t i = 1; i < lmdas.size(); ++i) {
        lmdas[i] = lmdas[i-1] * factor;
    }
}

/*
 * Initializes strong set. It adds n_add number of variables 
 * that are not in the strong set already (is_strong(i) == false) 
 * and achieve the highest absolute gradient values
 * into strong_set and strong_hashset.
 * strong_set is sorted to fit the invariant.
 */
template <class SSType, class SHType, class AbsGradType,
          class ISType>
GHOSTBASIL_STRONG_INLINE
void init_strong_set(
        SSType& strong_set, 
        SHType& strong_hashset,
        const AbsGradType& abs_grad, 
        const ISType& is_strong,
        size_t n_add,
        size_t capacity)
{
    strong_set.reserve(capacity); 
    screen(abs_grad, is_strong, n_add, strong_set);
    strong_hashset.insert(strong_set.begin(), strong_set.end());
    std::sort(strong_set.begin(), strong_set.end());
}

template <class SOType, class SSType>
GHOSTBASIL_STRONG_INLINE
void init_strong_order(
        SOType& strong_order,
        const SSType& strong_set,
        size_t old_strong_set_size,
        size_t capacity=0)
{
    strong_order.reserve(capacity);
    strong_order.resize(strong_set.size());
    std::iota(std::next(strong_order.begin(), old_strong_set_size), 
              strong_order.end(), 
              old_strong_set_size);
    std::sort(strong_order.begin(), strong_order.end(),
              [&](auto i, auto j) { return strong_set[i] < strong_set[j]; });
}

/*
 * Initializes the lambda sequence.
 * lmdas will be resized to size number of elements.
 * If user_lmdas is empty, it will generate a sequence by calling next_lambdas.
 * Otherwise, it will copy size number of elements from the beginning of user_lmdas.
 */
template <class LmdasType, class UserLmdasType, class SSType,
          class SGType, class ValueType>
void init_lambdas(
        LmdasType& lmdas,
        const UserLmdasType& user_lmdas,
        const SSType& strong_set,
        const SGType& strong_grad,
        size_t size,
        ValueType factor)
{
    using value_t = typename std::decay_t<SGType>::value_type;
    lmdas.resize(size);
    if (lmdas.size() == 0) return;
    if (user_lmdas.size() == 0) {
        // if no user-specific lambdas, find max abs grad 
        // and construct a sequence downwards by factor.
        Eigen::Map<const util::vec_type<value_t>> sg_map(
                strong_grad.data(),
                strong_grad.size());
        value_t max_abs_grad = sg_map.array().abs().maxCoeff();
        next_lambdas(max_abs_grad, factor, lmdas);
    } else {
        // if user-specific lambdas, copy a chunk.
        std::copy(user_lmdas.data(), 
                  std::next(user_lmdas.data(), lmdas.size()),
                  lmdas.data());
    }
}

/*
 * Initializes strong_grad. It must be of the same size as strong_set.
 * strong_grad[i] = gradient of strong_set[i] feature.
 * In the beginning, gradient is simply the correlation value.
 */
template <class SGType, class SSType, class GradType>
GHOSTBASIL_STRONG_INLINE
void init_strong_grad(
        SGType& strong_grad,
        const SSType& strong_set,
        const GradType& grad,
        size_t capacity)
{
    strong_grad.reserve(capacity);
    strong_grad.resize(strong_set.size());
    for (size_t i = 0; i < strong_set.size(); ++i) {
        strong_grad[i] = grad[strong_set[i]];
    }
}

/*
 * Initializes strong_A_diag with the diagonal of A on strong set.
 * strong_A_diag[i] = diagonal of A at index strong_set[i].
 */
template <class SADType, class AType, class SSType>
GHOSTBASIL_STRONG_INLINE
void init_strong_A_diag(
        SADType& strong_A_diag, 
        const AType& A, 
        const SSType& strong_set,
        size_t begin,
        size_t end,
        size_t capacity=0)
{
    assert((begin <= end) && (end <= strong_set.size()));

    // subsequent calls does not affect capacity
    strong_A_diag.reserve(capacity);
    strong_A_diag.resize(strong_set.size());

    for (size_t i = begin; i < end; ++i) {
        auto k = strong_set[i];
        strong_A_diag[i] = A.coeff(k, k);
    }
}

/*
 * Initializes active set. There is nothing to do except
 * optimize for initial capacity.
 */
template <class ASType>
GHOSTBASIL_STRONG_INLINE
void init_active_set(ASType& active_set, size_t capacity)
{
    active_set.reserve(capacity); 
}

/*
 * Initializes is_active. It must be the same size as strong_set.
 * is_active[i] = true if strong_set[i] is active.
 * In the beginning, no variables are active.
 */
template <class IAType, class SSType>
GHOSTBASIL_STRONG_INLINE
void init_is_active(
        IAType& is_active, 
        const SSType& strong_set, 
        size_t capacity)
{
    is_active.reserve(capacity);
    is_active.resize(strong_set.size(), false);
}

/*
 * Initialize coefficients on strong set.
 * In the beginning, is the 0 vector.
 */
template <class SBType, class SSType>
GHOSTBASIL_STRONG_INLINE
void init_strong_beta(
        SBType& strong_beta, 
        const SSType& strong_set,
        size_t capacity)
{
    strong_beta.reserve(capacity);
    strong_beta.resize(strong_set.size(), 0);
}

/*
 * Checks the KKT condition, which is that
 * 
 *      |\nabla_k f| \leq \lambda \quad \forall k
 *
 * The KKT condition is checked for a sequence of lambdas.
 *
 * @param   A       covariance matrix.
 * @param   r       correlation vector.
 * @param   s       regularization parameter of A towards identity.
 * @param   lmdas   downward sequence of L1 regularization parameter.
 * @param   betas   each element i corresponds to a (sparse) vector
 *                  of the solution at lmdas[i].
 * @param   is_strong   a functor that checks if feature i is strong.
 * @param   n_threads   number of threads to use in OpenMP.
 * @param   grad        a dense vector that represents the (negative) gradient
 *                      right before the first value in lmdas (going downwards)
 *                      that fails the KKT check.
 *                      If KKT fails at the first lambda, grad is unchanged.
 * @param   grad_next   a dense vector that represents the (negative) gradient
 *                      right at the first value in lmdas (going downwards)
 *                      that fails the KKT check.
 *                      This is really used for optimizing memory allocation.
 *                      User should not need to access this directly.
 *                      It is undefined-behavior accessing this after the call.
 *                      It just has to be initialized to the same size as grad.
 */
template <class AType, class RType, class ValueType, 
          class LmdasType, class BetasType, class ISType, class GradType>
GHOSTBASIL_STRONG_INLINE 
auto check_kkt(
    const AType& A, 
    const RType& r,
    ValueType s,
    const LmdasType& lmdas, 
    const BetasType& betas,
    const ISType& is_strong,
    size_t n_threads,
    GradType& grad,
    GradType& grad_next)
{
    assert(r.size() == grad.size());
    assert(grad.size() == grad_next.size());
    assert(betas.size() == lmdas.size());

    size_t i = 0;
    auto sc = 1-s;

    if (lmdas.size() == 0) return i;

    for (; i < lmdas.size(); ++i) {
        const auto& beta_i = betas[i];
        auto lmda = lmdas[i];

        std::atomic<bool> kkt_fail(false);

#pragma omp parallel for schedule(static) num_threads(n_threads)
        for (size_t k = 0; k < r.size(); ++k) {
            // Just omit the KKT check for strong variables.
            // If KKT failed previously, just do a no-op until loop finishes.
            // This is because OpenMP doesn't allow break statements.
            bool kkt_fail_raw = kkt_fail.load(std::memory_order_relaxed);
            if (kkt_fail_raw) continue;

            // we still need to save the gradients including strong variables
            auto gk = -sc * A.col_dot(k, beta_i) + r[k] - s * beta_i.coeff(k);
            grad_next[k] = gk;

            if (is_strong(k) || 
                (std::abs(gk) < lmda /* TODO: numerical prec window? */)) continue;
            
            kkt_fail.store(true, std::memory_order_relaxed);
        }

        if (kkt_fail.load(std::memory_order_relaxed)) break; 
        else grad.swap(grad_next);
    }

    return i;
}

/*
 * Append at most max_size number of (first) elements
 * that achieve the maximum absolute value in out.
 * If there are at least max_size number of such elements,
 * exactly max_size will be added.
 * Otherwise, all such elements will be added, but no more.
 */
template <class AbsGradType, class ISType, class SSType>
GHOSTBASIL_STRONG_INLINE 
void screen(
        const AbsGradType& abs_grad,
        const ISType& is_strong,
        size_t size,
        SSType& strong_set)
{
    assert(strong_set.size() <= abs_grad.size());

    size_t rem_size = abs_grad.size() - strong_set.size();
    size_t size_capped = std::min(size, rem_size);
    size_t old_strong_size = strong_set.size();
    strong_set.insert(strong_set.end(), size_capped, 0);
    util::k_imax(abs_grad, is_strong, size_capped, 
            std::next(strong_set.begin(), old_strong_size));
}

/*
 * Solves the following optimization problem:
 * \[
 *      \min\limits_{\beta} f(\beta) + \lambda ||\beta||_1 \\
 *      f(\beta) := 
 *          \frac{(1-s)}{2} \beta^\top A \beta
 *          - \beta^\top r
 *          + \frac{s}{2} \beta^\top \beta
 * \]
 * for a sequence of $\lambda$ values.
 *
 * @param   A   covariance matrix.
 * @param   r   covariance between covariates and response.
 * @param   s   regularization of A towards identity.
 * @param   user_lmdas      user provided lambda sequence.
 *                          Assumes it is in decreasing order.
 *                          If empty, then lambda sequence will be generated.
 * @param   max_n_lambdas   max number of lambdas to compute solutions for.
 *                          If user_lmdas is non-empty, it will be internally
 *                          reset to user_lmdas.size().
 *                          Assumes it is > 0.
 * @param   n_lambdas_iter  number of lambdas per BASIL iteration for fitting lasso on strong set.
 *                          Internally, it is capped at max_n_lambdas.
 *                          Assumes it is > 0.
 * @param   strong_size     initial strong set size.
 *                          Internally, it is capped at max_strong_size.
 *                          Assumes it is > 0.
 * @param   delta_strong_size   number of variables to add to strong set 
 *                              at every BASIL iteration.
 *                              Internally, it is capped at number of non-strong variables
 *                              at every BASIL iteration.
 *                              Assumes it is > 0.
 * @param   max_strong_size     max number of strong set variables.
 *                              Internally, it is capped at number of features.
 *                              Assumes it is > 0.
 * @param   max_n_cds           maximum number of coordinate descent per BASIL iteration.
 * @param   thr                 convergence threshold for coordinate descent.
 * @param   betas               vector of sparse vectors to store a list of solutions.
 * @param   lmdas               vector of values to store a list of lambdas
 *                              corresponding to the solutions in betas:
 *                              lmdas[i] is a lambda corresponding to the solution
 *                              at betas[i].
 * @param   rsqs                vector of values to store the list of (unnormalized) R^2 values.
 *                              rsqs[i] is the R^2 at lmdas[i] and betas[i].
 *
 * TODO:
 * The algorithm stops at a $\lambda$ value where the 
 * pseudo-validation loss stops decreasing.
 * For now, we return a vector of vector of lambdas, 
 * and vector of coefficient (sparse) matrices 
 * corresponding to each vector of lambdas.
 */

template <class AType, class RType, class ValueType,
          class ULmdasType,
          class BetasType, class LmdasType, class RsqsType,
          class CUIType = util::no_op>
inline void basil(
        const AType& A,
        const RType& r,
        ValueType s,
        const ULmdasType& user_lmdas,
        size_t max_n_lambdas,
        size_t n_lambdas_iter,
        size_t strong_size,
        size_t delta_strong_size,
        size_t max_strong_size,
        size_t max_n_cds,
        ValueType thr,
        ValueType min_ratio,
        size_t n_threads,
        BetasType& betas,
        LmdasType& lmdas,
        RsqsType& rsqs,
        CUIType check_user_interrupt = CUIType())
{
    using value_t = ValueType;
    using index_t = int32_t;
    using bool_t = index_t;
    using vec_t = util::vec_type<value_t>;
    using sp_vec_t = util::sp_vec_type<value_t, Eigen::ColMajor, index_t>;

    const size_t n_features = r.size();
    const size_t initial_size = std::min(n_features, 1uL << 20);

    // input cleaning
    const bool use_user_lmdas = user_lmdas.size() != 0;
    value_t factor = 0; // used only if user lambdas is empty
    if (!use_user_lmdas) {
        factor = std::pow(min_ratio, static_cast<value_t>(1.0)/(max_n_lambdas-1));
    } else {
        max_n_lambdas = user_lmdas.size();
    }
    n_lambdas_iter = std::min(n_lambdas_iter, max_n_lambdas);
    size_t n_lambdas_rem = max_n_lambdas;
    max_strong_size = std::min(max_strong_size, n_features);
    strong_size = std::min(strong_size, max_strong_size);

    // (negative) gradient: -(1-s)/2 A[k,:]^T * beta + r - s/2 * beta[k]
    vec_t grad = r; 
    vec_t grad_next(grad.size()); // just a common buffer to optimize alloc

    // used to determine if a feature is in strong set
    std::unordered_set<index_t> strong_hashset;
    const auto is_strong = [&](auto i) { 
        return strong_hashset.find(i) != strong_hashset.end(); 
    };

    // strong set
    std::vector<index_t> strong_set; 

    // initialize strong_set, strong_hashset based on current absolute gradient
    init_strong_set(strong_set, strong_hashset, grad.array().abs(), 
            is_strong, strong_size, initial_size);

    // strong set order
    std::vector<index_t> strong_order;
    init_strong_order(strong_order, strong_set, 0, initial_size);

    // (negative) gradient only on strong set variables.
    std::vector<value_t> strong_grad;
    init_strong_grad(strong_grad, strong_set, grad, initial_size);

    // diagonal of A only on strong set variables.
    std::vector<value_t> strong_A_diag;
    init_strong_A_diag(strong_A_diag, A, strong_set, 0, strong_set.size(), initial_size);

    // active set of indices corresponding to strong variables that are active.
    // invariant: 0 <= active_set.size() <= strong_set.size(), 
    // active_set.size() is exactly the number of true entries in is_active
    // and contains exactly the indices to is_active that are true.
    std::vector<index_t> active_set;
    std::vector<index_t> active_order;
    std::vector<index_t> active_set_ordered;
    init_active_set(active_set, initial_size);
    init_active_set(active_order, initial_size);
    init_active_set(active_set_ordered, initial_size);

    // map of indices corresponding to strong variables that indicate if they are active or not.
    // invariant: is_active.size() == strong_set.size().
    std::vector<bool_t> is_active;
    init_is_active(is_active, strong_set, initial_size);

    // coefficients for strong set (initialized to 0)
    // invariant: strong_beta.size() == strong_set.size().
    std::vector<value_t> strong_beta;
    init_strong_beta(strong_beta, strong_set, initial_size); 
    
    // previously valid strong beta
    auto strong_beta_prev_valid = strong_beta; 

    // current (unnormalized) R^2 at strong_beta.
    value_t rsq = 0;
    value_t rsq_prev_valid = rsq;
                                          
    // initialize lambda sequence
    vec_t lmdas_curr;
    init_lambdas(lmdas_curr, user_lmdas, 
            strong_set, strong_grad, n_lambdas_iter, factor);

    // list of coefficient outputs
    util::vec_type<sp_vec_t> betas_curr(lmdas_curr.size());

    // list of R^2 outputs
    vec_t rsqs_curr(lmdas_curr.size());

    while (1) 
    {
        // finish if all lmdas are finished
        if (lmdas_curr.size() == 0) break;

        /* Fit lasso */
        LassoParamPack<
            AType, value_t, index_t, bool_t
        > fit_pack(
            A, s, strong_set, strong_order, strong_A_diag,
            lmdas_curr, max_n_cds, thr, rsq, strong_beta, strong_grad,
            active_set, active_order, active_set_ordered,
            is_active, betas_curr, rsqs_curr, 0, 0       
        );
        auto& n_lmdas = fit_pack.n_lmdas;
        fit(fit_pack, check_user_interrupt);

        /* Checking KKT */

        // Get index of lambda of first failure.
        // grad will be the corresponding gradient vector at the returned lmdas_curr[index-1] if index >= 1,
        // and if idx <= 0, then grad is unchanged.
        // In any case, grad corresponds to the first smallest lambda where KKT check passes.
        size_t idx = check_kkt(
                    A, r, s, lmdas_curr.head(n_lmdas), betas_curr.head(n_lmdas), 
                    is_strong, n_threads, grad, grad_next);

        // decrement number of remaining lambdas
        n_lambdas_rem -= idx;

        /* Save output and check for any early stopping */

        // if first failure is not at the first lambda, save all previous solutions.
        for (size_t i = 0; i < idx; ++i) {
            betas.emplace_back(std::move(betas_curr[i]));
            lmdas.emplace_back(std::move(lmdas_curr[i]));
            rsqs.emplace_back(std::move(rsqs_curr[i]));
        }

        // check early termination 
        if (rsqs.size() >= 3) {
            const auto rsq_u = rsqs[rsqs.size()-1];
            const auto rsq_m = rsqs[rsqs.size()-2];
            const auto rsq_l = rsqs[rsqs.size()-3];
            if (check_early_stop_rsq(rsq_l, rsq_m, rsq_u)) break;
        }

        /* Screening */

        // screen to append to strong set and strong hashset.
        // Must use the previous valid gradient vector.
        // Only screen if KKT failure happened somewhere in the current lambda vector.
        // Otherwise, the current set might be enough for the next lambdas, so we try the current list first.
        bool new_strong_added = false;
        const auto old_strong_set_size = strong_set.size();

        if (idx < lmdas_curr.size()) {
            screen(grad.array().abs(), is_strong, delta_strong_size, strong_set);
            if (strong_set.size() > max_strong_size) throw util::max_basil_strong_set();
            new_strong_added = (old_strong_set_size < strong_set.size());

            const auto strong_set_new_begin = std::next(strong_set.begin(), old_strong_set_size);
            strong_hashset.insert(strong_set_new_begin, strong_set.end());

            // Note: DO NOT UPDATE strong_order YET!
            // Updating previously valid beta requires the old order.
            
            // only need to update on the new strong variables
            init_strong_A_diag(strong_A_diag, A, strong_set, 
                    old_strong_set_size, strong_set.size());

            // update ONLY on the new strong variable.
            // the old strong variables will be updated later!
            strong_beta.resize(strong_set.size(), 0);
            strong_beta_prev_valid.resize(strong_set.size(), 0);
            is_active.resize(strong_set.size(), false);
            strong_grad.resize(strong_set.size());
            for (size_t i = old_strong_set_size; i < strong_grad.size(); ++i) {
                strong_grad[i] = grad[strong_set[i]];
            }
        }

        // At this point, strong_set is ordered for all old variables
        // and unordered for the last few (new variables).
        // But all referencing quantities (strong_beta, strong_grad, is_active, strong_A_diag)
        // match up in size and positions with strong_set.
 
        // reset strong gradient to previous valid version.
        // We only need to update the old strong variables.
        // If new variables were added, these were updated before.
        for (size_t j = 0; j < old_strong_set_size; ++j) {
            strong_grad[j] = grad[strong_set[j]];
        }

        // create dense viewers of old strong betas
        Eigen::Map<util::vec_type<value_t>> old_strong_beta_view(
                strong_beta.data(), old_strong_set_size);
        Eigen::Map<util::vec_type<value_t>> strong_beta_prev_valid_view(
                strong_beta_prev_valid.data(),
                old_strong_set_size);

        // if the first lambda was the failure
        if (idx == 0) {
            // warm-start using previously valid solution.
            // Note: this is crucial for correctness in grad update step below.
            old_strong_beta_view = strong_beta_prev_valid_view;
            rsq = rsq_prev_valid;
        }
        else {
            // save last valid solution 
            assert(betas.size() > 0);
            const auto& last_valid_sol = betas.back();
            if (last_valid_sol.nonZeros() == 0) {
                old_strong_beta_view.setZero();
            } else {
                auto last_valid_sol_inner = last_valid_sol.innerIndexPtr();
                auto last_valid_sol_value = last_valid_sol.valuePtr();
                size_t osb_pos = 0;
                // zero-out all entries in the range (inner[i-1], inner[i]) 
                // and replace at inner[i] with valid solution.
                for (size_t i = 0; i < last_valid_sol.nonZeros(); ++i) {
                    assert(osb_pos < old_strong_beta_view.size());
                    auto lvs_i = last_valid_sol_inner[i];
                    auto lvs_x = last_valid_sol_value[i];
                    for (; strong_set[strong_order[osb_pos]] < lvs_i; ++osb_pos) {
                        assert(osb_pos < old_strong_beta_view.size());
                        old_strong_beta_view[strong_order[osb_pos]] = 0;
                    }
                    // here, we exploit the fact that the last valid solution
                    // is non-zero on the ever-active set, which is a subset
                    // of the old strong set, so we must have hit a common position.
                    auto ss_idx = strong_order[osb_pos];
                    assert((osb_pos < old_strong_beta_view.size()) &&
                           (strong_set[ss_idx] == lvs_i));
                    old_strong_beta_view[ss_idx] = lvs_x;
                    ++osb_pos;
                }
                for (; osb_pos < old_strong_beta_view.size(); ++osb_pos) {
                    old_strong_beta_view[strong_order[osb_pos]] = 0;
                }
            }

            strong_beta_prev_valid_view = old_strong_beta_view;

            // save last valid R^2
            rsq_prev_valid = rsq = rsqs.back();

            // shift lambda sequence
            // Find the maximum absolute gradient among the non-strong set
            // on the last valid solution.
            // The maximum must occur somewhere in the newly added strong variables.
            // If no new strong variables were added, it must be because 
            // we already added every variable. 
            // Use last valid lambda * factor as a cheap alternative to finding
            // the maximum absolute gradient at the last valid solution.
            if (!use_user_lmdas) {
                value_t max_abs_grad;
                if (!new_strong_added) {
                    max_abs_grad = lmdas.back() * factor;
                } else {
                    Eigen::Map<vec_t> strong_grad_view(
                            strong_grad.data() + old_strong_set_size,
                            strong_grad.size() - old_strong_set_size);
                    max_abs_grad = strong_grad_view.array().abs().maxCoeff();
                }             

                lmdas_curr.resize(std::min(n_lambdas_iter, n_lambdas_rem));
                next_lambdas(max_abs_grad, factor, lmdas_curr);
            } else {
                lmdas_curr.resize(std::min(n_lambdas_iter, n_lambdas_rem));
                auto begin = std::next(user_lmdas.data(), user_lmdas.size()-n_lambdas_rem);
                auto end = std::next(begin, lmdas_curr.size());
                std::copy(begin, end, lmdas_curr.data());
            }

            // reset current lasso estiamtes to next lambda sequence length
            betas_curr.resize(lmdas_curr.size());
            rsqs_curr.resize(lmdas_curr.size());

            // TODO: early stop
        }

        // update strong_order for new order of strong_set
        // only if new variables were added to the strong set.
        if (new_strong_added) {
            init_strong_order(strong_order, strong_set, old_strong_set_size);
        }
    }
}

} // namespace lasso
} // namespace ghostbasil
