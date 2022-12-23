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
template <class SSType, class SHType, class ValueType, 
          class ISType, class PenaltyType>
GHOSTBASIL_STRONG_INLINE
void init_strong_set(
        SSType& strong_set, 
        SHType& strong_hashset,
        ValueType alpha,
        const PenaltyType& penalty,
        const ISType& is_strong,
        size_t capacity)
{
    // if no L1 penalty, every variable is active
    if (alpha <= 0.0) {
        strong_set.resize(penalty.size());
        std::iota(strong_set.begin(), strong_set.end(), 0);
    } else {
        strong_set.reserve(capacity); 
        // add all non-penalized variables
        for (size_t i = 0; i < penalty.size(); ++i) {
            if (penalty[i] <= 0.0) {
                strong_set.push_back(i);        
            }
        }
    }
    strong_hashset.insert(strong_set.begin(), strong_set.end());
    
    // Note: in either case, strong_set contains increasing order of values.
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
template <class LmdasType, class UserLmdasType>
void init_lambdas(
        LmdasType& lmdas,
        const UserLmdasType& user_lmdas,
        size_t size)
{
    lmdas.resize(size);
    if (lmdas.size() == 0) return;
    std::copy(user_lmdas.data(), 
              std::next(user_lmdas.data(), lmdas.size()),
              lmdas.data());
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
template <class AType, class RType, class ValueType, class PenaltyType,
          class LmdasType, class BetasType, class ISType, class GradType>
GHOSTBASIL_STRONG_INLINE 
auto check_kkt(
    const AType& A, 
    const RType& r,
    ValueType alpha,
    const PenaltyType& penalty,
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
    auto alpha_c = 1 - alpha;

    if (lmdas.size() == 0) return i;

    for (; i < lmdas.size(); ++i) {
        const auto& beta_i = betas[i];
        const auto lmda = lmdas[i];

        std::atomic<bool> kkt_fail(false);

#pragma omp parallel for schedule(static) num_threads(n_threads)
        for (size_t k = 0; k < r.size(); ++k) {
            // Just omit the KKT check for strong variables.
            // If KKT failed previously, just do a no-op until loop finishes.
            // This is because OpenMP doesn't allow break statements.
            bool kkt_fail_raw = kkt_fail.load(std::memory_order_relaxed);
            if (kkt_fail_raw) continue;

            // we still need to save the gradients including strong variables
            auto gk = r[k] - A.col_dot(k, beta_i);
            grad_next[k] = gk;

            const auto pk = penalty[k];
            if (is_strong(k) || 
                (std::abs(gk - lmda * alpha_c * pk * beta_i.coeff(k)) <= 
                    lmda * alpha * pk)) continue;
            
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
template <class AbsGradType, class ValueType, class PenaltyType, class ISType, class SSType>
GHOSTBASIL_STRONG_INLINE 
void screen(
        const AbsGradType& abs_grad,
        ValueType lmda_prev,
        ValueType lmda_next,
        ValueType alpha,
        const PenaltyType& penalty,
        const ISType& is_strong,
        size_t size,
        SSType& strong_set,
        bool do_old_rule = false)
{
    using value_t = ValueType;

    assert(strong_set.size() <= abs_grad.size());
    if (do_old_rule) {
        size_t rem_size = abs_grad.size() - strong_set.size();
        size_t size_capped = std::min(size, rem_size);
        size_t old_strong_size = strong_set.size();
        strong_set.insert(strong_set.end(), size_capped, 0);
        const auto abs_grad_p = util::vec_type<value_t>::NullaryExpr(
            abs_grad.size(), [&](auto i) {
                return (penalty[i] <= 0) ? 0.0 : abs_grad[i] / penalty[i];
            }
        ) / std::max(alpha, 1e-3);
        util::k_imax(abs_grad_p, is_strong, size_capped, 
                std::next(strong_set.begin(), old_strong_size));
        return;
    }
    
    const auto strong_rule_lmda = (2 * lmda_next - lmda_prev) * alpha;
    for (size_t i = 0; i < abs_grad.size(); ++i) {
        if (is_strong(i)) continue;
        if (abs_grad[i] > strong_rule_lmda * penalty[i]) {
            strong_set.push_back(i);
        }
    }
}

/**
 * Solves the lasso objective for a sequence of \f$\lambda\f$ values.
 *
 * @param   A   covariance matrix.
 * @param   r   covariance between covariates and response.
 * @param   alpha   elastic net proportion.
 * @param   penalty penalty factor for each coefficient.
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
          class PenaltyType, class ULmdasType,
          class BetasType, class LmdasType, class RsqsType,
          class CUIType = util::no_op>
inline void basil(
        const AType& A,
        const RType& r,
        ValueType alpha,
        const PenaltyType& penalty,
        const ULmdasType& user_lmdas,
        size_t max_n_lambdas,
        size_t n_lambdas_iter,
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
    max_strong_size = std::min(max_strong_size, n_features);

    // (negative) gradient: r - A * beta
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
    init_strong_set(strong_set, strong_hashset, alpha, penalty, is_strong, initial_size);
    if (strong_set.size() > max_strong_size) {
        throw util::max_basil_strong_set();
    }

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

    // Fit lasso on lambda == inf to get non-penalized coefficients.
    util::vec_type<value_t, 1> lmda_inf(std::numeric_limits<value_t>::max());
    util::vec_type<sp_vec_t, 1> beta_inf;
    util::vec_type<value_t, 1> rsq_inf;
    LassoParamPack<
        AType, value_t, index_t, bool_t
    > fit_pack(
        A, alpha, penalty, strong_set, strong_order, strong_A_diag,
        lmda_inf, max_n_cds, thr, 0, strong_beta, strong_grad,
        active_set, active_order, active_set_ordered,
        is_active, beta_inf, rsq_inf, 0, 0       
    );
    fit(fit_pack, check_user_interrupt);
    
    // update states
    for (size_t i = 0; i < strong_set.size(); ++i) {
        grad[strong_set[i]] = strong_grad[i];
    }
    for (size_t i = 0; i < grad.size(); ++i) {
        if (is_strong(i)) continue;
        grad[i] -= A.col_dot(i, beta_inf[0]);
    }

    // current (unnormalized) R^2 at strong_beta.
    value_t rsq = fit_pack.rsq;
    value_t rsq_prev_valid = rsq;
    
    // previously valid strong beta
    auto strong_beta_prev_valid = strong_beta; 
                                          
    // initialize lambda sequence
    vec_t lmda_seq;
    const bool use_user_lmdas = user_lmdas.size() != 0;
    if (!use_user_lmdas) {
        // lmda_seq = [l_max, l_max * f, l_max * f^2, ..., l_max * f^(max_n_lambdas-1)]
        // l_max is the smallest lambda such that the penalized features (penalty > 0)
        // have 0 coefficients.
        value_t log_factor = std::log(min_ratio) * static_cast<value_t>(1.0)/(max_n_lambdas-1);
        value_t lmda_max = vec_t::NullaryExpr(
            grad.size(), [&](auto i) {
                return (penalty[i] <= 0.0) ? 0.0 : std::abs(grad[i]) / penalty[i];
            }
        ).maxCoeff() / std::max(alpha, 1e-3);
        lmda_seq.array() = lmda_max * (
            log_factor * vec_t::LinSpaced(max_n_lambdas, 0, max_n_lambdas-1)
        ).array().exp();
    } else {
        lmda_seq = user_lmdas;
        max_n_lambdas = user_lmdas.size();
    }
    n_lambdas_iter = std::min(n_lambdas_iter, max_n_lambdas);
    size_t n_lambdas_rem = max_n_lambdas;

    // current lambda sub-sequence
    // Take only lmda_max. Current state is the correct solution at lmda_max.
    // Lasso fitting should be trivial.
    vec_t lmdas_curr;
    init_lambdas(lmdas_curr, lmda_seq, 1);

    // list of coefficient outputs
    util::vec_type<sp_vec_t> betas_curr(lmdas_curr.size());

    // list of R^2 outputs
    vec_t rsqs_curr(lmdas_curr.size());

    while (1) 
    {
        /* Fit lasso */
        LassoParamPack<
            AType, value_t, index_t, bool_t
        > fit_pack(
            A, alpha, penalty, strong_set, strong_order, strong_A_diag,
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
                    A, r, alpha, penalty, lmdas_curr.head(n_lmdas), betas_curr.head(n_lmdas), 
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

        // finish if no more lambdas to process finished
        if (n_lambdas_rem == 0) break;

        /* Screening */

        const bool some_lambdas_failed = idx < lmdas_curr.size();

        // if some lambdas have valid solutions, shift the next lambda sequence
        if (idx > 0) {
            lmdas_curr.resize(std::min(n_lambdas_iter, n_lambdas_rem));
            auto begin = std::next(lmda_seq.data(), lmda_seq.size()-n_lambdas_rem);
            auto end = std::next(begin, lmdas_curr.size());
            std::copy(begin, end, lmdas_curr.data());

            // reset current lasso estimates to next lambda sequence length
            betas_curr.resize(lmdas_curr.size());
            rsqs_curr.resize(lmdas_curr.size());
        }

        // screen to append to strong set and strong hashset.
        // Must use the previous valid gradient vector.
        // Only screen if KKT failure happened somewhere in the current lambda vector.
        // Otherwise, the current set might be enough for the next lambdas, so we try the current list first.
        bool new_strong_added = false;
        const auto old_strong_set_size = strong_set.size();

        if (some_lambdas_failed) {
            const auto lmda_prev_valid = (lmdas.size() == 0) ? 
                std::numeric_limits<value_t>::max() : lmdas.back(); 
            const auto lmda_last = lmdas_curr[lmdas_curr.size()-1]; // well-defined
            const auto all_lmdas_failed = idx == 0;
            screen(grad.array().abs(), lmda_prev_valid, lmda_last, 
                alpha, penalty, is_strong, delta_strong_size, strong_set, all_lmdas_failed);
            if (strong_set.size() > max_strong_size) throw util::max_basil_strong_set();
            new_strong_added = (old_strong_set_size < strong_set.size());

            const auto strong_set_new_begin = std::next(strong_set.begin(), old_strong_set_size);
            strong_hashset.insert(strong_set_new_begin, strong_set.end());

            // Note: DO NOT UPDATE strong_order YET!
            // Updating previously valid beta requires the old order.
            
            // only need to update on the new strong variables
            init_strong_A_diag(strong_A_diag, A, strong_set, 
                    old_strong_set_size, strong_set.size());

            // updates on these will be done later!
            strong_beta.resize(strong_set.size(), 0);
            strong_beta_prev_valid.resize(strong_set.size(), 0);

            // update is_active to set the new strong variables to false
            is_active.resize(strong_set.size(), false);

            // update strong grad to last valid gradient
            strong_grad.resize(strong_set.size());
            for (size_t i = 0; i < strong_grad.size(); ++i) {
                strong_grad[i] = grad[strong_set[i]];
            }
        }

        // At this point, strong_set is ordered for all old variables
        // and unordered for the last few (new variables).
        // But all referencing quantities (strong_beta, strong_grad, is_active, strong_A_diag)
        // match up in size and positions with strong_set.
        // Note: strong_grad has been updated properly to previous valid version in all cases.

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
        else if (some_lambdas_failed) {
            // save last valid solution (this logic assumes the ordering is in old one)
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
