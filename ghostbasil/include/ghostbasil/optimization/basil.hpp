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
template <class SSType, class SHType, class GradType,
          class ISType>
GHOSTBASIL_STRONG_INLINE
void init_strong_set(
        SSType& strong_set, 
        SHType& strong_hashset,
        const GradType& grad, 
        const ISType& is_strong,
        size_t n_add,
        size_t capacity)
{
    strong_set.reserve(capacity); 
    screen(grad.array().abs(), is_strong, n_add, strong_set);
    strong_hashset.insert(strong_set.begin(), strong_set.end());
    std::sort(strong_set.begin(), strong_set.end());
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

template <class SADType, class MatType, class SSType>
GHOSTBASIL_STRONG_INLINE
void init_strong_A_diag(
        SADType& strong_A_diag, 
        const BlockMatrix<MatType>& A, 
        const SSType& strong_set,
        size_t begin,
        size_t end,
        size_t capacity=0)
{
    assert((begin <= end) && (end <= strong_set.size()));

    // subsequent calls does not affect capacity
    strong_A_diag.reserve(capacity);
    strong_A_diag.resize(strong_set.size());

    auto block_it = A.block_begin();
    const auto block_end = A.block_end();

    for (size_t i = begin; i < end; ++i) {
        auto k = strong_set[i];

        // update A block stride pointer if current feature is not in the block.
        if (!block_it.is_in_block(k)) block_it.advance_at(k);

        const auto k_shifted = block_it.shift(k);
        const auto& block = block_it.block();
        strong_A_diag[i] = block.coeff(k, k);
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
 * Initializes strong set ordering.
 */
template <class SSType, class SADType, class SBType,
          class SGType, class ASType, class IAType>
GHOSTBASIL_STRONG_INLINE
void preserve_strong_invariant(
        SSType& strong_set,
        SADType& strong_A_diag,
        SBType& strong_beta,
        SBType& strong_beta_prev_valid,
        SGType& strong_grad,
        ASType& active_set,
        IAType& is_active)
{
    using index_t = typename std::decay_t<SSType>::value_type;

    // initialize order
    util::vec_type<index_t> strong_order
        = util::vec_type<index_t>::LinSpaced(
                strong_set.size(), 0, strong_set.size()-1);

    // sort strong_set and get its order
    std::sort(strong_order.data(), strong_order.data() + strong_order.size(),
              [&](auto x, auto y) { return strong_set[x] < strong_set[y]; });

    // common buffers for repopulating
    std::decay_t<SBType> fbuff(strong_set.size());

    // permute all vectors via subsetting order
    auto permute = [&](auto& v, auto& buff, const auto& order) {
        assert(order.size() == v.size());
        assert(buff.size() == v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            buff[i] = v[order[i]];
        }
        v.swap(buff);
    };
    permute(strong_A_diag, fbuff, strong_order); 
    permute(strong_beta, fbuff, strong_order); 
    permute(strong_beta_prev_valid, fbuff, strong_order); 
    permute(strong_grad, fbuff, strong_order); 

    std::decay_t<IAType> bbuff(strong_set.size());
    permute(is_active, bbuff, strong_order);

    std::decay_t<SSType> ibuff(strong_set.size());
    permute(strong_set, ibuff, strong_order);

    // MUST repopulate active_set because indices of strong set
    // may be invalidated once new strong variables are added.
    // This also guarantees sorted values.
    size_t as_pos = 0;
    for (size_t i = 0; i < strong_set.size(); ++i) {
        if (as_pos >= active_set.size()) break;
        if (is_active[i]) {
            active_set[as_pos] = i;
            ++as_pos;
        }
    }
    assert(as_pos == active_set.size());
}

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
    GradType& grad)
{
    using value_t = ValueType;
    using vec_t = util::vec_type<value_t>;

    assert(r.size() == grad.size());
    assert(betas.size() == lmdas.size());

    auto sc = 1-s;

    vec_t grad_tmp(grad.size());

    size_t i = 0;
#pragma omp parallel num_threads(n_threads)
    {
#pragma omp single
        for (; i < lmdas.size(); ++i) {
            auto beta_i = betas[i];
            auto lmda = lmdas[i];

            std::atomic<bool> kkt_fail(false);

#pragma omp parallel for schedule(static)
            for (size_t k = 0; k < r.size(); ++k) {
                // we still need to save the gradients including strong variables
                auto gk = -sc * A.col_dot(k, beta_i) + r[k] - s * beta_i.coeff(k);
                grad_tmp[k] = gk;

                // just omit the KKT check for strong variables though.
                // if KKT failed previously, just do a no-op until loop finishes.
                // This is because OpenMP doesn't allow break statements.
                if (is_strong(k) || kkt_fail) continue;
                if (std::abs(gk) >= lmda /* TODO: numerical prec window? */) {
                    kkt_fail = true;
                }
            }

            if (kkt_fail) break; 
            else grad.swap(grad_tmp);
        }
    }

    return i;
}

/*
 * Find maximum absolute coefficient of vector grad.
 * Appends at most max_size number of (first) elements
 * that achieve the maximum absolute value in out.
 * Returns the maximum absolute value.
 * TODO: grad
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
          class BetasType, class LmdasType, class RsqsType>
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
        size_t n_threads,
        BetasType& betas,
        LmdasType& lmdas,
        RsqsType& rsqs)
{
    using value_t = ValueType;
    using index_t = int32_t;
    using vec_t = util::vec_type<value_t>;
    using sp_vec_t = util::sp_vec_type<value_t, Eigen::ColMajor, index_t>;

    const size_t n_features = r.size();
    const size_t initial_size = std::min(n_features, 1uL << 14);

    // input cleaning
    const bool use_user_lmdas = user_lmdas.size() != 0;
    auto eqs = 1e-6;    // TODO: generalize
    value_t factor = 0; // used only if user lambdas is empty
    if (!use_user_lmdas) {
        factor = std::pow(eqs, static_cast<value_t>(1.0)/(max_n_lambdas - 1));
    } else {
        max_n_lambdas = user_lmdas.size();
    }
    n_lambdas_iter = std::min(n_lambdas_iter, max_n_lambdas);
    size_t n_lambdas_rem = max_n_lambdas;
    max_strong_size = std::min(max_strong_size, n_features);
    strong_size = std::min(strong_size, max_strong_size);

    // (negative) gradient: -(1-s)/2 A[k,:]^T * beta + r - s/2 * beta[k]
    vec_t grad = r; 

    // used to determine if a feature is in strong set
    std::unordered_set<index_t> strong_hashset;
    const auto is_strong = [&](auto i) { 
        return strong_hashset.find(i) != strong_hashset.end(); 
    };

    // strong set
    std::vector<index_t> strong_set; 

    // initialize strong_set, strong_hashset based on current absolute gradient
    init_strong_set(strong_set, strong_hashset, grad, 
            is_strong, strong_size, initial_size);

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
    init_active_set(active_set, initial_size);

    // map of indices corresponding to strong variables that indicate if they are active or not.
    // invariant: is_active.size() == strong_set.size().
    std::vector<bool> is_active;
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

    // coefficient outputs
    util::vec_type<sp_vec_t> betas_curr(lmdas_curr.size()); // matrix of solutions

    vec_t rsqs_curr(lmdas_curr.size());

    // preserve strong invariants
    preserve_strong_invariant(
        strong_set, strong_A_diag, strong_beta, strong_beta_prev_valid,
        strong_grad, active_set, is_active);

    while (1) 
    {
        // finish if all lmdas are finished
        if (lmdas_curr.size() == 0) break;

        /* Fit lasso */
        
        size_t n_lmdas = 0;
        size_t n_cds = 0;
        lasso(A, s, strong_set, strong_A_diag, lmdas_curr, max_n_cds, thr, rsq,
              strong_beta, strong_grad, active_set, is_active, betas_curr, rsqs_curr, n_cds, n_lmdas);
        bool lasso_finished_early = n_lmdas < lmdas_curr.size();

        /* Checking KKT */

        // Get index of lambda of first failure.
        // grad will be the corresponding gradient vector at the returned lmdas_curr[index-1] if index >= 1,
        // and if idx <= 0, then grad is unchanged.
        // In any case, grad corresponds to the first smallest lambda where KKT check passes.
        size_t idx = check_kkt(
                A, r, s, lmdas_curr.head(n_lmdas), betas_curr.head(n_lmdas), 
                is_strong, n_threads, grad);

        // TODO: worth reverting active set back at idx also?
        // For now, don't because if KKT fails, active set probably needs more features.

        // decrement number of remaining lambdas
        n_lambdas_rem -= idx;

        /* Save output and check for any early stopping */

        // if first failure is not at the first lambda, save all previous solutions.
        for (size_t i = 0; i < idx; ++i) {
            betas.emplace_back(std::move(betas_curr[i]));
            lmdas.emplace_back(std::move(lmdas_curr[i]));
            rsqs.emplace_back(std::move(rsqs_curr[i]));
        }

        // if lmdas early stopped in processing (lasso terminated early),
        // terminate the whole BASIL framework also.
        if (lasso_finished_early) throw util::lasso_finished_early_error();

        /* Screening */

        // screen to append to strong set and strong hashset.
        // Must use the previous valid gradient vector.
        // Only screen if KKT failure happened somewhere in the current lambda vector.
        // Otherwise, the current set might be enough for the next lambdas, so we try the current list first.
        bool new_variables_added = false;
        const auto old_strong_set_size = strong_set.size();

        if (idx < lmdas_curr.size()) {
            screen(grad.array().abs(), is_strong, delta_strong_size, strong_set);

            if (strong_set.size() > max_strong_size) throw util::max_basil_strong_set();

            new_variables_added = (old_strong_set_size < strong_set.size());

            auto strong_set_new_begin = std::next(strong_set.begin(), old_strong_set_size);
            strong_hashset.insert(strong_set_new_begin, strong_set.end());

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
            const auto& last_valid_sol = betas_curr[idx-1];
            assert(last_valid_sol.nonZeros() == old_strong_beta_view.size());
            Eigen::Map<const vec_t> last_valid_sol_values(
                    last_valid_sol.valuePtr(),
                    last_valid_sol.nonZeros());
            old_strong_beta_view = last_valid_sol_values;
            strong_beta_prev_valid_view = old_strong_beta_view;

            // save last valid R^2
            rsq_prev_valid = rsq = rsqs_curr[idx-1];

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
                if (old_strong_set_size == strong_set.size()) {
                    max_abs_grad = lmdas_curr[idx-1] * factor;
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

        // preserve strong invariants.
        // Only need to call if new variables were added.
        if (new_variables_added) {
            preserve_strong_invariant(
                strong_set, strong_A_diag, strong_beta, strong_beta_prev_valid,
                strong_grad, active_set, is_active);
        }
    }
}

} // namespace ghostbasil
