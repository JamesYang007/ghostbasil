#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <atomic>
#include <vector>
#include <unordered_set>
#include <type_traits>
#include <cmath>
#include <ghostbasil/exceptions.hpp>
#include <ghostbasil/macros.hpp>
#include <ghostbasil/algorithm.hpp>

namespace ghostbasil {

// TODO: find starting value and then factor down from there
template <class ValueType, class LmdasType>
GHOSTBASIL_STRONG_INLINE
void next_lambdas(
        ValueType max_abs_grad,
        ValueType factor,
        LmdasType& lmdas_curr)
{
    if (lmdas_curr.size() == 0) return;
    lmdas_curr[0] = max_abs_grad;
    for (size_t i = 1; i < lmdas_curr.size(); ++i) {
        lmdas_curr[i] = lmdas_curr[i-1] * factor;
    }
}

template <class AType, class YType, class ValueType, 
          class LmdasType, class BetasType, class ISType, class GradType>
GHOSTBASIL_STRONG_INLINE 
auto check_kkt(
    const AType& A, 
    const YType& y,
    ValueType s,
    const LmdasType& lmdas, 
    const BetasType& betas,
    const ISType& is_strong,
    GradType& grad)
{
    assert(y.size() == grad.size());
    assert(betas.rows() == y.size());
    assert(betas.cols() == lmdas.size());

    auto sc = 1-s;

    Eigen::Matrix<ValueType, Eigen::Dynamic, 1> grad_tmp(grad.size());

    size_t i = 0;
    for (; i < lmdas.size(); ++i) {
        auto beta_i = betas.col(i);
        auto lmda = lmdas[i];

        std::atomic<bool> kkt_fail(false);

#pragma omp parallel for schedule(static)
        for (size_t k = 0; k < y.size(); ++k) {
            // we still need to save the gradients including strong variables
            auto gk = -sc * beta_i.dot(A.row(k)) + y[k] - s * beta_i.coeff(k);
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
    k_imax(abs_grad, is_strong, size_capped, 
            std::next(strong_set.begin(), old_strong_size));
}

template <class AType, class RType, class ValueType, class BetaType>
GHOSTBASIL_STRONG_INLINE 
auto objective(
        const AType& A,
        const RType& r,
        ValueType s,
        ValueType lmda,
        const BetaType& beta)
{
    return (1-s)/2 * beta.dot(A * beta) - beta.dot(r) + s/2 * beta.squaredNorm()
        + lmda * beta.cwiseAbs().sum();
}

template <class AType
        , class ValueType
        , class SSType
        , class ASType
        , class IAType
        , class BetaType
        , class SGType>
GHOSTBASIL_STRONG_INLINE 
void fit_lasso_active(
    const AType& A,
    ValueType s,
    const SSType& strong_set,
    const ASType& active_set,
    const IAType& is_active,
    size_t lmda_idx,
    ValueType lmda,
    size_t max_cds,
    ValueType thr,
    BetaType& beta,
    SGType& strong_grad,
    ValueType& rsq,
    bool& iz,
    size_t& n_cds)
{
    using value_t = ValueType;
    auto sc = 1-s;
    iz = true;

    auto beta_diff = beta;

    while (1) {
        ++n_cds;
        auto dlx = 0.0;
        for (size_t l = 0; l < active_set.size(); ++l) {
            auto kk = active_set[l]; // index to strong set
            auto k = strong_set[kk]; // actual feature
            auto ak = beta.coeff(k);
            auto gk = strong_grad[kk]; // corresponding gradient
            auto denom = sc * A.coeff(k,k) + s;
            auto u = gk + ak * denom;
            auto v = std::abs(u) - lmda;

            value_t new_ak = (v > 0.0) ? std::copysign(v,u)/denom : 0;

            if (new_ak == ak) continue;

            beta.coeffRef(k) = new_ak;
            auto del = new_ak - ak;

            // get next measure of prediction difference
            auto dlx_curr = A.coeff(k,k) * del * del;

            // update rsq
            rsq += del * (2*(gk+s*ak) - (1-s)*del*A.coeff(k,k)) - s*(new_ak*new_ak-ak*ak);

            // update gradient
            strong_grad[kk] -= s * del;
            for (size_t j = 0; j < active_set.size(); ++j) {
                auto jj = active_set[j];
                strong_grad[jj] -= sc * A.coeff(strong_set[jj],k) * del;
            }

            // update measure of convergence
            dlx = std::max(dlx_curr, dlx);
        }
        if (dlx < thr) break;
        if (n_cds >= max_cds) throw max_cds_error(lmda_idx);
    }

    beta_diff = beta - beta_diff;

    // update gradient in non-active positions
    for (size_t kk = 0; kk < strong_set.size(); ++kk) {
        if (is_active[kk]) continue;
        strong_grad[kk] -= sc * beta_diff.dot(A.row(strong_set[kk]));
    }
}

/*
 * Solves the lasso problem stated in fit_basil with the additional constraint:
 * \[
 *      \beta_{i} = 0, \, \forall i \notin S^{(k)}
 * \]
 * i.e. all betas not in strong_set are fixed to be 0.
 *
 * @param   A   A matrix in the objective.
 * @param   y   response vector.
 * @param   s   regularization parameter of objective.
 * @param   strong_set  vector of indices representing features in the strong set.
 * @param   lmdas       regularization parameter lambda sequence.
 * @param   warm_start  sparse vector of coefficients as warm start.
 * @param   betas       output coefficient sparse matrix.
 *                      betas(i,j) = ith coefficient for jth lambda.
 * @param   strong_grad gradient buffer just for strong variables.
 * @param   active_set  vector of indices representing features in the active set. 
 * @param   is_active   is_active[k] = true if strong_set[k] is active.
 *                      Note that only features in strong_set are allowed to be active.
 *                      It suffices that is_active.size() >= strong_set.size();
 *                      only strong_set.size() number of elements will be used.
 */
template <class AType, class ValueType,
          class SSType, class LmdasType, class WarmStartType,
          class BetasType, class StrongGradType,
          class ASType, class IAType>
inline void fit_lasso(
    const AType& A, 
    ValueType s, 
    const SSType& strong_set, 
    const LmdasType& lmdas, 
    size_t max_cds,
    ValueType thr,
    WarmStartType& warm_start, 
    BetasType& betas, 
    StrongGradType& strong_grad,
    ASType& active_set,
    IAType& is_active,
    size_t& n_cds,
    size_t& n_lmdas)
{
    assert(strong_grad.size() == strong_set.size());
    assert(betas.cols() == lmdas.size());

    using value_t = ValueType;
    
    auto sc = 1-s;
    bool iz = false;
    value_t rsq = 0.0;
    n_lmdas = 0;

    for (size_t l = 0; l < lmdas.size(); ++l) {
        auto lmda = lmdas[l];
        value_t rsq_prev = rsq;

        if (iz) {
            fit_lasso_active(
                    A, s, strong_set, active_set, is_active, l, lmda, max_cds,
                    thr, warm_start, strong_grad, rsq, iz, n_cds);
        }

        while (1) {

            ++n_cds;
            value_t dlx = 0.0;
            for (size_t kk = 0; kk < strong_set.size(); ++kk) {
                auto k = strong_set[kk];
                auto ak = warm_start.coeff(k);
                auto gk = strong_grad[kk];
                auto denom = sc * A.coeff(k,k) + s;
                auto u = gk + ak * denom;
                auto v = std::abs(u) - lmda;
                value_t new_ak = (v > 0.0) ? std::copysign(v,u)/denom : 0.0;

                if (new_ak == ak) continue;

                warm_start.coeffRef(k) = new_ak;

                if (!is_active[kk]) {
                    is_active[kk] = true;
                    active_set.push_back(kk);
                }

                auto del = new_ak - ak;

                // get next measure of prediction difference
                auto A_kk = A.coeff(k,k);
                auto dlx_curr = A_kk * del * del;

                // update rsq
                rsq += del * (2*gk - del * ((1-s)*A_kk + s));

                // update gradient
                strong_grad[kk] -= s * del;
                for (size_t j = 0; j < strong_set.size(); ++j) {
                    strong_grad[j] -= sc * A.coeff(strong_set[j],k) * del;
                }

                // update measure of convergence
                dlx = std::max(dlx_curr, dlx);
            }

            if (dlx < thr) break;

            if (n_cds >= max_cds) throw max_cds_error(l);

            fit_lasso_active(
                    A, s, strong_set, active_set, is_active, l, lmda, max_cds,
                    thr, warm_start, strong_grad, rsq, iz, n_cds);
        }

        betas.col(l) = warm_start;
        ++n_lmdas;

        if (l == 0) continue;
        if (rsq-rsq_prev < 1e-5*rsq) break;
    }
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
 * @param   y   covariance between covariates and response.
 * @param   s   regularization of A towards identity.
 * @param   user_lmdas      user provided lambda sequence.
 * @param   max_n_lambdas   max number of lambdas to compute solutions for.
 *                          If user_lmdas is non-empty, it will be internally
 *                          reset to user_lmdas.size().
 * @param   n_lambdas_iter  number of lambdas per BASIL iteration.
 *                          Internally, it is capped at max_n_lambdas.
 * @param   strong_size     initial strong set size.
 *                          Internally, it is capped at max_strong_size.
 * @param   delta_strong_size   number of variables to add to strong set 
 *                              at every BASIL iteration.
 *                              Internally, it is capped at number of non-strong variables
 *                              at every BASIL iteration.
 * @param   max_strong_size     max number of strong set variables.
 *                              Internally, it is capped at number of features.
 * @param   max_n_cds           maximum number of coordinate descent per BASIL iteration.
 * @param   thr                 convergence threshold for coordinate descent.
 * @param   betas               vector of sparse matrices to store a list of solutions.
 * @param   lmdas               vector of dense vectors to store a list of lambdas
 *                              corresponding to the solutions in betas:
 *                              lmdas[i] is a vector of lambdas corresponding to the solutions
 *                              as columns of betas[i].
 *
 * TODO:
 * The algorithm stops at a $\lambda$ value where the 
 * pseudo-validation loss stops decreasing.
 * For now, we return a vector of vector of lambdas, 
 * and vector of coefficient (sparse) matrices 
 * corresponding to each vector of lambdas.
 */

template <class AType, class YType, class ValueType,
          class ULmdasType,
          class BetaMatType, class LmdasType>
inline void fit_basil(
        const AType& A,
        const YType& y,
        ValueType s,
        const ULmdasType& user_lmdas,
        size_t max_n_lambdas,
        size_t n_lambdas_iter,
        size_t strong_size,
        size_t delta_strong_size,
        size_t max_strong_size,
        size_t max_n_cds,
        ValueType thr,
        BetaMatType& betas,
        LmdasType& lmdas)
{
    using value_t = ValueType;
    using vec_t = Eigen::Matrix<value_t, Eigen::Dynamic, 1>;
    using sp_mat_t = Eigen::SparseMatrix<value_t>;
    using sp_vec_t = Eigen::SparseVector<value_t>;

    //size_t n_blocks = (n_knockoffs + 1);
    //size_t p = y.size() / n_blocks;
    const size_t n_features = y.size();
    const size_t initial_size = std::min(n_features, 1uL << 20);

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

    //if (y.size() % n_blocks != 0) {
    //    throw std::runtime_error("y must have size multiple of M+1.");
    //}

    // initialize strong set
    std::vector<uint32_t> strong_set; 
    strong_set.reserve(initial_size); 

    std::unordered_set<uint32_t> strong_hashset;

    // checks if feature is in strong set
    const auto is_strong = [&](auto i) { 
        return strong_hashset.find(i) != strong_hashset.end(); 
    };
    
    // (negative) gradient: -(1-s)/2 A[k,:]^T * beta + y - s/2 * beta[k]
    vec_t grad = y; 

    // get max absolute gradient and the next strong set
    screen(grad.array().abs(), is_strong, strong_size, strong_set);
    strong_hashset.insert(strong_set.begin(), strong_set.end());

    // initialize lambda sequence
    vec_t lmdas_curr(n_lambdas_iter);
    if (!use_user_lmdas) {
        value_t max_abs_grad = 0;
        for (size_t i = 0; i < strong_set.size(); ++i) {
            max_abs_grad = std::max(max_abs_grad, std::abs(grad[strong_set[i]]));
        }
        next_lambdas(max_abs_grad, factor, lmdas_curr);
    } else {
        std::copy(user_lmdas.data(), 
                  std::next(user_lmdas.data(), lmdas_curr.size()),
                  lmdas_curr.data());
    }
    n_lambdas_rem -= lmdas_curr.size(); // remaining lambdas

    // coefficient outputs
    sp_vec_t beta_prev_valid(n_features); // previously valid beta
    sp_vec_t beta_warm_start(n_features); // warm start (initialized to 0)
    sp_mat_t betas_curr(n_features, lmdas_curr.size()); // matrix of solutions

    // (negative) gradient only on strong set variables.
    // For the warm-start value of 0, gradient is just y.
    std::vector<value_t> strong_grad;
    strong_grad.reserve(initial_size);
    strong_grad.resize(strong_set.size());
    for (size_t i = 0; i < strong_set.size(); ++i) {
        strong_grad[i] = y[strong_set[i]];
    }

    // active set of indices corresponding to strong variables that are active.
    // active_set[k] implicitly defined if strong_set[active_set[k]] is active.
    // invariant: 0 <= active_set.size() <= strong_set.size(), 
    // active_set.size() is exactly the number of true entries in is_active
    // and contains exactly the indices to is_active that are true.
    std::vector<uint32_t> active_set;
    active_set.reserve(initial_size); 

    // map of indices corresponding to strong variables that indicate if they are active or not.
    // is_active[k] = true if feature strong_set[k] is active.
    // invariant: is_active.size() == strong_set.size().
    std::vector<bool> is_active;
    is_active.reserve(initial_size);
    is_active.resize(strong_set.size(), false);

    while (1) 
    {
        // finish if all lmdas are finished
        if (lmdas_curr.size() == 0) break;

        /* Fit lasso */
        
        size_t n_lmdas = 0;
        size_t n_cds = 0;
        fit_lasso(A, s, strong_set, lmdas_curr, max_n_cds, thr,
                    beta_warm_start, betas_curr, strong_grad, 
                    active_set, is_active, n_cds, n_lmdas);
        bool fit_lasso_finished_early = n_lmdas < lmdas_curr.size();

        /* Checking KKT */

        // Get index of lambda of first failure.
        // grad will be the corresponding gradient vector at the returned lmdas_curr[index-1] if index >= 1,
        // and if idx <= 0, then grad is unchanged.
        // In any case, grad corresponds to the first smallest lambda where KKT check passes.
        size_t idx = check_kkt(
                A, y, s, lmdas_curr.head(n_lmdas), betas_curr.block(0,0,betas_curr.rows(),n_lmdas), 
                is_strong, grad);

        /* Save output and check for any early stopping */

        // if first failure is not at the first lambda, save all previous solutions.
        if (idx) {
            betas.emplace_back(betas_curr.block(0,0,betas_curr.rows(),idx));
            lmdas.emplace_back(lmdas_curr.head(idx));
        }

        // if lmdas early stopped in processing (fit_lasso terminated early),
        // terminate the whole BASIL framework also.
        if (fit_lasso_finished_early) throw lasso_finished_early_error();

        /* Screening */

        // screen to append to strong set and strong hashset.
        // Must use the previous valid gradient vector.
        // Only screen if KKT failure happened somewhere in the current lambda vector.
        // Otherwise, the current set might be enough for the next lambdas, so we try the current list first.
        const auto old_strong_set_size = strong_set.size();
        if (idx < lmdas_curr.size()) {
            screen(grad.array().abs(), is_strong, delta_strong_size, strong_set);
            if (strong_set.size() > max_strong_size) throw max_basil_strong_set();
            auto strong_set_new_begin = std::next(strong_set.begin(), old_strong_set_size);
            strong_hashset.insert(strong_set_new_begin, strong_set.end());
            strong_grad.resize(strong_set.size());
            is_active.resize(strong_set.size(), false); // insert false values
        }
        
        // update strong gradient to previous valid solution
        // Note: this is valid because beta_vec (warm-start) is corresondingly
        // set to the previously valid value where the gradient was grad.
        // We must rewrite every entry because strong_grad is updated by fit_lasso
        // until the end of lambda sequence, but we need the last valid state,
        // which may occur earlier in the sequence.
        for (size_t j = 0; j < strong_grad.size(); ++j) {
            strong_grad[j] = grad[strong_set[j]];
        }

        // if the first lambda was the failure
        if (idx == 0) {
            // warm-start using previously valid solution.
            // Note: this is crucial for correctness in grad update step below.
            beta_warm_start = beta_prev_valid;
        }
        else {
            // save last valid solution as warm-start
            beta_prev_valid = beta_warm_start = betas_curr.col(idx-1); 

            // update max number of lambdas left to process.
            // since only idx number of lambdas have full solutions,
            // we add back the rest of the lambdas that we didn't fully process.
            n_lambdas_rem += lmdas_curr.size() - idx;

            // shift lambda sequence
            // Find the maximum absolute gradient among the non-strong set
            // on the last valid solution.
            // The maximum must occur somewhere in the newly added strong variables.
            // If no new strong variables were added, it must be because 
            // we already added every variable. 
            // Use last valid lambda * factor as a cheaper alternative to finding
            // the maximum absolute gradient at the last valid solution.
            if (!use_user_lmdas) {
                value_t max_abs_grad = 0;
                if (old_strong_set_size == strong_grad.size()) {
                    max_abs_grad = lmdas_curr[idx-1] * factor;
                }             
                for (size_t i = old_strong_set_size; i < strong_grad.size(); ++i) {
                    max_abs_grad = std::max(max_abs_grad, std::abs(strong_grad[i]));
                }
                lmdas_curr.resize(std::min(n_lambdas_iter, n_lambdas_rem));
                next_lambdas(max_abs_grad, factor, lmdas_curr);
            } else {
                lmdas_curr.resize(std::min(n_lambdas_iter, n_lambdas_rem));
                auto begin = std::next(user_lmdas.data(), user_lmdas.size()-n_lambdas_rem);
                auto end = std::next(begin, lmdas_curr.size());
                std::copy(begin, end, lmdas_curr.data());
            }
            n_lambdas_rem -= lmdas_curr.size();

            // reset current betas
            betas_curr.resize(betas_curr.rows(), lmdas_curr.size());

            // TODO: early stop
        }
    }
}

} // namespace ghostbasil
