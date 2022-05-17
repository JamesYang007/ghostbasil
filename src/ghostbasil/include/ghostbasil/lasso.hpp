#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <unordered_set>
#include <type_traits>
#include <cmath>
#include <ghostbasil/exceptions.hpp>
#include <ghostbasil/macros.hpp>

namespace ghostbasil {

template <class ValueType, class LmdasType>
inline void next_lambdas(
        size_t idx,
        ValueType factor,
        LmdasType& lmdas_curr,
        size_t& n_lambdas)
{
    size_t n_lmdas_rem = lmdas_curr.size()-idx;
    size_t n_lmdas_to_fill = std::min(idx, n_lambdas);
    size_t n_lmdas_next = n_lmdas_rem + n_lmdas_to_fill;

    lmdas_curr.head(n_lmdas_rem) = lmdas_curr.tail(n_lmdas_rem);
    // if current set of lambdas has all been processed
    // and if no more lambdas need to be processed.
    if (n_lmdas_rem == 0 && n_lambdas == 0) {
        lmdas_curr.resize(0);
        return;
    }

    assert(lmdas_curr.size() > 0);
    size_t prev_lmda_idx = (n_lmdas_rem == 0) ? lmdas_curr.size()-1 : n_lmdas_rem-1;
    lmdas_curr[n_lmdas_rem] = lmdas_curr[prev_lmda_idx] * factor;
    for (size_t l = n_lmdas_rem+1; l < n_lmdas_next; ++l) {
        lmdas_curr[l] = lmdas_curr[l-1] * factor;
    }

    // resize the vector to a smaller one if we appended less than the original size
    if (n_lmdas_to_fill < idx) {
        lmdas_curr.conservativeResize(n_lmdas_next);
    }

    n_lambdas -= n_lmdas_to_fill;
}

template <class AType, class YType, class ValueType, 
          class LmdasType, class BetasType, class ISType, class GradType>
inline auto check_kkt(
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

        bool kkt_fail = false;

#pragma omp parallel for schedule(static)
        for (size_t k = 0; k < y.size(); ++k) {
            // we still need to save the gradients including strong variables
            grad_tmp[k] = -sc * beta_i.dot(A.row(k)) + y[k] - s * beta_i.coeff(k);

            // just omit the KKT check for strong variables though.
            // if KKT failed previously, just do a no-op until loop finishes.
            // This is because OpenMP doesn't allow break statements.
            if (is_strong(k) || kkt_fail) continue;
            if (std::abs(grad_tmp[k]) >= lmda /* TODO: numerical prec window? */) {
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
 */
template <class GradType, class OutType>
inline auto initial_screen(
        const GradType& grad,
        size_t max_size,
        OutType& out)
{
    // TODO: optimize?
    size_t count_left = max_size;
    auto max_abs = grad.array().abs().maxCoeff();
    for (int i = 0; i < grad.size(); ++i) {
        if (count_left == 0) break;
        if (std::abs(grad[i]) == max_abs) {
            out.push_back(i);
            --count_left;
        }
    }
    return max_abs;
}

template <class GradType, class IAType, class OutType>
inline auto screen(
        const GradType& grad,
        const IAType& is_active,
        size_t max_size,
        OutType& out)
{
    using value_t = typename std::decay_t<GradType>::Scalar;
    size_t count_left = max_size;
    value_t max_abs = 0;

    // find maximum over all non-active variables
    for (int j = 0; j < grad.size(); ++j) { 
        if (is_active(j)) continue;
        max_abs = std::max(max_abs, std::abs(grad[j]));
    }

    // add variables to the strong set that reach the maximum.
    for (int i = 0; i < grad.size(); ++i) {
        if (count_left == 0) break;
        if (is_active(i)) continue;
        if (std::abs(grad[i]) == max_abs) {
            out.push_back(i);
            --count_left;
        }
    }
    return max_abs;
}

template <class AType, class RType, class ValueType, class BetaType>
inline auto objective(
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
inline void fit_lasso_active(
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

            // update gradient
            strong_grad[kk] -= s * del;
            for (size_t j = 0; j < active_set.size(); ++j) {
                auto jj = active_set[j];
                strong_grad[jj] -= sc * A.coeff(strong_set[jj],k) * del;
            }

            dlx = std::max(A.coeff(k,k) * del * del, dlx);
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
          class ASType, class AHSType, class IAType>
void fit_lasso(
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
    AHSType& active_hashset,
    IAType& is_active,
    size_t& n_cds)
{
    assert(strong_grad.size() == strong_set.size());
    assert(betas.cols() == lmdas.size());

    using value_t = ValueType;
    
    auto sc = 1-s;
    bool iz = false;
    
    for (size_t l = 0; l < lmdas.size(); ++l) {
        auto lmda = lmdas[l];

        if (iz) {
            fit_lasso_active(
                    A, s, strong_set, active_set, is_active, l, lmda, max_cds,
                    thr, warm_start, strong_grad, iz, n_cds);
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
                    active_hashset.emplace(k);
                }

                auto del = new_ak - ak;

                // update gradient
                strong_grad[kk] -= s * del;
                for (size_t j = 0; j < strong_set.size(); ++j) {
                    strong_grad[j] -= sc * A.coeff(strong_set[j],k) * del;
                }

                dlx = std::max(A.coeff(k,k) * del * del, dlx);
            }

            if (dlx < thr) break;

            if (n_cds >= max_cds) throw max_cds_error(l);

            fit_lasso_active(
                    A, s, strong_set, active_set, is_active, l, lmda, max_cds,
                    thr, warm_start, strong_grad, iz, n_cds);
        }

        betas.col(l) = warm_start;
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
 * TODO:
 * The algorithm stops at a $\lambda$ value where the 
 * pseudo-validation loss stops decreasing.
 * For now, we return a vector of vector of lambdas, 
 * and vector of coefficient (sparse) matrices 
 * corresponding to each vector of lambdas.
 */
template <class AType, class YType, class ValueType,
          class BetaMatType, class LmdasType>
inline void fit_basil(
        const AType& A,
        const YType& y,
        ValueType s,
        size_t n_knockoffs,
        size_t n_lambdas,
        size_t n_lambdas_iter,
        size_t strong_size,
        size_t delta_strong_size,
        size_t n_iters,
        size_t max_cds,
        ValueType thr,
        BetaMatType& betas,
        LmdasType& lmdas)
{
    using value_t = ValueType;
    using vec_t = Eigen::Matrix<value_t, Eigen::Dynamic, 1>;
    using sp_mat_t = Eigen::SparseMatrix<value_t>;
    using sp_vec_t = Eigen::SparseVector<value_t>;

    size_t n_blocks = (n_knockoffs + 1);
    size_t p = y.size() / n_blocks;
    const size_t initial_size = std::min(y.size(), 1L << 20);

    if (y.size() % n_blocks != 0) {
        throw std::runtime_error("y must have size multiple of M+1.");
    }

    // initialize strong set
    std::vector<uint32_t> strong_set; 
    strong_set.reserve(initial_size); 

    // (negative) gradient: -(1-s) A[k,:]^T * beta + y - s * beta[k]
    vec_t grad = y; 

    // get max absolute gradient and the next strong set
    auto max_abs_grad = initial_screen(grad, strong_size, strong_set);

    // initialize lambda sequence
    auto eqs = 1e-6; // TODO: generalize and find 
    auto factor = std::pow(eqs, static_cast<value_t>(1.0)/(n_lambdas - 1));
    n_lambdas_iter = std::min(n_lambdas_iter, n_lambdas);
    vec_t lmdas_curr(n_lambdas_iter);
    lmdas_curr[0] = max_abs_grad;
    for (size_t i = 1; i < lmdas_curr.size(); ++i) {
        lmdas_curr[i] = lmdas_curr[i-1] * factor;
    }
    n_lambdas -= n_lambdas_iter; // remaining lambdas

    // coefficient outputs
    sp_vec_t beta_prev_valid(p); // previously valid beta
    sp_vec_t beta_warm_start(p); // warm start (initialized to 0)
    sp_mat_t betas_curr(p, lmdas_curr.size()); // matrix of solutions

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

    // hashset of active features.
    // active_hashset contains k if feature k is active.
    // invariant: active_hashset contains the same values as evaluating strong_set[active_set[j]] for all j.
    std::unordered_set<uint32_t> active_hashset;
    std::unordered_set<uint32_t> strong_hashset(strong_set.begin(), strong_set.end());

    // number of total coordinate descents
    size_t n_cds = 0;

    for (size_t i = 0; i < n_iters; ++i) {
    
        // finish if all lmdas are finished
        if (lmdas_curr.size() == 0) break;

        // fit lasso
        fit_lasso(A, s, strong_set, lmdas_curr, max_cds, thr,
                  beta_warm_start, betas_curr, strong_grad, active_set, active_hashset, is_active, n_cds);

        // check KKT and get index of lambda of first failure.
        // grad will be the corresponding gradient vector at the returned lambda index - 1 if it is >= 0.
        // if returned lambda index <= 0, then grad is unchanged.
        // in any case, grad corresponds to the first smallest lambda where KKT check passes.
        // Note that grad is only updated (well-defined) for indices NOT in the strong set.
        size_t idx = check_kkt(
                A, y, s, lmdas_curr, betas_curr, 
                [&](auto i) { return strong_hashset.find(i) != strong_hashset.end(); },
                grad);

        // if the first lambda was the failure
        if (idx == 0) {
            strong_size += delta_strong_size;

            // warm-start using previously valid solution.
            // Note: this is crucial for correctness in grad update step below.
            beta_warm_start = beta_prev_valid;
        }
        else {
            // TODO: temporarily save valid betas and lmdas
            betas.emplace_back(betas_curr.block(0,0,betas_curr.rows(),idx));
            lmdas.emplace_back(lmdas_curr.head(idx));

            // save last valid solution as warm-start
            beta_prev_valid = beta_warm_start = betas_curr.col(idx-1); 

            // shift lambda sequence
            next_lambdas(idx, factor, lmdas_curr, n_lambdas);

            // reset current betas
            betas_curr.resize(betas_curr.rows(), lmdas_curr.size());
            betas_curr.setZero();

            // TODO: early stop
        }

        // screen to append to strong set and strong hashset.
        // Must use the previous valid gradient vector.
        auto old_strong_set_size = strong_set.size();
        screen(grad,
               [&](auto i) { return active_hashset.find(i) != active_hashset.end(); }, 
               strong_size, strong_set);
        strong_hashset.insert(
                std::next(strong_set.begin(), old_strong_set_size),
                strong_set.end());
        
        strong_grad.resize(strong_set.size());
        // Note: this is valid because beta_vec (warm-start) is corresondingly
        // set to the previously valid value where the gradient was grad.
        // We must rewrite every entry because strong_grad is updated by fit_lasso
        // until the end of lambda sequence, but we need the last valid state,
        // which may occur earlier in the sequence.
        for (size_t j = 0; j < strong_grad.size(); ++j) {
            strong_grad[j] = grad[strong_set[j]];
        }
        is_active.resize(strong_set.size(), false); // insert false values
    }

    if (lmdas_curr.size() > 0) {
        throw max_basil_iters_error(n_iters);
    }
}

} // namespace ghostbasil
