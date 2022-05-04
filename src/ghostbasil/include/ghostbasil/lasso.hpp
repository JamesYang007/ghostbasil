#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <stdexcept>
#include <vector>
#include <type_traits>
#include <cmath>

namespace ghostbasil {

/*
 * Find maximum absolute coefficient of vector v.
 * Stores at most max_size number of (first) elements
 * that achieve the maximum absolute value in out.
 * Returns the maximum absolute value.
 */
template <class VType, class OutType>
inline auto screen(
        const VType& v,
        size_t max_size,
        OutType& out)
{
    // TODO: optimize?
    size_t count_left = max_size;
    auto max_abs = v.abs().maxCoeff();
    for (int i = 0; i < v.size(); ++i) {
        if (count_left == 0) break;
        if (std::abs(v[i]) == max_abs) {
            out.push_back(i);
            --count_left;
        }
    }
    return max_abs;
}

/*
 * Solves the following optimization problem:
 * \[
 *      \min\limits_{\beta} f(\beta) + \lambda ||\beta||_1 \\
 *      f(\beta) := 
 *          \frac{(1-s)}{2} \sum\limits_{l=1}^L \beta_l^\top A_l \beta_l
 *          - \beta^\top \begin{bmatrix} r \\ \tilde{r} \end{bmatrix}
 *          + \frac{s}{2} \beta^\top \beta
 * \]
 * for a sequence of $\lambda$ values.
 *
 * TODO:
 * The algorithm stops at a $\lambda$ value where the 
 * pseudo-validation loss stops decreasing.
 * For now, we return a vector of lambdas, 
 * and vector of coefficient (sparse) vectors.
 *
 */
template <class AVecType, class RType, class BetaMatType,
          class LmdasType>
inline void fit_basil(
        const AVecType& A_vec,
        const RType& r,
        size_t n_knockoffs,
        size_t n_lambdas,
        size_t n_lambdas_iter,
        size_t strong_size,
        size_t delta_strong_size,
        size_t n_iters,
        BetaMatType& betas,
        LmdasType& lmdas)
{
    using value_t = typename std::decay_t<RType>::Scalar;
    using vec_t = Eigen::Matrix<value_t, Eigen::Dynamic, 1>;
    using sp_mat_t = Eigen::SparseMatrix<value_t>;
    using sp_vec_t = Eigen::SparseVector<value_t>;
    //using arr_t = Eigen::Array<value_t, Eigen::Dynamic, Eigen::Dynamic>;

    size_t n_blocks = (n_knockoffs + 1);
    size_t p = r.size() / n_blocks;

    if (r.size() % n_blocks == 0) {
        throw std::runtime_error("grad must have size multiple of M+1.");
    }

    // initialize (ever) active set
    std::vector<uint32_t> active_set;
    active_set.reserve(p); // TODO: maybe too large in practice?

    // initialize strong set
    std::vector<uint32_t> strong_set; 
    strong_set.reserve(p); // TODO: maybe too large in practice?

    // get max absolute gradient and the next strong set
    auto max_abs_grad = screen(r, strong_size, strong_set);

    // initialize lambda sequence
    auto eqs = 1e-6; // TODO: generalize and find 
    auto factor = std::pow(eqs, static_cast<value_t>(1.0)/(n_lambdas - 1));
    vec_t lmdas_curr(n_lambdas_iter);
    lmdas_curr[0] = max_abs_grad;
    for (size_t i = 1; i < lmdas_curr.size(); ++i) {
        lmdas_curr[i] = lmdas_curr[i-1] * factor;
    }

    vec_t grad(p); // gradient
    sp_mat_t betas_curr(p, lmdas_curr.size()); // beta mat solutions
    sp_vec_t beta_vec(p); beta_vec.setZero();  // current beta vector

    for (size_t i = 0; i < n_iters; ++i) {
    
        // fit lasso
        fit_lasso(A_vec, strong_set, lmdas_curr, beta_vec, betas_curr, active_set);

        // check KKT and get index of lambda of first failure
        auto idx = check_kkt(
                A_vec, strong_set, active_set, lmdas_curr, betas_curr,
                grad, strong_size, delta_strong_size);

        if (idx > 0) {
            beta_vec = betas_curr.col(idx-1); // save last valid solution

            // TODO: temporarily save valid betas and lmdas
            betas.emplace_back(betas_curr.block(0,0,betas_curr.rows(),idx));
            lmdas.emplace_back(lmdas_curr.head(idx));
        }

        // get next lamda sequence
        next_lmdas(lmdas_curr, idx, factor);
        
        // reset current betas
        reset_betas(betas_curr);

        // TODO: early stop

        // screen to append to strong set
        screen(grad, strong_size, strong_set);
    }
}

} // namespace ghostbasil
